"""Shared lifecycle of the CT-PH-SAC algorithms: collection, optimizer steps, checkpoints."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import torch

from algorithms.ph_targets import (
    failure_charge,
    observed_end_reward,
    rollout_interval_reward,
    start_rate_reward,
)
from common.ph_config import (
    DEFAULT_REWARD_OBSERVATION,
    INPUT_MATRIX_START_SEED_OFFSET,
    decision_mode_setting,
    input_matrix_start_setting,
    plant_diagnostics_setting,
    plant_step_schedule,
    reward_observation_setting,
    warmup_decisions_setting,
)
from environment.ph_sac.registry import testbed_spec
from environment.ph_sac.tasks import (
    failure_mask_tensor,
    make_oracle_plant,
    reward_rate_tensor,
    terminal_payoff_tensor,
)
from models.ph_sac.networks import DecisionActor
from models.ph_sac.operators import _flat_decision_terms
from models.ph_sac.plant import PortHamiltonianPlant, diffusion_is_learned
from models.ph_sac.reward import RewardRate, end_state_fit_rows, end_state_reward_loss
from models.ph_sac.rollouts import (
    batch_contact_flags,
    grid_refresh_steps,
    integrate_endpoint_moments,
    path_jacobian_norms,
    start_state_flag_numpy,
    subset_batch,
)

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import NetworkConfig, TemperatureConfig, TrainingConfig
    from common.ph_replay import ReplayPartitions
    from common.ph_types import (
        AnalysisContext,
        Metrics,
        RewardFitRecord,
        Transition,
        TransitionBatch,
    )
    from environment.ph_sac.base import MujocoPHEnv
    from models.ph_sac.plant import Plant
    from models.ph_sac.rollouts import ContactExclusion, ReturnTerms, RolloutBatch
    from algorithms.ph_training import OptimizerStepTrace, TrainingBlock

# Block names that increment each resource counter; "critic" exists only in the explicit
# algorithm, so its traces fall outside ph_training's implicit-analysis TrainingBlock.
_BLOCK_COUNTERS = {
    "value": "value_gradient_steps",
    "rate": "rate_gradient_steps",
    "actor": "actor_gradient_steps",
    "model": "plant_gradient_steps",
    "temperature": "temperature_gradient_steps",
    "critic": "critic_gradient_steps",
}
_UPPER_TAIL_PROBABILITY = 0.05
_DURATION_BINS = 4
# policy_diagnostics (written with every evaluation since 2026-11-11): the latest replay
# rows it reads, the policy draws per row, and the seed of its private noise generator.
POLICY_DIAGNOSTIC_ROWS = 256
POLICY_DIAGNOSTIC_DRAWS = 4
POLICY_DIAGNOSTIC_SEED = 20_261_111
# The learned reward rate (the implicit arm's, and every arm's under reward_observation
# "end_state"): policy draws per state of its diagnostic, and the offsets of its
# initialization seed and diagnostic stream from the run seed.
_REWARD_DIAGNOSTIC_DRAWS = 4
_REWARD_INIT_SEED_OFFSET = 7919
_REWARD_DIAGNOSTIC_SEED_OFFSET = 104729


def _flatten(tensors: Sequence[Optional[Tensor]], like: Sequence[Tensor]) -> Tensor:
    """Concatenate per-parameter tensors, reading a missing gradient as zero."""
    return torch.cat(
        [
            (torch.zeros_like(p) if t is None else t).detach().reshape(-1)
            for t, p in zip(tensors, like)
        ]
    )


class PHAlgorithm:
    """Common collection/checkpoint contract; subclasses own their nested schedules.

    A learning run's budget is cumulative over learn calls. The callback fires once per
    interaction, after that interaction's updates, so a save inside it is a clean
    boundary; warm-up interactions count toward the budget and fire it too. Wall time
    runs from construction, so it includes any evaluation done before or during learn.

    Exact restarts: set_resume_hook installs a function the loops call at their safe
    points (_resume_point: the top of the warm-up loop, of the explicit loop and of the
    implicit inner loop, before the next row is collected), where resume_state captures
    everything the rest of the run depends on; an agent built afresh, given that state by
    load_resume_state and the global RNG states (evaluations/ph_sac/resume.py), continues
    bitwise as the uninterrupted run. A safe point inside the callback is not enough for
    the implicit arm (the K-th inner step's callback precedes the end of the outer
    iteration, and gate acquisitions fire it mid-attempt), hence the separate hook.

    The learned plant's online step follows the training keys plant_step_dense_until and
    plant_step_every (common.ph_config.plant_step_schedule; _plant_step_due).

    Contact exclusion (config key contact_handling, models/ph_sac/rollouts.ContactExclusion;
    the subclasses build self.exclusion): on a task with limited coordinates every
    collected row stores the start-state rule (near_limit) beside the environment's
    limit_touched, in either mode; with exclusion active (contact_handling "exclude") the
    plant fit (_plant_fit_batch) and the plant diagnostics leave the flagged rows out.

    Decisions are what the environment takes (env.decision_mode): held torques (the paper
    arms), or feedback laws (the impedance-law ablation). Every component reads the decision
    box from env.config["controller"] and maps decisions to torques through
    env.controller (FeedbackController or TorqueController), so nothing here depends
    on the mode; the config's testbed must declare the same mode (_check_decision_mode).

    The reward datum (config key reward_observation, common.ph_config.REWARD_OBSERVATION_KEY;
    the subclasses call _configure_reward_observation): under "end_state" every collected
    row stores the environment's end-state rate (Transition end_reward_rate, _collect), the
    replay hands out no interval reward, and the learned reward rate is fitted to it
    (_end_state_reward_step) and integrated over each interval by the targets
    (_generator_rewards, _observed_end_rewards, _rollout_rewards: each adds the failure
    charge). A checkpoint or warm start written under the other datum is refused.
    """

    actor: DecisionActor
    # Diagnostics read replay rows, so they are spaced by interactions; 0 disables them.
    # The training key plant_diagnostics_every sets it per run.
    plant_diagnostics_every = 10_000
    covariance_mode = "propagated"
    # Contact exclusion's rule (None on a task without limited coordinates).
    exclusion: Optional[ContactExclusion] = None
    # The reward datum (_configure_reward_observation sets it from the arm config).
    reward_observation = DEFAULT_REWARD_OBSERVATION
    # Reward steps between learned-reward diagnostics, recorded in reward_records; 0 disables.
    reward_diagnostics_every = 20_000

    def __init__(
        self, env: MujocoPHEnv, training: TrainingConfig, replay: ReplayPartitions
    ) -> None:
        self.env = env
        self.training = training
        self.replay = replay
        self.device = torch.device(training["device"])
        self.seed = int(training["seed"])
        self.testbed = env.config
        self.discount_rate = float(self.testbed["discount_rate"])
        self.physics_dt = float(self.testbed["physics_dt"])
        self.controller = env.controller
        # The model grids' refresh cadence: the controller's refresh_steps under
        # refresh_mode "held", None (no refresh points) under "continuous".
        self.refresh_steps = grid_refresh_steps(
            self.controller, int(self.testbed["controller"]["refresh_steps"])
        )
        self.decision_mode = env.decision_mode
        self.state_spec = env.state_spec

        streams = np.random.SeedSequence(self.seed).generate_state(6, dtype=np.uint32)
        self._env_seed = int(streams[0])
        self._rng = np.random.default_rng(int(streams[1]))
        self._generator = torch.Generator(self.device).manual_seed(int(streams[2]))
        self._predict_generator = torch.Generator(self.device).manual_seed(
            int(streams[3])
        )
        self._rollout_rng = np.random.default_rng(int(streams[4]))
        self._global_seed = int(streams[5])

        self._state: Optional[ndarray] = None
        self._episode_id = -1
        self._first_reset = True
        self._warmup_episodes_done = 0
        # The training key warmup_decisions (common.ph_config.warmup_decisions_setting):
        # None for the episode warm-up of every earlier run, else the decision warm-up's
        # length (warmup).
        self.warmup_decisions = warmup_decisions_setting(training)
        # Set by warm_start, which skips the decision warm-up (the episode warm-up is
        # marked done through _warmup_episodes_done); checkpointed only under the key.
        self._warmup_skipped = False
        # None until a checkpoint is loaded; then whether it carried the replay.
        self.replay_restored: Optional[bool] = None
        self._initialized = False
        self._budget: Optional[int] = None
        self._callback: Optional[Callable[[PHAlgorithm, Metrics], None]] = None
        self._observer: Optional[Callable[[OptimizerStepTrace], None]] = None
        self._resume_hook: Optional[Callable[[PHAlgorithm], None]] = None
        # The plant-step schedule and its position: rows offered after the dense phase.
        self.plant_step_dense_until, self.plant_step_every = plant_step_schedule(training)
        self._plant_rows_after_dense = 0
        diagnostics_every = plant_diagnostics_setting(training)
        if diagnostics_every is not None:
            self.plant_diagnostics_every = diagnostics_every
        self._block_steps: Dict[str, int] = {}
        self._wall_seconds = 0.0
        self._wall_start = time.perf_counter()
        self._counters: Dict[str, float] = {
            "interactions": 0,
            "simulated_seconds": 0.0,
            "critic_gradient_steps": 0,
            "value_gradient_steps": 0,
            "rate_gradient_steps": 0,
            "actor_gradient_steps": 0,
            "temperature_gradient_steps": 0,
            "plant_gradient_steps": 0,
            "model_rollout_paths": 0,
            "gate_simulated_paths": 0,
            "gate_real_rows_acquired": 0,
        }
        self.solver_records: List[Mapping[str, object]] = []
        self.admission_records: List[Mapping[str, object]] = []
        self.oracle_records: List[Mapping[str, object]] = []
        self.calibration_records: List[Mapping[str, object]] = []
        self.log_temperature: Optional[Tensor] = None
        self.temperature_optimizer: Optional[torch.optim.Optimizer] = None
        self._fixed_temperature: Optional[Tensor] = None

    # --- components ----------------------------------------------------------

    def initialize(self) -> None:
        """Build components, optimizers and target copies once; later calls do nothing."""
        if self._initialized:
            return
        self._initialized = True
        self._seed_global_stream()
        self._build_components()

    def _build_components(self) -> None:
        raise NotImplementedError

    def _check_decision_mode(self, config: Mapping[str, object]) -> None:
        """Require config["testbed"] to take the environment's decisions.

        A torque-mode environment needs a torque-decision config
        (common.ph_config.torque_decision_config), whose box and entropy target are for
        the m torques; a feedback environment needs a feedback config.
        """
        configured = decision_mode_setting(config["testbed"]["controller"])
        if configured != self.env.decision_mode:
            raise ValueError(
                f"The config's testbed takes {configured} decisions but the environment "
                f"takes {self.env.decision_mode} decisions; build a torque-decision config "
                "with common.ph_config.torque_decision_config."
            )

    def _check_input_matrix_start(self, config: Mapping[str, object]) -> None:
        """Refuse network input_matrix_start other than "true" ("random", "zero")
        without a learned plant.

        The start is the learned plant's (_build_learned_plant); an arm with the known
        plant (plant_source "oracle") has no input matrix to start, so the key would be
        silently ignored. Also checks the key's value; absent, nothing happens.
        """
        start = input_matrix_start_setting(config["network"])
        if start != "true" and config["plant_source"] != "learned":
            raise ValueError(
                f"network input_matrix_start {start!r} starts a learned plant's input "
                f"matrix; this arm's plant_source is {config['plant_source']!r}. Remove "
                "the key (or set 'true') for the known-plant arms."
            )

    def _configure_reward_observation(self, config: Mapping[str, object]) -> None:
        """Read the arm config's reward_observation (absent: "interval") and require the
        replay to hold that datum (common.ph_replay.ReplayPartitions reward_observation)."""
        self.reward_observation = reward_observation_setting(config)
        held = getattr(self.replay, "reward_observation", DEFAULT_REWARD_OBSERVATION)
        if held != self.reward_observation:
            raise ValueError(
                f"The replay holds the reward datum of reward_observation {held!r}, but the "
                f"config runs under {self.reward_observation!r}."
            )

    @property
    def end_state(self) -> bool:
        """Whether learners read the end-state rate only (reward_observation "end_state")."""
        return self.reward_observation == "end_state"

    def _check_reward_observation(self, config: Mapping[str, object], where: str) -> None:
        """Raise ValueError unless a stored config (a checkpoint's or a warm-start source's)
        ran under this agent's reward_observation (absent: "interval")."""
        stored = reward_observation_setting(config)
        if stored != self.reward_observation:
            raise ValueError(
                f"{where} was written under reward_observation {stored!r}, but this agent "
                f"runs under {self.reward_observation!r}: its replay, reward network and "
                "targets hold the other reward datum."
            )

    def _seed_global_stream(self) -> None:
        """Library calls that draw without a generator read this seeded global stream."""
        torch.manual_seed(self._global_seed)

    def _configure_temperature(self, temperature: TemperatureConfig) -> None:
        initial = float(temperature["initial_alpha_c"])
        self.temperature_config = temperature
        if temperature["learn_temperature"]:
            if initial <= 0.0:
                raise ValueError("A learned temperature needs a positive initial value.")
            self.log_temperature = torch.tensor(
                float(np.log(initial)),
                dtype=torch.get_default_dtype(),
                device=self.device,
                requires_grad=True,
            )
            self.temperature_optimizer = self._adam([self.log_temperature])
        else:
            self._fixed_temperature = torch.tensor(
                initial, dtype=torch.get_default_dtype(), device=self.device
            )

    @property
    def alpha_c(self) -> Tensor:
        """Current per-second temperature, detached from its tuning parameter."""
        if self.log_temperature is not None:
            return self.log_temperature.detach().exp()
        return self._fixed_temperature

    def _adam(self, parameters: Sequence[Tensor]) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=self.training["learning_rate"])

    def _build_learned_plant(self, network: NetworkConfig) -> PortHamiltonianPlant:
        """Build the online plant; network["diffusion"] "none" makes it drift-only.

        Its input matrix starts at the task's (initialize_from_physics), under network
        input_matrix_start "random" at _random_input_matrix's draw, or under "zero" at
        exactly zero (_zero_input_matrix); the callers copy the plant and build its
        optimizer only after this returns.
        """
        plant = PortHamiltonianPlant(
            self.state_spec, network, learn_diffusion=diffusion_is_learned(network)
        ).to(self.device)
        zero_state = np.zeros(int(self.env.state_dim))
        velocity_rows = self.env.oracle_diffusion(zero_state)[self.env.position_dim :]
        # The floor is a conditioning choice: a noiseless plant such as Acrobot has true
        # scale zero, and the likelihood gradient grows like 1/Sigma^2 as the scale falls.
        scale = np.maximum(np.linalg.norm(velocity_rows, axis=-1), 0.1)
        plant.initialize_from_physics(
            torch.as_tensor(self.env.oracle_input_matrix()),
            torch.as_tensor(scale),
        )
        start = input_matrix_start_setting(network)
        if start == "random":
            self._random_input_matrix(plant)
        elif start == "zero":
            self._zero_input_matrix(plant)
        return plant

    def _random_input_matrix(self, plant: PortHamiltonianPlant) -> None:
        """Overwrite the plant's input matrix G in place with i.i.d. standard normals.

        The draw is torch.randn of G's shape in float64 from a private CPU generator
        seeded INPUT_MATRIX_START_SEED_OFFSET + the training seed, cast to G's dtype and
        device; no other random stream of the run is touched. Prints one line with the
        true and the starting matrix.
        """
        true = plant.force_matrix.detach().clone()
        generator = torch.Generator(device="cpu").manual_seed(
            INPUT_MATRIX_START_SEED_OFFSET + self.seed
        )
        start = torch.randn(
            tuple(true.shape), generator=generator, dtype=torch.float64
        ).to(true)
        with torch.no_grad():
            plant.force_matrix.copy_(start)
        print(
            f"[input-start] random seed {self.seed} true {true.tolist()} "
            f"start {plant.force_matrix.detach().tolist()}",
            flush=True,
        )

    def _zero_input_matrix(self, plant: PortHamiltonianPlant) -> None:
        """Set every entry of the plant's input matrix G to exactly 0.0, in place.

        The same Parameter keeps its shape, dtype and device; nothing is drawn, so no
        random stream of the run is touched. Prints one line with the true and the
        starting matrix.
        """
        true = plant.force_matrix.detach().clone()
        with torch.no_grad():
            plant.force_matrix.zero_()
        print(
            f"[input-start] zero seed {self.seed} true {true.tolist()} "
            f"start {plant.force_matrix.detach().tolist()}",
            flush=True,
        )

    def _oracle_plant(self, oracle: Optional[Plant]) -> Plant:
        plant = make_oracle_plant(self.env) if oracle is None else oracle
        return plant.frozen_copy().to(self.device)

    def _reward_rate(self, states: Tensor, torques: Tensor) -> Tensor:
        return reward_rate_tensor(states, torques, self.testbed)

    def _failure_predicate(self, states: Tensor) -> Tensor:
        return failure_mask_tensor(states, self.testbed)

    # --- learned reward rate -------------------------------------------------

    def _learned_reward_rate(
        self, network: Mapping[str, object], controller: Mapping[str, object]
    ) -> RewardRate:
        """A RewardRate on the network config's features, its torques scaled by the motor
        limit, seeded apart from the global stream (a forked stream seeded with the run seed
        + _REWARD_INIT_SEED_OFFSET), so no other initialization or stream moves.

        Its output is a sigmoid onto the range the testbed's TestbedSpec declares in
        reward_rate_bounds ((0, 1) on the dm_control tolerance rewards), else linear.
        """
        limit = np.maximum(
            np.abs(np.asarray(controller["torque_low"], dtype=np.float64)),
            np.abs(np.asarray(controller["torque_high"], dtype=np.float64)),
        )
        declared = testbed_spec(self.testbed["name"]).reward_rate_bounds
        bounds = None if declared is None else declared(self.testbed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + _REWARD_INIT_SEED_OFFSET)
            network_module = RewardRate(
                self.state_spec, network, limit, bounds=bounds
            ).to(self.device)
        return network_module

    def _end_state_reward_step(self, batch: TransitionBatch) -> Metrics:
        """One Adam step (no clipping, as the implicit reward step) of the live reward_net
        on the end-state regression mean((r_hat(s', u_e) - rho_end)^2) over every row of the
        batch (models/ph_sac/reward.end_state_reward_loss: failed, truncated, shortened and
        contact-flagged rows included). Counts reward_gradient_steps and keeps
        last_reward_fit_loss; every reward_diagnostics_every steps it records
        reward_fit_diagnostics at the rows' (s', u_e). Draws from no random stream.
        """
        next_states, end_torques, end_rates = end_state_fit_rows(self.controller, batch)
        loss = end_state_reward_loss(self.reward_net, next_states, end_torques, end_rates)
        self.reward_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.reward_optimizer.step()
        self._counters["reward_gradient_steps"] += 1
        self.last_reward_fit_loss = float(loss.detach())
        every = self.reward_diagnostics_every
        if every and int(self._counters["reward_gradient_steps"]) % every == 0:
            self.reward_records.append(self.reward_fit_diagnostics(next_states, end_torques))
        return {"reward_fit_loss": self.last_reward_fit_loss}

    def reward_fit_diagnostics(self, states: Tensor, torques: Tensor) -> RewardFitRecord:
        """RMSE of the live learned rate against the task's rate, a sampled diagnostic.

        Compares at the given (state, taken torque) pairs and at fresh policy draws from
        a private stream, so calling it changes no training stream. It is never used in
        learning.
        """
        if getattr(self, "reward_net", None) is None:
            raise RuntimeError("The reward rate is the task's function; nothing is fitted.")
        with torch.no_grad():
            true_taken = reward_rate_tensor(states, torques, self.testbed)
            error_taken = (self.reward_net(states, torques) - true_taken).square().mean().sqrt()
            noise = torch.randn(
                (states.shape[0], _REWARD_DIAGNOSTIC_DRAWS, self.actor.decision_dim),
                generator=self._reward_log_generator,
                dtype=states.dtype,
            ).to(states.device)
            draws = self.actor.sample(
                states, sample_count=_REWARD_DIAGNOSTIC_DRAWS, noise=noise
            )["decisions"]
            torques_drawn, flat_states = _flat_decision_terms(self.controller, states, draws)
            true_drawn = reward_rate_tensor(flat_states, torques_drawn, self.testbed)
            error_drawn = (
                (self.reward_net(flat_states, torques_drawn) - true_drawn).square().mean().sqrt()
            )
        return {
            "interactions": self.interactions,
            "steps": int(self._counters["reward_gradient_steps"]),
            "fit_loss": self.last_reward_fit_loss,
            "rmse_taken": float(error_taken),
            "rmse_policy_draws": float(error_drawn),
            "true_rate_sd_taken": float(true_taken.std()),
            "true_rate_sd_draws": float(true_drawn.std()),
            "reward_output": self.reward_net.output,
        }

    def _terminal_payoff(self, states: Tensor) -> Tensor:
        """The task's failure payoff F at states, [..., 1]."""
        return terminal_payoff_tensor(states, self.testbed)

    def _failure_charges(self, batch: TransitionBatch) -> Tensor:
        """c = d e^(-lambda Delta) F(s'), [B,1] (ph_targets.failure_charge)."""
        return failure_charge(batch, self._terminal_payoff, discount_rate=self.discount_rate)

    def _generator_rewards(
        self,
        batch: TransitionBatch,
        reward_rate: Callable[[Tensor, Tensor], Tensor],
        flags: Optional[Tensor] = None,
    ) -> Tensor:
        """r_gen = Delta r_hat(s, u_0) + c, [B,1] detached (ph_targets.start_rate_reward):
        the generator target's reward under reward_observation "end_state"; at the rows of
        flags ([B] bool, contact exclusion) the observed ends' trapezoid + c instead."""
        rewards = start_rate_reward(self.controller, batch, reward_rate)
        if flags is not None and bool(flags.any()):
            observed = observed_end_reward(
                self.controller, batch, reward_rate, discount_rate=self.discount_rate
            )
            rewards = torch.where(flags.reshape(-1, 1), observed, rewards)
        return rewards + self._failure_charges(batch)

    def _observed_end_rewards(
        self, batch: TransitionBatch, reward_rate: Callable[[Tensor, Tensor], Tensor]
    ) -> Tensor:
        """r_obs + c = Delta/2 [r_hat(s, u_0) + e^(-lambda Delta) r_hat(s', u_e)] + c, [B,1]
        detached (ph_targets.observed_end_reward): the model-free targets' reward under
        reward_observation "end_state"."""
        observed = observed_end_reward(
            self.controller, batch, reward_rate, discount_rate=self.discount_rate
        )
        return observed + self._failure_charges(batch)

    def _rollout_rewards(
        self,
        batch: TransitionBatch,
        paths: RolloutBatch,
        reward_rate: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """r_roll + c, [B,1] detached (ph_targets.rollout_interval_reward of paths rolled
        out with _reward_return_terms(reward_rate)): the rollout targets' reward under
        reward_observation "end_state"; the observed ends' trapezoid stands in for truncated
        paths and for a row without a surviving path."""
        observed = observed_end_reward(
            self.controller, batch, reward_rate, discount_rate=self.discount_rate
        )
        return rollout_interval_reward(paths, observed) + self._failure_charges(batch)

    def _reward_return_terms(self, rate: Callable[[Tensor, Tensor], Tensor]) -> ReturnTerms:
        """stochastic_rollout's ReturnTerms of a learned rate: the paths then integrate rate
        by discounted trapezoids on their own knots and pay the task's failure charge."""
        return {
            "reward_rate": rate,
            "terminal_payoff": self._terminal_payoff,
            "discount_rate": self.discount_rate,
        }

    def _noise(self, shape: Sequence[int], generator: torch.Generator) -> Tensor:
        return torch.randn(
            tuple(shape),
            generator=generator,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )

    def _policy_sample(
        self, states: Tensor, *, sample_count: int = 1, actor: Optional[DecisionActor] = None
    ) -> Mapping[str, Tensor]:
        """Reparameterized draws from this run's stream; only callers decide on gradients."""
        actor = self.actor if actor is None else actor
        noise = self._noise(
            (states.shape[0], sample_count, actor.decision_dim), self._generator
        )
        return actor.sample(states, sample_count=sample_count, noise=noise)

    def _rollout_seed(self) -> int:
        return int(self._rollout_rng.integers(2**31))

    @property
    def exclusion_active(self) -> bool:
        """Whether contact exclusion acts (contact_handling "exclude")."""
        return self.exclusion is not None and self.exclusion.active

    def _plant_fit_batch(self, batch: TransitionBatch) -> Optional[TransitionBatch]:
        """The rows the plant fit uses: under contact exclusion the unflagged ones.

        The fit's loss is then the mean over the unflagged rows alone, the same as giving
        the flagged rows zero weight and normalizing by the unflagged count; no plant is
        integrated at a flagged row, so none of its terms reaches the backward pass. None
        when every row is flagged: the caller takes no step. Without exclusion the batch.
        """
        if not self.exclusion_active:
            return batch
        flags = batch_contact_flags(batch)
        if not bool(flags.any()):
            return batch
        if bool(flags.all()):
            return None
        return subset_batch(batch, ~flags)

    def _plant_step_due(self) -> bool:
        """Whether the row just collected gets its online-plant step (the schedule).

        Every row while self.interactions <= plant_step_dense_until, then every
        plant_step_every-th row offered. Call once per row offered; under the default
        (plant_step_every 1) it is always True and changes no state.
        """
        if self.plant_step_every == 1 or self.interactions <= self.plant_step_dense_until:
            return True
        self._plant_rows_after_dense += 1
        return self._plant_rows_after_dense % self.plant_step_every == 0

    # --- collection ----------------------------------------------------------

    @property
    def interactions(self) -> int:
        return int(self._counters["interactions"])

    def _budget_left(self) -> bool:
        return self._budget is None or self.interactions < self._budget

    def _begin_episode(self) -> None:
        seed = self._env_seed if self._first_reset else None
        self._first_reset = False
        self._state, _ = self.env.reset(seed=seed)
        self._episode_id += 1

    def _uniform_decision(self) -> ndarray:
        space = self.env.action_space
        return self._rng.uniform(space.low, space.high)

    def _behavior_decision(self, state: ndarray) -> ndarray:
        states = torch.as_tensor(
            state, dtype=torch.get_default_dtype(), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            sample = self._policy_sample(states)
        return sample["decisions"][0, 0].cpu().numpy().astype(np.float64)

    def _collect(
        self,
        decision_source: Callable[[ndarray], ndarray],
        *,
        partition: Optional[Literal["train", "gate"]] = None,
    ) -> Transition:
        if self._state is None:
            self._begin_episode()
        decision = decision_source(self._state)
        transition = self.env.step_dt(decision, self.env.sample_duration())
        if self.exclusion is not None:
            # Contact exclusion's start-state rule, stored with the row in either mode.
            transition["near_limit"] = start_state_flag_numpy(
                self.exclusion, transition["state"], transition["duration"]
            )
        if self.end_state:
            # The row's only reward datum under reward_observation "end_state" (gate
            # acquisitions included); reads no stream and not the interval reward.
            transition["end_reward_rate"] = self.env.end_reward_rate(transition)
        self.replay.add(transition, episode_id=self._episode_id, partition=partition)
        self._counters["interactions"] += 1
        self._counters["simulated_seconds"] += float(transition["duration"])
        ended = transition["terminated"] or transition["truncated"]
        self._state = None if ended else transition["next_state"]
        return transition

    def collect_transition(self) -> Transition:
        """Run one clock interval and store the pre-reset endpoint and actual duration."""
        return self._collect(self._behavior_decision)

    def warmup(self, episodes: int) -> None:
        """Collect initial episodes into replay partitions and account for their cost.

        Under the training key warmup_decisions (self.warmup_decisions not None) the
        warm-up is exactly that many uniform decisions instead, and episodes is ignored:
        an episode that ends inside it is reset as usual and the warm-up goes on, and the
        episode in progress after its last decision continues under the policy. Its count
        is the interaction counter (the warm-up's decisions are the run's first), so a
        restart inside it needs no state of its own; a warm start skips it.
        """
        if self.warmup_decisions is not None:
            while (
                not self._warmup_skipped
                and self.interactions < self.warmup_decisions
                and self._budget_left()
            ):
                self._resume_point()
                self._collect(lambda state: self._uniform_decision())
                self._after_interaction()
            return
        while self._warmup_episodes_done < int(episodes) and self._budget_left():
            self._resume_point()
            transition = self._collect(lambda state: self._uniform_decision())
            if transition["terminated"] or transition["truncated"]:
                self._warmup_episodes_done += 1
            self._after_interaction()

    def _train_ready(self) -> bool:
        return self.replay.partition_size("train") >= self.training["batch_size"]

    def _after_interaction(self) -> None:
        """End one interaction: update diagnostics, then fire the callback."""
        every = self.plant_diagnostics_every
        if (
            every
            and self.interactions % every == 0
            and self.replay.partition_size("train") >= 2
        ):
            size = min(self.training["batch_size"], self.replay.partition_size("train"))
            self.record_plant_diagnostics(self.replay.sample("train", size))
        if self._callback is not None:
            self._callback(self, {"interactions": float(self.interactions)})

    def policy_diagnostics(self) -> Dict[str, Optional[float]]:
        """The temperature and the policy's spread, as every evaluation records them.

        alpha_c: the per-second temperature (tuned or fixed); alpha_decision: alpha_c T, T
        the testbed's reference period, the same temperature as CT-SAC's alpha in CT-SAC's
        own units, per reference period (u alpha for an interval of u = Delta / T periods),
        the value the CT-SAC baseline starts from (under its default ct_sac_reward_units
        "second"; alpha_c under "reference_period"); log_std_mean: the actor's mean log
        standard deviation (its state-free parameter, clamped as sampling uses it);
        entropy_estimate: the mean of
        -log pi over POLICY_DIAGNOSTIC_DRAWS draws at each of the latest
        POLICY_DIAGNOSTIC_ROWS training rows (entropy_rows of them; None before the first
        row), the density over the normalized decision box that the temperature's target
        entropy refers to. The draws come from a private generator seeded with
        POLICY_DIAGNOSTIC_SEED and the rows are read by index, so nothing is drawn from the
        run's streams and a resumed run records the same values.
        """
        alpha_c = float(self.alpha_c)
        log_std = self.actor.log_standard_deviation().detach()
        record: Dict[str, Optional[float]] = {
            "alpha_c": alpha_c,
            "alpha_decision": alpha_c * float(self.testbed["reference_period"]),
            "log_std_mean": float(log_std.mean()),
            "entropy_estimate": None,
            "entropy_rows": 0,
        }
        buffer = self.replay.buffers["train"]
        count = min(POLICY_DIAGNOSTIC_ROWS, buffer.size())
        if count == 0:
            return record
        indices = (buffer.position - 1 - np.arange(count)) % buffer.capacity
        states = buffer.batch(indices)["states"]
        generator = torch.Generator(device=self.device).manual_seed(POLICY_DIAGNOSTIC_SEED)
        noise = self._noise((count, POLICY_DIAGNOSTIC_DRAWS, self.actor.decision_dim), generator)
        with torch.no_grad():
            sample = self.actor.sample(
                states, sample_count=POLICY_DIAGNOSTIC_DRAWS, noise=noise
            )
        record["entropy_estimate"] = float(-sample["log_probabilities"].mean())
        record["entropy_rows"] = int(count)
        return record

    def predict(self, state: ndarray, deterministic: bool = True) -> ndarray:
        """Return a decision (a feedback law, or a torque in torque mode); evaluation
        holds the mean over the interval."""
        states = torch.as_tensor(
            state, dtype=torch.get_default_dtype(), device=self.device
        ).unsqueeze(0)
        noise = None
        if not deterministic:
            noise = self._noise((1, 1, self.actor.decision_dim), self._predict_generator)
        with torch.no_grad():
            sample = self.actor.sample(states, noise=noise, deterministic=deterministic)
        return sample["decisions"][0, 0].cpu().numpy().astype(np.float64)

    # --- learning ------------------------------------------------------------

    def _run(self) -> None:
        """Advance this algorithm's own schedule until the budget is exhausted."""
        raise NotImplementedError

    def learn(
        self,
        total_interactions: int,
        *,
        callback: Optional[Callable[[PHAlgorithm, Metrics], None]] = None,
    ) -> PHAlgorithm:
        """Respect a decision-interaction budget, including warm-up and inner collection."""
        self._budget = int(total_interactions)
        self._callback = callback
        try:
            self.warmup(self.training["warmup_episodes"])
            self._run()
        finally:
            self._callback = None
            self._budget = None
        return self

    def counters(self) -> Metrics:
        """Report samples, physical seconds, steps per optimizer, and elapsed wall time."""
        wall = self._wall_seconds + time.perf_counter() - self._wall_start
        return {**self._counters, "wall_seconds": wall}

    # --- optimizer steps -----------------------------------------------------

    def set_optimizer_observer(
        self, observer: Optional[Callable[[OptimizerStepTrace], None]]
    ) -> None:
        """Optionally record actual increments/gradients/moments without changing the optimizer."""
        self._observer = observer

    # --- exact restarts ------------------------------------------------------

    def set_resume_hook(self, hook: Optional[Callable[[PHAlgorithm], None]]) -> None:
        """Call hook(agent) at every safe point of the loops (None removes it).

        The hook may save resume_state() or raise to stop the run; it must not change
        the agent or draw from any random stream.
        """
        self._resume_hook = hook

    def _resume_point(self) -> None:
        """A safe point: resume_state() here restarts the run exactly from this place."""
        if self._resume_hook is not None:
            self._resume_hook(self)

    def resume_state(self) -> Mapping[str, object]:
        """Everything a restart at a safe point needs: state_dict with the replay.

        The replay's filled rows are zero-copy tensor views of the live arrays, so the
        state must be saved before the agent moves on (the resume hook saves at once).
        The global python/numpy RNGs are the caller's (evaluations/ph_sac/resume.py);
        the torch global stream is in rng.global.
        """
        state = dict(self.state_dict(include_replay=False))
        state["replay"] = self.replay.state_dict(copy=False)
        return state

    def load_resume_state(self, state: Mapping[str, object]) -> None:
        """Restore resume_state() into an agent built from the same config."""
        self.load_state_dict(state)

    def _analysis_context(self) -> AnalysisContext:
        """Identify the candidate, policy and plant the next optimizer step refers to."""
        return {
            "outer_iteration": 0,
            "candidate_id": "V",
            "policy_id": "pi",
            "plant_id": "plant",
            "temperature_per_second": float(self.alpha_c),
            "policy_kind": "stochastic",
            "objective": "soft",
        }

    def _target_version(self) -> str:
        return "targets"

    def _optimizer_step(
        self,
        block: TrainingBlock,
        optimizer: torch.optim.Optimizer,
        parameters: Sequence[Tensor],
        loss: Tensor,
    ) -> float:
        """Take one clipped optimizer step, returning its loss; observers only read."""
        parameters = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._observer is not None:
            before = _flatten(parameters, parameters).clone()
            minibatch_gradient = _flatten([p.grad for p in parameters], parameters).clone()
        torch.nn.utils.clip_grad_norm_(parameters, self.training["gradient_clip"])
        if self._observer is not None:
            clipped_gradient = _flatten([p.grad for p in parameters], parameters).clone()
        optimizer.step()
        self._counters[_BLOCK_COUNTERS[block]] += 1
        step_index = self._block_steps.get(block, 0)
        self._block_steps[block] = step_index + 1
        loss_value = float(loss.detach())
        if self._observer is not None:
            moment, preconditioner = self._adam_moments(optimizer, parameters)
            after = _flatten(parameters, parameters)
            trace: OptimizerStepTrace = {
                "context": self._analysis_context(),
                "block": block,
                "step_index": step_index,
                "empirical_risk_id": f"{block}:{self.interactions}",
                "replay_version": str(self.interactions),
                "target_version": self._target_version(),
                "loss_value": loss_value,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "parameter_increment": after - before,
                "parameters_before": before,
                "empirical_gradient": None,
                "minibatch_gradient": minibatch_gradient,
                "clipped_gradient": clipped_gradient,
                "adam_first_moment": moment,
                "adam_preconditioner": preconditioner,
                "evidence": {
                    "kind": "sampled_diagnostic",
                    "domain": "replay minibatch of the recorded step",
                    "justification": "Recorded optimizer step; no uniform premise follows.",
                    "failure_probability": None,
                },
            }
            self._record_trace(trace)
            self._observer(trace)
        return loss_value

    def _record_trace(self, trace: OptimizerStepTrace) -> None:
        """Subclass hook for traces the observer also receives."""

    @staticmethod
    def _adam_moments(
        optimizer: torch.optim.Optimizer, parameters: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Return the bias-corrected first moment and diagonal preconditioner of Adam."""
        group = optimizer.param_groups[0]
        first_decay, second_decay = group["betas"]
        moments: List[Tensor] = []
        preconditioners: List[Tensor] = []
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            if not state:
                moments.append(torch.zeros_like(parameter).reshape(-1))
                preconditioners.append(torch.zeros_like(parameter).reshape(-1))
                continue
            steps = float(state["step"])
            first = state["exp_avg"] / (1.0 - first_decay**steps)
            second = state["exp_avg_sq"] / (1.0 - second_decay**steps)
            moments.append(first.reshape(-1))
            preconditioners.append((1.0 / (second.sqrt() + group["eps"])).reshape(-1))
        return torch.cat(moments).clone(), torch.cat(preconditioners).clone()

    def _polyak(
        self, target: torch.nn.Module, online: torch.nn.Module, rate: float
    ) -> None:
        with torch.no_grad():
            for slow, fast in zip(target.parameters(), online.parameters()):
                slow.mul_(1.0 - rate).add_(fast, alpha=rate)

    # --- diagnostics ---------------------------------------------------------

    def _learned_plant_for_diagnostics(self) -> Optional[Plant]:
        """Return the fitted plant to compare with the truth, or None for an oracle arm."""
        return None

    def _diagnostic_value(self) -> torch.nn.Module:
        """Return the value whose gradient weighs the value-direction drift error."""
        raise NotImplementedError

    def record_plant_diagnostics(self, batch: TransitionBatch) -> None:
        """Append sampled oracle errors and duration calibration; skip an oracle arm.

        The calibration scores the learned noise model, so a drift-only plant records
        oracle errors only.
        """
        plant = self._learned_plant_for_diagnostics()
        if plant is None:
            return
        from evaluations.ph_sac.diagnostics import (
            calibrate_by_duration,
            measure_oracle_errors,
        )

        flagged_share = None
        if self.exclusion_active:
            # Contact exclusion: the plant is the free mechanism, scored on unflagged rows.
            flags = batch_contact_flags(batch)
            flagged_share = float(flags.double().mean())
            if bool(flags.all()):
                return
            if bool(flags.any()):
                batch = subset_batch(batch, ~flags)
        states = batch["states"]
        resolved = self.controller.resolve_decision(states, batch["decisions"])
        with torch.no_grad():
            actions = self.controller.action(states, resolved)
        points = states.detach().requires_grad_(True)
        gradients = torch.autograd.grad(
            self._diagnostic_value()(points).sum(), points
        )[0]
        record = measure_oracle_errors(
            plant,
            make_oracle_plant(self.env).to(self.device),
            states.cpu().numpy(),
            actions.cpu().numpy(),
            gradients.cpu().numpy(),
            interactions=self.interactions,
        )
        if flagged_share is not None:
            record = {**record, "contact_flagged_share": flagged_share}
        self.oracle_records.append(record)
        if not getattr(plant, "learns_diffusion", True):
            return
        grid = {
            "maximum_step": self.model_step,
            "physics_dt": self.physics_dt,
            "controller_refresh_steps": self.refresh_steps,
            "integrator": self.model_integrator,
            "contact": getattr(self, "contact", None),
        }
        moments = integrate_endpoint_moments(
            plant,
            self.controller,
            states,
            batch["decisions"],
            batch["durations"],
            covariance_mode=self.covariance_mode,
            create_graph=False,
            **grid,
        )
        norms = path_jacobian_norms(
            plant, self.controller, states, batch["decisions"], batch["durations"], **grid
        )
        longest = float(self.testbed["clock"]["max_duration"])
        self.calibration_records.extend(
            calibrate_by_duration(
                batch,
                moments["means"].detach().cpu().numpy(),
                moments["covariances"].detach().cpu().numpy(),
                interactions=self.interactions,
                # Shortened final intervals fall in the lowest bin.
                duration_bin_edges=np.linspace(
                    0.0, longest + 0.5 * self.physics_dt, _DURATION_BINS + 1
                ).tolist(),
                upper_tail_probability=_UPPER_TAIL_PROBABILITY,
                duration_jacobian_norms=norms.detach().cpu().numpy(),
                saturated_fractions=moments["saturation_fractions"].detach().cpu().numpy(),
                state_spec=self.state_spec,
            )
        )

    # --- checkpoints ---------------------------------------------------------

    def _network_modules(self) -> Mapping[str, Optional[torch.nn.Module]]:
        raise NotImplementedError

    def _optimizers(self) -> Mapping[str, Optional[torch.optim.Optimizer]]:
        raise NotImplementedError

    def _loop_state(self) -> Mapping[str, object]:
        """Subclass schedule position and streams beyond the common fields."""
        return {}

    def _load_loop_state(self, state: Mapping[str, object]) -> None:
        pass

    def _config(self) -> Mapping[str, object]:
        raise NotImplementedError

    def state_dict(self, include_replay: bool = True) -> Mapping[str, object]:
        """Include networks, copies, optimizers, loop state, replay, config and RNG state.

        The replay keeps only its filled rows. Without it (include_replay False) the
        snapshot is small and restores everything else, but a run loaded from it trains
        on an empty replay, so it does not continue identically.
        """
        snapshot = copy.deepcopy(
            {
                "networks": {
                    name: module.state_dict()
                    for name, module in self._network_modules().items()
                    if module is not None
                },
                "optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in self._optimizers().items()
                    if optimizer is not None
                },
                "log_temperature": None
                if self.log_temperature is None
                else self.log_temperature.detach().clone(),
                "fixed_temperature": self._fixed_temperature,
                "counters": dict(self._counters),
                "wall_seconds": self.counters()["wall_seconds"],
                "block_steps": dict(self._block_steps),
                "environment": self.env.snapshot(),
                "state": self._state,
                "episode_id": self._episode_id,
                "first_reset": self._first_reset,
                "warmup_episodes_done": self._warmup_episodes_done,
                # The plant-step schedule's position (absent in checkpoints before
                # 2026-11-10, which ran the dense default: 0).
                "plant_schedule": {"rows_after_dense": self._plant_rows_after_dense},
                "records": {
                    "solver": self.solver_records,
                    "admission": self.admission_records,
                    "oracle": self.oracle_records,
                    "calibration": self.calibration_records,
                },
                "loop": self._loop_state(),
                "rng": {
                    "numpy": self._rng.bit_generator.state,
                    "rollout": self._rollout_rng.bit_generator.state,
                    "generator": self._generator.get_state(),
                    "predict": self._predict_generator.get_state(),
                    "global": torch.get_rng_state(),
                },
                "config": self._config(),
            }
        )
        if self.warmup_decisions is not None:
            # Only under a decision warm-up, so that every other checkpoint is as before.
            snapshot["warmup_skipped"] = self._warmup_skipped
        if include_replay:
            # The replay partitions already return copies, and they can be large.
            snapshot["replay"] = self.replay.state_dict()
        return snapshot

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a snapshot; one without a replay leaves the replay as it is.

        replay_restored records which: False means the replay was not in the snapshot. A
        snapshot written under another reward_observation raises before anything changes.
        """
        self._check_reward_observation(state.get("config", {}), "The checkpoint")
        modules = self._network_modules()
        for name, parameters in state["networks"].items():
            modules[name].load_state_dict(parameters)
        optimizers = self._optimizers()
        for name, parameters in state["optimizers"].items():
            optimizers[name].load_state_dict(parameters)
        if self.log_temperature is not None:
            with torch.no_grad():
                self.log_temperature.copy_(state["log_temperature"])
        else:
            self._fixed_temperature = state["fixed_temperature"].clone()
        self._counters = dict(state["counters"])
        self._wall_seconds = float(state["wall_seconds"])
        self._wall_start = time.perf_counter()
        self._block_steps = dict(state["block_steps"])
        self.replay_restored = "replay" in state
        if self.replay_restored:
            self.replay.load_state_dict(state["replay"])
        self.env.restore(state["environment"])
        self._state = copy.deepcopy(state["state"])
        self._episode_id = state["episode_id"]
        self._first_reset = state["first_reset"]
        self._warmup_episodes_done = state["warmup_episodes_done"]
        # Absent from every checkpoint written without a decision warm-up: not skipped.
        self._warmup_skipped = bool(state.get("warmup_skipped", False))
        self._plant_rows_after_dense = int(
            state.get("plant_schedule", {}).get("rows_after_dense", 0)
        )
        records = copy.deepcopy(state["records"])
        self.solver_records = records["solver"]
        self.admission_records = records["admission"]
        self.oracle_records = records["oracle"]
        self.calibration_records = records["calibration"]
        self._load_loop_state(state["loop"])
        rng = copy.deepcopy(state["rng"])
        self._rng.bit_generator.state = rng["numpy"]
        self._rollout_rng.bit_generator.state = rng["rollout"]
        self._generator.set_state(rng["generator"].cpu())
        self._predict_generator.set_state(rng["predict"].cpu())
        torch.set_rng_state(rng["global"].cpu())

    def save(self, path: Path, include_replay: bool = True) -> None:
        torch.save(self.state_dict(include_replay=include_replay), path)

    # --- warm start (the adaptation experiment) --------------------------------------

    def warm_start(
        self, path: Path, *, require_interactions: Optional[int] = None
    ) -> Dict[str, object]:
        """Start this fresh agent from a finished run's checkpoint.pt (save's state_dict).

        The subclass (_load_warm_start) takes the source's networks and temperature;
        the replay stays empty (the source's rows came from the robot before the change)
        and the optimizers, counters, records, schedules and random streams stay this
        agent's own, as built. The warm-up (training warmup_episodes episodes, or
        warmup_decisions decisions, of uniform decisions) is marked done, so the source's
        policy collects from the first decision. require_interactions, when given, must be
        the source's interaction
        count (a guard against an intermediate checkpoint of a run in progress). Returns
        what was loaded, for the run record.
        """
        if self.interactions or any(
            self.replay.partition_size(partition) for partition in ("train", "gate")
        ):
            raise RuntimeError("A warm start needs a fresh agent with an empty replay.")
        state = torch.load(path, map_location=self.device, weights_only=False)
        interactions = int(state["counters"]["interactions"])
        if require_interactions is not None and interactions != int(require_interactions):
            raise ValueError(
                f"{path} holds a run at {interactions} decisions, not the required "
                f"{int(require_interactions)}."
            )
        loaded = self._load_warm_start(state)
        self._warmup_episodes_done = int(self.training["warmup_episodes"])
        self._warmup_skipped = True
        return {"source_interactions": interactions, **loaded}

    def _load_warm_start(self, state: Mapping[str, object]) -> Dict[str, object]:
        """Load what a warm start takes from a source state_dict; the arm's own."""
        raise NotImplementedError(f"{type(self).__name__} has no warm start.")

    def load(self, path: Path) -> None:
        # The checkpoint holds this run's own numpy RNG states and config.
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
