"""Explicit CT-PH-SAC: the paper's single-level Algorithm 2."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Mapping, Optional

import numpy as np
import torch

from algorithms.ph_base import _REWARD_DIAGNOSTIC_SEED_OFFSET, PHAlgorithm
from algorithms.ph_targets import (
    generator_continuation,
    interval_critic_target,
    observed_continuation,
    rollout_continuation,
    td_continuation,
)
from common.ph_config import (
    exclusion_model_step,
    model_integrator_setting,
    reward_observation_setting,
)
from common.ph_replay import ReplayPartitions
from common.ph_time import discount_weight, interval_temperature
from environment.ph_sac.clock import DiscreteIntervalClock
from environment.ph_sac.tasks import terminal_boundary_functions
from models.ph_sac.losses import (
    critic_loss,
    explicit_actor_loss,
    plant_likelihood_loss,
    soft_value_loss,
    temperature_loss,
)
from models.ph_sac.networks import DecisionActor, SmoothValueHead, TwinQCritic
from models.ph_sac.rollouts import (
    batch_contact_flags,
    integrate_endpoint_moments,
    model_contact_exclusion,
    model_contact_substepping,
    sample_brownian_increments,
    stochastic_rollout,
)

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import ExplicitConfig
    from common.ph_types import Metrics, RewardFitRecord, TransitionBatch
    from environment.ph_sac.base import MujocoPHEnv
    from models.ph_sac.plant import Plant, PortHamiltonianPlant
    from models.ph_sac.reward import RewardRate


class ExplicitCTPHSAC(PHAlgorithm):
    """Actor, twin Qs, smooth V and plant; every target moves by Polyak averaging.

    Algorithm 2 holds no data out, so its default replay routes every row to training.
    The value head and its target carry the implicit agent's terminal boundary lift,
    V=H+B*network with V equal to the failure payoff on every failure limit; the
    critics, the actor and the target formulas reach the lift only through V.
    A learned plant follows network["diffusion"]; a drift-only plant is fitted by the
    drift loss, and its generator and rollout targets carry no noise terms. The plant
    fit and the rollout target integrate the plant with config["model_integrator"].
    Decisions are held torques when the environment and the config's testbed are in
    torque-decision mode (the explicit_*_torque paper arms): actor and critics then work
    on the m-dimensional torque box; or feedback laws (the impedance-law ablation arms).

    Contact exclusion (config contact_handling "exclude", self.exclusion active): the
    plant is fitted on unflagged rows only, and the critic target of a flagged row uses
    the observed continuation e^(-lambda Delta) Vbar(s') in place of the generator or
    rollout continuation (build_critic_target). The value, actor and temperature updates
    read no plant.

    reward_observation "end_state" (config key, common.ph_config.REWARD_OBSERVATION_KEY;
    every object below is None, and nothing changes, under the default): the agent builds
    a learned reward rate reward_net (PHAlgorithm._learned_reward_rate, its own seeded
    stream), a target copy target_reward that follows it by Polyak averaging at
    polyak_rate as the target plant follows the online plant, and its optimizer. Each
    train step fits reward_net to the end-state rates of the shared batch right after the
    plant step (update_reward; no warm fit, as the plant has none), and the critic target
    reads target_reward integrated over each interval in place of the interval reward
    (build_critic_target). Its steps count in reward_gradient_steps and its diagnostics go
    to reward_records.
    """

    value: SmoothValueHead
    target_value: SmoothValueHead
    critics: TwinQCritic
    target_critics: TwinQCritic
    online_plant: Optional[PortHamiltonianPlant]
    target_plant: Plant
    reward_net: Optional[RewardRate]
    target_reward: Optional[RewardRate]
    covariance_mode = "leading_order"

    def __init__(
        self,
        env: MujocoPHEnv,
        config: ExplicitConfig,
        *,
        replay: Optional[ReplayPartitions] = None,
        oracle: Optional[Plant] = None,
    ) -> None:
        training = config["training"]
        if replay is None:
            replay = ReplayPartitions(
                capacity=training["replay_capacity"],
                validation_fraction=0.0,
                device=training["device"],
                seed=int(training["seed"]),
                reward_observation=reward_observation_setting(config),
            )
        super().__init__(env, training, replay)
        self._check_decision_mode(config)
        self._check_input_matrix_start(config)
        self._configure_reward_observation(config)
        if self.end_state:
            # The learned reward rate's steps (under reward_observation "end_state" only,
            # so that every other agent's counters are as before).
            self._counters["reward_gradient_steps"] = 0
        self.config = config
        self._oracle = oracle
        # model_step, or under contact exclusion the task's coarse step.
        self.model_step = exclusion_model_step(config)
        self.model_integrator = model_integrator_setting(config)
        # Contact-triggered fine substepping (config model_coarse_step), None when off.
        self.contact = model_contact_substepping(config, self.state_spec, self.controller)
        # Contact exclusion's rule (config contact_handling), None without limited ranges.
        self.exclusion = model_contact_exclusion(config, env)
        self.continuation = config["continuation"]
        self.polyak_rate = float(config["polyak_rate"])
        self.rollout_paths = int(config["rollout_paths"])
        self.initialize()

    def _build_components(self) -> None:
        network = self.config["network"]
        controller = self.testbed["controller"]
        low = torch.as_tensor(np.asarray(controller["decision_low"]))
        high = torch.as_tensor(np.asarray(controller["decision_high"]))
        self.actor = DecisionActor(self.state_spec, low, high, network).to(self.device)
        self.critics = TwinQCritic(
            self.state_spec,
            self.actor.decision_dim,
            network,
            decision_low=low,
            decision_high=high,
        ).to(self.device)
        self.target_critics = self.critics.frozen_copy()
        self.value = SmoothValueHead(self.state_spec, network).to(self.device)
        self.value.configure_terminal_boundary(
            *terminal_boundary_functions(self.testbed)
        )
        self.target_value = self.value.frozen_copy()
        if self.config["plant_source"] == "oracle":
            self.online_plant = None
            self.target_plant = self._oracle_plant(self._oracle)
            self.plant_optimizer = None
        else:
            self.online_plant = self._build_learned_plant(network)
            self.target_plant = self.online_plant.frozen_copy()
            self.plant_optimizer = self._adam(list(self.online_plant.parameters()))
        self.actor_optimizer = self._adam(list(self.actor.parameters()))
        self.critic_optimizer = self._adam(list(self.critics.parameters()))
        self.value_optimizer = self._adam(list(self.value.parameters()))
        self._configure_temperature(self.config["temperature"])
        # Fresh next-interval draws for TD targets must not consume the environment's clock.
        self._next_clock = DiscreteIntervalClock(
            self.testbed["clock"], seed=int(self._rng.integers(2**31))
        )
        self._build_reward(network, controller)

    def _build_reward(
        self, network: Mapping[str, object], controller: Mapping[str, object]
    ) -> None:
        """Under reward_observation "end_state" build the learned reward rate, its Polyak
        target copy, its optimizer, records and private diagnostic stream; otherwise all
        None. Built last, on its own seeded stream, so every other initialization and
        stream is that of an agent without it."""
        self.reward_net = None
        self.target_reward = None
        self.reward_optimizer: Optional[torch.optim.Optimizer] = None
        self.reward_records: Optional[List[RewardFitRecord]] = None
        self.last_reward_fit_loss = float("nan")
        self._reward_log_generator: Optional[torch.Generator] = None
        if not self.end_state:
            return
        self.reward_net = self._learned_reward_rate(network, controller)
        self.target_reward = copy.deepcopy(self.reward_net).requires_grad_(False)
        self.reward_optimizer = self._adam(list(self.reward_net.parameters()))
        self.reward_records = []
        self._reward_log_generator = torch.Generator().manual_seed(
            self.seed + _REWARD_DIAGNOSTIC_SEED_OFFSET
        )

    def _run(self) -> None:
        while self._budget_left():
            self._resume_point()
            self.collect_transition()
            if self._train_ready():
                self.train(1, self.training["batch_size"])
            self._after_interaction()

    def train(self, gradient_steps: int, batch_size: int) -> Metrics:
        """Temperature, plant, V, twin Qs, actor, then all Polyak target updates.

        The learned plant steps on the batch when the plant-step schedule says so
        (_plant_step_due, once per call of train's loop); the target plant's Polyak
        update runs at every step either way. Under reward_observation "end_state" the
        learned reward rate steps on the same batch after the plant, at every step
        (update_reward; it draws from no stream).
        """
        metrics: Metrics = {}
        for _ in range(int(gradient_steps)):
            batch = self.replay.sample("train", batch_size)
            metrics.update(self.update_temperature(batch))
            if self.online_plant is not None and self._plant_step_due():
                metrics.update(self.update_plant(batch))
            if self.reward_net is not None:
                metrics.update(self.update_reward(batch))
            metrics.update(self.update_value(batch))
            metrics.update(self.update_critics(batch, self.build_critic_target(batch)))
            metrics.update(self.update_actor(batch))
            self.update_targets()
        return metrics

    def update_temperature(self, batch: TransitionBatch) -> Metrics:
        """Tune alpha_c, charging each replay row alpha_c w_lambda(Delta)."""
        if self.log_temperature is None:
            return {}
        with torch.no_grad():
            sample = self._policy_sample(batch["states"])
        loss = temperature_loss(
            self.log_temperature,
            sample["log_probabilities"],
            self.temperature_config["target_entropy"],
            interval_weights=discount_weight(
                batch["durations"], discount_rate=self.discount_rate
            ),
        )
        value = self._optimizer_step(
            "temperature", self.temperature_optimizer, [self.log_temperature], loss
        )
        return {"temperature_loss": value, "alpha_c": float(self.alpha_c)}

    def update_plant(self, batch: TransitionBatch) -> Metrics:
        """One likelihood step using the explicit leading-order endpoint covariance.

        plant_likelihood_loss scores a drift-only plant's moments by the drift loss.
        Under contact exclusion the fit reads the unflagged rows only; a batch of flagged
        rows alone takes no step (plant_loss 0, plant_rows 0).
        """
        if self.online_plant is None:
            return {}
        batch = self._plant_fit_batch(batch)
        if batch is None:
            return {"plant_loss": 0.0, "plant_rows": 0.0}
        moments = integrate_endpoint_moments(
            self.online_plant,
            self.controller,
            batch["states"],
            batch["decisions"],
            batch["durations"],
            maximum_step=self.model_step,
            physics_dt=self.physics_dt,
            controller_refresh_steps=self.refresh_steps,
            covariance_mode="leading_order",
            create_graph=True,
            integrator=self.model_integrator,
            contact=self.contact,
        )
        loss = plant_likelihood_loss(batch, moments, self.state_spec)
        value = self._optimizer_step(
            "model", self.plant_optimizer, list(self.online_plant.parameters()), loss
        )
        return {"plant_loss": value}

    def update_reward(self, batch: TransitionBatch) -> Metrics:
        """One step of the learned reward rate on the end-state rates of every row of the
        batch (PHAlgorithm._end_state_reward_step); nothing without one."""
        if self.reward_net is None:
            return {}
        return self._end_state_reward_step(batch)

    def _interval_temperature(self, batch: TransitionBatch) -> Tensor:
        return interval_temperature(
            self.alpha_c, batch["durations"], discount_rate=self.discount_rate
        )

    def update_value(self, batch: TransitionBatch) -> Metrics:
        """Regress on one-draw soft values from frozen target Qs."""
        states = batch["states"]
        with torch.no_grad():
            sample = self._policy_sample(states)
            decisions = sample["decisions"][:, 0]
            targets = self.target_critics.minimum(
                states, decisions
            ) - self._interval_temperature(batch) * sample["log_probabilities"][:, 0]
        loss = soft_value_loss(self.value, states, targets)
        value = self._optimizer_step(
            "value", self.value_optimizer, list(self.value.parameters()), loss
        )
        return {"value_loss": value}

    def build_critic_target(self, batch: TransitionBatch) -> Tensor:
        """Add reward once to masked generator, survivor-rollout or optional TD continuation.

        Contact exclusion: at a flagged row (start near a limit, or the interval touched
        one) the target is model-free, y = r + (1-d) e^(-lambda Delta) Vbar(s'), the
        observed interval's (CT-SAC's critic target with the soft value Vbar in place of
        its soft Q at s'); the generator continuation Vbar + Delta(G Vbar - lambda Vbar) and
        the rollout continuation are not formed there (a flagged row is not rolled out),
        and a model path from an unflagged row that comes near a limit is truncated, its
        continuation the observed endpoint's (rollouts.survivor_continuation). The td
        continuation reads no plant and is unchanged.

        reward_observation "end_state": the replay holds no interval reward, and r is the
        target reward rate r_hat (target_reward) integrated over the row's elapsed Delta,
        plus the failure charge c = d e^(-lambda Delta) F(s'): under "generator" the left
        rectangle Delta r_hat(s, u_0) + c at the point and torque of its explicit Euler
        continuation; under "rollout" the survivor mean of the same paths' discounted
        trapezoids of r_hat on their knots + c (truncated paths and a row without a
        survivor take the observed ends' trapezoid); under "td" and at flagged rows the
        observed ends' trapezoid Delta/2 [r_hat(s, u_0) + e^(-lambda Delta) r_hat(s', u_e)]
        + c (PHAlgorithm._generator_rewards, _rollout_rewards, _observed_end_rewards).
        """
        flags = batch_contact_flags(batch) if self.exclusion_active else None
        rewards = None
        if self.continuation == "generator":
            continuation = generator_continuation(
                self.target_plant,
                self.target_value,
                self.controller,
                batch["states"],
                batch["decisions"],
                batch["durations"],
                discount_rate=self.discount_rate,
            )
            if self.end_state:
                rewards = self._generator_rewards(batch, self.target_reward)
        elif self.continuation == "rollout":
            noise = sample_brownian_increments(
                batch["durations"],
                path_count=self.rollout_paths,
                noise_dim=self.target_plant.noise_dim,
                maximum_step=self.model_step,
                physics_dt=self.physics_dt,
                controller_refresh_steps=self.refresh_steps,
                seed=self._rollout_seed(),
            ).to(self.device)
            paths = stochastic_rollout(
                self.target_plant,
                self.controller,
                batch["states"],
                batch["decisions"],
                batch["durations"],
                self._failure_predicate,
                maximum_step=self.model_step,
                physics_dt=self.physics_dt,
                controller_refresh_steps=self.refresh_steps,
                path_count=self.rollout_paths,
                brownian_increments=noise,
                integrator=self.model_integrator,
                contact=self.contact,
                **(
                    {}
                    if flags is None
                    else {"truncation": self.exclusion, "skip_rows": flags}
                ),
                **(
                    {"return_terms": self._reward_return_terms(self.target_reward)}
                    if self.end_state
                    else {}
                ),
            )
            self._counters["model_rollout_paths"] += (
                batch["states"].shape[0] * self.rollout_paths
            )
            continuation = rollout_continuation(
                self.target_value, paths, batch, discount_rate=self.discount_rate
            )
            if self.end_state:
                rewards = self._rollout_rewards(batch, paths, self.target_reward)
        else:
            durations = batch["durations"]
            next_durations = torch.as_tensor(
                [self._next_clock.sample_duration() for _ in range(durations.shape[0])],
                dtype=durations.dtype,
                device=durations.device,
            ).reshape(-1, 1)
            continuation = td_continuation(
                self.target_critics,
                self.actor,
                batch["next_states"],
                durations,
                next_durations,
                alpha_c=self.alpha_c,
                discount_rate=self.discount_rate,
            )
            flags = None
            if self.end_state:
                rewards = self._observed_end_rewards(batch, self.target_reward)
        if flags is not None and bool(flags.any()):
            observed = observed_continuation(
                self.target_value, batch, discount_rate=self.discount_rate
            )
            continuation = torch.where(flags.reshape(-1, 1), observed, continuation)
            if rewards is not None:
                rewards = torch.where(
                    flags.reshape(-1, 1),
                    self._observed_end_rewards(batch, self.target_reward),
                    rewards,
                )
        return interval_critic_target(batch, continuation, rewards=rewards).detach()

    def update_critics(self, batch: TransitionBatch, targets: Tensor) -> Metrics:
        loss = critic_loss(self.critics, batch["states"], batch["decisions"], targets)
        value = self._optimizer_step(
            "critic", self.critic_optimizer, list(self.critics.parameters()), loss
        )
        return {"critic_loss": value}

    def update_actor(self, batch: TransitionBatch) -> Metrics:
        """Read min(Q1,Q2) with interval entropy temperature."""
        states = batch["states"]
        loss = explicit_actor_loss(
            self.critics,
            states,
            self._policy_sample(states),
            self._interval_temperature(batch),
        )
        value = self._optimizer_step(
            "actor", self.actor_optimizer, list(self.actor.parameters()), loss
        )
        return {"actor_loss": value}

    def update_targets(self) -> None:
        """Polyak-update twin Qs, V, learned plant and learned reward rate (under
        reward_observation "end_state"); leave an oracle fixed."""
        self._polyak(self.target_critics, self.critics, self.polyak_rate)
        self._polyak(self.target_value, self.value, self.polyak_rate)
        if self.online_plant is not None:
            self._polyak(self.target_plant, self.online_plant, self.polyak_rate)
        if self.target_reward is not None:
            self._polyak(self.target_reward, self.reward_net, self.polyak_rate)

    # --- checkpoints and diagnostics -----------------------------------------

    def _learned_plant_for_diagnostics(self) -> Optional[Plant]:
        return self.online_plant

    def _diagnostic_value(self) -> torch.nn.Module:
        return self.target_value

    def _network_modules(self) -> Mapping[str, Optional[torch.nn.Module]]:
        # "reward" and "target_reward" are None (left out of checkpoints) unless
        # reward_observation is "end_state".
        return {
            "actor": self.actor,
            "critics": self.critics,
            "target_critics": self.target_critics,
            "value": self.value,
            "target_value": self.target_value,
            "online_plant": self.online_plant,
            "target_plant": self.target_plant,
            "reward": self.reward_net,
            "target_reward": self.target_reward,
        }

    def _optimizers(self) -> Mapping[str, Optional[torch.optim.Optimizer]]:
        return {
            "actor": self.actor_optimizer,
            "critics": self.critic_optimizer,
            "value": self.value_optimizer,
            "plant": self.plant_optimizer,
            "temperature": self.temperature_optimizer,
            "reward": self.reward_optimizer,
        }

    def _loop_state(self) -> Mapping[str, object]:
        state = {"next_clock": self._next_clock.state_dict()}
        if self.end_state:
            # Only under reward_observation "end_state", so that every other checkpoint
            # is as before.
            state["reward_records"] = list(self.reward_records)
            state["last_reward_fit_loss"] = self.last_reward_fit_loss
            state["reward_diagnostic_rng"] = self._reward_log_generator.get_state()
        return state

    def _load_loop_state(self, state: Mapping[str, object]) -> None:
        self._next_clock.load_state_dict(state["next_clock"])
        if self.end_state:
            self.reward_records = list(state["reward_records"])
            self.last_reward_fit_loss = float(state["last_reward_fit_loss"])
            self._reward_log_generator.set_state(state["reward_diagnostic_rng"].cpu())

    def _config(self) -> Mapping[str, object]:
        return self.config
