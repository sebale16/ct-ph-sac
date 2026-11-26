"""Implicit CT-PH-SAC: the three nested loops of Algorithm 3.

Repository defaults beyond the paper (each a config switch, see ImplicitConfig): the
reward rate is a network fitted to observed rows, read through a copy frozen with the
operator; the plant and reward are warm-fitted once warm-up fills a batch; the online
plant is admitted untested at every cycle end; the plant likelihood uses the
leading-order covariance; and log alpha_c takes one step per temperature_every_rows
rows collected after warm-up, in a burst at each outer iteration's end.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import torch

from algorithms.ph_admission import (
    admission_spacing,
    admitted_fit_terms,
    budgeted_gate_width,
    corrected_gate_rows,
    gate_branch,
    gate_interval,
    gate_stage_paths,
    next_gate_counts,
    paired_gate_samples,
    pilot_variance_components,
    transfer_diagnostic,
    untested_admission_result,
)
from algorithms.ph_base import _REWARD_DIAGNOSTIC_SEED_OFFSET, PHAlgorithm
from algorithms.ph_resolvent import (
    all_pairs_loss,
    averaged_draw_loss,
    critic_resolvent_residuals,
    excluded_resolvent_residuals,
    explicit_step_residuals,
    operator_anchor_rates,
    resolvent_residuals,
    solver_status,
    solver_stopping_result,
)
from algorithms.ph_targets import (
    build_rate_targets,
    data_rate_target,
    data_rate_weights,
    excluded_rate_targets,
    generator_critic_target,
    generator_rate_target,
    observed_end_reward,
    rollout_interval_reward,
)
from common.ph_config import (
    check_step_ablation,
    decision_mode_setting,
    exclusion_model_step,
    flagged_value_gradient_setting,
    implicit_setting,
    model_integrator_setting,
    reward_observation_setting,
    step_ablation_active,
    trainable_plant_parts_setting,
)
from common.ph_replay import ReplayPartitions
from common.ph_time import interval_temperature
from environment.ph_sac.tasks import (
    terminal_boundary_functions,
    terminal_payoff_tensor,
)
from models.ph_sac.losses import (
    critic_loss,
    explicit_actor_loss,
    implicit_actor_loss,
    plant_likelihood_loss,
    rate_regression_loss,
    temperature_loss,
)
from models.ph_sac.networks import DecisionActor, RateHead, SmoothValueHead, TwinQCritic
from models.ph_sac.reward import RewardRate, reward_fit_rows, reward_rate_loss
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

    from common.ph_config import ImplicitConfig
    from common.ph_types import (
        AnalysisContext,
        DecisionSample,
        GateResult,
        Metrics,
        RewardFitRecord,
        SolverResult,
        Transition,
        TransitionBatch,
    )
    from environment.ph_sac.base import MujocoPHEnv
    from models.ph_sac.losses import RateTargets
    from models.ph_sac.plant import Plant, PortHamiltonianPlant
    from algorithms.ph_admission import PairedGateSamples
    from algorithms.ph_resolvent import FrozenOperator
    from algorithms.ph_training import OptimizerStepTrace, OuterStepTrace

# Replay states that score the transfer diagnostic J of an untested admission.
_TRANSFER_STATES = 256
# The learned reward's diagnostic draws and seed offsets are algorithms/ph_base's
# (_REWARD_DIAGNOSTIC_DRAWS, _REWARD_INIT_SEED_OFFSET, _REWARD_DIAGNOSTIC_SEED_OFFSET),
# shared with the explicit arms under reward_observation "end_state".
# The value-step ablation's divergence monitors of the K-th inner step (_monitor_value_step),
# written into the step-ablation arms' solver records (diagnostics.OptionalSolverFields).
_VALUE_MONITORS = ("value_loss", "residual_mean_square", "value_abs_max", "value_gradient_norm_max")


class ImplicitCTPHSAC(PHAlgorithm):
    """Solve V with a frozen operator, fit q, improve policy, then gate plant admissions.

    Version counters name the iterates: outer iteration k solves V_{k+1} from V_k under
    pi_k, then fits the rate head to V_{k+1} and steps the actor to pi_{k+1}.
    Decisions are held torques when the environment and the config's testbed are in
    torque-decision mode (the implicit_torque paper arm and the value-step ablation arms):
    actor, rate head or critic, residual draws, rollouts and the reward's torques then use
    the m-dimensional torque box and the identity controller; or feedback laws (the
    impedance-law and implicit-setting ablation arms).

    Contact exclusion (config contact_handling "exclude", self.exclusion active): the plant
    is fitted on unflagged rows only (_plant_fit_batch), the value residual of a flagged
    row is observed_interval_residuals' (update_value), the rate head's targets at
    flagged rows and model points are the observed interval's (_rate_targets,
    ph_targets.excluded_rate_targets), model paths are truncated near a limit, flagged
    gate rows carry no evidence, and the transfer score J reads unflagged states. The
    actor and the temperature read no plant. Config flagged_value_gradient
    "stop_successor" (common.ph_config.FLAGGED_VALUE_GRADIENT_KEY; absent: "full") cuts the
    flagged residual's gradient through V(s') in update_value's default value update.

    The value-step ablation (common.ph_config.STEP_ABLATION_DEFAULTS; the arms of
    protocol.step_ablation_arms), every key at its default the update above: value_step
    "explicit" fits V_(k+1) to one explicit Euler step V_k + h rate(V_k) of the operator
    the resolvent step solves (h = tau or each row's Delta; implicit_explicit_step and
    implicit_explicit_step_interval); decision_head "critic" replaces the rate head by
    explicit_generator's twin critic, fitted after each copy in the rate head's place
    (fit_critic) and read by the actor, with the resolvent step and learned reward
    unchanged under value_operator "model" (implicit_critic_head), and with value_operator
    "critic" (implicit_critic) also read by the resolvent step through (Q - V) / Delta_ref,
    with no learned reward. Those arms add divergence monitors to their solver records
    (_record_solver).

    reward_observation "end_state" (config key, common.ph_config.REWARD_OBSERVATION_KEY;
    needs reward_source "learned"): the replay hands out end-state rates in place of
    interval rewards, the reward steps (warm fit and one per inner step, unchanged in
    number) fit the end-state regression (update_reward), implicit_critic learns a reward
    rate too (its critic target needs one), and every site that read an interval reward
    reads the frozen copy integrated over the interval plus the failure charge: the
    critic head's target (fit_critic), the data and rollout rate targets
    (_model_rate_targets), the flagged rows' targets and residuals (_rate_targets,
    update_value, _ablation_residuals) and the gate's observed return (_gate_samples).
    The schedule, the plant fit and every random stream are as under "interval".
    """

    value: SmoothValueHead
    target_value: SmoothValueHead
    rate: Optional[RateHead]
    critics: Optional[TwinQCritic]
    online_plant: Optional[PortHamiltonianPlant]
    admitted_plant: Plant
    reward_net: Optional[RewardRate]

    def __init__(
        self,
        env: MujocoPHEnv,
        config: ImplicitConfig,
        *,
        replay: Optional[ReplayPartitions] = None,
        oracle: Optional[Plant] = None,
    ) -> None:
        training = config["training"]
        if replay is None:
            replay = ReplayPartitions(
                capacity=training["replay_capacity"],
                validation_fraction=training["validation_fraction"],
                device=training["device"],
                seed=int(training["seed"]),
                reward_observation=reward_observation_setting(config),
            )
        super().__init__(env, training, replay)
        self._check_decision_mode(config)
        self._check_input_matrix_start(config)
        self._configure_reward_observation(config)
        self.config = config
        self._oracle = oracle
        # The step of every model integration: model_step, or under contact exclusion the
        # task's coarse step (common.ph_config.exclusion_model_step).
        self.model_step = exclusion_model_step(config)
        # The substep map of the plant fit, the rate-head rollouts and the gate paths.
        self.model_integrator = model_integrator_setting(config)
        # Contact-triggered fine substepping (config model_coarse_step), None when off.
        self.contact = model_contact_substepping(config, self.state_spec, self.controller)
        # Contact exclusion's rule (config contact_handling), None without limited ranges.
        self.exclusion = model_contact_exclusion(config, env)
        # The flagged rows' value gradient (config flagged_value_gradient; absent: "full").
        self.flagged_value_gradient = flagged_value_gradient_setting(config)
        self.solver = config["solver"]
        self.rate_config = config["rate"]
        self.admission = config["admission"]
        self.resolvent_step = float(self.solver["resolvent_step"])
        self.contraction = 1.0 / (1.0 + self.discount_rate * self.resolvent_step)
        self.reward_source = str(implicit_setting(config, "reward_source"))
        if self.end_state and self.reward_source != "learned":
            raise ValueError(
                "reward_observation 'end_state' learns the reward rate from end-state rates: "
                "it needs reward_source 'learned', not "
                f"{self.reward_source!r}."
            )
        self.reward_warm_steps = int(implicit_setting(config, "reward_warm_steps"))
        self.plant_warm_steps = int(implicit_setting(config, "plant_warm_steps"))
        # The plant fit and the plant diagnostics' calibration share this covariance.
        self.covariance_mode = str(implicit_setting(config, "plant_fit_covariance"))
        self.temperature_every_rows = float(
            implicit_setting(config, "temperature_every_rows")
        )
        # The value-step ablation (common.ph_config.STEP_ABLATION_DEFAULTS); with every
        # choice at its default, default_value_update holds and nothing below changes.
        step = check_step_ablation(config)
        self.value_step = step["value_step"]
        self.explicit_step_size = step["explicit_step_size"]
        self.decision_head = step["decision_head"]
        self.value_operator = step["value_operator"]
        self.critic_reference_interval = step["critic_reference_interval"]
        self.default_value_update = (
            self.value_step == "implicit" and self.value_operator == "model"
        )
        self.step_monitors = step_ablation_active(step)
        # The K-th inner step's monitors, consumed by the solve's record (_record_solver).
        self._solve_monitor: Optional[Dict[str, float]] = None
        self._counters["reward_gradient_steps"] = 0
        self.initialize()

    def _build_components(self) -> None:
        """Initialize the terminal-constrained value, hard target and admitted copy.

        decision_head "critic" builds explicit_generator's twin critic where the rate head
        is built otherwise (the same place in the initialization stream).
        """
        network = self.config["network"]
        controller = self.testbed["controller"]
        low = torch.as_tensor(np.asarray(controller["decision_low"]))
        high = torch.as_tensor(np.asarray(controller["decision_high"]))
        self.actor = DecisionActor(self.state_spec, low, high, network).to(self.device)
        self.rate = None
        self.critics = None
        if self.decision_head == "critic":
            self.critics = TwinQCritic(
                self.state_spec,
                self.actor.decision_dim,
                network,
                decision_low=low,
                decision_high=high,
            ).to(self.device)
        else:
            self.rate = RateHead(
                self.state_spec,
                self.actor.decision_dim,
                network,
                decision_low=low,
                decision_high=high,
            ).to(self.device)
        self.value = SmoothValueHead(self.state_spec, network).to(self.device)
        self.value.configure_terminal_boundary(
            *terminal_boundary_functions(self.testbed)
        )
        self.target_value = self.value.frozen_copy()
        # The online plant's parameters its fit moves: every one, or the parts the config's
        # trainable_plant_parts names (the others frozen; PortHamiltonianPlant.
        # trainable_parameters).
        self.trainable_plant_parts = trainable_plant_parts_setting(self.config)
        self._plant_parameters: List[torch.nn.Parameter] = []
        if self.config["plant_source"] == "oracle":
            self.online_plant = None
            self.admitted_plant = self._oracle_plant(self._oracle)
            self.plant_optimizer = None
        else:
            self.online_plant = self._build_learned_plant(network)
            self.admitted_plant = self.online_plant.frozen_copy()
            self._plant_parameters = self.online_plant.trainable_parameters(
                self.trainable_plant_parts
            )
            self.plant_optimizer = self._adam(self._plant_parameters)
        self.actor_optimizer = self._adam(list(self.actor.parameters()))
        self.value_optimizer = self._adam(list(self.value.parameters()))
        self.rate_optimizer = (
            None if self.rate is None else self._adam(list(self.rate.parameters()))
        )
        self.critic_optimizer = (
            None if self.critics is None else self._adam(list(self.critics.parameters()))
        )
        self._configure_temperature(self.config["temperature"])
        self._build_reward(network, controller)

        self._gate_rng = np.random.default_rng(int(self._rng.integers(2**31)))
        self.base_spacing = admission_spacing(
            discount_rate=self.discount_rate,
            resolvent_step=self.resolvent_step,
            efolds=float(self.admission["efolds"]),
        )
        self.cycle_length = self.base_spacing
        self.outer_in_cycle = 0
        self.outer_iteration = 0
        self.inner_steps = 0
        self.admission_index = 0
        self.plant_version = 0
        self.policy_version = 0
        self.value_version = 0
        self.budget_stopped_outer_iterations = 0
        self.gate_look_index = 0
        self._pending_motion = 0.0
        self.warm_fit_done = False
        self.tuning_rows = 0
        self.tuning_steps = 0
        # True only inside warmup(), whose rows tuning_rows leaves out; not saved.
        self._collecting_warmup = False
        self._last_gate: Optional[Mapping[str, object]] = None
        self._completed_trace: Optional[OuterStepTrace] = None
        self._step_traces: Dict[str, List[OptimizerStepTrace]] = {}

    # --- schedule ------------------------------------------------------------

    def _build_reward(self, network: Mapping[str, object], controller: Mapping[str, object]) -> None:
        """Build the learned reward rate, seeded apart from the global stream, or none.

        Its output is a sigmoid onto the range the testbed's TestbedSpec declares in
        reward_rate_bounds ((0, 1) on the dm_control tolerance rewards), else linear
        (PHAlgorithm._learned_reward_rate). The frozen copies are deep copies, so they carry
        the same output. value_operator "critic" learns none under reward_observation
        "interval": the critic carries the observed rewards; under "end_state" it learns one,
        which its critic target integrates (fit_critic).
        """
        self.reward_net = None
        self.reward_optimizer: Optional[torch.optim.Optimizer] = None
        self._reward_frozen: Optional[RewardRate] = None
        self.reward_records: List[RewardFitRecord] = []
        self.last_reward_fit_loss = float("nan")
        self._reward_log_generator = torch.Generator().manual_seed(
            self.seed + _REWARD_DIAGNOSTIC_SEED_OFFSET
        )
        if self.reward_source != "learned" or (
            self.value_operator == "critic" and not self.end_state
        ):
            return
        self.reward_net = self._learned_reward_rate(network, controller)
        self.reward_optimizer = self._adam(list(self.reward_net.parameters()))

    def _run(self) -> None:
        self.warm_fit()
        while self._budget_left():
            self.run_admission_cycle(
                remaining_interactions=self._budget - self.interactions
            )

    def warm_fit(self) -> Metrics:
        """Once warm-up fills a batch, fit plant and reward before the first value step.

        The online plant takes plant_warm_steps steps and is then admitted untested; the
        reward takes reward_warm_steps steps. An operator frozen before the train
        partition held a batch has served no value step yet, so its reward copy is
        retaken. Later calls do nothing.
        """
        if self.warm_fit_done or not self._train_ready():
            return {}
        self.warm_fit_done = True
        size = self.training["batch_size"]
        metrics: Metrics = {}
        if self.online_plant is not None and self.plant_warm_steps > 0:
            for _ in range(self.plant_warm_steps):
                metrics.update(self.update_online_plant(self.replay.sample("train", size)))
            self.admitted_plant.load_state_dict(self.online_plant.state_dict())
            self.plant_version += 1
        if self.reward_net is not None and self.reward_warm_steps > 0:
            for _ in range(self.reward_warm_steps):
                metrics.update(self.update_reward(self.replay.sample("train", size)))
            if self._reward_frozen is not None:
                self._reward_frozen = self._frozen_reward()
        return metrics

    def _uses_gate(self) -> bool:
        return (
            self.online_plant is not None
            and self.admission["plant_update"] == "admission"
        )

    def _frozen_reward(self) -> RewardRate:
        return copy.deepcopy(self.reward_net).requires_grad_(False)

    def freeze_operator(self) -> FrozenOperator:
        """Freeze V_k, pi_k, alpha_c, admitted plant and reward copy for one inner solve.

        The operator's reward_rate is _reward_rate, which reads the reward copy taken
        here until the next freeze; only the reward loss moves the live network. A solve
        resumed from a checkpoint mid-way (inner_steps > 0) keeps the copy it started with.
        value_operator "critic" adds a frozen copy of the twin critic and Delta_ref; the
        critic steps only after the solve (fit_critic), so a copy taken on a resumed solve
        equals the one its start took.
        """
        if self.reward_net is not None and (
            self._reward_frozen is None or self.inner_steps == 0
        ):
            self._reward_frozen = self._frozen_reward()
        operator: FrozenOperator = {
            "target_value": self.target_value,
            "policy": self.actor.frozen_copy(),
            "plant": self.admitted_plant,
            "temperature_per_second": self.alpha_c.clone(),
            "discount_rate": self.discount_rate,
            "resolvent_step": self.resolvent_step,
            "reward_rate": self._reward_rate,
        }
        if self.value_operator == "critic":
            operator["critics"] = self.critics.frozen_copy()
            operator["reference_interval"] = float(self.critic_reference_interval)
        return operator

    def _reward_rate(self, states: Tensor, torques: Tensor) -> Tensor:
        """The task's rate, or the learned rate's copy frozen with the current operator.

        Before the first freeze the live network stands in.
        """
        if self.reward_net is None:
            return super()._reward_rate(states, torques)
        network = self.reward_net if self._reward_frozen is None else self._reward_frozen
        return network(states, torques)

    def run_admission_cycle(self, *, remaining_interactions: int) -> Metrics:
        """Run the scheduled outer iterations, then update the admitted plant.

        plant_update "always" admits the online plant untested, "admission" runs the
        staged gate, and "polyak" and the oracle arm do nothing here. A budget stop
        inside the cycle keeps its position, so a later learn call resumes it.
        """
        limit = self.interactions + int(remaining_interactions)
        while self.outer_in_cycle < self.cycle_length:
            remaining = limit - self.interactions
            if remaining <= 0:
                return {"gate_tested": 0.0}
            result = self.run_outer_iteration(remaining_interactions=remaining)
            if not result["completed"]:
                return {"gate_tested": 0.0}
        if self.online_plant is None or self.admission["plant_update"] == "polyak":
            self.outer_in_cycle = 0
            return {"gate_tested": 0.0}
        if self.admission["plant_update"] == "always":
            admitted = self.admit_without_test()
            self.update_admitted_plant(admitted)
            self.outer_in_cycle = 0
            return {"gate_tested": 0.0, "gate_accepted": 1.0}
        # The cycle stays complete until the attempt ends, because acquiring real rows
        # can fire a checkpoint and a resumed run must repeat the attempt.
        gate = self.test_admission()
        self.update_admitted_plant(gate)
        self.outer_in_cycle = 0
        return {"gate_tested": 1.0, "gate_accepted": float(gate["accepted"])}

    def run_outer_iteration(self, *, remaining_interactions: int) -> SolverResult:
        """Solve, hard-copy V, fit rates, step actor and temperature, preserving the budget.

        When the budget ends the solve early, the copy still happens and nothing else is
        fitted: the rate head and actor would otherwise read a value solved for fewer than K
        steps.
        """
        operator = self.freeze_operator()
        if self.inner_steps == 0:
            # A solve restarted mid-way (resume_state) keeps the traces it began with.
            self._step_traces = {}
        result = self.solve_value(operator, remaining_interactions=remaining_interactions)
        start_candidate = self._candidate_id()
        start_policy = self._policy_id()
        self.copy_solved_value()
        self._record_solver(result)
        if not result["completed"]:
            self.budget_stopped_outer_iterations += 1
            return result
        solved_context = self._context(self._candidate_id(), start_policy)
        if self._train_ready():
            fitted = self.fit_rate_head(int(self.rate_config["rate_steps"]))
            self.update_actor(int(self.rate_config["actor_steps"]))
            self.tune_temperature()
            self._record_fit_monitor(fitted)
        self._store_trace(self._context(start_candidate, start_policy), solved_context)
        self.outer_in_cycle += 1
        self.outer_iteration += 1
        return result

    def solve_value(
        self, operator: FrozenOperator, *, remaining_interactions: int
    ) -> SolverResult:
        """Collect per inner step and stop after K steps or at interaction exhaustion.

        self.inner_steps survives a checkpoint taken between inner iterations. Iterations
        before a full training batch exists collect only and do not count toward K.
        The top of each iteration is a safe point (_resume_point): a restart there
        refreezes an identical operator. The online plant steps on its own batch when the
        plant-step schedule says so (_plant_step_due); a skipped row draws no batch. The
        known-plant arm draws its (unused) batch on every row, as before the schedule.
        """
        limit = self.interactions + int(remaining_interactions)
        steps_per_solve = int(self.solver["inner_steps"])
        while True:
            if self.interactions >= limit or (
                self._budget is not None and not self._budget_left()
            ):
                final: SolverResult = {"inner_steps": self.inner_steps, "completed": False}
                self.inner_steps = 0
                return final
            self._resume_point()
            result: Optional[SolverResult] = None
            self.collect_transition()
            if self._train_ready():
                self.warm_fit()
                self.inner_steps += 1
                size = self.training["batch_size"]
                self.update_value(self.replay.sample("train", size), operator)
                if self.online_plant is None or self._plant_step_due():
                    self.update_online_plant(self.replay.sample("train", size))
                if self.inner_steps >= steps_per_solve:
                    result = self.check_solver(inner_steps=self.inner_steps)
            self._after_interaction()
            if result is not None:
                self.inner_steps = 0
                return result

    def update_value(self, batch: TransitionBatch, operator: FrozenOperator) -> Metrics:
        """Optimize only psi through the all-pairs residual (averaged-draw ablation).

        A learned reward first takes one step on the same batch; the residual reads the
        operator's frozen reward copy, so the order of the two steps does not matter.
        Under the value-step ablation the residuals are _ablation_residuals', with the
        same draws and estimator, and the K-th inner step records its monitors. Under
        reward_observation "end_state" a flagged row's observed-interval residual reads
        the operator's reward rate's observed-ends trapezoid plus the failure charge
        (_flagged_end_rewards) in place of the interval reward. Under flagged_value_gradient
        "stop_successor" that residual's gradient reaches psi through V(s) alone
        (ph_resolvent.observed_interval_residuals stop_successor), in either reward datum.
        """
        metrics = self.update_reward(batch)
        states = batch["states"]
        with torch.no_grad():
            sample = self._policy_sample(
                states,
                sample_count=int(self.solver["policy_draws"]),
                actor=operator["policy"],
            )
        if not self.default_value_update:
            residuals = self._ablation_residuals(batch, operator, sample)
        elif self.exclusion_active:
            flags = batch_contact_flags(batch)
            residuals = excluded_resolvent_residuals(
                self.value,
                operator,
                self.controller,
                batch,
                sample,
                flags,
                create_graph=True,
                rewards=self._flagged_end_rewards(batch, flags, operator["reward_rate"]),
                stop_successor=self.flagged_value_gradient == "stop_successor",
            )
        else:
            residuals = resolvent_residuals(
                self.value, operator, self.controller, states, sample, create_graph=True
            )
        loss = (
            all_pairs_loss(residuals)
            if self.solver["residual_estimator"] == "all_pairs"
            else averaged_draw_loss(residuals)
        )
        value = self._optimizer_step(
            "value", self.value_optimizer, list(self.value.parameters()), loss
        )
        if self.step_monitors and self.inner_steps >= int(self.solver["inner_steps"]):
            self._monitor_value_step(states, residuals, value)
        return {**metrics, "value_loss": value}

    def _ablation_residuals(
        self, batch: TransitionBatch, operator: FrozenOperator, sample: DecisionSample
    ) -> Tensor:
        """The value-step ablation's residuals per draw [B,N,1] (STEP_ABLATION_DEFAULTS).

        value_step "implicit" (then value_operator "critic"): critic_resolvent_residuals,
        R_i = V - V_k - tau [(Q_i - V) / Delta_ref - alpha_c log pi_i]. value_step
        "explicit": explicit_step_residuals, R_i = V - [V_k + h rate_i(V_k)], rate_i(V_k)
        the operator at the frozen value (operator_anchor_rates: the model operator, with
        contact exclusion's observed interval at flagged rows, or the critic's) and h from
        _explicit_steps. flagged_value_gradient changes nothing here: the explicit step's
        flagged rate reads V_k(s'), which carries no gradient, and the critic operator
        reads no observed interval.
        """
        states = batch["states"]
        if self.value_step == "implicit":
            return critic_resolvent_residuals(
                self.value, operator, states, sample, create_graph=True
            )
        flags = batch_contact_flags(batch) if self.exclusion_active else None
        rewards = None
        if "critics" not in operator:
            rewards = self._flagged_end_rewards(batch, flags, operator["reward_rate"])
        rates = operator_anchor_rates(
            operator, self.controller, batch, sample, flags, rewards=rewards
        )
        return explicit_step_residuals(
            self.value,
            operator,
            states,
            rates,
            self._explicit_steps(batch),
            create_graph=True,
        )

    def _flagged_end_rewards(
        self,
        batch: TransitionBatch,
        flags: Optional[Tensor],
        reward_rate: Callable[[Tensor, Tensor], Tensor],
    ) -> Optional[Tensor]:
        """The rewards of contact exclusion's observed-interval residual under
        reward_observation "end_state": r_obs + c of every row of the batch
        (PHAlgorithm._observed_end_rewards of reward_rate), the residual selecting the
        flagged ones. None (the batch's interval rewards, read only at flagged rows) under
        "interval" or when no row is flagged."""
        if not self.end_state or flags is None or not bool(flags.any()):
            return None
        return self._observed_end_rewards(batch, reward_rate)

    def _explicit_steps(self, batch: TransitionBatch) -> Union[float, Tensor]:
        """h of the explicit step: tau (explicit_step_size "resolvent_step"), or each
        row's elapsed interval Delta as [B,1,1] ("interval", explicit_generator's step)."""
        if self.explicit_step_size == "resolvent_step":
            return self.resolvent_step
        return batch["durations"].reshape(-1, 1, 1)

    def _monitor_value_step(self, states: Tensor, residuals: Tensor, loss: float) -> None:
        """Keep the K-th inner step's monitors for the solve's record (_record_solver).

        value_loss: that step's loss; residual_mean_square: mean_s (mean_i R_i)^2 before
        the step (for the explicit step the squared distance of V to the averaged explicit
        target); value_abs_max and value_gradient_norm_max: max |V| and max |grad_s V| of
        the stepped value (the one about to be copied) on the batch's states. No random
        stream is read and no parameter gradient is touched; the monitors live only from
        the K-th step to the record, which no safe point separates.
        """
        with torch.no_grad():
            square = float(residuals.detach().squeeze(-1).mean(dim=1).square().mean())
        with torch.enable_grad():
            points = states.detach().requires_grad_(True)
            values = self.value(points)
            gradients = torch.autograd.grad(values.sum(), points)[0]
        self._solve_monitor = {
            "value_loss": float(loss),
            "residual_mean_square": square,
            "value_abs_max": float(values.detach().abs().max()),
            "value_gradient_norm_max": float(gradients.detach().norm(dim=-1).max()),
        }

    def update_reward(self, batch: TransitionBatch) -> Metrics:
        """One Adam step of the learned reward on the batch's rows that did not fail.

        The trapezoid of the rate at the controller's torques at both interval ends
        fits each row's reward; failed rows are skipped and truncated rows kept. Every
        reward_diagnostics_every steps the fit is compared with the true rate. Under
        reward_observation "end_state" the step is the end-state regression on every row
        instead (PHAlgorithm._end_state_reward_step).
        """
        if self.reward_net is None:
            return {}
        if self.end_state:
            return self._end_state_reward_step(batch)
        rows = reward_fit_rows(self.controller, batch)
        if rows is None:
            return {}
        states, start, next_states, end, durations, rewards = rows
        loss = reward_rate_loss(
            self.reward_net,
            states,
            start,
            next_states,
            end,
            durations,
            rewards,
            discount_rate=self.discount_rate,
        )
        self.reward_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.reward_optimizer.step()
        self._counters["reward_gradient_steps"] += 1
        self.last_reward_fit_loss = float(loss.detach())
        every = self.reward_diagnostics_every
        if every and int(self._counters["reward_gradient_steps"]) % every == 0:
            self.reward_records.append(self.reward_fit_diagnostics(states, start))
        return {"reward_fit_loss": self.last_reward_fit_loss}

    # reward_fit_diagnostics, the learned reward's sampled diagnostic, is PHAlgorithm's.

    def update_online_plant(self, batch: TransitionBatch) -> Metrics:
        """Fit the endpoint likelihood with plant_fit_covariance; admitted stays fixed.

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
            covariance_mode=self.covariance_mode,
            create_graph=True,
            integrator=self.model_integrator,
            contact=self.contact,
        )
        loss = plant_likelihood_loss(batch, moments, self.state_spec)
        value = self._optimizer_step(
            "model", self.plant_optimizer, self._plant_parameters, loss
        )
        if self.admission["plant_update"] == "polyak":
            self.update_polyak_plant_ablation()
        return {"plant_loss": value}

    def check_solver(self, *, inner_steps: int) -> SolverResult:
        """Report whether inner_steps has reached K, the fixed step count of every solve."""
        return solver_stopping_result(
            inner_steps=inner_steps, steps_per_solve=int(self.solver["inner_steps"])
        )

    def copy_solved_value(self) -> None:
        """Hard copy psi to target psi after the inner loop, including budget exits."""
        self.target_value.load_state_dict(self.value.state_dict())
        self.value_version += 1

    def _record_solver(self, result: SolverResult) -> None:
        # J is logged as a sampled input only: Lemma moving's running bound needs actual
        # solver and motion bounds, and sampled diagnostics must not stand in for them.
        motion = self._pending_motion
        self._pending_motion = 0.0
        record = {
            "outer_iteration": self.outer_iteration,
            "inner_iterations": result["inner_steps"],
            "status": solver_status(result),
            "contraction_factor": self.contraction,
            "fixed_point_motion_input": motion,
        }
        if self.step_monitors:
            record.update(self._solve_monitor_record())
        self.solver_records.append(record)

    def _solve_monitor_record(self) -> Dict[str, float]:
        """The step-ablation arms' monitors of this solve (diagnostics.OptionalSolverFields):
        the K-th inner step's _VALUE_MONITORS (NaN after a budget exit), and the fitted
        head's loss, rate_loss or critic_loss, NaN until _record_fit_monitor writes it."""
        monitor = self._solve_monitor or {}
        self._solve_monitor = None
        record = {name: float(monitor.get(name, math.nan)) for name in _VALUE_MONITORS}
        record["critic_loss" if self.critics is not None else "rate_loss"] = math.nan
        return record

    def _record_fit_monitor(self, fitted: Metrics) -> None:
        """Write the last step's rate_loss or critic_loss of this outer iteration's head
        fit into its solver record (step-ablation arms only)."""
        if not self.step_monitors or not self.solver_records:
            return
        name = "critic_loss" if self.critics is not None else "rate_loss"
        if name in fitted:
            self.solver_records[-1][name] = float(fitted[name])

    # --- rate head, actor, temperature ---------------------------------------

    def fit_rate_head(self, gradient_steps: int) -> Metrics:
        """Fit replay, held-interval rollout and generator rates against the copied value.

        With decision_head "critic" the twin critic takes these steps instead (fit_critic).
        """
        if self.critics is not None:
            return self.fit_critic(gradient_steps)
        metrics: Metrics = {}
        config = self.rate_config
        for _ in range(int(gradient_steps)):
            batch = self.replay.sample("train", self.training["batch_size"])
            model_batch = self.replay.sample("train", int(config["model_points"]))
            model_states = model_batch["states"]
            with torch.no_grad():
                model_decisions = self._policy_sample(model_states)["decisions"][:, 0]
            if self.exclusion_active:
                # A flagged model point is scored at its row's observed decision, whose
                # interval gives its target (ph_targets.excluded_rate_targets).
                model_flags = batch_contact_flags(model_batch).reshape(-1, 1)
                model_decisions = torch.where(
                    model_flags, model_batch["decisions"], model_decisions
                )
            targets = self._rate_targets(
                batch, model_states, model_decisions, model_batch=model_batch
            )
            loss = rate_regression_loss(
                self.rate,
                batch,
                targets,
                model_states,
                model_decisions,
                rollout_weight=float(config["rollout_weight"]),
                model_weight=float(config["model_weight"]),
            )
            metrics["rate_loss"] = self._optimizer_step(
                "rate", self.rate_optimizer, list(self.rate.parameters()), loss
            )
        return metrics

    def _rate_targets(
        self,
        batch: TransitionBatch,
        model_states: Tensor,
        model_decisions: Tensor,
        *,
        model_batch: Optional[TransitionBatch] = None,
    ) -> RateTargets:
        """Build the targets, running no rollouts when their weight is zero (lambda_r=0).

        Under contact exclusion (model_batch: the replay rows of model_states) the targets
        at flagged rows and model points are the observed interval's
        (ph_targets.excluded_rate_targets); a flagged row is not rolled out and a path
        that comes near a limit is truncated (stochastic_rollout), its continuation the
        observed endpoint's.
        """
        targets = self._model_rate_targets(batch, model_states, model_decisions)
        if not self.exclusion_active:
            return targets
        if model_batch is None:
            raise ValueError("Contact exclusion's rate targets need the model points' rows.")
        model_flags = batch_contact_flags(model_batch)
        model_rewards = None
        if self.end_state and bool(model_flags.any()):
            # A flagged model point's data target, model-free: the observed ends'
            # trapezoid of the frozen reward copy plus the failure charge.
            model_rewards = self._observed_end_rewards(model_batch, self._reward_rate)
        return excluded_rate_targets(
            targets,
            self.target_value,
            batch,
            model_batch,
            batch_contact_flags(batch),
            model_flags,
            config=self.rate_config,
            discount_rate=self.discount_rate,
            model_rewards=model_rewards,
        )

    def _model_rate_targets(
        self, batch: TransitionBatch, model_states: Tensor, model_decisions: Tensor
    ) -> RateTargets:
        """The data, rollout and generator targets of build_rate_targets.

        Under reward_observation "end_state" the data target's reward is the observed
        ends' trapezoid of the frozen reward copy plus the failure charge, and the rollout
        target's the survivor mean of the paths' discounted trapezoids of the same copy on
        their own knots plus the charge (the rollout then carries ReturnTerms of the copy;
        PHAlgorithm._observed_end_rewards, ph_targets.rollout_interval_reward).
        """
        config = self.rate_config
        if float(config["rollout_weight"]) == 0.0:
            data = data_rate_target(
                self.target_value,
                batch,
                discount_rate=self.discount_rate,
                rewards=(
                    self._observed_end_rewards(batch, self._reward_rate)
                    if self.end_state
                    else None
                ),
            )
            return {
                "data": data,
                "rollout": data,
                "generator": generator_rate_target(
                    self.admitted_plant,
                    self.target_value,
                    self.controller,
                    model_states,
                    model_decisions,
                    self._reward_rate,
                    discount_rate=self.discount_rate,
                ),
                "data_weights": data_rate_weights(
                    self.admitted_plant,
                    self.target_value,
                    batch["states"],
                    batch["durations"],
                    weight_max=config["replay_weight_max"],
                    noise_variance_floor=config["noise_variance_floor"],
                ),
            }
        paths_per_row = int(config["rollout_paths"])
        noise = sample_brownian_increments(
            batch["durations"],
            path_count=paths_per_row,
            noise_dim=self.admitted_plant.noise_dim,
            maximum_step=self.model_step,
            physics_dt=self.physics_dt,
            controller_refresh_steps=self.refresh_steps,
            seed=self._rollout_seed(),
        ).to(self.device)
        paths = stochastic_rollout(
            self.admitted_plant,
            self.controller,
            batch["states"],
            batch["decisions"],
            batch["durations"],
            self._failure_predicate,
            maximum_step=self.model_step,
            physics_dt=self.physics_dt,
            controller_refresh_steps=self.refresh_steps,
            path_count=paths_per_row,
            brownian_increments=noise,
            integrator=self.model_integrator,
            contact=self.contact,
            **self._truncation(batch),
            **(
                {"return_terms": self._reward_return_terms(self._reward_rate)}
                if self.end_state
                else {}
            ),
        )
        self._counters["model_rollout_paths"] += batch["states"].shape[0] * paths_per_row
        rewards: Dict[str, Tensor] = {}
        if self.end_state:
            observed = observed_end_reward(
                self.controller, batch, self._reward_rate, discount_rate=self.discount_rate
            )
            charges = self._failure_charges(batch)
            rewards = {
                "data_rewards": observed + charges,
                "rollout_rewards": rollout_interval_reward(paths, observed) + charges,
            }
        return build_rate_targets(
            self.admitted_plant,
            self.target_value,
            self.controller,
            batch,
            paths,
            model_states,
            model_decisions,
            self._reward_rate,
            config=config,
            discount_rate=self.discount_rate,
            **rewards,
        )

    def _truncation(self, batch: TransitionBatch) -> Mapping[str, object]:
        """stochastic_rollout's contact-exclusion arguments for batch's rows: truncation
        near a limit and the flagged rows skipped; none when exclusion is off."""
        if not self.exclusion_active:
            return {}
        return {"truncation": self.exclusion, "skip_rows": batch_contact_flags(batch)}

    def fit_critic(self, gradient_steps: int) -> Metrics:
        """Fit the twin critic in the rate head's place (decision_head "critic").

        Each step draws a fresh training batch and regresses both critics on the
        explicit_generator arm's target r + (1-d) [V + Delta (G^kappa V - lambda V)] at the
        taken decision, each row's own Delta, V the copied value and G the admitted plant's,
        the rate head's value and plant (generator_critic_target; contact exclusion's
        observed continuation at flagged rows). Reports the last step's critic_loss. Under
        reward_observation "end_state" r is Delta r_hat(s, u_0) + c of the frozen reward
        copy, the observed ends' trapezoid + c at flagged rows
        (PHAlgorithm._generator_rewards).
        """
        metrics: Metrics = {}
        for _ in range(int(gradient_steps)):
            batch = self.replay.sample("train", self.training["batch_size"])
            flags = batch_contact_flags(batch) if self.exclusion_active else None
            targets = generator_critic_target(
                self.admitted_plant,
                self.target_value,
                self.controller,
                batch,
                discount_rate=self.discount_rate,
                flags=flags,
                rewards=(
                    self._generator_rewards(batch, self._reward_rate, flags)
                    if self.end_state
                    else None
                ),
            )
            loss = critic_loss(self.critics, batch["states"], batch["decisions"], targets)
            metrics["critic_loss"] = self._optimizer_step(
                "critic", self.critic_optimizer, list(self.critics.parameters()), loss
            )
        return metrics

    def update_actor(self, gradient_steps: int) -> Metrics:
        """Only chi receives gradients through fixed q_omega; use alpha_c per second.

        With decision_head "critic" the actor reads the twin critic's minimum with the
        interval temperature alpha_c w_lambda(Delta) of each row, as the explicit arms do.
        """
        metrics: Metrics = {}
        for _ in range(int(gradient_steps)):
            batch = self.replay.sample("train", self.training["batch_size"])
            states = batch["states"]
            if self.critics is None:
                loss = implicit_actor_loss(
                    self.rate, states, self._policy_sample(states), self.alpha_c
                )
            else:
                loss = explicit_actor_loss(
                    self.critics,
                    states,
                    self._policy_sample(states),
                    interval_temperature(
                        self.alpha_c, batch["durations"], discount_rate=self.discount_rate
                    ),
                )
            metrics["actor_loss"] = self._optimizer_step(
                "actor", self.actor_optimizer, list(self.actor.parameters()), loss
            )
        if int(gradient_steps) > 0:
            self.policy_version += 1
        return metrics

    def warmup(self, episodes: int) -> None:
        """PHAlgorithm.warmup, its rows left out of the temperature's row count."""
        self._collecting_warmup = True
        try:
            super().warmup(episodes)
        finally:
            self._collecting_warmup = False

    def _collect(self, decision_source, *, partition=None) -> Transition:
        """Collect one row; a learned temperature is owed a step per R rows once training
        is ready, warm-up rows not counted."""
        transition = super()._collect(decision_source, partition=partition)
        if (
            self.log_temperature is not None
            and not self._collecting_warmup
            and self._train_ready()
        ):
            self.tuning_rows += 1
        return transition

    @property
    def tuning_due(self) -> float:
        """Tuning steps owed: rows collected after warm-up since training was ready over R,
        less those taken."""
        return self.tuning_rows / self.temperature_every_rows - self.tuning_steps

    def tune_temperature(self) -> Metrics:
        """Take every whole owed tuning step, each on a fresh replay batch.

        Called at the end of each completed outer iteration, so the tuning rate is one
        step per temperature_every_rows rows whatever the inner step count K; warm-up
        rows are not counted, so the first outer iteration takes no burst for them.
        """
        if self.log_temperature is None:
            return {}
        steps = int(math.floor(self.tuning_due + 1e-9))
        metrics: Metrics = {}
        for _ in range(steps):
            metrics = self.update_temperature(
                self.replay.sample("train", self.training["batch_size"])
            )
            self.tuning_steps += 1
        return metrics

    def update_temperature(self, batch: TransitionBatch) -> Metrics:
        """One tuning step of log alpha_c on the batch; fixed temperatures do nothing."""
        if self.log_temperature is None:
            return {}
        with torch.no_grad():
            sample = self._policy_sample(batch["states"])
        log_probabilities = sample["log_probabilities"]
        loss = temperature_loss(
            self.log_temperature,
            log_probabilities,
            self.temperature_config["target_entropy"],
            interval_weights=torch.ones_like(batch["durations"]),
        )
        value = self._optimizer_step(
            "temperature", self.temperature_optimizer, [self.log_temperature], loss
        )
        return {"temperature_loss": value, "alpha_c": float(self.alpha_c)}

    # --- admissions ----------------------------------------------------------

    def _gate_noise(self, batch: TransitionBatch, paths: int) -> Tensor:
        return sample_brownian_increments(
            batch["horizons"],
            path_count=paths,
            noise_dim=self.online_plant.noise_dim,
            maximum_step=self.model_step,
            physics_dt=self.physics_dt,
            controller_refresh_steps=self.refresh_steps,
            seed=int(self._gate_rng.integers(2**31)),
        ).to(self.device)

    def _gate_samples(self, batch: TransitionBatch, paths: int) -> PairedGateSamples:
        """Fresh paired full returns for both plants, independent of the observed outcome.

        Under reward_observation "end_state" the observed return's reward is the observed
        ends' trapezoid of the frozen reward copy over the elapsed interval plus the failure
        charge (PHAlgorithm._observed_end_rewards), which reads neither plant.
        """
        self._counters["gate_simulated_paths"] += 2 * batch["states"].shape[0] * paths
        observed_rewards = None
        if self.end_state:
            observed_rewards = self._observed_end_rewards(batch, self._reward_rate)
        return paired_gate_samples(
            self.online_plant,
            self.admitted_plant,
            self.target_value,
            self.controller,
            batch,
            failure_predicate=self._failure_predicate,
            return_terms={
                "reward_rate": self._reward_rate,
                "terminal_payoff": lambda states: terminal_payoff_tensor(
                    states, self.testbed
                ),
                "discount_rate": self.discount_rate,
            },
            maximum_step=self.model_step,
            physics_dt=self.physics_dt,
            controller_refresh_steps=self.refresh_steps,
            path_count=paths,
            brownian_increments=self._gate_noise(batch, paths),
            integrator=self.model_integrator,
            contact=self.contact,
            observed_rewards=observed_rewards,
            **self._truncation(batch),
        )

    def _reserve_gate_rows(self, unused_needed: int) -> Optional[int]:
        """Acquire held-out rows for a whole stage, or None if data cannot supply them."""
        missing = unused_needed - self.replay.unused_rows("gate")
        if missing <= 0:
            return 0
        if unused_needed > self.replay.buffers["gate"].capacity or (
            self._budget is not None and self._budget - self.interactions < missing
        ):
            return None
        for _ in range(missing):
            self._collect(self._behavior_decision, partition="gate")
            self._after_interaction()
        self._counters["gate_real_rows_acquired"] += missing
        return missing

    def _production_width(self, samples: PairedGateSamples, rows: int) -> Optional[float]:
        """Count one look; confidence-budgeted mode returns its Hoeffding width."""
        self.gate_look_index += 1
        config = self.admission
        if config["inference"] != "confidence_budgeted":
            return None
        largest = max(
            float(samples[name].abs().max())
            for name in ("observed_returns", "candidate_returns", "admitted_returns")
        )
        if largest > config["return_bound"]:
            raise ValueError(f"Return {largest} breaks the declared return_bound.")
        if float(samples["horizons"].min()) < config["min_horizon"]:
            raise ValueError("A validation horizon is below the declared min_horizon.")
        return budgeted_gate_width(
            rows=rows,
            look=self.gate_look_index,
            gate_failure_budget=config["gate_failure_budget"],
            return_bound=config["return_bound"],
            min_horizon=config["min_horizon"],
        )

    def test_admission(self) -> GateResult:
        """Run the budgeted staged admission on a frozen candidate, value and policy.

        Pilots on reused rows only guide allocation. Every scoring batch uses rows no
        earlier read inspected and fresh simulation, with counts fixed beforehand. The
        fixed-allocation ablation drops pilots and only doubles rows. An attempt that
        cannot afford another complete stage ends inconclusive and keeps the plant.
        """
        config = self.admission
        batches = int(config["pilot_batches"])
        uses_pilot = config["allocation"] == "pilot"
        initial_rows, paths = int(config["initial_rows"]), int(config["initial_paths"])
        rows = initial_rows
        pilot_rows = initial_rows if uses_pilot else 0
        pilot: Optional[TransitionBatch] = None
        real_used = simulated_used = acquired = 0
        status = "inconclusive_at_validation_cap"
        simulation_variance = real_variance = float("nan")
        production: Optional[Mapping[str, object]] = None
        transfer = float("nan")
        stage_log: List[Dict[str, object]] = []

        def affordable(next_rows: int, next_paths: int, new_pilot: int) -> bool:
            cost = gate_stage_paths(
                next_rows, next_paths, pilot_rows=pilot_rows, pilot_batches=batches
            )
            return (
                real_used + new_pilot + next_rows <= config["real_row_budget"]
                and simulated_used + cost <= config["simulated_path_budget"]
            )

        while True:
            new_pilot = pilot_rows if uses_pilot and pilot is None else 0
            if not affordable(rows, paths, new_pilot):
                break
            inspected = self.replay.partition_size("gate") - self.replay.unused_rows("gate")
            added = self._reserve_gate_rows(rows + max(new_pilot - inspected, 0))
            if added is None:
                break
            acquired += added
            real_used += new_pilot + rows
            if uses_pilot:
                if pilot is None:
                    pilot = self.replay.draw_for_allocation("gate", pilot_rows)
                pilot_scores = torch.stack(
                    [
                        corrected_gate_rows(self._gate_samples(pilot, paths))[
                            "corrected_differences"
                        ].reshape(-1)
                        for _ in range(batches)
                    ],
                    dim=1,
                )
                simulation_variance, real_variance = pilot_variance_components(pilot_scores)
            batch = self.replay.draw_unused("gate", rows)
            samples = self._gate_samples(batch, paths)
            simulated_used += gate_stage_paths(
                rows, paths, pilot_rows=pilot_rows, pilot_batches=batches
            )
            scored = corrected_gate_rows(samples)
            interval = gate_interval(
                scored,
                margin_z=float(config["margin_z"]),
                width=self._production_width(samples, rows),
            )
            production = {"batch": batch, "samples": samples, "rows": scored, "interval": interval}
            branch = gate_branch(interval)
            stage_log.append(
                {
                    "rows": rows,
                    "paths": paths,
                    "pilot_rows": pilot_rows,
                    "pilot_variance_simulation": simulation_variance,
                    "pilot_variance_real": real_variance,
                    "corrected_mean": interval["corrected_mean"],
                    "corrected_std": interval["corrected_std"],
                    "width": interval["width"],
                    "branch": branch,
                    "real_rows_left": config["real_row_budget"] - real_used,
                    "simulated_paths_left": config["simulated_path_budget"] - simulated_used,
                }
            )
            if branch != "inconclusive":
                status = branch
                break
            if uses_pilot:
                counts = next_gate_counts(
                    rows,
                    paths,
                    simulation_variance=simulation_variance,
                    real_variance=real_variance,
                    fits=lambda n, l: affordable(n, l, 0),
                )
            else:
                counts = (2 * rows, paths) if affordable(2 * rows, paths, 0) else None
            if counts is None:
                break
            rows, paths = counts

        if status == "supported_improvement":
            transfer = float(
                transfer_diagnostic(
                    self.online_plant,
                    self.admitted_plant,
                    self.target_value,
                    self.actor,
                    self.controller,
                    production["batch"]["states"],
                    discount_rate=self.discount_rate,
                    policy_draws=int(self.solver["policy_draws"]),
                )
            )
            if not math.isfinite(transfer):
                status = "rejected_nonfinite"

        result: GateResult = {
            "status": status,
            "inference_mode": config["inference"],
            "raw_mean": float("nan"),
            "variance_correction": float("nan"),
            "corrected_mean": float("nan"),
            "corrected_std": float("nan"),
            "interval_lower": float("nan"),
            "interval_upper": float("nan"),
            "transfer_diagnostic": transfer,
            "accepted": status == "supported_improvement",
            "validation_rows": 0,
            "paths_per_model": 0,
            "stages": len(stage_log),
            "real_rows_used": real_used,
            "real_rows_acquired": acquired,
            "simulated_paths_used": simulated_used,
            "pilot_variance_simulation": simulation_variance,
            "pilot_variance_real": real_variance,
            "stage_log": stage_log,
        }
        self._last_gate = None
        if production is not None:
            samples, scored = production["samples"], production["rows"]
            interval = production["interval"]
            result.update(
                raw_mean=float(scored["raw_differences"].mean()),
                variance_correction=float(scored["variance_corrections"].mean()),
                corrected_mean=interval["corrected_mean"],
                corrected_std=interval["corrected_std"],
                interval_lower=interval["lower"],
                interval_upper=interval["upper"],
                validation_rows=interval["rows"],
                paths_per_model=samples["candidate_returns"].shape[1],
            )
            squares, variances = admitted_fit_terms(samples)
            self._last_gate = {
                "squares": squares.detach().cpu().numpy(),
                "variances": variances.detach().cpu().numpy(),
                "horizons": samples["horizons"].detach().cpu().numpy(),
                "candidate": samples["candidate_summary"],
                "admitted": samples["admitted_summary"],
                "observed_failure_rate": samples["observed_failure_rate"],
            }
        return result

    def admit_without_test(self) -> GateResult:
        """plant_update=always: record J on replay states; the copy follows regardless.

        No held-out row is read or acquired and no gate path is simulated. Under contact
        exclusion J reads the unflagged states only (NaN if none is).
        """
        sample = self.replay.sample("train", _TRANSFER_STATES)
        states = sample["states"]
        if self.exclusion_active:
            states = states[~batch_contact_flags(sample)]
        transfer = float("nan")
        if states.shape[0] > 0:
            transfer = float(
                transfer_diagnostic(
                    self.online_plant,
                    self.admitted_plant,
                    self.target_value,
                    self.actor,
                    self.controller,
                    states,
                    discount_rate=self.discount_rate,
                    policy_draws=int(self.solver["policy_draws"]),
                )
            )
        self._last_gate = None
        return untested_admission_result(transfer)

    def update_admitted_plant(self, result: GateResult) -> None:
        """Copy an accepted candidate; M outer iterations follow every attempt, whatever J."""
        if result["accepted"]:
            self.admitted_plant.load_state_dict(self.online_plant.state_dict())
            self.plant_version += 1
            self._pending_motion = float(result["transfer_diagnostic"])
        self.cycle_length = self.base_spacing
        self._record_admission(result)
        self.admission_index += 1

    def _record_admission(self, result: GateResult) -> None:
        from evaluations.ph_sac.diagnostics import measure_gate_diagnostics

        self.admission_records.append(
            measure_gate_diagnostics(
                admission_index=self.admission_index,
                outer_iteration=self.outer_iteration,
                result=result,
                production=self._last_gate,
                next_outer_iterations=self.cycle_length,
            )
        )
        self._last_gate = None

    def update_polyak_plant_ablation(self) -> None:
        """Expose the moving-plant ablation separately from the frozen-plant algorithm."""
        self._polyak(
            self.admitted_plant, self.online_plant, float(self.admission["polyak_rate"])
        )

    # --- lineage -------------------------------------------------------------

    def _candidate_id(self) -> str:
        return f"V_{self.value_version}"

    def _policy_id(self) -> str:
        return f"pi_{self.policy_version}"

    def _context(self, candidate: str, policy: str) -> AnalysisContext:
        return {
            "outer_iteration": self.outer_iteration,
            "candidate_id": candidate,
            "policy_id": policy,
            "plant_id": f"admitted_{self.plant_version}",
            "temperature_per_second": float(self.alpha_c),
            "policy_kind": "stochastic",
            "objective": "soft",
        }

    def _analysis_context(self) -> AnalysisContext:
        return self._context(self._candidate_id(), self._policy_id())

    def _target_version(self) -> str:
        return f"{self._candidate_id()}|admitted_{self.plant_version}"

    def _record_trace(self, trace: OptimizerStepTrace) -> None:
        self._step_traces.setdefault(trace["block"], []).append(trace)

    def _store_trace(
        self, input_context: AnalysisContext, solved_context: AnalysisContext
    ) -> None:
        if self._observer is None:
            self._completed_trace = None
            return

        def path_length(block: str) -> float:
            return float(
                sum(
                    trace["parameter_increment"].norm()
                    for trace in self._step_traces.get(block, [])
                )
            )

        self._completed_trace = {
            "input_context": input_context,
            "solve_output_context": solved_context,
            "actor_output_context": self._context(
                solved_context["candidate_id"], self._policy_id()
            ),
            "value_steps": list(self._step_traces.get("value", [])),
            "rate_steps": list(self._step_traces.get("rate", [])),
            "actor_steps": list(self._step_traces.get("actor", [])),
            "model_steps": list(self._step_traces.get("model", [])),
            "temperature_steps": list(self._step_traces.get("temperature", [])),
            "value_increment_norm_sum": path_length("value"),
            "actor_increment_norm_sum": path_length("actor"),
        }

    def outer_step_trace(self) -> OuterStepTrace:
        """Expose optional old/new value-policy provenance and actual parameter-path records."""
        if self._completed_trace is None:
            raise RuntimeError(
                "No outer-step trace: set an optimizer observer before an outer iteration."
            )
        return self._completed_trace

    # --- checkpoints and diagnostics -----------------------------------------

    def _learned_plant_for_diagnostics(self) -> Optional[Plant]:
        return self.online_plant

    def _diagnostic_value(self) -> torch.nn.Module:
        return self.target_value

    def _network_modules(self) -> Mapping[str, Optional[torch.nn.Module]]:
        # "critics" is None (left out of checkpoints) unless decision_head is "critic",
        # which leaves "rate" None instead.
        return {
            "actor": self.actor,
            "rate": self.rate,
            "value": self.value,
            "target_value": self.target_value,
            "online_plant": self.online_plant,
            "admitted_plant": self.admitted_plant,
            "reward": self.reward_net,
            "reward_frozen": self._reward_frozen,
            "critics": self.critics,
        }

    def _optimizers(self) -> Mapping[str, Optional[torch.optim.Optimizer]]:
        return {
            "actor": self.actor_optimizer,
            "value": self.value_optimizer,
            "rate": self.rate_optimizer,
            "plant": self.plant_optimizer,
            "temperature": self.temperature_optimizer,
            "reward": self.reward_optimizer,
            "critics": self.critic_optimizer,
        }

    _LOOP_FIELDS = (
        "cycle_length",
        "outer_in_cycle",
        "outer_iteration",
        "inner_steps",
        "admission_index",
        "plant_version",
        "policy_version",
        "value_version",
        "gate_look_index",
        "budget_stopped_outer_iterations",
        "_pending_motion",
    )
    # Fields added with the warm fit, the tuning cadence and the learned reward; a
    # checkpoint written before them keeps the values the agent was built with.
    _LATER_LOOP_FIELDS = (
        "warm_fit_done",
        "tuning_rows",
        "tuning_steps",
        "last_reward_fit_loss",
    )

    def _loop_state(self) -> Mapping[str, object]:
        names = self._LOOP_FIELDS + self._LATER_LOOP_FIELDS
        state: Dict[str, object] = {name: getattr(self, name) for name in names}
        state["gate_rng"] = self._gate_rng.bit_generator.state
        state["reward_records"] = list(self.reward_records)
        state["reward_diagnostic_rng"] = self._reward_log_generator.get_state()
        return state

    def _load_loop_state(self, state: Mapping[str, object]) -> None:
        for name in self._LOOP_FIELDS:
            setattr(self, name, state[name])
        for name in self._LATER_LOOP_FIELDS:
            setattr(self, name, state.get(name, getattr(self, name)))
        self._gate_rng.bit_generator.state = state["gate_rng"]
        self.reward_records = list(state.get("reward_records", []))
        if "reward_diagnostic_rng" in state:
            self._reward_log_generator.set_state(state["reward_diagnostic_rng"].cpu())

    # The networks a warm start takes from the source run (the frozen reward copy is
    # retaken at the first operator freeze, since a warm-started solve starts at step 0).
    _WARM_START_NETWORKS = (
        "actor",
        "rate",
        "value",
        "target_value",
        "online_plant",
        "admitted_plant",
        "reward",
        "critics",
    )

    def _load_warm_start(self, state: Mapping[str, object]) -> Dict[str, object]:
        """Take a finished run's networks and temperature (PHAlgorithm.warm_start).

        Loaded: actor, rate head, value and its target, online and admitted plant (each as
        the source left it) and the learned reward, all strictly (the source must have
        the same networks), and log alpha_c when the temperature is tuned. Kept fresh: the
        optimizers (built by this agent; Adam's moments start at zero), the replay
        (empty), every counter, record and schedule position (the cycle of M outer
        iterations, the temperature's row count, the plant-step schedule, all from zero),
        the random streams (this run's seed). The warm fit is marked done: the plant and
        reward are already fitted, so the first value step follows the first full batch.
        A plant with trainable_plant_parts keeps its frozen parts at the source's values.
        A source run under another reward_observation (its reward network fitted to the
        other reward datum) is refused.
        """
        source = state["config"]
        if source["testbed"]["name"] != self.testbed["name"]:
            raise ValueError(
                f"A warm start of {self.testbed['name']} needs a run of the same testbed, "
                f"not {source['testbed']['name']}."
            )
        if decision_mode_setting(source["testbed"]["controller"]) != self.decision_mode:
            raise ValueError("A warm start needs a source run with the same decisions.")
        self._check_reward_observation(source, "The warm start's source run")
        networks = state["networks"]
        modules = self._network_modules()
        loaded = []
        for name in self._WARM_START_NETWORKS:
            module = modules[name]
            if module is None:
                if networks.get(name) is not None:
                    raise ValueError(f"The source run has a {name} this agent does not have.")
                continue
            if name not in networks:
                raise ValueError(f"The source run has no {name} to start from.")
            module.load_state_dict(networks[name])
            loaded.append(name)
        self._reward_frozen = None
        if self.log_temperature is not None:
            if state.get("log_temperature") is None:
                raise ValueError("A tuned temperature needs a source run that tuned its own.")
            with torch.no_grad():
                self.log_temperature.copy_(state["log_temperature"])
        self.warm_fit_done = True
        return {
            "networks": loaded,
            "alpha_c": float(self.alpha_c),
            "trainable_plant_parts": None
            if self.trainable_plant_parts is None
            else list(self.trainable_plant_parts),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore, holding a frozen reward copy exactly when the checkpoint had one; a
        checkpoint of another reward_observation raises before anything changes."""
        self._check_reward_observation(state.get("config", {}), "The checkpoint")
        if self.reward_net is not None:
            saved = "reward_frozen" in state["networks"]
            self._reward_frozen = self._frozen_reward() if saved else None
        super().load_state_dict(state)

    def resume_state(self) -> Mapping[str, object]:
        """state_dict with the replay, plus the optimizer traces of the solve in progress
        (empty without an observer), which an outer-step trace collects."""
        state = dict(super().resume_state())
        state["step_traces"] = {
            block: list(traces) for block, traces in self._step_traces.items()
        }
        return state

    def load_resume_state(self, state: Mapping[str, object]) -> None:
        super().load_resume_state(state)
        self._step_traces = {
            block: list(traces) for block, traces in state.get("step_traces", {}).items()
        }

    def _config(self) -> Mapping[str, object]:
        return self.config
