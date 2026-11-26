"""Paper comparison arms, matched experiments, and fixed-seed evaluation APIs."""

from __future__ import annotations

import contextlib
import copy
import math
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

import numpy as np
import torch

from common.ph_config import (
    DEFAULT_CT_SAC_REWARD_SOURCE,
    DEFAULT_CT_SAC_REWARD_UNITS,
    DEFAULT_FLAGGED_VALUE_GRADIENT,
    DEFAULT_REWARD_OBSERVATION,
    FLAGGED_VALUE_GRADIENT_KEY,
    REWARD_OBSERVATION_KEY,
    STEP_ABLATION_CHOICES,
    contact_exclusion_setting,
    contact_handling_setting,
    contact_substepping_setting,
    ct_sac_reward_units_setting,
    effective_ct_sac_reward_source,
    flagged_value_gradient_setting,
    implicit_setting,
    model_coarse_step_setting,
    model_integrator_setting,
    plant_step_schedule,
    refresh_mode_setting,
    reward_observation_setting,
    torque_decision_config,
)

from .adaptation import (
    adaptation_record,
    apply_warm_start,
    validate_warm_start,
    validate_warm_start_arms,
)
from .diagnostics import summarize_episode

if TYPE_CHECKING:
    from .adaptation import WarmStartConfig
    from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
    from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC
    from algorithms.ph_baseline import TorqueCTSACBaseline
    from common.ph_config import ExplicitConfig, ImplicitConfig, TestbedConfig
    from common.ph_types import BoundEvidence, RewardFitRecord, TestbedName
    from environment.ph_sac.base import MujocoPHEnv
    from .resume import ResumeOptions
    from .diagnostics import (
        AdmissionMetrics,
        EpisodeMetrics,
        OracleMetrics,
        PlantCalibrationBin,
        SolverMetrics,
        TheoryDiagnosticsRecord,
        TrainingCounters,
    )


# In all_arms() order: paper arms, impedance-law, implicit-setting, value-step ablations.
ArmName = Literal[
    "ct_sac_torque",
    "implicit_torque",
    "explicit_rollout_torque",
    "explicit_generator_torque",
    "implicit_learned_admissions",
    "explicit_rollout",
    "explicit_generator",
    "implicit_oracle",
    "implicit_averaged_draw",
    "implicit_no_rollout_rate",
    "implicit_polyak_plant",
    "implicit_gated_admissions",
    "implicit_fixed_gate_allocation",
    "implicit_explicit_step",
    "implicit_explicit_step_interval",
    "implicit_critic",
    "implicit_critic_head",
]
# The CT-SAC arms with CT-SAC's released hyperparameters (baseline_released_arms), run only
# when named (--arms). They are not in all_arms() or ArmName (whose seventeen names the
# tests pin); selectable_arms() is all_arms() and these.
ReleasedBaselineArmName = Literal["ct_sac_released", "ct_sac_released_timeout"]
# The CT-SAC arms with the implicit value step (baseline_implicit_arms), run only when named
# (--arms); not in all_arms() or ArmName either.
ImplicitBaselineArmName = Literal[
    "ct_sac_implicit", "ct_sac_implicit_schedule", "ct_sac_implicit_released"
]
# The CT-SAC arms whose networks read CT-PH-SAC's state features (baseline_feature_arms),
# run only when named (--arms); not in all_arms() or ArmName either.
FeatureBaselineArmName = Literal[
    "ct_sac_features", "ct_sac_released_features", "ct_sac_released_timeout_features"
]
# The implicit CT-SAC arms whose networks read the state features
# (baseline_implicit_feature_arms), run only when named (--arms); not in all_arms() or
# ArmName either.
ImplicitFeatureBaselineArmName = Literal[
    "ct_sac_implicit_features",
    "ct_sac_implicit_schedule_features",
    "ct_sac_implicit_released_features",
]


class CTSACSettings(TypedDict):
    """CT-SAC hyperparameters an arm sets itself (baseline_released_arms), in place of the
    values the CT-SAC baseline otherwise takes from the explicit config.

    learning_rate     Adam step size of the actor, the critics and the temperature, in
                      place of the explicit config's training learning_rate
    replay_capacity   replay rows kept (first in, first out), in place of its
                      training replay_capacity
    warmup_decisions  CT-SAC's learning_starts, in place of its training warmup_episodes:
                      uniform random actions on exactly this many decisions, counted
                      across episodes, and training from the decision that reaches it
                      (algorithms/ph_baseline._PHCTSAC)
    polyak_rate       the target critics' soft-update rate tau, in place of its polyak_rate
    time_limit        "bootstrap": a time-limit truncation is stored with done 0 and the
                      critic target bootstraps through it (ct_sac_torque's convention);
                      "terminal": done = terminated or truncated, as CT-SAC's own loop
                      stores it (algorithms/off_policy.py learn)
    """

    learning_rate: float
    replay_capacity: int
    warmup_decisions: int
    polyak_rate: float
    time_limit: Literal["bootstrap", "terminal"]


class CTSACImplicitStep(TypedDict):
    """The implicit value step of a CT-SAC arm (baseline_implicit_arms;
    algorithms/ct_sac_implicit.py): the critics solve (1 + c) Q - c F(Q) = Qbar, c = tau / T,
    F CT-SAC's own target, by K residual-gradient steps, then a hard copy. Its numbers (tau,
    K, N, the outer schedule's actor steps and temperature cadence) are the explicit
    config's optional section common.ph_config.CT_SAC_IMPLICIT_KEY, its
    CT_SAC_IMPLICIT_DEFAULTS when absent.

    actor_schedule  "per_row": the actor and the temperature step on every row, as in
                    CT-SAC; "outer": the policy is frozen through each solve, and the actor
                    and temperature steps follow each copy, as in implicit CT-PH-SAC
                    (common.ph_config.CT_SAC_ACTOR_SCHEDULES)
    """

    actor_schedule: Literal["per_row", "outer"]


class OptionalArmFields(TypedDict, total=False):
    """The value-step ablation's choices (common.ph_config.STEP_ABLATION_CHOICES), carried
    only by step_ablation_arms() and copied into the arm's implicit config under the same
    names; every other arm leaves them out, so its spec, records and resume fingerprint
    are those of the code before them.

    ct_sac_settings (CTSACSettings) is carried only by baseline_released_arms(): the CT-SAC
    hyperparameters that arm runs with. Every other arm leaves it out, for the same reason.

    ct_sac_implicit_step (CTSACImplicitStep) is carried only by baseline_implicit_arms():
    the CT-SAC baseline with the implicit value step. Every other arm leaves it out.

    ct_sac_inputs "state_features" is carried only by baseline_feature_arms() and
    baseline_implicit_feature_arms(): the CT-SAC baseline's actor and critics read
    CT-PH-SAC's StateFeatures of the state (with the explicit config's network
    velocity_feature_scale) instead of the state itself. Absent, they read the state.
    Every other arm leaves it out.
    """

    value_step: Literal["implicit", "explicit"]
    explicit_step_size: Literal["resolvent_step", "interval"]
    decision_head: Literal["rate", "critic"]
    value_operator: Literal["model", "critic"]
    ct_sac_settings: CTSACSettings
    ct_sac_implicit_step: CTSACImplicitStep
    ct_sac_inputs: Literal["state_features"]


class ArmSpec(OptionalArmFields):
    """One comparison arm; each implicit ablation changes one learned implicit setting.

    plant_mode "admitted" copies the online plant into the admitted plant untested at
    every cycle end (plant_update "always", the default); "gated" runs the paper's staged
    gate (plant_update "admission"), whose allocation the fixed-allocation arm changes.
    decision_mode "torque" draws the torque itself, held over the interval: every paper
    arm (paper_arms: the CT-SAC baseline and the three CT-PH-SAC arms, whose configs are
    torque_decision_config's) and the step-ablation arms; "feedback" draws impedance laws
    (setpoint, stiffness, damping): the impedance-law ablation (impedance_decision_arms)
    and the implicit-setting ablations (implicit_ablation_arms). The step-ablation arms
    (step_ablation_arms) add OptionalArmFields' value-step choices, the released-baseline
    arms (baseline_released_arms: ct_sac_torque's spec with their own name) its
    ct_sac_settings.
    """

    name: Union[
        ArmName,
        ReleasedBaselineArmName,
        ImplicitBaselineArmName,
        FeatureBaselineArmName,
        ImplicitFeatureBaselineArmName,
    ]
    algorithm: Literal["ct_sac", "explicit", "implicit"]
    decision_mode: Literal["torque", "feedback"]
    target_mode: Optional[Literal["generator", "rollout"]]
    plant_mode: Literal["none", "oracle", "admitted", "gated", "polyak"]
    residual_loss: Optional[Literal["all_pairs", "averaged_draw"]]
    rollout_rate_enabled: bool
    gate_allocation: Optional[Literal["pilot", "fixed"]]


class MatchedQuantities(TypedDict):
    """Shared physical clock, objective, interaction budget, and reset seeds."""

    testbed_config: TestbedConfig
    interaction_budget: int
    evaluation_reset_seeds: Sequence[int]
    discount_rate: float
    temperature_per_second: float


class TheoryDiagnosticsConfig(TypedDict):
    """Optional theorem accounting; it adds measurements, never comparison arms."""

    policy_kind: Literal["stochastic", "deterministic_mean"]
    objective: Literal["soft", "reward_only"]
    every_outer_iterations: int
    sampled_states: int
    decision_draws: int
    holding_paths: int
    initial_state_law_id: str
    record_optimizer_steps: bool
    record_training_calibration: bool
    record_outer_budgets: bool
    uniform_evidence: Optional[BoundEvidence]
    task_return_threshold: Optional[float]


class OptionalProtocolFields(TypedDict, total=False):
    theory_diagnostics: TheoryDiagnosticsConfig
    baseline_expectation_samples: int
    # Whether checkpoint.pt carries the replay (default True); False keeps it small.
    checkpoint_replay: bool
    # The adaptation experiment: every run starts from a finished run's checkpoint
    # (evaluations/ph_sac/adaptation.py). Absent: every run starts fresh.
    warm_start: WarmStartConfig
    # The reward and temperature units of every CT-SAC arm's learner
    # (common.ph_config.CT_SAC_REWARD_UNITS; make_training_agent): "second" (absent, every
    # earlier run) or "reference_period" (each divided by T). No other arm reads it; it
    # enters only those arms' fingerprints, and only off "second" (resume._resolved_settings).
    ct_sac_reward_units: Literal["second", "reference_period"]
    # The interval part of every CT-SAC arm's learner reward
    # (common.ph_config.CT_SAC_REWARD_SOURCES; make_training_agent): "interval" (absent,
    # every earlier run), the environment's interval reward per period, or "end_state", T
    # times the reward rate at the interval's end state with the held torque. The same
    # scope and fingerprint rule as ct_sac_reward_units.
    ct_sac_reward_source: Literal["interval", "end_state"]
    # The reward datum every arm's learner may use (common.ph_config.REWARD_OBSERVATIONS,
    # the block at REWARD_OBSERVATION_KEY): "interval" (absent, every earlier run), the
    # environment's interval reward, or "end_state", the reward rate at each interval's end
    # state alone (the CT-SAC arms then run ct_sac_reward_source "end_state", an explicit
    # "interval" there is refused; every CT-PH-SAC arm learns a reward rate from it and
    # integrates it over the interval). The arm builders write it into a CT-PH-SAC arm's
    # config only off the default (_explicit_arm_config, _implicit_arm_config); it enters
    # the fingerprints only off the default (resume._resolved_settings).
    reward_observation: Literal["interval", "end_state"]


class EvaluationProtocol(OptionalProtocolFields):
    """Complete comparison settings; unspecified paper choices are required inputs."""

    testbed: TestbedName
    arms: Sequence[ArmSpec]
    matched: MatchedQuantities
    explicit_config: ExplicitConfig
    implicit_config: ImplicitConfig
    training_seeds: Sequence[int]
    evaluation_every_interactions: int
    deterministic_evaluation: bool
    ablation_polyak_rate: float
    output_directory: str


class RunSpec(TypedDict):
    """A single testbed, arm, and training-seed combination."""

    run_id: str
    protocol: EvaluationProtocol
    arm: ArmSpec
    training_seed: int


class _EvaluationCheckpointRequired(TypedDict):
    interactions: int
    episodes: Sequence[EpisodeMetrics]
    counters: TrainingCounters


class EvaluationCheckpoint(_EvaluationCheckpointRequired, total=False):
    """One fixed-seed evaluation point indexed by training interactions.

    policy (since 2026-11-11; absent from earlier records): the agent's
    policy_diagnostics() at that point, the keys POLICY_DIAGNOSTIC_KEYS: alpha_c (the
    per-second temperature), alpha_decision (the same temperature as CT-SAC's alpha, in
    CT-SAC's own units, per reference period T: alpha_c T), log_std_mean (the actor's mean
    log standard deviation), entropy_estimate (mean -log pi over the latest training rows,
    None for the CT-SAC baseline and before the first row) and entropy_rows. Records
    written before 2026-11-17 used alpha_c E[w_lambda(Delta)] in place of alpha_c T (the
    CT-PH-SAC arms' alpha_decision, the CT-SAC arms' alpha_c = alpha / E[w_lambda(Delta)]).
    A CT-SAC arm under ct_sac_reward_units "reference_period" (rewards and temperature
    divided by T) records its learner's alpha as alpha_decision, which is then alpha_c
    itself (alpha_c T / T); its alpha_c is the per-second temperature as in every arm.
    """

    policy: Mapping[str, Optional[float]]


POLICY_DIAGNOSTIC_KEYS = (
    "alpha_c", "alpha_decision", "log_std_mean", "entropy_estimate", "entropy_rows",
)


class OptionalRunResultFields(TypedDict, total=False):
    """decision_mode is the arm's ("feedback" or "torque"); records written before it
    existed load with the arm's, else "feedback" (reporting.record_decision_mode).
    refresh_mode is the testbed controller's (common.ph_config.REFRESH_MODES); records
    written before it existed ran the sample-and-hold controller and load as "held"
    (reporting.record_refresh_mode). plant_step_schedule is the training config's
    learned-plant step schedule, {"dense_until": D, "every": k}
    (common.ph_config.plant_step_schedule; the CT-SAC baseline has no plant and records
    its config's); records written before it existed took a step on every row and load
    as {"dense_until": 0, "every": 1} (reporting.record_plant_step_schedule).
    contact_substepping is the arm config's contact-triggered fine substepping
    (common.ph_config.contact_substepping_setting: coarse_step, fine_step,
    margin_fraction, velocity_factor), None when off and for the CT-SAC baseline; records
    written before it existed load as None (reporting.record_contact_substepping).
    contact_handling is the arm config's contact exclusion
    (common.ph_config.contact_exclusion_setting: mode "exclude", model_step and the rule's
    constants), None under "port" and for the CT-SAC baseline; records written before it
    existed load as None (reporting.record_contact_handling).
    The adaptation experiment's fields (adaptation.adaptation_record), each present only
    when the run has it: warm_start (the source checkpoint, its interaction count, the
    networks loaded and the carried temperature), physical_change (kind, scale and every
    changed plant parameter, nominal and physical) and trainable_plant_parts (an implicit
    arm's).
    reward_observation is the protocol's (common.ph_config.REWARD_OBSERVATIONS), written
    only when it is "end_state"; records without it ran "interval" and load as such
    (reporting.record_reward_observation). reward_fit holds the learned reward rate's
    diagnostics (the implicit arms; the explicit arms only under "end_state").
    flagged_value_gradient is an implicit arm's config key
    (common.ph_config.FLAGGED_VALUE_GRADIENTS), written only when it is "stop_successor";
    records without it load as "full" (reporting.record_flagged_value_gradient)."""

    theory_records: Sequence[TheoryDiagnosticsRecord]
    reward_fit: Sequence[RewardFitRecord]
    decision_mode: Literal["torque", "feedback"]
    refresh_mode: Literal["continuous", "held"]
    plant_step_schedule: Mapping[str, int]
    contact_substepping: Optional[Mapping[str, float]]
    contact_handling: Optional[Mapping[str, object]]
    warm_start: Mapping[str, object]
    physical_change: Mapping[str, object]
    trainable_plant_parts: Sequence[str]
    reward_observation: Literal["interval", "end_state"]
    flagged_value_gradient: Literal["full", "stop_successor"]


class RunResult(OptionalRunResultFields):
    """Return, failure, solver, admission, oracle, and calibration records."""

    run: RunSpec
    evaluations: Sequence[EvaluationCheckpoint]
    counters: TrainingCounters
    solver: Sequence[SolverMetrics]
    admissions: Sequence[AdmissionMetrics]
    oracle: Sequence[OracleMetrics]
    plant_calibration: Sequence[PlantCalibrationBin]
    checkpoint_paths: Sequence[str]


def _implicit_arm(name: ArmName, **changes: object) -> ArmSpec:
    """The learned implicit algorithm, admitted untested, with torque decisions unless the
    changes say otherwise, and the listed changes."""
    arm: ArmSpec = {
        "name": name,
        "algorithm": "implicit",
        "decision_mode": "torque",
        "target_mode": None,
        "plant_mode": "admitted",
        "residual_loss": "all_pairs",
        "rollout_rate_enabled": True,
        "gate_allocation": "pilot",
    }
    arm.update(changes)  # type: ignore[typeddict-item]
    return arm


def _explicit_arm(
    name: ArmName,
    target: Literal["generator", "rollout"],
    decision_mode: Literal["torque", "feedback"] = "torque",
) -> ArmSpec:
    """The explicit algorithm with the Polyak plant and the given target."""
    return {
        "name": name,
        "algorithm": "explicit",
        "decision_mode": decision_mode,
        "target_mode": target,
        "plant_mode": "polyak",
        "residual_loss": None,
        "rollout_rate_enabled": False,
        "gate_allocation": None,
    }


# Every arm's spec is fixed by its name: run records store it (run.arm) and the resume
# fingerprint hashes it (resume.run_settings), so a declared arm never changes
# (tests/test_ph_torque_decisions.py, TestFixedArmNames). The helpers' decision_mode
# default (torque) applies to arms declared without one; every arm below names its mode
# but the step-ablation arms.
def paper_arms() -> Sequence[ArmSpec]:
    """Declare the four paper arms, the default arm set of every protocol.

    In role order: the CT-SAC baseline (ct_sac_torque), then CT-PH-SAC's implicit
    (implicit_torque), rollout (explicit_rollout_torque) and generator
    (explicit_generator_torque) arms. Every one decides the torque itself, held over the
    interval; the three CT-PH-SAC arms' configs are torque_decision_config's (the torque
    box and -1 entropy per torque coordinate), built per run. implicit_torque admits the
    online plant untested at every cycle end.
    """
    baseline: ArmSpec = {
        "name": "ct_sac_torque",
        "algorithm": "ct_sac",
        "decision_mode": "torque",
        "target_mode": None,
        "plant_mode": "none",
        "residual_loss": None,
        "rollout_rate_enabled": False,
        "gate_allocation": None,
    }
    return [baseline, *torque_decision_arms()]


def torque_decision_arms() -> Sequence[ArmSpec]:
    """Declare the three CT-PH-SAC paper arms (paper_arms without the baseline).

    implicit_torque, explicit_rollout_torque and explicit_generator_torque: decisions are
    held torques, as in ct_sac_torque. Each is its impedance-law counterpart
    (impedance_decision_arms) with decision_mode "torque". Their configs are
    torque_decision_config's: the torque box and -1 entropy per torque coordinate.
    """
    return [
        _implicit_arm("implicit_torque", decision_mode="torque"),
        _explicit_arm("explicit_rollout_torque", "rollout", decision_mode="torque"),
        _explicit_arm("explicit_generator_torque", "generator", decision_mode="torque"),
    ]


def impedance_decision_arms() -> Sequence[ArmSpec]:
    """Declare the impedance-law ablation, run only when named (--arms).

    implicit_learned_admissions, explicit_rollout and explicit_generator: each is the
    torque_decision_arms arm of its role with decision_mode "feedback", decisions being
    impedance laws (setpoint, stiffness, damping per actuator) on the protocol's feedback
    box with the protocol's entropy target, -1 per feedback coordinate. No protocol
    includes them by default.
    """
    return [
        _implicit_arm("implicit_learned_admissions", decision_mode="feedback"),
        _explicit_arm("explicit_rollout", "rollout", decision_mode="feedback"),
        _explicit_arm("explicit_generator", "generator", decision_mode="feedback"),
    ]


def implicit_ablation_arms() -> Sequence[ArmSpec]:
    """Declare the six implicit-setting ablation arms, run only when named (--arms).

    Each changes one learned implicit setting of the impedance-law arm
    implicit_learned_admissions and takes its feedback decisions:
    implicit_gated_admissions runs the paper's staged gate in place of the untested
    admission, implicit_fixed_gate_allocation ablates that gate's pilot-guided
    allocation. No protocol includes them by default.
    """
    return [
        _implicit_arm("implicit_oracle", plant_mode="oracle", decision_mode="feedback"),
        _implicit_arm(
            "implicit_averaged_draw", residual_loss="averaged_draw", decision_mode="feedback"
        ),
        _implicit_arm(
            "implicit_no_rollout_rate", rollout_rate_enabled=False, decision_mode="feedback"
        ),
        _implicit_arm("implicit_polyak_plant", plant_mode="polyak", decision_mode="feedback"),
        _implicit_arm(
            "implicit_gated_admissions", plant_mode="gated", decision_mode="feedback"
        ),
        _implicit_arm(
            "implicit_fixed_gate_allocation",
            plant_mode="gated",
            gate_allocation="fixed",
            decision_mode="feedback",
        ),
    ]


def step_ablation_arms() -> Sequence[ArmSpec]:
    """Declare the four value-step ablation arms, run only when named (--arms).

    They take torque decisions, as the paper arms. explicit_generator_torque (twin critic,
    explicit Euler step of each row's Delta) and implicit_torque (rate head, resolvent
    step of tau) differ in two things; these arms take one at a time from implicit_torque
    (common.ph_config.STEP_ABLATION_DEFAULTS):
      implicit_explicit_step           the resolvent step replaced by one explicit Euler
                                       step of the same operator, V_(k+1) = V_k + tau
                                       E_pi[rate(V_k)] (h = tau: the like-for-like step)
      implicit_explicit_step_interval  the same with h = each row's Delta (the explicit
                                       generator arm's step)
      implicit_critic_head             the rate head replaced by the twin critic (fitted
                                       as explicit_generator_torque's) where the rate head
                                       is used: its fit after each copy and the actor; the
                                       resolvent step, which never reads the rate head,
                                       and the learned reward are unchanged
      implicit_critic                  the same critic also read by the resolvent step as
                                       (Q - V) / Delta_ref, no learned reward; Q holds G at
                                       V_k, so its root is an explicit step of tau Delta_ref
                                       / (tau + Delta_ref) in the generator
    The value-step code is the same for either decision mode; these arms take the paper
    arms' (the helper's default). No protocol includes them by default.
    """
    return [
        _implicit_arm(
            "implicit_explicit_step", value_step="explicit", explicit_step_size="resolvent_step"
        ),
        _implicit_arm(
            "implicit_explicit_step_interval", value_step="explicit", explicit_step_size="interval"
        ),
        _implicit_arm("implicit_critic", decision_head="critic", value_operator="critic"),
        _implicit_arm("implicit_critic_head", decision_head="critic", value_operator="model"),
    ]


def all_arms() -> Sequence[ArmSpec]:
    """Every arm a protocol may select, seventeen: the four paper arms (the default), then
    the ablations, impedance-law, implicit-setting and value-step."""
    return [
        *paper_arms(),
        *impedance_decision_arms(),
        *implicit_ablation_arms(),
        *step_ablation_arms(),
    ]


# CT-SAC's released hyperparameters (Nguyen et al. 2026; benchmarks/hyperparams/ct_sac.csv,
# the cheetah-run and walker-run rows: algo_learning_rate 0.00073, algo_buffer_size 300000,
# algo_learning_starts 10000, algo_tau 0.02 in most rows), the ones ct_sac_torque takes
# from the explicit config instead (learning_rate 3e-4, replay_capacity 1e6, 5 warm-up
# episodes, polyak_rate 0.0125 in the paper's configs). The rest of the released row is
# what ct_sac_torque already runs: batch_size 256 (the configs' training batch_size),
# train_freq 1 and gradient_steps 1 (algorithms/ct_sac.CTSAC's defaults), 400-300 ReLU
# networks with two critics and log_std_init -3 (algorithms/ph_baseline's defaults),
# alpha and target_entropy "auto" (a tuned temperature, -1 per unit action coordinate;
# the initial temperature stays the matched one) and num_expectation_samples 1. Its gamma
# 0.98 per control step is the configs' discount rate. A mapping that
# baseline_released_arms() reads at every call.
RELEASED_CT_SAC_SETTINGS = {
    "learning_rate": 7.3e-4,
    "replay_capacity": 300_000,
    "warmup_decisions": 10_000,
    "polyak_rate": 0.02,
}


def baseline_released_arms() -> Sequence[ArmSpec]:
    """Declare the two CT-SAC arms with CT-SAC's released hyperparameters, run only when
    named (--arms); neither is in paper_arms() or all_arms().

    Each is ct_sac_torque's spec (the CT-SAC baseline adapter, torque decisions, the same
    environment and evaluation) under its own name with ct_sac_settings:
    RELEASED_CT_SAC_SETTINGS in place of the explicit config's learning rate, replay
    capacity, warm-up and Polyak rate (make_training_agent).
      ct_sac_released          bootstraps through a time limit, as ct_sac_torque does
      ct_sac_released_timeout  ends the continuation at a time limit (done = terminated or
                               truncated), as CT-SAC's own loop does
    """

    def released(name: ReleasedBaselineArmName, time_limit: str) -> ArmSpec:
        return {
            "name": name,
            "algorithm": "ct_sac",
            "decision_mode": "torque",
            "target_mode": None,
            "plant_mode": "none",
            "residual_loss": None,
            "rollout_rate_enabled": False,
            "gate_allocation": None,
            "ct_sac_settings": {**RELEASED_CT_SAC_SETTINGS, "time_limit": time_limit},
        }

    return [
        released("ct_sac_released", "bootstrap"),
        released("ct_sac_released_timeout", "terminal"),
    ]


def baseline_implicit_arms() -> Sequence[ArmSpec]:
    """Declare the CT-SAC arms with the implicit value step, run only when named (--arms);
    none is in paper_arms() or all_arms().

    Each is ct_sac_torque's spec (the CT-SAC baseline adapter, torque decisions, the same
    environment, evaluation, networks, step size, batch, replay, warm-up, temperature and
    time-limit convention) under its own name with ct_sac_implicit_step
    (CTSACImplicitStep; algorithms/ct_sac_implicit.py), whose tau, K and N are the explicit
    config's ct_sac_implicit section (common.ph_config.CT_SAC_IMPLICIT_DEFAULTS: 1 s, 400,
    4 when absent):
      ct_sac_implicit           CT-SAC's value update replaced by the implicit step; the
                                actor and the temperature step on every row, as in CT-SAC
      ct_sac_implicit_schedule  the same value step with implicit CT-PH-SAC's outer
                                schedule: the policy frozen through each solve, actor_steps
                                (40) actor steps after each copy and one temperature step
                                per temperature_every_rows (40) rows
      ct_sac_implicit_released  ct_sac_implicit with ct_sac_released's ct_sac_settings
                                (RELEASED_CT_SAC_SETTINGS, bootstrapping through a time
                                limit); its polyak_rate is unused (hard copies)
    """

    def implicit(name: ImplicitBaselineArmName, schedule: str) -> ArmSpec:
        return {
            "name": name,
            "algorithm": "ct_sac",
            "decision_mode": "torque",
            "target_mode": None,
            "plant_mode": "none",
            "residual_loss": None,
            "rollout_rate_enabled": False,
            "gate_allocation": None,
            "ct_sac_implicit_step": {"actor_schedule": schedule},
        }

    released = implicit("ct_sac_implicit_released", "per_row")
    released["ct_sac_settings"] = {**RELEASED_CT_SAC_SETTINGS, "time_limit": "bootstrap"}
    return [
        implicit("ct_sac_implicit", "per_row"),
        implicit("ct_sac_implicit_schedule", "outer"),
        released,
    ]


def baseline_feature_arms() -> Sequence[ArmSpec]:
    """Declare the CT-SAC arms whose actor and critics read CT-PH-SAC's state features, run
    only when named (--arms); none is in paper_arms() or all_arms().

    Each is the named CT-SAC arm's spec under its own name with ct_sac_inputs
    "state_features": the networks read models.ph_sac.networks.StateFeatures of the state
    (periodic positions as sine and cosine, other positions raw, velocities divided by the
    explicit config's network velocity_feature_scale), the input map of the CT-PH-SAC
    arms' networks, in place of the state with its angles wrapped to [-pi, pi);
    everything else is that arm's (algorithms/ph_baseline.TorqueCTSACBaseline
    state_features).
      ct_sac_features                   ct_sac_torque
      ct_sac_released_features          ct_sac_released
      ct_sac_released_timeout_features  ct_sac_released_timeout
    """
    sources = {arm["name"]: arm for arm in (*paper_arms(), *baseline_released_arms())}
    arms = []
    for name, source in (
        ("ct_sac_features", "ct_sac_torque"),
        ("ct_sac_released_features", "ct_sac_released"),
        ("ct_sac_released_timeout_features", "ct_sac_released_timeout"),
    ):
        arm = copy.deepcopy(sources[source])
        arm["name"] = name
        arm["ct_sac_inputs"] = "state_features"
        arms.append(arm)
    return arms


def baseline_implicit_feature_arms() -> Sequence[ArmSpec]:
    """Declare the implicit CT-SAC arms whose actor and critics read CT-PH-SAC's state
    features, run only when named (--arms); none is in paper_arms() or all_arms().

    Declared as baseline_feature_arms() declares its arms: the named implicit arm's spec
    (baseline_implicit_arms) under its own name with ct_sac_inputs "state_features";
    everything else is that arm's (make_training_agent passes the explicit config's
    network as state_features to algorithms/ct_sac_implicit.ImplicitTorqueCTSACBaseline).
      ct_sac_implicit_features           ct_sac_implicit
      ct_sac_implicit_schedule_features  ct_sac_implicit_schedule
      ct_sac_implicit_released_features  ct_sac_implicit_released
    """
    sources = {arm["name"]: arm for arm in baseline_implicit_arms()}
    arms = []
    for name, source in (
        ("ct_sac_implicit_features", "ct_sac_implicit"),
        ("ct_sac_implicit_schedule_features", "ct_sac_implicit_schedule"),
        ("ct_sac_implicit_released_features", "ct_sac_implicit_released"),
    ):
        arm = copy.deepcopy(sources[source])
        arm["name"] = name
        arm["ct_sac_inputs"] = "state_features"
        arms.append(arm)
    return arms


def selectable_arms() -> Sequence[ArmSpec]:
    """Every arm a protocol may run (--arms): all_arms(), then baseline_released_arms(),
    baseline_implicit_arms(), baseline_feature_arms() and baseline_implicit_feature_arms()."""
    return [
        *all_arms(),
        *baseline_released_arms(),
        *baseline_implicit_arms(),
        *baseline_feature_arms(),
        *baseline_implicit_feature_arms(),
    ]


def build_protocol(
    *,
    testbed: TestbedName,
    testbed_config: TestbedConfig,
    explicit_config: ExplicitConfig,
    implicit_config: ImplicitConfig,
    interaction_budget: int,
    discount_rate: float,
    temperature_per_second: float,
    training_seeds: Sequence[int],
    evaluation_reset_seeds: Sequence[int],
    evaluation_every_interactions: int,
    deterministic_evaluation: bool,
    ablation_polyak_rate: float,
    output_directory: str,
    theory_diagnostics: Optional[TheoryDiagnosticsConfig] = None,
    baseline_expectation_samples: int = 1,
    checkpoint_replay: bool = True,
    warm_start: Optional[WarmStartConfig] = None,
    ct_sac_reward_units: Optional[str] = None,
    ct_sac_reward_source: Optional[str] = None,
    reward_observation: Optional[str] = None,
) -> EvaluationProtocol:
    """Lay out the paper arms (paper_arms), with optional separate theory measurements.

    The arm list is not part of a run's resume fingerprint (resume._UNFINGERPRINTED);
    --arms replaces it. warm_start (the adaptation experiment, adaptation.py),
    ct_sac_reward_units and ct_sac_reward_source (the CT-SAC arms' reward units and
    reward source) and reward_observation (every arm's reward datum,
    OptionalProtocolFields) join the protocol only when given.
    """
    protocol: EvaluationProtocol = {
        "testbed": testbed,
        "arms": list(paper_arms()),
        "matched": {
            "testbed_config": testbed_config,
            "interaction_budget": int(interaction_budget),
            "evaluation_reset_seeds": list(evaluation_reset_seeds),
            "discount_rate": float(discount_rate),
            "temperature_per_second": float(temperature_per_second),
        },
        "explicit_config": explicit_config,
        "implicit_config": implicit_config,
        "training_seeds": list(training_seeds),
        "evaluation_every_interactions": int(evaluation_every_interactions),
        "deterministic_evaluation": bool(deterministic_evaluation),
        "ablation_polyak_rate": float(ablation_polyak_rate),
        "output_directory": str(output_directory),
        "baseline_expectation_samples": int(baseline_expectation_samples),
        "checkpoint_replay": checkpoint_replay,
    }
    if theory_diagnostics is not None:
        protocol["theory_diagnostics"] = theory_diagnostics
    if warm_start is not None:
        protocol["warm_start"] = warm_start
    if ct_sac_reward_units is not None:
        protocol["ct_sac_reward_units"] = ct_sac_reward_units
    if ct_sac_reward_source is not None:
        protocol["ct_sac_reward_source"] = ct_sac_reward_source
    if reward_observation is not None:
        protocol[REWARD_OBSERVATION_KEY] = reward_observation
    validate_matched_quantities(protocol)
    return protocol


def _same_settings(left: object, right: object) -> bool:
    """Deep equality that compares arrays by value and tuples like lists."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _same_settings(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_settings(a, b) for a, b in zip(left, right)
        )
    return left == right


def validate_matched_quantities(protocol: EvaluationProtocol) -> None:
    """Check shared units/settings and reject legacy or policy/objective mismatches.

    The CT-PH-SAC arms share the initial per-second temperature, which is the matched
    temperature_per_second, the entropy target and whether the temperature is tuned; a
    tuned temperature is allowed. The CT-SAC baseline sets its own target in its units.
    The configs hold the feedback box and its target; the CT-PH-SAC arms with torque
    decisions (the paper's and the step-ablation arms) derive theirs per run from the
    shared one (torque_decision_config: the same entropy per coordinate on the m
    torques). Both CT-PH-SAC configs share the model_integrator and the matched testbed
    controller's refresh_mode. Every arm must equal its declaration in selectable_arms()
    (all_arms() and baseline_released_arms(); the step-ablation arms' choices and the
    released arms' ct_sac_settings included).
    """
    matched = protocol["matched"]
    testbed = matched["testbed_config"]
    if testbed["name"] != protocol["testbed"]:
        raise ValueError("The protocol testbed and its configuration disagree.")
    for label, config in (
        ("explicit", protocol["explicit_config"]),
        ("implicit", protocol["implicit_config"]),
    ):
        arm_testbed = config["testbed"]
        if not _same_settings(arm_testbed["clock"], testbed["clock"]):
            raise ValueError(f"The {label} arms must use the matched clock.")
        if arm_testbed["discount_rate"] != matched["discount_rate"]:
            raise ValueError(f"The {label} arms must use the matched discount rate.")
        if not _same_settings(arm_testbed, testbed):
            raise ValueError(f"The {label} arms must use the matched testbed.")
        temperature = config["temperature"]
        if temperature["initial_alpha_c"] != matched["temperature_per_second"]:
            raise ValueError(f"The {label} arms must use the matched temperature.")
    explicit_temperature = protocol["explicit_config"]["temperature"]
    implicit_temperature = protocol["implicit_config"]["temperature"]
    if explicit_temperature["target_entropy"] != implicit_temperature["target_entropy"]:
        raise ValueError("The CT-PH-SAC arms must share one entropy target.")
    if bool(explicit_temperature["learn_temperature"]) != bool(
        implicit_temperature["learn_temperature"]
    ):
        raise ValueError("The CT-PH-SAC arms must all tune or all fix the temperature.")
    if model_integrator_setting(protocol["explicit_config"]) != model_integrator_setting(
        protocol["implicit_config"]
    ):
        raise ValueError("The CT-PH-SAC arms must share one model_integrator.")
    if model_coarse_step_setting(protocol["explicit_config"]) != model_coarse_step_setting(
        protocol["implicit_config"]
    ):
        raise ValueError("The CT-PH-SAC arms must share one model_coarse_step.")
    if contact_handling_setting(protocol["explicit_config"]) != contact_handling_setting(
        protocol["implicit_config"]
    ):
        raise ValueError("The CT-PH-SAC arms must share one contact_handling.")
    modes = {
        refresh_mode_setting(config["testbed"]["controller"])
        for config in (protocol["explicit_config"], protocol["implicit_config"])
    }
    if modes != {refresh_mode_setting(testbed["controller"])}:
        raise ValueError(
            "The CT-PH-SAC arms must share the matched testbed's controller refresh_mode."
        )
    if matched["discount_rate"] != testbed["discount_rate"]:
        raise ValueError("The matched discount rate must be the testbed's.")
    if matched["interaction_budget"] < 1:
        raise ValueError("The interaction budget must be positive.")
    seeds = list(matched["evaluation_reset_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation reset seeds must be nonempty and distinct.")
    if not protocol["training_seeds"]:
        raise ValueError("The protocol needs at least one training seed.")
    if protocol["evaluation_every_interactions"] < 1:
        raise ValueError("The evaluation period must be positive.")
    if not 0.0 < protocol["ablation_polyak_rate"] <= 1.0:
        raise ValueError("The Polyak ablation rate must lie in (0, 1].")
    if protocol.get("baseline_expectation_samples", 1) != 1:
        raise ValueError(
            "The existing CT-SAC update supports exactly one expectation sample."
        )
    if not isinstance(protocol.get("checkpoint_replay", True), bool):
        raise ValueError("checkpoint_replay must be true or false.")
    # The CT-SAC arms' reward units: one of common.ph_config.CT_SAC_REWARD_UNITS, if given.
    ct_sac_reward_units_setting(protocol)
    # The reward datum: one of common.ph_config.REWARD_OBSERVATIONS, if given; the CT-SAC
    # arms' reward source, one of CT_SAC_REWARD_SOURCES if given, must not contradict it.
    observation = reward_observation_setting(protocol)
    effective_ct_sac_reward_source(protocol)
    for label in ("explicit_config", "implicit_config"):
        if REWARD_OBSERVATION_KEY in protocol[label]:
            raise ValueError(
                f"{REWARD_OBSERVATION_KEY} is a protocol key; the protocol's {label} may "
                "not set it (the arm builders write it into each arm's config)."
            )
    if (
        observation == "end_state"
        and implicit_setting(protocol["implicit_config"], "reward_source") != "learned"
    ):
        raise ValueError(
            f"{REWARD_OBSERVATION_KEY} 'end_state' needs implicit reward_source 'learned'."
        )
    validate_warm_start(protocol)

    catalog = {arm["name"]: arm for arm in selectable_arms()}
    if not protocol["arms"]:
        raise ValueError("The protocol needs at least one arm.")
    for arm in protocol["arms"]:
        if arm["name"] not in catalog or arm != catalog[arm["name"]]:
            raise ValueError(
                f"Arm {arm['name']!r} differs from the paper's specification; "
                "the CT-SAC arm must use the adapted torque baseline."
            )

    theory = protocol.get("theory_diagnostics")
    if theory is not None:
        validate_theory_diagnostics(theory)
        kind = "deterministic_mean" if protocol["deterministic_evaluation"] else "stochastic"
        if theory["policy_kind"] != kind:
            raise ValueError(
                "Theory diagnostics must analyze the policy kind that is evaluated."
            )


def validate_theory_diagnostics(config: TheoryDiagnosticsConfig) -> None:
    """Require reward-only mean-policy analysis; supplied evidence must justify bounds."""
    if config["policy_kind"] == "deterministic_mean" and config["objective"] != "reward_only":
        raise ValueError(
            "A deterministic-mean policy is analyzed under the reward-only alpha_c=0 "
            "objective; the soft objective belongs to the stochastic actor."
        )
    if config["every_outer_iterations"] < 1:
        raise ValueError("Theory diagnostics need a positive period.")
    if min(config["sampled_states"], config["decision_draws"], config["holding_paths"]) < 1:
        raise ValueError("Theory diagnostics need positive sample counts.")
    evidence = config["uniform_evidence"]
    if evidence is not None:
        if evidence["kind"] == "sampled_diagnostic":
            raise ValueError("A sampled diagnostic is not uniform evidence.")
        if not evidence["domain"] or not evidence["justification"]:
            raise ValueError("Uniform evidence needs its domain and justification.")
    if config["task_return_threshold"] is not None and evidence is None:
        raise ValueError("A return-threshold claim needs supplied uniform evidence.")


def build_run_matrix(protocol: EvaluationProtocol) -> Sequence[RunSpec]:
    """Expand arms and training seeds while retaining fixed evaluation seeds."""
    return [
        {
            "run_id": f"{protocol['testbed']}__{arm['name']}__seed{seed}",
            "protocol": protocol,
            "arm": arm,
            "training_seed": int(seed),
        }
        for arm in protocol["arms"]
        for seed in protocol["training_seeds"]
    ]


def _decision_config(
    config: ExplicitConfig | ImplicitConfig, arm: ArmSpec
) -> ExplicitConfig | ImplicitConfig:
    """The config itself for a feedback arm; torque_decision_config's for a torque arm."""
    if arm["decision_mode"] == "torque":
        return torque_decision_config(config)
    return config


def _with_reward_observation(
    config: ExplicitConfig | ImplicitConfig, protocol: EvaluationProtocol
) -> None:
    """Write the protocol's reward_observation into an arm's config, only off its default
    (so that every other arm config, checkpoint and agent is as before the key)."""
    observation = reward_observation_setting(protocol)
    if observation != DEFAULT_REWARD_OBSERVATION:
        config[REWARD_OBSERVATION_KEY] = observation


def _explicit_arm_config(run: RunSpec) -> ExplicitConfig:
    config = _decision_config(copy.deepcopy(run["protocol"]["explicit_config"]), run["arm"])
    config["training"]["seed"] = run["training_seed"]
    config["plant_source"] = "learned"
    config["continuation"] = run["arm"]["target_mode"]
    _with_reward_observation(config, run["protocol"])
    return config


def _implicit_arm_config(run: RunSpec) -> ImplicitConfig:
    protocol, arm = run["protocol"], run["arm"]
    config = _decision_config(copy.deepcopy(protocol["implicit_config"]), arm)
    config["training"]["seed"] = run["training_seed"]
    config["plant_source"] = "oracle" if arm["plant_mode"] == "oracle" else "learned"
    _with_reward_observation(config, protocol)

    config["solver"]["residual_estimator"] = arm["residual_loss"]
    config["admission"]["allocation"] = arm["gate_allocation"]

    if not arm["rollout_rate_enabled"]:
        config["rate"]["rollout_weight"] = 0.0

    if arm["plant_mode"] == "polyak":
        config["admission"]["plant_update"] = "polyak"
        config["admission"]["polyak_rate"] = protocol["ablation_polyak_rate"]
    elif arm["plant_mode"] == "gated":
        config["admission"]["plant_update"] = "admission"
        config["admission"]["polyak_rate"] = None
    else:
        config["admission"]["plant_update"] = "always"
        config["admission"]["polyak_rate"] = None
    # The step-ablation arms' choices (OptionalArmFields); no other arm carries one.
    for name in STEP_ABLATION_CHOICES:
        if name in arm:
            config[name] = arm[name]
    return config


def make_training_agent(
    run: RunSpec,
    env: MujocoPHEnv,
) -> Union[TorqueCTSACBaseline, ExplicitCTPHSAC, ImplicitCTPHSAC]:
    """Select the torque baseline adapter or PH algorithm and its ablation settings.

    The CT-SAC baseline takes the explicit config's training settings, temperature and
    Polyak rate; an arm with ct_sac_settings (baseline_released_arms) replaces the
    learning rate, replay capacity and Polyak rate by its own, warms up for its
    warmup_decisions decisions in place of the warm-up episodes, and sets the time-limit
    convention (CTSACSettings). Every other CT-SAC arm takes the training key
    warmup_decisions, when given, as its warmup_decisions (CT-SAC's learning_starts), as
    the CT-PH-SAC arms take it for their warm-up. An arm with ct_sac_inputs
    "state_features" (baseline_feature_arms) also passes the explicit config's network as
    state_features.
    Under the protocol's ct_sac_reward_units "reference_period" every CT-SAC arm also
    passes reward_units (absent or "second": no argument, the constructor's default), and
    under its ct_sac_reward_source "end_state", or its reward_observation "end_state"
    (common.ph_config.effective_ct_sac_reward_source), reward_source "end_state" (absent
    or "interval": no argument). The CT-PH-SAC arms' configs carry reward_observation
    "end_state" when the protocol has it.
    An arm with ct_sac_implicit_step (baseline_implicit_arms) gets the same arguments plus
    the resolved ct_sac_implicit section and its actor schedule, and builds
    algorithms/ct_sac_implicit.ImplicitTorqueCTSACBaseline.
    """
    arm = run["arm"]
    protocol = run["protocol"]
    seed = run["training_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arm["algorithm"] == "ct_sac":
        from algorithms.ph_baseline import TorqueCTSACBaseline

        explicit = protocol["explicit_config"]
        training = copy.deepcopy(explicit["training"])
        training["seed"] = seed
        polyak_rate = explicit["polyak_rate"]
        released: dict = {}
        # The training key warmup_decisions (absent or null: no argument, the warm-up
        # episodes); an arm with ct_sac_settings replaces it by its own below.
        if training.get("warmup_decisions") is not None:
            released["warmup_decisions"] = training["warmup_decisions"]
        settings = arm.get("ct_sac_settings")
        if settings is not None:
            training["learning_rate"] = settings["learning_rate"]
            training["replay_capacity"] = settings["replay_capacity"]
            polyak_rate = settings["polyak_rate"]
            released = {
                "warmup_decisions": settings["warmup_decisions"],
                "time_limit": settings["time_limit"],
            }
        if arm.get("ct_sac_inputs") == "state_features":
            # The networks read CT-PH-SAC's state features (baseline_feature_arms), at the
            # explicit config's network velocity_feature_scale; no other arm passes this.
            released["state_features"] = copy.deepcopy(explicit["network"])
        units = ct_sac_reward_units_setting(protocol)
        if units != DEFAULT_CT_SAC_REWARD_UNITS:
            # The legacy learner's rewards and temperature divided by T (every CT-SAC arm,
            # the implicit ones included); the default passes nothing, as before the key.
            released["reward_units"] = units
        source = effective_ct_sac_reward_source(protocol)
        if source != DEFAULT_CT_SAC_REWARD_SOURCE:
            # T times the reward rate at each interval's end state with the held torque
            # (every CT-SAC arm; ct_sac_reward_source or reward_observation "end_state");
            # the default passes nothing, as before the keys.
            released["reward_source"] = source
        step = arm.get("ct_sac_implicit_step")
        if step is not None:
            # The implicit value step (baseline_implicit_arms): ct_sac_torque's arguments
            # (or the released arm's) plus the explicit config's ct_sac_implicit section.
            from algorithms.ct_sac_implicit import ImplicitTorqueCTSACBaseline
            from common.ph_config import ct_sac_implicit_setting

            return ImplicitTorqueCTSACBaseline(
                env,
                training,
                copy.deepcopy(explicit["temperature"]),
                polyak_rate=polyak_rate,
                expectation_samples=protocol.get("baseline_expectation_samples", 1),
                implicit=ct_sac_implicit_setting(explicit),
                actor_schedule=step["actor_schedule"],
                **released,
            )
        return TorqueCTSACBaseline(
            env,
            training,
            copy.deepcopy(explicit["temperature"]),
            polyak_rate=polyak_rate,
            expectation_samples=protocol.get("baseline_expectation_samples", 1),
            **released,
        )
    if arm["algorithm"] == "explicit":
        from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC

        return ExplicitCTPHSAC(env, _explicit_arm_config(run))

    from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC
    from environment.ph_sac.tasks import make_oracle_plant

    oracle = make_oracle_plant(env) if arm["plant_mode"] == "oracle" else None
    return ImplicitCTPHSAC(env, _implicit_arm_config(run), oracle=oracle)


# Every link within this angle (rad) of upright counts as upright.
UPRIGHT_TUBE = 0.2
# An episode held upright this long (s) without a break counts as captured.
CAPTURE_HOLD_SECONDS = 1.0
EPISODE_LEVEL_KEYS = (
    "episode_return",
    "survival_seconds",
    "rate_executed",
    "rate_horizon",
    "upright_seconds",
    "max_hold_seconds",
    "mean_return_to_go",
)
# Fields that need the environment's undiscounted interval reward.
_REWARD_LEVEL_KEYS = ("episode_return", "rate_executed", "rate_horizon", "mean_return_to_go")
# Orbit diagnostics, on environments whose reward is the negative log of a Lyapunov
# function V (env.lyapunov_ratio(state) = V / V_down, not None: acrobot-swingup's rewards,
# the cart-pole tasks' optional Lyapunov rewards): the share of the time survived with
# V / V_down below each threshold ("on the orbit", the reward's zero set, to that tolerance).
ORBIT_THRESHOLDS = {"orbit_fraction_1e_2": 1e-2, "orbit_fraction_1e_3": 1e-3}
LYAPUNOV_LEVEL_KEYS = ("lyapunov_ratio_mean", "lyapunov_ratio_final", *ORBIT_THRESHOLDS)


def is_upright(testbed: TestbedConfig, state: np.ndarray) -> bool:
    """Whether every link is within UPRIGHT_TUBE of upright, by the testbed's registry entry.

    The pendulum is upright at angle 0; Acrobot-XK at shoulder pi/2 (hanging is -pi/2) and
    elbow 0; the cart-pole at pole angle 0, its second coordinate. A testbed registered
    without an upright predicate (no upright pose) is never upright; an unregistered
    name raises ValueError.
    """
    from environment.ph_sac.registry import testbed_spec

    predicate = testbed_spec(testbed["name"]).upright
    if predicate is None:
        return False
    return bool(predicate(testbed, state, UPRIGHT_TUBE))


def evaluate_fixed_seeds(
    policy: Callable[[np.ndarray, bool], np.ndarray],
    env_factory: Callable[[int], MujocoPHEnv],
    reset_seeds: Sequence[int],
    *,
    discount_rate: float,
    deterministic: bool,
) -> Sequence[EpisodeMetrics]:
    """Measure reward return/failures; mean deployment is distinct from the soft actor.

    Each episode record adds, beside the discounted return:
      episode_return     the reward rate integrated over the time survived, undiscounted,
                         without the failure charge (from env.last_undiscounted_reward)
      survival_seconds   time until failure, or the whole episode
      rate_executed      episode_return / survival_seconds
      rate_horizon       (episode_return + time lost to failure * lambda F) / episode
                         duration, with F the failure payoff, so lambda F is the worst rate
      upright_seconds    time with every link within UPRIGHT_TUBE of upright, judged at
                         each decision boundary
      max_hold_seconds   the longest unbroken stretch of that
      mean_return_to_go  the discounted return-to-go G_t averaged over t in [0, T]; after
                         a failure G_t is F, before it G_t includes F discounted from the
                         failure time, and reward after T is not observed. The rate is
                         taken as constant within each interval.
    An environment without last_undiscounted_reward leaves the reward-level fields NaN.
    An environment whose lyapunov_ratio(state) gives V / V_down (a reward -log(max(V /
    V_down, floor)); LYAPUNOV_LEVEL_KEYS) adds, judged at each decision boundary and
    weighted by the interval as upright_seconds:
      lyapunov_ratio_mean   the time average of V / V_down over the time survived
      lyapunov_ratio_final  V / V_down at the last state
      orbit_fraction_1e_2   the share of the time survived with V / V_down < 1e-2
      orbit_fraction_1e_3   the same below 1e-3 (ORBIT_THRESHOLDS)
    """
    kind = "deterministic_mean" if deterministic else "stochastic"
    episodes: list[EpisodeMetrics] = []
    for seed in reset_seeds:
        env = env_factory(int(seed))
        state, _ = env.reset(seed=int(seed))
        rewards: list[float] = []
        starts: list[float] = []
        transition = None
        observed = True
        lam = float(env.discount_rate)
        integral = weighted = upright_time = hold = best_hold = 0.0
        ratio_of = getattr(env, "lyapunov_ratio", None)
        ratio = None if ratio_of is None else ratio_of(state)
        lyapunov = ratio is not None
        ratio_time = 0.0
        orbit_time = dict.fromkeys(ORBIT_THRESHOLDS, 0.0)
        while not env.done:
            decision = policy(state, deterministic)
            transition = env.step_dt(decision, env.sample_duration())
            rewards.append(transition["reward"])
            starts.append(transition["time"])
            interval = getattr(env, "last_undiscounted_reward", None)
            start, duration = float(transition["time"]), float(transition["duration"])
            if interval is None:
                observed = False
            else:
                integral += float(interval)
                if duration > 0.0:
                    weighted += float(interval) / lam - (float(interval) / duration) * (
                        math.exp(-lam * start) - math.exp(-lam * (start + duration))
                    ) / lam**2
            state = transition["next_state"]
            if is_upright(env.config, state):
                upright_time += duration
                hold += duration
                best_hold = max(best_hold, hold)
            else:
                hold = 0.0
            if lyapunov:
                ratio = float(ratio_of(state))
                ratio_time += ratio * duration
                for key, threshold in ORBIT_THRESHOLDS.items():
                    if ratio < threshold:
                        orbit_time[key] += duration
        record = dict(
            summarize_episode(
                reset_seed=int(seed),
                interval_rewards=rewards,
                interval_start_times=starts,
                discount_rate=discount_rate,
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                duration_seconds=transition["next_time"],
                physics_steps=int(round(transition["next_time"] / env.physics_dt)),
                policy_kind=kind,
            )
        )
        survived = float(transition["next_time"])
        horizon = float(env.config["episode_duration"])
        payoff = float(env.terminal_payoff(state))
        failed = bool(transition["terminated"])
        lost = horizon - survived if failed else 0.0
        tail = payoff * ((1.0 - math.exp(-lam * survived)) / lam + lost) if failed else 0.0
        record.update(
            {
                "episode_return": integral,
                "survival_seconds": survived,
                "rate_executed": integral / survived if survived > 0.0 else math.nan,
                "rate_horizon": (integral + lost * lam * payoff) / horizon,
                "upright_seconds": upright_time,
                "max_hold_seconds": best_hold,
                "mean_return_to_go": (weighted + tail) / horizon,
            }
        )
        if not observed:
            record.update({key: math.nan for key in _REWARD_LEVEL_KEYS})
        if lyapunov:
            scale = 1.0 / survived if survived > 0.0 else math.nan
            record.update(
                {
                    "lyapunov_ratio_mean": ratio_time * scale,
                    "lyapunov_ratio_final": float(ratio),
                    **{key: seconds * scale for key, seconds in orbit_time.items()},
                }
            )
        episodes.append(record)
        env.close()
    return episodes


def episode_level_summary(episodes: Sequence[Mapping[str, object]]) -> dict:
    """Means over episodes of the episode-level fields, and the capture rate.

    capture_rate is the share of episodes held upright CAPTURE_HOLD_SECONDS or longer.
    The orbit diagnostics (LYAPUNOV_LEVEL_KEYS) join when every episode carries them.
    """
    keys = EPISODE_LEVEL_KEYS
    if episodes and all(key in episode for episode in episodes for key in LYAPUNOV_LEVEL_KEYS):
        keys = keys + LYAPUNOV_LEVEL_KEYS
    means = {
        key: float(np.mean([float(episode[key]) for episode in episodes]))
        for key in keys
    }
    means["capture_rate"] = float(
        np.mean([episode["max_hold_seconds"] >= CAPTURE_HOLD_SECONDS for episode in episodes])
    )
    return means


def _training_counters(agent: object, evaluation_seconds: float) -> TrainingCounters:
    """Agent counters with evaluation time taken out of the training wall time."""
    counters = dict(agent.counters())
    counters["wall_seconds"] = max(counters["wall_seconds"] - evaluation_seconds, 0.0)
    return counters


def _arm_training(run: RunSpec) -> Mapping[str, object]:
    """The training config the arm's agent receives (the CT-SAC baseline: explicit's, whose
    learning rate and replay capacity a released-baseline arm replaces; records read only
    its plant-step schedule here)."""
    key = "implicit_config" if run["arm"]["algorithm"] == "implicit" else "explicit_config"
    return run["protocol"][key]["training"]


def _arm_contact_substepping(run: RunSpec) -> Optional[Mapping[str, float]]:
    """The arm's contact substepping as records store it; None for the CT-SAC baseline
    (no model) and when off (common.ph_config.contact_substepping_setting)."""
    algorithm = run["arm"]["algorithm"]
    if algorithm not in ("explicit", "implicit"):
        return None
    config = run["protocol"][f"{algorithm}_config"]
    if run["arm"]["decision_mode"] == "torque":
        config = torque_decision_config(config)
    return contact_substepping_setting(config)


def _arm_contact_handling(run: RunSpec) -> Optional[Mapping[str, object]]:
    """The arm's contact exclusion as records store it; None under "port" and for the
    CT-SAC baseline (common.ph_config.contact_exclusion_setting)."""
    algorithm = run["arm"]["algorithm"]
    if algorithm not in ("explicit", "implicit"):
        return None
    return contact_exclusion_setting(run["protocol"][f"{algorithm}_config"])


def _arm_flagged_value_gradient(run: RunSpec) -> Optional[str]:
    """The implicit arm's flagged_value_gradient as records store it: only off its default
    ("full"), None otherwise and for every other arm (which never reads it), so that every
    other record is as before the key (common.ph_config.flagged_value_gradient_setting)."""
    if run["arm"]["algorithm"] != "implicit":
        return None
    gradient = flagged_value_gradient_setting(run["protocol"]["implicit_config"])
    return None if gradient == DEFAULT_FLAGGED_VALUE_GRADIENT else gradient


def run_output_directory(run: RunSpec) -> Path:
    """<output_directory>/<testbed>/<arm>/seed_<seed>, where a run writes everything."""
    protocol = run["protocol"]
    return (
        Path(protocol["output_directory"])
        / protocol["testbed"]
        / run["arm"]["name"]
        / f"seed_{run['training_seed']}"
    )


def run_seed(run: RunSpec, resume: Optional[ResumeOptions] = None) -> RunResult:
    """Train one arm to the matched budget, evaluating on the fixed reset seeds.

    resume None runs as always. With ResumeOptions (evaluations/ph_sac/resume.py) the
    run is restartable: a finished run's completion marker returns its stored record; a
    resumable checkpoint in the run's folder is continued exactly (its fingerprint must
    be this run's, else ResumeMismatch); otherwise the run starts afresh. While it runs it
    rewrites the resumable checkpoint at its next safe point after every
    every_interactions decisions, after every evaluation and after SIGUSR1/SIGTERM; after
    a signal it then writes its records and raises ResumableStop. The records always
    hold exactly the state of the agent they describe: on a restart they are rewritten
    from the checkpoint at once, dropping anything written after it.
    """
    from environment.ph_sac.tasks import make_env

    from . import resume as resuming
    from .reporting import RECORD_NAME, load_run_records, write_run_records

    protocol, arm = run["protocol"], run["arm"]
    matched = protocol["matched"]
    testbed = matched["testbed_config"]
    output = run_output_directory(run)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / resuming.RESUME_NAME
    complete_path = output / resuming.COMPLETE_NAME
    if resume is not None and complete_path.exists():
        resuming.read_completion(complete_path, run)
        resume_path.unlink(missing_ok=True)
        (stored,) = load_run_records([str(output / RECORD_NAME)])
        return stored
    payload = None
    if resume is not None and resume_path.exists():
        payload = resuming.load_resume_payload(resume_path, run)
    with contextlib.ExitStack() as stack:
        if resume is not None:
            stack.enter_context(resuming.stop_on_signals())
        return _train_run(
            run, output, payload, resume, make_env=make_env, write_run_records=write_run_records
        )


def _train_run(
    run: RunSpec,
    output: Path,
    payload: Optional[Mapping[str, object]],
    resume: Optional[ResumeOptions],
    *,
    make_env: Callable[..., MujocoPHEnv],
    write_run_records: Callable[[RunResult, str], Sequence[str]],
) -> RunResult:
    """run_seed's training: fresh, or continued from a resumable checkpoint (payload)."""
    from . import resume as resuming

    protocol, arm = run["protocol"], run["arm"]
    matched = protocol["matched"]
    testbed = matched["testbed_config"]
    resume_path = output / resuming.RESUME_NAME

    env = make_env(testbed, decision_mode=arm["decision_mode"], seed=run["training_seed"])
    agent = make_training_agent(run, env)
    # The adaptation experiment's warm start (protocol warm_start, adaptation.py): before
    # the evaluation at 0, so that it scores the source's policy on this run's robot; a
    # restart takes everything from its resumable checkpoint instead (and the record of
    # the warm start from the runner's state).
    warm_start: Optional[Mapping[str, object]] = None
    if payload is None and "warm_start" in protocol:
        warm_start = apply_warm_start(agent, run)

    def env_factory(seed: int) -> MujocoPHEnv:
        return make_env(testbed, decision_mode=arm["decision_mode"], seed=seed)

    theory = protocol.get("theory_diagnostics")
    optimizer_steps: list[Mapping[str, object]] = []
    if theory is not None and theory["record_optimizer_steps"]:

        def observe(trace: Mapping[str, object]) -> None:
            outer = trace["context"]["outer_iteration"]
            if outer % theory["every_outer_iterations"] == 0:
                optimizer_steps.append(trace)

        agent.set_optimizer_observer(observe)

    evaluations: list[EvaluationCheckpoint] = []
    evaluation_seconds = 0.0

    def evaluate(interactions: int) -> None:
        nonlocal evaluation_seconds
        started = time.perf_counter()
        episodes = evaluate_fixed_seeds(
            agent.predict,
            env_factory,
            matched["evaluation_reset_seeds"],
            discount_rate=matched["discount_rate"],
            deterministic=protocol["deterministic_evaluation"],
        )
        evaluation_seconds += time.perf_counter() - started
        point: EvaluationCheckpoint = {
            "interactions": int(interactions),
            "episodes": list(episodes),
            "counters": _training_counters(agent, evaluation_seconds),
        }
        # Temperature and policy spread at this point (POLICY_DIAGNOSTIC_KEYS, since
        # 2026-11-11); draws nothing from the run's streams.
        diagnostics = getattr(agent, "policy_diagnostics", None)
        if diagnostics is not None:
            point["policy"] = dict(diagnostics())
        evaluations.append(point)

    checkpoint = output / "checkpoint.pt"
    dense_until, every = plant_step_schedule(_arm_training(run))
    contact = _arm_contact_substepping(run)
    exclusion = _arm_contact_handling(run)
    gradient = _arm_flagged_value_gradient(run)
    observation = reward_observation_setting(protocol)

    def current_result() -> RunResult:
        result: RunResult = {
            "run": run,
            "evaluations": evaluations,
            "counters": _training_counters(agent, evaluation_seconds),
            "solver": list(getattr(agent, "solver_records", [])),
            "admissions": list(getattr(agent, "admission_records", [])),
            "oracle": list(getattr(agent, "oracle_records", [])),
            "plant_calibration": list(getattr(agent, "calibration_records", [])),
            "checkpoint_paths": [str(checkpoint)],
            "decision_mode": arm["decision_mode"],
            "refresh_mode": refresh_mode_setting(testbed["controller"]),
            "plant_step_schedule": {"dense_until": dense_until, "every": every},
            "contact_substepping": contact,
            "contact_handling": exclusion,
        }
        if theory is not None:
            result["theory_records"] = list(getattr(agent, "theory_records", [])) + (
                _optimizer_step_records(optimizer_steps)
            )
        # The implicit arms always keep reward records; the explicit arms only under
        # reward_observation "end_state" (None otherwise).
        if getattr(agent, "reward_records", None) is not None:
            result["reward_fit"] = list(agent.reward_records)
        if observation != DEFAULT_REWARD_OBSERVATION:
            # Only off the default, so that every other record is as before the key.
            result[REWARD_OBSERVATION_KEY] = observation
        if gradient is not None:
            # An implicit arm's flagged_value_gradient, only off its default (the same).
            result[FLAGGED_VALUE_GRADIENT_KEY] = gradient
        # Only the fields a run of the adaptation experiment has (none for any other).
        result.update(adaptation_record(run, warm_start))
        return result

    def save_checkpoint() -> None:
        # Replace atomically (written, fsynced, renamed) so that a crash mid-save keeps
        # the previous checkpoint. The CT-SAC baseline's counters sidecar follows its file.
        partial = checkpoint.with_suffix(".partial")
        agent.save(partial, include_replay=protocol.get("checkpoint_replay", True))
        resuming.fsync_file(partial)
        sidecar = partial.with_name(partial.name + ".counters.json")
        partial.replace(checkpoint)
        if sidecar.exists():
            resuming.fsync_file(sidecar)
            sidecar.replace(checkpoint.with_name(checkpoint.name + ".counters.json"))

    period = protocol["evaluation_every_interactions"]
    next_evaluation = period
    # The resumable checkpoint: due at the next safe point after an evaluation, every
    # resume_every decisions and on a stop request (signal or stop_after_interactions).
    resume_due = False
    resume_every = 0 if resume is None else resume["every_interactions"]
    stop_after = None if resume is None else resume["stop_after_interactions"]
    last_written = -1

    if payload is not None:
        agent.load_resume_state(payload["agent"])
        saved = payload["runner"]
        evaluations[:] = saved["evaluations"]
        optimizer_steps[:] = saved["optimizer_steps"]
        evaluation_seconds = float(saved["evaluation_seconds"])
        next_evaluation = int(saved["next_evaluation"])
        warm_start = saved.get("warm_start")
        last_written = int(payload["interactions"])
        # Last, so that building and loading the agent leave no trace in them.
        resuming.restore_global_rng(payload["global_rng"])
        # Drop any rows written after this checkpoint (an evaluation before the kill).
        write_run_records(current_result(), str(output))
    start = int(agent.counters()["interactions"])
    next_resume = (start // resume_every + 1) * resume_every if resume_every else None

    def callback(_: object, metrics: Mapping[str, float]) -> None:
        nonlocal next_evaluation, next_resume, resume_due
        interactions = int(metrics["interactions"])
        if interactions >= next_evaluation:
            evaluate(interactions)
            next_evaluation = (interactions // period + 1) * period
            save_checkpoint()
            write_run_records(current_result(), str(output))
            resume_due = True
        if next_resume is not None and interactions >= next_resume:
            next_resume = (interactions // resume_every + 1) * resume_every
            resume_due = True
        if stop_after is not None and start < stop_after <= interactions:
            resuming.STOP.request(0)

    def resume_point(at_safe_point: object) -> None:
        # Called by the agent at a safe point (set_resume_hook): nothing has been drawn
        # for the next row yet, so the saved state restarts exactly here.
        nonlocal resume_due, last_written
        stopping = resuming.STOP.requested
        if not (resume_due or stopping):
            return
        resume_due = False
        interactions = int(at_safe_point.counters()["interactions"])
        if interactions > 0 and interactions != last_written:
            runner = {
                "evaluations": evaluations,
                "evaluation_seconds": evaluation_seconds,
                "next_evaluation": next_evaluation,
                "optimizer_steps": optimizer_steps,
            }
            if warm_start is not None:
                runner["warm_start"] = warm_start
            resuming.atomic_torch_save(
                resuming.resume_payload(run, at_safe_point, runner), resume_path
            )
            last_written = interactions
        if stopping:
            raise resuming.ResumableStop(
                f"{run['run_id']} stopped at {interactions} decisions"
                + (f" (signal {resuming.STOP.signal})" if resuming.STOP.signal else "")
                + f"; resumable checkpoint {resume_path if interactions else 'none'}"
            )

    if resume is not None:
        agent.set_resume_hook(resume_point)
    if payload is None:
        evaluate(0)
    try:
        agent.learn(matched["interaction_budget"], callback=callback)
    except resuming.ResumableStop:
        write_run_records(current_result(), str(output))
        env.close()
        raise
    except Exception:
        # Keep what a failed run measured; its state may be invalid, so don't checkpoint it.
        write_run_records(current_result(), str(output))
        raise
    final = int(agent.counters()["interactions"])
    if evaluations[-1]["interactions"] != final:
        evaluate(final)

    save_checkpoint()
    env.close()

    result = current_result()
    write_run_records(result, str(output))
    if resume is not None:
        resuming.write_completion(output / resuming.COMPLETE_NAME, run, final)
        resume_path.unlink(missing_ok=True)
    return result


def _optimizer_step_records(
    steps: Sequence[Mapping[str, object]],
) -> list[TheoryDiagnosticsRecord]:
    """Group recorded optimizer steps by their exact analysis context."""
    records: list[TheoryDiagnosticsRecord] = []
    for step in steps:
        if not records or records[-1]["context"] != step["context"]:
            records.append(
                {
                    "context": step["context"],
                    "solve_context": None,
                    "sampled": [],
                    "optimizer_steps": [],
                    "training_accuracy": None,
                    "outer_step": None,
                    "original_iteration_budget": None,
                    "performance_bound": None,
                }
            )
        records[-1]["optimizer_steps"].append(step)
    return records


def run_protocol(
    protocol: EvaluationProtocol, resume: Optional[ResumeOptions] = None
) -> Sequence[RunResult]:
    """Run all arms and seeds, reporting interactions, gradient steps, and time.

    With resume (run_seed), finished runs are skipped, a stopped run continues, and a
    stop request (SIGUSR1/SIGTERM) between runs raises ResumableStop before the next.
    """
    from . import resume as resuming
    from .reporting import write_protocol_manifest

    validate_matched_quantities(protocol)
    validate_warm_start_arms(protocol)
    output = Path(protocol["output_directory"]) / protocol["testbed"]
    output.mkdir(parents=True, exist_ok=True)
    write_protocol_manifest(protocol, str(output / "protocol.json"))
    if resume is None:
        return [run_seed(run) for run in build_run_matrix(protocol)]
    results = []
    with resuming.stop_on_signals():
        for run in build_run_matrix(protocol):
            if resuming.STOP.requested:
                raise resuming.ResumableStop(
                    f"stop requested (signal {resuming.STOP.signal}) before {run['run_id']}"
                )
            results.append(run_seed(run, resume))
    return results
