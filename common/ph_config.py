"""Configuration schemas and the in-house testbeds' preset factories.

The dm_control ports' factories are in common/ph_dmc/ (cartpole_swingup_config is
re-exported here).
"""

from __future__ import annotations

import math
from typing import (
    TYPE_CHECKING,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
)

import numpy as np

# The dm_control ports' factories live in common/ph_dmc/; re-exported for existing callers.
from common.ph_dmc.cartpole import cartpole_swingup_config  # noqa: F401
from common.ph_types import StateSpec, TestbedName

if TYPE_CHECKING:
    from numpy import ndarray


class ClockConfig(TypedDict):
    kind: Literal["fixed", "two_tail_uniform", "two_tail_beta"]
    physics_dt: float
    reference_period: float
    min_duration: float
    max_duration: float
    tail_probability: Optional[float]
    tail_split: Optional[float]
    beta_shape: Optional[float]


class ControllerConfig(TypedDict):
    """The decision box, the motor limits and the feedback law's settings.

    decision_mode (NotRequired, default "feedback", see DECISION_MODES) says what a
    decision is. "feedback": the impedance law (setpoint, stiffness, damping) per
    actuator, a box of dimension 3m, the box every factory, config and table row holds.
    "torque": the motor torque itself, held over the interval as in the CT-SAC baseline
    (the paper arms); the box is then the torque box [torque_low, torque_high] of
    dimension m (torque_decision_controller builds it per run from the feedback
    controller), and setpoint_mode and setpoint_error are unused.

    refresh_mode (NotRequired, default "continuous", see REFRESH_MODES) says when a
    feedback law is evaluated. "continuous": from the current state at every stage of
    every integrator step, in the environment (MuJoCo's control callback) and in the
    model; refresh_steps is then unused. "held": the sample-and-hold of every run before
    2026-11-09, the torque refreshed every refresh_steps physics steps and held between,
    and the model's substep grid split at every refresh. A torque decision is held over
    its interval in either mode; the mode only sets its model grid as for a feedback law.
    """

    decision_low: ndarray
    decision_high: ndarray
    torque_low: ndarray
    torque_high: ndarray
    setpoint_mode: Literal["absolute", "offset"]
    setpoint_error: Literal["wrapped", "unwrapped"]
    refresh_steps: int
    decision_mode: NotRequired[Literal["feedback", "torque"]]
    refresh_mode: NotRequired[Literal["continuous", "held"]]


class TestbedConfig(TypedDict):
    """A testbed's settings. A config of the adaptation experiment also carries
    physical_change ({"kind", "scale"}) and physical_plant_parameters, the simulated
    robot's parameters (common.ph_dmc._common.with_physical_change); plant_parameters then
    stay the nominal robot's, which the reward and the state spec read."""

    name: TestbedName
    state_spec: StateSpec
    physics_dt: float
    reference_period: float
    episode_duration: float
    discount_rate: float
    clock: ClockConfig
    controller: ControllerConfig
    integrator: Literal["Euler", "RK4"]
    plant_parameters: Mapping[str, float]
    failure_limits: Mapping[str, float]
    reward_parameters: Mapping[str, float]
    reset_parameters: Mapping[str, object]


class OptionalNetworkFields(TypedDict, total=False):
    """Network settings a config may leave out (NotRequired is not honoured here: with
    postponed annotations this module's TypedDicts count every NotRequired key as
    required, so optional keys sit in a total=False base, as OptionalAdmissionFields)."""

    damping_velocity: bool
    limit_port: bool


class NetworkConfig(OptionalNetworkFields):
    """hidden_sizes shape actor/value/rate/critics; plant_hidden_sizes shape the plant.

    diffusion (default "learned") selects the learned plant's noise model: "learned" fits
    a diffusion above diffusion_floor; "none" is a drift-only plant with zero diffusion,
    one noise column, zero covariance and the drift loss 0.5*|e|^2/Delta.

    damping_velocity (default False) lets the learned plant's damping factor read the
    velocities as well, D(y, v) = L(y, v) L(y, v)^T (velocities divided by
    velocity_feature_scale), for drag that grows with speed (the swimmer's fluid);
    False keeps D(y), the plant of every row before 2026-11-09.

    limit_port (default False) adds to the learned plant a one-sided spring and damper
    beyond each range of the state spec's limited_position_ranges (learned stiffness,
    damping and ramp width per coordinate, exactly zero inside the range, starting at the
    state spec's limit_port_init when given), for tasks that keep MuJoCo's soft joint
    limits (the swimmers, cart-poles, reachers and point mass).

    actor_log_std_init and input_matrix_start (NETWORK_COLUMNLESS_KEYS; absent: every
    earlier run) are read from a network config but are no fields here, as training's
    warmup_decisions: every field has a column of the hyperparameter table, whose CSV
    stays fixed, so they are JSON-config keys (benchmarks/run_ph_sac.py
    _COLUMNLESS_NETWORK). See the block at NETWORK_COLUMNLESS_KEYS.
    """

    hidden_sizes: Tuple[int, ...]
    plant_hidden_sizes: Tuple[int, ...]
    activation: Literal["tanh"]
    mass_floor: float
    diffusion_floor: float
    damping_rank: int
    velocity_feature_scale: NotRequired[float]
    diffusion: NotRequired[Literal["learned", "none"]]


class OptionalTrainingFields(TypedDict, total=False):
    """Training settings a config may leave out (see OptionalNetworkFields for why a base).

    plant_step_dense_until and plant_step_every (defaults PLANT_STEP_DEFAULTS: 0 and 1,
    one step per row, the schedule of every run before 2026-11-10) schedule the learned
    plant's online gradient step, one per collected row (the implicit arms' inner step,
    the explicit arms' train step): every row while the interaction count is at most
    plant_step_dense_until, then every plant_step_every-th row. See plant_step_schedule.

    plant_diagnostics_every (default PLANT_DIAGNOSTICS_EVERY, 10,000; 0 disables): the
    interaction cadence of the sampled plant diagnostics (oracle[] and the calibration
    records, algorithms.ph_base.PHAlgorithm.record_plant_diagnostics). Each one draws a
    replay batch, so the cadence is part of the run (and of its resume fingerprint when
    given).

    warmup_decisions (absent or None: the warmup_episodes episodes of every earlier run;
    warmup_decisions_setting) is read from a training config but is no field here: every
    field has a column of the hyperparameter table (tests/test_ph_hyperparams.py), whose
    CSV stays fixed, so it is a JSON-config key (benchmarks/run_ph_sac.py
    _COLUMNLESS_TRAINING) and enters a run's fingerprint only when given.
    """

    plant_step_dense_until: int
    plant_step_every: int
    plant_diagnostics_every: int


class TrainingConfig(OptionalTrainingFields):
    seed: int
    device: str
    replay_capacity: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    warmup_episodes: int
    validation_fraction: float


class TemperatureConfig(TypedDict):
    """target_entropy is for the actor's density over the normalized decision box."""

    initial_alpha_c: float
    learn_temperature: bool
    target_entropy: float


class SolverConfig(TypedDict):
    """inner_steps is K, the fixed number of value steps in every inner solve."""

    resolvent_step: float
    policy_draws: int
    inner_steps: int
    residual_estimator: Literal["all_pairs", "averaged_draw"]


class RateConfig(TypedDict):
    replay_weight_max: float
    noise_variance_floor: float
    rollout_weight: float
    model_weight: float
    rollout_paths: int
    model_points: int
    rate_steps: int
    actor_steps: int


class OptionalAdmissionFields(TypedDict, total=False):
    """Declared inputs of inference=confidence_budgeted; empirical mode needs none.

    return_bound and min_horizon bound observed and numerical model returns; they are
    assumptions, not empirical maxima, and a row that breaks one raises an error.
    """

    gate_failure_budget: float
    return_bound: float
    min_horizon: float


class AdmissionConfig(OptionalAdmissionFields):
    """How the admitted plant follows the online plant.

    plant_update "always" (the default) copies the online plant into the admitted plant
    at every cycle end without a test; the transfer score J is recorded and decides
    nothing. "admission" runs the budgeted staged gate, whose allocation=fixed drops
    pilots and only doubles rows. "polyak" is the moving-plant ablation at polyak_rate.
    The cycle is M=admission_spacing(lambda, tau, efolds) outer iterations in every mode.
    """

    efolds: float
    margin_z: float
    initial_rows: int
    initial_paths: int
    pilot_batches: int
    real_row_budget: int
    simulated_path_budget: int
    allocation: Literal["pilot", "fixed"]
    inference: Literal["empirical", "confidence_budgeted"]
    plant_update: Literal["always", "admission", "polyak"]
    polyak_rate: Optional[float]


class ExplicitConfig(TypedDict):
    """Explicit CT-PH-SAC settings; model_integrator defaults to DEFAULT_MODEL_INTEGRATOR.

    model_coarse_step (NotRequired; absent or None: off, as in every earlier run)
    turns on contact-triggered fine substepping (model_coarse_step_setting,
    models/ph_sac/rollouts.ContactSubstepping): the model integrates at the coarse step
    and re-integrates at model_step the rows that come near a joint limit.

    contact_handling (NotRequired; absent: "port", every earlier run) is how a task with
    joint limits treats intervals near them (CONTACT_HANDLINGS, contact_handling_setting):
    "exclude" leaves them out of the plant fit and replaces every model use at them by
    the observed transition (the block at CONTACT_HANDLINGS).

    ct_sac_implicit (NotRequired; absent: CT_SAC_IMPLICIT_DEFAULTS) holds the numbers of
    the CT-SAC arms with the implicit value step (ct_sac_implicit_setting); no other arm
    reads it.

    reward_observation (NotRequired; absent: "interval", every earlier run) is the
    protocol's key (the block at REWARD_OBSERVATION_KEY), written into an arm's config only
    as "end_state": the agent then learns from the end-state reward rate alone.
    """

    testbed: TestbedConfig
    network: NetworkConfig
    training: TrainingConfig
    temperature: TemperatureConfig
    plant_source: Literal["learned", "oracle"]
    continuation: Literal["generator", "rollout", "td"]
    polyak_rate: float
    rollout_paths: int
    model_step: float
    model_integrator: NotRequired[Literal["rk4", "semi_implicit"]]
    model_coarse_step: NotRequired[Optional[float]]
    contact_handling: NotRequired[Literal["port", "exclude"]]
    ct_sac_implicit: NotRequired[Mapping[str, object]]
    reward_observation: NotRequired[Literal["interval", "end_state"]]


class ImplicitConfig(TypedDict):
    """Implicit CT-PH-SAC settings; the NotRequired keys default to IMPLICIT_DEFAULTS.

    reward_source: "learned" fits a reward-rate network to observed rows and the value
    step reads its copy frozen with the operator; "known" reads the task's function.
    reward_warm_steps / plant_warm_steps: reward and online-plant steps on replay batches
    once warm-up has filled a batch, before the first value step; the warm-fitted plant
    is admitted untested. plant_fit_covariance: endpoint covariance of the plant
    likelihood. temperature_every_rows: rows collected per tuning step of log alpha_c
    (warm-up rows not counted), taken in a burst at the end of each outer iteration
    (learned temperature only).
    model_integrator (both arms; default DEFAULT_MODEL_INTEGRATOR): the substep map of
    every model integration, see MODEL_INTEGRATORS. model_coarse_step (both arms;
    absent or None: off): contact-triggered fine substepping, see ExplicitConfig.
    contact_handling (both arms; absent: "port"): see ExplicitConfig.
    flagged_value_gradient (absent: "full", every earlier run): the value residual's
    gradient at contact exclusion's flagged rows, see the block at
    FLAGGED_VALUE_GRADIENT_KEY; "stop_successor" needs contact_handling "exclude".
    trainable_plant_parts (absent: every part, every earlier run): the learned plant's
    parts (PLANT_PARTS) its online fit moves; the others keep their values
    (trainable_plant_parts_setting; the adaptation experiment's structured fine-tune).
    value_step, explicit_step_size, decision_head, value_operator and
    critic_reference_interval (absent: STEP_ABLATION_DEFAULTS, every earlier run): the
    value-step ablation, see the block at STEP_ABLATION_DEFAULTS.
    reward_observation (both arms; absent: "interval"): see ExplicitConfig; "end_state"
    needs reward_source "learned".
    """

    testbed: TestbedConfig
    network: NetworkConfig
    training: TrainingConfig
    temperature: TemperatureConfig
    plant_source: Literal["learned", "oracle"]
    solver: SolverConfig
    rate: RateConfig
    admission: AdmissionConfig
    model_step: float
    model_integrator: NotRequired[Literal["rk4", "semi_implicit"]]
    model_coarse_step: NotRequired[Optional[float]]
    contact_handling: NotRequired[Literal["port", "exclude"]]
    flagged_value_gradient: NotRequired[Literal["full", "stop_successor"]]
    reward_source: NotRequired[Literal["learned", "known"]]
    reward_warm_steps: NotRequired[int]
    plant_warm_steps: NotRequired[int]
    plant_fit_covariance: NotRequired[Literal["leading_order", "propagated"]]
    temperature_every_rows: NotRequired[float]
    trainable_plant_parts: NotRequired[Tuple[str, ...]]
    value_step: NotRequired[Literal["implicit", "explicit"]]
    explicit_step_size: NotRequired[Literal["resolvent_step", "interval"]]
    decision_head: NotRequired[Literal["rate", "critic"]]
    value_operator: NotRequired[Literal["model", "critic"]]
    critic_reference_interval: NotRequired[Optional[float]]
    reward_observation: NotRequired[Literal["interval", "end_state"]]


# The learned plant's parts (models/ph_sac/plant.PortHamiltonianPlant.part_parameters),
# each a separate set of parameters: the potential network U(y), the mass network
# (Cholesky factor of M(y)), the damping network (factor of D(y) or D(y, v)), the input map
# G (force_matrix), the diffusion network (absent from a drift-only plant) and the limit
# port's stiffness, damping and width (present with network limit_port). The implicit key
# trainable_plant_parts names the ones the online fit moves (trainable_plant_parts_setting).
PLANT_PARTS: Tuple[str, ...] = ("potential", "mass", "damping", "input", "diffusion", "limit_port")


def trainable_plant_parts_setting(config: Mapping[str, object]) -> Optional[Tuple[str, ...]]:
    """config["trainable_plant_parts"] as a tuple, None when absent (every part trains).

    Raises ValueError unless it is a nonempty list of distinct names from PLANT_PARTS;
    whether the plant has each part (diffusion, limit_port) is checked when it is built.
    """
    if "trainable_plant_parts" not in config:
        return None
    parts = config["trainable_plant_parts"]
    if isinstance(parts, str) or not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError(
            f"trainable_plant_parts must be a nonempty list of plant parts, not {parts!r}."
        )
    unknown = [part for part in parts if part not in PLANT_PARTS]
    if unknown or len(set(parts)) != len(parts):
        raise ValueError(
            f"trainable_plant_parts must name distinct parts of {PLANT_PARTS}, not {list(parts)!r}."
        )
    return tuple(str(part) for part in parts)


IMPLICIT_DEFAULTS: Mapping[str, object] = {
    "reward_source": "learned",
    "reward_warm_steps": 2000,
    "plant_warm_steps": 2000,
    "plant_fit_covariance": "leading_order",
    "temperature_every_rows": 40,
}
DEFAULT_DIFFUSION = "learned"
WINDY_TASKS = ("balance", "swingup")

# The learned plant's online-step schedule (training keys plant_step_dense_until and
# plant_step_every, OptionalTrainingFields). Each collected row offers the plant one
# gradient step: the implicit arms' inner step (after the value step, on its own replay
# batch) and the explicit arms' train step (on the shared batch). The step is taken on
# every row whose interaction count (the row's 1-based index, warm-up included) is at most
# plant_step_dense_until, and after that on every plant_step_every-th row offered (a
# counter that checkpoints carry). The defaults take every row: the schedule of every run
# before 2026-11-10, reproduced bitwise, and of any config or record without the keys.
# A skipped row draws no plant batch. The implicit warm fit (plant_warm_steps), the reward
# head's per-row step, the explicit arms' Polyak update of their target plant (every train
# step, toward the online plant as it stands) and the known-plant arm are unchanged.
# Motivation (2026-11-10): the plant step is 63-70% of an implicit decision's time and
# ~50% of a rollout arm's; learned plants reach 0.2-0.3% drift error by ~100k decisions
# and stay flat, but are ~6% off right after the warm fit, so the early phase stays dense.
PLANT_STEP_DEFAULTS: Mapping[str, int] = {"plant_step_dense_until": 0, "plant_step_every": 1}
# The default cadence (interactions) of the sampled plant diagnostics (training key
# plant_diagnostics_every, OptionalTrainingFields): the class attribute
# algorithms.ph_base.PHAlgorithm.plant_diagnostics_every of every earlier run.
PLANT_DIAGNOSTICS_EVERY = 10_000


def plant_diagnostics_setting(training: Mapping[str, object]) -> Optional[int]:
    """training["plant_diagnostics_every"], None when absent (the agent's default).

    Raises ValueError unless it is a nonnegative integer (0 disables the diagnostics).
    """
    if "plant_diagnostics_every" not in training:
        return None
    value = training["plant_diagnostics_every"]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            f"training plant_diagnostics_every must be a nonnegative integer, not {value!r}."
        )
    return int(value)


def warmup_decisions_setting(training: Mapping[str, object]) -> Optional[int]:
    """training["warmup_decisions"], None when absent or None (the episode warm-up).

    An integer makes the CT-PH-SAC warm-up exactly that many uniform decisions, counted
    across episode ends, in place of warmup_episodes episodes
    (algorithms.ph_base.PHAlgorithm.warmup); the CT-SAC arms without released settings
    take it as their learning_starts (evaluations/ph_sac/protocol.make_training_agent).
    Raises ValueError unless it is a nonnegative integer.
    """
    value = training.get("warmup_decisions")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            f"training warmup_decisions must be a nonnegative integer or null, not {value!r}."
        )
    return int(value)


# The CT-PH-SAC networks' column-less keys (NetworkConfig; JSON configs only, since
# 2026-11-20). Without them every agent is built as before, bitwise.
#   actor_log_std_init  the start of every entry of the actor's state-free log standard
#                       deviation (models.ph_sac.networks.DecisionActor, every CT-PH-SAC
#                       arm), a finite number in [LOG_STD_MIN, LOG_STD_MAX] of that module;
#                       absent: its LOG_STD_INIT (-1). The CT-SAC arms' actor ignores it.
#   input_matrix_start  where the learned plant's input matrix G (force_matrix) starts:
#                       "true" (DEFAULT_INPUT_MATRIX_START, the task's G set by
#                       initialize_from_physics), "random": right after that, before any
#                       copy of the plant or its optimizer exists, every entry of G is
#                       replaced by an i.i.d. standard normal, drawn in float64 from a
#                       private CPU generator seeded INPUT_MATRIX_START_SEED_OFFSET + the
#                       training seed and cast to G's dtype, so no other random stream of
#                       the run moves (algorithms.ph_base.PHAlgorithm._random_input_matrix),
#                       or "zero": right after initialize_from_physics, before any copy of
#                       the plant or its optimizer exists, every entry of G is set to
#                       exactly 0.0 in place; no random stream is touched
#                       (algorithms.ph_base.PHAlgorithm._zero_input_matrix). An arm with
#                       the known plant (plant_source "oracle") refuses "random" and
#                       "zero".
# A run's resume fingerprint holds each only off its default, resolved
# (evaluations/ph_sac/resume._fingerprint_protocol, network_start_settings), so that a
# config stating a default resumes a run started without the key, and back.
NETWORK_COLUMNLESS_KEYS = ("actor_log_std_init", "input_matrix_start")
INPUT_MATRIX_STARTS = ("true", "random", "zero")
DEFAULT_INPUT_MATRIX_START = "true"
INPUT_MATRIX_START_SEED_OFFSET = 977_000


def actor_log_std_init_setting(network: Mapping[str, object]) -> Optional[float]:
    """network["actor_log_std_init"] as a float, None when absent (LOG_STD_INIT).

    Raises ValueError unless it is a finite number (not a bool) within
    [LOG_STD_MIN, LOG_STD_MAX] of models.ph_sac.networks.
    """
    if "actor_log_std_init" not in network:
        return None
    # models.ph_sac.networks imports this module, so its bounds are imported here.
    from models.ph_sac.networks import LOG_STD_MAX, LOG_STD_MIN

    value = network["actor_log_std_init"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not LOG_STD_MIN <= float(value) <= LOG_STD_MAX
    ):
        raise ValueError(
            f"network actor_log_std_init must be a finite number in [{LOG_STD_MIN}, "
            f"{LOG_STD_MAX}], not {value!r}."
        )
    return float(value)


def input_matrix_start_setting(network: Mapping[str, object]) -> str:
    """network["input_matrix_start"], DEFAULT_INPUT_MATRIX_START ("true") when absent.

    Raises ValueError unless it is one of INPUT_MATRIX_STARTS.
    """
    value = network.get("input_matrix_start", DEFAULT_INPUT_MATRIX_START)
    if not isinstance(value, str) or value not in INPUT_MATRIX_STARTS:
        raise ValueError(
            f"network input_matrix_start must be one of {INPUT_MATRIX_STARTS}, not {value!r}."
        )
    return value


def network_start_settings(network: Mapping[str, object]) -> dict:
    """The column-less keys a network config sets off their defaults, resolved (checked):
    actor_log_std_init as a float unless it equals LOG_STD_INIT, input_matrix_start
    unless it is "true". {} for a config without them (every earlier one)."""
    from models.ph_sac.networks import LOG_STD_INIT

    settings: dict = {}
    start = actor_log_std_init_setting(network)
    if start is not None and start != float(LOG_STD_INIT):
        settings["actor_log_std_init"] = start
    matrix = input_matrix_start_setting(network)
    if matrix != DEFAULT_INPUT_MATRIX_START:
        settings["input_matrix_start"] = matrix
    return settings


# Values of the top-level config key "model_integrator": the map that advances the learned
# (or oracle) plant over one model substep wherever a model integrates it over a decision
# interval (plant fit, rollout continuation, rate-head rollouts, admission paths, plant
# diagnostics); models/ph_sac/rollouts.model_substep applies it. Within a substep the
# controller torque is held, refreshed only at the substep's start when the refresh grid
# flags it.
#   "rk4"            classical fourth-order Runge-Kutta of the drift at the held torque:
#                    MuJoCo's RK4 step (mjINT_RK4), so on a testbed simulated with RK4 at
#                    the model step the true drift reproduces the simulator's endpoint.
#   "semi_implicit"  the velocity-first semi-implicit Euler(-Maruyama) step, the only
#                    integrator before 2026-11-09, kept so that earlier runs stay
#                    reproducible; it is MuJoCo's Euler step (mjINT_EULER) when no joint
#                    has damping, which MuJoCo's Euler treats implicitly (eulerdamp:
#                    M + h diag(dof_damping) in the velocity update). The rows simulated
#                    with Euler use it at their physics step: the windy pendulum
#                    (damping 0) and, since 2026-11-10, both swimmers (dm_control's own
#                    2 ms Euler; no dof damping, the fluid force and the soft joint limits
#                    evaluated at the step's start by mj_forward, as the drift is).
# A config without the key (every run before 2026-11-09) reads DEFAULT_MODEL_INTEGRATOR.
MODEL_INTEGRATORS = ("rk4", "semi_implicit")
DEFAULT_MODEL_INTEGRATOR = "rk4"


def implicit_setting(config: ImplicitConfig, name: str) -> object:
    """Read an optional implicit setting, falling back to IMPLICIT_DEFAULTS."""
    return config.get(name, IMPLICIT_DEFAULTS[name])


# The value-step ablation (implicit config keys, since 2026-11-13; the arms of
# evaluations/ph_sac/protocol.step_ablation_arms set them, every other arm leaves them
# absent). With the operator's soft rate at a value W and a frozen policy draw kappa_i,
#     model   rate_i(W) = rho_i + G^(kappa_i) W - lambda W - alpha_c log pi_i
#             (the learned plant and reward frozen with the operator; contact exclusion's
#             observed-interval rate at flagged rows, algorithms/ph_resolvent.py),
#     critic  rate_i(W) = (Q(s, kappa_i) - W(s)) / Delta_ref - alpha_c log pi_i
#             (the twin critic's minimum, frozen with the operator),
# the value step of an outer iteration fits V_(k+1) to the root, over the policy average
# of the N draws, of
#     implicit  R_i = V - V_k - tau rate_i(V)    (the resolvent step, tau = resolvent_step)
#     explicit  R_i = V - V_k - h rate_i(V_k)    (one explicit Euler step of length h)
# with the solver's K inner steps and residual_estimator either way. Keys:
#   value_step          "implicit" (default) or "explicit".
#   explicit_step_size  h of the explicit step: "resolvent_step" (default, h = tau, the
#                       like-for-like ablation) or "interval" (h = each replay row's own
#                       elapsed Delta, the explicit_generator arm's step). Only with
#                       value_step "explicit".
#   decision_head       what the actor reads and what is fitted after each copy:
#                       "rate" (default, the rate head on its three targets) or "critic"
#                       (explicit_generator's twin critic on r + (1-d)[V + Delta(G V -
#                       lambda V)], V the copied value, the actor on its minimum with the
#                       interval temperature alpha_c w_lambda(Delta)).
#   value_operator      "model" (default) or "critic"; "critic" needs decision_head
#                       "critic", learns no reward rate, and needs a plant_update other
#                       than "admission" (the gate simulates returns with a reward rate).
#   critic_reference_interval  Delta_ref in seconds of the critic operator; absent or
#                       None: the clock's mean decision interval
#                       (common.ph_time.mean_decision_interval). The only key a JSON
#                       config's implicit section may set.
# With every choice at its default the agent runs the code before the keys, bitwise.
VALUE_STEPS = ("implicit", "explicit")
EXPLICIT_STEP_SIZES = ("resolvent_step", "interval")
DECISION_HEADS = ("rate", "critic")
VALUE_OPERATORS = ("model", "critic")
STEP_ABLATION_CHOICES: Mapping[str, Tuple[str, ...]] = {
    "value_step": VALUE_STEPS,
    "explicit_step_size": EXPLICIT_STEP_SIZES,
    "decision_head": DECISION_HEADS,
    "value_operator": VALUE_OPERATORS,
}
STEP_ABLATION_DEFAULTS: Mapping[str, object] = {
    "value_step": "implicit",
    "explicit_step_size": "resolvent_step",
    "decision_head": "rate",
    "value_operator": "model",
    "critic_reference_interval": None,
}


class StepAblationSetting(TypedDict):
    """step_ablation_setting's result; critic_reference_interval is resolved (seconds)
    under value_operator "critic" and as given (None or seconds) otherwise."""

    value_step: str
    explicit_step_size: str
    decision_head: str
    value_operator: str
    critic_reference_interval: Optional[float]


def step_ablation_setting(config: Mapping[str, object]) -> StepAblationSetting:
    """Read and check the value-step ablation keys, STEP_ABLATION_DEFAULTS when absent.

    Raises ValueError for a value outside STEP_ABLATION_CHOICES, an explicit_step_size
    other than the default without value_step "explicit", value_operator "critic"
    without decision_head "critic", or a critic_reference_interval that is not None or a
    positive finite number. Under value_operator "critic" an absent (None)
    critic_reference_interval resolves to the mean decision interval of
    config["testbed"]["clock"].
    """
    setting = {}
    for name, allowed in STEP_ABLATION_CHOICES.items():
        value = config.get(name, STEP_ABLATION_DEFAULTS[name])
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}.")
        setting[name] = value
    if (
        setting["value_step"] != "explicit"
        and setting["explicit_step_size"] != STEP_ABLATION_DEFAULTS["explicit_step_size"]
    ):
        raise ValueError("explicit_step_size applies to value_step explicit only.")
    if setting["value_operator"] == "critic" and setting["decision_head"] != "critic":
        raise ValueError("value_operator critic reads the critic: set decision_head critic.")
    reference = config.get("critic_reference_interval")
    if reference is not None and (
        isinstance(reference, bool)
        or not isinstance(reference, (int, float, np.integer, np.floating))
        or not math.isfinite(float(reference))
        or float(reference) <= 0.0
    ):
        raise ValueError(
            f"critic_reference_interval must be a positive number of seconds or null, "
            f"not {reference!r}."
        )
    if reference is not None:
        reference = float(reference)
    elif setting["value_operator"] == "critic":
        # common.ph_time imports this module, so it is imported where it is needed.
        from common.ph_time import mean_decision_interval

        reference = mean_decision_interval(config["testbed"]["clock"])
    setting["critic_reference_interval"] = reference
    return setting  # type: ignore[return-value]


def step_ablation_active(setting: Mapping[str, object]) -> bool:
    """Whether a step_ablation_setting departs from the default value update: any choice
    key off its default (critic_reference_interval alone changes nothing)."""
    return any(setting[name] != STEP_ABLATION_DEFAULTS[name] for name in STEP_ABLATION_CHOICES)


# CT-SAC with the implicit value step (since 2026-11-14; the arms of
# evaluations/ph_sac/protocol.baseline_implicit_arms, algorithms/ct_sac_implicit.py). The
# explicit config's optional section CT_SAC_IMPLICIT_KEY (a JSON config's "explicit"
# object; no table column) sets its numbers; absent keys take CT_SAC_IMPLICIT_DEFAULTS,
# implicit CT-PH-SAC's values in the paper's configs:
#   resolvent_step          tau (s), the implicit step's size (implicit solver resolvent_step)
#   inner_steps             K, training rows (residual steps) per solve, then the hard copy
#                           (implicit solver inner_steps)
#   policy_draws            N >= 2 joint draws per row at s and s' (implicit solver policy_draws)
#   actor_steps             actor steps after each copy, actor schedule "outer" only
#                           (implicit rate actor_steps)
#   temperature_every_rows  training rows per temperature step, schedule "outer" only
#                           (implicit temperature_every_rows)
# No other arm reads the section, and a config without it leaves every fingerprint as it
# was; the implicit CT-SAC arms' fingerprints hold the resolved values
# (evaluations/ph_sac/resume._resolved_settings).
CT_SAC_IMPLICIT_KEY = "ct_sac_implicit"
CT_SAC_IMPLICIT_DEFAULTS: Mapping[str, object] = {
    "resolvent_step": 1.0,
    "inner_steps": 400,
    "policy_draws": 4,
    "actor_steps": 40,
    "temperature_every_rows": 40,
}
# The arms' actor schedules (protocol.CTSACImplicitStep): "per_row", the actor and the
# temperature step on every row as in CT-SAC; "outer", after each copy as in implicit
# CT-PH-SAC, the policy frozen through each solve.
CT_SAC_ACTOR_SCHEDULES = ("per_row", "outer")


def ct_sac_implicit_setting(config: Mapping[str, object]) -> dict:
    """config[CT_SAC_IMPLICIT_KEY] over CT_SAC_IMPLICIT_DEFAULTS, checked.

    Raises ValueError for an unknown key, a resolvent_step that is not a positive finite
    number, inner_steps below 1, policy_draws below 2, a negative actor_steps (integers
    all three) or a temperature_every_rows that is not positive. Returns floats for
    resolvent_step and temperature_every_rows and integers for the rest.
    """
    given = config.get(CT_SAC_IMPLICIT_KEY)
    if given is None:
        given = {}
    if not isinstance(given, Mapping):
        raise ValueError(f"{CT_SAC_IMPLICIT_KEY} must be an object, not {given!r}.")
    unknown = sorted(set(given) - set(CT_SAC_IMPLICIT_DEFAULTS))
    if unknown:
        raise ValueError(f"{CT_SAC_IMPLICIT_KEY} has unknown settings: {unknown}.")
    setting = {**CT_SAC_IMPLICIT_DEFAULTS, **given}

    def number(name: str) -> float:
        value = setting[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{CT_SAC_IMPLICIT_KEY} {name} must be positive, not {value!r}.")
        return float(value)

    def integer(name: str, minimum: int) -> int:
        value = setting[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(
                f"{CT_SAC_IMPLICIT_KEY} {name} must be an integer of at least {minimum}, "
                f"not {value!r}."
            )
        return int(value)

    return {
        "resolvent_step": number("resolvent_step"),
        "inner_steps": integer("inner_steps", 1),
        "policy_draws": integer("policy_draws", 2),
        "actor_steps": integer("actor_steps", 0),
        "temperature_every_rows": number("temperature_every_rows"),
    }


# Values of the protocol key CT_SAC_REWARD_UNITS_KEY (since 2026-11-18;
# evaluations/ph_sac/protocol.OptionalProtocolFields, a JSON config's "protocol" section or
# benchmarks/run_ph_sac.py --ct-sac-reward-units; no table column): the units of the reward
# and the temperature that the legacy learner of every CT-SAC arm (algorithm "ct_sac":
# ct_sac_torque and the released, feature and implicit CT-SAC arms) sees
# (algorithms/ph_baseline.TorqueCTSACBaseline reward_units). With T the reference period
# (s), u = Delta / T, r the environment's discounted integral over the interval of a reward
# rate per second and c the failure charge (value units) of a terminated interval:
#   "second"            reward (r - c) / u + c, about T times the rate; temperature
#                       alpha = alpha_c T for the per-second alpha_c. Every run before the key.
#   "reference_period"  every value the learner sees divided by T: reward
#                       ((r - c) / u + c) / T, about the rate itself (a rate of 1 per second is
#                       a reward of 1 per reference period, CT-SAC's own scale on dm_control,
#                       whose rewards are at most 1 per control step), the failure charge c / T
#                       with it; temperature alpha = alpha_c T / T = alpha_c, so
#                       initial_alpha_c 1 is CT-SAC's default alpha 1.
# The recorded per-second alpha_c is the CT-PH-SAC arms' in both (policy_diagnostics). No other
# arm reads the key. A protocol without it reads DEFAULT_CT_SAC_REWARD_UNITS; it enters a
# run's resume fingerprint only for a CT-SAC arm and only off the default
# (evaluations/ph_sac/resume._resolved_settings), so that every other fingerprint is that of
# the code before it.
CT_SAC_REWARD_UNITS_KEY = "ct_sac_reward_units"
CT_SAC_REWARD_UNITS = ("second", "reference_period")
DEFAULT_CT_SAC_REWARD_UNITS = "second"


def check_ct_sac_reward_units(units: object) -> str:
    """Return units if it is one of CT_SAC_REWARD_UNITS; raise ValueError otherwise."""
    if not isinstance(units, str) or units not in CT_SAC_REWARD_UNITS:
        raise ValueError(
            f"{CT_SAC_REWARD_UNITS_KEY} must be one of {CT_SAC_REWARD_UNITS}, not {units!r}."
        )
    return units


def ct_sac_reward_units_setting(protocol: Mapping[str, object]) -> str:
    """Read protocol[CT_SAC_REWARD_UNITS_KEY], DEFAULT_CT_SAC_REWARD_UNITS when absent;
    check it."""
    return check_ct_sac_reward_units(
        protocol.get(CT_SAC_REWARD_UNITS_KEY, DEFAULT_CT_SAC_REWARD_UNITS)
    )


# Values of the protocol key CT_SAC_REWARD_SOURCE_KEY (since 2026-11-22;
# evaluations/ph_sac/protocol.OptionalProtocolFields, a JSON config's "protocol" section or
# benchmarks/run_ph_sac.py --ct-sac-reward-source; no table column): what the interval part
# of the reward that the legacy learner of every CT-SAC arm sees is
# (algorithms/ph_baseline.TorqueCTSACBaseline reward_source, _adapt_transition). With T, u,
# r and c as for CT_SAC_REWARD_UNITS and rho(s, a) the task's reward rate per second:
#   "interval"   (r - c) / u, the environment's discounted integral of the rate over the
#                interval per reference period: about T times the rate averaged over the
#                interval. Every run before the key.
#   "end_state"  T rho(s', a): the rate at the state s' that ends the interval, with the
#                torque a held over the interval (the environment's own reward_rate and
#                torque, those of its last trapezoid knot), times T and not scaled by the
#                interval's duration, as CT-SAC's own wrapper gives the task's reward at the
#                state after the interval's physics steps (environment/dmc.py _step_physics).
# The failure charge c of a terminated interval is added to either, and ct_sac_reward_units
# "reference_period" divides the sum by T as before, so "end_state" under it is rho(s', a)
# itself in an interval without failure. No other arm reads the key. A protocol without it
# reads DEFAULT_CT_SAC_REWARD_SOURCE; it enters a run's resume fingerprint only for a CT-SAC
# arm and only off the default (evaluations/ph_sac/resume._resolved_settings), so that every
# other fingerprint is that of the code before it.
CT_SAC_REWARD_SOURCE_KEY = "ct_sac_reward_source"
CT_SAC_REWARD_SOURCES = ("interval", "end_state")
DEFAULT_CT_SAC_REWARD_SOURCE = "interval"


def check_ct_sac_reward_source(source: object) -> str:
    """Return source if it is one of CT_SAC_REWARD_SOURCES; raise ValueError otherwise."""
    if not isinstance(source, str) or source not in CT_SAC_REWARD_SOURCES:
        raise ValueError(
            f"{CT_SAC_REWARD_SOURCE_KEY} must be one of {CT_SAC_REWARD_SOURCES}, "
            f"not {source!r}."
        )
    return source


def ct_sac_reward_source_setting(protocol: Mapping[str, object]) -> str:
    """Read protocol[CT_SAC_REWARD_SOURCE_KEY], DEFAULT_CT_SAC_REWARD_SOURCE when absent;
    check it."""
    return check_ct_sac_reward_source(
        protocol.get(CT_SAC_REWARD_SOURCE_KEY, DEFAULT_CT_SAC_REWARD_SOURCE)
    )


# Values of the protocol key REWARD_OBSERVATION_KEY (since 2026-11-22;
# evaluations/ph_sac/protocol.OptionalProtocolFields, a JSON config's "protocol" section or
# benchmarks/run_ph_sac.py --reward-observation; no table column): which reward datum of a
# transition every learner may use. With rho(s, u) the task's reward rate per second, s' the
# state that ends the interval (the first failing knot of a terminated one), u_e the torque
# at s' (the held torque in torque mode, the feedback law at s' resolved at s otherwise) and
# c = d e^(-lambda Delta) F(s') the failure charge of a terminated row (F the task's payoff):
#   "interval"   the environment's discounted integral of the rate over the interval, with
#                c (Transition reward). Every run before the key.
#   "end_state"  rho_end = rho(s', u_e) alone (environment/ph_sac/base.MujocoPHEnv.
#                end_reward_rate). The interval reward is still stored in the replay but no
#                learner reads it: replay batches carry end_rewards in place of rewards
#                (common/ph_replay.py). The CT-SAC arms take ct_sac_reward_source "end_state"
#                (effective_ct_sac_reward_source; an explicit "interval" contradicts the key
#                and raises). Every CT-PH-SAC arm fits a reward-rate network r_hat to rho_end
#                at (s', u_e) on every row (models/ph_sac/reward.end_state_reward_loss) and
#                integrates it over the interval wherever a target read the interval reward
#                (algorithms/ph_targets.py): the generator target Delta r_hat(s, u_0) + c, the
#                rollout target the survivor mean of each model path's discounted trapezoids
#                of r_hat on its own knots + c, the model-free targets (data rate target,
#                contact exclusion's flagged rows, the gate's observed return) the observed
#                ends' trapezoid Delta/2 [r_hat(s, u_0) + e^(-lambda Delta) r_hat(s', u_e)] + c.
#                The implicit arms need reward_source "learned" (and implicit_critic then
#                learns one); the explicit arms build the network with a Polyak target copy.
# The key is the protocol's only: an arm config carries it (as reward_observation
# "end_state") only off the default, written by the protocol's arm builders, and a
# protocol's explicit_config or implicit_config may not set it. A protocol without it reads
# DEFAULT_REWARD_OBSERVATION; it enters a run's resume fingerprint only off the default (a
# CT-PH-SAC arm's resolved reward_observation, a CT-SAC arm's resolved ct_sac_reward_source,
# evaluations/ph_sac/resume._resolved_settings), so that every other fingerprint is that of
# the code before it.
REWARD_OBSERVATION_KEY = "reward_observation"
REWARD_OBSERVATIONS = ("interval", "end_state")
DEFAULT_REWARD_OBSERVATION = "interval"


def check_reward_observation(observation: object) -> str:
    """Return observation if it is one of REWARD_OBSERVATIONS; raise ValueError otherwise."""
    if not isinstance(observation, str) or observation not in REWARD_OBSERVATIONS:
        raise ValueError(
            f"{REWARD_OBSERVATION_KEY} must be one of {REWARD_OBSERVATIONS}, "
            f"not {observation!r}."
        )
    return observation


def reward_observation_setting(mapping: Mapping[str, object]) -> str:
    """Read mapping[REWARD_OBSERVATION_KEY] (a protocol or an arm config),
    DEFAULT_REWARD_OBSERVATION when absent; check it."""
    return check_reward_observation(
        mapping.get(REWARD_OBSERVATION_KEY, DEFAULT_REWARD_OBSERVATION)
    )


def effective_ct_sac_reward_source(protocol: Mapping[str, object]) -> str:
    """The reward source the CT-SAC arms run with: ct_sac_reward_source_setting, and under
    reward_observation "end_state" always "end_state".

    Raises ValueError under "end_state" when ct_sac_reward_source is explicitly "interval"
    (the interval reward is then no learner's datum), and for invalid values of either key.
    """
    source = ct_sac_reward_source_setting(protocol)
    if reward_observation_setting(protocol) != "end_state":
        return source
    # Absent, the source follows the key; given, it must agree with it.
    if CT_SAC_REWARD_SOURCE_KEY in protocol and source != "end_state":
        raise ValueError(
            f"{REWARD_OBSERVATION_KEY} 'end_state' gives the CT-SAC arms the end-state rate; "
            f"{CT_SAC_REWARD_SOURCE_KEY} {source!r} contradicts it (remove it or set "
            "'end_state')."
        )
    return "end_state"


def plant_step_schedule(training: Mapping[str, object]) -> Tuple[int, int]:
    """(dense_until, every) of a training config, PLANT_STEP_DEFAULTS for absent keys.

    Raises ValueError unless dense_until is a nonnegative and every a positive integer.
    """
    values = []
    for name, low in (("plant_step_dense_until", 0), ("plant_step_every", 1)):
        value = training.get(name, PLANT_STEP_DEFAULTS[name])
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            kind = "nonnegative" if low == 0 else "positive"
            raise ValueError(f"training {name} must be a {kind} integer, not {value!r}.")
        values.append(int(value))
    return values[0], values[1]


def plant_step_is_dense(training: Mapping[str, object]) -> bool:
    """Whether the schedule takes the plant step on every row (plant_step_every 1)."""
    return plant_step_schedule(training)[1] == 1


def check_model_integrator(integrator: object) -> str:
    """Return integrator if it is one of MODEL_INTEGRATORS; raise ValueError otherwise."""
    if not isinstance(integrator, str) or integrator not in MODEL_INTEGRATORS:
        raise ValueError(
            f"model_integrator must be one of {MODEL_INTEGRATORS}, not {integrator!r}."
        )
    return integrator


def model_integrator_setting(config: Mapping[str, object]) -> str:
    """Read config["model_integrator"], DEFAULT_MODEL_INTEGRATOR when absent; check it."""
    return check_model_integrator(
        config.get("model_integrator", DEFAULT_MODEL_INTEGRATOR)
    )


# Contact-triggered fine substepping (top-level config key "model_coarse_step", both
# CT-PH-SAC arms; table column model_coarse_step; since 2026-11-10). On a task with joint
# limits (StateSpec limited_position_ranges) the model step model_step must be the physics
# step wherever an interval touches a limit (the limit's damping switches on with a jump
# that a longer RK4 step straddles), but limit-free intervals pass the accuracy criterion
# at a coarser step. With model_coarse_step set, every model integration of a decision
# interval (plant fit, its propagated covariance, rate-head and explicit rollouts, gate
# paths, path Jacobian norms) first integrates every row at model_coarse_step; a row whose
# coarse path comes within a margin of a range edge (or past it) at any knot is integrated
# again over the whole interval at model_step, and that result replaces the coarse one
# (models/ph_sac/rollouts.ContactSubstepping, contact_flags: the detector and the
# assumption under which it cannot miss a contact). The margin of limited coordinate i
# over a coarse step of length h_c is
#     mu_i = CONTACT_MARGIN_FRACTION * (high_i - low_i) + CONTACT_VELOCITY_FACTOR * h_c
#            * max(|qdot_i| at the step's start, |qdot_i| at its end).
# Absent or None: off, every row integrates at model_step (every run before the key, and
# every row without a MODEL_COARSE_STEPS entry). Under refresh_mode "held" the scheme is
# inactive (the grid splits at every refresh anyway). Requirements (validate_configuration):
# a whole multiple of model_step above it, a whole number of physics steps, a fixed clock
# whose interval and episode it divides, and limited_position_ranges in the state spec;
# under contact_handling "exclude" (where it is only the step of every integration) no
# fixed clock and no divisibility (irregular clocks, since 2026-11-11).
CONTACT_MARGIN_FRACTION = 0.01
CONTACT_VELOCITY_FACTOR = 2.0


def model_coarse_step_setting(config: Mapping[str, object]) -> Optional[float]:
    """Read config["model_coarse_step"]: None when absent or None, else a positive float.

    Raises ValueError for anything else (a bool, a string, zero, a negative or non-finite
    number).
    """
    value = config.get("model_coarse_step")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"model_coarse_step must be a positive number or null, not {value!r}.")
    return float(value)


def contact_substepping_setting(config: Mapping[str, object]) -> Optional[Mapping[str, float]]:
    """The contact substepping a CT-PH-SAC config runs, as run records store it.

    None when it is off: no model_coarse_step, or a controller under refresh_mode "held"
    (where the scheme is inactive). Otherwise {"coarse_step", "fine_step" (model_step),
    "margin_fraction", "velocity_factor"} (CONTACT_MARGIN_FRACTION,
    CONTACT_VELOCITY_FACTOR). Records written before the key existed load as None
    (evaluations/ph_sac/reporting.record_contact_substepping).
    """
    coarse = model_coarse_step_setting(config)
    if coarse is None:
        return None
    if refresh_mode_setting(config["testbed"]["controller"]) != "continuous":
        return None
    if contact_handling_setting(config) == "exclude":
        # Contact exclusion integrates every model path at the coarse step and never
        # re-integrates (exclusion_model_step): no substepping.
        return None
    return {
        "coarse_step": coarse,
        "fine_step": float(config["model_step"]),
        "margin_fraction": float(CONTACT_MARGIN_FRACTION),
        "velocity_factor": float(CONTACT_VELOCITY_FACTOR),
    }


def _check_contact_substepping(config: Mapping[str, object]) -> None:
    """validate_configuration's checks of model_coarse_step (see the block above)."""
    coarse = model_coarse_step_setting(config)
    if coarse is None:
        return
    testbed = config["testbed"]
    fine = float(config["model_step"])
    ratio = coarse / fine
    if ratio < 2.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-8:
        raise ValueError(
            f"model_coarse_step {coarse} must be a whole multiple (at least 2) of "
            f"model_step {fine}."
        )
    _whole_steps("model_coarse_step", coarse, testbed["physics_dt"])
    if not testbed["state_spec"].get("limited_position_ranges"):
        raise ValueError(
            "model_coarse_step needs a state spec with limited_position_ranges: the "
            "contact detector watches those coordinates."
        )
    if contact_handling_setting(config) == "exclude":
        # Under contact exclusion the coarse step is only the step of every model
        # integration (exclusion_model_step, no fine re-integration): the continuous grid
        # (models/ph_sac/rollouts.refresh_aligned_grid) splits any interval of whole
        # physics steps into equal substeps of at most it, so neither a fixed clock nor an
        # interval or episode it divides is needed (irregular clocks, since 2026-11-11).
        return
    clock = testbed["clock"]
    if clock["kind"] != "fixed":
        raise ValueError("model_coarse_step needs a fixed clock (one decision interval).")
    for name, value in (
        ("the decision interval", clock["min_duration"]),
        ("the episode duration", testbed["episode_duration"]),
    ):
        pieces = value / coarse
        if abs(pieces - round(pieces)) > 1e-8:
            raise ValueError(f"model_coarse_step {coarse} must divide {name} {value}.")


# Contact exclusion (top-level config key "contact_handling", both CT-PH-SAC arms; table
# column model_contact_handling; since 2026-11-10). On a task with joint limits (StateSpec
# limited_position_ranges), "port" (the default, every earlier run) fits the limits with the
# learned plant's limit port (network limit_port). MuJoCo's soft limit adds a spring-damper
# AND cancels about 90-95% of every other force on the joint (the constraint's added
# inertia), which no potential and damping represent: held-out one-interval endpoint errors
# of the port were 17-34% on intervals touching a limit (under 1% on free ones). "exclude"
# leaves the plant the free mechanism (no limit port; validation requires network
# limit_port false) and keeps it away from the limits:
#   * a replay transition is flagged when, at its START state, a limited coordinate i is
#     within the reach margin
#         mu_i = EXCLUSION_MARGIN_FRACTION (high_i - low_i) + x_i(v, Delta),
#         v = EXCLUSION_VELOCITY_FACTOR |qdot_i|, z = k_i Delta,
#         x_i = max(v Delta, v Delta phi1(z) + a_i Delta^2 phi2(z)),
#         phi1(z) = (1 - e^-z) / z, phi2(z) = (z - 1 + e^-z) / z^2 (1 and 1/2 at z = 0)
#     of an edge of its range or past it (a_i = EXCLUSION_ACCELERATION_FACTOR times the
#     acceleration the motors alone give coordinate i at the model's reference pose,
#     environment.ph_sac.base.MujocoPHEnv.motor_acceleration_bounds; k_i = d_i (M^-1)_ii
#     its damping rate there, MujocoPHEnv.velocity_decay_rates), or when the environment
#     recorded that the real interval was past a range at a physics knot (Transition
#     limit_touched); the flag is stored in the replay. x_i bounds the travel of the
#     damped double integrator qdd = s a_i - k_i qd, |s| <= 1, from speed v over Delta
#     (models/ph_sac/rollouts.start_state_margins); without damping (k_i = 0) it is
#     v Delta + a_i Delta^2 / 2, the rule before EXCLUSION_RULE_VERSION 2;
#   * flagged rows get no weight in the plant fit (warm fit included) and the plant
#     diagnostics; the reward-head fit keeps every row;
#   * every use of the plant at a flagged state falls back to the observed transition
#     (the value residual, the generator and rollout targets, the rate head's targets);
#     a model path that comes near a limit (models/ph_sac/rollouts.contact_flags with
#     margin fraction EXCLUSION_MARGIN_FRACTION and EXCLUSION_PATH_VELOCITY_FACTOR) is
#     truncated at its last clean knot;
#   * every model integration runs at exclusion_model_step: model_coarse_step when the
#     task has one (no fine re-integration: the plant has no limit force), else model_step.
# models/ph_sac/rollouts.ContactExclusion holds the rule; the docstrings of the agents'
# fallbacks state the formulas.
CONTACT_HANDLINGS = ("port", "exclude")
DEFAULT_CONTACT_HANDLING = "port"
EXCLUSION_MARGIN_FRACTION = CONTACT_MARGIN_FRACTION
EXCLUSION_VELOCITY_FACTOR = 1.5
EXCLUSION_ACCELERATION_FACTOR = 2.0
EXCLUSION_PATH_VELOCITY_FACTOR = CONTACT_VELOCITY_FACTOR
# The start-state rule's version, in run records and the resume fingerprint of every
# "exclude" run (contact_exclusion_setting), so that a run started under another rule
# (other rows flagged) refuses to resume. 1 (2026-11-10, absent from the records): the
# undamped allowance a_i Delta^2 / 2; 2 (2026-11-10 evening): the damped reach bound.
EXCLUSION_RULE_VERSION = 2


def contact_handling_setting(config: Mapping[str, object]) -> str:
    """config["contact_handling"], DEFAULT_CONTACT_HANDLING ("port") when absent; checked."""
    value = config.get("contact_handling", DEFAULT_CONTACT_HANDLING)
    if not isinstance(value, str) or value not in CONTACT_HANDLINGS:
        raise ValueError(f"contact_handling must be one of {CONTACT_HANDLINGS}, not {value!r}.")
    return value


def exclusion_model_step(config: Mapping[str, object]) -> float:
    """The step of every model integration: under contact_handling "exclude" the task's
    model_coarse_step when it has one, else (and under "port") model_step."""
    if contact_handling_setting(config) == "exclude":
        coarse = model_coarse_step_setting(config)
        if coarse is not None:
            return float(coarse)
    return float(config["model_step"])


def contact_exclusion_setting(config: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """The contact exclusion a CT-PH-SAC config runs, as run records store it; None when
    contact_handling is "port" (absent: every earlier run)."""
    if contact_handling_setting(config) != "exclude":
        return None
    return {
        "mode": "exclude",
        "model_step": exclusion_model_step(config),
        "margin_fraction": float(EXCLUSION_MARGIN_FRACTION),
        "velocity_factor": float(EXCLUSION_VELOCITY_FACTOR),
        "acceleration_factor": float(EXCLUSION_ACCELERATION_FACTOR),
        "path_velocity_factor": float(EXCLUSION_PATH_VELOCITY_FACTOR),
        "rule_version": int(EXCLUSION_RULE_VERSION),
    }


# The value gradient at contact exclusion's flagged rows (implicit config key
# FLAGGED_VALUE_GRADIENT_KEY, a JSON config's "implicit" section; no table column; since
# 2026-11-23). Under contact_handling "exclude" the implicit value residual at a flagged row
# is the observed interval's (algorithms/ph_resolvent.observed_interval_residuals),
#     R_i = (1 + tau/Delta) V(s) - V_k(s) - (tau/Delta) [r + (1-d) e^(-lambda Delta) V(s')]
#           + tau alpha_c (w_lambda(Delta)/Delta) log pi_k(kappa_i | s),
# V the online value at both s and s'. Its one observed s' serves all N draws (only the
# entropy charge differs per draw), so the all-pairs loss squares a one-transition estimate,
# and the gradient through V(s') carries the double-sampling bias: it can lower the loss by
# flattening V between s and s'.
#   "full"            the gradient reaches psi through V(s) and V(s'). Every run before the
#                     key.
#   "stop_successor"  V(s') is the same online value, evaluated as under "full" and then
#                     detached: the residuals' values are unchanged, the gradient reaches psi
#                     through V(s) alone. V(s') is not replaced by the target copy V_k. The
#                     unflagged rows' residuals and gradients are untouched.
# It acts wherever a value step trains on observed_interval_residuals: the implicit arms with
# the default value update (value_step "implicit", value_operator "model"): implicit_torque,
# implicit_critic_head, the impedance-law arm and the implicit-setting ablations, in either
# reward_observation. The explicit-step arms read V_k(s') there (no gradient reaches V(s'))
# and implicit_critic reads no observed interval: the key changes nothing in them, nor in the
# explicit and CT-SAC arms, which never read it. "stop_successor" needs contact_handling
# "exclude" (validate_configuration; without flagged rows it would change nothing). A config
# without the key reads DEFAULT_FLAGGED_VALUE_GRADIENT. The JSON loader writes it into the
# implicit config only off the default (benchmarks/run_ph_sac.py), so that a stated "full"
# is the absent key: the config, records and resume fingerprint of the code before it. Off
# the default it enters every run's fingerprint through the protocol's implicit_config (as
# the implicit config's other optional keys do) and an implicit arm's record as
# flagged_value_gradient (evaluations/ph_sac/protocol.py).
FLAGGED_VALUE_GRADIENT_KEY = "flagged_value_gradient"
FLAGGED_VALUE_GRADIENTS = ("full", "stop_successor")
DEFAULT_FLAGGED_VALUE_GRADIENT = "full"


def flagged_value_gradient_setting(config: Mapping[str, object]) -> str:
    """config[FLAGGED_VALUE_GRADIENT_KEY], DEFAULT_FLAGGED_VALUE_GRADIENT ("full") when
    absent; raise ValueError unless it is one of FLAGGED_VALUE_GRADIENTS."""
    value = config.get(FLAGGED_VALUE_GRADIENT_KEY, DEFAULT_FLAGGED_VALUE_GRADIENT)
    if not isinstance(value, str) or value not in FLAGGED_VALUE_GRADIENTS:
        raise ValueError(
            f"{FLAGGED_VALUE_GRADIENT_KEY} must be one of {FLAGGED_VALUE_GRADIENTS}, "
            f"not {value!r}."
        )
    return value


def _check_contact_handling(config: Mapping[str, object]) -> None:
    """validate_configuration's checks of contact_handling (the block at CONTACT_HANDLINGS)
    and of flagged_value_gradient (the block above)."""
    gradient = flagged_value_gradient_setting(config)
    if contact_handling_setting(config) != "exclude":
        if gradient != DEFAULT_FLAGGED_VALUE_GRADIENT:
            raise ValueError(
                f"{FLAGGED_VALUE_GRADIENT_KEY} {gradient!r} acts at contact exclusion's "
                "flagged rows only: it needs contact_handling 'exclude'."
            )
        return
    if not config["testbed"]["state_spec"].get("limited_position_ranges"):
        raise ValueError(
            "contact_handling exclude needs a state spec with limited_position_ranges."
        )
    if config["network"].get("limit_port", False):
        raise ValueError(
            "contact_handling exclude fits the free mechanism: set network limit_port false."
        )


# Values of the controller key "decision_mode" (ControllerConfig): what a decision is.
#   "feedback"  the impedance law (setpoint, stiffness, damping) per actuator, mapped to a
#               torque from the current state at every controller refresh: the
#               impedance-law ablation arms (implicit_learned_admissions, explicit_rollout,
#               explicit_generator) and the implicit-setting ablations.
#   "torque"    the motor torque itself, held over the interval as by the CT-SAC
#               baseline; the decision box is the torque box: every paper arm
#               (ct_sac_torque, implicit_torque, explicit_rollout_torque,
#               explicit_generator_torque) and the value-step ablation arms.
# A controller without the key (every config and table row) reads DEFAULT_DECISION_MODE.
# The key describes the controller's box, not the arm: configs and table rows carry the
# feedback box (decision_low/high of dimension 3m) and its entropy target, and an arm
# with torque decisions converts its own copy per run (torque_decision_config; the arm's
# decision_mode, evaluations/ph_sac/protocol.ArmSpec). So the default stays "feedback":
# reading the unconverted feedback box as torque decisions would leave a 3m box that
# validation rejects.
DECISION_MODES = ("feedback", "torque")
DEFAULT_DECISION_MODE = "feedback"
# Decision coordinates per actuator in each mode.
DECISION_WIDTHS: Mapping[str, int] = {"feedback": 3, "torque": 1}


def check_decision_mode(mode: object) -> str:
    """Return mode if it is one of DECISION_MODES; raise ValueError otherwise."""
    if not isinstance(mode, str) or mode not in DECISION_MODES:
        raise ValueError(f"decision_mode must be one of {DECISION_MODES}, not {mode!r}.")
    return mode


def decision_mode_setting(controller: Mapping[str, object]) -> str:
    """Read controller["decision_mode"], DEFAULT_DECISION_MODE when absent; check it."""
    return check_decision_mode(controller.get("decision_mode", DEFAULT_DECISION_MODE))


def decision_dim(controller: Mapping[str, object], action_dim: int) -> int:
    """The decision dimension: 3m feedback coordinates or m torques, m = action_dim."""
    return DECISION_WIDTHS[decision_mode_setting(controller)] * int(action_dim)


# Values of the controller key "refresh_mode" (ControllerConfig): when a feedback law
# turns the state into a torque.
#   "continuous"  at every stage of every integrator step from that stage's state: the
#                 environment evaluates the law in MuJoCo's control callback
#                 (mujoco.set_mjcb_control) at each RK4 stage of each 1 ms physics step,
#                 and the model's RK4 at each of its stages, so the torque follows the
#                 state continuously and a model step above the physics step is a
#                 consistent discretization of the same closed loop. refresh_steps is
#                 unused.
#   "held"        sample and hold: the torque is computed from the state every
#                 refresh_steps physics steps and held through each physics step and
#                 its RK4 stages, and the model's substep grid splits at every refresh.
#                 Every run before 2026-11-09 used it; it reproduces them bitwise.
# A torque decision (decision_mode "torque") is held over its interval in either mode;
# "continuous" only drops the refresh points from its model grid.
# A controller without the key reads DEFAULT_REFRESH_MODE; a run record without
# refresh_mode (written before 2026-11-09) loads as "held"
# (evaluations/ph_sac/reporting.record_refresh_mode).
REFRESH_MODES = ("continuous", "held")
DEFAULT_REFRESH_MODE = "continuous"
# The mode of every run record written before the key existed.
LEGACY_REFRESH_MODE = "held"


def check_refresh_mode(mode: object) -> str:
    """Return mode if it is one of REFRESH_MODES; raise ValueError otherwise."""
    if not isinstance(mode, str) or mode not in REFRESH_MODES:
        raise ValueError(f"refresh_mode must be one of {REFRESH_MODES}, not {mode!r}.")
    return mode


def refresh_mode_setting(controller: Mapping[str, object]) -> str:
    """Read controller["refresh_mode"], DEFAULT_REFRESH_MODE when absent; check it."""
    return check_refresh_mode(controller.get("refresh_mode", DEFAULT_REFRESH_MODE))


def torque_decision_controller(controller: ControllerConfig) -> ControllerConfig:
    """Copy controller in torque-decision mode: the decision box becomes the torque box.

    decision_low/high are float64 copies of torque_low/high and decision_mode is
    "torque"; the motor limits, setpoint settings, refresh_steps and refresh_mode are
    kept. A controller already in torque mode is copied unchanged.
    """
    converted = dict(controller)
    if decision_mode_setting(controller) == "torque":
        return converted  # type: ignore[return-value]
    converted["decision_low"] = np.array(controller["torque_low"], dtype=np.float64).reshape(-1)
    converted["decision_high"] = np.array(
        controller["torque_high"], dtype=np.float64
    ).reshape(-1)
    converted["decision_mode"] = "torque"
    return converted  # type: ignore[return-value]


def torque_decision_testbed(testbed: TestbedConfig) -> TestbedConfig:
    """Shallow copy of testbed whose controller is torque_decision_controller's.

    Every other entry is the same object; the testbed's plant, reward, clock and
    failure set do not depend on the decision mode.
    """
    return {**testbed, "controller": torque_decision_controller(testbed["controller"])}


def torque_decision_config(
    config: ExplicitConfig | ImplicitConfig,
) -> ExplicitConfig | ImplicitConfig:
    """The arm config with torque decisions: torque-box testbed and rescaled entropy target.

    The testbed is torque_decision_testbed's, and temperature["target_entropy"] keeps the
    config's entropy per decision coordinate on the m torque coordinates: the feedback
    target (decision_box_entropy_target, -1 per coordinate, -3m, in every protocol file
    and table row) times m / 3m, so -m. Plant, reward, value, clock and every other
    setting are unchanged. A config already in torque mode is returned as a copy. The
    result shares its sub-dicts with config except the testbed's controller and the
    temperature; the protocol's arm builders pass a deep copy.
    """
    testbed = config["testbed"]
    if decision_mode_setting(testbed["controller"]) == "torque":
        return dict(config)  # type: ignore[return-value]
    action_dim = int(testbed["state_spec"]["action_dim"])
    feedback_dim = decision_dim(testbed["controller"], action_dim)
    temperature = dict(config["temperature"])
    per_coordinate = float(temperature["target_entropy"]) / feedback_dim
    converted = torque_decision_testbed(testbed)
    temperature["target_entropy"] = decision_box_entropy_target(
        converted["controller"], entropy_per_coordinate=per_coordinate
    )
    converted_config = {**config, "testbed": converted, "temperature": temperature}
    return converted_config  # type: ignore[return-value]


def decision_box_entropy_target(
    controller: ControllerConfig, *, entropy_per_coordinate: float = -1.0
) -> float:
    """Entropy target d*H1 of the policy density over the normalized decision box.

    DecisionActor's density is over u = (kappa - centre) / h in [-1,1]^d, d the decision
    dimension (3m feedback coordinates, or m torques in torque-decision mode), so the
    target is entropy_per_coordinate per decision dimension, -1 by default as in SAC:
    -3 for one actuator's feedback law, -6 for two, -m for m torques, whatever the box's
    physical units.
    """
    low = np.asarray(controller["decision_low"], dtype=np.float64).reshape(-1)
    return float(entropy_per_coordinate * low.size)


def windy_pendulum_config(
    *,
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    tail_probability: float,
    tail_split: float,
    clock_kind: Literal["two_tail_uniform", "two_tail_beta"],
    beta_shape: Optional[float] = None,
    task: Literal["balance", "swingup"] = "balance",
    torque_limit: Optional[float] = None,
) -> TestbedConfig:
    """Declare the paper preset; require its unspecified reward/reset/clock details.

    task "balance" is the paper's task. task "swingup" is the swing-up of
    ../ct-rl-cpht-sac/docs/windy_pendulum_swingup.md: it starts hanging
    (reset_parameters start_angle pi, which it requires), the setpoint ranges over
    +-pi, the motor is limited to torque_limit, sigma_w is 1 and the angle never fails;
    the speed limit 4*pi stays. torque_limit is required for swing-up and must be
    absent (None) for balance, whose motor is the paper's 12 N m.
    """
    if task not in WINDY_TASKS:
        raise ValueError(f"Unknown windy pendulum task {task!r}.")
    if task == "balance" and torque_limit is not None:
        raise ValueError("The balance task fixes the paper's 12 N m motor; omit torque_limit.")
    if task == "swingup":
        if torque_limit is None or not float(torque_limit) > 0.0:
            raise ValueError("The swing-up task needs a positive torque_limit.")
        start = reset_parameters.get("start_angle")
        if start is None or abs(float(start) - math.pi) > 1e-9:
            raise ValueError("The swing-up task starts hanging: set start_angle to pi.")
    physics_dt = 0.001
    reference_period = 0.01
    state_spec: StateSpec = {
        "position_dim": 1,
        "velocity_dim": 1,
        "action_dim": 1,
        "periodic_position_indices": (0,),
        "actuated_position_indices": (0,),
        "observation_projection": np.eye(1),
    }
    clock: ClockConfig = {
        "kind": clock_kind,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "min_duration": 0.002,
        "max_duration": 0.018,
        "tail_probability": tail_probability,
        "tail_split": tail_split,
        "beta_shape": beta_shape,
    }
    controller: ControllerConfig = {
        "decision_low": np.array([-0.5, 0.0, 0.0]),
        "decision_high": np.array([0.5, 100.0, 20.0]),
        "torque_low": np.array([-12.0]),
        "torque_high": np.array([12.0]),
        "setpoint_mode": "absolute",
        "setpoint_error": "wrapped",
        "refresh_steps": 1,
        "refresh_mode": DEFAULT_REFRESH_MODE,
    }
    plant_parameters = {
        "mass": 1.0,
        "length": 1.0,
        "gravity": 9.81,
        "damping": 0.0,
        "wind_intensity": 3.0,
    }
    failure_limits = {"angle": math.pi / 2.0, "velocity": 4.0 * math.pi}
    if task == "swingup":
        limit = float(torque_limit)
        controller["decision_low"] = np.array([-math.pi, 0.0, 0.0])
        controller["decision_high"] = np.array([math.pi, 100.0, 20.0])
        controller["torque_low"] = np.array([-limit])
        controller["torque_high"] = np.array([limit])
        plant_parameters["wind_intensity"] = 1.0
        failure_limits["angle"] = math.inf
    return {
        "name": "windy_pendulum",
        "state_spec": state_spec,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "episode_duration": 10.0,
        "discount_rate": 0.1,
        "clock": clock,
        "controller": controller,
        "integrator": "Euler",
        "plant_parameters": plant_parameters,
        "failure_limits": failure_limits,
        "reward_parameters": reward_parameters,
        "reset_parameters": reset_parameters,
    }


def acrobot_xk_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare the MuJoCo Xin--Kaneda preset with explicit geometry and task details.

    The shoulder observation is wrapped and the elbow observation stays unwrapped for
    its travel limit; both are sin/cos network features and wrapped residuals, which
    is harmless for the elbow because endpoint residuals are far smaller than pi. The
    elbow is also a network feature divided by its travel limit, because the r1 reward
    reads the unwrapped elbow: one full turn costs about as much as hanging at rest.
    """
    physics_dt = 0.001
    reference_period = 0.01
    state_spec: StateSpec = {
        "position_dim": 2,
        "velocity_dim": 2,
        "action_dim": 1,
        "periodic_position_indices": (0, 1),
        "actuated_position_indices": (1,),
        "observation_projection": np.eye(2),
        "travel_position_scales": ((1, float(failure_limits["elbow_angle"])),),
    }
    clock: ClockConfig = {
        "kind": "fixed",
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "min_duration": reference_period,
        "max_duration": reference_period,
        "tail_probability": None,
        "tail_split": None,
        "beta_shape": None,
    }
    controller: ControllerConfig = {
        "decision_low": np.array([-1.0, 0.0, 0.0]),
        "decision_high": np.array([1.0, 100.0, 20.0]),
        "torque_low": np.array([-10.0]),
        "torque_high": np.array([10.0]),
        "setpoint_mode": "offset",
        "setpoint_error": "unwrapped",
        "refresh_steps": 1,
        "refresh_mode": DEFAULT_REFRESH_MODE,
    }
    return {
        "name": "acrobot_xk",
        "state_spec": state_spec,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "episode_duration": 20.0,
        "discount_rate": 0.5,
        "clock": clock,
        "controller": controller,
        "integrator": "RK4",
        "plant_parameters": plant_parameters,
        "failure_limits": failure_limits,
        "reward_parameters": reward_parameters,
        "reset_parameters": reset_parameters,
    }


def _whole_steps(name: str, value: float, physics_dt: float) -> None:
    ratio = value / physics_dt
    if value <= 0.0 or abs(ratio - round(ratio)) > 1e-8:
        raise ValueError(f"{name}={value} must be a positive multiple of {physics_dt}.")


def _validate_testbed(testbed: TestbedConfig) -> None:
    physics_dt = testbed["physics_dt"]
    clock = testbed["clock"]
    if physics_dt <= 0.0 or testbed["discount_rate"] <= 0.0:
        raise ValueError("physics_dt and discount_rate must be positive.")
    if clock["physics_dt"] != physics_dt:
        raise ValueError("Clock and testbed must share one physics step.")
    if clock["reference_period"] != testbed["reference_period"]:
        raise ValueError("Clock and testbed must share one reference period.")
    for name, value in (
        ("reference_period", testbed["reference_period"]),
        ("episode_duration", testbed["episode_duration"]),
        ("min_duration", clock["min_duration"]),
        ("max_duration", clock["max_duration"]),
    ):
        _whole_steps(name, value, physics_dt)
    if clock["max_duration"] < clock["min_duration"]:
        raise ValueError("Clock max_duration must not lie below min_duration.")
    if clock["kind"] == "fixed":
        if clock["min_duration"] != clock["max_duration"]:
            raise ValueError("A fixed clock needs min_duration == max_duration.")
    else:
        tail_probability = clock["tail_probability"]
        tail_split = clock["tail_split"]
        if tail_probability is None or not 0.0 <= tail_probability <= 1.0:
            raise ValueError("A two-tailed clock needs tail_probability in [0, 1].")
        if tail_split is None or not 0.0 <= tail_split <= 1.0:
            raise ValueError("A two-tailed clock needs tail_split in [0, 1].")
        if clock["kind"] == "two_tail_beta" and not (clock["beta_shape"] or 0.0) > 0.0:
            raise ValueError("A two_tail_beta clock needs a positive beta_shape.")

    controller = testbed["controller"]
    action_dim = testbed["state_spec"]["action_dim"]
    if controller["refresh_steps"] < 1:
        raise ValueError("Controller refresh_steps must be at least one.")
    refresh_mode_setting(controller)
    mode = decision_mode_setting(controller)
    if np.size(controller["decision_low"]) != decision_dim(controller, action_dim):
        if mode == "torque":
            raise ValueError("A torque decision needs one torque per actuator.")
        raise ValueError("A decision needs (setpoint, stiffness, damping) per actuator.")
    if np.any(np.asarray(controller["decision_low"]) >= controller["decision_high"]):
        raise ValueError("The decision box must have positive width.")
    if mode == "torque" and not (
        np.array_equal(
            np.asarray(controller["decision_low"], dtype=np.float64).reshape(-1),
            np.asarray(controller["torque_low"], dtype=np.float64).reshape(-1),
        )
        and np.array_equal(
            np.asarray(controller["decision_high"], dtype=np.float64).reshape(-1),
            np.asarray(controller["torque_high"], dtype=np.float64).reshape(-1),
        )
    ):
        raise ValueError("A torque decision box must be the torque box.")


def validate_configuration(config: ExplicitConfig | ImplicitConfig) -> None:
    """Check units, refresh-aligned (possibly unequal) grids, clocks, and N/L/n minima."""
    _validate_testbed(config["testbed"])
    if config["model_step"] <= 0.0:
        raise ValueError("model_step must be positive.")
    model_integrator_setting(config)
    _check_contact_substepping(config)
    _check_contact_handling(config)
    # The reward datum (absent: "interval", no check fails); the implicit check below.
    observation = reward_observation_setting(config)

    training = config["training"]
    if not 0.0 <= training["validation_fraction"] < 1.0:
        raise ValueError("The held-out fraction must be nonnegative and leave training rows.")
    if training["batch_size"] < 1 or training["replay_capacity"] < training["batch_size"]:
        raise ValueError("replay_capacity must hold at least one batch.")
    plant_step_schedule(training)
    plant_diagnostics_setting(training)
    warmup_decisions_setting(training)
    if config["temperature"]["initial_alpha_c"] < 0.0:
        raise ValueError("The per-second temperature must be nonnegative.")
    diffusion = config["network"].get("diffusion", DEFAULT_DIFFUSION)
    if diffusion not in ("learned", "none"):
        raise ValueError(f"Unknown network diffusion {diffusion!r}.")
    for key in ("damping_velocity", "limit_port"):
        if not isinstance(config["network"].get(key, False), bool):
            raise ValueError(f"network {key} must be True or False.")
    # The column-less network keys (absent: no check fails); the known-plant arms' refusal
    # of input_matrix_start "random" and "zero" is the agents' (their plant_source is set
    # per arm).
    actor_log_std_init_setting(config["network"])
    input_matrix_start_setting(config["network"])

    if "solver" not in config:
        if not 0.0 < config["polyak_rate"] <= 1.0:
            raise ValueError("polyak_rate must lie in (0, 1].")
        if config["rollout_paths"] < 1:
            raise ValueError("The rollout target needs at least one path.")
        # The implicit CT-SAC arms' optional section (absent: the defaults, no check fails).
        ct_sac_implicit_setting(config)
        return

    solver = config["solver"]
    if solver["resolvent_step"] <= 0.0:
        raise ValueError("resolvent_step must be positive.")
    if solver["policy_draws"] < 2:
        raise ValueError("The residual needs N >= 2 policy draws per state.")
    if solver["inner_steps"] < 1:
        raise ValueError("A solve needs inner_steps K >= 1.")

    rate = config["rate"]
    if rate["rollout_paths"] < 1 or rate["rate_steps"] < 0 or rate["actor_steps"] < 0:
        raise ValueError("Rate rollouts and step counts are out of range.")

    admission = config["admission"]
    if admission["efolds"] <= 0.0:
        raise ValueError("Admission e-folds must be positive.")
    if admission["initial_paths"] < 2:
        raise ValueError("The gate correction needs L >= 2 paths per model.")
    if admission["initial_rows"] < 2:
        raise ValueError("The gate needs n >= 2 validation rows.")
    if admission["pilot_batches"] < 2:
        raise ValueError("The pilot variance needs B >= 2 batches per row.")
    if admission["allocation"] not in ("pilot", "fixed"):
        raise ValueError(f"Unknown gate allocation {admission['allocation']!r}.")
    if admission["inference"] == "confidence_budgeted":
        if not 0.0 < admission["gate_failure_budget"] < 1.0:
            raise ValueError("gate_failure_budget must lie in (0, 1).")
        if admission["return_bound"] <= 0.0 or admission["min_horizon"] <= 0.0:
            raise ValueError("return_bound and min_horizon must be positive.")
    elif admission["inference"] != "empirical":
        raise ValueError(f"Unknown gate inference {admission['inference']!r}.")
    if admission["plant_update"] not in ("always", "admission", "polyak"):
        raise ValueError(f"Unknown plant_update {admission['plant_update']!r}.")
    if admission["plant_update"] == "polyak" and not (
        admission["polyak_rate"] and 0.0 < admission["polyak_rate"] <= 1.0
    ):
        raise ValueError("The Polyak plant ablation needs polyak_rate in (0, 1].")

    reward_source = implicit_setting(config, "reward_source")
    if reward_source not in ("learned", "known"):
        raise ValueError(f"Unknown reward_source {reward_source!r}.")
    if observation == "end_state" and reward_source != "learned":
        raise ValueError(
            "reward_observation 'end_state' learns the reward rate from end-state rates: "
            "it needs implicit reward_source 'learned'."
        )
    for name in ("reward_warm_steps", "plant_warm_steps"):
        steps = implicit_setting(config, name)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    covariance = implicit_setting(config, "plant_fit_covariance")
    if covariance not in ("leading_order", "propagated"):
        raise ValueError(f"Unknown plant_fit_covariance {covariance!r}.")
    rows = implicit_setting(config, "temperature_every_rows")
    if isinstance(rows, bool) or not isinstance(rows, (int, float)) or not rows > 0:
        raise ValueError("temperature_every_rows must be positive.")
    trainable_plant_parts_setting(config)
    check_step_ablation(config)


def check_step_ablation(config: Mapping[str, object]) -> StepAblationSetting:
    """step_ablation_setting, also refusing value_operator "critic" under plant_update
    "admission": that arm learns no reward rate, and the gate's simulated returns need one."""
    setting = step_ablation_setting(config)
    if (
        setting["value_operator"] == "critic"
        and config["admission"]["plant_update"] == "admission"
    ):
        raise ValueError(
            "value_operator critic learns no reward rate for the admission gate's returns; "
            "use plant_update always or polyak."
        )
    return setting


# The model step h (seconds) of each testbed, by name: the longest substep of every model
# integration of the plant (the plant fit's endpoint mean, the explicit rollout target,
# the rate head's rollouts, the gate paths). benchmarks/make_ct_ph_sac_table.py writes it
# into each row's model_step through default_model_step, and the JSON configs carry the
# same value. The rule: the model step is coarsened only on tasks without joint limits.
# A limit-free task takes the largest step tried (1, 2, 5, 10, 15, 20, 30 ms, the divisors
# of the decision interval among them) at which the oracle plant's model endpoint over one
# decision interval stays within 1e-3 relative error of the environment's, at the 99th
# percentile over 2000 rows with decisions uniform in the box (half of them with stiffness
# and damping in the top 10% of their range), states from 3 s uniform-decision rollouts
# from the reset law; reference: the environment's 1 ms RK4 with the feedback law at every
# stage (refresh_mode "continuous"); measured 2026-11-09 (pendulum 10 ms, acrobot 2 ms,
# lqr 10 ms). A task with joint limits (MuJoCo's soft limits kept: the six cart-poles,
# both reachers, the point mass and both swimmers) integrates at its physics step (1 ms
# RK4; the swimmers' own 2 ms Euler since 2026-11-10, with the semi_implicit map, which is
# that Euler step): with the limit in the oracle, an interval that touches a limit meets
# the 1e-3 criterion only there, since the limit's damping switches on with a jump
# (impedance 0.9 at zero penetration) that a longer RK4 step straddles. Intervals that do
# not touch a limit may instead take MODEL_COARSE_STEPS (contact-triggered fine
# substepping, below): the cart-poles 5 ms, the point mass 10 ms, the reachers none (their
# 1 ms is the RK4 stability bound of the wrist's closed loop, not the limit), the swimmers
# none (the native step). Measured 2026-11-10 (the limits agent,
# scratchpad limits/step/ and limits/step_full/): on rows started at a limit the p99 error
# is about 1e-15 at 1 ms and 0.30 to 0.67 at 2 ms (cart-poles 0.35 to 0.65, point mass
# 0.30, reacher 0.047); the point mass's 20 ms also fails on whole-episode state pools
# (p99 3.0e-3); in short training runs 30% (point mass) to 48% (cart-pole) of intervals
# ended pressed on a limit, and the swimmers' joints sat on a limit 45 to 92% of the
# warm-up time (their 2 ms bound of 2026-11-09 did not separate rows at a limit). The
# reacher needs 1 ms anyway: its RK4 closed loop at the box's damping maximum is unstable
# above 1.1 ms (the wrist). The bounds are pinned by tests/test_ph_model_step.py. Every
# value divides its fixed decision interval, so the grid's substeps are all exactly h. A
# testbed not listed (the windy pendulum: Euler physics, model integrator semi_implicit;
# the in-house Acrobot) integrates at its physics step. Under refresh_mode "held" a step
# above the physics step changes nothing while the controller refreshes every physics
# step: the grid then splits at every refresh. The swimmers (2026-11-10) run dm_control's
# own physics, Euler at 2 ms, and their model step is that physics step: the
# semi_implicit map at h = 2 ms is MuJoCo's Euler step (no dof damping), so the model
# needs one drift evaluation per physics step, 15 per 30 ms interval (120 at 1 ms RK4).
MODEL_STEPS: Mapping[str, float] = {
    "pendulum_swingup": 0.01,
    "acrobot_swingup": 0.002,
    "acrobot_swingup_sparse": 0.002,
    "cartpole_balance": 0.001,
    "cartpole_balance_sparse": 0.001,
    "cartpole_swingup": 0.001,
    "cartpole_swingup_sparse": 0.001,
    "cartpole_two_poles": 0.001,
    "cartpole_three_poles": 0.001,
    "reacher_easy": 0.001,
    "reacher_hard": 0.001,
    "point_mass_easy": 0.001,
    "lqr_2_1": 0.01,
    "lqr_6_2": 0.01,
    "swimmer_swimmer6": 0.002,
    "swimmer_swimmer15": 0.002,
}

# The coarse step of contact-triggered fine substepping (config key model_coarse_step, the
# block at CONTACT_MARGIN_FRACTION) per testbed with joint limits: limit-free intervals
# integrate at this step, intervals that come near a limit at MODEL_STEPS (the physics
# step). benchmarks/make_ct_ph_sac_table.py writes it into each row's model_coarse_step
# (testbed_model_coarse_step; an empty cell for a testbed not listed: off), and a JSON
# config must state the same value. Values: the free-state bound of each task (the
# largest step tried whose p99 one-interval relative endpoint error stays below 1e-3 on
# rows that do not touch a limit; tests/test_ph_model_step.py MEASURED): the cart-poles
# 5 ms (p99 3.2e-4 to 5.0e-4 at 5 ms; 1.1e-3 to 1.7e-3 at 10 ms), the point mass 10 ms (its
# 20 ms failed on whole-episode state pools, p99 3.0e-3, against 2.6e-4 at 10 ms). Not
# listed: the reachers (their 1 ms is the RK4 stability bound of the wrist's closed loop
# under stiff gains, 1.1 ms, not the wrist limit) and the swimmers (their model step is
# the native 2 ms Euler step); every task without joint limits (no detector to run).
MODEL_COARSE_STEPS: Mapping[str, float] = {
    "cartpole_balance": 0.005,
    "cartpole_balance_sparse": 0.005,
    "cartpole_swingup": 0.005,
    "cartpole_swingup_sparse": 0.005,
    "cartpole_two_poles": 0.005,
    "cartpole_three_poles": 0.005,
    "point_mass_easy": 0.01,
}


def testbed_model_step(name: str, physics_dt: float) -> float:
    """The model step h of the testbed called name: MODEL_STEPS, else physics_dt."""
    return float(MODEL_STEPS.get(name, physics_dt))


def testbed_model_coarse_step(name: str) -> Optional[float]:
    """The contact scheme's coarse step of the testbed called name (MODEL_COARSE_STEPS),
    None (off) for a testbed not listed."""
    value = MODEL_COARSE_STEPS.get(name)
    return None if value is None else float(value)


def default_model_step(testbed: TestbedConfig) -> float:
    """The model step h of testbed: MODEL_STEPS by its name, else its physics step."""
    return testbed_model_step(testbed["name"], testbed["physics_dt"])


def default_model_coarse_step(testbed: TestbedConfig) -> Optional[float]:
    """The contact scheme's coarse step of testbed: MODEL_COARSE_STEPS by its name, else
    None (off)."""
    return testbed_model_coarse_step(testbed["name"])
