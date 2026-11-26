"""Building blocks of the dm_control config factories.

The task_* columns of benchmarks/hyperparams/ct_ph_sac.csv fix every dm_control row to
1 ms RK4 physics, a fixed clock at dm_control's control step, the discount rate 0.5 /s,
controller refresh every physics step and the decision-box rule of
benchmarks/make_ct_ph_sac_table.py. These helpers produce exactly those values, so a
factory only states what differs between tasks.

Import discipline: common.ph_config imports this package (it re-exports
cartpole_swingup_config), so modules here import common.ph_config only under
TYPE_CHECKING; at run time they may import common.ph_types, numpy and math. A factory of
a task with joint limits compiles its MuJoCo model to read the limits
(with_limit_port_init); it imports the environment module that builds the model inside
the function, since that module imports this package.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from common.ph_config import ClockConfig, ControllerConfig, TestbedConfig
    from common.ph_types import StateSpec
    from environment.ph_sac.dmc.joint_limits import JointLimits

DMC_PHYSICS_DT = 0.001
DMC_INTEGRATOR = "RK4"
DMC_DISCOUNT_RATE = 0.5
# common.ph_config.DEFAULT_REFRESH_MODE (not imported: see the import discipline above):
# the feedback law is evaluated at every RK4 stage, in the environment and the model.
DMC_REFRESH_MODE = "continuous"
# The decision-box rule: stiffness up to 10 * torque_limit / offset, damping up to 0.2 s
# times that (benchmarks/make_ct_ph_sac_table.py, scaled_box).
STIFFNESS_OVER_LIMIT = 10.0
DAMPING_RATIO = 0.2
# MuJoCo's soft joint-limit parameters as plant_parameters keys, in the order of the
# MJCF attributes solreflimit (timeconst, dampratio) and solimplimit (d0, dmax, width,
# midpoint, power): the keys of every port that keeps a dm_control joint limit (the
# swimmer, and since 2026-11-10 the cart-pole slider, the reacher wrist and the point
# mass's slides; environment/ph_sac/dmc/joint_limits.py).
LIMIT_KEYS: Tuple[str, ...] = (
    "limit_timeconst",
    "limit_dampratio",
    "limit_d0",
    "limit_dmax",
    "limit_width",
    "limit_midpoint",
    "limit_power",
)
# MuJoCo's defaults (solref 0.02 1, solimp 0.9 0.95 0.001 0.5 2), which apply to a limited
# joint whose dm_control XML omits the attributes (the reacher's wrist, the point mass's
# slides, and the cart-pole slider's solimplimit).
MUJOCO_DEFAULT_LIMIT: Mapping[str, float] = {
    "limit_timeconst": 0.02,
    "limit_dampratio": 1.0,
    "limit_d0": 0.9,
    "limit_dmax": 0.95,
    "limit_width": 0.001,
    "limit_midpoint": 0.5,
    "limit_power": 2.0,
}


def fixed_clock(reference_period: float, physics_dt: float = DMC_PHYSICS_DT) -> ClockConfig:
    """A fixed clock holding every decision for the reference period."""
    return {
        "kind": "fixed",
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "min_duration": reference_period,
        "max_duration": reference_period,
        "tail_probability": None,
        "tail_split": None,
        "beta_shape": None,
    }


# Irregular clocks of the dm_control testbeds (since 2026-11-11; the table's rows of mode
# "irregular"). The optional testbed_settings CLOCK_SETTINGS_KEYS of a testbed whose factory
# is wrapped by with_clock_settings: clock_kind (absent, None or "fixed": dm_control's
# control step, every top row) or a two-tailed law, "two_tail_uniform" or "two_tail_beta",
# with tail_probability, tail_split and (beta only) beta_shape, as the windy pendulum's
# testbed_settings. The duration bounds follow one rule, the CT-SAC benchmark's (Nguyen 2026,
# arXiv:2602.14587, App. B.1.3 Table 1; this repository's benchmarks/hyperparams/ct_sac.csv,
# irregular rows of cheetah-run and walker-run): IRREGULAR_MIN_FRACTION T and
# IRREGULAR_MAX_FRACTION T, T the task's dm_control control step (cheetah 2 and 30 ms at
# T = 10 ms, walker 5 and 75 ms at T = 25 ms), rounded to whole physics steps. The reference
# period stays dm_control's control step, CT-SAC's dt_default (environment/dmc.py).
# (2026-11-11 to 11-12 the rule was T/5 to 9T/5, the windy pendulum's law scaled.)
CLOCK_SETTINGS_KEYS: Tuple[str, ...] = ("clock_kind", "tail_probability", "tail_split", "beta_shape")
TWO_TAILED_CLOCKS: Tuple[str, ...] = ("two_tail_uniform", "two_tail_beta")
IRREGULAR_MIN_FRACTION = 0.2
IRREGULAR_MAX_FRACTION = 3.0


def irregular_clock_bounds(
    reference_period: float, physics_dt: float = DMC_PHYSICS_DT
) -> Tuple[float, float]:
    """(min_duration, max_duration): IRREGULAR_*_FRACTION times T, each a whole number of
    physics steps.

    Raises ValueError unless T, T/5 and 3T are whole numbers of physics steps (within 1e-9
    of one) and the minimum is at least one step. CT-SAC's sampler rounds bounds to the
    physics grid (environment/base.py sample_dt_from_distribution); here a bound that would
    need rounding is refused, so the stated law is the one sampled (every final task: T of
    10 or 20 ms at 1 ms physics).
    """
    exact = {
        "T": reference_period / physics_dt,
        "T/5": IRREGULAR_MIN_FRACTION * reference_period / physics_dt,
        "3T": IRREGULAR_MAX_FRACTION * reference_period / physics_dt,
    }
    for name, value in exact.items():
        if abs(value - round(value)) > 1e-9:
            raise ValueError(
                f"An irregular clock about T = {reference_period} needs {name} to be a whole "
                f"number of physics steps of {physics_dt}, not {value:.6g}."
            )
    steps, low, high = (round(exact[name]) for name in ("T", "T/5", "3T"))
    if not 1 <= low < steps < high:
        raise ValueError(
            f"An irregular clock about T = {reference_period} needs more physics steps of "
            f"{physics_dt} per period."
        )
    return round(low * physics_dt, 12), round(high * physics_dt, 12)


def dmc_clock(
    reference_period: float,
    physics_dt: float = DMC_PHYSICS_DT,
    *,
    clock_kind: Optional[str] = None,
    tail_probability: Optional[float] = None,
    tail_split: Optional[float] = None,
    beta_shape: Optional[float] = None,
) -> ClockConfig:
    """fixed_clock without clock_kind (or "fixed"); else the two-tailed law of the
    CLOCK_SETTINGS_KEYS on irregular_clock_bounds (common.ph_config._validate_testbed
    checks the values)."""
    if clock_kind is None or clock_kind == "fixed":
        given = {
            key: value
            for key, value in (
                ("tail_probability", tail_probability),
                ("tail_split", tail_split),
                ("beta_shape", beta_shape),
            )
            if value is not None
        }
        if given:
            raise ValueError(f"A fixed clock takes no {sorted(given)}.")
        return fixed_clock(reference_period, physics_dt)
    if clock_kind not in TWO_TAILED_CLOCKS:
        raise ValueError(f"clock_kind must be fixed or one of {TWO_TAILED_CLOCKS}, not {clock_kind!r}.")
    if tail_probability is None or tail_split is None:
        raise ValueError(f"A {clock_kind} clock needs tail_probability and tail_split.")
    if clock_kind == "two_tail_uniform" and beta_shape is not None:
        raise ValueError("A two_tail_uniform clock takes no beta_shape.")
    low, high = irregular_clock_bounds(reference_period, physics_dt)
    return {
        "kind": clock_kind,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "min_duration": low,
        "max_duration": high,
        "tail_probability": float(tail_probability),
        "tail_split": float(tail_split),
        "beta_shape": None if beta_shape is None else float(beta_shape),
    }


def with_clock_settings(factory: Callable[..., TestbedConfig]) -> Callable[..., TestbedConfig]:
    """factory, also taking the optional CLOCK_SETTINGS_KEYS: its config with the clock
    dmc_clock builds on the config's reference period and physics step (unchanged, the
    fixed clock, without them). Register it with optional_settings_keys
    CLOCK_SETTINGS_KEYS; the module-level factory keeps the fixed clock."""

    @functools.wraps(factory)
    def build(
        *,
        clock_kind: Optional[str] = None,
        tail_probability: Optional[float] = None,
        tail_split: Optional[float] = None,
        beta_shape: Optional[float] = None,
        **settings: object,
    ) -> TestbedConfig:
        config = factory(**settings)
        clock = dmc_clock(
            config["reference_period"],
            config["physics_dt"],
            clock_kind=clock_kind,
            tail_probability=tail_probability,
            tail_split=tail_split,
            beta_shape=beta_shape,
        )
        if clock["kind"] == "fixed":
            return config
        return {**config, "clock": clock}

    return build


# The adaptation experiment's change of the robot (docs/ct_ph_sac_layout.md, "Adaptation
# experiment"): the testbed_settings key PHYSICAL_CHANGE_KEY, {"kind": one of
# PHYSICAL_CHANGE_KINDS, "scale": a positive factor}, on a testbed whose TestbedSpec
# declares physical_change (the task's map from its plant_parameters to the changed ones;
# acrobot_swingup and cartpole_swingup). "length" scales a link's length with its mass
# held (centre of mass and inertia follow the geometry), "damping" every joint's damping.
# with_physical_change adds two TestbedConfig keys and leaves plant_parameters as they
# are: physical_change (the checked change) and physical_plant_parameters (the changed
# parameters). The MuJoCo model, the environment's oracle and the oracle plant read
# physical_plant_parameters (physical_plant_parameters()); the reward, its constants and
# the state spec (the learned plant's initialisation) keep the nominal plant_parameters,
# so the reward stays the task's function of the state and no learner is told the change.
# Neither key is a table column: a config without them is the nominal robot, bitwise.
PHYSICAL_CHANGE_KEY = "physical_change"
PHYSICAL_CHANGE_KINDS: Tuple[str, ...] = ("length", "damping")


def check_physical_change(change: object) -> dict:
    """{"kind": str, "scale": float} of a physical_change setting; raise ValueError unless
    it holds exactly kind (one of PHYSICAL_CHANGE_KINDS) and scale (finite, positive)."""
    if not isinstance(change, Mapping) or set(change) != {"kind", "scale"}:
        raise ValueError(
            f"{PHYSICAL_CHANGE_KEY} needs exactly the keys kind and scale, not {change!r}."
        )
    kind, scale = change["kind"], change["scale"]
    if kind not in PHYSICAL_CHANGE_KINDS:
        raise ValueError(f"{PHYSICAL_CHANGE_KEY} kind must be one of {PHYSICAL_CHANGE_KINDS}, not {kind!r}.")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (
        math.isfinite(float(scale)) and float(scale) > 0.0
    ):
        raise ValueError(f"{PHYSICAL_CHANGE_KEY} scale must be a finite positive number, not {scale!r}.")
    return {"kind": str(kind), "scale": float(scale)}


def with_physical_change(
    config: TestbedConfig,
    change: object,
    plant_map: Callable[[Mapping[str, float], str, float], Mapping[str, float]],
) -> TestbedConfig:
    """config with physical_change (check_physical_change's) and physical_plant_parameters,
    plant_map(plant_parameters, kind, scale); every other entry is the same object."""
    checked = check_physical_change(change)
    changed = dict(plant_map(config["plant_parameters"], checked["kind"], checked["scale"]))
    if set(changed) != set(config["plant_parameters"]):
        raise ValueError(f"{config['name']}: a physical change must keep the plant_parameters' keys.")
    return {**config, PHYSICAL_CHANGE_KEY: checked, "physical_plant_parameters": changed}


def physical_plant_parameters(config: TestbedConfig) -> Mapping[str, float]:
    """The plant parameters MuJoCo simulates: physical_plant_parameters when the config
    carries a physical change, else plant_parameters."""
    return config.get("physical_plant_parameters", config["plant_parameters"])


# A run's discount rate in place of the row's (testbed_settings key DISCOUNT_RATE_KEY, per
# second; DMC_DISCOUNT_RATE in every row) on a testbed whose TestbedSpec sets
# discount_setting (benchmarks/run_ph_sac.py _build_testbed; not a table column).
# with_discount_rate replaces the config's discount_rate after the factory, so it combines
# with the clock settings and lqr's torque_limit. Everything that discounts reads the
# config's discount_rate when it runs (the environment's interval reward and failure
# charge, the CT-SAC baseline's period discount exp(-lambda T) and failure charge, both
# CT-PH-SAC agents, the protocol's matched rate and the evaluation's discounted returns,
# the entropy weights w_lambda of common/ph_time.py, the failure payoff failure_rate /
# lambda), so the override holds everywhere in the run; no factory derives a constant from
# the discount.
DISCOUNT_RATE_KEY = "discount_rate"


def with_discount_rate(config: TestbedConfig, rate: object) -> TestbedConfig:
    """config with discount_rate replaced by rate (per second); every other entry is the
    same object. Raises ValueError unless rate is a finite positive number."""
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not (
        math.isfinite(float(rate)) and float(rate) > 0.0
    ):
        raise ValueError(f"{DISCOUNT_RATE_KEY} must be a finite positive rate per second, not {rate!r}.")
    return {**config, "discount_rate": float(rate)}


def nominal_testbed(config: TestbedConfig) -> TestbedConfig:
    """config without its physical change: the robot the task's reward constants describe
    (a shallow copy; config itself when it carries no change)."""
    if PHYSICAL_CHANGE_KEY not in config and "physical_plant_parameters" not in config:
        return config
    return {
        key: value
        for key, value in config.items()
        if key not in (PHYSICAL_CHANGE_KEY, "physical_plant_parameters")
    }


def box_rule_controller(
    *,
    offsets: Sequence[float],
    torque_limit: float,
    setpoint_mode: Literal["absolute", "offset"],
    setpoint_error: Literal["wrapped", "unwrapped"],
    refresh_steps: int = 1,
    refresh_mode: Literal["continuous", "held"] = DMC_REFRESH_MODE,
) -> ControllerConfig:
    """Decision box by the table's rule, one setpoint offset per actuator, symmetric motors.

    Per actuator: setpoint in [-offset, offset], stiffness in [0, k_max] with
    k_max = 10 * torque_limit / offset, damping in [0, 0.2 * k_max]; torques within
    +-torque_limit. The floats equal the table's task_decision_* cells exactly.
    refresh_mode is the table's task_refresh_mode ("continuous" on every row).
    """
    offsets = [float(offset) for offset in offsets]
    stiffness = [STIFFNESS_OVER_LIMIT * torque_limit / offset for offset in offsets]
    count = len(offsets)
    return {
        "decision_low": np.array([-offset for offset in offsets] + [0.0] * count + [0.0] * count),
        "decision_high": np.array(
            offsets + stiffness + [DAMPING_RATIO * k_max for k_max in stiffness]
        ),
        "torque_low": np.array([-float(torque_limit)] * count),
        "torque_high": np.array([float(torque_limit)] * count),
        "setpoint_mode": setpoint_mode,
        "setpoint_error": setpoint_error,
        "refresh_steps": int(refresh_steps),
        "refresh_mode": refresh_mode,
    }


def dmc_state_spec(
    *,
    physical_dim: int,
    action_dim: int,
    periodic: Sequence[int],
    actuated: Sequence[int],
    context_dim: int = 0,
    travel_position_scales: Optional[Sequence[Tuple[int, float]]] = None,
    limited_position_ranges: Optional[Sequence[Tuple[int, float, float]]] = None,
) -> StateSpec:
    """State (q, context, qd) with every physical position observed: P = [I; 0].

    position_dim = physical_dim + context_dim and velocity_dim = physical_dim; the
    context rows of the observation projection are zero. limited_position_ranges,
    (position index, low, high) triples, names the coordinates the task limits with
    MuJoCo's soft constraint (read by the learned plant's limit port).
    """
    projection = np.vstack((np.eye(physical_dim), np.zeros((context_dim, physical_dim))))
    spec: StateSpec = {
        "position_dim": physical_dim + context_dim,
        "velocity_dim": physical_dim,
        "action_dim": action_dim,
        "periodic_position_indices": tuple(int(index) for index in periodic),
        "actuated_position_indices": tuple(int(index) for index in actuated),
        "observation_projection": projection,
    }
    if context_dim:
        spec["context_dim"] = int(context_dim)
    if travel_position_scales:
        spec["travel_position_scales"] = tuple(
            (int(index), float(scale)) for index, scale in travel_position_scales
        )
    if limited_position_ranges:
        spec["limited_position_ranges"] = tuple(
            (int(index), float(low), float(high)) for index, low, high in limited_position_ranges
        )
    return spec


def with_limit_port_init(config: TestbedConfig, limits: JointLimits) -> TestbedConfig:
    """Add StateSpec limit_port_init to config's state spec (in place) and return config.

    limits are the compiled model's soft limits of the joints that the state spec's
    limited_position_ranges declares, in its order (joint_limits.joint_limits_from_model);
    each joint's dof must be the velocity its entry's position projects to and its
    compiled range the declared one (to 1e-12: MuJoCo converts degree ranges itself).
    The values are joint_limits.limit_port_init's, at the config's physics step: the
    learned plant's limit port then starts at the compiled limit's stiffness, damping
    and width instead of the plant's generic constants.
    """
    from environment.ph_sac.dmc.joint_limits import limit_port_init

    spec = config["state_spec"]
    ranges = tuple(spec.get("limited_position_ranges", ()))
    if len(ranges) != limits.count:
        raise ValueError(
            f"{config['name']}: {limits.count} compiled limits for {len(ranges)} limited_position_ranges."
        )
    projection = np.asarray(spec["observation_projection"])
    for (index, low, high), dof, lower, upper in zip(ranges, limits.dofs, limits.lower, limits.upper):
        if int(np.argmax(np.abs(projection[int(index)]))) != int(dof):
            raise ValueError(f"{config['name']}: limited position {index} is not the compiled joint's dof {dof}.")
        if not all(
            math.isclose(float(declared), float(compiled), rel_tol=1e-12, abs_tol=1e-12)
            for declared, compiled in ((low, lower), (high, upper))
        ):
            raise ValueError(
                f"{config['name']}: declared range {(low, high)} of position {index} is not "
                f"the compiled {(float(lower), float(upper))}."
            )
    spec["limit_port_init"] = limit_port_init(limits)
    return config


def require_no_failure_set(name: str, failure_limits: Mapping[str, float]) -> None:
    """Raise unless failure_limits is empty: the task keeps dm_control's soft joint limits.

    The cart-pole, reacher and point-mass rows made their joint stops failure boundaries
    until 2026-11-10 (failure_limits cart_position, wrist_angle, position); a config of
    that convention no longer builds, so no run silently mixes the two.
    """
    if failure_limits:
        raise ValueError(
            f"{name} has no failure set (since 2026-11-10 dm_control's joint limits stay "
            f"MuJoCo's soft constraints); failure_limits must be empty, not {dict(failure_limits)!r}."
        )


def require_settings(
    name: str,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
    *,
    failure_keys: Sequence[str],
    reward_keys: Sequence[str],
    reset_keys: Sequence[str],
    plant_keys: Sequence[str],
) -> None:
    """Raise ValueError naming the first settings dict that misses a required key."""
    for label, mapping, keys in (
        ("failure_limits", failure_limits, failure_keys),
        ("reward_parameters", reward_parameters, reward_keys),
        ("reset_parameters", reset_parameters, reset_keys),
        ("plant_parameters", plant_parameters, plant_keys),
    ):
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise ValueError(f"{name} {label} is missing required keys {missing}.")


def dmc_testbed_config(
    *,
    name: str,
    state_spec: StateSpec,
    reference_period: float,
    episode_duration: float,
    controller: ControllerConfig,
    plant_parameters: Mapping[str, float],
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    physics_dt: float = DMC_PHYSICS_DT,
    discount_rate: float = DMC_DISCOUNT_RATE,
    integrator: Literal["Euler", "RK4"] = DMC_INTEGRATOR,
) -> TestbedConfig:
    """Assemble a TestbedConfig with the dm_control rows' shared physics and discount."""
    return {
        "name": name,
        "state_spec": state_spec,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "episode_duration": episode_duration,
        "discount_rate": discount_rate,
        "clock": fixed_clock(reference_period, physics_dt),
        "controller": controller,
        "integrator": integrator,
        "plant_parameters": plant_parameters,
        "failure_limits": failure_limits,
        "reward_parameters": reward_parameters,
        "reset_parameters": reset_parameters,
    }


def factory_todo(testbed: str) -> NotImplementedError:
    """The error a stub factory raises."""
    return NotImplementedError(
        f"The config factory of testbed {testbed!r} is a stub; port the testbed first "
        "(see environment/ph_sac/dmc/README.md)."
    )
