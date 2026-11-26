"""Config factories of the dm_control lqr ports (lqr_2_1, lqr_6_2).

Each factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The environment, the tensor task functions and the
registration live in environment/ph_sac/dmc/lqr.py; environment/ph_sac/dmc/README.md
describes the contract. Import common.ph_config only under TYPE_CHECKING here (it imports
this package).

plant_parameters (the row's keys, all required):
    n_bodies, n_actuated   2 and 1 (lqr_2_1) or 6 and 2 (lqr_6_2), checked against the task
    body_mass              kg per sphere. The row writes 4.19; dm_control's compiled mass is
                           the radius-0.1 sphere at density 1000, 4.188790204786391 kg.
    stiffness_low/high     dm_control's spring range U[15, 25] N/m; checks the stiffnesses
    damping                N s/m on every joint (dm_control's damping range is [0, 0])
and one OPTIONAL key the row does not give:
    stiffness              the n spring stiffnesses, joint by joint (a sequence, or the text
                           "k1,k2,..." that a table dict cell carries). dm_control draws them
                           when the model is built, from the task's RandomState, alternating
                           a stiffness and a damping draw per body; without a seed the draw
                           is unreproducible. Absent, the factory uses the draw of
                           task_kwargs={"random": 0} (DM_CONTROL_SEED0_STIFFNESS, read from
                           dm_control's compiled models), and the returned config states it.
reward_parameters: control_cost (0.1). reset_parameters: position_radius (sqrt 2),
velocity_noise (0). failure_limits: {} (no failure set).

The motor limit +-20 N is the table's choice: dm_control's motors are unlimited. The
discounted (lambda 0.5) Riccati optimum of the unconstrained problem stays below 2 N over
10 s from the reset sphere, peaking at about 1.3-1.6 N depending on the reset (5.7 N
undiscounted), so the limit does not bind on the optimal closed loop
(tests/test_ph_dmc_lqr.py, TestRiccatiDiagnostic).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

from common.ph_dmc._common import (
    box_rule_controller,
    dmc_state_spec,
    dmc_testbed_config,
)

if TYPE_CHECKING:
    from common.ph_config import TestbedConfig

# dm_control's spring stiffnesses with task_kwargs={"random": 0}, joint by joint: the
# draws of np.random.RandomState(0) at even positions (each body draws its stiffness
# from U[15, 25], then its damping from U[0, 0]). lqr_2_1 compiles the first two.
DM_CONTROL_SEED0_STIFFNESS: Tuple[float, ...] = (
    20.488135039273246,
    21.02763376071644,
    19.236547993389046,
    19.375872112626926,
    24.636627605010293,
    22.917250380826644,
)
# The table's motor limit, N (dm_control's motors have no control limit).
LQR_TORQUE_LIMIT = 20.0
LQR_SETPOINT_OFFSET = 1.0
LQR_CONTROL_STEP = 0.03
LQR_EPISODE_DURATION = 10.0

PLANT_KEYS = ("n_bodies", "n_actuated", "body_mass", "stiffness_low", "stiffness_high", "damping")
REWARD_KEYS = ("control_cost",)
RESET_KEYS = ("position_radius", "velocity_noise")


def _require(mapping: Mapping[str, object], keys: Tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required keys {missing}.")


def lqr_stiffness(plant_parameters: Mapping[str, object], n_bodies: int) -> Tuple[float, ...]:
    """The n spring stiffnesses: plant_parameters["stiffness"], else dm_control's seed-0 draw.

    The key holds a sequence or comma-separated text (a table cell's form, since dict
    cells hold scalars). Each stiffness must lie in [stiffness_low, stiffness_high].
    """
    if "stiffness" in plant_parameters:
        given = plant_parameters["stiffness"]
        if isinstance(given, str):
            given = given.split(",")
        elif isinstance(given, (int, float)):
            given = (given,)
        stiffness = tuple(float(value) for value in given)
    elif n_bodies <= len(DM_CONTROL_SEED0_STIFFNESS):
        stiffness = DM_CONTROL_SEED0_STIFFNESS[:n_bodies]
    else:
        raise ValueError(f"No default stiffness for {n_bodies} bodies; give 'stiffness'.")
    if len(stiffness) != n_bodies:
        raise ValueError(f"The lqr needs {n_bodies} stiffnesses, got {len(stiffness)}.")
    low, high = float(plant_parameters["stiffness_low"]), float(plant_parameters["stiffness_high"])
    if any(not (low <= value <= high and math.isfinite(value)) for value in stiffness):
        raise ValueError(f"The stiffnesses {stiffness} leave [{low}, {high}].")
    return stiffness


def _lqr_config(
    name: str,
    n_bodies: int,
    n_actuated: int,
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, object],
    torque_limit: Optional[float] = None,
) -> TestbedConfig:
    """n nested slides, the first n_actuated driven by +-torque_limit motors (default
    LQR_TORQUE_LIMIT, 20 N; every table row); 30 ms decisions, 10 s. The decision box
    follows the box rule of the motor (box_rule_controller): stiffness up to 10 x
    torque_limit / 1 m, damping up to 0.2 s times that. These factories keep the fixed
    clock; the registered ones (environment/ph_sac/dmc/lqr.py) are wrapped by
    _common.with_clock_settings and also take the optional irregular clock settings."""
    if torque_limit is None:
        torque_limit = LQR_TORQUE_LIMIT
    if isinstance(torque_limit, bool) or not float(torque_limit) > 0.0 or not math.isfinite(float(torque_limit)):
        raise ValueError(f"{name} needs a positive finite torque_limit, not {torque_limit!r}.")
    _require(plant_parameters, PLANT_KEYS, f"{name} plant_parameters")
    _require(reward_parameters, REWARD_KEYS, f"{name} reward_parameters")
    _require(reset_parameters, RESET_KEYS, f"{name} reset_parameters")
    if failure_limits:
        raise ValueError(f"{name} has no failure set; got failure_limits {dict(failure_limits)}.")
    sizes = (plant_parameters["n_bodies"], plant_parameters["n_actuated"])
    if sizes != (n_bodies, n_actuated):
        raise ValueError(
            f"{name} has {n_bodies} bodies and {n_actuated} actuators; "
            f"plant_parameters give {sizes}."
        )
    plant = dict(plant_parameters)
    plant["stiffness"] = lqr_stiffness(plant_parameters, n_bodies)
    return dmc_testbed_config(
        name=name,
        state_spec=dmc_state_spec(
            physical_dim=n_bodies,
            action_dim=n_actuated,
            periodic=(),
            actuated=tuple(range(n_actuated)),
        ),
        reference_period=LQR_CONTROL_STEP,
        episode_duration=LQR_EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(LQR_SETPOINT_OFFSET,) * n_actuated,
            torque_limit=float(torque_limit),
            setpoint_mode="offset",
            setpoint_error="unwrapped",
        ),
        plant_parameters=plant,
        failure_limits=dict(failure_limits),
        reward_parameters=dict(reward_parameters),
        reset_parameters=dict(reset_parameters),
    )


def lqr_2_1_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
    torque_limit: Optional[float] = None,
) -> TestbedConfig:
    """dm_control's lqr/lqr_2_1 (table row lqr-lqr_2_1): 2 bodies, the first actuated.

    State (q_1, q_2, qd_1, qd_2), relative slide displacements; 1 ms RK4, 30 ms fixed
    clock, 10 s episodes, lambda 0.5; decisions (setpoint offset in +-1 m, stiffness
    0-200 N/m, damping 0-40 N s/m), torque +-20 N. Without plant_parameters["stiffness"]
    the springs are dm_control's seed-0 draw (20.488135039273246, 21.02763376071644).
    """
    return _lqr_config(
        "lqr_2_1",
        2,
        1,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
        torque_limit=torque_limit,
    )


def lqr_6_2_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
    torque_limit: Optional[float] = None,
) -> TestbedConfig:
    """dm_control's lqr/lqr_6_2 (table row lqr-lqr_6_2): 6 bodies, the first 2 actuated.

    State (q_1..q_6, qd_1..qd_6); clock, physics, box and torque limit as lqr_2_1. Without
    plant_parameters["stiffness"] the springs are dm_control's seed-0 draw
    DM_CONTROL_SEED0_STIFFNESS.
    """
    return _lqr_config(
        "lqr_6_2",
        6,
        2,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
        torque_limit=torque_limit,
    )
