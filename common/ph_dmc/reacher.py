"""Config factories of the dm_control reacher ports (reacher_easy, reacher_hard).

Each factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The two tasks differ only in reach_radius (the
target plus finger radius: 0.06 easy, 0.025 hard). The environment, the tensor task
functions and the registration live in environment/ph_sac/dmc/reacher.py;
environment/ph_sac/dmc/README.md describes the contract. Import common.ph_config only
under TYPE_CHECKING here (it imports this package).

State (theta_s, theta_w, x_target, y_target, theta_dot_s, theta_dot_w): the shoulder and
the wrist (relative to the arm), both periodic and actuated, then the episode-constant
target as a two-dimensional context, then the two joint rates.

The wrist keeps dm_control's soft joint limit (since 2026-11-10): range +-160 deg
(plant_parameters wrist_range_degrees, as dm_control's XML states it) with MuJoCo's
default solreflimit and solimplimit (LIMIT_KEYS). No failure set (failure_limits must be
empty); the state spec declares the wrist as limited_position_ranges ((1, -w, w),), w in
radians, read by the learned plant's limit port (network limit_port, set by the rows),
and the port's start limit_port_init, read from the compiled model since 2026-11-10
(with_limit_port_init: k 0.1205 N m/rad, c 4.82e-3 N m s/rad, w 1.053e-3 rad).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Tuple

from common.ph_dmc._common import (
    LIMIT_KEYS,
    box_rule_controller,
    dmc_state_spec,
    dmc_testbed_config,
    require_no_failure_set,
    require_settings,
    with_limit_port_init,
)

if TYPE_CHECKING:
    from common.ph_config import TestbedConfig

# Required keys of each settings dict; the environment checks them again.
PLANT_KEYS: Tuple[str, ...] = (
    "arm_length",
    "hand_length",
    "finger_offset",
    "arm_mass",
    "hand_mass",
    "finger_mass",
    "link_radius",
    "finger_radius",
    "damping",
    # dm_control's wrist limit: range +-wrist_range_degrees (160, in degrees as its XML
    # writes it) with MuJoCo's default solreflimit and solimplimit.
    "wrist_range_degrees",
) + LIMIT_KEYS
REWARD_KEYS: Tuple[str, ...] = ("reach_radius", "margin", "failure_rate")
RESET_KEYS: Tuple[str, ...] = (
    "shoulder_range",
    "wrist_range",
    "target_radius_low",
    "target_radius_high",
    "velocity_noise",
)
FAILURE_KEYS: Tuple[str, ...] = ()  # no failure set: the wrist limit stays a soft constraint

TORQUE_LIMIT = 0.05  # N m per joint: dm_control's gear 0.05 times control in [-1, 1]
SETPOINT_OFFSET = 1.0  # rad, the table's decision-box offset for both joints
REFERENCE_PERIOD = 0.02  # s, dm_control's control step
EPISODE_DURATION = 20.0  # s


def wrist_range(plant_parameters: Mapping[str, float]) -> float:
    """The wrist's limit in radians, math.radians(wrist_range_degrees) (160 deg: 2.7925...)."""
    return math.radians(float(plant_parameters["wrist_range_degrees"]))


def _check_settings(
    name: str,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> None:
    require_settings(
        name, failure_limits, reward_parameters, reset_parameters, plant_parameters,
        failure_keys=FAILURE_KEYS, reward_keys=REWARD_KEYS, reset_keys=RESET_KEYS,
        plant_keys=PLANT_KEYS,
    )
    require_no_failure_set(name, failure_limits)
    if float(reset_parameters["wrist_range"]) != wrist_range(plant_parameters):
        raise ValueError(
            f"{name} reset_parameters wrist_range must be the wrist's limit "
            f"{wrist_range(plant_parameters)!r} rad (dm_control draws it within its range)."
        )


def _reacher_config(
    name: str,
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    _check_settings(name, failure_limits, reward_parameters, reset_parameters, plant_parameters)
    config = dmc_testbed_config(
        name=name,
        state_spec=dmc_state_spec(
            physical_dim=2,
            action_dim=2,
            periodic=(0, 1),
            actuated=(0, 1),
            context_dim=2,
            limited_position_ranges=((1, -wrist_range(plant_parameters), wrist_range(plant_parameters)),),
        ),
        reference_period=REFERENCE_PERIOD,
        episode_duration=EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(SETPOINT_OFFSET, SETPOINT_OFFSET),
            torque_limit=TORQUE_LIMIT,
            setpoint_mode="offset",
            setpoint_error="wrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
    )
    # The wrist's limit_port_init from the compiled model; environment/ph_sac/dmc/reacher.py
    # imports this module, so it is imported here.
    from environment.ph_sac.dmc.reacher import build_reacher_model, reacher_wrist_limits

    return with_limit_port_init(config, reacher_wrist_limits(build_reacher_model(config)))


def reacher_easy_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's reacher/easy (table row reacher-easy, mode top).

    The row passes these testbed_settings (its env_* cells):
        failure_limits     {} (none: the wrist limit is a soft constraint)
        reward_parameters  reach_radius=0.06;margin=0.0;failure_rate=0.0
        reset_parameters   shoulder_range=3.141592653589793;
                           wrist_range=2.792526803190927;target_radius_low=0.05;
                           target_radius_high=0.2;velocity_noise=0.0
        plant_parameters   arm_length=0.12;hand_length=0.1;finger_offset=0.12;
                           arm_mass=0.0418879...;hand_mass=0.0356047...;
                           finger_mass=0.00418879...;link_radius=0.01;finger_radius=0.01;
                           damping=0.01;wrist_range_degrees=160.0;limit_timeconst=0.02;
                           limit_dampratio=1.0;limit_d0=0.9;limit_dmax=0.95;
                           limit_width=0.001;limit_midpoint=0.5;limit_power=2.0
    and the TestbedConfig reproduces the row's task_* cells: 2 physical positions plus
    context_dim 2 (position_dim 4), nv 2, nu 2, both angles periodic and actuated (the
    wrist limited to +-160 deg: limited_position_ranges), 1 ms RK4, a fixed 20 ms clock, 20 s episodes, lambda 0.5, decision box setpoint offsets
    +-1 rad, stiffness 0-0.5 N m/rad, damping 0-0.1 N m s/rad, torques +-0.05 N m,
    wrapped setpoint errors, refresh every physics step.

    The masses are dm_control's rounded values: the model takes its masses and inertias
    from the geometry at density 1000, as dm_control compiles them, and only checks that
    they round to these (environment/ph_sac/dmc/reacher.py, build_reacher_model).
    """
    return _reacher_config(
        "reacher_easy",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def reacher_hard_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's reacher/hard (table row reacher-hard, mode top).

    As reacher_easy_config, with reward_parameters reach_radius=0.025 (target 0.015 plus
    finger 0.01); every task_* cell is the same.
    """
    return _reacher_config(
        "reacher_hard",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )
