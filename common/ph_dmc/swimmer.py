"""Config factories of the dm_control swimmer ports (swimmer_swimmer6, swimmer_swimmer15).

Each factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The two tasks differ only in the link count n
(plant_parameters n_links: 6 or 15), which fixes the joint range 2 pi / n, the state
size and the decision box. The environment, the tensor task functions and the
registration live in environment/ph_sac/dmc/swimmer.py; environment/ph_sac/dmc/README.md
describes the contract. Import common.ph_config only under TYPE_CHECKING here (it
imports this package).

State (x, y, psi, theta_1 .. theta_{n-1}, x_target, y_target, rates of the first n + 2):
MuJoCo's qpos of dm_control's swimmer (root slides x, y, root yaw psi, the n - 1 joint
angles), then the episode-constant target as a two-dimensional context, then qvel. Only
the yaw is periodic; x and y are travel coordinates; the joints are actuated and keep
dm_control's soft limits at +-2 pi / n (no failure set: a joint may pass its range, and
MuJoCo's limit constraint pushes it back). x, y and the target are cyclic for the
dynamics (no force depends on them), which the state spec declares
(cyclic_position_indices) so that the learned plant leaves them out of its features; the
joint ranges are declared as limited_position_ranges, which the learned plant's limit
port (network limit_port) reads, starting (since 2026-11-10) at limit_port_init, read
from the compiled model (with_limit_port_init: k 2.7e-3 to 3.7e-3 N m/rad, c 2.7e-4 to
3.7e-4 N m s/rad, w 0.101 rad).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Tuple

from common.ph_dmc._common import (
    box_rule_controller,
    dmc_state_spec,
    dmc_testbed_config,
    with_limit_port_init,
)

if TYPE_CHECKING:
    from common.ph_config import TestbedConfig
    from common.ph_types import StateSpec

# Required keys of each settings dict; the environment checks them again.
PLANT_KEYS: Tuple[str, ...] = (
    "n_links",
    "link_mass",
    "link_length",
    "box_half_width",
    "box_half_height",
    "armature",
    "density",
    "viscosity",
    "nose_offset",
    # dm_control's joint limits (solreflimit and solimplimit of its swimmer.xml); the
    # range itself is dm_control's 360 / n deg.
    "limit_timeconst",
    "limit_dampratio",
    "limit_d0",
    "limit_dmax",
    "limit_width",
    "limit_midpoint",
    "limit_power",
)
REWARD_KEYS: Tuple[str, ...] = ("target_radius", "margin", "failure_rate")
RESET_KEYS: Tuple[str, ...] = (
    "yaw_range",
    "joint_range",
    "close_target_probability",
    "close_target_range",
    "far_target_range",
    "velocity_noise",
)
FAILURE_KEYS: Tuple[str, ...] = ()  # no failure set: the limits stay soft constraints

TORQUE_LIMIT = 5e-4  # N m per joint: dm_control's gear 5e-4 times control in [-1, 1]
REFERENCE_PERIOD = 0.03  # s, dm_control's control step
# dm_control's physics (swimmer.xml: timestep 0.002, no integrator attribute, so MuJoCo's
# default Euler), the swimmers' physics since 2026-11-10 (1 ms RK4 before): 15 physics
# steps per 30 ms decision. The rows' model step is this step and their model integrator
# semi_implicit, MuJoCo's Euler step without dof damping (common.ph_config.MODEL_STEPS).
PHYSICS_DT = 0.002
INTEGRATOR = "Euler"
EPISODE_DURATION = 30.0  # s
CONTEXT_DIM = 2  # the target (x*, y*)
LINK_COUNTS = {"swimmer_swimmer6": 6, "swimmer_swimmer15": 15}


def swimmer_joint_range(n_links: int) -> float:
    """dm_control's joint range 360 deg / n in radians: 2 pi / n.

    It is the range of every joint's soft limit, the reset range and the decision box's
    setpoint offset; the table generator computes the same float through this function.
    """
    return 2.0 * math.pi / int(n_links)


def swimmer_state_spec(n_links: int) -> StateSpec:
    """(x, y, psi, theta_1 .. theta_{n-1}, x*, y*; rates): yaw periodic, joints actuated.

    The root position (0, 1) and the context (the last two positions) are cyclic: no
    force depends on them, so the learned plant drops them from its features. The joints
    (3 .. n + 1) are limited to +-2 pi / n (limited_position_ranges, read by the learned
    plant's limit port; _swimmer_config adds the port's start limit_port_init, which
    needs the compiled model).
    """
    physical = int(n_links) + 2
    joint_range = swimmer_joint_range(n_links)
    spec = dmc_state_spec(
        physical_dim=physical,
        action_dim=physical - 3,
        periodic=(2,),
        actuated=tuple(range(3, physical)),
        context_dim=CONTEXT_DIM,
    )
    spec["cyclic_position_indices"] = (0, 1, physical, physical + 1)
    spec["limited_position_ranges"] = tuple(
        (index, -joint_range, joint_range) for index in range(3, physical)
    )
    return spec


def _check_settings(
    name: str,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> int:
    for label, mapping, keys in (
        ("failure_limits", failure_limits, FAILURE_KEYS),
        ("reward_parameters", reward_parameters, REWARD_KEYS),
        ("reset_parameters", reset_parameters, RESET_KEYS),
        ("plant_parameters", plant_parameters, PLANT_KEYS),
    ):
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise ValueError(f"{name} {label} is missing required keys {missing}.")
    n_links = plant_parameters["n_links"]
    if isinstance(n_links, bool) or not isinstance(n_links, int) or n_links != LINK_COUNTS[name]:
        raise ValueError(f"{name} needs plant_parameters n_links={LINK_COUNTS[name]}.")
    if failure_limits:
        raise ValueError(f"{name} has no failure set: its joint limits stay soft constraints.")
    joint_range = swimmer_joint_range(n_links)
    if float(reset_parameters["joint_range"]) != joint_range:
        raise ValueError(
            f"{name} reset_parameters joint_range must be dm_control's range 2 pi / n = {joint_range!r}."
        )
    return int(n_links)


def _swimmer_config(
    name: str,
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    n_links = _check_settings(
        name, failure_limits, reward_parameters, reset_parameters, plant_parameters
    )
    joint_range = swimmer_joint_range(n_links)
    config = dmc_testbed_config(
        name=name,
        state_spec=swimmer_state_spec(n_links),
        reference_period=REFERENCE_PERIOD,
        episode_duration=EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(joint_range,) * (n_links - 1),
            torque_limit=TORQUE_LIMIT,
            setpoint_mode="offset",
            setpoint_error="unwrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        physics_dt=PHYSICS_DT,
        integrator=INTEGRATOR,
    )
    # The joints' limit_port_init from the compiled model; environment/ph_sac/dmc/swimmer.py
    # imports this module, so it is imported here.
    from environment.ph_sac.dmc.joint_limits import joint_limits_from_model
    from environment.ph_sac.dmc.swimmer import build_swimmer_model

    joints = tuple(f"joint_{index}" for index in range(n_links - 1))
    return with_limit_port_init(config, joint_limits_from_model(build_swimmer_model(config), joints))


def swimmer_swimmer6_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's swimmer/swimmer6 (table row swimmer-swimmer6, mode top).

    The row passes these testbed_settings (its env_* cells):
        failure_limits     {} (none: the joint limits stay soft constraints)
        reward_parameters  target_radius=0.1;margin=0.5;failure_rate=0.0
        reset_parameters   yaw_range=3.141592653589793;joint_range=1.0471975511965976;
                           close_target_probability=0.2;close_target_range=0.3;
                           far_target_range=2.0;velocity_noise=0.0
        plant_parameters   n_links=6;link_mass=0.01;link_length=0.1;box_half_width=0.001;
                           box_half_height=0.01;armature=1e-06;density=3000.0;
                           viscosity=0.0;nose_offset=0.01;limit_timeconst=0.05;
                           limit_dampratio=1.0;limit_d0=0.0;limit_dmax=0.8;
                           limit_width=0.1;limit_midpoint=0.5;limit_power=2.0
    and the TestbedConfig reproduces the row's task_* cells: 8 physical positions plus
    context_dim 2 (position_dim 10), nv 8, nu 5, the yaw (index 2) periodic, the joints
    (indices 3-7) actuated, dm_control's 2 ms Euler (1 ms RK4 before 2026-11-10), a fixed
    30 ms clock (15 physics steps), 30 s episodes, lambda 0.5,
    decision box setpoint offsets +-pi/3 rad, stiffness 0-0.0047746 N m/rad, damping
    0-0.00095493 N m s/rad, torques +-5e-4 N m, unwrapped setpoint errors, refresh every
    physics step.
    """
    return _swimmer_config(
        "swimmer_swimmer6",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def swimmer_swimmer15_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's swimmer/swimmer15 (table row swimmer-swimmer15, mode top).

    As swimmer_swimmer6_config with n_links=15: joint_range 2 pi / 15 =
    0.41887902047863906, 17 physical positions plus context_dim 2, nv 17, nu 14, the
    joints at indices 3-16, setpoint offsets +-2 pi/15 rad, stiffness 0-0.011937 N m/rad,
    damping 0-0.0023873 N m s/rad.
    """
    return _swimmer_config(
        "swimmer_swimmer15",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )
