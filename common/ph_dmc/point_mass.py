"""Config factories of the dm_control point_mass ports (point_mass_easy).

The factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The environment, the tensor task functions and the
registration live in environment/ph_sac/dmc/point_mass.py;
environment/ph_sac/dmc/README.md describes the contract. Import common.ph_config only
under TYPE_CHECKING here (it imports this package).

State (x, y, x_dot, y_dot): the two slide positions, travel coordinates (not periodic),
both actuated, then their rates; no context.

Both slides keep dm_control's soft joint limits (since 2026-11-10): range +-slide_range
(0.29 m) with MuJoCo's default solreflimit and solimplimit (LIMIT_KEYS). No failure set
(failure_limits must be empty); the state spec declares both slides as
limited_position_ranges, read by the learned plant's limit port (network limit_port, set
by the row), and the port's start limit_port_init, read from the compiled model since
2026-11-10 (with_limit_port_init: k 750 N/m, c 30 N s/m, w 1.053e-3 m per slide).
"""

from __future__ import annotations

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
# mass, damping, and dm_control's slide limits: range +-slide_range with MuJoCo's default
# solreflimit and solimplimit.
PLANT_KEYS: Tuple[str, ...] = ("mass", "damping", "slide_range") + LIMIT_KEYS
REWARD_KEYS: Tuple[str, ...] = ("target_radius", "margin", "failure_rate")
RESET_KEYS: Tuple[str, ...] = ("position_range", "velocity_noise")
FAILURE_KEYS: Tuple[str, ...] = ()  # no failure set: the slide limits stay soft constraints

FORCE_LIMIT = 0.1  # N per axis: dm_control's gear 0.1 times control in [-1, 1]
SETPOINT_OFFSET = 0.3  # m, the table's decision-box offset for both axes
REFERENCE_PERIOD = 0.02  # s, dm_control's control step
EPISODE_DURATION = 20.0  # s


def point_mass_easy_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's point_mass/easy (table row point_mass-easy, mode top).

    The row passes these testbed_settings (its env_* cells):
        failure_limits     {} (none: the slide limits are soft constraints)
        reward_parameters  target_radius=0.015;margin=0.015;failure_rate=0.0
        reset_parameters   position_range=0.29;velocity_noise=0.0
        plant_parameters   mass=0.3;damping=1.0;slide_range=0.29;limit_timeconst=0.02;
                           limit_dampratio=1.0;limit_d0=0.9;limit_dmax=0.95;
                           limit_width=0.001;limit_midpoint=0.5;limit_power=2.0
    and the TestbedConfig reproduces the row's task_* cells: nq 2, context_dim 0, nv 2,
    nu 2, no periodic position, both actuated and limited to +-0.29 m
    (limited_position_ranges), 1 ms RK4, a fixed 20 ms clock, 20 s
    episodes, lambda 0.5, decision box setpoint offsets +-0.3 m, stiffness 0-3.3333 N/m,
    damping 0-0.66667 N s/m, forces +-0.1 N, unwrapped setpoint errors, refresh every
    physics step.
    """
    name = "point_mass_easy"
    require_settings(
        name, failure_limits, reward_parameters, reset_parameters, plant_parameters,
        failure_keys=FAILURE_KEYS, reward_keys=REWARD_KEYS, reset_keys=RESET_KEYS,
        plant_keys=PLANT_KEYS,
    )
    require_no_failure_set(name, failure_limits)
    slide = float(plant_parameters["slide_range"])
    if float(reset_parameters["position_range"]) != slide:
        raise ValueError(
            f"{name} reset_parameters position_range must be the slides' range {slide!r} "
            "(dm_control draws each slide within its range)."
        )
    config = dmc_testbed_config(
        name=name,
        state_spec=dmc_state_spec(
            physical_dim=2,
            action_dim=2,
            periodic=(),
            actuated=(0, 1),
            limited_position_ranges=((0, -slide, slide), (1, -slide, slide)),
        ),
        reference_period=REFERENCE_PERIOD,
        episode_duration=EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(SETPOINT_OFFSET, SETPOINT_OFFSET),
            torque_limit=FORCE_LIMIT,
            setpoint_mode="offset",
            setpoint_error="unwrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
    )
    # The slides' limit_port_init from the compiled model; environment/ph_sac/dmc/
    # point_mass.py imports this module, so it is imported here.
    from environment.ph_sac.dmc.point_mass import build_point_mass_model, point_mass_limits

    return with_limit_port_init(config, point_mass_limits(build_point_mass_model(config)))
