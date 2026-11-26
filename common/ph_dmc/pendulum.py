"""Config factory of the dm_control pendulum port (pendulum_swingup).

The factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The environment, the tensor task functions and the
registration live in environment/ph_sac/dmc/pendulum.py; environment/ph_sac/dmc/README.md
describes the contract. Import common.ph_config only under TYPE_CHECKING here (it imports
this package).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

from common.ph_dmc._common import box_rule_controller, dmc_state_spec, dmc_testbed_config

if TYPE_CHECKING:
    from common.ph_config import TestbedConfig

# dm_control's pendulum: motor gear 1 with control in [-1, 1], so +-1 N m on the hinge.
PENDULUM_TORQUE_LIMIT = 1.0
# dm_control's control step (its Euler physics step, one substep) and time limit.
PENDULUM_CONTROL_PERIOD = 0.02
PENDULUM_EPISODE_DURATION = 20.0


def pendulum_swingup_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's pendulum/swingup (table row pendulum-swingup, mode top).

    The row passes these testbed_settings (its env_* cells):
        failure_limits     {}
        reward_parameters  cosine_bound=0.9902680687415704;margin=2.0 (dense since
                           2026-11-09; dm_control's margin is 0)
        reset_parameters   angle_range=3.141592653589793;velocity_noise=0.0
        plant_parameters   mass=1.0;length=0.5;bob_radius=0.05;damping=0.1;gravity=9.81
    and the TestbedConfig reproduces the row's task_* cells: state (theta, theta_dot),
    theta = 0 upright, periodic (wrapped) and actuated; 1 ms RK4 physics; a fixed clock at
    dm_control's 20 ms control step; 20 s episodes; discount rate 0.5 /s; the box rule's
    decisions (absolute setpoint within +-pi with a wrapped error, stiffness 0-10/pi,
    damping 0-2/pi) and torques within +-1 N m; refresh every physics step. The settings'
    keys are checked by the environment (PendulumSwingupEnv), as the cart-pole's are.
    """
    return dmc_testbed_config(
        name="pendulum_swingup",
        state_spec=dmc_state_spec(physical_dim=1, action_dim=1, periodic=(0,), actuated=(0,)),
        reference_period=PENDULUM_CONTROL_PERIOD,
        episode_duration=PENDULUM_EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(math.pi,),
            torque_limit=PENDULUM_TORQUE_LIMIT,
            setpoint_mode="absolute",
            setpoint_error="wrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
    )
