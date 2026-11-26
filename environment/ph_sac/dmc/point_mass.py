"""dm_control's point_mass/easy as the testbed point_mass_easy.

dm_control (suite/point_mass.xml, point_mass.py): a 0.3 kg point (sphere geom) on two
slide joints root_x, root_y (axes +x, +y), each with damping 1 and limited to
[-0.29, 0.29] by the constraint solver, contact disabled; two motors of gear 0.1 drive
the slides through fixed tendons t1 = root_x, t2 = root_y, control in [-1, 1] (force
+-0.1 N per axis); Euler at 0.02 s. Reset (randomize_limited_and_rotational_joints):
each slide uniform within its range, velocities 0. The target is fixed at the origin
with radius 0.015. Reward: tolerance(|p|, (0, 0.015), margin 0.015) (gaussian, value 0.1
at the margin) times (4 + mean_i tolerance(u_i, margin 1, quadratic, value_at_margin
0)) / 5, u the control in [-1, 1]. 20 s episodes. (point_mass/hard redraws the tendon
gains per episode and is excluded.)

The port (row point_mass-easy; factory in common/ph_dmc/point_mass.py):
  state       (x, y, x_dot, y_dot), both travel coordinates (raw, not periodic), both
              actuated with offset setpoints and unwrapped errors; no context.
  model       the slides KEEP dm_control's soft joint limits (since 2026-11-10: limited,
              range +-slide_range = 0.29 m, MuJoCo's default solreflimit 0.02 1 and
              solimplimit 0.9 0.95 0.001 0.5 2 from plant_parameters, the constraint
              solver on with its default Newton settings) and the motors act on the joints
              directly in newtons (gear 1, ctrlrange +-0.1): dm_control's tendons are an
              identity map, so the generalized forces are the same; the configured step
              and integrator (1 ms RK4 in the table). MuJoCo floors the limits' time
              constant at 2 x timestep: 0.02 s at our 1 ms, 0.04 s in dm_control's 20 ms
              model (the same compiled limit, softer at dm_control's step).
  reward      tolerance(|p|, (0, target_radius), margin) (gaussian) *
              (4 + mean_i max(0, 1 - (u_i / 0.1)^2)) / 5, u_i the force in newtons, as
              dm_control's with its control u_i / 0.1.
  failure     none (payoff failure_rate / lambda = 0 never charged, H = 0, B = 1): a slide
              past +-0.29 m meets dm_control's soft limit and continues, as in dm_control.
              (Until 2026-11-10 the limits were removed and max(|x|, |y|) >= 0.29 m was a
              failure with payoff 0 and lift B = (1 - (x / 0.29)^2)(1 - (y / 0.29)^2).)
  oracle      linear and closed form: M = mass I, D = damping I, U = 0, G = I in newtons
              (dm_control's G = 0.1 I on its control), qdd = (u - damping qd) / mass, plus
              the slides' soft-limit forces (joint_limits.constrained_accelerations with
              the compiled model's limits; M diagonal, so each slide's row is its own),
              exactly 0 within the box; zero diffusion. The state spec declares both
              slides as limited_position_ranges for the learned plant's limit port,
              which starts at these limits (limit_port_init, the factory's, from the model).
  upright     None: the point mass has no upright pose, so the evaluation counts no
              state as upright.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc._common import require_no_failure_set
from common.ph_dmc.point_mass import (
    FAILURE_KEYS,
    PLANT_KEYS,
    RESET_KEYS,
    REWARD_KEYS,
    point_mass_easy_config,
)

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
# The irregular clocks (optional testbed_settings, since 2026-11-11; _common.py).
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, with_clock_settings  # noqa: E402
from ._common import (
    NO_FAILURE,
    failure_rate_payoff,
    limit_attributes,
    limit_boundary_functions,
    limit_failure_mask,
    require_keys,
    tolerance,
    tolerance_tensor,
    torque_limit,
    unit_interval_bounds,
    zero_diffusion,
)
from .joint_limits import (
    JointLimits,
    constrained_accelerations,
    constrained_accelerations_numpy,
    joint_limits_from_model,
)
from .limited_chain import limits_engaged_numpy

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

TESTBED = "point_mass_easy"
SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")

# dm_control's point_mass.xml with its soft slide limits, the motors on the joints in
# newtons. The walls mark the +-0.3 m arena; contact is disabled, so they only draw.
_POINT_MASS_XML = """
<mujoco model="planar point mass">
  <option timestep="{timestep}" integrator="{integrator}">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="slide" damping="{damping}" limited="true" range="-{slide} {slide}" {limit}/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" rgba=".25 .3 .35 1"/>
    <geom name="wall_x" type="box" pos="-.3 0 .02" size=".005 .3 .02" rgba=".5 .5 .5 1"/>
    <geom name="wall_y" type="box" pos="0 -.3 .02" size=".3 .005 .02" rgba=".5 .5 .5 1"/>
    <geom name="wall_neg_x" type="box" pos=".3 0 .02" size=".005 .3 .02" rgba=".5 .5 .5 1"/>
    <geom name="wall_neg_y" type="box" pos="0 .3 .02" size=".3 .005 .02" rgba=".5 .5 .5 1"/>
    <body name="pointmass" pos="0 0 .01">
      <joint name="root_x" pos="0 0 0" axis="1 0 0"/>
      <joint name="root_y" pos="0 0 0" axis="0 1 0"/>
      <geom name="pointmass" type="sphere" size=".01" mass="{mass}" rgba=".8 .3 .2 1"/>
    </body>
    <geom name="target" pos="0 0 .01" type="sphere" size="{target_radius}" rgba=".3 .8 .3 .5"/>
  </worldbody>
  <actuator>
    <motor name="t1" joint="root_x" gear="1" ctrllimited="true" ctrlrange="{x_low} {x_high}"/>
    <motor name="t2" joint="root_y" gear="1" ctrllimited="true" ctrlrange="{y_low} {y_high}"/>
  </actuator>
</mujoco>
"""


def _require_point_mass_keys(config: TestbedConfig) -> None:
    require_keys(config["plant_parameters"], PLANT_KEYS, "Point-mass plant_parameters")
    require_keys(config["failure_limits"], FAILURE_KEYS, "Point-mass failure_limits")
    require_no_failure_set(config["name"], config["failure_limits"])
    require_keys(config["reward_parameters"], REWARD_KEYS, "Point-mass reward_parameters")
    require_keys(config["reset_parameters"], RESET_KEYS, "Point-mass reset_parameters")


def build_point_mass_model(config: TestbedConfig) -> MjModel:
    """Compile dm_control's point mass with its soft slide limits, motors in newtons."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    return mujoco.MjModel.from_xml_string(
        _POINT_MASS_XML.format(
            timestep=float(config["physics_dt"]),
            integrator=config["integrator"],
            damping=float(plant["damping"]),
            slide=float(plant["slide_range"]),
            limit=limit_attributes(plant),
            mass=float(plant["mass"]),
            target_radius=float(config["reward_parameters"]["target_radius"]),
            x_low=float(controller["torque_low"][0]),
            x_high=float(controller["torque_high"][0]),
            y_low=float(controller["torque_low"][1]),
            y_high=float(controller["torque_high"][1]),
        )
    )


def point_mass_limits(model: object) -> JointLimits:
    """The slides' soft limits of a compiled point mass (ours or dm_control's)."""
    return joint_limits_from_model(model, ("root_x", "root_y"))


class PointMassEnv(MujocoPHEnv):
    """dm_control's point mass with its soft slide limits. State (x, y, x_dot, y_dot)."""

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require_point_mass_keys(config)
        super().__init__(
            config,
            build_point_mass_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        if (self.physical_position_dim, self.context_dim, self.velocity_dim) != (2, 0, 2):
            raise ValueError("The point mass needs 2 positions, no context and 2 velocities.")
        plant = config["plant_parameters"]
        self._mass = float(plant["mass"])
        self._damping = float(plant["damping"])
        self._force_limit = torque_limit(config)
        self._limits = point_mass_limits(self.model)

    @property
    def slide_limits(self) -> JointLimits:
        """The slides' soft limits, read from the compiled model."""
        return self._limits

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's reset: each slide uniform within its range (x, then y), at rest."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        rng = self._reset_rng
        span = float(reset["position_range"])
        positions = np.array([rng.uniform(-span, span), rng.uniform(-span, span)])
        velocities = reset["velocity_noise"] * rng.standard_normal(2)
        self._set_state(np.concatenate((positions, velocities)))

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's reward; point_mass_reward_rate_tensor must stay its copy."""
        reward = self.config["reward_parameters"]
        radius = float(reward["target_radius"])
        near = tolerance(math.hypot(float(state[0]), float(state[1])), (0.0, radius), float(reward["margin"]))
        control = np.asarray(torque, dtype=np.float64).reshape(-1) / self._force_limit
        small = tolerance(control, margin=1.0, sigmoid="quadratic", value_at_margin=0.0)
        return float(near * (4.0 + float(np.mean(small))) / 5.0)

    def failed(self, state: ndarray) -> bool:
        """Never: a slide past +-0.29 m meets dm_control's soft limit and continues."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return failure_rate_payoff(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """The linear drift, plus the slides' soft-limit forces beyond +-0.29 m."""
        state = np.asarray(state, dtype=np.float64)
        velocities = state[2:4]
        force = np.asarray(torque, dtype=np.float64).reshape(-1)
        accelerations = (force - self._damping * velocities) / self._mass
        if limits_engaged_numpy(self._limits, state[:2]):
            accelerations = constrained_accelerations_numpy(
                self._limits, self._mass * np.eye(2), accelerations, state[:2], velocities
            )["qacc"]
        return np.concatenate((velocities, accelerations))

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The point mass has no noise."""
        return np.zeros((self.state_dim, 1))


def point_mass_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """Near-target tolerance times the control factor, [B,1]."""
    reward = config["reward_parameters"]
    radius = float(reward["target_radius"])
    distance = torch.linalg.vector_norm(states[..., 0:2], dim=-1, keepdim=True)
    near = tolerance_tensor(distance, (0.0, radius), float(reward["margin"]))
    control = torques.to(states.dtype) / torque_limit(config)
    small = tolerance_tensor(control, margin=1.0, sigmoid="quadratic", value_at_margin=0.0)
    return near * (4.0 + small.mean(dim=-1, keepdim=True)) / 5.0


def point_mass_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """All False, [B]: no failure set (the slide limits are soft constraints)."""
    return limit_failure_mask(states, NO_FAILURE)


def point_mass_terminal_payoff_value(config: TestbedConfig) -> float:
    """failure_rate / lambda (0 in the table)."""
    return failure_rate_payoff(config)


def point_mass_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure boundary: H = failure_rate / lambda (0) and B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), NO_FAILURE)


def point_mass_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The linear drift (qd, (u - damping qd) / mass) with the compiled model's soft slide
    limits (exactly 0 within the box), and a zero diffusion."""
    plant = config["plant_parameters"]
    mass, damping = float(plant["mass"]), float(plant["damping"])
    limits = point_mass_limits(build_point_mass_model(config))
    root_mass = math.sqrt(mass)

    def point_mass_drift(states: Tensor, torques: Tensor) -> Tensor:
        velocities = states[..., 2:4]
        accelerations = (torques.to(states.dtype) - damping * velocities) / mass
        # M = mass I: its Cholesky factor is sqrt(mass) I.
        factor = root_mass * torch.eye(2, dtype=states.dtype, device=states.device).expand(
            *states.shape[:-1], 2, 2
        )
        accelerations = constrained_accelerations(
            limits, factor, accelerations, states[..., 0:2], velocities
        )["qacc"]
        return torch.cat((velocities, accelerations), dim=-1)

    return point_mass_drift, zero_diffusion(4)


register_testbed(
    TestbedSpec(
        name=TESTBED,
        env_class=PointMassEnv,
        config_factory=with_clock_settings(point_mass_easy_config),
        settings_keys=SETTINGS_KEYS,
        optional_settings_keys=CLOCK_SETTINGS_KEYS,
        reward_rate_tensor=point_mass_reward_rate_tensor,
        failure_mask_tensor=point_mass_failure_mask_tensor,
        terminal_payoff_value=point_mass_terminal_payoff_value,
        terminal_boundary_functions=point_mass_terminal_boundary_functions,
        oracle_callbacks=point_mass_oracle_callbacks,
        upright=None,
        reward_rate_bounds=unit_interval_bounds,
        # The testbed_settings key discount_rate (_common.py, with_discount_rate).
        discount_setting=True,
    )
)
