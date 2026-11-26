"""dm_control's pendulum/swingup as the testbed pendulum_swingup.

dm_control (suite/pendulum.xml, pendulum.py): one hinge about +y at height 0.6 m, a
massless capsule pole of 0.5 m carrying a 1 kg sphere (radius 0.05 m) at its end, so the
body has mass 1, its centre of mass 0.5 m from the hinge and inertia 2/5 m r^2 = 0.001
about it; gravity 9.81, joint damping 0.1, contact disabled, no joint limit; motor gear 1
with control in [-1, 1] (torque +-1 N m); Euler at 0.02 s, one substep per control step.
Reset: the hinge uniform on [-pi, pi), velocity 0. Reward: tolerance(cos theta,
(cos 8 deg, 1)), margin 0: the indicator of the pole within 8 degrees of upright
(dm_control's xmat zz of the pole, which is cos theta). Episodes of 20 s.

The port (table row pendulum-swingup; the factory is common/ph_dmc/pendulum.py's
pendulum_swingup_config):
  state       (theta, theta_dot), theta = 0 upright (dm_control's hinge angle, positive
              toward +x), periodic, wrapped and actuated; no context.
  settings    plant_parameters mass, length, bob_radius, damping, gravity;
              reward_parameters cosine_bound, margin (a positive margin takes dm_control's
              default gaussian shape); reset_parameters angle_range, velocity_noise;
              failure_limits {} (no failure set).
  model       dm_control's geometry from plant_parameters: the bob a sphere geom with the
              mass (MuJoCo compiles its inertia 2/5 m r^2), the pole massless, the motor in
              N m (gear 1, ctrlrange the torque limits), contact and constraints disabled,
              timestep and integrator from the config (1 ms RK4).
  reward      r(s, u) = tolerance(cos theta, (cosine_bound, 1), margin); the torque does
              not enter. margin 0 is dm_control's indicator; the row's margin 2 (below)
              is dm_control's gaussian tolerance with value 0.1 at the margin.
  failure     none: an all-False mask, payoff failure_rate_payoff = 0 (no failure_rate),
              lift H = 0, B = 1.
  reset       theta uniform on [-angle_range, angle_range), velocity velocity_noise N(0,1)
              (0 in the row); options={"state": s} sets s.
  oracle      chain.chain_callbacks of the one-link PlanarChain (mass, length,
              0.4 mass bob_radius^2, damping, gravity, G = [[1]]); zero diffusion.
  upright     |wrap(theta)| <= tube.

Deviations from dm_control: 1 ms RK4 physics instead of 20 ms Euler (as every dm_control
row of the table), with the decision clock at dm_control's 20 ms; and, since 2026-11-09,
the row's reward margin. The row sets margin 2 where dm_control sets 0, so its reward is
tolerance(cos theta, (cos 8 deg, 1), margin=2) with dm_control's default gaussian shape:
1 within 8 degrees of upright, 0.569 with the pole horizontal and 0.102 hanging (the
margin spans the whole range of cos theta). It is the same function with another margin,
so no code changed, and it stays in [0, 1], so unit_interval_bounds and the sigmoid
reward head stay valid. With dm_control's indicator no arm learned the swing-up: the 1 N m
motor against 4.9 N m of gravity torque needs about 6.6 s of coherent pumping, random
exploration gains 1-3% of the energy needed, and in 14,600 training episodes none entered
the window from beyond 30 degrees, so the value stayed flat outside the upright basin
(the learned reward and plant were accurate). The tests compare a margin-0 config with
dm_control's task reward and the row's margin 2 with dm_control's rewards.tolerance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc.pendulum import pendulum_swingup_config

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
# The irregular clocks (optional testbed_settings, since 2026-11-11; _common.py).
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, with_clock_settings  # noqa: E402
from ._common import (
    failure_rate_payoff,
    limit_boundary_functions,
    limit_failed,
    limit_failure_mask,
    require_keys,
    tolerance,
    tolerance_tensor,
    unit_interval_bounds,
)
from .chain import PlanarChain, chain_callbacks, chain_drift_numpy

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

TESTBED = "pendulum_swingup"
SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")
PLANT_KEYS = ("mass", "length", "bob_radius", "damping", "gravity")
REWARD_KEYS = ("cosine_bound", "margin")
RESET_KEYS = ("angle_range", "velocity_noise")
# No joint stop and no failure set: (state index, limit) pairs for the _common helpers.
_LIMITS: Tuple[Tuple[int, float], ...] = ()
# dm_control's hinge height (rendering only; the hinge is fixed).
_HINGE_HEIGHT = 0.6

# dm_control's pendulum with the motor in N m (gear 1) and the bob's inertia compiled
# from its sphere geom. The pole turns about +y, so the bob sits at
# (l sin theta, h + l cos theta).
_PENDULUM_XML = """
<mujoco model="pendulum-swingup">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable" constraint="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light name="light" pos="0 0 2"/>
    <camera name="fixed" pos="0 -1.5 2" xyaxes="1 0 0 0 1 1"/>
    <body name="pole" pos="0 0 {hinge_height}">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{damping}" limited="false"/>
      <geom name="base" type="cylinder" fromto="0 -.03 0 0 .03 0" size="0.021" mass="0"/>
      <geom name="pole" type="capsule" fromto="0 0 0 0 0 {length}" size="0.02" mass="0"/>
      <geom name="mass" type="sphere" pos="0 0 {length}" size="{bob_radius}" mass="{mass}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="true"
           ctrlrange="{torque_low} {torque_high}"/>
  </actuator>
</mujoco>
"""


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def build_pendulum_model(config: TestbedConfig) -> MjModel:
    """dm_control's pendulum with the config's timestep, integrator and motor limits."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    return mujoco.MjModel.from_xml_string(
        _PENDULUM_XML.format(
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=plant["gravity"],
            hinge_height=_HINGE_HEIGHT,
            damping=plant["damping"],
            length=plant["length"],
            bob_radius=plant["bob_radius"],
            mass=plant["mass"],
            torque_low=float(controller["torque_low"][0]),
            torque_high=float(controller["torque_high"][0]),
        )
    )


def pendulum_chain(config: TestbedConfig) -> PlanarChain:
    """The one-link chain: the bob a solid sphere of radius bob_radius at the pole's end."""
    plant = config["plant_parameters"]
    return PlanarChain(
        link_masses=(plant["mass"],),
        link_com_distances=(plant["length"],),
        link_inertias=(0.4 * plant["mass"] * plant["bob_radius"] ** 2,),
        link_lengths=(plant["length"],),
        joint_damping=(plant["damping"],),
        input_matrix=[[1.0]],
        gravity=plant["gravity"],
    )


class PendulumSwingupEnv(MujocoPHEnv):
    """dm_control's pendulum swing-up: torque on the hinge, the angle wrapped, no noise.

    The state is (theta, theta_dot) with theta = 0 upright.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        require_keys(config["plant_parameters"], PLANT_KEYS, "Pendulum plant_parameters")
        require_keys(config["reward_parameters"], REWARD_KEYS, "Pendulum reward_parameters")
        require_keys(config["reset_parameters"], RESET_KEYS, "Pendulum reset_parameters")
        super().__init__(
            config,
            build_pendulum_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        self._chain = pendulum_chain(config)

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's swing-up start: the angle uniform on +-angle_range, at rest."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        bound = float(reset["angle_range"])
        angle = self._reset_rng.uniform(-bound, bound)
        velocity = float(reset["velocity_noise"]) * self._reset_rng.standard_normal()
        self._set_state(np.array([angle, velocity]))

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's tolerance at the config's margin (the row's 2, dm_control's 0);
        pendulum_reward_rate_tensor must stay its copy."""
        reward = self.config["reward_parameters"]
        return tolerance(
            math.cos(float(state[0])), (float(reward["cosine_bound"]), 1.0), float(reward["margin"])
        )

    def failed(self, state: ndarray) -> bool:
        """No failure set."""
        return limit_failed(state, _LIMITS)

    def terminal_payoff(self, state: ndarray) -> float:
        return failure_rate_payoff(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        return chain_drift_numpy(self._chain, state, np.asarray(torque, dtype=np.float64).reshape(-1))

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The pendulum has no noise."""
        return np.zeros((self.state_dim, 1))


def pendulum_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """tolerance(cos theta, (cosine_bound, 1), margin) on tensor batches, [B,1]."""
    reward = config["reward_parameters"]
    return tolerance_tensor(
        torch.cos(states[..., :1]), (float(reward["cosine_bound"]), 1.0), float(reward["margin"])
    )


def pendulum_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """No failure set: all False, [B]."""
    return limit_failure_mask(states, _LIMITS)


def pendulum_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure set: H = the payoff (0), B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), _LIMITS)


def pendulum_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """chain.chain_callbacks of the one-link PlanarChain; zero diffusion."""
    return chain_callbacks(pendulum_chain(config))


def pendulum_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    """The pole within tube of upright."""
    return abs(_wrap(float(state[0]))) <= tube


register_testbed(
    TestbedSpec(
        name=TESTBED,
        env_class=PendulumSwingupEnv,
        config_factory=with_clock_settings(pendulum_swingup_config),
        settings_keys=SETTINGS_KEYS,
        optional_settings_keys=CLOCK_SETTINGS_KEYS,
        reward_rate_tensor=pendulum_reward_rate_tensor,
        failure_mask_tensor=pendulum_failure_mask_tensor,
        terminal_payoff_value=failure_rate_payoff,
        terminal_boundary_functions=pendulum_terminal_boundary_functions,
        oracle_callbacks=pendulum_oracle_callbacks,
        upright=pendulum_upright,
        reward_rate_bounds=unit_interval_bounds,
        # The testbed_settings key discount_rate (_common.py, with_discount_rate).
        discount_setting=True,
    )
)
