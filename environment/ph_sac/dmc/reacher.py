"""dm_control's reacher/easy and reacher/hard as the testbeds reacher_easy and reacher_hard.

dm_control (suite/reacher.xml, reacher.py): a planar two-link arm in the horizontal plane
(hinges about +z, so gravity does no work), contact disabled. Arm capsule 0..0.12 m along
+x and hand capsule 0..0.10 m (radius 0.01, density 1000: masses 0.0418879 and
0.0356047 kg), a finger sphere (radius 0.01, 0.00418879 kg) in a jointless body 0.12 m
along the hand; damping 0.01 on both joints, both actuated with gear 0.05 and control in
[-1, 1] (torque +-0.05 N m); the wrist is limited to +-160 deg, the shoulder free; Euler
at 0.02 s. Reset (Reacher.initialize_episode): the shoulder uniform on [-pi, pi], the
wrist uniform within its range, velocities 0; then a target angle U[0, 2 pi) and radius
U[0.05, 0.20] with (x, y) = radius (sin angle, cos angle). Reward:
tolerance(|finger - target|, (0, rho)), margin 0, with rho the target plus finger
radius: 0.06 (easy), 0.025 (hard). 20 s episodes.

The port (rows reacher-easy, reacher-hard; factories in common/ph_dmc/reacher.py):
  state       (theta_s, theta_w, x_target, y_target, theta_dot_s, theta_dot_w):
              position_dim 4 = 2 physical + context_dim 2, velocity_dim 2, P = [I; 0];
              both angles periodic (wrapped in observations) and actuated; theta_w is
              the wrist angle relative to the arm, as MuJoCo's joint.
  context     the target (x*, y*), drawn in _reset_state from self._reset_rng with
              dm_control's law and order (shoulder, wrist, target angle, target radius)
              and written with _set_state; _apply_context moves the target geom so that
              rendering and dm_control comparisons line up. MuJoCo's dynamics never read
              it; options={"state": s} and snapshot/restore carry it.
  model       dm_control's geometry from plant_parameters at density 1000, so the
              compiled masses, centres of mass and inertias are dm_control's (the
              plant_parameters masses are dm_control's rounded values and are only
              checked to round to the compiled ones); the wrist KEEPS dm_control's soft
              joint limit (since 2026-11-10: limited, range +-wrist_range_degrees = 160
              deg written in degrees as dm_control's XML, MuJoCo's default solreflimit
              0.02 1 and solimplimit 0.9 0.95 0.001 0.5 2 from plant_parameters, the
              constraint solver on with its default Newton settings); motors in N m
              (gear 1, ctrlrange +-0.05); the configured step and integrator (1 ms RK4 in
              the table). MuJoCo floors the limit's time constant at 2 x timestep: 0.02 s
              at our 1 ms, but 0.04 s in dm_control's 20 ms model, so at dm_control's step
              the same compiled limit is softer (a departure of the physics step, not of
              the model).
  reward      finger = arm_length (cos theta_s, sin theta_s)
                       + finger_offset (cos(theta_s + theta_w), sin(theta_s + theta_w));
              r = tolerance(|finger - target|, (0, reach_radius), margin), the indicator
              of reaching the target when margin is 0, as dm_control's.
  failure     none (payoff failure_rate / lambda = 0 never charged, H = 0, B = 1): a
              wrist past +-160 deg meets dm_control's soft limit and continues, as in
              dm_control. (Until 2026-11-10 the stop was removed and |theta_w| >= 160 deg
              was a failure with payoff 0 and lift B = 1 - (theta_w / wrist_angle)^2.)
              The wrist stays periodic in the observation: the soft limit holds it within
              a few degrees of 160, far from the wrap at 180, so wrapped and unwrapped
              wrist agree and the oracle and the limit port read the observed value.
  oracle      limited_chain.limited_chain_callbacks(chain, limits, context_dim=2): the
              two-link PlanarChain read from the compiled model (planar_chain_from_model,
              the finger merged into the hand, gravity 0, input matrix eye(2) in N m) plus
              the wrist's soft-limit force (joint_limits, the compiled model's limit),
              exactly 0 within +-160 deg. The state spec declares the wrist as
              limited_position_ranges ((1, -w, w),) for the learned plant's limit port,
              which starts at this limit (limit_port_init, the factory's, from the model).
  upright     None: the reacher has no upright pose, so the evaluation counts no state as
              upright.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc._common import require_no_failure_set
from common.ph_dmc.reacher import (
    FAILURE_KEYS,
    PLANT_KEYS,
    RESET_KEYS,
    REWARD_KEYS,
    reacher_easy_config,
    reacher_hard_config,
)

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
from ._common import (
    NO_FAILURE,
    failure_rate_payoff,
    limit_attributes,
    limit_boundary_functions,
    limit_failure_mask,
    planar_chain_from_model,
    require_keys,
    tolerance,
    tolerance_tensor,
    unit_interval_bounds,
)
from .chain import PlanarChain
from .joint_limits import JointLimits, joint_limits_from_model
from .limited_chain import limited_chain_callbacks, limited_chain_drift_numpy

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")
LINK_BODIES = ("arm", "hand")
# The plant_parameters masses are dm_control's masses rounded to 1 g / 1000: the compiled
# geometric masses must lie within half a unit of the last digit of them.
MASS_ROUNDING = 5e-4

# dm_control's reacher.xml with its soft wrist limit (range in degrees, MuJoCo's default
# compiler angle, as dm_control writes it) and motors in N m. Geoms carry the masses at
# MuJoCo's default density 1000, as in dm_control; the finger stays a welded body on the
# hand. Gravity is MuJoCo's default along -z, normal to the plane of motion.
_REACHER_XML = """
<mujoco model="two-link planar reacher">
  <option timestep="{timestep}" integrator="{integrator}">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="hinge" axis="0 0 1" damping="{damping}" limited="false"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 10" rgba=".25 .3 .35 1"/>
    <geom name="root" type="cylinder" fromto="0 0 0 0 0 0.02" size=".011" rgba=".5 .5 .5 1"/>
    <body name="arm" pos="0 0 .01">
      <geom name="arm" type="capsule" fromto="0 0 0 {arm_length} 0 0" size="{link_radius}"
            rgba=".7 .5 .3 1"/>
      <joint name="shoulder"/>
      <body name="hand" pos="{arm_length} 0 0">
        <geom name="hand" type="capsule" fromto="0 0 0 {hand_length} 0 0" size="{link_radius}"
              rgba=".7 .5 .3 1"/>
        <joint name="wrist" limited="true" range="-{wrist_degrees} {wrist_degrees}" {limit}/>
        <body name="finger" pos="{finger_offset} 0 0">
          <geom name="finger" type="sphere" size="{finger_radius}" rgba=".8 .3 .2 1"/>
        </body>
      </body>
    </body>
    <geom name="target" pos="0 0 .01" type="sphere" size="{target_size}" rgba=".3 .8 .3 .5"/>
  </worldbody>
  <actuator>
    <motor name="shoulder" joint="shoulder" gear="1" ctrllimited="true"
           ctrlrange="{shoulder_low} {shoulder_high}"/>
    <motor name="wrist" joint="wrist" gear="1" ctrllimited="true"
           ctrlrange="{wrist_low} {wrist_high}"/>
  </actuator>
</mujoco>
"""


def _require_reacher_keys(config: TestbedConfig) -> None:
    require_keys(config["plant_parameters"], PLANT_KEYS, "Reacher plant_parameters")
    require_keys(config["failure_limits"], FAILURE_KEYS, "Reacher failure_limits")
    require_no_failure_set(config["name"], config["failure_limits"])
    require_keys(config["reward_parameters"], REWARD_KEYS, "Reacher reward_parameters")
    require_keys(config["reset_parameters"], RESET_KEYS, "Reacher reset_parameters")


def build_reacher_model(config: TestbedConfig) -> MjModel:
    """Compile dm_control's reacher geometry with its soft wrist limit, motors in N m.

    The target geom's radius is the reach radius less the finger radius (dm_control's
    target size); it only draws. Raises ValueError if the compiled masses do not round
    to the plant_parameters masses.
    """
    plant = config["plant_parameters"]
    controller = config["controller"]
    reward = config["reward_parameters"]
    target_size = float(reward["reach_radius"]) - float(plant["finger_radius"])
    model = mujoco.MjModel.from_xml_string(
        _REACHER_XML.format(
            timestep=float(config["physics_dt"]),
            integrator=config["integrator"],
            damping=float(plant["damping"]),
            arm_length=float(plant["arm_length"]),
            hand_length=float(plant["hand_length"]),
            finger_offset=float(plant["finger_offset"]),
            link_radius=float(plant["link_radius"]),
            finger_radius=float(plant["finger_radius"]),
            wrist_degrees=float(plant["wrist_range_degrees"]),
            limit=limit_attributes(plant),
            target_size=target_size if target_size > 0.0 else float(reward["reach_radius"]),
            shoulder_low=float(controller["torque_low"][0]),
            shoulder_high=float(controller["torque_high"][0]),
            wrist_low=float(controller["torque_low"][1]),
            wrist_high=float(controller["torque_high"][1]),
        )
    )
    for body, key in (("arm", "arm_mass"), ("hand", "hand_mass"), ("finger", "finger_mass")):
        compiled = float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])
        if abs(compiled - float(plant[key])) > MASS_ROUNDING:
            raise ValueError(
                f"Reacher plant_parameters {key}={plant[key]} does not round the {body}'s "
                f"compiled mass {compiled:.8f} kg (density 1000, as dm_control); the masses "
                "follow from the geometry and are not free parameters."
            )
    return model


def reacher_chain(model: object, config: TestbedConfig) -> PlanarChain:
    """The two-link PlanarChain of a compiled reacher (ours or dm_control's physics).

    The chain's "up" is +x and "side" +y; its tip is the finger, finger_offset along the
    hand. Its input matrix maps the model's ctrl: eye(2) for ours, 0.05 eye(2) for
    dm_control's.
    """
    return planar_chain_from_model(
        model,
        LINK_BODIES,
        along=0,
        joint_axis=2,
        last_length=float(config["plant_parameters"]["finger_offset"]),
    )


def reacher_wrist_limits(model: object) -> JointLimits:
    """The wrist's soft limit of a compiled reacher (ours or dm_control's)."""
    return joint_limits_from_model(model, ("wrist",))


def _finger_numpy(plant: Mapping[str, float], state: ndarray) -> ndarray:
    shoulder, wrist = float(state[0]), float(state[1])
    absolute = shoulder + wrist
    return np.array(
        [
            plant["arm_length"] * math.cos(shoulder) + plant["finger_offset"] * math.cos(absolute),
            plant["arm_length"] * math.sin(shoulder) + plant["finger_offset"] * math.sin(absolute),
        ]
    )


def reacher_finger_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """World (x, y) of the finger, [..., 2]."""
    plant = config["plant_parameters"]
    shoulder, wrist = states[..., 0:1], states[..., 1:2]
    absolute = shoulder + wrist
    return torch.cat(
        (
            plant["arm_length"] * torch.cos(shoulder) + plant["finger_offset"] * torch.cos(absolute),
            plant["arm_length"] * torch.sin(shoulder) + plant["finger_offset"] * torch.sin(absolute),
        ),
        dim=-1,
    )


class ReacherEnv(MujocoPHEnv):
    """dm_control's reacher (both target sizes) with the target as episode context.

    The state is (theta_s, theta_w, x_target, y_target, theta_dot_s, theta_dot_w).
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require_reacher_keys(config)
        super().__init__(
            config,
            build_reacher_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        if (self.physical_position_dim, self.context_dim, self.velocity_dim) != (2, 2, 2):
            raise ValueError("The reacher needs 2 physical positions, context_dim 2 and 2 velocities.")
        self._target_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self._chain = reacher_chain(self.model, config)
        self._limits = reacher_wrist_limits(self.model)

    @property
    def wrist_limits(self) -> JointLimits:
        """The wrist's soft limit, read from the compiled model."""
        return self._limits

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's reset: the joints, then the target angle and radius."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        rng = self._reset_rng
        shoulder = rng.uniform(-reset["shoulder_range"], reset["shoulder_range"])
        wrist = rng.uniform(-reset["wrist_range"], reset["wrist_range"])
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(reset["target_radius_low"], reset["target_radius_high"])
        velocities = reset["velocity_noise"] * rng.standard_normal(2)
        self._set_state(
            np.array(
                [
                    shoulder,
                    wrist,
                    radius * math.sin(angle),
                    radius * math.cos(angle),
                    velocities[0],
                    velocities[1],
                ]
            )
        )

    def _apply_context(self) -> None:
        """Move the target geom to the context (it is a world geom: no dynamics)."""
        geom = getattr(self, "_target_geom", None)
        if geom is None:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.model.geom_pos[geom, :2] = self._context
        self.data.geom_xpos[geom, :2] = self._context

    @property
    def target(self) -> ndarray:
        """The episode's target (x*, y*), the context."""
        return self.context

    def finger(self, state: ndarray) -> ndarray:
        """World (x, y) of the finger in a state."""
        return _finger_numpy(self.config["plant_parameters"], state)

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's reach reward; reacher_reward_rate_tensor must stay its copy."""
        reward = self.config["reward_parameters"]
        distance = float(np.linalg.norm(self.finger(state) - np.asarray(state[2:4], dtype=np.float64)))
        return float(
            tolerance(distance, (0.0, float(reward["reach_radius"])), float(reward["margin"]))
        )

    def failed(self, state: ndarray) -> bool:
        """Never: a wrist past +-160 deg meets dm_control's soft limit and continues."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return failure_rate_payoff(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """The chain, plus the wrist's soft-limit force beyond +-160 deg."""
        return limited_chain_drift_numpy(self._chain, self._limits, state, torque, context_dim=2)

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The reacher has no noise."""
        return np.zeros((self.state_dim, 1))


def reacher_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """The reach reward from the angles and the context target, [B,1]."""
    reward = config["reward_parameters"]
    offset = reacher_finger_tensor(states, config) - states[..., 2:4]
    distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    return tolerance_tensor(
        distance, (0.0, float(reward["reach_radius"])), float(reward["margin"])
    )


def reacher_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """All False, [B]: no failure set (the wrist limit is a soft constraint)."""
    return limit_failure_mask(states, NO_FAILURE)


def reacher_terminal_payoff_value(config: TestbedConfig) -> float:
    """failure_rate / lambda (0 in the table)."""
    return failure_rate_payoff(config)


def reacher_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure boundary: H = failure_rate / lambda (0) and B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), NO_FAILURE)


def reacher_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The compiled model's chain with its soft wrist limit, zero rate on the context."""
    model = build_reacher_model(config)
    return limited_chain_callbacks(
        reacher_chain(model, config), reacher_wrist_limits(model), context_dim=2
    )


for _name, _factory in (
    ("reacher_easy", reacher_easy_config),
    ("reacher_hard", reacher_hard_config),
):
    register_testbed(
        TestbedSpec(
            name=_name,
            env_class=ReacherEnv,
            config_factory=_factory,
            settings_keys=SETTINGS_KEYS,
            reward_rate_tensor=reacher_reward_rate_tensor,
            failure_mask_tensor=reacher_failure_mask_tensor,
            terminal_payoff_value=reacher_terminal_payoff_value,
            terminal_boundary_functions=reacher_terminal_boundary_functions,
            oracle_callbacks=reacher_oracle_callbacks,
            upright=None,
            reward_rate_bounds=unit_interval_bounds,
        )
    )
