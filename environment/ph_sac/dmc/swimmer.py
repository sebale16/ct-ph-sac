"""dm_control's swimmer/swimmer6 and swimmer/swimmer15 as swimmer_swimmer6 and swimmer_swimmer15.

dm_control (suite/swimmer.py, swimmer.xml): a planar chain of n bodies (the head plus
n - 1 segments, n = 6 or 15) in the horizontal plane, contact disabled, in a fluid of
density 3000 and viscosity 0 with gravity normal to the plane. Each body carries an
invisible inertial box geom of half-sizes (0.001, 0.05, 0.01) m and mass 0.01 kg (the
visual capsules, the head ellipsoid, the nose and the eyes have mass 0), centred 0.05 m
behind its joint, the next joint 0.1 m behind; the head's root is planar and free (slides
rootx, rooty along world x and y and the hinge rootz about z, all through the point
0.05 m ahead of the head's centre, unlimited, no armature), the n - 1 hinges joint_i
about z have armature 1e-6, no damping and are limited to +-360/n deg by a soft limit
constraint (solreflimit 0.05 1, solimplimit 0 0.8 0.1); n - 1 motors of gear 5e-4 and
control in [-1, 1] (torque +-5e-4 N m). Euler at 2 ms, control step 30 ms (15 substeps),
30 s episodes. The fluid force is MuJoCo's inertia-box model (mj_passive, body by body):
the body's equivalent box of the compiled inertia, full sizes
b_k = sqrt(6 (I_{k+1} + I_{k+2} - I_k) / m) = 2 x the box geom's half-sizes, gets the
quadratic drag f_k = -rho/2 b_{k+1} b_{k+2} |v_k| v_k of its centre-of-mass velocity and
tau_k = -rho/64 b_k (b_{k+1}^4 + b_{k+2}^4) |w_k| w_k of its angular velocity, in its
principal frame (and the viscous terms when viscosity > 0). Reset
(randomize_limited_and_rotational_joints, then Swimmer.initialize_episode): root at the
origin, yaw uniform on [-pi, pi], each joint uniform within its range, velocities 0;
then the target uniform in a +-0.3 m box with probability 0.2, else in a +-2 m box.
Reward: tolerance(|nose - target|, (0, 0.1), margin 0.5, sigmoid long_tail),
1 / (1 + 9 (d / 0.5)^2) outside the target, d the distance beyond 0.1 m.

The port (rows swimmer-swimmer6, swimmer-swimmer15; factories in common/ph_dmc/swimmer.py):
  state       (x, y, psi, theta_1 .. theta_{n-1}, x*, y*, rates of the first n + 2):
              MuJoCo's qpos (the slides, the root yaw, the joint angles relative to the
              previous body) and qvel, position_dim n + 4 = n + 2 physical + context_dim 2,
              velocity_dim n + 2, P = [I; 0]. psi is periodic (wrapped in observations,
              sine and cosine in the networks); x and y are travel coordinates (raw); the
              joints are actuated (offset setpoints, unwrapped errors: a joint passes its
              range by at most a fraction of a radian before its soft limit turns it).
  context     the target (x*, y*), drawn in _reset_state from self._reset_rng with
              dm_control's law and order (yaw, joints from the head back, the near/far
              coin, x*, y*) and written with _set_state; _apply_context moves the target
              geom (rendering only). The reward reads the absolute nose position, a known
              kinematic function of (x, y, psi), against the absolute target, so neither
              moves the other; MuJoCo's dynamics never read the context.
  cyclic      x, y and the context (cyclic_position_indices): no force depends on them,
              so the learned plant leaves them out of its features (PortHamiltonianPlant)
              and fits forces that cannot depend on where the swimmer is. M(q) and the
              fluid drag in world-frame velocities depend on psi and the joint angles.
  model       dm_control's swimmer.xml at the configured step and integrator (the table:
              dm_control's own 2 ms Euler since 2026-11-10, 1 ms RK4 before), with the
              motors in N m (gear 1, ctrlrange +-5e-4). No joint has damping
              (dof_damping 0), so MuJoCo's Euler step is the plain velocity-first
              semi-implicit step, v' = v + h qacc(s), q' = q + h v', with the fluid force
              and the limit forces of qacc evaluated at the step's start: the rows' model
              integrator semi_implicit at their 2 ms model step is that map, and under
              refresh_mode "continuous" the feedback law is evaluated once per step, at
              its start, in the environment (the held path, refreshed every step) and in
              the model alike. The joint
              limits are KEPT as dm_control has them (limited, range 360 / n deg,
              solreflimit 0.05 1, solimplimit 0 0.8 0.1, MuJoCo's default Newton solver),
              so the dynamics are dm_control's, limits included; no departure there.
  failure     none (payoff 0, B = 1): a joint past its range meets the soft limit and
              continues, as in dm_control. The limits are the one non-smooth force: zero
              inside the range, a one-sided spring-damper-and-added-inertia beyond it
              whose impedance ramps up over the first 0.1 rad (joint_limits.py).
  reward      nose = (x, y) + head_anchor + R(psi) (0, -nose_offset), the nose sphere
              0.01 m ahead of the root hinge (head_anchor (0, -0.05) is the root hinge's
              offset from the head body); r = tolerance(|nose - target|,
              (0, target_radius), margin, long_tail): dm_control's, with target_radius 0.1
              and margin 0.5 = 5 target sizes.
  oracle      SwimmerDynamics read from the compiled model: Lagrange's equations of the
              planar chain on a free planar base in absolute coordinates
              z = (x, y, phi_0 .. phi_{n-1}) = T q (phi_0 = psi, phi_i = psi + theta_1 ..
              theta_i), M(q) = T^T M_z(phi) T + diag(armature), the centrifugal terms, the
              inertia-box fluid force above as the generalized force J^T f of each body,
              and the input matrix eye on the joints (N m), then MuJoCo's soft
              joint-limit forces (joint_limits.constrained_accelerations: impedance,
              reference acceleration, regularizer, and the one-sided complementarity of
              several active limits); torch.func-compatible (Cholesky solves, fixed
              pivoting passes); zero diffusion. The fluid force is -D_fl(q, v) v with
              D_fl = sum_i J_i^T diag(c |J_i v|) J_i positive semidefinite and full rank
              in v, linear in |v|: the learned plant needs network.damping_velocity. The
              limits need network.limit_port (a learned one-sided spring and damper at
              the known ranges, limited_position_ranges, starting at the compiled
              limit's stiffness, damping and width, limit_port_init, which the factory
              reads from build_swimmer_model); the rows set both.
  upright     None: the swimmer has no upright pose, so the evaluation counts no state as
              upright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc.swimmer import (
    CONTEXT_DIM,
    FAILURE_KEYS,
    PLANT_KEYS,
    RESET_KEYS,
    REWARD_KEYS,
    swimmer_joint_range,
    swimmer_swimmer6_config,
    swimmer_swimmer15_config,
)

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
from ._common import (
    body_inertials,
    failure_rate_payoff,
    limit_boundary_functions,
    limit_failure_mask,
    raw_model,
    require_keys,
    tolerance,
    tolerance_tensor,
    unit_interval_bounds,
    zero_diffusion,
)
from .joint_limits import (
    JointLimits,
    constrained_accelerations,
    constrained_accelerations_numpy,
    joint_limits_from_model,
)

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")
REWARD_SIGMOID = "long_tail"  # dm_control's Swimmer.get_reward
# MuJoCo's mjMINVAL, the floor of the inertia-box model's squared box sizes.
_MJ_MINVAL = 1e-15

# dm_control's swimmer.xml with the motors in N m (its joint limits kept). Geoms and
# masses as dm_control's: only the inertial boxes carry mass; the visual geoms draw. No
# joint sets damping, as in dm_control (MuJoCo's Euler then has no implicit damping term).
_SWIMMER_XML = """
<mujoco model="swimmer">
  <option timestep="{timestep}" integrator="{integrator}" density="{density}"
          viscosity="{viscosity}">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <geom contype="0" conaffinity="0"/>
    <default class="swimmer">
      <joint type="hinge" pos="0 {joint_y} 0" axis="0 0 1" limited="true"
             solreflimit="{solref}" solimplimit="{solimp}" armature="{armature}"/>
      <default class="inertial">
        <geom type="box" size="{half_width} {half_length} {half_height}" rgba="0 0 0 0"
              mass="{link_mass}"/>
      </default>
      <default class="visual">
        <geom type="capsule" size=".01" fromto="0 {joint_y} 0 0 {half_length} 0"
              rgba=".7 .5 .3 1" mass="0"/>
      </default>
      <site size=".01" rgba="0 0 0 0"/>
    </default>
    <default class="free">
      <joint limited="false" stiffness="0" armature="0"/>
    </default>
  </default>
  <worldbody>
    <light name="light" pos="0 0 4"/>
    <camera name="fixed" pos="0 0 {camera_height}" quat="1 0 0 0"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba=".25 .3 .35 1"/>
    <body name="head" pos="0 0 .05" childclass="swimmer">
      <geom name="head" type="ellipsoid" size=".02 .04 .017" pos="0 -.022 0"
            rgba=".7 .5 .3 1" mass="0"/>
      <geom name="nose" type="sphere" pos="0 {nose_y} 0" size=".004" rgba=".8 .3 .2 1"
            mass="0"/>
      <geom name="eyes" type="capsule" fromto="-.006 -.054 .005 .006 -.054 .005"
            size=".004" rgba=".1 .1 .1 1" mass="0"/>
      <camera name="tracking1" pos="0 -.2 {tracking_height}" xyaxes="1 0 0 0 1 1"
              mode="trackcom" fovy="60"/>
      <joint name="rootx" class="free" type="slide" axis="1 0 0" pos="0 {joint_y} 0"/>
      <joint name="rooty" class="free" type="slide" axis="0 1 0" pos="0 {joint_y} 0"/>
      <joint name="rootz" class="free" type="hinge" axis="0 0 1" pos="0 {joint_y} 0"/>
      <geom name="inertial" class="inertial"/>
      <geom name="visual" class="visual"/>
      <site name="head"/>
{segments}
    </body>
    <geom name="target" type="sphere" pos="1 1 .05" size="{target_radius}"
          rgba=".3 .8 .3 .5"/>
  </worldbody>
  <actuator>
{motors}
  </actuator>
  <sensor>
    <framepos name="nose_pos" objtype="geom" objname="nose"/>
  </sensor>
</mujoco>
"""


def _require_swimmer_keys(config: TestbedConfig) -> None:
    require_keys(config["plant_parameters"], PLANT_KEYS, "Swimmer plant_parameters")
    require_keys(config["failure_limits"], FAILURE_KEYS, "Swimmer failure_limits")
    require_keys(config["reward_parameters"], REWARD_KEYS, "Swimmer reward_parameters")
    require_keys(config["reset_parameters"], RESET_KEYS, "Swimmer reset_parameters")


def build_swimmer_model(config: TestbedConfig) -> MjModel:
    """Compile dm_control's swimmer of n_links bodies, motors in N m.

    The joints keep dm_control's soft limits (range 360 / n deg, solreflimit and
    solimplimit from plant_parameters); MuJoCo's default solver (Newton, 100
    iterations, tolerance 1e-8) solves them, as in dm_control.
    """
    plant = config["plant_parameters"]
    controller = config["controller"]
    n_links = int(plant["n_links"])
    half_length = 0.5 * float(plant["link_length"])
    # dm_control writes the range in degrees, '{} {}'.format(-360.0 / n, 360.0 / n), and
    # MuJoCo converts it; the same text compiles to the same jnt_range.
    limit = 360.0 / n_links
    joint_range = "{} {}".format(-limit, limit)
    lines, parent_indent = [], "      "
    for index in range(n_links - 1):
        indent = parent_indent + "  " * index
        lines.append(f'{indent}<body name="segment_{index}" pos="0 {2.0 * half_length!r} 0">')
        lines.append(f'{indent}  <geom class="visual" name="visual_{index}"/>')
        lines.append(f'{indent}  <geom class="inertial" name="inertial_{index}"/>')
        lines.append(f'{indent}  <site name="site_{index}"/>')
        lines.append(f'{indent}  <joint name="joint_{index}" range="{joint_range}"/>')
    for index in reversed(range(n_links - 1)):
        lines.append(parent_indent + "  " * index + "</body>")
    motors = [
        f'    <motor name="motor_{index}" joint="joint_{index}" gear="1" ctrllimited="true" '
        f'ctrlrange="{float(controller["torque_low"][index])!r} '
        f'{float(controller["torque_high"][index])!r}"/>'
        for index in range(n_links - 1)
    ]
    scale = n_links / 6.0
    solref = f'{float(plant["limit_timeconst"])!r} {float(plant["limit_dampratio"])!r}'
    solimp = " ".join(
        repr(float(plant[key]))
        for key in ("limit_d0", "limit_dmax", "limit_width", "limit_midpoint", "limit_power")
    )
    return mujoco.MjModel.from_xml_string(
        _SWIMMER_XML.format(
            timestep=float(config["physics_dt"]),
            integrator=config["integrator"],
            density=float(plant["density"]),
            viscosity=float(plant["viscosity"]),
            joint_y=repr(-half_length),
            half_width=repr(float(plant["box_half_width"])),
            half_length=repr(half_length),
            half_height=repr(float(plant["box_half_height"])),
            link_mass=repr(float(plant["link_mass"])),
            armature=repr(float(plant["armature"])),
            solref=solref,
            solimp=solimp,
            nose_y=repr(-half_length - float(plant["nose_offset"])),
            target_radius=float(config["reward_parameters"]["target_radius"]),
            camera_height=3.0 * scale,
            tracking_height=0.5 * scale,
            segments="\n".join(lines),
            motors="\n".join(motors),
        )
    )


def link_body_names(n_links: int) -> Tuple[str, ...]:
    """The bodies from the head back: head, segment_0 .. segment_{n-2}."""
    return ("head",) + tuple(f"segment_{index}" for index in range(int(n_links) - 1))


def _rotation(quaternion: ndarray) -> ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


@dataclass(frozen=True, eq=False)
class SwimmerDynamics:
    """Constants of the swimmer's planar dynamics, read from a compiled model.

    Coordinates q = (x, y, psi, theta_1 .. theta_{n-1}) (nv = n + 2) and absolute
    z = T q = (x, y, phi_0 .. phi_{n-1}). Body i points along u_i = (-sin phi_i,
    cos phi_i) (its local +y) with local +x e_i = (cos phi_i, sin phi_i); its hinge
    (the root hinge for the head) sits at A_i, A_{i+1} = A_i + lengths[i] u_i, and its
    centre of mass at A_i + com[i] u_i, so the centre of mass of body i moves with
    xdot_b - sum_j W_ij phidot_j e_j (W_ij = lengths[j] for j < i, com[i] for j = i).

    coupling (K) and moments (h) are the link block K_jk cos(phi_j - phi_k) of the mass
    matrix and the base coupling -h_j e_j, as chain.PlanarChain's. drag_x, drag_y and
    drag_spin are the inertia-box model's quadratic coefficients per body along its local
    x and y and about z; viscous_linear and viscous_spin its linear (viscosity) ones.
    input_matrix maps the model's ctrl to forces in q. limits are the joints' soft limits
    (joint_limits.JointLimits), None for a model without limited joints.
    """

    masses: np.ndarray
    inertias: np.ndarray
    com: np.ndarray
    lengths: np.ndarray
    armature: np.ndarray
    drag_x: np.ndarray
    drag_y: np.ndarray
    drag_spin: np.ndarray
    viscous_linear: np.ndarray
    viscous_spin: np.ndarray
    input_matrix: np.ndarray
    head_anchor: np.ndarray
    nose_offset: float
    limits: Optional[JointLimits] = None

    def __post_init__(self) -> None:
        n = len(self.masses)
        beyond = np.concatenate((np.cumsum(self.masses[::-1])[::-1][1:], [0.0]))
        moments = self.masses * self.com + self.lengths * beyond
        coupling = np.empty((n, n))
        for j in range(n):
            for k in range(n):
                if j == k:
                    coupling[j, k] = (
                        self.inertias[j] + self.masses[j] * self.com[j] ** 2
                        + self.lengths[j] ** 2 * beyond[j]
                    )
                else:
                    coupling[j, k] = self.lengths[min(j, k)] * moments[max(j, k)]
        lever = np.zeros((n, n))
        for i in range(n):
            lever[i, :i] = self.lengths[:i]
            lever[i, i] = self.com[i]
        transform = np.zeros((n + 2, n + 2))
        transform[:2, :2] = np.eye(2)
        transform[2:, 2:] = np.tril(np.ones((n, n)))
        object.__setattr__(self, "moments", moments)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "lever", lever)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "total_mass", float(np.sum(self.masses)))

    @property
    def n_links(self) -> int:
        return len(self.masses)

    @property
    def nv(self) -> int:
        return self.n_links + 2


def swimmer_dynamics(model: object, n_links: int) -> SwimmerDynamics:
    """SwimmerDynamics of a compiled swimmer: ours, or dm_control's physics.

    Reads the bodies' masses, centres of mass, inertias, principal frames, joint anchors,
    armature, the fluid's density and viscosity, the nose geom and the motors' gears
    (the input matrix maps the model's ctrl: eye on the joints for ours, 5e-4 eye for
    dm_control's). Raises ValueError if the model is not a planar swimmer chain.
    """
    model = raw_model(model)
    names = link_body_names(n_links)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    if min(ids) < 0:
        raise ValueError(f"The model is not a swimmer of {n_links} links.")
    inertials = body_inertials(model, names, include_welded=False)
    hinge = {}
    for name, body in zip(names, ids):
        joints = range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body])
        hinges = [j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if len(hinges) != 1 or not np.allclose(model.jnt_axis[hinges[0]], [0.0, 0.0, 1.0]):
            raise ValueError(f"Body {name!r} needs exactly one hinge about +z.")
        if not np.allclose(model.body_quat[body], [1.0, 0.0, 0.0, 0.0]):
            raise ValueError(f"Body {name!r} must not be rotated relative to its parent.")
        hinge[name] = hinges[0]
    density, viscosity = float(model.opt.density), float(model.opt.viscosity)
    masses, inertias, com, lengths = [], [], [], []
    drag_x, drag_y, drag_spin, viscous_linear, viscous_spin = [], [], [], [], []
    for position, (name, body) in enumerate(zip(names, ids)):
        anchor = model.jnt_pos[hinge[name]]
        inertial = inertials[name]
        if abs(inertial.com[0] - anchor[0]) > 1e-12:
            raise ValueError(f"The centre of mass of {name!r} must lie on its link line.")
        masses.append(inertial.mass)
        inertias.append(float(inertial.inertia[2, 2]))
        com.append(float(inertial.com[1] - anchor[1]))
        if position + 1 < n_links:
            child = ids[position + 1]
            offset = model.body_pos[child] + model.jnt_pos[hinge[names[position + 1]]] - anchor
            if abs(offset[0]) > 1e-12:
                raise ValueError(f"The joint after {name!r} must sit on its link line.")
            lengths.append(float(offset[1]))
        else:
            lengths.append(0.0)
        # MuJoCo's inertia-box fluid model of this body (mj_inertiaBoxFluidModel).
        mass, moment = float(model.body_mass[body]), np.asarray(model.body_inertia[body])
        box = np.array(
            [
                math.sqrt(max(_MJ_MINVAL, moment[(k + 1) % 3] + moment[(k + 2) % 3] - moment[k]) / mass * 6.0)
                for k in range(3)
            ]
        )
        axes = _rotation(model.body_iquat[body])  # columns: principal axes in the body frame
        linear = [0.5 * density * box[(k + 1) % 3] * box[(k + 2) % 3] for k in range(3)]
        spin = [
            density * box[k] * (box[(k + 1) % 3] ** 4 + box[(k + 2) % 3] ** 4) / 64.0
            for k in range(3)
        ]
        along = {}
        for k in range(3):
            direction = np.abs(axes[:, k])
            local = int(np.argmax(direction))
            if abs(direction[local] - 1.0) > 1e-9:
                raise ValueError(f"The principal axes of {name!r} must be its body axes.")
            along[local] = k
        diameter = float(np.mean(box))
        drag_x.append(linear[along[0]])
        drag_y.append(linear[along[1]])
        drag_spin.append(spin[along[2]])
        viscous_linear.append(3.0 * math.pi * diameter * viscosity)
        viscous_spin.append(math.pi * diameter**3 * viscosity)
    dofs = [int(model.jnt_dofadr[j]) for j in range(model.njnt)]
    if model.nv != n_links + 2 or dofs != list(range(n_links + 2)):
        raise ValueError("The swimmer's dofs must be rootx, rooty, rootz, joint_0 ...")
    input_matrix = np.zeros((model.nv, model.nu))
    for actuator in range(model.nu):
        dof = int(model.jnt_dofadr[model.actuator_trnid[actuator, 0]])
        input_matrix[dof, actuator] = float(model.actuator_gear[actuator, 0])
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, hinge[name]) for name in names[1:]]
    limited = [bool(model.jnt_limited[hinge[name]]) for name in names[1:]]
    if any(limited) and not all(limited):
        raise ValueError("Either every swimmer joint is limited or none is.")
    if all(limited) and limited and model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT:
        raise ValueError("The swimmer's limits need the constraint solver enabled.")
    limits = joint_limits_from_model(model, joints) if all(limited) and limited else None
    head = ids[0]
    nose = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "nose")
    root_anchor = model.jnt_pos[hinge["head"]]
    nose_local = model.geom_pos[nose] - root_anchor
    if abs(nose_local[0]) > 1e-12 or model.geom_bodyid[nose] != head:
        raise ValueError("The nose must sit on the head's line.")
    return SwimmerDynamics(
        masses=np.array(masses),
        inertias=np.array(inertias),
        com=np.array(com),
        lengths=np.array(lengths),
        armature=np.array(model.dof_armature[: model.nv], dtype=np.float64),
        drag_x=np.array(drag_x),
        drag_y=np.array(drag_y),
        drag_spin=np.array(drag_spin),
        viscous_linear=np.array(viscous_linear),
        viscous_spin=np.array(viscous_spin),
        input_matrix=input_matrix,
        head_anchor=np.asarray(model.body_pos[head][:2] + root_anchor[:2], dtype=np.float64),
        nose_offset=float(-nose_local[1]),
        limits=limits,
    )


# ------------------------------------------------------------------------------- torch


def _constant(array: np.ndarray, like: Tensor) -> Tensor:
    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def swimmer_terms(
    dynamics: SwimmerDynamics, positions: Tensor, velocities: Tensor
) -> Dict[str, Tensor]:
    """The terms of M(q) qdd = G u - bias + fluid in q: "mass" [..., nv, nv], "bias" and
    "fluid" [..., nv] (MuJoCo's qM, qfrc_bias and qfrc_fluid).

    bias = T^T c_z with c_z the centrifugal terms in absolute coordinates, fluid = T^T f_z
    with f_z the inertia-box fluid force J^T f summed over the bodies; torch.func-
    compatible (no in-place writes).
    """
    phi = torch.cumsum(positions[..., 2:], dim=-1)
    rates = torch.cumsum(velocities[..., 2:], dim=-1)
    cos, sin = phi.cos(), phi.sin()
    base = velocities[..., :2]
    coupling = _constant(dynamics.coupling, positions)
    moments = _constant(dynamics.moments, positions)
    lever = _constant(dynamics.lever, positions)
    transform = _constant(dynamics.transform, positions)

    # cos(phi_j - phi_k) and sin(phi_j - phi_k), [..., n, n] (row j, column k).
    cos_difference = cos.unsqueeze(-1) * cos.unsqueeze(-2) + sin.unsqueeze(-1) * sin.unsqueeze(-2)
    sin_difference = sin.unsqueeze(-1) * cos.unsqueeze(-2) - cos.unsqueeze(-1) * sin.unsqueeze(-2)

    # Mass matrix in z: total mass on the base, -h_j e_j coupling, K cos on the links.
    total = torch.ones_like(cos[..., :1]) * dynamics.total_mass
    zero = torch.zeros_like(total)
    base_block = torch.stack(
        (torch.cat((total, zero), dim=-1), torch.cat((zero, total), dim=-1)), dim=-2
    )
    coupling_block = -torch.stack((moments * cos, moments * sin), dim=-2)  # [..., 2, n]
    top = torch.cat((base_block, coupling_block), dim=-1)
    bottom = torch.cat((coupling_block.mT, coupling * cos_difference), dim=-1)
    mass = transform.mT @ torch.cat((top, bottom), dim=-2) @ transform
    mass = mass + torch.diag(_constant(dynamics.armature, positions))

    # Centrifugal terms: base sum_j h_j phidot_j^2 (sin phi_j, -cos phi_j); links
    # sum_k K_jk sin(phi_j - phi_k) phidot_k^2.
    squares = rates.square()
    weighted = moments * squares
    bias = torch.cat(
        (
            (weighted * sin).sum(-1, keepdim=True),
            -(weighted * cos).sum(-1, keepdim=True),
            (coupling * sin_difference * squares.unsqueeze(-2)).sum(-1),
        ),
        dim=-1,
    )

    # Fluid: body i's centre-of-mass velocity in its local (x, y) frame; with
    # cos(phi_k - phi_i) = cos_difference[i, k] and sin(phi_k - phi_i) = -sin_difference[i, k].
    lever_cos = lever * cos_difference
    lever_sin = -lever * sin_difference
    local_x = (
        base[..., :1] * cos + base[..., 1:] * sin
        - (lever_cos @ rates.unsqueeze(-1)).squeeze(-1)
    )
    local_y = (
        -base[..., :1] * sin + base[..., 1:] * cos
        - (lever_sin @ rates.unsqueeze(-1)).squeeze(-1)
    )
    viscous_linear = _constant(dynamics.viscous_linear, positions)
    force_x = -_constant(dynamics.drag_x, positions) * local_x.abs() * local_x - viscous_linear * local_x
    force_y = -_constant(dynamics.drag_y, positions) * local_y.abs() * local_y - viscous_linear * local_y
    spin = (
        -_constant(dynamics.drag_spin, positions) * rates.abs() * rates
        - _constant(dynamics.viscous_spin, positions) * rates
    )
    fluid = torch.cat(
        (
            (force_x * cos - force_y * sin).sum(-1, keepdim=True),
            (force_x * sin + force_y * cos).sum(-1, keepdim=True),
            -(lever_cos.mT @ force_x.unsqueeze(-1)).squeeze(-1)
            - (lever_sin.mT @ force_y.unsqueeze(-1)).squeeze(-1)
            + spin,
        ),
        dim=-1,
    )
    return {"mass": mass, "bias": bias @ transform, "fluid": fluid @ transform}


def swimmer_accelerations(
    dynamics: SwimmerDynamics, positions: Tensor, velocities: Tensor, torques: Tensor
) -> Tensor:
    """qdd [..., nv]: M(q) qdd = G u - bias + fluid (swimmer_terms) by Cholesky, then
    MuJoCo's soft joint-limit forces (joint_limits.constrained_accelerations)."""
    terms = swimmer_terms(dynamics, positions, velocities)
    applied = torques.to(positions.dtype) @ _constant(dynamics.input_matrix, positions).T
    forces = applied + terms["fluid"] - terms["bias"]
    factor = torch.linalg.cholesky(terms["mass"])
    smooth = torch.cholesky_solve(forces.unsqueeze(-1), factor).squeeze(-1)
    if dynamics.limits is None:
        return smooth
    return constrained_accelerations(dynamics.limits, factor, smooth, positions, velocities)["qacc"]


def swimmer_drift(
    dynamics: SwimmerDynamics, states: Tensor, torques: Tensor, *, context_dim: int = CONTEXT_DIM
) -> Tensor:
    """(qd, 0, qdd) [..., D] for states (q, context, qd), D = 2 nv + context_dim."""
    nv = dynamics.nv
    if states.shape[-1] != 2 * nv + context_dim:
        raise ValueError(f"Swimmer states need {2 * nv + context_dim} entries, not {states.shape[-1]}.")
    positions = states[..., :nv]
    velocities = states[..., nv + context_dim :]
    accelerations = swimmer_accelerations(dynamics, positions, velocities, torques)
    parts = [velocities]
    if context_dim:
        parts.append(torch.zeros_like(states[..., nv : nv + context_dim]))
    parts.append(accelerations)
    return torch.cat(parts, dim=-1)


def swimmer_nose_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """World (x, y) of the nose, [..., 2]: (x, y) + head anchor + R(psi) (0, -nose_offset)."""
    plant = config["plant_parameters"]
    offset = float(plant["nose_offset"])
    anchor_y = -0.5 * float(plant["link_length"])
    yaw = states[..., 2:3]
    return torch.cat(
        (states[..., 0:1] + offset * torch.sin(yaw), states[..., 1:2] + anchor_y - offset * torch.cos(yaw)),
        dim=-1,
    )


# ------------------------------------------------------------------------------- numpy


def swimmer_accelerations_numpy(
    dynamics: SwimmerDynamics, positions: ndarray, velocities: ndarray, torques: ndarray
) -> ndarray:
    """The numpy twin of swimmer_accelerations, written from the same equations."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    torques = np.asarray(torques, dtype=np.float64)
    n, nv = dynamics.n_links, dynamics.nv
    phi = np.cumsum(positions[..., 2:], axis=-1)
    rates = np.cumsum(velocities[..., 2:], axis=-1)
    cos, sin = np.cos(phi), np.sin(phi)
    difference = phi[..., :, None] - phi[..., None, :]
    mass_z = np.zeros(positions.shape[:-1] + (nv, nv))
    mass_z[..., 0, 0] = mass_z[..., 1, 1] = dynamics.total_mass
    mass_z[..., 0, 2:] = -dynamics.moments * cos
    mass_z[..., 1, 2:] = -dynamics.moments * sin
    mass_z[..., 2:, 0] = mass_z[..., 0, 2:]
    mass_z[..., 2:, 1] = mass_z[..., 1, 2:]
    mass_z[..., 2:, 2:] = dynamics.coupling * np.cos(difference)
    transform = dynamics.transform
    mass = transform.T @ mass_z @ transform + np.diag(dynamics.armature)

    squares = rates**2
    bias = np.zeros(positions.shape[:-1] + (nv,))
    bias[..., 0] = np.sum(dynamics.moments * squares * sin, axis=-1)
    bias[..., 1] = -np.sum(dynamics.moments * squares * cos, axis=-1)
    bias[..., 2:] = np.sum(dynamics.coupling * np.sin(difference) * squares[..., None, :], axis=-1)

    # Centre-of-mass velocity of body i: xdot_b - sum_j W_ij phidot_j e_j, then local axes.
    lever = dynamics.lever
    com_velocity = velocities[..., None, :2] - np.einsum(
        "ij,...j,...jc->...ic", lever, rates, np.stack((cos, sin), axis=-1)
    )
    axis_x = np.stack((cos, sin), axis=-1)
    axis_y = np.stack((-sin, cos), axis=-1)
    local_x = np.sum(com_velocity * axis_x, axis=-1)
    local_y = np.sum(com_velocity * axis_y, axis=-1)
    force_x = -dynamics.drag_x * np.abs(local_x) * local_x - dynamics.viscous_linear * local_x
    force_y = -dynamics.drag_y * np.abs(local_y) * local_y - dynamics.viscous_linear * local_y
    spin = -dynamics.drag_spin * np.abs(rates) * rates - dynamics.viscous_spin * rates
    world_force = force_x[..., None] * axis_x + force_y[..., None] * axis_y  # [..., n, 2]
    # Generalized force in z: J_i^T f_i with dp_i/dxb = I and dp_i/dphi_j = -W_ij e_j.
    fluid = np.zeros(positions.shape[:-1] + (nv,))
    fluid[..., :2] = np.sum(world_force, axis=-2)
    fluid[..., 2:] = -np.einsum("ij,...ic,...jc->...j", lever, world_force, axis_x) + spin

    forces = torques @ dynamics.input_matrix.T + (fluid - bias) @ transform
    smooth = np.linalg.solve(mass, forces[..., None])[..., 0]
    if dynamics.limits is None:
        return smooth
    flat = smooth.reshape(-1, nv)
    masses = np.broadcast_to(mass, smooth.shape + (nv,)).reshape(-1, nv, nv)
    q = np.broadcast_to(positions, smooth.shape).reshape(-1, nv)
    v = np.broadcast_to(velocities, smooth.shape).reshape(-1, nv)
    constrained = [
        constrained_accelerations_numpy(dynamics.limits, masses[i], flat[i], q[i], v[i])["qacc"]
        for i in range(len(flat))
    ]
    return np.array(constrained).reshape(smooth.shape)


def swimmer_drift_numpy(
    dynamics: SwimmerDynamics, state: ndarray, torque: ndarray, *, context_dim: int = CONTEXT_DIM
) -> ndarray:
    """(qd, 0, qdd) for one state [D] or a batch [..., D], as swimmer_drift."""
    state = np.asarray(state, dtype=np.float64)
    nv = dynamics.nv
    if state.shape[-1] != 2 * nv + context_dim:
        raise ValueError(f"Swimmer states need {2 * nv + context_dim} entries, not {state.shape[-1]}.")
    velocities = state[..., nv + context_dim :]
    accelerations = swimmer_accelerations_numpy(dynamics, state[..., :nv], velocities, torque)
    context = np.zeros(state.shape[:-1] + (context_dim,))
    return np.concatenate((velocities, context, accelerations), axis=-1)


def _nose_numpy(plant: Mapping[str, float], state: ndarray) -> ndarray:
    offset = float(plant["nose_offset"])
    yaw = float(state[2])
    return np.array(
        [
            float(state[0]) + offset * math.sin(yaw),
            float(state[1]) - 0.5 * float(plant["link_length"]) - offset * math.cos(yaw),
        ]
    )


# No failure set: the joint limits stay MuJoCo's soft constraints.
_NO_FAILURE: Tuple[Tuple[int, float], ...] = ()


class SwimmerEnv(MujocoPHEnv):
    """dm_control's swimmer (6 or 15 links) with the target as episode context.

    The state is (x, y, psi, theta_1 .. theta_{n-1}, x*, y*, xdot, ydot, psidot,
    thetadot_1 .. thetadot_{n-1}).
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require_swimmer_keys(config)
        n_links = int(config["plant_parameters"]["n_links"])
        super().__init__(
            config,
            build_swimmer_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        expected = (n_links + 2, CONTEXT_DIM, n_links + 2)
        if (self.physical_position_dim, self.context_dim, self.velocity_dim) != expected:
            raise ValueError(f"The swimmer of {n_links} links needs (positions, context, velocities) {expected}.")
        self.n_links = n_links
        self._target_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self._dynamics = swimmer_dynamics(self.model, n_links)

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's reset: yaw, the joints from the head back, then the target."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        rng = self._reset_rng
        yaw_range, joint_range = float(reset["yaw_range"]), float(reset["joint_range"])
        yaw = rng.uniform(-yaw_range, yaw_range)
        joints = np.array([rng.uniform(-joint_range, joint_range) for _ in range(self.n_links - 1)])
        close = rng.random() < float(reset["close_target_probability"])
        box = float(reset["close_target_range"] if close else reset["far_target_range"])
        target = rng.uniform(-box, box, size=2)
        velocities = float(reset["velocity_noise"]) * rng.standard_normal(self.velocity_dim)
        self._set_state(np.concatenate(([0.0, 0.0, yaw], joints, target, velocities)))

    def _apply_context(self) -> None:
        """Move the target geom to the context (a world geom: no dynamics)."""
        geom = getattr(self, "_target_geom", None)
        if geom is None:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.model.geom_pos[geom, :2] = self._context
        self.data.geom_xpos[geom, :2] = self._context

    @property
    def target(self) -> ndarray:
        """The episode's target (x*, y*), the context."""
        return self.context

    @property
    def dynamics(self) -> SwimmerDynamics:
        return self._dynamics

    def nose(self, state: ndarray) -> ndarray:
        """World (x, y) of the nose in a state."""
        return _nose_numpy(self.config["plant_parameters"], state)

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's swim-to-target reward; swimmer_reward_rate_tensor stays its copy."""
        reward = self.config["reward_parameters"]
        target = np.asarray(state[self.physical_position_dim : self.position_dim], dtype=np.float64)
        distance = float(np.linalg.norm(self.nose(state) - target))
        return float(
            tolerance(
                distance,
                (0.0, float(reward["target_radius"])),
                float(reward["margin"]),
                sigmoid=REWARD_SIGMOID,
            )
        )

    def failed(self, state: ndarray) -> bool:
        """Never: a joint past its range meets dm_control's soft limit and continues."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return failure_rate_payoff(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """The numpy oracle, joint-limit forces included (one state or a batch)."""
        return swimmer_drift_numpy(self._dynamics, state, torque)

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The swimmer has no noise."""
        return np.zeros((self.state_dim, 1))


def swimmer_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """The long-tail swim-to-target reward from the root pose and the context, [B,1]."""
    reward = config["reward_parameters"]
    physical = int(config["plant_parameters"]["n_links"]) + 2
    offset = swimmer_nose_tensor(states, config) - states[..., physical : physical + 2]
    distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    return tolerance_tensor(
        distance,
        (0.0, float(reward["target_radius"])),
        float(reward["margin"]),
        sigmoid=REWARD_SIGMOID,
    )


def swimmer_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """All False, [B]: no failure set."""
    return limit_failure_mask(states, _NO_FAILURE)


def swimmer_terminal_payoff_value(config: TestbedConfig) -> float:
    """failure_rate / lambda (0 in the table); never charged without a failure set."""
    return failure_rate_payoff(config)


def swimmer_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure boundary: H = failure_rate / lambda (0) and B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), _NO_FAILURE)


def swimmer_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The analytic drift of the compiled model's SwimmerDynamics, zero rate on the context."""
    n_links = int(config["plant_parameters"]["n_links"])
    dynamics = swimmer_dynamics(build_swimmer_model(config), n_links)

    def drift(states: Tensor, torques: Tensor) -> Tensor:
        return swimmer_drift(dynamics, states, torques)

    return drift, zero_diffusion(2 * dynamics.nv + CONTEXT_DIM)


for _name, _factory in (
    ("swimmer_swimmer6", swimmer_swimmer6_config),
    ("swimmer_swimmer15", swimmer_swimmer15_config),
):
    register_testbed(
        TestbedSpec(
            name=_name,
            env_class=SwimmerEnv,
            config_factory=_factory,
            settings_keys=SETTINGS_KEYS,
            reward_rate_tensor=swimmer_reward_rate_tensor,
            failure_mask_tensor=swimmer_failure_mask_tensor,
            terminal_payoff_value=swimmer_terminal_payoff_value,
            terminal_boundary_functions=swimmer_terminal_boundary_functions,
            oracle_callbacks=swimmer_oracle_callbacks,
            upright=None,
            reward_rate_bounds=unit_interval_bounds,
        )
    )
