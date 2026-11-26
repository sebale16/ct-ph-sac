"""Helpers shared by the dm_control ports (and the in-house tasks' boundary functions).

tolerance / tolerance_tensor
    dm_control.utils.rewards.tolerance in numpy and torch for the shapes the ports use:
    margin 0 (the indicator of the bounds), "gaussian", "quadratic" and "long_tail" (the
    swimmer's 1 / (1 + (d s)^2), s = sqrt(1 / value_at_margin - 1)), with dm_control's
    default value 0.1 at the margin (pass value_at_margin=0.0 for the quadratic control
    cost of cartpole and point_mass, as dm_control does).
limit_failed / limit_failure_mask / failure_rate_payoff / limit_boundary_functions
    a failure set given as (state index, limit) pairs, |s[index]| >= limit failing; the
    payoff failure_rate / lambda; and the terminal lift (H, B) with
    B = prod (1 - (s[index] / limit)^2) over the finite limits.
zero_diffusion
    the diffusion callback of a noiseless task, a zero column [..., D, 1].
limit_attributes
    the solreflimit and solimplimit MJCF attributes of a joint that keeps a dm_control
    soft limit, from plant_parameters (common.ph_dmc._common.LIMIT_KEYS).
unit_interval_bounds
    TestbedSpec.reward_rate_bounds of a task whose reward rate lies in [0, 1], as
    dm_control's tolerance-shaped rewards do: the learned reward head then outputs a
    sigmoid onto [0, 1] (acrobot and lqr declare none).
body_inertials / planar_chain_from_model
    masses, centres of mass and inertias read from a compiled MuJoCo model (ours, or
    dm_control's physics.model at test time), and the PlanarChain they define.
todo
    the NotImplementedError a stub raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np
import torch

from .chain import PlanarChain

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import TestbedConfig

DEFAULT_VALUE_AT_MARGIN = 0.1
# -2 log(0.1): dm_control's gaussian tolerance at value 0.1 is exp(-0.5 * this * (d/m)^2).
TOLERANCE_SCALE_SQUARED = -2.0 * math.log(DEFAULT_VALUE_AT_MARGIN)
_SIGMOIDS = ("gaussian", "quadratic", "long_tail")


def todo(testbed: str, what: str) -> NotImplementedError:
    """The error a stub raises: the testbed is registered but not ported yet."""
    return NotImplementedError(
        f"Testbed {testbed!r} is registered as a stub: {what} is not implemented yet "
        "(see environment/ph_sac/dmc/README.md)."
    )


def require_keys(mapping: Mapping[str, object], keys: Tuple[str, ...], name: str) -> None:
    """Raise ValueError naming the missing keys of a configuration mapping."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required keys {missing}.")


def unit_interval_bounds(config: TestbedConfig) -> Tuple[float, float]:
    """(0, 1): the reward rate range of dm_control's tolerance-shaped rewards.

    The value of TestbedSpec.reward_rate_bounds for every port whose reward is a product
    of tolerances (pendulum, cart-pole, reacher, point mass); a learned reward head maps
    a sigmoid onto it. Not for lqr, whose 1 - cost is unbounded below.
    """
    return (0.0, 1.0)


def torque_limit(config: TestbedConfig) -> float:
    """The largest motor limit in absolute value (dm_control's control = torque / limit)."""
    controller = config["controller"]
    return float(
        max(np.max(np.abs(controller["torque_low"])), np.max(np.abs(controller["torque_high"])))
    )


# ---------------------------------------------------------------------------- tolerance


def _check_tolerance(
    bounds: Tuple[float, float], margin: float, sigmoid: str, value_at_margin: float
) -> None:
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative.")
    if margin > 0.0:
        if sigmoid not in _SIGMOIDS:
            raise ValueError(f"Unsupported sigmoid {sigmoid!r}; use one of {_SIGMOIDS}.")
        if sigmoid in ("gaussian", "long_tail") and not 0.0 < value_at_margin < 1.0:
            raise ValueError(f"A {sigmoid} tolerance needs value_at_margin in (0, 1).")
        if sigmoid == "quadratic" and not 0.0 <= value_at_margin < 1.0:
            raise ValueError("A quadratic tolerance needs value_at_margin in [0, 1).")


def tolerance(
    x: object,
    bounds: Tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = DEFAULT_VALUE_AT_MARGIN,
) -> object:
    """dm_control's rewards.tolerance in numpy: 1 inside the bounds, decaying outside.

    A scalar x gives a float, an array an array of the same shape.
    """
    _check_tolerance(bounds, margin, sigmoid, value_at_margin)
    lower, upper = bounds
    values = np.asarray(x, dtype=np.float64)
    if margin == 0.0:
        result = np.where((lower <= values) & (values <= upper), 1.0, 0.0)
    else:
        distance = (np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)) / margin
        if sigmoid == "gaussian":
            scale_squared = -2.0 * math.log(value_at_margin)
            result = np.exp(-0.5 * scale_squared * distance**2)
        elif sigmoid == "long_tail":
            scale = math.sqrt(1.0 / value_at_margin - 1.0)
            result = 1.0 / ((distance * scale) ** 2 + 1.0)
        else:
            scaled = distance * math.sqrt(1.0 - value_at_margin)
            result = np.maximum(1.0 - scaled**2, 0.0)
    return float(result) if np.ndim(x) == 0 else result


def tolerance_tensor(
    x: Tensor,
    bounds: Tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = DEFAULT_VALUE_AT_MARGIN,
) -> Tensor:
    """dm_control's rewards.tolerance in torch, elementwise; compatible with torch.func.

    The gaussian, long_tail and quadratic shapes are differentiable in x (the quadratic
    has a kink where it reaches 0); the indicator (margin 0) is piecewise constant.
    """
    _check_tolerance(bounds, margin, sigmoid, value_at_margin)
    lower, upper = bounds
    if margin == 0.0:
        return ((x >= lower) & (x <= upper)).to(x.dtype)
    distance = (torch.clamp(lower - x, min=0.0) + torch.clamp(x - upper, min=0.0)) / margin
    if sigmoid == "gaussian":
        scale_squared = -2.0 * math.log(value_at_margin)
        return torch.exp(-0.5 * scale_squared * distance.square())
    if sigmoid == "long_tail":
        scale = math.sqrt(1.0 / value_at_margin - 1.0)
        return 1.0 / ((distance * scale).square() + 1.0)
    scaled = distance * math.sqrt(1.0 - value_at_margin)
    return torch.clamp(1.0 - scaled.square(), min=0.0)


# ------------------------------------------------------------------- failure boundary


def finite_limits(limited: Sequence[Tuple[int, float]]) -> Tuple[Tuple[int, float], ...]:
    """Drop infinite limits, which bound nothing."""
    return tuple((int(index), float(limit)) for index, limit in limited if math.isfinite(limit))


def limit_failed(state: object, limited: Sequence[Tuple[int, float]]) -> bool:
    """Whether any |state[index]| >= limit; the numpy twin of limit_failure_mask."""
    state = np.asarray(state)
    return bool(any(abs(state[index]) >= limit for index, limit in finite_limits(limited)))


def limit_failure_mask(states: Tensor, limited: Sequence[Tuple[int, float]]) -> Tensor:
    """Boolean [B]: any |states[..., index]| >= limit; all False without finite limits."""
    mask = torch.zeros(states.shape[:-1], dtype=torch.bool, device=states.device)
    for index, limit in finite_limits(limited):
        mask = mask | (states[..., index].abs() >= limit)
    return mask


def failure_rate_payoff(config: TestbedConfig) -> float:
    """F = failure_rate / lambda, reading reward_parameters["failure_rate"] (default 0)."""
    rate = config["reward_parameters"].get("failure_rate", 0.0)
    return rate / config["discount_rate"]


def limit_boundary_functions(
    payoff: float, limited: Sequence[Tuple[int, float]]
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """(H, B) for V = H + B * network: H the constant payoff, B = prod 1 - (x / x_lim)^2.

    Each finite limit contributes its factor, so B is positive strictly inside the limits
    and zero on them; with no finite limit B is 1. Travel coordinates must be unwrapped.
    """
    limited = finite_limits(limited)

    def lift(states: Tensor) -> Tensor:
        return torch.full_like(states[..., :1], payoff)

    def vanishing(states: Tensor) -> Tensor:
        factors = [1.0 - (states[..., index : index + 1] / limit).square() for index, limit in limited]
        if not factors:
            return torch.ones_like(states[..., :1])
        return torch.stack(factors, dim=0).prod(dim=0)

    return lift, vanishing


# A task without a failure set: an all-False mask, H = payoff (0) and B = 1.
NO_FAILURE: Tuple[Tuple[int, float], ...] = ()


def limit_attributes(plant: Mapping[str, float]) -> str:
    """'solreflimit="tc dr" solimplimit="d0 dmax width mid power"' from plant_parameters.

    The keys are common.ph_dmc._common.LIMIT_KEYS; repr keeps every float exact, so the
    compiled jnt_solref and jnt_solimp equal the values (MuJoCo's defaults where
    dm_control's XML omits them, compiled bitwise as MuJoCo's own defaults).
    """
    solref = " ".join(repr(float(plant[key])) for key in ("limit_timeconst", "limit_dampratio"))
    solimp = " ".join(
        repr(float(plant[key]))
        for key in ("limit_d0", "limit_dmax", "limit_width", "limit_midpoint", "limit_power")
    )
    return f'solreflimit="{solref}" solimplimit="{solimp}"'


def zero_diffusion(state_dim: int) -> Callable[[Tensor], Tensor]:
    """Diffusion callback of a noiseless task: zeros [..., state_dim, 1]."""

    def diffusion(states: Tensor) -> Tensor:
        return torch.zeros(
            *states.shape[:-1], state_dim, 1, dtype=states.dtype, device=states.device
        )

    return diffusion


# ------------------------------------------------------------- compiled model inertials


@dataclass(frozen=True, eq=False)
class BodyInertial:
    """Mass, centre of mass (3,) and inertia (3, 3) about it, both in the body's frame."""

    mass: float
    com: np.ndarray
    inertia: np.ndarray


def raw_model(model: object) -> mujoco.MjModel:
    """A mujoco.MjModel from itself, a dm_control model wrapper (.ptr) or a Physics (.model)."""
    if isinstance(model, mujoco.MjModel):
        return model
    inner = getattr(model, "model", None)
    if inner is not None:
        model = inner
    if isinstance(model, mujoco.MjModel):
        return model
    pointer = getattr(model, "ptr", None)
    if isinstance(pointer, mujoco.MjModel):
        return pointer
    raise TypeError(f"Cannot read a MuJoCo model from {type(model).__name__}.")


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f"The model has no body {name!r}.")
    return int(body)


def _inertial_parts(
    model: mujoco.MjModel,
    body: int,
    position: np.ndarray,
    rotation: np.ndarray,
    include_welded: bool,
) -> list:
    """(mass, com, inertia about com) of body and its welded descendants, in one frame."""
    principal = _rotation(model.body_iquat[body])
    inertia = principal @ np.diag(model.body_inertia[body]) @ principal.T
    parts = [
        (
            float(model.body_mass[body]),
            position + rotation @ model.body_ipos[body],
            rotation @ inertia @ rotation.T,
        )
    ]
    if include_welded:
        for child in range(1, model.nbody):
            if child != body and model.body_parentid[child] == body and model.body_jntnum[child] == 0:
                parts.extend(
                    _inertial_parts(
                        model,
                        child,
                        position + rotation @ model.body_pos[child],
                        rotation @ _rotation(model.body_quat[child]),
                        include_welded,
                    )
                )
    return parts


def body_inertials(
    model: object, names: Sequence[str], *, include_welded: bool = True
) -> Dict[str, BodyInertial]:
    """Read compiled masses, centres of mass and inertias of the named bodies.

    model is a mujoco.MjModel, a dm_control model wrapper or a dm_control Physics. Each
    body's inertial is expressed in that body's frame; with include_welded (the default),
    jointless descendants, which move rigidly with the body (the reacher's finger on its
    hand), are merged into it by the parallel-axis theorem.
    """
    model = raw_model(model)
    result: Dict[str, BodyInertial] = {}
    for name in names:
        parts = _inertial_parts(model, _body_id(model, name), np.zeros(3), np.eye(3), include_welded)
        mass = sum(part[0] for part in parts)
        if mass > 0.0:
            com = sum(part[0] * part[1] for part in parts) / mass
        else:
            com = np.zeros(3)
        inertia = np.zeros((3, 3))
        for part_mass, part_com, part_inertia in parts:
            offset = part_com - com
            inertia += part_inertia + part_mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
        result[name] = BodyInertial(mass=float(mass), com=com, inertia=inertia)
    return result


def _single_dof(model: mujoco.MjModel, body: int, kind: int, name: str) -> int:
    if model.body_jntnum[body] != 1:
        raise ValueError(f"Body {name!r} needs exactly one joint.")
    joint = int(model.body_jntadr[body])
    if model.jnt_type[joint] != kind:
        raise ValueError(f"Body {name!r} has a joint of the wrong type.")
    dof = int(model.jnt_dofadr[joint])
    for label, values in (
        ("armature", model.dof_armature),
        ("frictionloss", model.dof_frictionloss),
        ("stiffness", model.jnt_stiffness),
    ):
        index = joint if label == "stiffness" else dof
        if values[index] != 0.0:
            raise ValueError(f"Joint of {name!r} has {label}, which PlanarChain lacks.")
    return joint


def planar_chain_from_model(
    model: object,
    link_bodies: Sequence[str],
    *,
    along: int,
    joint_axis: int,
    cart_body: Optional[str] = None,
    last_length: float = 0.0,
) -> PlanarChain:
    """The PlanarChain of a compiled MuJoCo model's serial chain (see chain.py).

    link_bodies name the link bodies from the base outward, each with one hinge about
    +e[joint_axis] (welded descendants merged in); along is the body-frame axis a link
    points along at zero angle ("up"); the side axis is the remaining one. dm_control's
    vertical models: along=2 (z), joint_axis=1 (y); the reacher: along=0 (x),
    joint_axis=2 (z). cart_body names a body with one slide along +e[side] carrying the
    first link. Gravity is the model's along -e[along]; link lengths are the next link
    body's offset along the link; the last link gets last_length. The input matrix maps
    each joint-transmission actuator's control to its joint with its gear, so the input is
    the model's ctrl. Armature, joint stiffness and friction loss are rejected.
    """
    model = raw_model(model)
    side = 3 - along - joint_axis
    if sorted((along, joint_axis, side)) != [0, 1, 2]:
        raise ValueError("along and joint_axis must be two different axes in 0, 1, 2.")
    axis = np.eye(3)[joint_axis]
    names = ([cart_body] if cart_body is not None else []) + list(link_bodies)
    for name in names:
        if not np.allclose(model.body_quat[_body_id(model, name)], [1.0, 0.0, 0.0, 0.0]):
            raise ValueError(f"Body {name!r} must not be rotated relative to its parent.")
    inertials = body_inertials(model, names)
    dofs = []
    cart_mass = None
    cart_damping = 0.0
    if cart_body is not None:
        body = _body_id(model, cart_body)
        joint = _single_dof(model, body, mujoco.mjtJoint.mjJNT_SLIDE, cart_body)
        if not np.allclose(model.jnt_axis[joint], np.eye(3)[side]):
            raise ValueError(f"The cart's slide must point along +e[{side}].")
        cart_mass = inertials[cart_body].mass
        cart_damping = float(model.dof_damping[model.jnt_dofadr[joint]])
        dofs.append(int(model.jnt_dofadr[joint]))
    masses, com, inertias, lengths, damping = [], [], [], [], []
    for position, name in enumerate(link_bodies):
        body = _body_id(model, name)
        joint = _single_dof(model, body, mujoco.mjtJoint.mjJNT_HINGE, name)
        if not np.allclose(model.jnt_axis[joint], axis):
            raise ValueError(f"The hinge of {name!r} must turn about +e[{joint_axis}].")
        if not np.allclose(model.jnt_pos[joint], 0.0):
            raise ValueError(f"The hinge of {name!r} must sit at its body's origin.")
        inertial = inertials[name]
        if abs(inertial.com[side]) > 1e-12:
            raise ValueError(f"The centre of mass of {name!r} must lie on the link line.")
        masses.append(inertial.mass)
        com.append(float(inertial.com[along]))
        inertias.append(float(inertial.inertia[joint_axis, joint_axis]))
        damping.append(float(model.dof_damping[model.jnt_dofadr[joint]]))
        dofs.append(int(model.jnt_dofadr[joint]))
        if position + 1 < len(link_bodies):
            offset = model.body_pos[_body_id(model, link_bodies[position + 1])]
            if abs(offset[side]) > 1e-12:
                raise ValueError(f"The link after {name!r} must sit along it.")
            lengths.append(float(offset[along]))
        else:
            lengths.append(float(last_length))
    input_matrix = np.zeros((len(dofs), model.nu))
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        dof = int(model.jnt_dofadr[model.actuator_trnid[actuator, 0]])
        if dof in dofs:
            input_matrix[dofs.index(dof), actuator] = float(model.actuator_gear[actuator, 0])
    return PlanarChain(
        link_masses=masses,
        link_com_distances=com,
        link_inertias=inertias,
        link_lengths=lengths,
        joint_damping=damping,
        input_matrix=input_matrix,
        gravity=-float(model.opt.gravity[along]),
        cart_mass=cart_mass,
        cart_damping=cart_damping,
    )
