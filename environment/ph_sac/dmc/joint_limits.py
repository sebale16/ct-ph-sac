"""MuJoCo's soft joint-limit constraint, in torch and numpy, for the oracle plants.

dm_control's models keep some joints `limited` (the swimmer's hinges, the cart-pole's
slider, the reacher's wrist, the point mass's slides). MuJoCo turns each violated limit
into a soft, one-sided constraint row and solves for the constraint forces together with
the dynamics. This module reproduces that computation for hinge and slide limits, so an
oracle drift equals MuJoCo's forward dynamics with the limits active (mj_forward's qacc
and qfrc_constraint, to rounding), batched and torch.func-compatible.

The model (MuJoCo's documentation, Computation: Constraint model, Solver parameters, and
engine_core_constraint.c), for a limited scalar joint with range [lo, hi], margin 0:

  rows        one row per violated side: the upper side when q > hi, with Jacobian
              J = -e_dof, distance pos = hi - q; the lower side when q < lo, with
              J = +e_dof, pos = q - lo. A row exists only while pos < margin (= 0), so
              pos < 0 is the penetration. vel = J qdot.
  impedance   d(pos) from solimp (d0, dmax, width, midpoint, power), with d0 and dmax
              first clipped to [mjMINIMP, mjMAXIMP] = [1e-4, 0.9999] (so a solimp d0 of 0
              acts as 1e-4): with x = |pos - margin| / width, d = d0 + y(x) (dmax - d0),
              y(x) = a x^p up to the midpoint and 1 - b (1 - x)^p beyond
              (a = mid^(1-p), b = (1-mid)^(1-p)), y = 1 for x >= 1.
  reference   with dmax clipped likewise and timeconst = max(solref[0], 2 timestep)
              (MuJoCo's refsafe), b = 2 / (dmax timeconst),
              k = 1 / (dmax^2 timeconst^2 dampratio^2), and
              aref = -b vel - k d (pos - margin).
  regularizer R = max(mjMINVAL, (1 - d) / d * diagApprox), D = 1 / R, with diagApprox the
              model's dof_invweight0 of the joint's dof (computed by mj_setConst at
              qpos0, not the current J M^-1 J^T).
  solution    the forces f >= 0 of the convex problem MuJoCo's solvers minimize,
              (M + J_S^T D_S J_S) qacc = M qacc_smooth + J_S^T D_S aref_S on the active set
              S = {rows with J qacc < aref}, i.e. the linear complementarity problem
              (A + R) f - (aref - J qacc_smooth) = w, f >= 0, w >= 0, f w = 0, with
              A = J M^-1 J^T; then qacc = qacc_smooth + M^-1 J^T f and
              qfrc_constraint = J^T f. A limit only pushes the joint back into its range.
              MuJoCo's default Newton solver (opt.iterations 100, opt.tolerance 1e-8,
              exact line search on the piecewise quadratic) reaches this minimizer to
              rounding; it is solved here by principal pivoting on the active set, a fixed
              number of passes of tensor operations (no data-dependent branching, so
              torch.func.vmap works), each a Cholesky solve on the current set.

Equivalently, while a row is active it adds the constraint mass D to the joint's inertia
and the force D aref: a spring D k d |pos| and a damper D b, both gated by the
penetration through d(pos). The added inertia makes the constraint cancel a fraction of
every other force on the joint, which a potential plus damping cannot reproduce exactly
(see PortHamiltonianPlant's limit port). limit_port_init derives that port's initial
stiffness, damping and width from the compiled limit (its docstring has the derivation);
the dm_control factories store them in the state spec (StateSpec limit_port_init).

Use (any port): build JointLimits from the compiled model with
joint_limits_from_model(model, joint_names), compute the smooth acceleration
qacc_smooth = M^-1 (G u - bias + passive), then
constrained_accelerations(limits, mass_factor, qacc_smooth, positions, velocities)
(torch; mass_factor the Cholesky factor of M) or constrained_accelerations_numpy(...,
mass, ...). positions and velocities are the generalized coordinates in MuJoCo's dof
order (an oracle with a context strips it first). Joint ranges, solref, solimp and margin
are the compiled model's, so MuJoCo's defaults (solref 0.02 1, solimp 0.9 0.95 0.001 0.5 2)
apply where the XML omits them. mujoco_limit_rows(model, data) reads MuJoCo's efc_* rows
of the same joints after mj_forward, for cross-checks in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np
import torch
from torch import Tensor

from ._common import raw_model

MJ_MINIMP = 1e-4  # mjMINIMP
MJ_MAXIMP = 0.9999  # mjMAXIMP
MJ_MINVAL = 1e-15  # mjMINVAL


@dataclass(frozen=True, eq=False)
class JointLimits:
    """Limited scalar joints (hinge or slide) of a model, one entry per joint.

    dofs: their dof (velocity) indices; lower, upper: the ranges; timeconst, dampratio:
    solref (timeconst already max(solref[0], 2 timestep) when refsafe is on);
    d0, dmax, width, midpoint, power: solimp; margin; invweight: dof_invweight0
    (MuJoCo's diagApprox of the row).
    """

    dofs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    timeconst: np.ndarray
    dampratio: np.ndarray
    d0: np.ndarray
    dmax: np.ndarray
    width: np.ndarray
    midpoint: np.ndarray
    power: np.ndarray
    margin: np.ndarray
    invweight: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.dofs)
        for name in ("lower", "upper", "timeconst", "dampratio", "d0", "dmax", "width",
                     "midpoint", "power", "margin", "invweight"):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.shape != (count,):
                raise ValueError(f"JointLimits {name} needs one value per joint.")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "dofs", np.asarray(self.dofs, dtype=np.int64).reshape(-1))
        if np.any(self.lower >= self.upper):
            raise ValueError("JointLimits need lower < upper.")
        if np.any(self.timeconst <= 0.0):
            raise ValueError("Only positive solref (timeconst, dampratio) is supported.")
        dmax = np.clip(self.dmax, MJ_MINIMP, MJ_MAXIMP)
        object.__setattr__(self, "damping_coefficient", 2.0 / (dmax * self.timeconst))
        object.__setattr__(
            self,
            "stiffness_coefficient",
            1.0 / (dmax * dmax * self.timeconst**2 * self.dampratio**2),
        )

    @property
    def count(self) -> int:
        return len(self.dofs)


def joint_limits_from_model(model: object, joints: Sequence[str]) -> JointLimits:
    """JointLimits of the named limited hinge or slide joints of a compiled model.

    model is a mujoco.MjModel, a dm_control model wrapper or a Physics. Raises ValueError
    for an unlimited joint, another joint type, or a negative (direct) solref.
    """
    model = raw_model(model)
    ids = []
    for name in joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"The model has no joint {name!r}.")
        if model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint {name!r} is not a hinge or slide.")
        if not model.jnt_limited[joint]:
            raise ValueError(f"Joint {name!r} is not limited.")
        ids.append(joint)
    ids = np.array(ids, dtype=np.int64)
    solref = np.array(model.jnt_solref[ids], dtype=np.float64).reshape(len(ids), -1)
    solimp = np.array(model.jnt_solimp[ids], dtype=np.float64).reshape(len(ids), -1)
    timeconst = solref[:, 0]
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        timeconst = np.maximum(timeconst, 2.0 * float(model.opt.timestep))
    dofs = np.array(model.jnt_dofadr[ids], dtype=np.int64)
    return JointLimits(
        dofs=dofs,
        lower=model.jnt_range[ids, 0],
        upper=model.jnt_range[ids, 1],
        timeconst=timeconst,
        dampratio=solref[:, 1],
        d0=solimp[:, 0],
        dmax=solimp[:, 1],
        width=solimp[:, 2],
        midpoint=solimp[:, 3],
        power=solimp[:, 4],
        margin=model.jnt_margin[ids],
        invweight=model.dof_invweight0[dofs],
    )


# ------------------------------------------------------------------------------- torch


def _constant(array: np.ndarray, like: Tensor) -> Tensor:
    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def impedance(limits: JointLimits, distance: Tensor) -> Tensor:
    """MuJoCo's d(pos) of every joint for distance = |pos - margin| [..., L].

    d0 and dmax are clipped to [mjMINIMP, mjMAXIMP] before the interpolation, as MuJoCo
    does (with dm_control's swimmer d0 = 0 the impedance starts at 1e-4).
    """
    d0 = _constant(np.clip(limits.d0, MJ_MINIMP, MJ_MAXIMP), distance)
    dmax = _constant(np.clip(limits.dmax, MJ_MINIMP, MJ_MAXIMP), distance)
    width = _constant(limits.width, distance)
    mid, power = _constant(limits.midpoint, distance), _constant(limits.power, distance)
    x = distance / width
    inner = torch.clamp(x, 0.0, 1.0)
    low = torch.pow(inner, power) / torch.pow(mid, power - 1.0)
    high = 1.0 - torch.pow(1.0 - inner, power) / torch.pow(1.0 - mid, power - 1.0)
    y = torch.where(power == 1.0, inner, torch.where(inner <= mid, low, high))
    value = d0 + y * (dmax - d0)
    value = torch.where(x >= 1.0, dmax, torch.where(x <= 0.0, d0, value))
    flat = (d0 == dmax) | (width <= MJ_MINVAL)
    value = torch.where(flat, 0.5 * (d0 + dmax), value)
    return torch.clamp(value, MJ_MINIMP, MJ_MAXIMP)


def limit_rows(limits: JointLimits, positions: Tensor, velocities: Tensor) -> Dict[str, Tensor]:
    """MuJoCo's constraint rows of the limits, one slot per joint, [..., L].

    "active": a row exists (pos < margin); "side": +1 upper, -1 lower (J = -side e_dof);
    "pos", "vel", "imp", "aref", "R", "D" as MuJoCo's efc_pos, efc_vel, efc_KBIP[2],
    efc_aref, efc_R, efc_D. Slots without a row hold harmless finite values.
    """
    dofs = _constant(limits.dofs, positions).long()
    q = positions.index_select(-1, dofs)
    qd = velocities.index_select(-1, dofs)
    lower, upper = _constant(limits.lower, q), _constant(limits.upper, q)
    margin = _constant(limits.margin, q)
    above = upper - q < margin
    below = q - lower < margin
    side = torch.where(above, torch.ones_like(q), torch.where(below, -torch.ones_like(q), torch.zeros_like(q)))
    pos = torch.where(above, upper - q, torch.where(below, q - lower, margin))
    vel = -side * qd
    imp = impedance(limits, (pos - margin).abs())
    aref = (
        -_constant(limits.damping_coefficient, q) * vel
        - _constant(limits.stiffness_coefficient, q) * imp * (pos - margin)
    )
    regularizer = torch.clamp((1.0 - imp) * _constant(limits.invweight, q) / imp, min=MJ_MINVAL)
    return {
        "active": above | below,
        "side": side,
        "pos": pos,
        "vel": vel,
        "imp": imp,
        "aref": aref,
        "R": regularizer,
        "D": 1.0 / regularizer,
    }


def constrained_accelerations(
    limits: JointLimits,
    mass_factor: Tensor,
    qacc_smooth: Tensor,
    positions: Tensor,
    velocities: Tensor,
    *,
    passes: Optional[int] = None,
) -> Dict[str, Tensor]:
    """qacc with the limit forces, from the Cholesky factor of M(q) and qacc_smooth.

    Returns "qacc" [..., nv], "force" (f >= 0 per joint, 0 without a row) [..., L],
    "qfrc_constraint" = J^T f [..., nv] and the rows of limit_rows. The complementarity
    problem is solved by principal pivoting from the set of rows that exist: each pass
    solves (A + R) f = aref - J qacc_smooth on the current set, then keeps the rows with
    f > 0 and adds those with w < 0. passes defaults to L + 1 (L joints).
    """
    rows = limit_rows(limits, positions, velocities)
    count = limits.count
    nv = qacc_smooth.shape[-1]
    dofs = _constant(limits.dofs, qacc_smooth).long()
    eye = torch.eye(nv, dtype=qacc_smooth.dtype, device=qacc_smooth.device)
    inverse = torch.cholesky_solve(eye.expand(*qacc_smooth.shape[:-1], nv, nv), mass_factor)
    # J = -side e_dof per slot: M^-1 J^T [..., nv, L] and A = J M^-1 J^T [..., L, L].
    side = rows["side"]
    columns = -inverse.index_select(-1, dofs) * side.unsqueeze(-2)
    block = columns.index_select(-2, dofs) * (-side).unsqueeze(-1)
    system = block + torch.diag_embed(rows["R"])
    rhs = rows["aref"] + side * qacc_smooth.index_select(-1, dofs)
    exists = rows["active"]
    eye_l = torch.eye(count, dtype=qacc_smooth.dtype, device=qacc_smooth.device)

    def solve(mask: Tensor) -> Tensor:
        weight = mask.to(system.dtype)
        masked = system * weight.unsqueeze(-1) * weight.unsqueeze(-2) + eye_l * (1.0 - weight).unsqueeze(-1)
        factor = torch.linalg.cholesky(masked)
        return torch.cholesky_solve((rhs * weight).unsqueeze(-1), factor).squeeze(-1)

    mask = exists
    for _ in range(count + 1 if passes is None else int(passes)):
        force = solve(mask)
        slack = (system @ force.unsqueeze(-1)).squeeze(-1) - rhs
        mask = exists & ((mask & (force > 0.0)) | (~mask & (slack < 0.0)))
    force = solve(mask)
    qfrc = (-side * force) @ torch.nn.functional.one_hot(dofs, nv).to(force.dtype)
    qacc = qacc_smooth + (columns @ force.unsqueeze(-1)).squeeze(-1)
    return {**rows, "qacc": qacc, "force": force, "qfrc_constraint": qfrc, "rows_active": mask}


# ------------------------------------------------------------------------------- numpy


def _impedance_numpy(limits: JointLimits, distance: np.ndarray) -> np.ndarray:
    d0 = np.clip(limits.d0, MJ_MINIMP, MJ_MAXIMP)
    dmax = np.clip(limits.dmax, MJ_MINIMP, MJ_MAXIMP)
    width = limits.width
    mid, power = limits.midpoint, limits.power
    x = distance / width
    inner = np.clip(x, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        low = inner**power / mid ** (power - 1.0)
        high = 1.0 - (1.0 - inner) ** power / (1.0 - mid) ** (power - 1.0)
    y = np.where(power == 1.0, inner, np.where(inner <= mid, low, high))
    value = np.where(x >= 1.0, dmax, np.where(x <= 0.0, d0, d0 + y * (dmax - d0)))
    value = np.where((d0 == dmax) | (width <= MJ_MINVAL), 0.5 * (d0 + dmax), value)
    return np.clip(value, MJ_MINIMP, MJ_MAXIMP)


def constrained_accelerations_numpy(
    limits: JointLimits,
    mass: np.ndarray,
    qacc_smooth: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> Dict[str, np.ndarray]:
    """The numpy twin of constrained_accelerations for one state (mass [nv, nv]).

    Solves the same complementarity problem by principal pivoting, iterating until the
    active set repeats (at most 2^L + 1 passes).
    """
    q = np.asarray(positions, dtype=np.float64)[limits.dofs]
    qd = np.asarray(velocities, dtype=np.float64)[limits.dofs]
    qacc_smooth = np.asarray(qacc_smooth, dtype=np.float64)
    above = limits.upper - q < limits.margin
    below = q - limits.lower < limits.margin
    side = np.where(above, 1.0, np.where(below, -1.0, 0.0))
    pos = np.where(above, limits.upper - q, np.where(below, q - limits.lower, limits.margin))
    vel = -side * qd
    imp = _impedance_numpy(limits, np.abs(pos - limits.margin))
    aref = -limits.damping_coefficient * vel - limits.stiffness_coefficient * imp * (pos - limits.margin)
    regularizer = np.maximum((1.0 - imp) * limits.invweight / imp, MJ_MINVAL)
    nv = len(qacc_smooth)
    jacobian = np.zeros((limits.count, nv))
    jacobian[np.arange(limits.count), limits.dofs] = -side
    inverse = np.linalg.inv(np.asarray(mass, dtype=np.float64))
    system = jacobian @ inverse @ jacobian.T + np.diag(regularizer)
    rhs = aref - jacobian @ qacc_smooth
    exists = above | below
    mask = exists.copy()
    seen = set()
    force = np.zeros(limits.count)
    while True:
        force = np.zeros(limits.count)
        if mask.any():
            force[mask] = np.linalg.solve(system[np.ix_(mask, mask)], rhs[mask])
        slack = system @ force - rhs
        new = exists & ((mask & (force > 0.0)) | (~mask & (slack < 0.0)))
        key = tuple(new.tolist())
        if np.array_equal(new, mask) or key in seen:
            mask = new
            force = np.zeros(limits.count)
            if mask.any():
                force[mask] = np.linalg.solve(system[np.ix_(mask, mask)], rhs[mask])
            break
        seen.add(key)
        mask = new
    qfrc = jacobian.T @ force
    return {
        "active": exists,
        "side": side,
        "pos": pos,
        "vel": vel,
        "imp": imp,
        "aref": aref,
        "R": regularizer,
        "D": 1.0 / regularizer,
        "force": force,
        "qfrc_constraint": qfrc,
        "qacc": qacc_smooth + inverse @ qfrc,
        "rows_active": mask,
    }


# ------------------------------------------------------------------- MuJoCo cross-check


def mujoco_limit_rows(model: object, data: object, limits: JointLimits) -> Dict[str, np.ndarray]:
    """MuJoCo's efc_* rows of the limits after mj_forward, one slot per joint (test utility).

    Returns "active" (a row exists), "side" (+1 for the upper limit, read from the
    joint's position against its range), "pos", "margin", "vel" (efc_vel), "imp"
    (efc_KBIP[2]), "aref", "R", "D", "diagApprox", "force" and "state" (efc_state;
    mjCNSTRSTATE_QUADRATIC while it pushes) per joint; NaN where no row. model and data
    may be dm_control wrappers or a Physics.
    """
    model = raw_model(model)
    if hasattr(data, "data"):
        data = data.data
    data = getattr(data, "ptr", data)
    count = limits.count
    out = {key: np.full(count, np.nan) for key in
           ("side", "pos", "margin", "vel", "imp", "aref", "R", "D", "diagApprox", "force", "state")}
    out["active"] = np.zeros(count, dtype=bool)
    dof_to_slot = {int(dof): slot for slot, dof in enumerate(limits.dofs)}
    kbip = np.array(data.efc_KBIP).reshape(-1, 4)
    for row in range(data.nefc):
        if data.efc_type[row] != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            continue
        joint = int(data.efc_id[row])
        dof = int(model.jnt_dofadr[joint])
        if dof not in dof_to_slot:
            continue
        slot = dof_to_slot[dof]
        position = float(data.qpos[model.jnt_qposadr[joint]])
        out["active"][slot] = True
        out["side"][slot] = 1.0 if position > 0.5 * (model.jnt_range[joint, 0] + model.jnt_range[joint, 1]) else -1.0
        out["pos"][slot] = data.efc_pos[row]
        out["margin"][slot] = data.efc_margin[row]
        out["vel"][slot] = data.efc_vel[row]
        out["imp"][slot] = kbip[row, 2]
        out["aref"][slot] = data.efc_aref[row]
        out["R"][slot] = data.efc_R[row]
        out["D"][slot] = data.efc_D[row]
        out["diagApprox"][slot] = data.efc_diagApprox[row]
        out["force"][slot] = data.efc_force[row]
        out["state"][slot] = data.efc_state[row]
    return out


def limit_port_init(limits: JointLimits) -> Tuple[Tuple[float, float, float], ...]:
    """The learned plant's limit-port start (stiffness k, damping c, width w) per joint.

    Derivation. While one limit row is active, with A = J M^-1 J^T the joint's inverse
    inertia and R = (1 - d) / d * invweight0 (invweight0 = dof_invweight0, MuJoCo's
    diagApprox), the force is f = (aref - J qacc_smooth) / (A + R). At the inertia of
    qpos0, A = invweight0, so A + R = invweight0 / d and along the joint
        J qacc = (1 - d) J qacc_smooth + d aref,  aref = -b vel + k d r,
    r = -pos >= 0 the penetration and vel = J qdot. In force units (times the joint's mass
    1 / invweight0) the limit is a spring and a damper gated by the impedance,
        K(r) = k d(r)^2 / invweight0,   C(r) = b d(r) / invweight0,
    plus -d J qacc_smooth / invweight0, which cancels the fraction d of every other force on
    the joint (the constraint's added inertia; no potential or damping reproduces it).
    Beyond the impedance width d = dmax, and with b = 2 / (dmax tc) and
    k = 1 / (dmax^2 tc^2 zeta^2) (solref's timeconst and dampratio zeta),
        k_port = 1 / (invweight0 tc^2 zeta^2),   c_port = 2 / (invweight0 tc):
    the joint's own inertia on a spring and damper of time constant tc and damping ratio
    zeta, which is what solref means. tc keeps MuJoCo's floor tc = max(solref[0],
    2 timestep) (refsafe; JointLimits.timeconst holds it): at our 1 ms physics step the
    reacher's and point mass's 0.02 s stays 0.02 s (at dm_control's 20 ms step it would act
    as 0.04 s, a quarter of the stiffness), the cart-pole's 0.08 s and the swimmer's 0.05 s
    are above the floor either way. The impedance rises from d0 at zero penetration to
    dmax over solimp's width W, while the port's stiffness and damping rise from 0 to
    k_port and c_port over its width w by a smoothstep, and w cannot go below the plant's
    LIMIT_WIDTH_FLOOR (1e-3); the port starts at
        w = LIMIT_WIDTH_FLOOR + W (1 - d0 / dmax),
    the floor plus the part of W over which the impedance still has to rise (d0 and dmax
    clipped to [mjMINIMP, mjMAXIMP] as MuJoCo does).

    Values at the table's 1 ms physics step: cart-pole slider k 160.5 N/m (161.0 with two
    or three poles), c 25.7 N s/m, w 1.053e-3 m (solref 0.08 1, MuJoCo's default solimp
    0.9 0.95 0.001); reacher wrist k 0.1205 N m/rad, c 4.82e-3 N m s/rad, w 1.053e-3 rad
    (MuJoCo's defaults); point-mass slides k 750 N/m, c 30 N s/m, w 1.053e-3 m (MuJoCo's
    defaults; invweight0 = 1 / 0.3 kg); swimmer joints k 2.7e-3 to 3.7e-3 N m/rad, c 2.7e-4
    to 3.7e-4 N m s/rad, w 0.1010 rad (solref 0.05 1, solimp 0 0.8 0.1). A short fit of the
    port at a high learning rate from its earlier start (k 1e-2) came to 164 N/m on the
    cart-pole and 746 to 764 N/m on the point mass (tests/test_ph_dmc_joint_limits.py).

    Raises ValueError for a nonzero margin (the port starts exactly at the range).
    """
    if np.any(limits.margin != 0.0):
        raise ValueError("The limit port starts at the range: limit_port_init needs margin 0.")
    from models.ph_sac.plant import LIMIT_WIDTH_FLOOR  # deferred, as tasks.py imports models

    d0 = np.clip(limits.d0, MJ_MINIMP, MJ_MAXIMP)
    dmax = np.clip(limits.dmax, MJ_MINIMP, MJ_MAXIMP)
    stiffness = limits.stiffness_coefficient * dmax**2 / limits.invweight
    damping = limits.damping_coefficient * dmax / limits.invweight
    width = LIMIT_WIDTH_FLOOR + limits.width * np.clip(1.0 - d0 / dmax, 0.0, 1.0)
    return tuple(
        (float(k), float(c), float(w)) for k, c, w in zip(stiffness, damping, width)
    )


def limit_constants(limits: JointLimits) -> Mapping[str, np.ndarray]:
    """b and k of every joint's reference acceleration (MuJoCo's efc_KBIP[1], [0])."""
    return {"b": limits.damping_coefficient, "k": limits.stiffness_coefficient}
