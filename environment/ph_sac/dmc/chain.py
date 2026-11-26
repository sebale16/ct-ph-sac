"""Forward dynamics of a planar serial chain of rigid links, on a fixed base or a cart.

The chain covers dm_control's pendulum (one link), acrobot (two links), reacher (two
links, no gravity) and the cart-pole family (one to three links on a cart). Every
function exists twice: a batched torch version, differentiable and compatible with
torch.func (no in-place writes, no .item()), and a numpy version with the same name and a
_numpy suffix. Both solve M(q) qdd = G u - D qd - C(q, qd) qd - g(q) per batch row: numpy
by np.linalg.solve, torch by a Cholesky factorization of the positive-definite M and
torch.cholesky_solve. Not torch.linalg.solve: in torch 2.12 its forward-mode derivative
is wrong under torch.func.vmap (vmap(jacfwd(...)), which OraclePlant.drift_jacobian
uses), while jacrev and unbatched jacfwd are right; tests/test_ph_dmc.py guards this.

Coordinates, as MuJoCo's nested bodies and dm_control's models give them:

    q = (x, theta_1, ..., theta_n) with a cart, (theta_1, ..., theta_n) without.
    x        the cart position along the "side" axis (dm_control: world +x).
    theta_1  the angle of link 1 from the "up" axis (dm_control: world +z), positive
             toward the side axis: link 1 points along (sin theta_1, cos theta_1) in
             (side, up) coordinates. Zero is upright, pi hanging.
    theta_i  (i > 1) the angle of link i relative to link i-1, positive in the same
             sense; link i's absolute angle is phi_i = theta_1 + ... + theta_i.

In dm_control's vertical models (pendulum, acrobot, cartpole) the hinges turn about +y,
links rest along +z and the cart slides along +x, which is exactly this convention. For
the reacher (hinges about +z, links along +x, gravity normal to the plane) take "up" as
+x and "side" as +y with gravity 0: then link i points along (cos phi_i, sin phi_i) in
world (x, y), and the dynamics are the same.

Parameters (PlanarChain), per link i:
    link_masses[i]          m_i, kg
    link_com_distances[i]   c_i, distance from joint i to the link's centre of mass along
                            the link (the centre of mass must lie on the link's line)
    link_inertias[i]        I_i, inertia about the centre of mass, about the joint axis
    link_lengths[i]         l_i, distance from joint i to joint i+1 (the last link's
                            length only places the tip in chain_points)
    joint_damping[i]        viscous damping of joint i on the relative rate theta_dot_i
and for the chain: gravity (m/s^2 along -up; 0 for the reacher), cart_mass and
cart_damping (cart_mass None: fixed base), and input_matrix G (nv x m), the constant map
from the motor inputs u to generalized forces in q (MuJoCo's gear on each actuated
joint, so u is either the torque or dm_control's normalized control, as G says).

The state layout is (q, context, qd): positions, then context_dim episode-constant
context coordinates (zero rate, see StateSpec), then velocities. chain_drift returns
(qd, 0, qdd) in that layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class PlanarChain:
    """Inertial, damping, gravity and actuation parameters of a planar chain (see module)."""

    link_masses: Sequence[float]
    link_com_distances: Sequence[float]
    link_inertias: Sequence[float]
    link_lengths: Sequence[float]
    joint_damping: Sequence[float]
    input_matrix: object
    gravity: float = 0.0
    cart_mass: Optional[float] = None
    cart_damping: float = 0.0

    def __post_init__(self) -> None:
        names = (
            "link_masses",
            "link_com_distances",
            "link_inertias",
            "link_lengths",
            "joint_damping",
        )
        values = {}
        for name in names:
            entries = tuple(float(value) for value in getattr(self, name))
            if not all(math.isfinite(value) for value in entries):
                raise ValueError(f"PlanarChain {name} must be finite.")
            values[name] = entries
            object.__setattr__(self, name, entries)
        count = len(values["link_masses"])
        if count < 1 or any(len(entries) != count for entries in values.values()):
            raise ValueError("PlanarChain needs one entry per link in every link sequence.")
        if min(values["link_masses"]) < 0.0 or min(values["link_inertias"]) < 0.0:
            raise ValueError("Link masses and inertias must be nonnegative.")
        object.__setattr__(self, "gravity", float(self.gravity))
        object.__setattr__(self, "cart_damping", float(self.cart_damping))
        if self.cart_mass is not None:
            object.__setattr__(self, "cart_mass", float(self.cart_mass))
            if not self.cart_mass > 0.0:
                raise ValueError("The cart mass must be positive.")
        nv = count + (self.cart_mass is not None)
        matrix = np.array(self.input_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != nv:
            raise ValueError(f"input_matrix must have shape ({nv}, action_dim).")
        object.__setattr__(self, "input_matrix", matrix)

        masses = np.array(values["link_masses"])
        com = np.array(values["link_com_distances"])
        lengths = np.array(values["link_lengths"])
        inertias = np.array(values["link_inertias"])
        # Mass carried beyond each link's far joint: sum_{i>j} m_i.
        beyond = np.concatenate((np.cumsum(masses[::-1])[::-1][1:], [0.0]))
        # h_j = m_j c_j + l_j sum_{i>j} m_i: link j's first moment about its joint; the
        # potential is g sum_j h_j cos(phi_j) and the cart coupling h_j cos(phi_j).
        first_moments = masses * com + lengths * beyond
        # K_jk = l_min(j,k) h_max(j,k) off the diagonal, J_j on it, so the link block of the
        # mass matrix in absolute angles is K_jk cos(phi_j - phi_k).
        couplings = np.empty((count, count))
        for j in range(count):
            for k in range(count):
                if j == k:
                    couplings[j, k] = inertias[j] + masses[j] * com[j] ** 2 + lengths[j] ** 2 * beyond[j]
                else:
                    couplings[j, k] = lengths[min(j, k)] * first_moments[max(j, k)]
        # z = T q with z = (x, phi): phi = lower-triangular ones times theta.
        transform = np.tril(np.ones((count, count)))
        damping = np.array(values["joint_damping"])
        if self.cart_mass is not None:
            transform = np.block(
                [[np.ones((1, 1)), np.zeros((1, count))], [np.zeros((count, 1)), transform]]
            )
            damping = np.concatenate(([self.cart_damping], damping))
        object.__setattr__(self, "_first_moments", first_moments)
        object.__setattr__(self, "_couplings", couplings)
        object.__setattr__(self, "_transform", transform)
        object.__setattr__(self, "_damping", damping)

    @property
    def n_links(self) -> int:
        return len(self.link_masses)

    @property
    def has_cart(self) -> bool:
        return self.cart_mass is not None

    @property
    def nv(self) -> int:
        """Generalized coordinates: links plus the cart."""
        return self.n_links + int(self.has_cart)

    @property
    def action_dim(self) -> int:
        return int(self.input_matrix.shape[1])

    @property
    def total_mass(self) -> float:
        return float(sum(self.link_masses)) + (self.cart_mass or 0.0)

    @property
    def first_moments(self) -> np.ndarray:
        """h_j = m_j c_j + l_j sum_{i>j} m_i, [n]."""
        return self._first_moments

    @property
    def coupling_matrix(self) -> np.ndarray:
        """K with M_links(phi)_jk = K_jk cos(phi_j - phi_k), [n, n]."""
        return self._couplings

    @property
    def transform(self) -> np.ndarray:
        """T with (x, phi) = T q, [nv, nv]."""
        return self._transform

    @property
    def damping(self) -> np.ndarray:
        """Diagonal of D in q: (cart_damping, joint_damping...) or joint_damping, [nv]."""
        return self._damping


# ------------------------------------------------------------------------------- torch


def _constant(array: np.ndarray, like: Tensor) -> Tensor:
    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def _absolute_angles(chain: PlanarChain, joint_values: Tensor) -> Tensor:
    """phi (or phi_dot) [..., n] from q (or qd) [..., nv]."""
    return torch.cumsum(joint_values[..., int(chain.has_cart) :], dim=-1)


def _absolute_mass(chain: PlanarChain, cos: Tensor, sin: Tensor) -> Tensor:
    """Mass matrix in (x, phi) coordinates, [..., nv, nv]."""
    cos_difference = cos.unsqueeze(-1) * cos.unsqueeze(-2) + sin.unsqueeze(-1) * sin.unsqueeze(-2)
    links = _constant(chain.coupling_matrix, cos) * cos_difference
    if not chain.has_cart:
        return links
    coupling = _constant(chain.first_moments, cos) * cos
    total = torch.ones_like(cos[..., :1]) * chain.total_mass
    top = torch.cat((total, coupling), dim=-1).unsqueeze(-2)
    bottom = torch.cat((coupling.unsqueeze(-1), links), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def chain_mass_matrix(chain: PlanarChain, positions: Tensor) -> Tensor:
    """M(q) in joint coordinates, [..., nv, nv], for positions q [..., nv]."""
    phi = _absolute_angles(chain, positions)
    transform = _constant(chain.transform, positions)
    return transform.mT @ _absolute_mass(chain, phi.cos(), phi.sin()) @ transform


def chain_potential_energy(chain: PlanarChain, positions: Tensor) -> Tensor:
    """Gravity potential g sum_j h_j cos(phi_j), zero with every link horizontal, [..., 1]."""
    phi = _absolute_angles(chain, positions)
    moments = _constant(chain.first_moments, positions)
    return chain.gravity * (moments * phi.cos()).sum(dim=-1, keepdim=True)


def chain_energy(chain: PlanarChain, positions: Tensor, velocities: Tensor) -> Tensor:
    """Kinetic plus potential energy, [..., 1]."""
    mass = chain_mass_matrix(chain, positions)
    kinetic = 0.5 * (velocities.unsqueeze(-2) @ mass @ velocities.unsqueeze(-1)).squeeze(-1)
    return kinetic + chain_potential_energy(chain, positions)


def chain_accelerations(
    chain: PlanarChain, positions: Tensor, velocities: Tensor, torques: Tensor
) -> Tensor:
    """qdd [..., nv] solving M qdd = G u - D qd - C qd - dU/dq, for u = torques [..., m]."""
    phi = _absolute_angles(chain, positions)
    phi_rates = _absolute_angles(chain, velocities)
    cos, sin = phi.cos(), phi.sin()
    transform = _constant(chain.transform, positions)
    moments = _constant(chain.first_moments, positions)
    mass = transform.mT @ _absolute_mass(chain, cos, sin) @ transform

    # Coriolis/centrifugal and gravity forces in (x, phi): for link j,
    # sum_k K_jk sin(phi_j - phi_k) phi_dot_k^2 - g h_j sin(phi_j); for the cart,
    # -sum_j h_j sin(phi_j) phi_dot_j^2. Forces in q are T^T times those in (x, phi).
    squares = phi_rates.square()
    sin_difference = sin.unsqueeze(-1) * cos.unsqueeze(-2) - cos.unsqueeze(-1) * sin.unsqueeze(-2)
    centrifugal = (_constant(chain.coupling_matrix, positions) * sin_difference * squares.unsqueeze(-2)).sum(-1)
    bias = centrifugal - chain.gravity * moments * sin
    if chain.has_cart:
        cart = -(moments * sin * squares).sum(dim=-1, keepdim=True)
        bias = torch.cat((cart, bias), dim=-1)
    applied = torques.to(positions.dtype) @ _constant(chain.input_matrix, positions).T
    damping = _constant(chain.damping, positions) * velocities
    forces = applied - damping - bias @ transform
    # Cholesky, not torch.linalg.solve, whose vmap(jacfwd) derivative is wrong (module).
    factor = torch.linalg.cholesky(mass)
    return torch.cholesky_solve(forces.unsqueeze(-1), factor).squeeze(-1)


def chain_drift(
    chain: PlanarChain, states: Tensor, torques: Tensor, *, context_dim: int = 0
) -> Tensor:
    """(qd, 0, qdd) [..., D] for states (q, context, qd) [..., D], D = 2 nv + context_dim."""
    nv = chain.nv
    if states.shape[-1] != 2 * nv + context_dim:
        raise ValueError(f"Chain states need {2 * nv + context_dim} entries, not {states.shape[-1]}.")
    positions = states[..., :nv]
    velocities = states[..., nv + context_dim :]
    accelerations = chain_accelerations(chain, positions, velocities, torques)
    parts = [velocities]
    if context_dim:
        parts.append(torch.zeros_like(states[..., nv : nv + context_dim]))
    parts.append(accelerations)
    return torch.cat(parts, dim=-1)


def chain_callbacks(
    chain: PlanarChain, *, context_dim: int = 0
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """(drift, diffusion) oracle callbacks for OraclePlant; the diffusion is a zero column."""
    size = 2 * chain.nv + context_dim

    def drift(states: Tensor, torques: Tensor) -> Tensor:
        return chain_drift(chain, states, torques, context_dim=context_dim)

    def diffusion(states: Tensor) -> Tensor:
        return torch.zeros(*states.shape[:-1], size, 1, dtype=states.dtype, device=states.device)

    return drift, diffusion


def chain_points(chain: PlanarChain, positions: Tensor) -> Tensor:
    """(side, up) of the base joint, every later joint and the last link's end, [..., n+1, 2].

    The base joint is at (x, 0) on a cart and at the origin otherwise; joint i+1 lies
    l_i along link i. Add the base's world position to place them.
    """
    phi = _absolute_angles(chain, positions)
    lengths = _constant(np.array(chain.link_lengths), positions)
    steps = torch.stack((phi.sin(), phi.cos()), dim=-1) * lengths.unsqueeze(-1)
    base = torch.zeros_like(steps[..., :1, :])
    if chain.has_cart:
        base = torch.cat((positions[..., :1], torch.zeros_like(positions[..., :1])), dim=-1).unsqueeze(-2)
    return torch.cat((base, base + torch.cumsum(steps, dim=-2)), dim=-2)


# ------------------------------------------------------------------------------- numpy


def _absolute_angles_numpy(chain: PlanarChain, joint_values: np.ndarray) -> np.ndarray:
    return np.cumsum(joint_values[..., int(chain.has_cart) :], axis=-1)


def _absolute_mass_numpy(chain: PlanarChain, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    cos_difference = cos[..., :, None] * cos[..., None, :] + sin[..., :, None] * sin[..., None, :]
    links = chain.coupling_matrix * cos_difference
    if not chain.has_cart:
        return links
    coupling = chain.first_moments * cos
    total = np.full(cos.shape[:-1] + (1,), chain.total_mass)
    top = np.concatenate((total, coupling), axis=-1)[..., None, :]
    bottom = np.concatenate((coupling[..., :, None], links), axis=-1)
    return np.concatenate((top, bottom), axis=-2)


def chain_mass_matrix_numpy(chain: PlanarChain, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    phi = _absolute_angles_numpy(chain, positions)
    transform = chain.transform
    return transform.T @ _absolute_mass_numpy(chain, np.cos(phi), np.sin(phi)) @ transform


def chain_potential_energy_numpy(chain: PlanarChain, positions: np.ndarray) -> np.ndarray:
    phi = _absolute_angles_numpy(chain, np.asarray(positions, dtype=np.float64))
    return chain.gravity * np.sum(chain.first_moments * np.cos(phi), axis=-1, keepdims=True)


def chain_energy_numpy(
    chain: PlanarChain, positions: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    velocities = np.asarray(velocities, dtype=np.float64)
    mass = chain_mass_matrix_numpy(chain, positions)
    kinetic = 0.5 * np.einsum("...i,...ij,...j->...", velocities, mass, velocities)[..., None]
    return kinetic + chain_potential_energy_numpy(chain, positions)


def chain_accelerations_numpy(
    chain: PlanarChain, positions: np.ndarray, velocities: np.ndarray, torques: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    torques = np.asarray(torques, dtype=np.float64)
    phi = _absolute_angles_numpy(chain, positions)
    phi_rates = _absolute_angles_numpy(chain, velocities)
    cos, sin = np.cos(phi), np.sin(phi)
    transform = chain.transform
    moments = chain.first_moments
    mass = transform.T @ _absolute_mass_numpy(chain, cos, sin) @ transform

    squares = phi_rates**2
    sin_difference = sin[..., :, None] * cos[..., None, :] - cos[..., :, None] * sin[..., None, :]
    centrifugal = np.sum(chain.coupling_matrix * sin_difference * squares[..., None, :], axis=-1)
    bias = centrifugal - chain.gravity * moments * sin
    if chain.has_cart:
        cart = -np.sum(moments * sin * squares, axis=-1, keepdims=True)
        bias = np.concatenate((cart, bias), axis=-1)
    forces = torques @ chain.input_matrix.T - chain.damping * velocities - bias @ transform
    return np.linalg.solve(mass, forces[..., None])[..., 0]


def chain_drift_numpy(
    chain: PlanarChain, state: np.ndarray, torque: np.ndarray, *, context_dim: int = 0
) -> np.ndarray:
    """(qd, 0, qdd) for one state [D] or a batch [..., D], as chain_drift."""
    state = np.asarray(state, dtype=np.float64)
    nv = chain.nv
    if state.shape[-1] != 2 * nv + context_dim:
        raise ValueError(f"Chain states need {2 * nv + context_dim} entries, not {state.shape[-1]}.")
    velocities = state[..., nv + context_dim :]
    accelerations = chain_accelerations_numpy(chain, state[..., :nv], velocities, torque)
    context = np.zeros(state.shape[:-1] + (context_dim,))
    return np.concatenate((velocities, context, accelerations), axis=-1)


def chain_points_numpy(chain: PlanarChain, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    phi = _absolute_angles_numpy(chain, positions)
    steps = np.stack((np.sin(phi), np.cos(phi)), axis=-1) * np.array(chain.link_lengths)[:, None]
    base = np.zeros(steps.shape[:-2] + (1, 2))
    if chain.has_cart:
        base[..., 0, 0] = positions[..., 0]
    return np.concatenate((base, base + np.cumsum(steps, axis=-2)), axis=-2)
