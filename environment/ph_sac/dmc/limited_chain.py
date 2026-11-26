"""Planar-chain dynamics with MuJoCo's soft joint limits: the oracle of a chain port that
keeps a dm_control joint limit (the reacher's wrist, the multi-pole cart-poles' slider).

The smooth acceleration is chain.chain_accelerations (M(q) qdd = G u - D qd - C qd - g);
the limit forces follow from joint_limits.constrained_accelerations with the Cholesky
factor of chain.chain_mass_matrix, exactly as MuJoCo's forward dynamics adds them
(qacc = qacc_smooth + M^-1 J^T f, f >= 0 from the one-sided complementarity of the
violated limits). Inside every range no limit row exists and the force is exactly 0, so
the drift equals chain.chain_drift there (the numpy twin returns chain_drift_numpy's
value unchanged for such states). The chain's coordinates must be MuJoCo's dofs in order
(q = (x, theta_1, ..., theta_n) with a cart), as joint_limits_from_model reads them.
torch.func-compatible: Cholesky solves and fixed pivoting passes only.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .chain import (
    PlanarChain,
    chain_accelerations,
    chain_drift_numpy,
    chain_mass_matrix,
    chain_mass_matrix_numpy,
)
from .joint_limits import JointLimits, constrained_accelerations, constrained_accelerations_numpy


def limited_chain_accelerations(
    chain: PlanarChain,
    limits: Optional[JointLimits],
    positions: Tensor,
    velocities: Tensor,
    torques: Tensor,
) -> Tensor:
    """qdd [..., nv] with the soft joint-limit forces (limits None: the smooth chain)."""
    smooth = chain_accelerations(chain, positions, velocities, torques)
    if limits is None:
        return smooth
    factor = torch.linalg.cholesky(chain_mass_matrix(chain, positions))
    return constrained_accelerations(limits, factor, smooth, positions, velocities)["qacc"]


def limited_chain_drift(
    chain: PlanarChain,
    limits: Optional[JointLimits],
    states: Tensor,
    torques: Tensor,
    *,
    context_dim: int = 0,
) -> Tensor:
    """(qd, 0, qdd) [..., D] for states (q, context, qd), as chain.chain_drift, limits included."""
    nv = chain.nv
    if states.shape[-1] != 2 * nv + context_dim:
        raise ValueError(f"Chain states need {2 * nv + context_dim} entries, not {states.shape[-1]}.")
    positions = states[..., :nv]
    velocities = states[..., nv + context_dim :]
    accelerations = limited_chain_accelerations(chain, limits, positions, velocities, torques)
    parts = [velocities]
    if context_dim:
        parts.append(torch.zeros_like(states[..., nv : nv + context_dim]))
    parts.append(accelerations)
    return torch.cat(parts, dim=-1)


def limited_chain_callbacks(
    chain: PlanarChain, limits: Optional[JointLimits], *, context_dim: int = 0
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """(drift, diffusion) oracle callbacks for OraclePlant; the diffusion is a zero column."""
    size = 2 * chain.nv + context_dim

    def drift(states: Tensor, torques: Tensor) -> Tensor:
        return limited_chain_drift(chain, limits, states, torques, context_dim=context_dim)

    def diffusion(states: Tensor) -> Tensor:
        return torch.zeros(*states.shape[:-1], size, 1, dtype=states.dtype, device=states.device)

    return drift, diffusion


def limits_engaged_numpy(limits: JointLimits, positions: np.ndarray) -> np.ndarray:
    """Boolean [...]: some limited joint of the positions [..., nv] lies past its range
    (a constraint row exists: pos < margin)."""
    q = np.asarray(positions, dtype=np.float64)[..., limits.dofs]
    return np.any((limits.upper - q < limits.margin) | (q - limits.lower < limits.margin), axis=-1)


def limited_chain_drift_numpy(
    chain: PlanarChain,
    limits: Optional[JointLimits],
    state: np.ndarray,
    torque: np.ndarray,
    *,
    context_dim: int = 0,
) -> np.ndarray:
    """The numpy twin of limited_chain_drift for one state [D] or a batch [..., D].

    States with no limit row keep chain_drift_numpy's value; the others get the limit
    forces from constrained_accelerations_numpy, state by state.
    """
    state = np.asarray(state, dtype=np.float64)
    drift = chain_drift_numpy(chain, state, torque, context_dim=context_dim)
    if limits is None:
        return drift
    nv = chain.nv
    flat_states = state.reshape(-1, state.shape[-1])
    flat_drift = drift.reshape(-1, drift.shape[-1]).copy()
    engaged = limits_engaged_numpy(limits, flat_states[:, :nv])
    for row in np.flatnonzero(engaged):
        positions = flat_states[row, :nv]
        velocities = flat_states[row, nv + context_dim :]
        mass = chain_mass_matrix_numpy(chain, positions)
        result = constrained_accelerations_numpy(
            limits, mass, flat_drift[row, nv + context_dim :], positions, velocities
        )
        flat_drift[row, nv + context_dim :] = result["qacc"]
    return flat_drift.reshape(drift.shape)
