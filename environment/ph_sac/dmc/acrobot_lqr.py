"""The upright LQR of the dm_control acrobot and a simulated region of attraction.

The acrobot_swingup reward (environment/ph_sac/dmc/acrobot.py) is the reward of Zhang,
Sathuluri and Zimmermann, "Solving the swing-up and balance task for the Acrobot and
Pendubot with SAC", arXiv:2312.11311 (2023), Eq. 4 and Table I. Its bonus r_lqr is paid
while the state lies in an estimate {e : e^T S e <= rho} of the region of attraction of
the upright under a saturated LQR law, where S is the LQR cost-to-go matrix. The paper
gives the LQR weights (acrobot: Q_lqr = diag(0.97, 0.93, 0.39, 0.26), R_lqr = 0.11) but
not S or rho, which it took from a sums-of-squares estimate for its own robot. This
module computes both for the port's plant, dm_control's acrobot; the table generator
(benchmarks/make_ct_ph_sac_table.py) writes the numbers into the acrobot-swingup row,
which is the record: nothing here runs when an environment is built.

  linearization  upright_linearization(chain): the exact Jacobians A, B of the planar
                 chain's drift (chain.py) at the upright (theta_1 = theta_2 = 0, at rest,
                 zero torque), in the state (theta_1, theta_2, theta_dot_1, theta_dot_2)
                 and the elbow torque in N m: M0 qdd = g T^T diag(h) T q - D qd + G u to
                 first order, with M0 the mass matrix at upright, h the links' first
                 moments and T the map to absolute angles.
  LQR            upright_lqr(chain): S from scipy.linalg.solve_continuous_are(A, B, Q_lqr,
                 R_lqr) and K = R_lqr^{-1} B^T S, so the law is u = -K e.
  rollouts       saturated_lqr_converges: u = clip(-K e, -limit, limit) with e the state
                 with both angles wrapped (the upright is (0, 0, 0, 0)), held over each
                 physics step, the chain's own drift integrated by RK4 at 1 ms (the
                 port's physics). A start converges when e^T S e falls to
                 convergence_fraction x level within the horizon without the upper arm
                 ever passing the horizontal (|theta_1| >= pi/2 counts as fallen).
  rho            region_of_attraction_level: fixed directions (seeded) on the unit level
                 set of S, half on the surface and half uniform inside the ellipsoid
                 (radius fraction U^(1/4) in four dimensions), scaled by sqrt(level). The
                 largest level at which every start converges is bracketed by doubling or
                 halving from initial_level and bisected geometrically to a relative
                 width relative_tolerance; rho is margin (0.9) times the last level at
                 which every start converged, confirmed on an independent sample
                 (check_samples, check_seed; shrunk by margin again until it passes).
  record         acrobot_lqr_reward_terms(config): {"rho": ..., "s11": ..., ..., "s44": ...},
                 the upper triangle of S row by row, for the row's reward_parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

from .chain import PlanarChain, chain_drift_numpy, chain_mass_matrix_numpy

if TYPE_CHECKING:
    from numpy import ndarray

    from common.ph_config import TestbedConfig

# Zhang et al. (2023), Table I, acrobot column: the LQR weights of the region of attraction.
LQR_STATE_WEIGHTS: Tuple[float, ...] = (0.97, 0.93, 0.39, 0.26)
LQR_TORQUE_WEIGHT = 0.11
# The upper-triangular entries of S in the row's reward_parameters, row by row.
COST_TO_GO_KEYS: Tuple[str, ...] = tuple(
    f"s{row + 1}{column + 1}" for row in range(4) for column in range(row, 4)
)
# The port's physics step and integrator (RK4 at 1 ms) for the rollouts.
ROLLOUT_PHYSICS_DT = 0.001
ROLLOUT_HORIZON = 5.0
# Converged: e^T S e at most this fraction of the level tested.
CONVERGENCE_FRACTION = 1e-6
# Fallen: the upper arm at or past the horizontal.
FALLEN_SHOULDER_ANGLE = 0.5 * math.pi


def wrapped_errors(states: ndarray) -> ndarray:
    """(wrap theta_1, wrap theta_2, theta_dot_1, theta_dot_2): the error from the upright."""
    states = np.asarray(states, dtype=np.float64)
    angles = np.arctan2(np.sin(states[..., :2]), np.cos(states[..., :2]))
    return np.concatenate((angles, states[..., 2:4]), axis=-1)


def cost_to_go_matrix(entries: Dict[str, float]) -> ndarray:
    """The symmetric 4 x 4 S from its upper-triangular entries s11, s12, ..., s44."""
    matrix = np.empty((4, 4))
    for row in range(4):
        for column in range(row, 4):
            value = float(entries[f"s{row + 1}{column + 1}"])
            matrix[row, column] = matrix[column, row] = value
    return matrix


def cost_to_go_entries(matrix: ndarray) -> Dict[str, float]:
    """The upper-triangular entries of a symmetric 4 x 4 S, keyed s11, s12, ..., s44."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return {
        f"s{row + 1}{column + 1}": float(matrix[row, column])
        for row in range(4)
        for column in range(row, 4)
    }


def upright_linearization(chain: PlanarChain) -> Tuple[ndarray, ndarray]:
    """A [4, 4] and B [4, 1] of the chain's drift at the upright at rest, zero input."""
    if chain.n_links != 2 or chain.has_cart:
        raise ValueError("upright_linearization expects the two-link acrobot chain.")
    mass = chain_mass_matrix_numpy(chain, np.zeros(2))
    transform = chain.transform
    # Gravity force -dU/dq = g T^T (h sin phi) ~ g T^T diag(h) T q near phi = 0.
    stiffness = chain.gravity * transform.T @ np.diag(chain.first_moments) @ transform
    position = np.linalg.solve(mass, stiffness)
    velocity = -np.linalg.solve(mass, np.diag(chain.damping))
    a = np.block([[np.zeros((2, 2)), np.eye(2)], [position, velocity]])
    b = np.vstack((np.zeros((2, chain.action_dim)), np.linalg.solve(mass, chain.input_matrix)))
    return a, b


def upright_lqr(
    chain: PlanarChain,
    state_weights: Sequence[float] = LQR_STATE_WEIGHTS,
    torque_weight: float = LQR_TORQUE_WEIGHT,
) -> Tuple[ndarray, ndarray]:
    """(S [4, 4], K [1, 4]) of the continuous-time LQR at the upright; the law is u = -K e."""
    from scipy.linalg import solve_continuous_are

    a, b = upright_linearization(chain)
    weights = np.diag(np.asarray(state_weights, dtype=np.float64))
    control = np.array([[float(torque_weight)]])
    cost_to_go = solve_continuous_are(a, b, weights, control)
    cost_to_go = 0.5 * (cost_to_go + cost_to_go.T)
    gain = np.linalg.solve(control, b.T @ cost_to_go)
    return cost_to_go, gain


def _rk4_step(chain: PlanarChain, states: ndarray, torques: ndarray, dt: float) -> ndarray:
    """One RK4 step of the chain's drift with the torques held over it."""
    k1 = chain_drift_numpy(chain, states, torques)
    k2 = chain_drift_numpy(chain, states + 0.5 * dt * k1, torques)
    k3 = chain_drift_numpy(chain, states + 0.5 * dt * k2, torques)
    k4 = chain_drift_numpy(chain, states + dt * k3, torques)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def saturated_lqr_converges(
    chain: PlanarChain,
    cost_to_go: ndarray,
    gain: ndarray,
    starts: ndarray,
    *,
    torque_limit: float,
    threshold: float,
    horizon: float = ROLLOUT_HORIZON,
    physics_dt: float = ROLLOUT_PHYSICS_DT,
    stop_at_first_failure: bool = False,
) -> ndarray:
    """Which starts [N, 4] reach e^T S e <= threshold within the horizon, bool [N].

    u = clip(-K e, -torque_limit, torque_limit) with e = wrapped_errors(state), held over
    each physics step of RK4 on the chain's drift. A start whose upper arm reaches the
    horizontal (|wrap theta_1| >= pi/2) has fallen and does not converge; one still
    moving at the horizon does not converge either. stop_at_first_failure ends every
    rollout once one start has fallen (the remaining starts then read False).
    """
    states = np.array(starts, dtype=np.float64, copy=True).reshape(-1, 4)
    converged = np.zeros(len(states), dtype=bool)
    active = np.arange(len(states))
    for _ in range(int(round(horizon / physics_dt))):
        if not active.size:
            break
        current = states[active]
        torques = np.clip(-(wrapped_errors(current) @ gain.T), -torque_limit, torque_limit)
        current = _rk4_step(chain, current, torques, physics_dt)
        states[active] = current
        errors = wrapped_errors(current)
        values = np.einsum("ni,ij,nj->n", errors, cost_to_go, errors)
        reached = values <= threshold
        fallen = (np.abs(errors[:, 0]) >= FALLEN_SHOULDER_ANGLE) & ~reached
        converged[active[reached]] = True
        if stop_at_first_failure and fallen.any():
            break
        active = active[~(reached | fallen)]
    return converged


def level_set_directions(cost_to_go: ndarray, samples: int, seed: int) -> ndarray:
    """2 x samples errors with e^T S e = 1 (the first half) or uniform inside it, [2N, 4]."""
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(cost_to_go)
    normals = rng.standard_normal((2 * samples, 4))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[samples:] *= rng.uniform(0.0, 1.0, (samples, 1)) ** 0.25
    # e = L^{-T} z gives e^T S e = z^T L^{-1} L L^T L^{-T} z = |z|^2.
    return np.linalg.solve(factor.T, normals.T).T


def region_of_attraction_level(
    chain: PlanarChain,
    cost_to_go: ndarray,
    gain: ndarray,
    *,
    torque_limit: float,
    samples: int = 256,
    seed: int = 0,
    check_samples: int = 1024,
    check_seed: int = 1,
    initial_level: float = 1.0,
    relative_tolerance: float = 1e-3,
    margin: float = 0.9,
    horizon: float = ROLLOUT_HORIZON,
    physics_dt: float = ROLLOUT_PHYSICS_DT,
) -> float:
    """rho: margin x the largest level of e^T S e at which every sampled start converges.

    See the module docstring: seeded directions on and inside the unit level set, scaled
    by sqrt(level); a bracket by doubling or halving, geometric bisection to the relative
    tolerance, the margin, and a confirmation on an independent sample.
    """

    def every_start_converges(level: float, directions: ndarray) -> bool:
        return bool(
            saturated_lqr_converges(
                chain,
                cost_to_go,
                gain,
                math.sqrt(level) * directions,
                torque_limit=torque_limit,
                threshold=CONVERGENCE_FRACTION * level,
                horizon=horizon,
                physics_dt=physics_dt,
                stop_at_first_failure=True,
            ).all()
        )

    directions = level_set_directions(cost_to_go, samples, seed)
    low: Optional[float] = None
    high: Optional[float] = None
    level = float(initial_level)
    while low is None or high is None:
        if every_start_converges(level, directions):
            low = level
            level *= 2.0
        else:
            high = level
            level *= 0.5
        if level < 1e-12 or level > 1e12:
            raise RuntimeError("No bracket of the region of attraction was found.")
    while high / low > 1.0 + relative_tolerance:
        middle = math.sqrt(low * high)
        if every_start_converges(middle, directions):
            low = middle
        else:
            high = middle
    level = margin * low
    check = level_set_directions(cost_to_go, check_samples, check_seed)
    while not every_start_converges(level, check):
        level *= margin
    return level


def acrobot_lqr_reward_terms(config: TestbedConfig) -> Dict[str, float]:
    """{"rho", "s11", ..., "s44"} of the config's plant: the Riccati S and simulated rho.

    The chain is the config's compiled acrobot (environment/ph_sac/dmc/acrobot.py), the
    torque limit the config's elbow motor limit (dm_control's 2 N m).
    """
    from .acrobot import acrobot_chain

    chain = acrobot_chain(config)
    cost_to_go, gain = upright_lqr(chain)
    controller = config["controller"]
    limit = float(min(-controller["torque_low"][0], controller["torque_high"][0]))
    rho = region_of_attraction_level(chain, cost_to_go, gain, torque_limit=limit)
    return {"rho": rho, **cost_to_go_entries(cost_to_go)}
