"""Model integration of a plant over decision intervals, separate from real MuJoCo physics.

Model fitting, rate rollouts, admission paths and plant diagnostics share the same
physical noise factor, substep map, substep grid and controller refresh times.

Substep map. model_substep is the one place that advances a plant by one substep of
length h; its integrator is the config key model_integrator (values
common.ph_config.MODEL_INTEGRATORS, default "rk4"):

  "rk4"            s' = Phi_h(s) + K_h Sigma(s) dW (rk4_step), Phi_h the classical
                   fourth-order Runge-Kutta map of the closed-loop drift: stages
                   k1 = b(s_1, a_1), k2 = b(s_2, a_2), k3 = b(s_3, a_3), k4 = b(s_4, a_4)
                   at s_1 = s, s_2 = s + h k1 / 2, s_3 = s + h k2 / 2, s_4 = s + h k3 and
                   Phi_h = s + h (k1 + 2 k2 + 2 k3 + k4) / 6. Under refresh_mode
                   "continuous" (a feedback law, the default) a_i = g(s_i, kappa), the
                   law at each stage's state; otherwise a_i = a, the torque held through
                   the stages. Either way this is MuJoCo's RK4 step as the environment
                   runs it (mjINT_RK4, positions advanced by the stage velocities, the
                   control callback setting ctrl at each stage or ctrl held), so with the
                   oracle drift and h the physics step it reproduces a testbed simulated
                   with RK4 to rounding error.
  "semi_implicit"  v' = v + h a(s) + Sigma_v dW, y' = y + Sigma_y dW + h P v'
                   (semi_implicit_step), the velocity-first semi-implicit
                   Euler-Maruyama step: the only map before 2026-11-09, kept so that
                   earlier runs stay reproducible.

Both maps add the same noise increment K_h Sigma(s) dW, with Sigma at the substep's
start, K_h = I + h J_kin carrying the velocity kick into the position through h P, and
dW ~ N(0, h I) already scaled (sample_brownian_increments): the noise model and the
leading-order endpoint covariance Delta A(s) do not depend on the integrator, only the
deterministic map does. With covariance_mode "propagated" the recursion
F Lambda F^T + h K_h A K_h^T uses the Jacobian F of the chosen map.

Controller torque. The controller's refresh_mode (common.ph_config.REFRESH_MODES) sets
when a feedback law g(s, kappa) (the environment's PD law and clip) is evaluated:

  "continuous"  at every stage of every substep from the stage's state, as the
                environment's MuJoCo control callback does at every RK4 stage of every
                physics step: rk4_step's four stages, semi_implicit_step's single one
                (the substep's start). The grid has no refresh points
                (grid_refresh_steps None): it splits each interval at the interval end
                and into equal substeps of at most model_step only, so a model step above
                the physics step is a coarser RK4 of the same closed loop, and at the
                physics step the oracle reproduces the environment to rounding. The
                step Jacobians (_rk4_closed_loop_jacobian, augmented_closed_loop_step,
                path_jacobian_norms) carry the law's state dependence dg/ds at every
                stage; the augmented held-torque block of the propagated covariance stays
                zero, since no torque is held.
  "held"        at the start of every substep the refresh grid flags
                (refresh_aligned_grid: every controller refresh, i.e. every
                refresh_steps physics steps) the torque is controller.action of the
                substep's starting state; otherwise the previous torque is held. It is
                never re-evaluated inside a substep or between RK4 stages, as the "held"
                environment holds it through each physics step; model_step above the
                refresh period has no effect. Every run before 2026-11-09.

A controller without state feedback (TorqueController, decision_mode "torque") is
evaluated at each interval's first substep only (controller_refresh_flags) in either
mode: its torque, the clipped decision, is held over the whole interval as the
torque-mode environment holds it; the mode only sets its grid as for a feedback law.

Contact-triggered fine substepping (config key model_coarse_step, ContactSubstepping,
since 2026-11-10; off unless given, and then every function below is bitwise what it was).
On a task with joint limits the model step model_step is the physics step, because an
interval that touches a limit is accurate only there; with a ContactSubstepping (contact=)
integrate_endpoint_moments, stochastic_rollout and path_jacobian_norms first integrate
every row at the coarse step, flag each row whose coarse path comes near a limit
(contact_flags, decided per row and per decision interval: one call integrates one
interval per row, so a multi-interval use decides again for each interval), integrate the
flagged rows again over the whole interval at model_step (the caller's maximum_step, the
fine step) and put their results in place of the coarse ones. The selection is discrete:
no gradient flows through the mask, and every output, its gradient (create_graph) and its
propagated covariance come from whichever integration a row used. Gathering and
scattering keep the rows' order (no data-dependent nondeterminism beyond the mask), so
with no row flagged the result is the plain coarse integration's and with every row
flagged the plain fine integration's, bitwise. Noise lives on the fine grid (the caller's
increments, drawn at model_step as without the scheme); a coarse substep takes the sum of
the fine increments it spans, the same Brownian path, so a re-integrated path keeps its
noise. In stochastic_rollout a row is a path (each of the L paths of a state is flagged on
its own), and each path's reward is integrated on the knots of the step it used. The
scheme needs refresh_mode "continuous" (no refresh points on the grid) and a coarse grid
that nests in the fine one (the coarse step divides the interval).

Coordinates. Positions are integrated unwrapped within an interval. The drift reads a
periodic position only through 2 pi-periodic functions (sine and cosine features of the
learned plant, trigonometric oracles), and every registered testbed wraps the setpoint
error of an actuated coordinate it observes wrapped, so neither the drift nor the torque
depends on the branch; endpoint residuals are wrapped (wrapped_endpoint_residual).
Context coordinates have zero rows in P, in the drift and in the diffusion, so both
maps leave them exactly constant.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Mapping, Optional, Tuple, TypedDict

import numpy as np
import torch

from common.ph_config import (
    CONTACT_MARGIN_FRACTION,
    CONTACT_VELOCITY_FACTOR,
    DEFAULT_MODEL_INTEGRATOR,
    EXCLUSION_ACCELERATION_FACTOR,
    EXCLUSION_MARGIN_FRACTION,
    EXCLUSION_PATH_VELOCITY_FACTOR,
    EXCLUSION_RULE_VERSION,
    EXCLUSION_VELOCITY_FACTOR,
    check_model_integrator,
    contact_handling_setting,
    contact_substepping_setting,
    exclusion_model_step,
)

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_types import StateSpec
    from environment.ph_sac.controller import Controller
    from models.ph_sac.networks import SmoothValueHead
    from models.ph_sac.plant import Plant

WHOLE_STEP_TOLERANCE = 1e-3


class EndpointMoments(TypedDict):
    """Noise-free mean proxy and first-order physical covariance, not the exact law.

    step_counts has shape [B]; step_sizes holds a zero-padded [B,N_max]
    refresh-aligned grid, allowing unequal substeps. drift_only marks the moments of
    a plant without a diffusion model (zero covariance); plant_likelihood_loss then
    fits the drift alone. Hand-built moments may omit it, meaning False. Under contact
    substepping each row's step_counts and step_sizes are those of the grid it used
    (padded to the widest), and contact_rows [B] (bool, only present then) flags the rows
    integrated at the fine step.
    """

    means: Tensor
    covariances: Tensor
    step_counts: Tensor
    step_sizes: Tensor
    saturation_fractions: Tensor
    drift_only: bool


class ReturnTerms(TypedDict):
    """Known task terms each model path earns on its own simulated states."""

    reward_rate: Callable[[Tensor, Tensor], Tensor]
    terminal_payoff: Callable[[Tensor], Tensor]
    discount_rate: float


class RolloutBatch(TypedDict):
    """Stopped path endpoints and shared kicks on the same padded substep grid.

    running_rewards and failure_payoffs are double [B,L,1], zero without ReturnTerms.
    Under contact substepping brownian_increments, step_counts and step_sizes are those
    of the fine grid (the caller's maximum_step, on which the increments live), and
    contact_paths [B,L,1] (bool, only present then) flags the paths integrated on it;
    the others used the coarse grid with the summed increments.
    """

    endpoints: Tensor
    survived: Tensor
    failure_times: Tensor
    durations_used: Tensor
    running_rewards: Tensor
    failure_payoffs: Tensor
    brownian_increments: Tensor
    step_counts: Tensor
    step_sizes: Tensor


@functools.lru_cache(maxsize=4096)
def _segment_grid(
    physics_steps: int, physics_dt: float, maximum_step: float, refresh_steps: int
) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
    """Substeps and refresh flags of one interval, in whole-step arithmetic."""
    sizes: list[float] = []
    flags: list[bool] = []
    for start in range(0, physics_steps, refresh_steps):
        length = min(refresh_steps, physics_steps - start)
        pieces = max(1, math.ceil(length * physics_dt / maximum_step - 1e-9))
        sizes += [length * physics_dt / pieces] * pieces
        flags += [True] + [False] * (pieces - 1)
    return tuple(sizes), tuple(flags)


@functools.lru_cache(maxsize=4096)
def _continuous_grid(
    physics_steps: int, physics_dt: float, maximum_step: float
) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
    """Equal substeps of at most maximum_step over a whole interval; one flag at its start.

    A substep of a whole number of physics steps is that multiple of physics_dt, so a
    model step equal to the physics step gives the "held" grid of refresh_steps 1 exactly.
    """
    pieces = max(1, math.ceil(physics_steps * physics_dt / maximum_step - 1e-9))
    if physics_steps % pieces == 0:
        size = (physics_steps // pieces) * physics_dt
    else:
        size = physics_steps * physics_dt / pieces
    return (size,) * pieces, (True,) + (False,) * (pieces - 1)


def grid_refresh_steps(
    controller: Controller, controller_refresh_steps: Optional[int]
) -> Optional[int]:
    """The refresh cadence that lays out the model grid of controller's intervals.

    None (no refresh points: each interval splits into equal substeps of at most the
    model step) under refresh_mode "continuous", for a feedback law evaluated at every
    stage and for a held torque alike; controller_refresh_steps under "held". A
    controller without refresh_mode (a hand-built test double) counts as "held".
    """
    if getattr(controller, "refresh_mode", "held") == "continuous":
        return None
    return controller_refresh_steps


def stage_feedback(controller: Controller) -> bool:
    """Whether controller's torque is its law at every stage's state (a continuous law)."""
    return bool(getattr(controller, "continuous", False))


def refresh_aligned_grid(
    durations: Tensor,
    maximum_step: float,
    *,
    physics_dt: Optional[float] = None,
    controller_refresh_steps: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return step counts [B], padded sizes [B,N_max] and refresh flags [B,N_max].

    With physics_dt and controller_refresh_steps, every controller refresh lies on the
    grid and a refresh segment splits into equal substeps of at most maximum_step
    (refresh_mode "held"). With physics_dt alone (controller_refresh_steps None, as
    grid_refresh_steps gives for refresh_mode "continuous") the interval has no refresh
    points and splits into equal substeps of at most maximum_step; its first substep is
    flagged. Durations must be whole physics steps whenever physics_dt is given. Without
    clock arguments, each row takes N=ceil(Delta/h) equal steps, all flagged.
    """
    flat = durations.reshape(-1)
    if physics_dt is None and controller_refresh_steps is not None:
        raise ValueError("Give physics_dt with controller_refresh_steps.")
    if physics_dt is None:
        counts = torch.ceil(flat.double() / maximum_step - 1e-6).clamp_min(1).long()
        widest = int(counts.max())
        active = torch.arange(widest, device=flat.device).unsqueeze(0) < counts.unsqueeze(1)
        sizes = (flat / counts.to(flat.dtype)).unsqueeze(1) * active.to(flat.dtype)
        return counts, sizes, active

    ratio = flat.double() / physics_dt
    whole = ratio.round()
    if torch.any((ratio - whole).abs() > WHOLE_STEP_TOLERANCE) or torch.any(whole < 1):
        raise ValueError(
            f"Durations must be positive multiples of the physics step {physics_dt}."
        )
    unique, inverse = torch.unique(whole.long(), return_inverse=True)
    grids = [
        _continuous_grid(int(steps), physics_dt, maximum_step)
        if controller_refresh_steps is None
        else _segment_grid(
            int(steps), physics_dt, maximum_step, int(controller_refresh_steps)
        )
        for steps in unique.tolist()
    ]
    widest = max(len(grid[0]) for grid in grids)
    table_sizes = np.zeros((len(grids), widest))
    table_flags = np.zeros((len(grids), widest), dtype=bool)
    table_counts = np.zeros(len(grids), dtype=np.int64)
    for row, (sizes, flags) in enumerate(grids):
        table_counts[row] = len(sizes)
        table_sizes[row, : len(sizes)] = sizes
        table_flags[row, : len(flags)] = flags
    device = flat.device
    return (
        torch.as_tensor(table_counts, device=device)[inverse],
        torch.as_tensor(table_sizes, dtype=flat.dtype, device=device)[inverse],
        torch.as_tensor(table_flags, device=device)[inverse],
    )


def controller_refresh_flags(controller: Controller, flags: Tensor) -> Tensor:
    """The substeps [B,N_max] at which controller.action sets the held torque.

    A feedback law (state_feedback True) is re-evaluated at every flagged refresh. A
    controller without state feedback (TorqueController) is evaluated at each row's
    first substep only: its torque is constant over the interval, so refreshing it
    would change nothing, and holding it is what the torque-mode environment does.
    """
    if getattr(controller, "state_feedback", True):
        return flags
    first = torch.zeros_like(flags)
    first[:, :1] = True
    return flags & first


def integration_grid(
    durations: Tensor,
    maximum_step: float,
    *,
    physics_dt: Optional[float] = None,
    controller_refresh_steps: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Return counts and padded sizes of refresh_aligned_grid (refreshes on the grid when
    controller_refresh_steps is given, none when it is None)."""
    counts, sizes, _ = refresh_aligned_grid(
        durations,
        maximum_step,
        physics_dt=physics_dt,
        controller_refresh_steps=controller_refresh_steps,
    )
    return counts, sizes


def _projection(state_spec: StateSpec, reference: Tensor) -> Tensor:
    return torch.as_tensor(
        state_spec["observation_projection"],
        dtype=reference.dtype,
        device=reference.device,
    )


def kinematic_noise_transport(state_spec: StateSpec, step_sizes: Tensor) -> Tensor:
    """Return K_h=I+h*J_kin for J_kin=[[0,P],[0,0]] in s=(observed q,velocity)."""
    n_o = int(state_spec["position_dim"])
    n = int(state_spec["velocity_dim"])
    h = step_sizes.reshape(-1, 1, 1)
    top = torch.cat((h.new_zeros(n_o, n_o), _projection(state_spec, h)), dim=-1)
    kinematic = torch.cat((top, h.new_zeros(n, n_o + n)), dim=0)
    return torch.eye(n_o + n, dtype=h.dtype, device=h.device) + h * kinematic


def physical_noise_injection(
    plant: Plant, states: Tensor, step_sizes: Tensor
) -> Tensor:
    """Return B=K_h Sigma, retaining direct position noise and transported velocity kicks."""
    return kinematic_noise_transport(plant.state_spec, step_sizes) @ plant.diffusion(
        states
    )


def semi_implicit_step(
    plant: Plant,
    states: Tensor,
    actions: Tensor,
    step_sizes: Tensor,
    *,
    brownian_increments: Optional[Tensor] = None,
) -> Tensor:
    """Update v then y; supplied dW~N(0,hI) is already scaled and is not rescaled again.

    Use the same dW in Sigma_v*dW and Sigma_y*dW, including the velocity
    kick transported into the new position through h*P*v_next.
    """
    n_o = plant.position_dim
    h = step_sizes.reshape(-1, 1)
    projection = plant.observation_projection.to(states.dtype)
    positions, velocities = states[:, :n_o], states[:, n_o:]
    acceleration = plant.drift(states, actions)[:, n_o:]
    next_velocities = velocities + h * acceleration
    next_positions = positions
    if brownian_increments is not None:
        kicks = (plant.diffusion(states) @ brownian_increments.unsqueeze(-1)).squeeze(-1)
        next_velocities = next_velocities + kicks[:, n_o:]
        next_positions = next_positions + kicks[:, :n_o]
    next_positions = next_positions + h * (next_velocities @ projection.T)
    return torch.cat((next_positions, next_velocities), dim=-1)


# RK4 stage offsets c_2..c_4 (stage i is evaluated at s + c_i h k_(i-1)) and the weights
# w_1..w_4 of s' = s + h (w_1 k_1 + ... + w_4 k_4) / 6.
_RK4_OFFSETS = (0.5, 0.5, 1.0)
_RK4_WEIGHTS = (1.0, 2.0, 2.0, 1.0)


def rk4_step(
    plant: Plant,
    states: Tensor,
    actions: Tensor,
    step_sizes: Tensor,
    *,
    brownian_increments: Optional[Tensor] = None,
    feedback: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Classical RK4 of the closed-loop drift, plus the noise increment K_h Sigma dW.

    s' = s + h (k1 + 2 k2 + 2 k3 + k4) / 6 with k1 = b(s, a_1), k2 = b(s_2, a_2),
    k3 = b(s_3, a_3), k4 = b(s_4, a_4), s_2 = s + h k1 / 2, s_3 = s + h k2 / 2,
    s_4 = s + h k3. Without feedback every a_i is actions, the torque held through the
    stages (MuJoCo's mjINT_RK4 step with ctrl held); with feedback (a function of the
    stage states, the controller's law at the substep's resolved decisions) a_1 is
    actions, which must be feedback(s), and a_i = feedback(s_i) at the later stages
    (MuJoCo's step with the control callback). In observation coordinates (the drift's
    position block P v is linear, so stepping y = P q is stepping q). With
    already-scaled dW ~ N(0, h I) it adds K_h Sigma(s) dW, the increment
    semi_implicit_step adds (Sigma at the step's start, the velocity kick carried into
    the position through h P). Ordinary tensor operations only: autograd reaches the
    plant's parameters through all four stages, and h = 0 (grid padding) returns s
    unchanged.
    """
    h = step_sizes.reshape(-1, 1)
    slope = plant.drift(states, actions)
    total = slope
    for offset, weight in zip(_RK4_OFFSETS, _RK4_WEIGHTS[1:]):
        point = states + (offset * h) * slope
        slope = plant.drift(point, actions if feedback is None else feedback(point))
        total = total + weight * slope
    following = states + (h / 6.0) * total
    if brownian_increments is not None:
        injection = physical_noise_injection(plant, states, step_sizes)
        following = following + (injection @ brownian_increments.unsqueeze(-1)).squeeze(-1)
    return following


def model_substep(
    plant: Plant,
    states: Tensor,
    actions: Tensor,
    step_sizes: Tensor,
    *,
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
    brownian_increments: Optional[Tensor] = None,
    feedback: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Advance [B,D] states by one substep [B] from torques [B,m] with the chosen map.

    "rk4" is rk4_step, "semi_implicit" semi_implicit_step; both take the same already
    scaled dW [B,K] (None: noise-free) and add the same noise increment K_h Sigma dW.
    actions is the torque at the substep's start. feedback (refresh_mode "continuous":
    the law at the substep's resolved decisions, with actions = feedback(states))
    makes rk4_step evaluate the torque at each later stage's state; the semi-implicit
    map has one stage, at the start, so it reads actions alone. Without feedback the
    torque is held through the substep. Every model integration over a decision
    interval goes through this function.
    """
    if check_model_integrator(integrator) == "rk4":
        return rk4_step(
            plant,
            states,
            actions,
            step_sizes,
            brownian_increments=brownian_increments,
            feedback=feedback,
        )
    return semi_implicit_step(
        plant, states, actions, step_sizes, brownian_increments=brownian_increments
    )


def _rk4_step_jacobian(
    plant: Plant, states: Tensor, actions: Tensor, step_sizes: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return Phi_h(s; a) [B,D], dPhi_h/ds [B,D,D] and dPhi_h/da [B,D,m] of one RK4 step.

    The chain rule through the stages: stage i is evaluated at s_i = s + c_i h k_(i-1)
    (s_1 = s), so dk_i/ds = b_s(s_i) ds_i/ds with ds_i/ds = I + c_i h dk_(i-1)/ds, and
    dk_i/da = b_s(s_i) ds_i/da + b_a(s_i) with ds_i/da = c_i h dk_(i-1)/da; Phi_h and its
    Jacobians are the weighted sums s + h sum_i w_i k_i / 6. Each stage reads
    plant.drift_jacobian, so the Jacobians stay differentiable in the parameters.
    """
    h = step_sizes.reshape(-1, 1, 1)
    batch, size = states.shape[0], states.shape[-1]
    eye = torch.eye(size, dtype=states.dtype, device=states.device)
    point = states
    point_by_state = eye.expand(batch, size, size)
    point_by_action = states.new_zeros(batch, size, plant.action_dim)
    total = states.new_zeros(states.shape)
    total_by_state = states.new_zeros(batch, size, size)
    total_by_action = states.new_zeros(batch, size, plant.action_dim)
    for stage, weight in enumerate(_RK4_WEIGHTS):
        slope, by_state, by_action = plant.drift_jacobian(point, actions)
        slope_by_state = by_state @ point_by_state
        slope_by_action = by_state @ point_by_action + by_action
        total = total + weight * slope
        total_by_state = total_by_state + weight * slope_by_state
        total_by_action = total_by_action + weight * slope_by_action
        if stage < len(_RK4_OFFSETS):
            offset = _RK4_OFFSETS[stage] * h
            point = states + offset[:, 0] * slope
            point_by_state = eye + offset * slope_by_state
            point_by_action = offset * slope_by_action
    return (
        states + (h[:, 0] / 6.0) * total,
        eye + (h / 6.0) * total_by_state,
        (h / 6.0) * total_by_action,
    )


def _stage_law(
    controller: Controller, resolved_decisions: Tensor
) -> Callable[[Tensor], Tensor]:
    """The law at fixed resolved decisions as a function of stage states (rk4_step's
    feedback under refresh_mode "continuous")."""

    def law(points: Tensor) -> Tensor:
        return controller.action(points, resolved_decisions)

    return law


def _rk4_closed_loop_jacobian(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    resolved_decisions: Tensor,
    step_sizes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a_1 = g(s) [B,m], Phi_h(s) [B,D] and dPhi_h/ds [B,D,D] of one closed-loop RK4 step.

    The continuous-law version of _rk4_step_jacobian: stage i reads the torque
    a_i = g(s_i, kappa) of its own state, so its slope's state derivative is the
    closed-loop db/ds + db/da dg/ds at s_i (_action_state_jacobian, piecewise constant
    for the clipped PD law), chained through ds_i/ds = I + c_i h dk_(i-1)/ds. The map
    does not depend on a held torque, so there is no action derivative. Each stage
    reads plant.drift_jacobian, so the Jacobian stays differentiable in the parameters.
    """
    h = step_sizes.reshape(-1, 1, 1)
    batch, size = states.shape[0], states.shape[-1]
    eye = torch.eye(size, dtype=states.dtype, device=states.device)
    point = states
    point_by_state = eye.expand(batch, size, size)
    total = states.new_zeros(states.shape)
    total_by_state = states.new_zeros(batch, size, size)
    first = None
    for stage, weight in enumerate(_RK4_WEIGHTS):
        torque = controller.action(point, resolved_decisions)
        if first is None:
            first = torque
        slope, by_state, by_action = plant.drift_jacobian(point, torque)
        closed = by_state + by_action @ _action_state_jacobian(
            controller, point, resolved_decisions
        )
        slope_by_state = closed @ point_by_state
        total = total + weight * slope
        total_by_state = total_by_state + weight * slope_by_state
        if stage < len(_RK4_OFFSETS):
            offset = _RK4_OFFSETS[stage] * h
            point = states + offset[:, 0] * slope
            point_by_state = eye + offset * slope_by_state
    return first, states + (h[:, 0] / 6.0) * total, eye + (h / 6.0) * total_by_state


def augment_noise_injection(physical_injection: Tensor, action_dim: int) -> Tensor:
    """Return Gamma=(K_h Sigma;0): Brownian noise enters only the physical state block."""
    padding = physical_injection.new_zeros(
        physical_injection.shape[0], action_dim, physical_injection.shape[-1]
    )
    return torch.cat((physical_injection, padding), dim=-2)


def propagate_linearized_covariance(
    covariances: Tensor,
    jacobians: Tensor,
    noise_injections: Tensor,
    step_sizes: Tensor,
) -> Tensor:
    """Return F Lambda F^T+h B B^T (or augmented Gamma), without inverting singular A."""
    h = step_sizes.reshape(-1, 1, 1)
    return (
        jacobians @ covariances @ jacobians.mT
        + h * noise_injections @ noise_injections.mT
    )


def project_physical_covariance(
    augmented_covariances: Tensor, state_spec: StateSpec
) -> Tensor:
    """Return C Lambda_x C^T, C=(I,0), retaining transported state/held-torque correlation."""
    size = int(state_spec["position_dim"]) + int(state_spec["velocity_dim"])
    return augmented_covariances[..., :size, :size]


def _action_state_jacobian(
    controller: Controller, states: Tensor, resolved_decisions: Tensor
) -> Tensor:
    """dg/ds [B,m,D]; the clipped PD law is piecewise linear, so this is locally constant.

    Zero for a controller without state feedback, whose torque does not read the state.
    """
    if not getattr(controller, "state_feedback", True):
        return states.new_zeros(
            states.shape[0], resolved_decisions.shape[-1], states.shape[-1]
        )
    with torch.enable_grad():
        points = states.detach().requires_grad_(True)
        torques = controller.action(points, resolved_decisions.detach())
        rows = [
            torch.autograd.grad(torques[:, i].sum(), points, retain_graph=True)[0]
            for i in range(torques.shape[-1])
        ]
    return torch.stack(rows, dim=1)


def augmented_closed_loop_step(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    held_actions: Tensor,
    resolved_decisions: Tensor,
    step_sizes: Tensor,
    refresh_mask: Tensor,
    *,
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return next state, held torque and the full Jacobian of the step including torque refresh.

    The refresh a=g(s,kappa) acts at the start of a flagged step. The Jacobian maps
    (s_j,a_j) to (s_(j+1),a_(j+1)), both of dimension D+m. The state block is the
    noise-free map of model_substep with the given integrator: RK4's stage chain rule
    (_rk4_step_jacobian) or the semi-implicit d Psi_h = I + h K_h Db.

    Under refresh_mode "continuous" (stage_feedback) the law is evaluated at every stage
    whatever the mask. RK4: the state block is the closed-loop map's Jacobian
    (_rk4_closed_loop_jacobian), the returned torque is g(s_j), the next state does not
    read a held torque and the torque block of the Jacobian is zero, so the held-torque
    covariance stays zero. Semi-implicit: its one stage is the substep's start, so the
    step is the "held" step refreshed at every substep, computed by the same arithmetic.
    """
    n_o, n, m = plant.position_dim, plant.velocity_dim, plant.action_dim
    h = step_sizes.reshape(-1, 1, 1)
    rk4 = check_model_integrator(integrator) == "rk4"
    if stage_feedback(controller) and not rk4:
        refresh_mask = torch.ones_like(refresh_mask)
    elif stage_feedback(controller):
        size = n_o + n
        actions, next_states, psi_state = _rk4_closed_loop_jacobian(
            plant, controller, states, resolved_decisions, step_sizes
        )
        jacobian = torch.cat(
            (
                torch.cat((psi_state, states.new_zeros(states.shape[0], size, m)), dim=-1),
                states.new_zeros(states.shape[0], m, size + m),
            ),
            dim=1,
        )
        return next_states, actions, jacobian

    mask = refresh_mask.reshape(-1, 1)
    any_refresh = bool(refresh_mask.any())
    actions = held_actions
    if any_refresh:
        actions = torch.where(
            mask, controller.action(states, resolved_decisions), held_actions
        )

    if check_model_integrator(integrator) == "rk4":
        next_states, psi_state, psi_action = _rk4_step_jacobian(
            plant, states, actions, step_sizes
        )
    else:
        drift, by_state, by_action = plant.drift_jacobian(states, actions)
        velocities = states[:, n_o:] + h[:, 0] * drift[:, n_o:]
        projection = plant.observation_projection.to(states.dtype)
        positions = states[:, :n_o] + h[:, 0] * (velocities @ projection.T)
        next_states = torch.cat((positions, velocities), dim=-1)

        # Psi_h=s+h b+h^2 J_kin b, so d Psi_h=I+h K_h Db at fixed torque.
        transport = kinematic_noise_transport(plant.state_spec, step_sizes)
        eye = torch.eye(n_o + n, dtype=states.dtype, device=states.device)
        psi_state = eye + h * transport @ by_state
        psi_action = h * transport @ by_action

    keep = (~mask).to(states.dtype).unsqueeze(-1)
    torque_state = states.new_zeros(states.shape[0], m, n_o + n)
    if any_refresh:
        torque_state = _action_state_jacobian(
            controller, states, resolved_decisions
        ) * mask.to(states.dtype).unsqueeze(-1)
    torque_action = keep * torch.eye(m, dtype=states.dtype, device=states.device)

    jacobian = torch.cat(
        (
            torch.cat(
                (psi_state + psi_action @ torque_state, psi_action @ torque_action),
                dim=-1,
            ),
            torch.cat((torque_state, torque_action), dim=-1),
        ),
        dim=1,
    )
    return next_states, actions, jacobian


def _saturated(controller: Controller, actions: Tensor) -> Tensor:
    """Flag rows [n] whose torque sits on a limit of the clip."""
    low = torch.as_tensor(
        controller.config["torque_low"], dtype=actions.dtype, device=actions.device
    )
    high = torch.as_tensor(
        controller.config["torque_high"], dtype=actions.dtype, device=actions.device
    )
    return ((actions <= low) | (actions >= high)).any(dim=-1)


# ------------------------------------------------------ contact-triggered fine substepping


@dataclass(frozen=True, eq=False)
class ContactSubstepping:
    """The contact-triggered fine substepping of one task (config key model_coarse_step).

    coarse_step: the step of the first, every-row integration; fine_step: model_step, the
    step of the flagged rows (the physics step on every row that sets the key).
    positions: the state indices of the limited coordinates (StateSpec
    limited_position_ranges), velocities: the state indices of their rates (position_dim
    plus the coordinate's dof, P's column), lows/highs: their ranges, margins: the
    absolute margins m_i = margin_fraction * (high_i - low_i), velocity_factor: beta.
    contact_flags states the detector. Build it with contact_substepping or, from an arm
    config, model_contact_substepping.
    """

    coarse_step: float
    fine_step: float
    positions: Tuple[int, ...]
    velocities: Tuple[int, ...]
    lows: Tuple[float, ...]
    highs: Tuple[float, ...]
    margins: Tuple[float, ...]
    margin_fraction: float
    velocity_factor: float

    def record(self) -> Mapping[str, float]:
        """The setting as run records store it (common.ph_config.contact_substepping_setting)."""
        return {
            "coarse_step": float(self.coarse_step),
            "fine_step": float(self.fine_step),
            "margin_fraction": float(self.margin_fraction),
            "velocity_factor": float(self.velocity_factor),
        }


def contact_substepping(
    state_spec: StateSpec,
    coarse_step: Optional[float],
    fine_step: float,
    *,
    margin_fraction: float = CONTACT_MARGIN_FRACTION,
    velocity_factor: float = CONTACT_VELOCITY_FACTOR,
) -> Optional[ContactSubstepping]:
    """The ContactSubstepping of state_spec's limited coordinates; None when coarse_step is None.

    Raises ValueError without limited_position_ranges, for a coarse step that is not a
    whole multiple (at least 2) of fine_step, a margin_fraction that is not positive, or a
    velocity_factor below 1.5.
    """
    if coarse_step is None:
        return None
    ranges = tuple(state_spec.get("limited_position_ranges", ()))
    if not ranges:
        raise ValueError("Contact substepping needs StateSpec limited_position_ranges.")
    ratio = float(coarse_step) / float(fine_step)
    if ratio < 2.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-8:
        raise ValueError(
            f"The coarse step {coarse_step} must be a whole multiple (at least 2) of the "
            f"fine step {fine_step}."
        )
    if not margin_fraction > 0.0:
        raise ValueError("The contact margin fraction must be positive.")
    if not velocity_factor >= 1.5:
        raise ValueError("The contact velocity factor beta must be at least 1.5.")
    projection = np.asarray(state_spec["observation_projection"], dtype=np.float64)
    position_dim = int(state_spec["position_dim"])
    positions = tuple(int(entry[0]) for entry in ranges)
    lows = tuple(float(entry[1]) for entry in ranges)
    highs = tuple(float(entry[2]) for entry in ranges)
    if any(low >= high for low, high in zip(lows, highs)):
        raise ValueError("limited_position_ranges need low < high.")
    return ContactSubstepping(
        coarse_step=float(coarse_step),
        fine_step=float(fine_step),
        positions=positions,
        velocities=tuple(
            position_dim + int(np.argmax(np.abs(projection[index]))) for index in positions
        ),
        lows=lows,
        highs=highs,
        margins=tuple(margin_fraction * (high - low) for low, high in zip(lows, highs)),
        margin_fraction=float(margin_fraction),
        velocity_factor=float(velocity_factor),
    )


def model_contact_substepping(
    config: Mapping[str, object], state_spec: StateSpec, controller: Controller
) -> Optional[ContactSubstepping]:
    """The ContactSubstepping an arm config runs, None when off.

    Off without model_coarse_step and under refresh_mode "held" (controller), where the
    model grid splits at every refresh and the scheme is inactive
    (common.ph_config.contact_substepping_setting, which run records store).
    """
    setting = contact_substepping_setting(config)
    if setting is None or getattr(controller, "refresh_mode", "held") != "continuous":
        return None
    return contact_substepping(
        state_spec,
        setting["coarse_step"],
        setting["fine_step"],
        margin_fraction=setting["margin_fraction"],
        velocity_factor=setting["velocity_factor"],
    )


def contact_flags(
    contact: ContactSubstepping, before: Tensor, after: Tensor, step_sizes: Tensor
) -> Tensor:
    """Rows [n] whose substep from before [n,D] to after [n,D] (lengths [n]) may touch a limit.

    Detector. For each limited coordinate i (position y_i, rate v_i, range
    [low_i, high_i]) the margin over a substep of length h is
        mu_i = m_i + beta * h * max(|v_i(before)|, |v_i(after)|),
    with m_i = margin_fraction * (high_i - low_i) and beta = velocity_factor >= 1.5. A row
    is left unflagged only if, for every i, y_i(before) and y_i(after) both lie strictly
    inside (low_i + mu_i, high_i - mu_i). So a knot past the range, a knot within the
    margin of an edge, and a non-finite knot (a comparison with NaN is False) flag the
    row. The integrators call this on every substep of the coarse path, both ends
    included, so an interval's first and last knots are checked too; values are read
    detached (the mask carries no gradient).

    Claim: under (A) below, a row whose coarse path is unflagged on every substep never
    touches a limit, so the coarse integration of it is the limit-free integration the
    coarse step was measured accurate for; equivalently, a contact cannot go undetected.
    (A) On each coarse substep [t_k, t_k + h], the exact closed-loop path y(t) started
    from the coarse knot s_k has rates bounded by the knot rates of the coarse path:
    |v_i(t)| <= beta * max(|v_i^k|, |v_i^(k+1)|) + m_i / h for every t in the substep.
    Proof sketch: |y_i(t) - y_i^k| <= h * max |v_i| <= mu_i on the substep by (A), and
    y_i^k lies inside (low_i + mu_i, high_i - mu_i), so low_i < y_i(t) < high_i throughout:
    MuJoCo's limit row exists only past the range (margin 0), so no limit force acts. The
    same bound holds for the RK4 stages of the coarse map, y^k + c h k_pos with c <= 1
    and stage rates approximating v on the substep, so the coarse map never evaluates the
    limit force on an unflagged row either. Tunnelling, a path that crosses an edge and
    comes back within one substep (or a fast approach that crosses the margin between
    two knots), needs |y_i(t*) - y_i^k| >= dist(y_i^k, edge), which (A) bounds by mu_i:
    the knot is within the margin and the row is flagged. (A) asks only that the rate not
    exceed beta times its larger knot value within one coarse step (beta = 2: the speed
    may double mid-step) plus m_i / h, which covers a start from rest: travel under an
    acceleration a over h is a h^2 / 2, and m_i (1% of the range: 36 mm on the cart's
    rail, 5.8 mm on the point mass's slides) exceeds it by two decades at the coarse steps
    used (5 ms, 10 ms) and the motor's accelerations. Rows with a coordinate at rest near
    an edge are flagged by m_i alone.
    """
    with torch.no_grad():
        start, end = before.detach(), after.detach()
        device, dtype = start.device, start.dtype
        positions = torch.as_tensor(contact.positions, dtype=torch.long, device=device)
        rates = torch.as_tensor(contact.velocities, dtype=torch.long, device=device)
        low = torch.as_tensor(contact.lows, dtype=dtype, device=device)
        high = torch.as_tensor(contact.highs, dtype=dtype, device=device)
        absolute = torch.as_tensor(contact.margins, dtype=dtype, device=device)
        h = step_sizes.detach().reshape(-1, 1).to(dtype)
        speed = torch.maximum(start[:, rates].abs(), end[:, rates].abs())
        margin = absolute + contact.velocity_factor * h * speed
        inner_low, inner_high = low + margin, high - margin
        clear = torch.ones(start.shape[0], dtype=torch.bool, device=device)
        for knot in (start[:, positions], end[:, positions]):
            clear = clear & ((knot > inner_low) & (knot < inner_high)).all(dim=-1)
        return ~clear


# ------------------------------------------------------------------------ contact exclusion


@dataclass(frozen=True, eq=False)
class ContactExclusion:
    """Contact exclusion's rule for one task (config key contact_handling).

    active: True under contact_handling "exclude" (the agents then exclude and fall back);
    False under "port", where the agents still store the flags (diagnostics only).
    model_step: the step of every model integration under "exclude"
    (common.ph_config.exclusion_model_step). positions, velocities, lows, highs: the
    limited coordinates' state indices, their rates' state indices and their ranges;
    margins: m_i = margin_fraction (high_i - low_i); accelerations: a_i, the acceleration
    allowance (acceleration_factor times the motors' acceleration bound of the
    coordinate, MujocoPHEnv.motor_acceleration_bounds); decay_rates: k_i = d_i (M^-1)_ii,
    the damping rate of the coordinate (MujocoPHEnv.velocity_decay_rates; 0 without
    damping); start_velocity_factor: beta_s of the start-state rule (start_state_flags);
    velocity_factor: beta of the per-substep path detector (contact_flags reads it, as for
    ContactSubstepping).
    """

    active: bool
    model_step: float
    positions: Tuple[int, ...]
    velocities: Tuple[int, ...]
    lows: Tuple[float, ...]
    highs: Tuple[float, ...]
    margins: Tuple[float, ...]
    accelerations: Tuple[float, ...]
    margin_fraction: float
    start_velocity_factor: float
    acceleration_factor: float
    velocity_factor: float
    decay_rates: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.decay_rates:
            object.__setattr__(self, "decay_rates", tuple(0.0 for _ in self.positions))
        if len(self.decay_rates) != len(self.positions) or any(
            not (math.isfinite(k) and k >= 0.0) for k in self.decay_rates
        ):
            raise ValueError("decay_rates need one finite nonnegative rate per limited coordinate.")

    def record(self) -> Mapping[str, object]:
        """The rule with its per-coordinate constants (diagnostics, test-job output)."""
        return {
            "mode": "exclude" if self.active else "port",
            "model_step": float(self.model_step),
            "margin_fraction": float(self.margin_fraction),
            "velocity_factor": float(self.start_velocity_factor),
            "acceleration_factor": float(self.acceleration_factor),
            "path_velocity_factor": float(self.velocity_factor),
            "rule_version": int(EXCLUSION_RULE_VERSION),
            "positions": list(self.positions),
            "margins": [float(m) for m in self.margins],
            "accelerations": [float(a) for a in self.accelerations],
            "decay_rates": [float(k) for k in self.decay_rates],
        }


def contact_exclusion(
    state_spec: StateSpec,
    motor_accelerations: object,
    *,
    active: bool,
    model_step: float,
    decay_rates: object = None,
    margin_fraction: float = EXCLUSION_MARGIN_FRACTION,
    start_velocity_factor: float = EXCLUSION_VELOCITY_FACTOR,
    acceleration_factor: float = EXCLUSION_ACCELERATION_FACTOR,
    path_velocity_factor: float = EXCLUSION_PATH_VELOCITY_FACTOR,
) -> Optional[ContactExclusion]:
    """The ContactExclusion of state_spec's limited coordinates; None without any.

    motor_accelerations [velocity_dim]: per velocity coordinate the motors' acceleration
    bound (MujocoPHEnv.motor_acceleration_bounds); coordinate i's allowance is
    acceleration_factor times the entry of its rate (P's column). decay_rates
    [velocity_dim]: per velocity coordinate the damping rate d_i (M^-1)_ii
    (MujocoPHEnv.velocity_decay_rates); None: no damping (the undamped bound).
    """
    ranges = tuple(state_spec.get("limited_position_ranges", ()))
    if not ranges:
        return None
    projection = np.asarray(state_spec["observation_projection"], dtype=np.float64)
    position_dim = int(state_spec["position_dim"])
    bounds = np.asarray(motor_accelerations, dtype=np.float64).reshape(-1)
    rates = (
        np.zeros_like(bounds)
        if decay_rates is None
        else np.asarray(decay_rates, dtype=np.float64).reshape(-1)
    )
    positions = tuple(int(entry[0]) for entry in ranges)
    dofs = tuple(int(np.argmax(np.abs(projection[index]))) for index in positions)
    lows = tuple(float(entry[1]) for entry in ranges)
    highs = tuple(float(entry[2]) for entry in ranges)
    if any(low >= high for low, high in zip(lows, highs)):
        raise ValueError("limited_position_ranges need low < high.")
    return ContactExclusion(
        active=bool(active),
        model_step=float(model_step),
        positions=positions,
        velocities=tuple(position_dim + dof for dof in dofs),
        lows=lows,
        highs=highs,
        margins=tuple(margin_fraction * (high - low) for low, high in zip(lows, highs)),
        accelerations=tuple(float(acceleration_factor * bounds[dof]) for dof in dofs),
        margin_fraction=float(margin_fraction),
        start_velocity_factor=float(start_velocity_factor),
        acceleration_factor=float(acceleration_factor),
        velocity_factor=float(path_velocity_factor),
        decay_rates=tuple(float(rates[dof]) for dof in dofs),
    )


def model_contact_exclusion(config: Mapping[str, object], env: object) -> Optional[ContactExclusion]:
    """The ContactExclusion of an arm config on env (active under contact_handling
    "exclude"); None on a task without limited coordinates."""
    spec = env.state_spec
    if not spec.get("limited_position_ranges"):
        return None
    zeros = np.zeros(int(spec["velocity_dim"]))
    bounds = getattr(env, "motor_acceleration_bounds", None)
    decay = getattr(env, "velocity_decay_rates", None)
    return contact_exclusion(
        spec,
        bounds() if callable(bounds) else zeros,
        decay_rates=decay() if callable(decay) else zeros,
        active=contact_handling_setting(config) == "exclude",
        model_step=exclusion_model_step(config),
    )


# Below this z = k Delta the reach functions use their Taylor series (truncation below
# 1e-14 relative; z + expm1(-z) would lose about 2 eps / z there).
_REACH_SERIES_BELOW = 1e-3


def reach_travel(speeds: Tensor, durations: Tensor, accelerations: Tensor, decay_rates: Tensor) -> Tensor:
    """The largest travel of the damped double integrator over an interval, torch.

        x = max(v Delta, v Delta phi1(z) + a Delta^2 phi2(z)),  z = k Delta,
        phi1(z) = (1 - e^-z) / z,  phi2(z) = (z - 1 + e^-z) / z^2  (1 and 1/2 at z = 0),

    for start speeds v = |speeds| (broadcast with durations Delta, accelerations a >= 0
    and decay rates k >= 0). It bounds sup_{0<=t<=Delta} |y(t) - y(0)| for
    ydd = s(t) a - k yd with any measurable |s(t)| <= 1 and |yd(0)| = v: with speed below
    the terminal speed a / k the full push (s = 1 toward the edge) gives the second term
    exactly; above it the speed only decreases, so v Delta bounds the travel. At k = 0 it
    is v Delta + a Delta^2 / 2. Derivation: docs/ct_ph_sac_layout.md (contact exclusion).
    """
    v = speeds.abs()
    z = decay_rates * durations
    small = z < _REACH_SERIES_BELOW
    safe = torch.where(small, torch.ones_like(z), z)
    decay = torch.expm1(-safe)
    phi1 = torch.where(small, 1.0 - z / 2.0 + z * z / 6.0 - z * z * z / 24.0, -decay / safe)
    phi2 = torch.where(
        small, 0.5 - z / 6.0 + z * z / 24.0 - z * z * z / 120.0, (safe + decay) / (safe * safe)
    )
    held = v * durations
    rise = held * phi1 + accelerations * durations.square() * phi2
    return torch.maximum(held, rise)


def reach_travel_numpy(speeds: object, durations: object, accelerations: object, decay_rates: object) -> np.ndarray:
    """reach_travel in numpy (double), the arithmetic the agents store."""
    v = np.abs(np.asarray(speeds, dtype=np.float64))
    delta = np.asarray(durations, dtype=np.float64)
    a = np.asarray(accelerations, dtype=np.float64)
    z = np.asarray(decay_rates, dtype=np.float64) * delta
    small = z < _REACH_SERIES_BELOW
    safe = np.where(small, 1.0, z)
    decay = np.expm1(-safe)
    phi1 = np.where(small, 1.0 - z / 2.0 + z * z / 6.0 - z * z * z / 24.0, -decay / safe)
    phi2 = np.where(small, 0.5 - z / 6.0 + z * z / 24.0 - z * z * z / 120.0, (safe + decay) / (safe * safe))
    held = v * delta
    rise = held * phi1 + a * (delta * delta) * phi2
    return np.maximum(held, rise)


def start_state_margins(exclusion: ContactExclusion, speeds: Tensor, durations: Tensor) -> Tensor:
    """mu_i = m_i + reach_travel(beta_s |qdot_i|, Delta, a_i, k_i), [B, n_limited]."""
    dtype, device = speeds.dtype, speeds.device
    absolute = torch.as_tensor(exclusion.margins, dtype=dtype, device=device)
    accelerations = torch.as_tensor(exclusion.accelerations, dtype=dtype, device=device)
    rates = torch.as_tensor(exclusion.decay_rates, dtype=dtype, device=device)
    delta = durations.reshape(-1, 1).to(dtype)
    travel = reach_travel(exclusion.start_velocity_factor * speeds.abs(), delta, accelerations, rates)
    return absolute + travel


def start_state_margins_numpy(exclusion: ContactExclusion, speeds: object, duration: object) -> np.ndarray:
    """start_state_margins in numpy: speeds [..., n_limited], duration broadcastable."""
    travel = reach_travel_numpy(
        exclusion.start_velocity_factor * np.abs(np.asarray(speeds, dtype=np.float64)),
        np.asarray(duration, dtype=np.float64),
        np.asarray(exclusion.accelerations, dtype=np.float64),
        np.asarray(exclusion.decay_rates, dtype=np.float64),
    )
    return np.asarray(exclusion.margins, dtype=np.float64) + travel


def start_state_flags(exclusion: ContactExclusion, states: Tensor, durations: Tensor) -> Tensor:
    """Contact exclusion's start-state rule: rows [B] (bool) near or past a limit.

    For each limited coordinate i (position y_i, rate v_i, range [low_i, high_i]) and the
    row's interval Delta, the reach margin is (rule version 2,
    common.ph_config.EXCLUSION_RULE_VERSION)
        mu_i = m_i + reach_travel(beta_s |v_i|, Delta, a_i, k_i)
             = m_i + max(w Delta, w Delta phi1(k_i Delta) + a_i Delta^2 phi2(k_i Delta)),
    w = beta_s |v_i|, with m_i = margin_fraction (high_i - low_i) (1% of the range), beta_s
    = start_velocity_factor (1.5), a_i the acceleration allowance (2 times the motors'
    acceleration of the coordinate at the reference pose) and k_i = d_i (M^-1)_ii its
    damping rate there. A row is flagged unless every y_i lies strictly inside
    (low_i + mu_i, high_i - mu_i); a non-finite start is flagged. The travel term bounds
    the interval's travel of the damped double integrator ydd = s a_i - k_i yd, |s| <= 1
    (reach_travel); without damping it is beta_s |v_i| Delta + a_i Delta^2 / 2, rule
    version 1. Left over for what the model omits: the factor 2 in a_i (also doubling the
    terminal speed a_i / k_i, which covers the other links' coupling forces: their
    damping, Coriolis and centripetal terms and motors through M^-1), beta_s - 1 of the
    speed, and m_i (gravity, limit forces, the pose dependence of M). It reads the start
    state and the interval only, never the outcome. The agents combine it with the
    environment's limit_touched.
    """
    with torch.no_grad():
        states = states.detach()
        positions = torch.as_tensor(exclusion.positions, dtype=torch.long, device=states.device)
        rates = torch.as_tensor(exclusion.velocities, dtype=torch.long, device=states.device)
        low = torch.as_tensor(exclusion.lows, dtype=states.dtype, device=states.device)
        high = torch.as_tensor(exclusion.highs, dtype=states.dtype, device=states.device)
        margin = start_state_margins(exclusion, states[:, rates], durations.detach())
        knot = states[:, positions]
        clear = ((knot > low + margin) & (knot < high - margin)).all(dim=-1)
        return ~clear


def start_state_flag_numpy(exclusion: ContactExclusion, state: np.ndarray, duration: float) -> bool:
    """start_state_flags for one state [D] (numpy, double), the rule the agents store."""
    state = np.asarray(state, dtype=np.float64)
    positions = state[list(exclusion.positions)]
    margin = start_state_margins_numpy(
        exclusion, state[list(exclusion.velocities)], float(duration)
    )
    clear = np.all(
        (positions > np.asarray(exclusion.lows) + margin)
        & (positions < np.asarray(exclusion.highs) - margin)
    )
    return not bool(clear)


def batch_contact_flags(batch: Mapping[str, Tensor]) -> Tensor:
    """The batch's contact-exclusion flags, bool [B] (all False without the column)."""
    flags = batch.get("contact_flags")
    if flags is None:
        return torch.zeros(
            batch["states"].shape[0], dtype=torch.bool, device=batch["states"].device
        )
    return flags.reshape(-1) > 0.5


def subset_batch(batch: Mapping[str, Tensor], rows: Tensor) -> dict:
    """The batch's rows (a bool mask or an index tensor [B]), every column, in row order."""
    if rows.dtype == torch.bool:
        rows = torch.nonzero(rows).squeeze(-1)
    return {key: value.index_select(0, rows) for key, value in batch.items()}


# Rows integrated by the contact scheme since the last reset_contact_statistics(), per
# consumer: [rows, rows integrated at the fine step]. Diagnostics only (timing scripts,
# accuracy measurements); never part of an agent's state.
_CONTACT_STATISTICS: Dict[str, list] = {}


def contact_statistics() -> Dict[str, Tuple[int, int]]:
    """{consumer: (rows, fine rows)} since the last reset_contact_statistics()."""
    return {key: (int(value[0]), int(value[1])) for key, value in _CONTACT_STATISTICS.items()}


def reset_contact_statistics() -> None:
    _CONTACT_STATISTICS.clear()


def _count_contact(consumer: str, touched: Tensor) -> None:
    entry = _CONTACT_STATISTICS.setdefault(consumer, [0, 0])
    entry[0] += int(touched.numel())
    entry[1] += int(touched.sum())


def _check_contact_grid(
    contact: ContactSubstepping,
    controller: Controller,
    maximum_step: float,
    controller_refresh_steps: Optional[int],
) -> None:
    if abs(float(maximum_step) - contact.fine_step) > 1e-12 * max(1.0, contact.fine_step):
        raise ValueError(
            f"Contact substepping's fine step {contact.fine_step} must be the maximum_step "
            f"{maximum_step} of the integration."
        )
    if grid_refresh_steps(controller, controller_refresh_steps) is not None:
        raise ValueError(
            "Contact substepping needs refresh_mode continuous (no refresh points on the grid)."
        )


def _scatter_rows(kept_rows: Tensor, flagged_rows: Tensor, touched: Tensor) -> Tensor:
    """[B,...] from kept_rows (one row per False of touched [B], in row order) and
    flagged_rows (one per True, in row order): a stable, differentiable scatter. Zero-padded
    grids [n,N] of different widths are padded to the wider."""
    kept = torch.nonzero(~touched).squeeze(-1)
    flagged = torch.nonzero(touched).squeeze(-1)
    inverse = torch.argsort(torch.cat((kept, flagged)))
    if kept_rows.dim() == 2 and flagged_rows.dim() == 2 and kept_rows.shape[1] != flagged_rows.shape[1]:
        width = max(kept_rows.shape[1], flagged_rows.shape[1])
        kept_rows = torch.nn.functional.pad(kept_rows, (0, width - kept_rows.shape[1]))
        flagged_rows = torch.nn.functional.pad(flagged_rows, (0, width - flagged_rows.shape[1]))
    return torch.cat((kept_rows, flagged_rows), dim=0).index_select(0, inverse)


def _coarse_increments(
    increments: Tensor,
    durations: Tensor,
    *,
    fine_step: float,
    coarse_step: float,
    physics_dt: float,
) -> Tensor:
    """Sum fine-grid increments [B,L,N_f,K] over each coarse substep: [B,L,N_c,K].

    Each coarse substep (the continuous grid at coarse_step) takes the fine substeps (the
    grid at fine_step) inside it, the same Brownian path. Knots are compared in physics
    steps to WHOLE_STEP_TOLERANCE: the grids carry the durations' dtype, and in float32 a
    knot of a 20 ms interval is off by about 1e-6 steps. Raises ValueError where a coarse
    knot is not a fine knot.
    """
    counts_f, sizes_f, _ = refresh_aligned_grid(durations, fine_step, physics_dt=physics_dt)
    counts_c, sizes_c, _ = refresh_aligned_grid(durations, coarse_step, physics_dt=physics_dt)
    fine_steps = sizes_f.double() / physics_dt
    coarse_steps = sizes_c.double() / physics_dt
    fine_ends = torch.cumsum(fine_steps, dim=1)
    coarse_ends = torch.cumsum(coarse_steps, dim=1)
    width_f, width_c = sizes_f.shape[1], sizes_c.shape[1]
    active_f = torch.arange(width_f, device=sizes_f.device).unsqueeze(0) < counts_f.unsqueeze(1)
    active_c = torch.arange(width_c, device=sizes_c.device).unsqueeze(0) < counts_c.unsqueeze(1)
    matched = (
        (coarse_ends.unsqueeze(2) - fine_ends.unsqueeze(1)).abs() < WHOLE_STEP_TOLERANCE
    ) & active_f.unsqueeze(1)
    if not bool((matched.any(dim=2) | ~active_c).all()):
        raise ValueError(
            f"Contact substepping needs the coarse grid ({coarse_step}) to nest in the fine "
            f"grid ({fine_step}): the coarse step must divide every interval."
        )
    middles = fine_ends - 0.5 * fine_steps
    owner = (coarse_ends.unsqueeze(1) <= middles.unsqueeze(2)).sum(dim=2)  # [B,N_f]
    assign = (
        (owner.unsqueeze(1) == torch.arange(width_c, device=owner.device).view(1, -1, 1))
        & active_f.unsqueeze(1)
        & active_c.unsqueeze(2)
    )
    return torch.einsum("bcf,blfk->blck", assign.to(increments.dtype), increments)


def integrate_endpoint_moments(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    covariance_mode: Literal["leading_order", "propagated"],
    create_graph: bool,
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
    contact: Optional[ContactSubstepping] = None,
) -> EndpointMoments:
    """Differentiable endpoint moments; Delta*A(s) or the transported B/Gamma recursion.

    The mean is the noise-free endpoint of model_substep with the given integrator over
    the controller's grid (grid_refresh_steps: refresh-aligned under refresh_mode
    "held", where the torque is refreshed at flagged substeps and held through each
    substep and all RK4 stages; split at maximum_step alone under "continuous", where
    the law is evaluated at every stage and every substep counts toward the saturation
    fraction). "leading_order" covariance is Delta*A(s) for either integrator;
    "propagated" runs the recursion with that integrator's step Jacobian.
    Rows are sorted by step count, so each step evaluates only the rows still active.
    A drift-only plant (learns_diffusion False) has zero covariance in either mode, so
    its moments skip the propagation, whose means are the same, and say drift_only.
    A held-torque controller is evaluated at the first substep only
    (controller_refresh_flags), so its saturation fraction is 1 or 0.

    With contact (a ContactSubstepping whose fine_step is maximum_step) every row is
    integrated at contact.coarse_step first, the rows contact_flags flags on any substep
    are integrated again at maximum_step, and their means, covariances (propagated or
    leading order), saturation fractions and grids replace the coarse ones; the result
    carries contact_rows. With create_graph the plant-fit gradient flows through the
    integration each row used; should a coarse row be non-finite (it is then flagged), the
    kept rows are integrated again on their own so that no non-finite discarded row
    reaches the backward pass.
    """
    if contact is None:
        moments, _ = _endpoint_moments(
            plant, controller, states, decisions, durations,
            maximum_step=maximum_step, physics_dt=physics_dt,
            controller_refresh_steps=controller_refresh_steps,
            covariance_mode=covariance_mode, create_graph=create_graph,
            integrator=integrator, detector=None,
        )
        return moments
    _check_contact_grid(contact, controller, maximum_step, controller_refresh_steps)
    grid = {
        "physics_dt": physics_dt,
        "controller_refresh_steps": controller_refresh_steps,
        "covariance_mode": covariance_mode,
        "create_graph": create_graph,
        "integrator": integrator,
        "detector": None,
    }
    coarse, touched = _endpoint_moments(
        plant, controller, states, decisions, durations,
        maximum_step=contact.coarse_step, **{**grid, "detector": contact},
    )
    _count_contact("endpoint_moments", touched)
    if not bool(touched.any()):
        return {**coarse, "contact_rows": touched}
    if bool(touched.all()):
        fine, _ = _endpoint_moments(
            plant, controller, states, decisions, durations, maximum_step=maximum_step, **grid
        )
        return {**fine, "contact_rows": touched}
    flagged = torch.nonzero(touched).squeeze(-1)
    kept = torch.nonzero(~touched).squeeze(-1)
    fine, _ = _endpoint_moments(
        plant, controller, states[flagged], decisions[flagged], durations[flagged],
        maximum_step=maximum_step, **grid,
    )
    keys = ("means", "covariances", "step_counts", "step_sizes", "saturation_fractions")
    if create_graph and not (
        bool(torch.isfinite(coarse["means"].detach()).all())
        and bool(torch.isfinite(coarse["covariances"].detach()).all())
    ):
        # A non-finite discarded row would turn its zero gradient into NaN in the backward
        # pass of the shared batch: integrate the kept rows again, without it.
        clean, _ = _endpoint_moments(
            plant, controller, states[kept], decisions[kept], durations[kept],
            maximum_step=contact.coarse_step, **grid,
        )
        kept_values = {key: clean[key] for key in keys}
    else:
        kept_values = {key: coarse[key].index_select(0, kept) for key in keys}
    merged = {key: _scatter_rows(kept_values[key], fine[key], touched) for key in keys}
    return {**merged, "drift_only": fine["drift_only"], "contact_rows": touched}


def _endpoint_moments(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    covariance_mode: Literal["leading_order", "propagated"],
    create_graph: bool,
    integrator: str,
    detector: Optional[ContactSubstepping],
) -> Tuple[EndpointMoments, Optional[Tensor]]:
    """integrate_endpoint_moments without contact substepping; with a detector also the
    rows [B] (bool) that contact_flags flags on any substep, else None."""
    if covariance_mode not in ("leading_order", "propagated"):
        raise ValueError(f"Unknown covariance mode {covariance_mode!r}.")
    check_model_integrator(integrator)
    drift_only = not getattr(plant, "learns_diffusion", True)
    propagated = covariance_mode == "propagated" and not drift_only
    continuous = stage_feedback(controller)
    counts, sizes, flags = refresh_aligned_grid(
        durations,
        maximum_step,
        physics_dt=physics_dt,
        controller_refresh_steps=grid_refresh_steps(controller, controller_refresh_steps),
    )
    flags = controller_refresh_flags(controller, flags)
    total = states.shape[0]
    state_dim, action_dim = plant.state_dim, plant.action_dim
    order = torch.argsort(counts, descending=True)
    sorted_counts = counts[order]

    with torch.set_grad_enabled(create_graph):
        current = states[order]
        resolved = controller.resolve_decision(current, decisions[order])
        held = current.new_zeros(total, action_dim)
        covariance = None
        if propagated:
            size = state_dim + action_dim
            covariance = current.new_zeros(total, size, size)
        saturated = current.new_zeros(total)
        refreshes = current.new_zeros(total)
        finished_states, finished_covariances = [], []
        # Rows (in the sorted order) the detector flags; only with a detector.
        touched = None
        if detector is not None:
            touched = torch.zeros(total, dtype=torch.bool, device=states.device)

        active = total
        for step in range(int(sorted_counts[0])):
            still = int((sorted_counts > step).sum())
            if still < active:
                finished_states.append(current[still:])
                current, held, resolved = (
                    current[:still],
                    held[:still],
                    resolved[:still],
                )
                if propagated:
                    finished_covariances.append(covariance[still:])
                    covariance = covariance[:still]
                active = still
            h = sizes[order[:active], step]
            mask = flags[order[:active], step]
            before = current
            if propagated:
                injection = physical_noise_injection(plant, current, h)
                current, held, jacobian = augmented_closed_loop_step(
                    plant, controller, current, held, resolved, h, mask,
                    integrator=integrator,
                )
                covariance = propagate_linearized_covariance(
                    covariance,
                    jacobian,
                    augment_noise_injection(injection, action_dim),
                    h,
                )
            elif continuous:
                held = controller.action(current, resolved)
                current = model_substep(
                    plant,
                    current,
                    held,
                    h,
                    integrator=integrator,
                    feedback=_stage_law(controller, resolved),
                )
            else:
                if mask.any():
                    held = torch.where(
                        mask.unsqueeze(-1), controller.action(current, resolved), held
                    )
                current = model_substep(plant, current, held, h, integrator=integrator)
            if touched is not None:
                touched[:active] |= contact_flags(detector, before, current, h)
            flagged = current.new_ones(current.shape[0]) if continuous else mask.to(current.dtype)
            hit = _saturated(controller, held.detach()).to(current.dtype) * flagged
            saturated = torch.cat((saturated[:active] + hit, saturated[active:]))
            refreshes = torch.cat((refreshes[:active] + flagged, refreshes[active:]))

        finished_states.append(current)
        ranked = torch.cat(finished_states[::-1], dim=0)
        inverse = torch.argsort(order)
        means = ranked[inverse]
        if propagated:
            finished_covariances.append(covariance)
            covariances = project_physical_covariance(
                torch.cat(finished_covariances[::-1], dim=0)[inverse], plant.state_spec
            )
        else:
            covariances = durations.reshape(-1, 1, 1).to(states.dtype) * plant.covariance(
                states
            )
    fractions = (saturated / refreshes.clamp_min(1.0))[inverse].unsqueeze(-1)
    moments: EndpointMoments = {
        "means": means,
        "covariances": covariances,
        "step_counts": counts,
        "step_sizes": sizes,
        "saturation_fractions": fractions.detach(),
        "drift_only": drift_only,
    }
    return moments, (None if touched is None else touched[inverse])


def path_jacobian_norms(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
    contact: Optional[ContactSubstepping] = None,
) -> Tensor:
    """Delta times the largest closed-loop ||d_s b|| along the noise-free path, [B,1].

    The path is the noise-free model_substep path of the given integrator on the
    controller's grid (grid_refresh_steps); the norm is read at each substep's starting
    state. A held torque adds no closed-loop term; a law under refresh_mode "continuous"
    adds db/da dg/ds at every substep. With contact (a ContactSubstepping whose fine_step
    is maximum_step) the rows whose coarse path contact_flags flags take the norms of
    their path at maximum_step, the others those of the coarse path.
    """
    if contact is None:
        norms, _ = _path_jacobian_norms(
            plant, controller, states, decisions, durations,
            maximum_step=maximum_step, physics_dt=physics_dt,
            controller_refresh_steps=controller_refresh_steps, integrator=integrator,
            detector=None,
        )
        return norms
    _check_contact_grid(contact, controller, maximum_step, controller_refresh_steps)
    grid = {
        "physics_dt": physics_dt,
        "controller_refresh_steps": controller_refresh_steps,
        "integrator": integrator,
    }
    coarse, touched = _path_jacobian_norms(
        plant, controller, states, decisions, durations,
        maximum_step=contact.coarse_step, detector=contact, **grid,
    )
    _count_contact("jacobian_norms", touched)
    if not bool(touched.any()):
        return coarse
    if bool(touched.all()):
        fine, _ = _path_jacobian_norms(
            plant, controller, states, decisions, durations,
            maximum_step=maximum_step, detector=None, **grid,
        )
        return fine
    flagged = torch.nonzero(touched).squeeze(-1)
    kept = torch.nonzero(~touched).squeeze(-1)
    fine, _ = _path_jacobian_norms(
        plant, controller, states[flagged], decisions[flagged], durations[flagged],
        maximum_step=maximum_step, detector=None, **grid,
    )
    return _scatter_rows(coarse.index_select(0, kept), fine, touched)


def _path_jacobian_norms(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    integrator: str,
    detector: Optional[ContactSubstepping],
) -> Tuple[Tensor, Optional[Tensor]]:
    """path_jacobian_norms without contact substepping; with a detector also the rows [B]
    (bool) that contact_flags flags on any substep, else None."""
    check_model_integrator(integrator)
    continuous = stage_feedback(controller)
    counts, sizes, flags = refresh_aligned_grid(
        durations,
        maximum_step,
        physics_dt=physics_dt,
        controller_refresh_steps=grid_refresh_steps(controller, controller_refresh_steps),
    )
    flags = controller_refresh_flags(controller, flags)
    if continuous:
        flags = torch.ones_like(flags)
    touched = None
    if detector is not None:
        touched = torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
    with torch.no_grad():
        current = states
        resolved = controller.resolve_decision(states, decisions)
        held = states.new_zeros(states.shape[0], plant.action_dim)
        largest = states.new_zeros(states.shape[0])
        for step in range(int(counts.max())):
            mask = flags[:, step].unsqueeze(-1)
            held = torch.where(mask, controller.action(current, resolved), held)
            _, by_state, by_action = plant.drift_jacobian(current, held)
            closed = by_state + by_action @ _action_state_jacobian(
                controller, current, resolved
            ) * mask.to(states.dtype).unsqueeze(-1)
            norm = torch.linalg.matrix_norm(closed, ord=2)
            largest = torch.where(counts > step, torch.max(largest, norm), largest)
            before = current
            current = model_substep(
                plant,
                current,
                held,
                sizes[:, step],
                integrator=integrator,
                feedback=_stage_law(controller, resolved) if continuous else None,
            )
            if touched is not None:
                touched |= contact_flags(detector, before, current, sizes[:, step]) & (
                    counts > step
                )
    norms = durations.reshape(-1, 1).to(states.dtype) * largest.unsqueeze(-1)
    return norms, touched


def wrapped_endpoint_residual(
    observed_endpoints: Tensor, predicted_endpoints: Tensor, state_spec: StateSpec
) -> Tensor:
    """Wrap only periodic position residuals to [-pi, pi)."""
    residual = observed_endpoints - predicted_endpoints
    periodic = list(state_spec["periodic_position_indices"])
    if not periodic:
        return residual
    mask = torch.zeros(
        residual.shape[-1], dtype=torch.bool, device=residual.device
    )
    mask[periodic] = True
    return torch.where(
        mask, torch.remainder(residual + math.pi, 2.0 * math.pi) - math.pi, residual
    )


def sample_brownian_increments(
    durations: Tensor,
    *,
    path_count: int,
    noise_dim: int,
    maximum_step: float,
    physics_dt: Optional[float] = None,
    controller_refresh_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """Return dW~N(0,h_j I), already scaled per substep, on the shared refresh-aligned grid."""
    _, sizes = integration_grid(
        durations,
        maximum_step,
        physics_dt=physics_dt,
        controller_refresh_steps=controller_refresh_steps,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (sizes.shape[0], path_count, sizes.shape[1], noise_dim)
    standard = torch.randn(shape, generator=generator, dtype=sizes.dtype)
    return standard.to(sizes.device) * sizes.sqrt()[:, None, :, None]


@torch.no_grad()
def stochastic_rollout(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    failure_predicate: Callable[[Tensor], Tensor],
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    path_count: int = 4,
    brownian_increments: Optional[Tensor] = None,
    return_terms: Optional[ReturnTerms] = None,
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
    contact: Optional[ContactSubstepping] = None,
    truncation: Optional[ContactExclusion] = None,
    skip_rows: Optional[Tensor] = None,
) -> RolloutBatch:
    """Paths of model_substep with the given integrator, already-scaled dW used once per substep.

    Contact exclusion (truncation, a ContactExclusion; skip_rows [B] bool; both None: off,
    bitwise the paths without them). A path whose substep contact_flags flags (it comes
    within margin_fraction of the range plus velocity_factor h times its larger knot speed
    of a limited coordinate's edge, or past it) is truncated at the substep's start, its
    last clean knot: it keeps that knot as endpoint, its elapsed time there as
    durations_used and its reward up to it, and integrates no further; it has not failed
    (survived stays True). The rows of skip_rows are not integrated at all: their paths
    are truncated at the start state (duration 0). The batch then carries truncated
    [B,L,1] (bool). Not with contact substepping (contact).

    Each path stops at its first failing state (checked at the end of every substep) and
    keeps that state as its endpoint. The grid is the controller's (grid_refresh_steps;
    supplied increments must be drawn on it). Under refresh_mode "held" the torque is
    held between refreshes and through each substep, and refreshes follow the row's
    grid; under "continuous" the law is evaluated at every stage of every substep. With
    return_terms, each path integrates its reward as the environment does, by
    discounted trapezoids on the substep knots, each knot's rate at its state and the
    torque applied there: the law at that knot's state for a continuous law under
    "rk4", else the substep's torque at both of its knots (the held torque; the
    semi-implicit map's one stage is the substep's start, as the Euler environment's
    is). It pays one discounted charge where it fails. A held-torque controller is
    evaluated at the first substep only (controller_refresh_flags).

    With contact (a ContactSubstepping whose fine_step is maximum_step) the increments
    stay on the grid at maximum_step (supplied or drawn there, as without contact); every
    path is integrated first at contact.coarse_step, each coarse substep taking the sum of
    the fine increments it spans, and each path contact_flags flags (paths are flagged one
    by one) is integrated again at maximum_step with its own fine increments: its
    endpoint, survival, failure time, duration, reward (trapezoids on the fine knots) and
    payoff replace the coarse ones. The batch carries contact_paths.
    """
    if truncation is not None or skip_rows is not None:
        if contact is not None:
            raise ValueError("Contact exclusion and contact substepping do not combine.")
        batch, _ = _rollout(
            plant, controller, states, decisions, durations, failure_predicate,
            maximum_step=maximum_step, physics_dt=physics_dt,
            controller_refresh_steps=controller_refresh_steps, path_count=path_count,
            brownian_increments=brownian_increments, return_terms=return_terms,
            integrator=integrator, detector=None, truncation=truncation, skip_rows=skip_rows,
        )
        return batch
    if contact is None:
        batch, _ = _rollout(
            plant, controller, states, decisions, durations, failure_predicate,
            maximum_step=maximum_step, physics_dt=physics_dt,
            controller_refresh_steps=controller_refresh_steps, path_count=path_count,
            brownian_increments=brownian_increments, return_terms=return_terms,
            integrator=integrator, detector=None,
        )
        return batch
    _check_contact_grid(contact, controller, maximum_step, controller_refresh_steps)
    counts, sizes, _ = refresh_aligned_grid(durations, maximum_step, physics_dt=physics_dt)
    rows = states.shape[0]
    if brownian_increments is None:
        brownian_increments = sample_brownian_increments(
            durations,
            path_count=path_count,
            noise_dim=plant.noise_dim,
            maximum_step=maximum_step,
            physics_dt=physics_dt,
            controller_refresh_steps=None,
        ).to(states.dtype)
    path_count = brownian_increments.shape[1]
    if brownian_increments.shape[:3] != (rows, path_count, sizes.shape[1]):
        raise ValueError("Brownian increments do not match the integration grid.")
    shared = {
        "physics_dt": physics_dt,
        "controller_refresh_steps": controller_refresh_steps,
        "return_terms": return_terms,
        "integrator": integrator,
    }
    coarse, touched = _rollout(
        plant, controller, states, decisions, durations, failure_predicate,
        maximum_step=contact.coarse_step,
        path_count=path_count,
        brownian_increments=_coarse_increments(
            brownian_increments,
            durations,
            fine_step=maximum_step,
            coarse_step=contact.coarse_step,
            physics_dt=physics_dt,
        ),
        detector=contact,
        **shared,
    )
    _count_contact("rollouts", touched)
    extra = {
        "brownian_increments": brownian_increments,
        "step_counts": counts,
        "step_sizes": sizes,
        "contact_paths": touched.reshape(rows, path_count, 1),
    }
    if not bool(touched.any()):
        return {**coarse, **extra}
    if bool(touched.all()):
        fine, _ = _rollout(
            plant, controller, states, decisions, durations, failure_predicate,
            maximum_step=maximum_step, path_count=path_count,
            brownian_increments=brownian_increments, detector=None, **shared,
        )
        return {**fine, **extra}
    flagged = torch.nonzero(touched).squeeze(-1)
    kept = torch.nonzero(~touched).squeeze(-1)
    owners = torch.div(flagged, path_count, rounding_mode="floor")
    owned = durations[owners]
    _, owned_sizes, _ = refresh_aligned_grid(owned, maximum_step, physics_dt=physics_dt)
    noise = brownian_increments.flatten(0, 1).index_select(0, flagged).unsqueeze(1)
    fine, _ = _rollout(
        plant, controller, states[owners], decisions[owners], owned, failure_predicate,
        maximum_step=maximum_step, path_count=1,
        brownian_increments=noise[:, :, : owned_sizes.shape[1]], detector=None, **shared,
    )
    merged = {}
    for key in (
        "endpoints", "survived", "failure_times", "durations_used", "running_rewards",
        "failure_payoffs",
    ):
        merged[key] = _scatter_rows(
            coarse[key].flatten(0, 1).index_select(0, kept), fine[key].flatten(0, 1), touched
        ).unflatten(0, (rows, path_count))
    return {**merged, **extra}


@torch.no_grad()
def _rollout(
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    failure_predicate: Callable[[Tensor], Tensor],
    *,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    path_count: int,
    brownian_increments: Optional[Tensor],
    return_terms: Optional[ReturnTerms],
    integrator: str,
    detector: Optional[ContactSubstepping],
    truncation: Optional[ContactExclusion] = None,
    skip_rows: Optional[Tensor] = None,
) -> Tuple[RolloutBatch, Optional[Tensor]]:
    """stochastic_rollout without contact substepping; with a detector also the paths
    [B*L] (bool, row-major) that contact_flags flags on any substep, else None. With
    truncation or skip_rows (contact exclusion) the batch carries truncated [B,L,1]."""
    rk4 = check_model_integrator(integrator) == "rk4"
    continuous = stage_feedback(controller)
    refresh_steps = grid_refresh_steps(controller, controller_refresh_steps)
    counts, sizes, flags = refresh_aligned_grid(
        durations,
        maximum_step,
        physics_dt=physics_dt,
        controller_refresh_steps=refresh_steps,
    )
    flags = controller_refresh_flags(controller, flags)
    batch = states.shape[0]
    if brownian_increments is None:
        brownian_increments = sample_brownian_increments(
            durations,
            path_count=path_count,
            noise_dim=plant.noise_dim,
            maximum_step=maximum_step,
            physics_dt=physics_dt,
            controller_refresh_steps=refresh_steps,
        ).to(states.dtype)
    path_count = brownian_increments.shape[1]
    if brownian_increments.shape[:3] != (batch, path_count, sizes.shape[1]):
        raise ValueError("Brownian increments do not match the integration grid.")

    resolved = controller.resolve_decision(states, decisions)
    paths = batch * path_count
    current = states.repeat_interleave(path_count, dim=0)
    resolved = resolved.repeat_interleave(path_count, dim=0)
    row_counts = counts.repeat_interleave(path_count, dim=0)
    row_sizes = sizes.repeat_interleave(path_count, dim=0)
    row_flags = flags.repeat_interleave(path_count, dim=0)
    kicks = brownian_increments.flatten(0, 1)
    held = states.new_zeros(paths, plant.action_dim)
    alive = torch.ones(paths, dtype=torch.bool, device=states.device)
    elapsed = states.new_zeros(paths)
    failure_times = torch.full_like(elapsed, math.inf)
    running = torch.zeros(paths, dtype=torch.float64, device=states.device)
    payoffs = torch.zeros(paths, dtype=torch.float64, device=states.device)
    touched = None
    if detector is not None:
        touched = torch.zeros(paths, dtype=torch.bool, device=states.device)
    excluding = truncation is not None or skip_rows is not None
    stopped = None
    if excluding:
        stopped = torch.zeros(paths, dtype=torch.bool, device=states.device)
        if skip_rows is not None:
            stopped = skip_rows.reshape(-1).to(torch.bool).repeat_interleave(path_count, dim=0)

    for step in range(int(counts.max())):
        moving = alive & (row_counts > step)
        if stopped is not None:
            moving = moving & ~stopped
        rows = torch.nonzero(moving).squeeze(-1)
        if rows.numel() == 0:
            break
        here = current[rows]
        mask = row_flags[rows, step]
        actions = held[rows]
        law = None
        if continuous:
            law = _stage_law(controller, resolved[rows])
            actions = law(here)
        elif mask.any():
            actions = torch.where(
                mask.unsqueeze(-1), controller.action(here, resolved[rows]), actions
            )
        held[rows] = actions
        h = row_sizes[rows, step]
        following = model_substep(
            plant,
            here,
            actions,
            h,
            integrator=integrator,
            brownian_increments=kicks[rows, step],
            feedback=law,
        )
        if truncation is not None:
            # Contact exclusion: a path whose substep comes near a limit stops at the
            # substep's start (its last clean knot), which stays its endpoint.
            hit = contact_flags(truncation, here, following, h)
            if bool(hit.any()):
                stopped[rows[hit]] = True
                keep = ~hit
                rows, here, following = rows[keep], here[keep], following[keep]
                actions, h = actions[keep], h[keep]
                if continuous:
                    law = _stage_law(controller, resolved[rows])
                if rows.numel() == 0:
                    continue
        current[rows] = following
        if touched is not None:
            touched[rows] |= contact_flags(detector, here, following, h)
        if return_terms is not None:
            lam, rate = return_terms["discount_rate"], return_terms["reward_rate"]
            start = elapsed[rows].double()
            before = torch.exp(-lam * start) * rate(here, actions).squeeze(-1).double()
            after = torch.exp(-lam * (start + h.double())) * rate(
                following, law(following) if law is not None and rk4 else actions
            ).squeeze(-1).double()
            running[rows] += 0.5 * h.double() * (before + after)
        elapsed[rows] += h
        failed = failure_predicate(following).reshape(-1).bool()
        if failed.any():
            ended = rows[failed]
            alive[ended] = False
            failure_times[ended] = elapsed[ended]
            if return_terms is not None:
                payoffs[ended] = (
                    torch.exp(-return_terms["discount_rate"] * elapsed[ended].double())
                    * return_terms["terminal_payoff"](current[ended]).squeeze(-1).double()
                )

    result: RolloutBatch = {
        "endpoints": current.unflatten(0, (batch, path_count)),
        "survived": alive.reshape(batch, path_count, 1),
        "failure_times": failure_times.reshape(batch, path_count, 1),
        "durations_used": elapsed.reshape(batch, path_count, 1),
        "running_rewards": running.reshape(batch, path_count, 1),
        "failure_payoffs": payoffs.reshape(batch, path_count, 1),
        "brownian_increments": brownian_increments,
        "step_counts": counts,
        "step_sizes": sizes,
    }
    if stopped is not None:
        result["truncated"] = stopped.reshape(batch, path_count, 1)
    return result, touched


@torch.no_grad()
def path_continuations(
    value: SmoothValueHead, rollouts: RolloutBatch, horizons: Tensor, discount_rate: float
) -> Tensor:
    """e^(-lambda H) 1{survives} V(endpoint), double [B,L,1]; failed paths continue with zero.

    A path contact exclusion truncated (rollouts["truncated"]) is bootstrapped where it
    stopped: e^(-lambda t_c) V(endpoint) with t_c its durations_used (its running reward
    covers [0, t_c]); a row skipped altogether thus continues with V(start).
    """
    values = value(rollouts["endpoints"]).double()
    survived = rollouts["survived"].double()
    elapsed = horizons.reshape(-1, 1, 1).double()
    truncated = rollouts.get("truncated")
    if truncated is not None:
        elapsed = torch.where(truncated, rollouts["durations_used"].double(), elapsed)
    discount = torch.exp(-discount_rate * elapsed)
    return discount * survived * values


@torch.no_grad()
def survivor_continuation(
    value: SmoothValueHead,
    rollouts: RolloutBatch,
    observed_next_states: Tensor,
    durations: Tensor,
    discount_rate: float,
) -> Tensor:
    """Discount the survivor mean; use the observed endpoint if no path survives.

    A path contact exclusion truncated (rollouts["truncated"]: it came near a limit, or
    its row was skipped) takes the observed endpoint's value in place of its own: every
    rollout here spans one decision interval, so its last clean interval is the observed
    one, and the continuation of such a row is e^(-lambda Delta) V(s'_observed) (with every
    path truncated: exactly the data target's continuation).
    """
    values = value(rollouts["endpoints"])
    truncated = rollouts.get("truncated")
    if truncated is not None:
        observed = value(observed_next_states).unsqueeze(1).expand_as(values)
        values = torch.where(truncated, observed, values)
    survived = rollouts["survived"].to(values.dtype)
    survivors = survived.sum(dim=1)
    mean = (values * survived).sum(dim=1) / survivors.clamp_min(1.0)
    fallback = value(observed_next_states)
    chosen = torch.where(survivors > 0.0, mean, fallback)
    return torch.exp(-discount_rate * durations.reshape(-1, 1).to(values.dtype)) * chosen
