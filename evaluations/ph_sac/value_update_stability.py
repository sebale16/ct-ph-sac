"""Stability of three value-update rules on the lqr testbeds: a policy-evaluation ablation.

Explicit CT-PH-SAC with the generator continuation diverged on lqr in the Vista sweep while
the rollout and implicit arms did not. This module isolates the value step: one value
network (the 256-256 tanh network and StateFeatures of SmoothValueHead, the explicit arm's
Adam at learning rate 3e-4 with gradient-norm clip 10, minibatches of 256), one fixed
batch of states, and the known linear plant of the testbed (environment/ph_sac/dmc/lqr.py:
drift b(x, u) = A x + B u from lqr_oracle_callbacks, no diffusion, reward rate
r(x, u) = 1 - |q|^2/2 - c |u|^2/2 from lqr_reward_rate_tensor). It compares three rules
for the value flow dV/ds = r^pi + (L^pi - lambda) V (Section 4, eq. valueflow), each with
a step T and the target head V_bar:

  generator   eq. explicit-gen, the explicit (forward) Euler step:
              y = V_bar + T (r + b . grad V_bar - lambda V_bar); M Adam steps of V on
              (V - y)^2, then the Polyak step V_bar <- (1 - eta) V_bar + eta V.
  implicit    eq. resolvent with the residual of eq. residual, the implicit (backward)
              Euler step: K Adam steps of V on ((1 + lambda T) V - V_bar - T (b . grad V
              + r))^2, then V_bar <- V (hard copy), as ImplicitCTPHSAC.solve_value and
              copy_solved_value; one deterministic decision, so the all-pairs estimator of
              algorithms/ph_resolvent.py reduces to the squared residual.
  rollout     eq. explicit-roll, the interval backup (method of characteristics):
              y = int_0^T e^{-lambda s} r(x_s, u_s) ds + e^{-lambda T} V_bar(x_T), x_s the
              exact path of the known plant (matrix exponential; the integral by Van Loan's
              block exponential); M Adam steps and the Polyak rate eta of the generator.

Mode analysis (Section 4.2, eq. thresholds, Lemma modes; frozen coefficients, pure drift,
exact regressions). With z = i k.b - lambda an error mode e^{i k.x} is multiplied per
update (target step) by
  generator   1 + eta T z, modulus sqrt((1 - eta T lambda)^2 + (eta T k.b)^2), which
              exceeds one iff |k.b| > k_c = sqrt(lambda (2/(eta T) - lambda))
              ~ sqrt(2 lambda/(eta T)) (every mode grows once eta T lambda >= 2);
  implicit    1/(1 - T z), modulus 1/sqrt((1 + lambda T)^2 + (T k.b)^2) < 1 for every k and
              falling like 1/(T |k.b|): unconditionally stable and smoothing; the
              resolvent's norm is at most 1/(1 + lambda T) (Theorem 5.2, Hille-Yosida);
  rollout     1 - eta + eta e^{T z}, modulus at most 1 - eta (1 - e^{-lambda T}) < 1, every
              mode alike: stable, no smoothing.
On a linear plant under a linear policy u = K x the functions c + x^T E x are invariant,
with z running over mu_i + mu_j - lambda (mu the eigenvalues of A + B K) and -lambda, so
quadratic_spectral_radius is the exact per-update factor of each rule on that subspace.
The pairing M = eta * UPDATE_STEPS (80 Adam steps per hard copy, 1 per Polyak step at
eta = 0.0125) and K = UPDATE_STEPS gives every rule the pseudo-time T per UPDATE_STEPS
Adam steps.

Ground truth. For u = K x, V^pi(x) = 1/lambda - x^T P x with P from the discounted
Lyapunov equation
  (A_cl - lambda/2)^T P + P (A_cl - lambda/2) + Q_pi = 0,   A_cl = A + B K,
  Q_pi = diag(I_n, 0)/2 + c K^T K/2,
which makes the evaluation residual r + b . grad V^pi - lambda V^pi vanish identically
(hjb_residual). V*(x) = 1/lambda - x^T P* x with P* from the discounted Riccati equation of
(A - lambda/2, B, diag(I_n, 0)/2, c I/2), u* = K* x, K* = -(2/c) B^T P*; the torque that
maximizes r + b . grad V pointwise is clip(B^T grad V / c, -limit, limit).

Stage A (fixed policy u = K* x, and u = 0.5 K* x as a clearly suboptimal target): each rule
on the same network initialisation, data and minibatch sequence per seed, over the sweep
T in {reference period, 0.1 s, 1 s} x eta in {1, explicit Polyak rate}; the implicit rule has
no eta and its runs serve both rows. Two starts: SmoothValueHead's initialisation
('random', where the error is dominated by the smooth offset 1/lambda - x^T P x), and a
supervised fit of V^pi by PRETRAIN_STEPS Adam steps ('fit', the fixed point up to the
fit error, so any growth from that floor is the rule's). Data: DATA_SIZE states uniform in
a box whose half-widths are 1.1 times the largest coordinate reached by the K* and 0.5 K*
closed loops over 10 s from 200 reset draws; there |K* x| stays far below the motor limit,
so V^pi is exactly quadratic. Errors are measured on a separate batch from the same box.

Stage B (actor coupling): the policy is the greedy torque of the target head V_bar, read
anew at every Adam step by the generator and rollout rules and frozen during each
implicit solve (V_bar is fixed then, as the repo freezes the operator's policy). The
data is a FIFO replay of DATA_SIZE states refilled with one state per Adam step: every
COLLECT_EVERY steps, COLLECT_PATHS paths of COLLECT_DECISIONS decisions from the task's
reset law under the current greedy torque plus Gaussian exploration noise, held over each
reference period (exact zero-order hold). The rollout target holds the start torque over
T, as the repo's rollout holds the decision, so its fixed point is the held-decision value
(an O(T) shift from V*). Errors are measured against V* on the Stage A evaluation batch.
The implicit rule runs at T = reference period with K = UPDATE_STEPS and at the repo's
resolvent step 1 s with the repo's K = 400 inner steps. Controls separate the two
couplings for the generator at Stage B's (T, eta): the policy fixed at K* and/or the data
fixed to the Stage A box batch, or to one replay batch collected once under K* and never
refreshed (STAGE_B_CONTROL_STEPS Adam steps each).

Divergence: a member stops when its loss is non-finite, or its RMS value or gradient
error exceeds DIVERGENCE_FACTOR times its error at step 0. Members run as one batched
ensemble per rule (ValueEnsemble), each with its own Adam moments and gradient clip, so
they are independent runs computed together.

Usage (from the repository root):
    python evaluations/ph_sac/value_update_stability.py --task lqr_2_1 --stage both \
        --out out/generator_stability [--quick] [--plot-only]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from benchmarks.ph_hyperparams import find_ph_entry  # noqa: E402
from environment.ph_sac.dmc.lqr import (  # noqa: E402
    lqr_oracle_callbacks,
    lqr_reward_rate_tensor,
    lqr_state_matrices,
)
from environment.ph_sac.registry import testbed_spec  # noqa: E402
from models.ph_sac.networks import SmoothValueHead  # noqa: E402

RULES = ("generator", "implicit", "rollout")
UPDATE_STEPS = 80
DATA_SIZE = 4096
EVALUATION_SIZE = 2048
LINE_POINTS = 256
BOX_MARGIN = 1.1
DIVERGENCE_FACTOR = 1e3
STAGE_A_STEPS = 8000
STAGE_A_FIT_STEPS = 4000
STAGE_A_HALF_GAIN_STEPS = 3000
STAGE_A_EVALUATE_EVERY = 160
PRETRAIN_STEPS = 4000
STAGE_B_STEPS = 40000
STAGE_B_IMPLICIT_STEPS = 20000
STAGE_B_CONTROL_STEPS = 30000
STAGE_B_EVALUATE_EVERY = 400
STAGE_B_IMPLICIT_LONG_STEP = 1.0
COLLECT_EVERY = 400
COLLECT_PATHS = 4
COLLECT_DECISIONS = 100
EXPLORATION_STD = 2.0
FIGURE_DIRECTORY = REPOSITORY_ROOT / "docs" / "paper" / "figures"


# ----------------------------------------------------------------------------- problem


@dataclass(frozen=True)
class LQRProblem:
    """The testbed's linear plant, reward constants and the protocol values of its table row."""

    task: str
    config: Mapping[str, object]
    network: Mapping[str, object]
    state_matrix: np.ndarray
    input_matrix: np.ndarray
    discount_rate: float
    control_cost: float
    torque_limit: float
    position_radius: float
    velocity_noise: float
    reference_period: float
    learning_rate: float
    gradient_clip: float
    batch_size: int
    explicit_polyak_rate: float
    resolvent_step: float
    inner_steps: int

    @property
    def state_dim(self) -> int:
        return int(self.state_matrix.shape[0])

    @property
    def n_bodies(self) -> int:
        return self.state_dim // 2

    @property
    def action_dim(self) -> int:
        return int(self.input_matrix.shape[1])

    def state_cost(self) -> np.ndarray:
        """Q = diag(I_n, 0)/2: r = 1 - x^T Q x - u^T R u."""
        return 0.5 * np.diag([1.0] * self.n_bodies + [0.0] * self.n_bodies)

    def control_weight(self) -> np.ndarray:
        """R = c I/2."""
        return 0.5 * self.control_cost * np.eye(self.action_dim)


def load_problem(task: str) -> LQRProblem:
    """Build the problem of table row lqr-<task> (top mode) through the registry factory."""
    entry = find_ph_entry(f"lqr-{task}", "top").config
    config = testbed_spec(entry["testbed"]).config_factory(**entry["testbed_settings"])
    state_matrix, input_matrix = lqr_state_matrices(config)
    controller = config["controller"]
    limit = float(
        min(np.min(np.abs(controller["torque_low"])), np.min(np.abs(controller["torque_high"])))
    )
    training = entry["training"]
    solver = entry["implicit"]["solver"]
    return LQRProblem(
        task=task,
        config=config,
        network=dict(entry["network"]),
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        discount_rate=float(config["discount_rate"]),
        control_cost=float(config["reward_parameters"]["control_cost"]),
        torque_limit=limit,
        position_radius=float(config["reset_parameters"]["position_radius"]),
        velocity_noise=float(config["reset_parameters"]["velocity_noise"]),
        reference_period=float(config["reference_period"]),
        learning_rate=float(training["learning_rate"]),
        gradient_clip=float(training["gradient_clip"]),
        batch_size=int(training["batch_size"]),
        explicit_polyak_rate=float(entry["explicit"]["polyak_rate"]),
        resolvent_step=float(solver["resolvent_step"]),
        inner_steps=int(solver["inner_steps"]),
    )


def riccati_solution(problem: LQRProblem) -> Tuple[np.ndarray, np.ndarray]:
    """(P*, K*): V*(x) = 1/lambda - x^T P* x, u*(x) = K* x, the discounted LQR optimum."""
    shifted = problem.state_matrix - 0.5 * problem.discount_rate * np.eye(problem.state_dim)
    weight = problem.control_weight()
    riccati = scipy.linalg.solve_continuous_are(
        shifted, problem.input_matrix, problem.state_cost(), weight
    )
    riccati = 0.5 * (riccati + riccati.T)
    return riccati, -np.linalg.solve(weight, problem.input_matrix.T @ riccati)


def closed_loop(problem: LQRProblem, gain: np.ndarray) -> np.ndarray:
    """A_cl = A + B K."""
    return problem.state_matrix + problem.input_matrix @ gain


def policy_value_matrix(problem: LQRProblem, gain: np.ndarray) -> np.ndarray:
    """P with V^pi(x) = 1/lambda - x^T P x for u = K x: the discounted Lyapunov equation."""
    shifted = closed_loop(problem, gain) - 0.5 * problem.discount_rate * np.eye(problem.state_dim)
    cost = problem.state_cost() + gain.T @ problem.control_weight() @ gain
    solution = scipy.linalg.solve_continuous_lyapunov(shifted.T, -cost)
    return 0.5 * (solution + solution.T)


def quadratic_value(
    value_matrix: np.ndarray, discount_rate: float, states: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(1/lambda - x^T P x, -2 P x) for states [B,D]."""
    values = 1.0 / discount_rate - np.einsum("bi,ij,bj->b", states, value_matrix, states)
    return values, -2.0 * states @ value_matrix


def hjb_residual(
    problem: LQRProblem, gain: np.ndarray, value_matrix: np.ndarray, states: np.ndarray
) -> np.ndarray:
    """r(x, K x) + grad V(x) . b(x, K x) - lambda V(x), [B], with the testbed's tensor functions."""
    points = torch.as_tensor(states, dtype=torch.float64)
    torques = points @ torch.as_tensor(gain, dtype=torch.float64).T
    drift_function, _ = lqr_oracle_callbacks(problem.config)
    rates = lqr_reward_rate_tensor(points, torques, problem.config).squeeze(-1).numpy()
    drift = drift_function(points, torques).numpy()
    values, gradients = quadratic_value(value_matrix, problem.discount_rate, states)
    return rates + np.sum(gradients * drift, axis=-1) - problem.discount_rate * values


def discounted_flow(
    dynamics: np.ndarray, cost: np.ndarray, discount_rate: float, step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """(e^{G T}, W_T), W_T = int_0^T e^{-lambda s} e^{G^T s} cost e^{G s} ds by Van Loan's method.

    With F = G - lambda/2 I, exp([[-F^T, cost], [0, F]] T) has the blocks [[., X], [0, e^{F T}]]
    and W_T = e^{F T}^T X.
    """
    size = dynamics.shape[0]
    shifted = dynamics - 0.5 * discount_rate * np.eye(size)
    block = np.block([[-shifted.T, cost], [np.zeros((size, size)), shifted]])
    exponential = scipy.linalg.expm(block * step)
    weight = exponential[size:, size:].T @ exponential[:size, size:]
    return scipy.linalg.expm(dynamics * step), 0.5 * (weight + weight.T)


def held_torque_dynamics(problem: LQRProblem) -> Tuple[np.ndarray, np.ndarray]:
    """The augmented (x, u) system with u held, [[A, B], [0, 0]], and its cost diag(Q, R)."""
    size, inputs = problem.state_dim, problem.action_dim
    dynamics = np.zeros((size + inputs, size + inputs))
    dynamics[:size, :size] = problem.state_matrix
    dynamics[:size, size:] = problem.input_matrix
    return dynamics, scipy.linalg.block_diag(problem.state_cost(), problem.control_weight())


def zero_order_hold(problem: LQRProblem, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """(Phi, Gamma) with x(t + period) = Phi x + Gamma u for a torque held over the period."""
    dynamics, _ = held_torque_dynamics(problem)
    exponential = scipy.linalg.expm(dynamics * period)
    size = problem.state_dim
    return exponential[:size, :size], exponential[:size, size:]


# ----------------------------------------------------------------------------- mode analysis


def drift_threshold(step: float, polyak_rate: float, discount_rate: float) -> float:
    """k_c = sqrt(lambda (2/(eta T) - lambda)): generator modes with |k.b| > k_c grow; 0 if all do."""
    effective = polyak_rate * step
    if effective * discount_rate >= 2.0:
        return 0.0
    return math.sqrt(discount_rate * (2.0 / effective - discount_rate))


def amplification(rule: str, exponent, step: float, polyak_rate: float = 1.0) -> np.ndarray:
    """Per-update multiplier of an error mode with generator exponent z (z = i k.b - lambda).

    generator 1 + eta T z; implicit 1/(1 - T z) (hard copy, eta unused); rollout
    1 - eta + eta e^{T z}.
    """
    exponent = np.asarray(exponent, dtype=complex)
    if rule == "generator":
        return 1.0 + polyak_rate * step * exponent
    if rule == "implicit":
        return 1.0 / (1.0 - step * exponent)
    if rule == "rollout":
        return 1.0 - polyak_rate + polyak_rate * np.exp(step * exponent)
    raise ValueError(f"Unknown rule {rule!r}; expected one of {RULES}.")


def quadratic_exponents(closed: np.ndarray, discount_rate: float) -> np.ndarray:
    """The exponents of L - lambda on c + x^T E x: mu_i + mu_j - lambda (i <= j) and -lambda."""
    eigenvalues = np.linalg.eigvals(closed)
    upper = np.triu_indices(len(eigenvalues))
    pairs = (eigenvalues[:, None] + eigenvalues[None, :])[upper]
    return np.concatenate((pairs - discount_rate, [-discount_rate + 0j]))


def quadratic_spectral_radius(
    rule: str, closed: np.ndarray, discount_rate: float, step: float, polyak_rate: float
) -> float:
    """The exact per-update factor of a rule on the quadratic subspace of a linear closed loop."""
    exponents = quadratic_exponents(closed, discount_rate)
    rate = 1.0 if rule == "implicit" else polyak_rate
    return float(np.abs(amplification(rule, exponents, step, rate)).max())


# ----------------------------------------------------------------------------- network


class ValueEnsemble(nn.Module):
    """E independent SmoothValueHead networks evaluated in one batched pass.

    Member e holds the Linear weights of heads[e], transposed and stacked on a leading
    member axis, and the heads' parameter-free StateFeatures, so forward(states)[e] equals
    heads[e](states[e]) for states [E,B,D]: tanh hidden layers, a linear output. The lqr
    head carries no terminal lift (no failure set: H = 0, B = 1), so V is the network.
    """

    def __init__(self, heads: Sequence[SmoothValueHead]) -> None:
        super().__init__()
        if not heads:
            raise ValueError("The ensemble needs at least one head.")
        if any(head._lift is not None for head in heads):
            raise ValueError("ValueEnsemble mirrors heads without a terminal lift.")
        self.features = copy.deepcopy(heads[0].features)
        layers = [[module for module in head.network if isinstance(module, nn.Linear)] for head in heads]
        self.weights = nn.ParameterList(
            nn.Parameter(torch.stack([layer[index].weight.detach().T.clone() for layer in layers]))
            for index in range(len(layers[0]))
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.stack([layer[index].bias.detach().clone()[None, :] for layer in layers]))
            for index in range(len(layers[0]))
        )

    @property
    def members(self) -> int:
        return int(self.weights[0].shape[0])

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.features(states)
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.baddbmm(bias, hidden, weight)
            if index < last:
                hidden = torch.tanh(hidden)
        return hidden

    def frozen_copy(self) -> ValueEnsemble:
        frozen = copy.deepcopy(self)
        frozen.requires_grad_(False)
        return frozen

    def select(self, indices: Sequence[int]) -> ValueEnsemble:
        """A trainable ensemble whose member j copies member indices[j] of this one."""
        chosen = copy.deepcopy(self)
        index = torch.as_tensor(list(indices), dtype=torch.long)
        with torch.no_grad():
            for parameter in chosen.parameters():
                parameter.data = parameter.data[index].clone()
        chosen.requires_grad_(True)
        return chosen


def value_heads(
    problem: LQRProblem, seeds: Sequence[int], network: Optional[Mapping[str, object]] = None
) -> List[SmoothValueHead]:
    """One SmoothValueHead per seed, each initialised under torch.manual_seed(seed)."""
    heads = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        heads.append(SmoothValueHead(problem.config["state_spec"], network or problem.network))
    return heads


def value_and_gradient(
    network: nn.Module, states: torch.Tensor, *, create_graph: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V and grad_x V at states [E,B,D]; with create_graph the gradient stays differentiable."""
    with torch.enable_grad():
        points = states.detach().contiguous().requires_grad_(True)
        values = network(points)
        gradients = torch.autograd.grad(values.sum(), points, create_graph=create_graph)[0]
    return values, gradients


# ----------------------------------------------------------------------------- rules


@dataclass(frozen=True)
class LinearPolicy:
    """u = K x (Stage A); reads no value."""

    gain: torch.Tensor
    needs_gradient = False

    def __call__(self, states: torch.Tensor, gradients: Optional[torch.Tensor]) -> torch.Tensor:
        return states @ self.gain.T


@dataclass(frozen=True)
class GreedyPolicy:
    """u = clip(B^T grad V_bar / c, -limit, limit), the maximizer of r + b . grad V_bar (Stage B)."""

    input_matrix: torch.Tensor
    control_cost: float
    torque_limit: float
    needs_gradient = True

    def __call__(self, states: torch.Tensor, gradients: Optional[torch.Tensor]) -> torch.Tensor:
        raw = gradients @ self.input_matrix / self.control_cost
        return raw.clamp(-self.torque_limit, self.torque_limit)


@dataclass(frozen=True)
class RolloutFlows:
    """Per-member exact interval maps: x_T = z @ endpoint, int = offset - z^T W z.

    z = x under a linear policy (closed loop), z = (x, u) for a torque held over T.
    """

    endpoint: torch.Tensor
    weight: torch.Tensor
    offset: torch.Tensor
    discount: torch.Tensor
    held_torque: bool


def rollout_flows(
    problem: LQRProblem, steps: Sequence[float], gain: Optional[np.ndarray] = None
) -> RolloutFlows:
    """The closed loop of u = gain x, or the held-torque system when gain is None."""
    if gain is None:
        dynamics, cost = held_torque_dynamics(problem)
    else:
        dynamics = closed_loop(problem, gain)
        cost = problem.state_cost() + gain.T @ problem.control_weight() @ gain
    endpoints, weights = [], []
    for step in steps:
        exponential, weight = discounted_flow(dynamics, cost, problem.discount_rate, step)
        endpoints.append(exponential[: problem.state_dim, :].T)
        weights.append(weight)
    rate = problem.discount_rate
    steps_array = np.asarray(steps, dtype=np.float64)
    dtype = torch.get_default_dtype()
    return RolloutFlows(
        endpoint=torch.as_tensor(np.stack(endpoints), dtype=dtype),
        weight=torch.as_tensor(np.stack(weights), dtype=dtype),
        offset=torch.as_tensor((1.0 - np.exp(-rate * steps_array)) / rate, dtype=dtype).view(-1, 1, 1),
        discount=torch.as_tensor(np.exp(-rate * steps_array), dtype=dtype).view(-1, 1, 1),
        held_torque=gain is None,
    )


@dataclass(frozen=True)
class PlantFunctions:
    """The testbed's tensor drift and reward rate, with the discount rate."""

    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    reward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    discount_rate: float


def plant_functions(problem: LQRProblem) -> PlantFunctions:
    drift, _ = lqr_oracle_callbacks(problem.config)
    config = problem.config
    return PlantFunctions(
        drift=drift,
        reward=lambda states, torques: lqr_reward_rate_tensor(states, torques, config),
        discount_rate=problem.discount_rate,
    )


def rule_losses(
    rule: str,
    online: ValueEnsemble,
    target: ValueEnsemble,
    states: torch.Tensor,
    policy,
    plant: PlantFunctions,
    steps: torch.Tensor,
    flows: Optional[RolloutFlows] = None,
) -> torch.Tensor:
    """Per-member losses [E] of one Adam step of a rule at states [E,B,D]; steps is [E,1,1]."""
    rate = plant.discount_rate
    anchor, anchor_gradients = None, None
    if policy.needs_gradient or rule == "generator":
        anchor, anchor_gradients = value_and_gradient(target, states)
        anchor, anchor_gradients = anchor.detach(), anchor_gradients.detach()
    elif rule == "implicit":
        with torch.no_grad():
            anchor = target(states)
    with torch.no_grad():
        torques = policy(states, anchor_gradients)
    if rule == "rollout":
        if flows is None:
            raise ValueError("The rollout rule needs its interval flows.")
        with torch.no_grad():
            inputs = torch.cat((states, torques), dim=-1) if flows.held_torque else states
            endpoints = torch.bmm(inputs, flows.endpoint)
            integral = flows.offset - (torch.bmm(inputs, flows.weight) * inputs).sum(-1, keepdim=True)
            goal = integral + flows.discount * target(endpoints)
        return (online(states) - goal).square().mean(dim=(1, 2))
    with torch.no_grad():
        rates = plant.reward(states, torques)
        drift = plant.drift(states, torques)
    if rule == "generator":
        transport = (anchor_gradients * drift).sum(-1, keepdim=True)
        goal = anchor + steps * (rates + transport - rate * anchor)
        return (online(states) - goal).square().mean(dim=(1, 2))
    if rule == "implicit":
        values, gradients = value_and_gradient(online, states, create_graph=True)
        transport = (gradients * drift).sum(-1, keepdim=True)
        residual = (1.0 + rate * steps) * values - anchor - steps * (transport + rates)
        return residual.square().mean(dim=(1, 2))
    raise ValueError(f"Unknown rule {rule!r}; expected one of {RULES}.")


# ----------------------------------------------------------------------------- members


@dataclass(frozen=True)
class Member:
    """One run: rule, step T, Polyak rate eta (1 for a hard copy), Adam steps per update, seed."""

    rule: str
    step: float
    polyak_rate: float
    steps_per_update: int
    seed: int

    def outer_updates(self, gradient_steps: int) -> int:
        return gradient_steps // self.steps_per_update

    def pseudo_time(self, gradient_steps: int) -> float:
        """The value-flow time advanced: eta T per Polyak update (T per hard copy)."""
        return self.outer_updates(gradient_steps) * self.polyak_rate * self.step


def paired_update_steps(polyak_rate: float) -> int:
    """M = eta * UPDATE_STEPS: the same pseudo-time T per UPDATE_STEPS Adam steps for every eta."""
    steps = int(round(polyak_rate * UPDATE_STEPS))
    if steps < 1 or not math.isclose(steps, polyak_rate * UPDATE_STEPS, rel_tol=1e-9):
        raise ValueError(f"eta * {UPDATE_STEPS} must be a positive integer, got eta = {polyak_rate}.")
    return steps


def stage_a_members(
    rule: str, steps: Sequence[float], polyak_rates: Sequence[float], seeds: Sequence[int]
) -> List[Member]:
    if rule == "implicit":
        return [Member(rule, step, 1.0, UPDATE_STEPS, seed) for step in steps for seed in seeds]
    return [
        Member(rule, step, rate, paired_update_steps(rate), seed)
        for step in steps
        for rate in polyak_rates
        for seed in seeds
    ]


def stage_b_members(
    rule: str, problem: LQRProblem, seeds: Sequence[int], reference_step: float
) -> List[Member]:
    """Generator and rollout at (T, eta) = (reference period, explicit rate); implicit at T and 1 s."""
    if rule == "implicit":
        return [
            Member(rule, reference_step, 1.0, UPDATE_STEPS, seed) for seed in seeds
        ] + [
            Member(rule, STAGE_B_IMPLICIT_LONG_STEP, 1.0, problem.inner_steps, seed)
            for seed in seeds
        ]
    rate = problem.explicit_polyak_rate
    return [Member(rule, reference_step, rate, paired_update_steps(rate), seed) for seed in seeds]


def clip_per_member(parameters, max_norm: float) -> None:
    """clip_grad_norm_ applied to each member's gradient (leading axis) separately."""
    parameters = list(parameters)
    with torch.no_grad():
        norms = torch.sqrt(sum(p.grad.square().flatten(1).sum(1) for p in parameters))
        scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
        for parameter in parameters:
            parameter.grad.mul_(scale.view(-1, *([1] * (parameter.dim() - 1))))


def pretrain_values(
    problem: LQRProblem,
    seeds: Sequence[int],
    data: torch.Tensor,
    value_matrix: np.ndarray,
    steps: int,
) -> ValueEnsemble:
    """Per-seed networks regressed on V^pi = 1/lambda - x^T P x over their seed's data [S,N,D].

    The same Adam, clip and minibatch size as the rules; the result is the 'fit' start.
    """
    network = ValueEnsemble(value_heads(problem, seeds))
    optimizer = torch.optim.Adam(network.parameters(), lr=problem.learning_rate)
    matrix = torch.as_tensor(value_matrix, dtype=torch.get_default_dtype())
    sampler = torch.Generator().manual_seed(7)
    rows = torch.arange(len(seeds))[:, None]
    for _ in range(int(steps)):
        states = data[rows, torch.randint(data.shape[1], (len(seeds), problem.batch_size), generator=sampler)]
        truth = 1.0 / problem.discount_rate - torch.einsum("ebi,ij,ebj->eb", states, matrix, states)
        loss = (network(states).squeeze(-1) - truth).square().mean(dim=1).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_per_member(network.parameters(), problem.gradient_clip)
        optimizer.step()
    return network


class EnsembleTrainer:
    """Online and target ensembles, one Adam over the stacked weights, per-member clipping.

    Adam is elementwise, so stacking members changes nothing; the gradient-norm clip is
    applied per member as torch.nn.utils.clip_grad_norm_ would to each network alone.
    """

    def __init__(self, online: ValueEnsemble, members: Sequence[Member], problem: LQRProblem) -> None:
        self.online = online
        self.target = self.online.frozen_copy()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=problem.learning_rate)
        self.gradient_clip = problem.gradient_clip
        count = len(members)
        dtype = torch.get_default_dtype()
        self.steps = torch.tensor([member.step for member in members], dtype=dtype).view(count, 1, 1)
        self.polyak = torch.tensor([member.polyak_rate for member in members], dtype=dtype)
        self.update_every = torch.tensor([member.steps_per_update for member in members])
        self.active = torch.ones(count, dtype=torch.bool)
        self.diverged_at = [-1] * count
        self.stopped_by_loss = [False] * count

    def step(self, losses: torch.Tensor, gradient_step: int) -> None:
        """One Adam step on the active members' summed losses; non-finite losses stop a member."""
        for index in torch.nonzero(self.active & ~torch.isfinite(losses)).flatten().tolist():
            self.deactivate(index, gradient_step)
            self.stopped_by_loss[index] = True
        total = torch.where(self.active, losses, torch.zeros_like(losses)).sum()
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        with torch.no_grad():
            for parameter in self.online.parameters():
                parameter.grad[~self.active] = 0.0
        clip_per_member(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()

    def update_targets(self, completed_steps: int) -> None:
        """V_bar <- V_bar + eta (V - V_bar) for the members whose update is due."""
        due = (completed_steps % self.update_every == 0) & self.active
        rates = torch.where(due, self.polyak, torch.zeros_like(self.polyak))
        with torch.no_grad():
            for slow, fast in zip(self.target.parameters(), self.online.parameters()):
                slow.lerp_(fast, rates.view(-1, *([1] * (slow.dim() - 1))))

    def deactivate(self, index: int, gradient_step: int) -> None:
        """Stop a diverged member: zero its weights and Adam moments and record the step."""
        if not self.active[index]:
            return
        self.active[index] = False
        self.diverged_at[index] = int(gradient_step)
        with torch.no_grad():
            for parameter in list(self.online.parameters()) + list(self.target.parameters()):
                parameter[index] = 0.0
            for state in self.optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][index] = 0.0


# ----------------------------------------------------------------------------- data


def state_box(
    problem: LQRProblem, gains: Sequence[np.ndarray], *, paths: int = 200, duration: float = 10.0
) -> np.ndarray:
    """BOX_MARGIN times the largest |x_i| of the closed loops of the gains from reset draws."""
    rng = np.random.default_rng(0)
    largest = np.zeros(problem.state_dim)
    starts = reset_states_numpy(problem, paths, rng)
    for gain in gains:
        transition = scipy.linalg.expm(closed_loop(problem, gain) * 0.01)
        states = starts.copy()
        for _ in range(int(round(duration / 0.01)) + 1):
            largest = np.maximum(largest, np.abs(states).max(axis=0))
            states = states @ transition.T
    return BOX_MARGIN * largest


def reset_states_numpy(problem: LQRProblem, count: int, rng: np.random.Generator) -> np.ndarray:
    """The task's reset law: q uniform on the sphere of position_radius, qd velocity_noise N(0, I)."""
    directions = rng.standard_normal((count, problem.n_bodies))
    positions = problem.position_radius * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    velocities = problem.velocity_noise * rng.standard_normal((count, problem.n_bodies))
    return np.concatenate((positions, velocities), axis=1)


def reset_states(problem: LQRProblem, shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Reset-law states [*shape, D] drawn from a torch generator."""
    directions = torch.randn(*shape, problem.n_bodies, generator=generator)
    positions = problem.position_radius * directions / directions.norm(dim=-1, keepdim=True)
    velocities = problem.velocity_noise * torch.randn(*shape, problem.n_bodies, generator=generator)
    return torch.cat((positions, velocities), dim=-1)


def uniform_box_states(half_widths: np.ndarray, count: int, generator: torch.Generator) -> torch.Tensor:
    widths = torch.as_tensor(half_widths, dtype=torch.get_default_dtype())
    return (2.0 * torch.rand(count, len(half_widths), generator=generator) - 1.0) * widths


@dataclass(frozen=True)
class EvaluationSet:
    """States, ground truth and the drift of the reference closed loop, plus one line along b."""

    states: torch.Tensor
    values: torch.Tensor
    gradients: torch.Tensor
    drift: torch.Tensor
    line_states: torch.Tensor
    line_values: torch.Tensor
    line_spacing: float
    line_speed: float


def evaluation_set(
    problem: LQRProblem, value_matrix: np.ndarray, gain: np.ndarray, box: np.ndarray, seed: int
) -> EvaluationSet:
    """EVALUATION_SIZE box states and a LINE_POINTS line through x0 along b(x0) inside the box.

    x0 is the closed loop of gain 1 s after the reset (sqrt(2/n), ..., 0): a typical state.
    """
    generator = torch.Generator().manual_seed(seed)
    states = uniform_box_states(box, EVALUATION_SIZE, generator).double().numpy()
    closed = closed_loop(problem, gain)
    values, gradients = quadratic_value(value_matrix, problem.discount_rate, states)
    start = np.concatenate(
        (np.full(problem.n_bodies, problem.position_radius / math.sqrt(problem.n_bodies)), np.zeros(problem.n_bodies))
    )
    centre = scipy.linalg.expm(closed * 1.0) @ start
    velocity = closed @ centre
    direction = velocity / np.linalg.norm(velocity)
    with np.errstate(divide="ignore"):
        reach = np.where(np.abs(direction) > 1e-12, (box - np.abs(centre)) / np.abs(direction), np.inf)
    half_length = float(reach.min())
    offsets = np.linspace(-half_length, half_length, LINE_POINTS)
    line = centre[None, :] + offsets[:, None] * direction[None, :]
    line_values, _ = quadratic_value(value_matrix, problem.discount_rate, line)
    dtype = torch.get_default_dtype()
    return EvaluationSet(
        states=torch.as_tensor(states, dtype=dtype),
        values=torch.as_tensor(values, dtype=dtype),
        gradients=torch.as_tensor(gradients, dtype=dtype),
        drift=torch.as_tensor(states @ closed.T, dtype=dtype),
        line_states=torch.as_tensor(line, dtype=dtype),
        line_values=torch.as_tensor(line_values, dtype=dtype),
        line_spacing=float(offsets[1] - offsets[0]),
        line_speed=float(np.linalg.norm(velocity)),
    )


def line_spectrum(errors: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rough RMS, power-weighted mean k, 90%-power k) of the error along the line, [E] each.

    Wavenumbers are in radians per unit of arc length.

    The least-squares quadratic in the arc length (the part the quadratic analysis covers)
    is removed first; a Hann window precedes the FFT; the zero wavenumber is excluded.
    """
    count, points = errors.shape
    arc = (np.arange(points) - 0.5 * (points - 1)) * spacing
    basis = np.stack((np.ones(points), arc, arc**2), axis=1)
    finite = np.all(np.isfinite(errors), axis=1)
    rough = np.full(count, np.nan)
    centroid = np.full(count, np.nan)
    quantile = np.full(count, np.nan)
    if not finite.any():
        return rough, centroid, quantile
    coefficients = np.linalg.lstsq(basis, errors[finite].T, rcond=None)[0]
    residual = errors[finite] - (basis @ coefficients).T
    power = np.abs(np.fft.rfft(residual * np.hanning(points), axis=1)) ** 2
    power[:, 0] = 0.0
    wavenumbers = 2.0 * math.pi * np.fft.rfftfreq(points, d=spacing)
    total = np.maximum(power.sum(axis=1, keepdims=True), 1e-300)
    cumulative = np.cumsum(power, axis=1) / total
    rough[finite] = np.sqrt(np.mean(residual**2, axis=1))
    centroid[finite] = (power * wavenumbers).sum(axis=1) / total[:, 0]
    quantile[finite] = wavenumbers[np.argmax(cumulative >= 0.9, axis=1)]
    return rough, centroid, quantile


def measure(network: ValueEnsemble, evaluation: EvaluationSet) -> Dict[str, np.ndarray]:
    """RMS value and gradient errors, max |grad V|, kappa_eff and the line spectrum, [E] each.

    kappa_eff = RMS(b . grad e) / RMS(e - mean e), e = V - V_ref and b the reference closed
    loop's drift: the |k.b| of a single mode, an energy-weighted average otherwise.
    """
    members = network.members
    states = evaluation.states.expand(members, -1, -1)
    values, gradients = value_and_gradient(network, states)
    with torch.no_grad():
        errors = values.squeeze(-1) - evaluation.values
        gradient_errors = gradients - evaluation.gradients
        transport = (gradient_errors * evaluation.drift).sum(-1)
        centred = errors - errors.mean(dim=1, keepdim=True)
        line = network(evaluation.line_states.expand(members, -1, -1)).squeeze(-1) - evaluation.line_values
    rough, centroid, quantile = line_spectrum(line.double().numpy(), evaluation.line_spacing)
    return {
        "value_rms_error": errors.square().mean(1).sqrt().double().numpy(),
        "gradient_rms_error": gradient_errors.square().sum(-1).mean(1).sqrt().double().numpy(),
        "max_gradient_norm": gradients.norm(dim=-1).amax(1).double().numpy(),
        "kappa_eff": (transport.square().mean(1).sqrt() / centred.square().mean(1).sqrt()).double().numpy(),
        "line_rough_rms": rough,
        "line_kb_centroid": centroid * evaluation.line_speed,
        "line_kb_90": quantile * evaluation.line_speed,
    }


# ----------------------------------------------------------------------------- runs


@dataclass(frozen=True)
class RunSpec:
    """One ensemble run: a rule's members, their policy and data, the step budget."""

    stage: str
    policy_label: str
    rule: str
    members: Tuple[Member, ...]
    gradient_steps: int
    evaluate_every: int
    reference_gain: np.ndarray
    initialisation: str = "random"
    data_source: str = "fixed"


def _member_seed_indices(members: Sequence[Member], seeds: Sequence[int]) -> torch.Tensor:
    return torch.tensor([list(seeds).index(member.seed) for member in members])


def _predictions(problem: LQRProblem, member: Member, gain: np.ndarray) -> Dict[str, float]:
    return {
        "predicted_threshold": drift_threshold(member.step, member.polyak_rate, problem.discount_rate)
        if member.rule == "generator"
        else float("inf"),
        "quadratic_radius": quadratic_spectral_radius(
            member.rule, closed_loop(problem, gain), problem.discount_rate, member.step, member.polyak_rate
        ),
    }


def run_ensemble(
    problem: LQRProblem,
    spec: RunSpec,
    seeds: Sequence[int],
    evaluation: EvaluationSet,
    *,
    policy,
    data: Optional[torch.Tensor] = None,
    initial: Optional[ValueEnsemble] = None,
    progress: Callable[[str], None] = print,
) -> List[Dict[str, object]]:
    """Train one rule's ensemble for spec.gradient_steps and return a row per member per evaluation.

    Members start from their seed's network: SmoothValueHead's initialisation, or the
    per-seed ensemble initial (the 'fit' start). Stage A reads the fixed data [S,N,D] (one
    batch per seed); Stage B keeps a replay per member, refilled from the greedy closed loop
    (ReplayCollector).
    """
    members = spec.members
    count = len(members)
    seed_index = _member_seed_indices(members, seeds)
    start = ValueEnsemble(value_heads(problem, seeds)) if initial is None else initial
    trainer = EnsembleTrainer(start.select(seed_index.tolist()), members, problem)
    plant = plant_functions(problem)
    flows = None
    if spec.rule == "rollout":
        held = policy.needs_gradient
        flows = rollout_flows(problem, [m.step for m in members], None if held else policy_gain(policy))
    batch = problem.batch_size
    sampler = torch.Generator().manual_seed(20261108)
    replay = None
    collector = None
    if spec.data_source in ("replay", "frozen replay"):
        collector = ReplayCollector(problem, count, seed_index, len(seeds))
        replay = collector.initial(trainer.target, policy)
    predictions = [_predictions(problem, member, spec.reference_gain) for member in members]
    initial = None
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    for gradient_step in range(spec.gradient_steps + 1):
        if gradient_step % spec.evaluate_every == 0 or gradient_step == spec.gradient_steps:
            iterate = measure(trainer.target, evaluation)
            online = measure(trainer.online, evaluation)
            extra = {}
            if spec.stage == "B":
                pool = replay if replay is not None else data[seed_index]
                extra = stage_b_measures(problem, trainer.target, evaluation, pool, policy)
            if initial is None:
                initial = (iterate["value_rms_error"].copy(), iterate["gradient_rms_error"].copy())
            for index in range(count):
                if not trainer.active[index]:
                    continue
                value_error = iterate["value_rms_error"][index]
                gradient_error = iterate["gradient_rms_error"][index]
                if not (
                    math.isfinite(value_error)
                    and math.isfinite(gradient_error)
                    and value_error <= DIVERGENCE_FACTOR * initial[0][index]
                    and gradient_error <= DIVERGENCE_FACTOR * initial[1][index]
                ):
                    trainer.deactivate(index, gradient_step)
            elapsed = time.perf_counter() - started
            for index, member in enumerate(members):
                alive = bool(trainer.active[index]) or (
                    trainer.diverged_at[index] == gradient_step and not trainer.stopped_by_loss[index]
                )
                row: Dict[str, object] = {
                    "stage": spec.stage,
                    "task": problem.task,
                    "policy": spec.policy_label,
                    "init": spec.initialisation,
                    "rule": member.rule,
                    "step": member.step,
                    "polyak_rate": member.polyak_rate,
                    "steps_per_update": member.steps_per_update,
                    "seed": member.seed,
                    "grad_steps": gradient_step,
                    "outer_updates": member.outer_updates(gradient_step),
                    "pseudo_time": member.pseudo_time(gradient_step),
                    **predictions[index],
                }
                for key, values in iterate.items():
                    row[key] = float(values[index]) if alive else float("nan")
                row["online_value_rms_error"] = float(online["value_rms_error"][index]) if alive else float("nan")
                row["online_gradient_rms_error"] = (
                    float(online["gradient_rms_error"][index]) if alive else float("nan")
                )
                for key, values in extra.items():
                    row[key] = float(values[index]) if alive else float("nan")
                row["diverged"] = int(trainer.diverged_at[index] >= 0)
                row["diverged_at_step"] = trainer.diverged_at[index]
                row["wall_time_s"] = elapsed
                row["members"] = count
                rows.append(row)
            alive_errors = iterate["gradient_rms_error"][trainer.active.numpy()]
            progress(
                f"[{spec.stage} {spec.policy_label} {spec.initialisation} {spec.rule}] step {gradient_step}/{spec.gradient_steps} "
                f"alive {int(trainer.active.sum())}/{count} grad-err median "
                f"{np.median(alive_errors) if alive_errors.size else float('nan'):.3g} {elapsed:.0f}s"
            )
            if not trainer.active.any() or gradient_step == spec.gradient_steps:
                break
        if spec.data_source == "replay" and gradient_step > 0 and gradient_step % COLLECT_EVERY == 0:
            collector.refill(replay, trainer.target, policy)
        indices = torch.randint(DATA_SIZE, (len(seeds), batch), generator=sampler)[seed_index]
        if replay is None:
            states = data[seed_index[:, None], indices]
        else:
            states = replay[torch.arange(count)[:, None], indices]
        losses = rule_losses(spec.rule, trainer.online, trainer.target, states, policy, plant, trainer.steps, flows)
        trainer.step(losses, gradient_step + 1)
        trainer.update_targets(gradient_step + 1)
    return rows


def policy_gain(policy: LinearPolicy) -> np.ndarray:
    return policy.gain.double().numpy()


class ReplayCollector:
    """Stage B's FIFO replay [E,N,D], one state per Adam step from the greedy closed loop.

    Paths start from the reset law; each decision holds clip(greedy + N(0, EXPLORATION_STD^2))
    over the reference period exactly (zero-order hold). Members with one seed share the
    reset and noise draws.
    """

    def __init__(self, problem: LQRProblem, count: int, seed_index: torch.Tensor, seeds: int) -> None:
        self.problem = problem
        self.count = count
        self.seed_index = seed_index
        self.seeds = seeds
        self.generator = torch.Generator().manual_seed(4242)
        transition, inputs = zero_order_hold(problem, problem.reference_period)
        dtype = torch.get_default_dtype()
        self.transition = torch.as_tensor(transition, dtype=dtype)
        self.inputs = torch.as_tensor(inputs, dtype=dtype)
        self.pointer = 0

    def collect(self, target: ValueEnsemble, policy, paths: int) -> torch.Tensor:
        """States [E, paths * COLLECT_DECISIONS, D] of fresh paths under the current greedy torque."""
        states = reset_states(self.problem, (self.seeds, paths), self.generator)[self.seed_index]
        limit = self.problem.torque_limit
        records = []
        for _ in range(COLLECT_DECISIONS):
            records.append(states)
            _, gradients = value_and_gradient(target, states)
            noise = torch.randn(self.seeds, paths, self.problem.action_dim, generator=self.generator)
            torques = (policy(states, gradients) + EXPLORATION_STD * noise[self.seed_index]).clamp(-limit, limit)
            states = states @ self.transition.T + torques @ self.inputs.T
        return torch.cat(records, dim=1).detach()

    def initial(self, target: ValueEnsemble, policy) -> torch.Tensor:
        paths = math.ceil(DATA_SIZE / COLLECT_DECISIONS)
        return self.collect(target, policy, paths)[:, -DATA_SIZE:].clone()

    def refill(self, replay: torch.Tensor, target: ValueEnsemble, policy) -> None:
        fresh = self.collect(target, policy, COLLECT_PATHS)
        slots = (self.pointer + torch.arange(fresh.shape[1])) % DATA_SIZE
        replay[:, slots] = fresh
        self.pointer = int((self.pointer + fresh.shape[1]) % DATA_SIZE)


def stage_b_measures(
    problem: LQRProblem, target: ValueEnsemble, evaluation: EvaluationSet, replay: torch.Tensor, policy
) -> Dict[str, np.ndarray]:
    """Torque statistics of the policy on the evaluation states, data extent and error on the data.

    The torque is the greedy one of V_bar (unclipped, then capped at the limit for max |u|)
    or the fixed linear policy's; replay is the member's data [E,N,D], replay or fixed.
    """
    members = target.members
    states = evaluation.states.expand(members, -1, -1)
    _, gradients = value_and_gradient(target, states)
    limit = problem.torque_limit
    with torch.no_grad():
        raw = gradients @ policy.input_matrix / policy.control_cost if policy.needs_gradient else policy(states, None)
        saturated = (raw.abs() >= limit).double().mean(dim=(1, 2))
        largest = raw.abs().amax(dim=(1, 2)).clamp(max=limit)
        sample = replay[:, : EVALUATION_SIZE]
        values = target(sample).squeeze(-1).double().numpy()
    points = sample.double().numpy()
    riccati, _ = riccati_solution(problem)
    truth = np.stack([quadratic_value(riccati, problem.discount_rate, p)[0] for p in points])
    return {
        "max_abs_torque": largest.double().numpy(),
        "saturated_fraction": saturated.numpy(),
        "data_max_abs_state": replay.abs().amax(dim=(1, 2)).double().numpy(),
        "data_value_rms_error": np.sqrt(np.mean((values - truth) ** 2, axis=1)),
    }


# ----------------------------------------------------------------------------- stages


STAGE_A_CASES = (("K*", 1.0, "random"), ("K*", 1.0, "fit"), ("0.5K*", 0.5, "random"))


def stage_a_specs(
    problem: LQRProblem,
    steps: Sequence[float],
    polyak_rates: Sequence[float],
    seeds: Sequence[int],
    budgets: Mapping[Tuple[str, str], int],
    evaluate_every: int,
) -> List[RunSpec]:
    """One spec per (policy, initialisation, rule) of STAGE_A_CASES with a positive budget."""
    _, optimal = riccati_solution(problem)
    specs = []
    for label, scale, initialisation in STAGE_A_CASES:
        budget = int(budgets.get((label, initialisation), 0))
        if budget <= 0:
            continue
        for rule in RULES:
            members = tuple(stage_a_members(rule, steps, polyak_rates, seeds))
            specs.append(
                RunSpec("A", label, rule, members, budget, evaluate_every, scale * optimal, initialisation)
            )
    return specs


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], *, append: bool) -> None:
    if not rows:
        return
    exists = path.exists() and append
    fields = list(rows[0].keys())
    if exists:
        with path.open("r", newline="") as handle:
            fields = next(csv.reader(handle))
    with path.open("a" if exists else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = []
    for row in rows:
        parsed: Dict[str, object] = {}
        for key, value in row.items():
            if key in ("stage", "task", "policy", "init", "rule"):
                parsed[key] = value
            else:
                parsed[key] = float(value) if value not in ("", None) else float("nan")
        numeric.append(parsed)
    return numeric


def problem_settings(
    problem: LQRProblem, box: np.ndarray, settings: Mapping[str, object]
) -> Dict[str, object]:
    """The settings and closed-form facts summary.md quotes."""
    riccati, optimal = riccati_solution(problem)
    info: Dict[str, object] = dict(settings)
    info.update(
        task=problem.task,
        discount_rate=problem.discount_rate,
        control_cost=problem.control_cost,
        torque_limit=problem.torque_limit,
        reference_period=problem.reference_period,
        explicit_polyak_rate=problem.explicit_polyak_rate,
        repo_resolvent_step=problem.resolvent_step,
        repo_inner_steps=problem.inner_steps,
        learning_rate=problem.learning_rate,
        gradient_clip=problem.gradient_clip,
        batch_size=problem.batch_size,
        hidden_sizes=list(problem.network["hidden_sizes"]),
        velocity_feature_scale=float(problem.network["velocity_feature_scale"]),
        box_half_widths=[float(value) for value in box],
        optimal_gain=optimal.tolist(),
        open_loop_eigenvalues=[str(np.round(v, 4)) for v in np.linalg.eigvals(problem.state_matrix)],
        optimal_closed_loop_eigenvalues=[
            str(np.round(v, 4)) for v in np.linalg.eigvals(closed_loop(problem, optimal))
        ],
        half_gain_closed_loop_eigenvalues=[
            str(np.round(v, 4)) for v in np.linalg.eigvals(closed_loop(problem, 0.5 * optimal))
        ],
        max_optimal_torque_on_box=float(np.max(np.abs(optimal) @ box)),
        optimal_value_matrix=riccati.tolist(),
    )
    return info


# ----------------------------------------------------------------------------- summary


def _final_rows(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple, List[Mapping[str, object]]]:
    """The last row of each run, grouped by (policy, init, rule, step, polyak_rate, steps_per_update)."""
    last: Dict[Tuple, Mapping[str, object]] = {}
    for row in rows:
        key = _run_key(row)
        if key not in last or row["grad_steps"] >= last[key]["grad_steps"]:
            last[key] = row
    grouped: Dict[Tuple, List[Mapping[str, object]]] = {}
    for key, row in last.items():
        grouped.setdefault(key[:6], []).append(row)
    return grouped


def _run_key(row: Mapping[str, object]) -> Tuple:
    return (
        row["policy"], row["init"], row["rule"], row["step"], row["polyak_rate"], row["steps_per_update"], row["seed"]
    )


def _run_series(rows: Sequence[Mapping[str, object]], key: Tuple) -> List[Mapping[str, object]]:
    return sorted((r for r in rows if _run_key(r) == key), key=lambda r: r["grad_steps"])


def _last_finite(series: Sequence[Mapping[str, object]], field: str) -> float:
    values = [r[field] for r in series if math.isfinite(r[field])]
    return values[-1] if values else float("nan")


def growth_statistics(series: Sequence[Mapping[str, object]], field: str = "gradient_rms_error") -> Dict[str, float]:
    """Observed per-update factor (log-linear fit over the second half of the finite record),
    and the ratio of the last finite error to the smallest one."""
    finite = [r for r in series if math.isfinite(r[field]) and r[field] > 0]
    if len(finite) < 4:
        return {"observed_factor": float("nan"), "final_over_min": float("nan")}
    half = finite[len(finite) // 2:]
    updates = np.array([r["outer_updates"] for r in half], dtype=float)
    logs = np.log([r[field] for r in half])
    slope = np.polyfit(updates, logs, 1)[0] if np.ptp(updates) > 0 else float("nan")
    errors = np.array([r[field] for r in finite])
    return {"observed_factor": float(np.exp(slope)), "final_over_min": float(errors[-1] / errors.min())}


GROWTH_FACTOR = 10.0


def grew(series: Sequence[Mapping[str, object]], field: str = "gradient_rms_error") -> bool:
    """Whether the error rose above GROWTH_FACTOR times its smallest earlier value, or the run stopped."""
    smallest = float("inf")
    for row in series:
        value = row[field]
        if not math.isfinite(value):
            continue
        if value > GROWTH_FACTOR * smallest:
            return True
        smallest = min(smallest, value)
    return any(row["diverged"] for row in series)


def _power(base: float, exponent: int) -> float:
    try:
        return float(base) ** int(exponent)
    except OverflowError:
        return float("inf")


def _mean_sd(values: Sequence[float]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "--"
    if len(finite) == 1:
        return f"{finite[0]:.3g}"
    return f"{np.mean(finite):.3g} ± {np.std(finite, ddof=1):.2g}"


def stage_a_table(rows: Sequence[Mapping[str, object]], policy: str, initialisation: str) -> List[str]:
    lines = [
        "| rule | T (s) | eta | M or K | updates | k_c (1/s) | rho_quad | rho_quad^U | diverged | grew >10x | "
        "final RMS V err | final RMS grad err | final max abs grad V | obs. factor/update | final/min grad err | "
        "kappa_eff start -> end | line k.b (centroid, 90%) end |",
        "|" + "---|" * 17,
    ]
    grouped = _final_rows([r for r in rows if r["policy"] == policy and r["init"] == initialisation])
    for key in sorted(grouped, key=lambda k: (RULES.index(k[2]), k[3], -k[4])):
        finals = grouped[key]
        first = finals[0]
        series = [_run_series(rows, key + (r["seed"],)) for r in finals]
        statistics = [growth_statistics(s) for s in series]
        diverged = [r for r in finals if r["diverged"]]
        divergence = f"{len(diverged)}/{len(finals)}"
        if diverged:
            divergence += f" (first at {int(min(r['diverged_at_step'] for r in diverged))})"
        starts = [s[0]["kappa_eff"] for s in series]
        ends = [_last_finite(s, "kappa_eff") for s in series]
        threshold = first["predicted_threshold"]
        radius = first["quadratic_radius"]
        total_updates = max(int(s[-1]["outer_updates"]) for s in series)
        lines.append(
            "| {rule} | {step:g} | {eta} | {m} | {u} | {kc} | {rho:.5f} | {rhou:.3g} | {div} | {grew} | {v} | {g} | {gm} | "
            "{of} | {fm} | {ks} -> {ke} | {kd}, {k9} |".format(
                rule=key[2],
                step=key[3],
                eta="--" if key[2] == "implicit" else f"{key[4]:g}",
                m=int(key[5]),
                u=total_updates,
                kc="--" if not math.isfinite(threshold) else f"{threshold:.3g}",
                rho=radius,
                rhou=_power(radius, total_updates),
                div=divergence,
                grew=f"{sum(grew(s) for s in series)}/{len(series)}",
                v=_mean_sd([r["value_rms_error"] for r in finals]),
                g=_mean_sd([r["gradient_rms_error"] for r in finals]),
                gm=_mean_sd([r["max_gradient_norm"] for r in finals]),
                of=_mean_sd([s["observed_factor"] for s in statistics]),
                fm=_mean_sd([s["final_over_min"] for s in statistics]),
                ks=_mean_sd(starts),
                ke=_mean_sd(ends),
                kd=_mean_sd([_last_finite(s, "line_kb_centroid") for s in series]),
                k9=_mean_sd([_last_finite(s, "line_kb_90") for s in series]),
            )
        )
    return lines


POLICY_NAMES = {"greedy": "greedy, replay data"}
CONTROL_ORDER = ("greedy", "greedy, box data", "K*, replay data", "K*, frozen replay data", "K*, box data")


def _policy_order(label: str) -> int:
    return CONTROL_ORDER.index(label) if label in CONTROL_ORDER else len(CONTROL_ORDER)


def stage_b_table(rows: Sequence[Mapping[str, object]]) -> List[str]:
    lines = [
        "| policy, data | rule | T (s) | eta | M or K | Adam steps | diverged | grew >10x (grad V*, own-data V) | final RMS V err vs V* | final RMS grad err vs V* | "
        "min RMS grad err | final max abs u (N) | final saturated fraction | max over run of saturated fraction | "
        "final data max abs x | min RMS V err on own data | final RMS V err on own data | max kappa_eff |",
        "|" + "---|" * 18,
    ]
    grouped = _final_rows(rows)
    for key in sorted(grouped, key=lambda k: (_policy_order(k[0]), RULES.index(k[2]), k[3])):
        finals = grouped[key]
        series = [_run_series(rows, key + (r["seed"],)) for r in finals]
        diverged = [r for r in finals if r["diverged"]]
        divergence = f"{len(diverged)}/{len(finals)}"
        if diverged:
            divergence += f" (first at {int(min(r['diverged_at_step'] for r in diverged))})"
        lines.append(
            "| {policy} | {rule} | {step:g} | {eta} | {m} | {n} | {div} | {grew} | {v} | {g} | {gmin} | {u} | {s} | {smax} | {x} | "
            "{dmin} | {dv} | {kmax} |".format(
                policy=POLICY_NAMES.get(key[0], key[0]),
                rule=key[2],
                step=key[3],
                eta="--" if key[2] == "implicit" else f"{key[4]:g}",
                m=int(key[5]),
                n=int(max(s[-1]["grad_steps"] for s in series)),
                div=divergence,
                grew=f"{sum(grew(s) for s in series)}/{len(series)}, "
                     f"{sum(grew(s, 'data_value_rms_error') for s in series)}/{len(series)}",
                v=_mean_sd([_last_finite(s, "value_rms_error") for s in series]),
                g=_mean_sd([_last_finite(s, "gradient_rms_error") for s in series]),
                gmin=_mean_sd([min((r["gradient_rms_error"] for r in s if math.isfinite(r["gradient_rms_error"])), default=float("nan")) for s in series]),
                u=_mean_sd([_last_finite(s, "max_abs_torque") for s in series]),
                s=_mean_sd([_last_finite(s, "saturated_fraction") for s in series]),
                smax=_mean_sd([max((r["saturated_fraction"] for r in s if math.isfinite(r["saturated_fraction"])), default=float("nan")) for s in series]),
                x=_mean_sd([_last_finite(s, "data_max_abs_state") for s in series]),
                dv=_mean_sd([_last_finite(s, "data_value_rms_error") for s in series]),
                dmin=_mean_sd([min((r["data_value_rms_error"] for r in s if math.isfinite(r["data_value_rms_error"])),
                                   default=float("nan")) for s in series]),
                kmax=_mean_sd([max((r["kappa_eff"] for r in s if math.isfinite(r["kappa_eff"])), default=float("nan"))
                               for s in series]),
            )
        )
    return lines


def write_summary(out: Path, rows_a, rows_b, info: Mapping[str, object]) -> Path:
    lines = [
        f"# Value-update stability on {info.get('task')} (evaluations/ph_sac/value_update_stability.py)",
        "",
        "Generated from results_stageA.csv and results_stageB.csv; every number below is computed by the script.",
        "",
        "## Settings",
        "",
    ]
    for key, value in info.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "Columns. k_c = sqrt(lambda (2/(eta T) - lambda)), the frozen-coefficient threshold on |k.b| above which the "
        "generator's error modes grow (eq. thresholds); rho_quad = exact per-update factor of the rule on the quadratic "
        "functions c + x^T E x of the linear closed loop (exact regressions assumed), rho_quad^U its product over the "
        "run's U updates; 'obs. factor/update' = exp(slope) of a log-linear fit of the target head's RMS gradient error "
        "against updates over the second half of its finite record; 'grew >10x' = runs whose RMS gradient error (Stage B: "
        "also the value error on the run's own data) rose above 10 times its smallest earlier value, or that stopped; "
        "'final/min grad err' = last finite RMS gradient "
        "error over its smallest value in the run (>1 means it grew after its minimum); kappa_eff = "
        "RMS(b . grad e)/RMS(e - mean e) on the evaluation batch (APPROXIMATE: the energy-weighted |k.b| of the error "
        "e = V_bar - V^pi); line k.b = power-weighted mean and 90%-power wavenumber of the error along one line through the box "
        "in the direction of b at a typical state, quadratic trend removed, times |b| there (APPROXIMATE, frozen "
        "coefficients at one state). Errors are those of the target head V_bar (the iterate) on the evaluation batch; "
        "mean ± sd over the seeds that did not diverge; diverged = runs stopped (non-finite loss or error above 1e3 "
        "times its step-0 value) with the first Adam step at which one stopped.",
        "",
    ]
    starts = {"random": "SmoothValueHead initialisation", "fit": "a supervised fit of V^pi"}
    for policy, _, initialisation in STAGE_A_CASES:
        if any(r["policy"] == policy and r["init"] == initialisation for r in rows_a):
            lines += [
                f"## Stage A, policy u = {policy} x, start: {starts[initialisation]}, errors against V^pi",
                "",
            ]
            lines += stage_a_table(rows_a, policy, initialisation)
            lines.append("")
    if rows_b:
        lines += [
            "## Stage B, greedy policy from grad V_bar (and the generator controls), errors against V*",
            "",
            "Controls: the generator at the same (T, eta) with the policy fixed at K* and/or the data fixed to "
            "the Stage A box batch, or to one replay batch collected under K* and never refreshed ('frozen "
            "replay'); 'max abs u' is then the fixed policy's torque. Under K*, V^pi = V*. 'RMS V err on replay' "
            "is measured on the member's own data.",
            "",
        ]
        lines += stage_b_table(rows_b)
        lines.append("")
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------------------- figures


COLORS = {"generator": "#2a78d6", "implicit": "#eb6834", "rollout": "#1baf7a"}
MARKERS = {"generator": "o", "implicit": "s", "rollout": "D"}
LABELS = {
    "generator": "generator (explicit Euler)",
    "implicit": "implicit (resolvent)",
    "rollout": "rollout (interval backup)",
}
INK, SECONDARY, MUTED, GRIDLINE, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.0, "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY,
        "axes.linewidth": 0.7, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY, "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True,
        "axes.grid.axis": "y", "grid.color": GRIDLINE, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "axes.titlesize": 8.0, "axes.titlecolor": INK,
    })


def _log_ticks(axis) -> None:
    """Decade ticks only, labelled 10^n; no minor labels."""
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _plot_runs(axis, rows, field, color, marker, linestyle="-") -> None:
    """Every seed as a thin line, the median over the seeds still running as a thick one."""
    seeds = sorted({r["seed"] for r in rows})
    curves = {}
    for seed in seeds:
        series = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["grad_steps"])
        steps = np.array([r["grad_steps"] for r in series]) / 1000.0
        values = np.array([r[field] for r in series], dtype=float)
        curves[seed] = (steps, values)
        axis.plot(steps, values, color=color, linewidth=0.6, alpha=0.45, linestyle=linestyle)
        finite = np.isfinite(values)
        if any(r["diverged"] for r in series) and finite.any():
            last = np.flatnonzero(finite)[-1]
            axis.plot(steps[last], values[last], marker="x", color=color, markersize=6, markeredgewidth=1.4)
    if curves:
        length = min(len(c[0]) for c in curves.values())
        steps = next(iter(curves.values()))[0][:length]
        stacked = np.stack([c[1][:length] for c in curves.values()])
        alive = np.all(np.isfinite(stacked), axis=0)
        median = np.where(alive, np.median(stacked, axis=0), np.nan)
        axis.plot(steps, median, color=color, linewidth=1.6, linestyle=linestyle,
                  marker=marker, markevery=max(1, length // 6), markersize=3.5,
                  markeredgecolor="#fcfcfb", markeredgewidth=0.5)


def plot_stage_a(rows, policy: str, initialisation: str, stems: Sequence[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    rows = [r for r in rows if r["policy"] == policy and r["init"] == initialisation]
    steps = sorted({r["step"] for r in rows})
    rates = sorted({r["polyak_rate"] for r in rows if r["rule"] != "implicit"}, reverse=True)
    panels = [(field, rate) for rate in rates for field in ("value_rms_error", "gradient_rms_error")]
    figure, axes = plt.subplots(len(panels), len(steps), figsize=(6.6, 1.6 * len(panels) + 0.7), sharex=True,
                                sharey="row", squeeze=False)
    names = {"value_rms_error": r"RMS $|\bar V-V^\pi|$", "gradient_rms_error": r"RMS $|\nabla\bar V-\nabla V^\pi|$"}
    for row_index, (field, rate) in enumerate(panels):
        for column, step in enumerate(steps):
            axis = axes[row_index][column]
            axis.set_yscale("log")
            for rule in RULES:
                selected = [
                    r for r in rows
                    if r["rule"] == rule and r["step"] == step and (rule == "implicit" or r["polyak_rate"] == rate)
                ]
                if selected:
                    _plot_runs(axis, selected, field, COLORS[rule], MARKERS[rule])
            _log_ticks(axis)
            generator = [r for r in rows if r["rule"] == "generator" and r["step"] == step and r["polyak_rate"] == rate]
            if generator and field == "value_rms_error":
                threshold = generator[0]["predicted_threshold"]
                radius = generator[0]["quadratic_radius"]
                axis.set_title(f"T = {step:g} s, $\\eta$ = {rate:g}\ngenerator: $k_c$ = {threshold:.3g}/s, "
                               f"$\\rho_q$ = {radius:.4f}", fontsize=7.0)
            if column == 0:
                axis.set_ylabel(f"{names[field]}\n$\\eta$ = {rate:g}", fontsize=7.2)
            if row_index == len(panels) - 1:
                axis.set_xlabel("Adam steps (thousands)")
    handles = [
        matplotlib.lines.Line2D([], [], color=COLORS[rule], marker=MARKERS[rule], markersize=4, label=LABELS[rule])
        for rule in RULES
    ]
    handles.append(matplotlib.lines.Line2D([], [], color=MUTED, marker="x", linestyle="none", label="run stopped (diverged)"))
    figure.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
                  handlelength=1.8)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".pdf"))
        figure.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(figure)


def plot_stage_b(rows, stems: Sequence[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    configurations = sorted({(r["rule"], r["step"]) for r in rows}, key=lambda k: (RULES.index(k[0]), k[1]))
    long_step = max(step for _, step in configurations) if configurations else 0.0
    fields = (("value_rms_error", r"RMS $|\bar V-V^*|$", True),
              ("gradient_rms_error", r"RMS $|\nabla\bar V-\nabla V^*|$", True),
              ("max_abs_torque", r"max $|u|$ of the greedy torque (N)", False))
    figure, axes = plt.subplots(1, len(fields), figsize=(6.6, 2.4), squeeze=False)
    handles = []
    drawing = sorted(configurations, key=lambda k: -RULES.index(k[0]))
    for rule, step in drawing:
        selected = [r for r in rows if r["rule"] == rule and r["step"] == step]
        linestyle = "--" if (rule == "implicit" and step == long_step) else "-"
        for axis, (field, _, logarithmic) in zip(axes[0], fields):
            _plot_runs(axis, selected, field, COLORS[rule], MARKERS[rule], linestyle)
    for rule, step in configurations:
        selected = [r for r in rows if r["rule"] == rule and r["step"] == step]
        linestyle = "--" if (rule == "implicit" and step == long_step) else "-"
        first = selected[0]
        label = f"{rule}, T = {step:g} s" + (
            f", $\\eta$ = {first['polyak_rate']:g}" if rule != "implicit" else f", K = {int(first['steps_per_update'])}"
        )
        handles.append(matplotlib.lines.Line2D([], [], color=COLORS[rule], marker=MARKERS[rule], markersize=4,
                                               linestyle=linestyle, label=label))
    for axis, (_, name, logarithmic) in zip(axes[0], fields):
        if logarithmic:
            axis.set_yscale("log")
            _log_ticks(axis)
        axis.set_title(name)
        axis.set_xlabel("Adam steps (thousands)")
    figure.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".pdf"))
        figure.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(figure)


def plot_stage_b_controls(rows, stems: Sequence[Path]) -> None:
    """The generator at Stage B's (T, eta) under the four policy x data combinations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    palette = {"greedy": "#2a78d6", "greedy, box data": "#eb6834", "K*, replay data": "#1baf7a",
               "K*, frozen replay data": "#e87ba4", "K*, box data": "#eda100"}
    markers = {"greedy": "o", "greedy, box data": "s", "K*, replay data": "D", "K*, frozen replay data": "v",
               "K*, box data": "^"}
    fields = (("value_rms_error", r"RMS $|\bar V-V^*|$", True),
              ("gradient_rms_error", r"RMS $|\nabla\bar V-\nabla V^*|$", True),
              ("max_abs_torque", r"max $|u|$ on the evaluation states (N)", False))
    figure, axes = plt.subplots(1, len(fields), figsize=(6.6, 2.4), squeeze=False)
    handles = []
    for label in CONTROL_ORDER:
        selected = [r for r in rows if r["policy"] == label]
        if not selected:
            continue
        for axis, (field, _, _) in zip(axes[0], fields):
            _plot_runs(axis, selected, field, palette[label], markers[label])
        handles.append(matplotlib.lines.Line2D([], [], color=palette[label], marker=markers[label], markersize=4,
                                               label=POLICY_NAMES.get(label, label)))
    for axis, (_, name, logarithmic) in zip(axes[0], fields):
        if logarithmic:
            axis.set_yscale("log")
            _log_ticks(axis)
        axis.set_title(name)
        axis.set_xlabel("Adam steps (thousands)")
    first = rows[0]
    figure.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.0,
                  title=f"generator, T = {first['step']:g} s, $\\eta$ = {first['polyak_rate']:g}: policy, data",
                  title_fontsize=7.5)
    figure.tight_layout(rect=(0, 0, 1, 0.74))
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".pdf"))
        figure.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(figure)


def make_outputs(out: Path, figure_directory: Path) -> None:
    """summary.md and the figures (PDF and PNG, in out and in figure_directory) from the CSVs."""
    rows_a = read_rows(out / "results_stageA.csv")
    rows_b = read_rows(out / "results_stageB.csv")
    info_path = out / "settings.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    write_summary(out, rows_a, rows_b, info)
    task = info.get("task", "lqr")
    names = {"K*": "optimal", "0.5K*": "half_gain"}
    for policy, _, initialisation in STAGE_A_CASES:
        if any(r["policy"] == policy and r["init"] == initialisation for r in rows_a):
            name = f"value_update_stability_{task}_stageA_{names[policy]}_{initialisation}"
            plot_stage_a(rows_a, policy, initialisation, [out / name, figure_directory / name])
    main_b = [r for r in rows_b if r["policy"] == "greedy"]
    if main_b:
        name = f"value_update_stability_{task}_stageB"
        plot_stage_b(main_b, [out / name, figure_directory / name])
    controls = [r for r in rows_b if r["rule"] == "generator"]
    if len({r["policy"] for r in controls}) > 1:
        name = f"value_update_stability_{task}_stageB_controls"
        plot_stage_b_controls(controls, [out / name, figure_directory / name])


# ----------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="lqr_2_1", choices=("lqr_2_1", "lqr_6_2"))
    parser.add_argument("--stage", default="both", choices=("A", "B", "both"))
    parser.add_argument("--out", default="out/generator_stability")
    parser.add_argument("--figure-dir", default=str(FIGURE_DIRECTORY))
    parser.add_argument("--quick", action="store_true", help="smoke run: one seed, a few hundred steps")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--stage-a-steps", type=int, default=STAGE_A_STEPS,
                        help="Adam steps per run, K* policy, random start")
    parser.add_argument("--stage-a-fit-steps", type=int, default=STAGE_A_FIT_STEPS,
                        help="Adam steps per run, K* policy, start at the fit of V^pi (0 skips)")
    parser.add_argument("--stage-a-half-gain-steps", type=int, default=STAGE_A_HALF_GAIN_STEPS,
                        help="Adam steps per run, 0.5 K* policy, random start (0 skips)")
    parser.add_argument("--pretrain-steps", type=int, default=PRETRAIN_STEPS)
    parser.add_argument("--stage-b-steps", type=int, default=STAGE_B_STEPS,
                        help="Adam steps per generator and rollout run in Stage B")
    parser.add_argument("--stage-b-implicit-steps", type=int, default=STAGE_B_IMPLICIT_STEPS)
    parser.add_argument("--stage-b-control-steps", type=int, default=STAGE_B_CONTROL_STEPS,
                        help="Adam steps per generator control run (fixed policy and/or fixed data); 0 skips")
    parser.add_argument("--controls", nargs="*", default=None,
                        help="labels of the controls to run (default all), e.g. 'K*, box data'")
    parser.add_argument("--controls-only", action="store_true",
                        help="with --stage B: run only the controls, appending to results_stageB.csv")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true", help="redraw figures and summary from the CSVs")
    arguments = parser.parse_args(argv)
    torch.set_num_threads(arguments.threads)
    out = Path(arguments.out)
    if not out.is_absolute():
        out = REPOSITORY_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    figure_directory = Path(arguments.figure_dir)
    if arguments.plot_only:
        make_outputs(out, figure_directory)
        return 0
    quick = arguments.quick
    problem = load_problem(arguments.task)
    seeds = list(range(1 if quick else arguments.seeds))
    budgets = {
        ("K*", "random"): 480 if quick else arguments.stage_a_steps,
        ("K*", "fit"): 320 if quick else arguments.stage_a_fit_steps,
        ("0.5K*", "random"): 240 if quick else arguments.stage_a_half_gain_steps,
    }
    pretrain_steps = 200 if quick else arguments.pretrain_steps
    stage_b_budgets = {
        "generator": 1200 if quick else arguments.stage_b_steps,
        "rollout": 1200 if quick else arguments.stage_b_steps,
        "implicit": 800 if quick else arguments.stage_b_implicit_steps,
        "generator controls": 600 if quick else arguments.stage_b_control_steps,
    }
    steps = (problem.reference_period, 0.1, problem.resolvent_step)
    polyak_rates = (1.0, problem.explicit_polyak_rate)
    riccati, optimal = riccati_solution(problem)
    box = state_box(problem, [optimal, 0.5 * optimal])
    settings: Dict[str, object] = {
        "steps_T": list(steps),
        "polyak_rates": list(polyak_rates),
        "update_steps": UPDATE_STEPS,
        "seeds": seeds,
        "stage_a_adam_steps": {f"{policy} {start}": value for (policy, start), value in budgets.items()},
        "pretrain_adam_steps": pretrain_steps,
        "stage_b_adam_steps": stage_b_budgets,
        "data_size": DATA_SIZE,
        "evaluation_size": EVALUATION_SIZE,
        "stage_b_collection": f"{COLLECT_PATHS} paths x {COLLECT_DECISIONS} decisions every {COLLECT_EVERY} "
                              f"Adam steps, exploration std {EXPLORATION_STD} N, FIFO {DATA_SIZE}",
        "threads": arguments.threads,
    }
    timings: Dict[str, float] = {}
    if arguments.stage in ("A", "both"):
        path = out / "results_stageA.csv"
        if path.exists():
            path.unlink()
        data = torch.stack(
            [uniform_box_states(box, DATA_SIZE, torch.Generator().manual_seed(1000 + seed)) for seed in seeds]
        )
        fitted: Dict[str, ValueEnsemble] = {}
        for spec in stage_a_specs(problem, steps, polyak_rates, seeds, budgets, STAGE_A_EVALUATE_EVERY):
            gain = spec.reference_gain
            value_matrix = policy_value_matrix(problem, gain)
            initial = None
            if spec.initialisation == "fit":
                if spec.policy_label not in fitted:
                    started = time.perf_counter()
                    fitted[spec.policy_label] = pretrain_values(problem, seeds, data, value_matrix, pretrain_steps)
                    timings[f"A {spec.policy_label} pretrain"] = time.perf_counter() - started
                initial = fitted[spec.policy_label]
            evaluation = evaluation_set(problem, value_matrix, gain, box, seed=999)
            policy = LinearPolicy(torch.as_tensor(gain, dtype=torch.get_default_dtype()))
            started = time.perf_counter()
            rows = run_ensemble(problem, spec, seeds, evaluation, policy=policy, data=data, initial=initial)
            timings[f"A {spec.policy_label} {spec.initialisation} {spec.rule}"] = time.perf_counter() - started
            write_rows(path, rows, append=True)
    if arguments.stage in ("B", "both"):
        path = out / "results_stageB.csv"
        if path.exists() and not arguments.controls_only:
            path.unlink()
        evaluation = evaluation_set(problem, riccati, optimal, box, seed=999)
        policy = GreedyPolicy(
            torch.as_tensor(problem.input_matrix, dtype=torch.get_default_dtype()),
            problem.control_cost,
            problem.torque_limit,
        )
        for rule in RULES if not arguments.controls_only else ():
            members = tuple(stage_b_members(rule, problem, seeds, problem.reference_period))
            spec = RunSpec("B", "greedy", rule, members, stage_b_budgets[rule], STAGE_B_EVALUATE_EVERY, optimal,
                           data_source="replay")
            started = time.perf_counter()
            rows = run_ensemble(problem, spec, seeds, evaluation, policy=policy)
            timings[f"B {rule}"] = time.perf_counter() - started
            write_rows(path, rows, append=True)
        control_steps = 600 if quick else arguments.stage_b_control_steps
        if control_steps > 0:
            box_data = torch.stack(
                [uniform_box_states(box, DATA_SIZE, torch.Generator().manual_seed(1000 + seed)) for seed in seeds]
            )
            fixed = LinearPolicy(torch.as_tensor(optimal, dtype=torch.get_default_dtype()))
            members = tuple(stage_b_members("generator", problem, seeds, problem.reference_period))
            for label, control_policy, source in (
                ("K*, box data", fixed, "fixed"),
                ("greedy, box data", policy, "fixed"),
                ("K*, replay data", fixed, "replay"),
                ("K*, frozen replay data", fixed, "frozen replay"),
            ):
                if arguments.controls and label not in arguments.controls:
                    continue
                spec = RunSpec("B", label, "generator", members, control_steps, STAGE_B_EVALUATE_EVERY, optimal,
                               data_source=source)
                started = time.perf_counter()
                rows = run_ensemble(problem, spec, seeds, evaluation, policy=control_policy, data=box_data)
                timings[f"B control {label}"] = time.perf_counter() - started
                write_rows(path, rows, append=True)
    settings["wall_time_s"] = {key: round(value, 1) for key, value in timings.items()}
    info_path = out / "settings.json"
    previous = json.loads(info_path.read_text()) if info_path.exists() else {}
    if arguments.stage != "both" and "wall_time_s" in previous:
        previous["wall_time_s"].update(settings["wall_time_s"])
        settings["wall_time_s"] = previous["wall_time_s"]
    info_path.write_text(json.dumps(problem_settings(problem, box, settings), indent=1, default=str))
    make_outputs(out, figure_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
