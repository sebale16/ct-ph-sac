"""Frozen-policy residual, unbiased regression and the fixed-step solve outcome."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Optional, TypedDict, Union

import torch

from common.ph_time import discount_weight
from models.ph_sac.operators import policy_generator_terms, soft_reward_rate

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_types import DecisionSample, SolverResult, TransitionBatch
    from environment.ph_sac.controller import Controller
    from models.ph_sac.networks import DecisionActor, SmoothValueHead, TwinQCritic
    from models.ph_sac.plant import Plant


class OptionalOperatorFields(TypedDict, total=False):
    """The critic operator's inputs (implicit config value_operator "critic", the
    implicit_critic arm): the twin critic frozen with the operator and Delta_ref (s)."""

    critics: TwinQCritic
    reference_interval: float


class FrozenOperator(OptionalOperatorFields):
    """All inputs except candidate V stay fixed until this resolvent step is solved."""

    target_value: SmoothValueHead
    policy: DecisionActor
    plant: Plant
    temperature_per_second: Tensor
    discount_rate: float
    resolvent_step: float
    reward_rate: Callable[[Tensor, Tensor], Tensor]


def resolvent_residuals(
    value: SmoothValueHead,
    operator: FrozenOperator,
    controller: Controller,
    states: Tensor,
    samples: DecisionSample,
    *,
    create_graph: bool,
) -> Tensor:
    """R_i=(1+lambda tau)V - V_k - tau(G_i V + rho_i - alpha_c log pi_i), per draw [B,N,1].

    The value derivatives come from one pass and only the bracket depends on the draw.
    With create_graph the gradient reaches psi alone: draws, plant and target are frozen.
    """
    decisions = samples["decisions"].detach()
    log_probabilities = samples["log_probabilities"].detach()
    tau = operator["resolvent_step"]
    derivatives, generated = policy_generator_terms(
        value,
        operator["plant"],
        controller,
        states,
        decisions,
        create_graph=create_graph,
    )
    with torch.no_grad():
        target = operator["target_value"](states)
        soft_rates = soft_reward_rate(
            states,
            decisions,
            log_probabilities,
            controller,
            operator["reward_rate"],
            torch.as_tensor(operator["temperature_per_second"]).detach(),
        )
    return (1.0 + operator["discount_rate"] * tau) * derivatives["values"].unsqueeze(
        1
    ) - target.unsqueeze(1) - tau * (generated + soft_rates)


def observed_interval_residuals(
    value: SmoothValueHead,
    operator: FrozenOperator,
    batch: TransitionBatch,
    samples: DecisionSample,
    *,
    create_graph: bool,
    rewards: Optional[Tensor] = None,
    stop_successor: bool = False,
) -> Tensor:
    """Contact exclusion's model-free residual at flagged replay rows, per draw [B,N,1].

    rewards [B,1] (None: the batch's interval rewards) stand for r below; under
    reward_observation "end_state" the caller passes the observed ends' trapezoid of the
    operator's reward rate plus the failure charge (ph_targets.observed_end_reward).
    stop_successor (implicit config flagged_value_gradient "stop_successor"; False, the
    default, is "full") detaches V(s') after evaluating it as below: the residuals keep
    their values and the gradient reaches psi through V(s) alone.

    The generator residual R_i = (1 + lambda tau) V - V_k - tau (G^(kappa_i) V + rho_i -
    alpha_c log pi_i) reads the plant at s. At a flagged row the observed interval
    (Delta, reward r, next state s', termination d) stands in for the generator and the
    reward: G^kappa V(s) + rho(s) ~ [r + (1-d) e^(-lambda Delta) V(s') - V(s)] / Delta
    + lambda V(s), the interval difference of the SAME candidate V (the generator acts on
    the candidate in R_i; its O(Delta) discretization), and the entropy charge is the
    interval's alpha_c w_lambda(Delta) log pi over Delta, so
        R_i = (1 + tau/Delta) V(s) - V_k(s) - (tau/Delta) [r + (1-d) e^(-lambda Delta) V(s')]
              + tau alpha_c (w_lambda(Delta)/Delta) log pi_k(kappa_i | s),
    with w_lambda(Delta) = (1 - e^(-lambda Delta))/lambda and kappa_i the operator policy's
    draws at s (the entropy term is on-policy; the reward and s' are those of the observed,
    behaviour decision). Expanding V(s') = V(s) + Delta G V + O(Delta^2) and r = Delta rho
    + O(Delta^2) gives back R_i to O(tau Delta). Its root is the resolvent step of the
    interval chain, V = [V_k + (tau/Delta)(r + gamma_Delta V(s') - alpha_Delta log pi)] /
    (1 + tau/Delta), whose contraction per outer iteration is 1/(1 + (tau/Delta)(1 -
    gamma_Delta)) = 1/(1 + lambda tau) + O(Delta), the generator rows'. The gradient
    reaches psi through V(s) and V(s') (the full residual gradient, as through G V in R_i),
    or with stop_successor through V(s) alone; draws and V_k are frozen.
    """
    tau = float(operator["resolvent_step"])
    lam = float(operator["discount_rate"])
    durations = batch["durations"].reshape(-1, 1)
    with torch.set_grad_enabled(create_graph):
        here = value(batch["states"])
        there = value(batch["next_states"])
    if stop_successor:
        # The same online value at s' (not V_k), its gradient path cut after the same
        # evaluation, so every residual keeps its value.
        there = there.detach()
    with torch.no_grad():
        anchor = operator["target_value"](batch["states"])
        alpha = torch.as_tensor(operator["temperature_per_second"]).detach()
        ratio = tau / durations.to(here.dtype)
        discount = torch.exp(-lam * durations.to(here.dtype))
        weight = discount_weight(durations.to(here.dtype), discount_rate=lam) / durations.to(
            here.dtype
        )
        rewards = (batch["rewards"] if rewards is None else rewards).reshape(-1, 1).to(
            here.dtype
        )
        live = 1.0 - batch["terminated"].reshape(-1, 1).to(here.dtype)
    base = (1.0 + ratio) * here - anchor - ratio * (rewards + live * discount * there)
    charge = (tau * alpha * weight).unsqueeze(1) * samples["log_probabilities"].detach()
    return base.unsqueeze(1) + charge


def excluded_resolvent_residuals(
    value: SmoothValueHead,
    operator: FrozenOperator,
    controller: Controller,
    batch: TransitionBatch,
    samples: DecisionSample,
    flags: Tensor,
    *,
    create_graph: bool,
    rewards: Optional[Tensor] = None,
    stop_successor: bool = False,
) -> Tensor:
    """resolvent_residuals at the unflagged rows, observed_interval_residuals at the
    flagged ones (flags [B] bool), per draw [B,N,1] in row order.

    Each part is computed on its own rows only, so no plant is evaluated at a flagged
    state and no flagged row's terms reach the backward pass of the other part. With no
    row flagged this is resolvent_residuals of the whole batch. rewards [B,1] (None: the
    batch's interval rewards) are observed_interval_residuals' r, selected to the flagged
    rows here; stop_successor is its argument (the flagged rows only).
    """
    states = batch["states"]
    if not bool(flags.any()):
        return resolvent_residuals(
            value, operator, controller, states, samples, create_graph=create_graph
        )
    kept = torch.nonzero(~flags).squeeze(-1)
    flagged = torch.nonzero(flags).squeeze(-1)

    def draws(rows: Tensor) -> DecisionSample:
        return {
            "decisions": samples["decisions"].index_select(0, rows),
            "log_probabilities": samples["log_probabilities"].index_select(0, rows),
        }

    observed = observed_interval_residuals(
        value,
        operator,
        {key: column.index_select(0, flagged) for key, column in batch.items()},
        draws(flagged),
        create_graph=create_graph,
        rewards=None if rewards is None else rewards.index_select(0, flagged),
        stop_successor=stop_successor,
    )
    if kept.numel() == 0:
        return observed
    modelled = resolvent_residuals(
        value,
        operator,
        controller,
        states.index_select(0, kept),
        draws(kept),
        create_graph=create_graph,
    )
    order = torch.argsort(torch.cat((kept, flagged)))
    return torch.cat((modelled, observed), dim=0).index_select(0, order)


# --- the value-step ablation (common.ph_config.STEP_ABLATION_DEFAULTS) -------------------


def critic_soft_rates(
    operator: FrozenOperator, states: Tensor, samples: DecisionSample, values: Tensor
) -> Tensor:
    """rate_i(W) = (Q(s, kappa_i) - W(s)) / Delta_ref - alpha_c log pi_i, per draw [B,N,1].

    The critic operator's soft rate at a value W, values = W(s) [B,1] (with or without a
    graph; the rate reaches W only through it). Q is the operator's frozen twin minimum
    at the draws, alpha_c log pi_i the implicit residual's per-second charge. A critic
    fitted to r + V + Delta (G V - lambda V) over rows of every Delta of the clock holds
    Q - V ~ E[Delta] (rho + G V - lambda V) to first order in Delta when Delta is drawn
    independently of (s, kappa), so Delta_ref is the clock's mean interval by default;
    the O(E[Delta^2] / E[Delta]) remainder of the observed reward and the shortened rows
    at the episode end are the approximation the implicit_critic arm carries.
    """
    decisions = samples["decisions"].detach()
    log_probabilities = samples["log_probabilities"].detach()
    with torch.no_grad():
        repeated = states.unsqueeze(1).expand(-1, decisions.shape[1], -1)
        scores = operator["critics"].minimum(repeated, decisions)
        alpha = torch.as_tensor(operator["temperature_per_second"]).detach()
    reference = float(operator["reference_interval"])
    return (scores - values.unsqueeze(1)) / reference - alpha * log_probabilities


def critic_resolvent_residuals(
    value: SmoothValueHead,
    operator: FrozenOperator,
    states: Tensor,
    samples: DecisionSample,
    *,
    create_graph: bool,
) -> Tensor:
    """The resolvent residual with the critic operator, per draw [B,N,1]:
        R_i = V - V_k - tau [(Q_i - V) / Delta_ref - alpha_c log pi_i]
            = (1 + tau / Delta_ref) V - V_k - (tau / Delta_ref) Q_i + tau alpha_c log pi_i,
    the candidate V inside the rate as in resolvent_residuals. Its root over the policy
    average is V = [V_k + c (E_pi Q - Delta_ref alpha_c E_pi log pi)] / (1 + c), c = tau /
    Delta_ref. Q is frozen and holds G at the value it was fitted to (V_k), so the step is
    implicit only in the -V / Delta_ref part: with Q = V_k + Delta_ref q(V_k) it moves V_k by
    tau Delta_ref / (tau + Delta_ref) times the policy-averaged soft rate at V_k, an explicit
    Euler step a little shorter than Delta_ref. With create_graph the gradient reaches psi
    through V alone; critic, draws and V_k are frozen.
    """
    tau = float(operator["resolvent_step"])
    with torch.set_grad_enabled(create_graph):
        values = value(states)
    with torch.no_grad():
        anchor = operator["target_value"](states)
    rates = critic_soft_rates(operator, states, samples, values)
    return (values - anchor).unsqueeze(1) - tau * rates


def operator_anchor_rates(
    operator: FrozenOperator,
    controller: Controller,
    batch: TransitionBatch,
    samples: DecisionSample,
    flags: Optional[Tensor] = None,
    *,
    rewards: Optional[Tensor] = None,
) -> Tensor:
    """rate_i(V_k), the value step's operator at the frozen value V_k, per draw [B,N,1],
    detached (the explicit step's input).

    With a critic in the operator: critic_soft_rates at V_k. Otherwise the model rate
    rho_i + G^(kappa_i) V_k - lambda V_k - alpha_c log pi_i, read off the resolvent residual
    at V = V_k, R_i(V_k) = (1 + lambda tau) V_k - V_k - tau (G_i V_k + rho_i - alpha_c log
    pi_i) = -tau rate_i(V_k): the operator the implicit step solves, evaluated at the old
    value instead of the new one. flags ([B] bool, contact exclusion) give the flagged rows
    excluded_resolvent_residuals' observed-interval rate, [r + (1-d) e^(-lambda Delta)
    V_k(s') - V_k(s)] / Delta - alpha_c (w_lambda(Delta) / Delta) log pi_i, by the same
    identity; rewards [B,1] (None: the batch's interval rewards) stand for r there.
    """
    states = batch["states"]
    if "critics" in operator:
        with torch.no_grad():
            anchor = operator["target_value"](states)
        return critic_soft_rates(operator, states, samples, anchor).detach()
    target = operator["target_value"]
    if flags is None:
        residuals = resolvent_residuals(
            target, operator, controller, states, samples, create_graph=False
        )
    else:
        residuals = excluded_resolvent_residuals(
            target, operator, controller, batch, samples, flags, create_graph=False,
            rewards=rewards,
        )
    return -residuals.detach() / float(operator["resolvent_step"])


def explicit_step_residuals(
    value: SmoothValueHead,
    operator: FrozenOperator,
    states: Tensor,
    anchor_rates: Tensor,
    steps: Union[float, Tensor],
    *,
    create_graph: bool,
) -> Tensor:
    """R_i = V - [V_k + h rate_i(V_k)], per draw [B,N,1]: one explicit Euler step.

    anchor_rates [B,N,1] are operator_anchor_rates; steps h is a float (the resolvent step
    tau) or [B,1,1] (each row's elapsed interval). The targets are detached, so the
    estimators regress V on the policy-averaged explicit target V_k + h E_pi[rate(V_k)]
    (all_pairs_loss's variance term carries no gradient here).
    """
    with torch.set_grad_enabled(create_graph):
        values = value(states)
    with torch.no_grad():
        anchor = operator["target_value"](states)
        targets = anchor.unsqueeze(1) + steps * anchor_rates.detach()
    return values.unsqueeze(1) - targets


def _draws(residuals: Tensor) -> Tensor:
    """Accept [B,N,1] or [B,N] residuals and return [B,N]."""
    return residuals.squeeze(-1) if residuals.dim() == 3 else residuals


def all_pairs_loss(residuals: Tensor) -> Tensor:
    """For N>=2: mean_s[mean(R)^2 - unbiased_sample_variance(R)/N]; can be negative."""
    draws = _draws(residuals)
    count = draws.shape[1]
    if count < 2:
        raise ValueError(f"The all-pairs loss needs N >= 2 draws per state, got {count}.")
    mean = draws.mean(dim=1)
    return (mean.square() - draws.var(dim=1, unbiased=True) / count).mean()


def averaged_draw_loss(residuals: Tensor) -> Tensor:
    """Ablation: square the sampled policy mean without its variance correction."""
    return _draws(residuals).mean(dim=1).square().mean()


def solver_stopping_result(*, inner_steps: int, steps_per_solve: int) -> SolverResult:
    """A solve completes after exactly K inner steps; completion certifies nothing."""
    return {"inner_steps": int(inner_steps), "completed": inner_steps >= steps_per_solve}


def solver_status(result: SolverResult) -> Literal["completed", "budget_exit"]:
    """Name how a copied solve ended: all K steps, or the interaction budget first."""
    return "completed" if result["completed"] else "budget_exit"


def moving_operator_bound(
    previous_bound: float,
    *,
    contraction: float,
    solver_error_bound: float,
    fixed_point_motion_bound: float,
) -> float:
    """Lemma moving: c*previous + epsilon + delta, conditional on supplied true bounds."""
    if not 0.0 <= contraction < 1.0:
        raise ValueError(f"The contraction must lie in [0, 1), got {contraction}.")
    for term in (previous_bound, solver_error_bound, fixed_point_motion_bound):
        if term < 0.0 or math.isnan(term):
            raise ValueError(f"Error bounds must be nonnegative, got {term}.")
    return contraction * previous_bound + solver_error_bound + fixed_point_motion_bound
