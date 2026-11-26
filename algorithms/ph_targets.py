"""Continuation and rate target construction for both algorithms; all targets are detached.

Every target that adds a row's reward reads the interval reward batch["rewards"] by default.
Under reward_observation "end_state" (common.ph_config.REWARD_OBSERVATION_KEY) the replay
holds no interval reward, and the callers pass the reward of the interval (rewards=, and
the like) integrated from the learned rate r_hat by the helpers below, each detached:
    failure_charge          c = d e^(-lambda Delta) F(s'), F the task's payoff
    start_rate_reward       Delta r_hat(s, u_0), the left rectangle at the start state and
                            torque (the generator target: its continuation is one explicit
                            Euler step at s)
    observed_end_reward     Delta/2 [r_hat(s, u_0) + e^(-lambda Delta) r_hat(s', u_e)], the
                            trapezoid of the observed ends (the model-free targets)
    rollout_interval_reward the survivor mean of the model paths' discounted trapezoids of
                            r_hat on their own knots (the rollout targets)
with u_0 and u_e the controller's torques at s and s' for the decision resolved at s; the
callers add c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from common.ph_time import interval_temperature
from models.ph_sac.operators import policy_generator_terms
from models.ph_sac.rollouts import survivor_continuation

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import RateConfig
    from common.ph_types import TransitionBatch
    from environment.ph_sac.controller import Controller
    from models.ph_sac.losses import RateTargets
    from models.ph_sac.networks import DecisionActor, SmoothValueHead, TwinQCritic
    from models.ph_sac.plant import Plant
    from models.ph_sac.rollouts import RolloutBatch


def _column(values: Tensor, reference: Tensor) -> Tensor:
    return values.reshape(-1, 1).to(reference.dtype)


def _value_and_generator(
    plant: Plant,
    value: SmoothValueHead,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
) -> Tuple[Tensor, Tensor]:
    """V(s) and G^kappa V(s), each [B,1], from one value pass; raw decisions resolve at s."""
    derivatives, generated = policy_generator_terms(
        value, plant, controller, states, decisions.unsqueeze(1), create_graph=False
    )
    return derivatives["values"], generated.squeeze(1)


def generator_continuation(
    plant: Plant,
    value: SmoothValueHead,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    durations: Tensor,
    *,
    discount_rate: float,
) -> Tensor:
    """V(s)+Delta*(G^kappa V-lambda V), [B,1], using initial torque and each row's Delta.

    durations ([B] or [B,1]) are the rows' elapsed intervals, the Delta the critic target's
    reward r is integrated over, so r + this is one explicit Euler step of the same
    interval on every row: an irregular clock's short and long rows, and a row shortened
    at the episode end. On a fixed clock a full row has Delta = T.
    """
    values, generated = _value_and_generator(
        plant, value, controller, states, decisions
    )
    return values + _column(durations, values) * (generated - discount_rate * values)


def rollout_continuation(
    value: SmoothValueHead,
    paths: RolloutBatch,
    batch: TransitionBatch,
    *,
    discount_rate: float,
) -> Tensor:
    """Survivor mean with the observed-endpoint fallback when no model path survives."""
    return survivor_continuation(
        value, paths, batch["next_states"], batch["durations"], discount_rate
    )


@torch.no_grad()
def td_continuation(
    critics: TwinQCritic,
    actor: DecisionActor,
    next_states: Tensor,
    durations: Tensor,
    next_durations: Tensor,
    *,
    alpha_c: Tensor,
    discount_rate: float,
) -> Tensor:
    """Optional explicit TD mode; charge entropy using a fresh next-clock interval."""
    sample = actor.sample(next_states, sample_count=1)
    decisions = sample["decisions"].squeeze(1)
    log_probabilities = sample["log_probabilities"].squeeze(1)
    minimum = critics.minimum(next_states, decisions)
    charge = interval_temperature(
        alpha_c,
        _column(next_durations, minimum),
        discount_rate=discount_rate,
    )
    discount = torch.exp(-discount_rate * _column(durations, minimum))
    return discount * (minimum - charge * log_probabilities)


def interval_critic_target(
    batch: TransitionBatch, continuation: Tensor, *, rewards: Optional[Tensor] = None
) -> Tensor:
    """r+(1-terminated)*C; reward is already integrated, and truncation bootstraps.

    rewards [B,1] (None: the batch's interval rewards) stand for r.
    """
    rewards = batch["rewards"] if rewards is None else rewards
    return rewards + (1.0 - batch["terminated"]) * continuation.detach()


@torch.no_grad()
def data_rate_target(
    value: SmoothValueHead,
    batch: TransitionBatch,
    *,
    discount_rate: float,
    rewards: Optional[Tensor] = None,
) -> Tensor:
    """[r+(1-d)*exp(-lambda Delta)*V(s')-V(s)]/Delta; rewards [B,1] (None: the batch's
    interval rewards) stand for r."""
    durations = batch["durations"]
    rewards = batch["rewards"] if rewards is None else rewards
    continuation = torch.exp(-discount_rate * durations) * value(batch["next_states"])
    return (
        rewards
        + (1.0 - batch["terminated"]) * continuation
        - value(batch["states"])
    ) / durations


@torch.no_grad()
def rollout_rate_target(
    value: SmoothValueHead,
    batch: TransitionBatch,
    continuation: Tensor,
    *,
    rewards: Optional[Tensor] = None,
) -> Tensor:
    """[r+(1-d)*C_roll-V(s)]/Delta, relative to the newly copied value; rewards [B,1]
    (None: the batch's interval rewards) stand for r."""
    rewards = batch["rewards"] if rewards is None else rewards
    return (
        rewards
        + (1.0 - batch["terminated"]) * continuation
        - value(batch["states"])
    ) / batch["durations"]


def failure_charge(
    batch: TransitionBatch,
    terminal_payoff: Callable[[Tensor], Tensor],
    *,
    discount_rate: float,
) -> Tensor:
    """c = d e^(-lambda Delta) F(s'), [B,1]: the failure charge of each terminated row at
    its elapsed Delta (the environment's, in a reward of reward_observation "interval")."""
    with torch.no_grad():
        return (
            batch["terminated"]
            * torch.exp(-discount_rate * batch["durations"])
            * terminal_payoff(batch["next_states"])
        )


@torch.no_grad()
def start_rate_reward(
    controller: Controller,
    batch: TransitionBatch,
    reward_rate: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Delta r_hat(s, u_0), [B,1]: the left rectangle of the rate over each row's elapsed
    Delta at its start state and torque, without the failure charge.

    The explicit generator continuation V + Delta (G V - lambda V) is one explicit Euler
    step at s, so the rectangle at the same point and torque is the consistent first-order
    rule: y - V = Delta (r_hat + G V - lambda V), Delta times the generator rate target.
    """
    states = batch["states"]
    torques = controller.action(states, controller.resolve_decision(states, batch["decisions"]))
    return batch["durations"] * reward_rate(states, torques)


@torch.no_grad()
def observed_end_reward(
    controller: Controller,
    batch: TransitionBatch,
    reward_rate: Callable[[Tensor, Tensor], Tensor],
    *,
    discount_rate: float,
) -> Tensor:
    """Delta/2 [r_hat(s, u_0) + e^(-lambda Delta) r_hat(s', u_e)], [B,1]: the discounted
    trapezoid of the rate at the observed ends of each row, without the failure charge.

    The model-free targets (the data rate target, contact exclusion's flagged rows, the
    gate's observed return) read only the observed s and s', whose continuation carries no
    time-discretization error; the two-point trapezoid is the best rule there.
    """
    states, next_states = batch["states"], batch["next_states"]
    resolved = controller.resolve_decision(states, batch["decisions"])
    start = controller.action(states, resolved)
    end = controller.action(next_states, resolved)
    durations = batch["durations"]
    return 0.5 * durations * (
        reward_rate(states, start)
        + torch.exp(-discount_rate * durations) * reward_rate(next_states, end)
    )


@torch.no_grad()
def rollout_interval_reward(paths: RolloutBatch, observed: Tensor) -> Tensor:
    """The rows' interval rewards from model paths, [B,1] in observed's dtype, no charge.

    Each path's running_rewards R_l (stochastic_rollout with ReturnTerms of the learned
    rate: discounted trapezoids on the path's own substep knots, the environment's rule)
    over the paths that survive, sum_l 1{survived_l} R_l / sum_l 1{survived_l}, the weights
    and fallback of survivor_continuation: a path contact exclusion truncated
    (paths["truncated"]) takes observed (observed_end_reward of the row) in place of its
    own, and a row without a surviving path takes observed.
    """
    running = paths["running_rewards"].double()
    fallback = observed.double().reshape(-1, 1)
    truncated = paths.get("truncated")
    if truncated is not None:
        running = torch.where(truncated, fallback.unsqueeze(1).expand_as(running), running)
    survived = paths["survived"].double()
    survivors = survived.sum(dim=1)
    mean = (running * survived).sum(dim=1) / survivors.clamp_min(1.0)
    return torch.where(survivors > 0.0, mean, fallback).to(observed.dtype)


def generator_rate_target(
    plant: Plant,
    value: SmoothValueHead,
    controller: Controller,
    states: Tensor,
    decisions: Tensor,
    reward_rate: Callable[[Tensor, Tensor], Tensor],
    *,
    discount_rate: float,
) -> Tensor:
    """rho+G^kappa V-lambda V at model-sampled points; no iteration of this score."""
    values, generated = _value_and_generator(
        plant, value, controller, states, decisions
    )
    with torch.no_grad():
        torques = controller.action(
            states, controller.resolve_decision(states, decisions)
        )
        rates = reward_rate(states, torques)
    return rates + generated - discount_rate * values


def value_noise_square_norm(
    plant: Plant, value: SmoothValueHead, states: Tensor
) -> Tensor:
    """||Sigma^T grad V||^2, [B,1], the coefficient of the rate target's 1/Delta variance."""
    with torch.enable_grad():
        points = states.detach().requires_grad_(True)
        gradients = torch.autograd.grad(value(points).sum(), points)[0]
    with torch.no_grad():
        projected = gradients.unsqueeze(1) @ plant.diffusion(states)
    return projected.squeeze(1).square().sum(dim=-1, keepdim=True)


def data_rate_weights(
    plant: Plant,
    value: SmoothValueHead,
    states: Tensor,
    durations: Tensor,
    *,
    weight_max: float,
    noise_variance_floor: float,
) -> Tensor:
    """min(w_max,1/(||Sigma^T grad V||^2/Delta + sigma_0^2)).

    A drift-only plant's zero Sigma leaves min(w_max, 1/sigma_0^2) on every row.
    """
    square_norm = value_noise_square_norm(plant, value, states)
    variance = square_norm / _column(durations, square_norm) + noise_variance_floor
    return torch.clamp(1.0 / variance, max=weight_max)


@torch.no_grad()
def observed_continuation(
    value: SmoothValueHead, batch: TransitionBatch, *, discount_rate: float
) -> Tensor:
    """e^(-lambda Delta) V(s'), [B,1], at the observed next state: contact exclusion's
    model-free continuation (interval_critic_target masks termination)."""
    return torch.exp(-discount_rate * batch["durations"]) * value(batch["next_states"])


def generator_critic_target(
    plant: Plant,
    value: SmoothValueHead,
    controller: Controller,
    batch: TransitionBatch,
    *,
    discount_rate: float,
    flags: Optional[Tensor] = None,
    rewards: Optional[Tensor] = None,
) -> Tensor:
    """r + (1-d) [V + Delta (G^kappa V - lambda V)] at the taken decisions, [B,1], detached.

    The explicit_generator arm's critic target (ExplicitCTPHSAC.build_critic_target with
    continuation "generator"), for the implicit_critic arm: plant and value are the ones the
    caller pairs with the critic (there the admitted plant and the copied value). At rows
    flagged by contact exclusion (flags [B] bool) the observed continuation e^(-lambda
    Delta) V(s') replaces the generator one, as there. rewards [B,1] (None: the batch's
    interval rewards) stand for r (reward_observation "end_state": the caller's).
    """
    continuation = generator_continuation(
        plant,
        value,
        controller,
        batch["states"],
        batch["decisions"],
        batch["durations"],
        discount_rate=discount_rate,
    )
    if flags is not None and bool(flags.any()):
        observed = observed_continuation(value, batch, discount_rate=discount_rate)
        continuation = torch.where(flags.reshape(-1, 1), observed, continuation)
    return interval_critic_target(batch, continuation, rewards=rewards).detach()


def excluded_rate_targets(
    targets: RateTargets,
    value: SmoothValueHead,
    batch: TransitionBatch,
    model_batch: TransitionBatch,
    batch_flags: Tensor,
    model_flags: Tensor,
    *,
    config: RateConfig,
    discount_rate: float,
    model_rewards: Optional[Tensor] = None,
) -> RateTargets:
    """Contact exclusion's rate targets at flagged states, all model-free.

    At a flagged replay row (batch_flags [B]) the rollout target is the data target,
        Y_roll = Y_Delta = [r + (1-d) e^(-lambda Delta) V(s') - V(s)] / Delta,
    the observed interval in place of model paths, and the data weight is the noise-free
    weight min(w_max, 1/sigma_0^2) (the plant's diffusion is not trusted there). At a
    flagged model point (model_flags [M], a replay row of model_batch) the caller has put
    the row's observed decision in place of the policy draw, and the generator target is
    that row's data target Y_Delta. Unflagged rows keep their targets. V is the copied
    value the targets refer to. model_rewards [M,1] (None: model_batch's interval rewards)
    stand for r of model_batch's rows.
    """
    flags = batch_flags.reshape(-1, 1)
    data = targets["data"]
    floor = torch.clamp(
        torch.ones_like(targets["data_weights"]) / float(config["noise_variance_floor"]),
        max=float(config["replay_weight_max"]),
    )
    generator = targets["generator"]
    if bool(model_flags.any()):
        observed = data_rate_target(
            value, model_batch, discount_rate=discount_rate, rewards=model_rewards
        )
        generator = torch.where(model_flags.reshape(-1, 1), observed, generator)
    return {
        "data": data,
        "rollout": torch.where(flags, data, targets["rollout"]),
        "generator": generator,
        "data_weights": torch.where(flags, floor, targets["data_weights"]),
    }


def build_rate_targets(
    plant: Plant,
    value: SmoothValueHead,
    controller: Controller,
    batch: TransitionBatch,
    paths: RolloutBatch,
    model_states: Tensor,
    model_decisions: Tensor,
    reward_rate: Callable[[Tensor, Tensor], Tensor],
    *,
    config: RateConfig,
    discount_rate: float,
    data_rewards: Optional[Tensor] = None,
    rollout_rewards: Optional[Tensor] = None,
) -> RateTargets:
    """Construct detached data/rollout/model rates; compare decisions with common path noise.

    data, rollout and data_weights are [B,1] at the replay rows, generator is [M,1].
    data_rewards and rollout_rewards [B,1] (None: the batch's interval rewards) stand for r
    in the data and the rollout target (reward_observation "end_state": the caller's).
    """
    continuation = rollout_continuation(
        value, paths, batch, discount_rate=discount_rate
    )
    return {
        "data": data_rate_target(
            value, batch, discount_rate=discount_rate, rewards=data_rewards
        ),
        "rollout": rollout_rate_target(value, batch, continuation, rewards=rollout_rewards),
        "generator": generator_rate_target(
            plant,
            value,
            controller,
            model_states,
            model_decisions,
            reward_rate,
            discount_rate=discount_rate,
        ),
        "data_weights": data_rate_weights(
            plant,
            value,
            batch["states"],
            batch["durations"],
            weight_max=config["replay_weight_max"],
            noise_variance_floor=config["noise_variance_floor"],
        ),
    }
