"""Regression and policy losses; every optimizer step belongs to algorithms."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterator, TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from common.ph_types import context_dim, dynamic_state_indices
from models.ph_sac.rollouts import wrapped_endpoint_residual

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_types import DecisionSample, StateSpec, TransitionBatch
    from models.ph_sac.networks import RateHead, SmoothValueHead, TwinQCritic
    from models.ph_sac.rollouts import EndpointMoments


class RateTargets(TypedDict):
    data: Tensor
    rollout: Tensor
    generator: Tensor
    data_weights: Tensor


@contextlib.contextmanager
def _parameters_held_fixed(module: nn.Module) -> Iterator[None]:
    """Read a module with its parameters out of the graph; decisions keep gradients."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


def plant_drift_loss(
    batch: TransitionBatch, moments: EndpointMoments, state_spec: StateSpec
) -> Tensor:
    """0.5*|e|^2/Delta averaged over rows, e the wrapped endpoint residual.

    This is the Gaussian NLL with the endpoint covariance fixed at Delta*I, constants
    dropped: a drift-only plant fits its drift and nothing about noise.
    """
    residual = wrapped_endpoint_residual(
        batch["next_states"], moments["means"], state_spec
    )
    durations = batch["durations"].reshape(-1, 1).to(residual.dtype).clamp_min(1e-6)
    return (0.5 * residual.square().sum(-1, keepdim=True) / durations).mean()


def plant_likelihood_loss(
    batch: TransitionBatch, moments: EndpointMoments, state_spec: StateSpec
) -> Tensor:
    """Wrapped Gaussian NLL using physical endpoint moments and the learned model covariance floor.

    The covariance is rescaled to unit diagonal before factoring, because its position
    and velocity blocks differ by many orders of magnitude at short durations.
    Moments of a drift-only plant (moments["drift_only"], set by
    integrate_endpoint_moments) have zero covariance and are scored by
    plant_drift_loss instead, so every caller fits such a plant the same way.
    An episode-constant context (StateSpec context_dim) neither moves nor has noise, so
    its rows are left out of the Gaussian, whose covariance is singular there.
    """
    if moments.get("drift_only", False):
        return plant_drift_loss(batch, moments, state_spec)
    residual = wrapped_endpoint_residual(
        batch["next_states"], moments["means"], state_spec
    )
    covariance = moments["covariances"]
    if context_dim(state_spec):
        keep = torch.as_tensor(dynamic_state_indices(state_spec), device=residual.device)
        residual = residual[..., keep]
        covariance = covariance[..., keep, :][..., :, keep]
    covariance = 0.5 * (covariance + covariance.mT)
    scale = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    correlation = covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    factor = torch.linalg.cholesky(correlation)
    whitened = torch.linalg.solve_triangular(
        factor, (residual / scale).unsqueeze(-1), upper=False
    ).squeeze(-1)
    half_log_determinant = (
        factor.diagonal(dim1=-2, dim2=-1).log().sum(-1) + scale.log().sum(-1)
    )
    return (0.5 * whitened.square().sum(-1) + half_log_determinant).mean()


def critic_loss(
    critics: TwinQCritic, states: Tensor, decisions: Tensor, targets: Tensor
) -> Tensor:
    first, second = critics(states, decisions)
    return F.mse_loss(first, targets) + F.mse_loss(second, targets)


def soft_value_loss(
    value: SmoothValueHead, states: Tensor, soft_value_targets: Tensor
) -> Tensor:
    return F.mse_loss(value(states), soft_value_targets)


def rate_regression_loss(
    rate: RateHead,
    replay_batch: TransitionBatch,
    targets: RateTargets,
    model_states: Tensor,
    model_decisions: Tensor,
    *,
    rollout_weight: float,
    model_weight: float,
) -> Tensor:
    """Fit data and rollout targets at replay decisions, generator rates at model points."""
    replay_rates = rate(replay_batch["states"], replay_batch["decisions"])
    loss = (targets["data_weights"] * (replay_rates - targets["data"]).square()).mean()
    if rollout_weight != 0.0:
        loss = loss + rollout_weight * F.mse_loss(replay_rates, targets["rollout"])
    if model_weight != 0.0:
        model_rates = rate(model_states, model_decisions)
        loss = loss + model_weight * F.mse_loss(model_rates, targets["generator"])
    return loss


def _repeat_states(states: Tensor, decisions: Tensor) -> Tensor:
    return states.unsqueeze(1).expand(-1, decisions.shape[1], -1)


def explicit_actor_loss(
    critics: TwinQCritic,
    states: Tensor,
    policy_sample: DecisionSample,
    interval_temperature: Tensor,
) -> Tensor:
    """Read frozen critics while retaining gradients through sampled decisions."""
    decisions = policy_sample["decisions"]
    with _parameters_held_fixed(critics):
        scores = critics.minimum(_repeat_states(states, decisions), decisions)
    temperature = interval_temperature
    if temperature.dim() == 2:
        temperature = temperature.unsqueeze(1)
    return (temperature * policy_sample["log_probabilities"] - scores).mean()


def implicit_actor_loss(
    rate: RateHead,
    states: Tensor,
    policy_sample: DecisionSample,
    temperature_per_second: Tensor,
) -> Tensor:
    """Read frozen rate parameters; only the actor receives reparameterized gradients."""
    decisions = policy_sample["decisions"]
    with _parameters_held_fixed(rate):
        scores = rate(_repeat_states(states, decisions), decisions)
    return (
        temperature_per_second * policy_sample["log_probabilities"] - scores
    ).mean()


def temperature_loss(
    log_temperature: Tensor,
    log_probabilities: Tensor,
    target_entropy: float,
    *,
    interval_weights: Tensor,
) -> Tensor:
    """Tune log alpha_c using detached policy log densities and the entropy contract."""
    weights = interval_weights
    if weights.dim() == log_probabilities.dim() - 1:
        weights = weights.unsqueeze(1)
    return -(
        log_temperature * weights * (log_probabilities.detach() + target_entropy)
    ).mean()
