"""Differentiable generators shared by both value-update schemes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple, TypedDict

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from environment.ph_sac.controller import Controller
    from models.ph_sac.networks import SmoothValueHead
    from models.ph_sac.plant import Plant


class ValueDerivatives(TypedDict):
    """values [B,1], gradients [B,D] and diffusion_traces [B,1]."""

    values: Tensor
    gradients: Tensor
    diffusion_traces: Tensor


def compress_diffusion_factor(diffusion: Tensor) -> Tensor:
    """Use QR to preserve A with at most state_dim columns, including singular factors.

    With Sigma^T=QR, R^T R=Sigma Sigma^T. A factor already within state_dim columns
    is returned unchanged.
    """
    if diffusion.shape[-1] <= diffusion.shape[-2]:
        return diffusion
    return torch.linalg.qr(diffusion.mT, mode="reduced").R.mT


def hessian_vector_product(
    value: SmoothValueHead, states: Tensor, vectors: Tensor, *, create_graph: bool
) -> Tensor:
    """Return (Hess V) u row by row, for states and vectors of shape [B,D]."""
    with torch.enable_grad():
        points = states.detach().requires_grad_(True)
        gradient = torch.autograd.grad(
            value(points).sum(), points, create_graph=True
        )[0]
        product = torch.autograd.grad(
            (gradient * vectors).sum(), points, create_graph=create_graph
        )[0]
    return product if create_graph else product.detach()


def value_derivatives(
    value: SmoothValueHead,
    states: Tensor,
    diffusion: Tensor,
    *,
    create_graph: bool,
) -> ValueDerivatives:
    """Evaluate V, grad V, and Tr(Sigma Sigma^T Hess V) using HVPs."""
    factor = compress_diffusion_factor(diffusion)
    with torch.enable_grad():
        points = states.detach().requires_grad_(True)
        values = value(points)
        gradients = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        trace = torch.zeros_like(values)
        for column in factor.unbind(dim=-1):
            product = torch.autograd.grad(
                (gradients * column).sum(),
                points,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            trace = trace + (product * column).sum(dim=-1, keepdim=True)
    if create_graph:
        return {"values": values, "gradients": gradients, "diffusion_traces": trace}
    return {
        "values": values.detach(),
        "gradients": gradients.detach(),
        "diffusion_traces": trace.detach(),
    }


def generator(
    value: SmoothValueHead,
    plant: Plant,
    controller: Controller,
    states: Tensor,
    resolved_decisions: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    """Apply the continuous generator using physical A, rather than step-transported K_h A K_h^T."""
    with torch.set_grad_enabled(create_graph):
        actions = controller.action(states, resolved_decisions)
        drift = plant.drift(states, actions)
        diffusion = plant.diffusion(states)
    derivatives = value_derivatives(
        value, states, diffusion, create_graph=create_graph
    )
    return (drift * derivatives["gradients"]).sum(
        dim=-1, keepdim=True
    ) + 0.5 * derivatives["diffusion_traces"]


def _flat_decision_terms(
    controller: Controller,
    states: Tensor,
    decision_draws: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return torques [B*N,m] and matching flat states for draws resolved at their state."""
    draws = decision_draws.shape[1]
    repeated = states.unsqueeze(1).expand(-1, draws, -1)
    resolved = controller.resolve_decision(repeated, decision_draws)
    actions = controller.action(repeated, resolved)
    return actions.flatten(0, 1), repeated.flatten(0, 1)


def policy_generator_terms(
    value: SmoothValueHead,
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decision_draws: Tensor,
    *,
    create_graph: bool,
    average_plant_terms_first: bool = False,
) -> Tuple[ValueDerivatives, Tensor]:
    """One value-head pass for all draws: its derivatives and the generator values."""
    with torch.set_grad_enabled(create_graph):
        actions, flat_states = _flat_decision_terms(
            controller, states, decision_draws
        )
        drift = plant.drift(flat_states, actions).unflatten(
            0, decision_draws.shape[:2]
        )
        diffusion = plant.diffusion(states)
    if average_plant_terms_first:
        drift = drift.mean(dim=1, keepdim=True)
    derivatives = value_derivatives(
        value, states, diffusion, create_graph=create_graph
    )
    generated = (drift * derivatives["gradients"].unsqueeze(1)).sum(
        dim=-1, keepdim=True
    ) + 0.5 * derivatives["diffusion_traces"].unsqueeze(1)
    return derivatives, generated


def policy_generator(
    value: SmoothValueHead,
    plant: Plant,
    controller: Controller,
    states: Tensor,
    decision_draws: Tensor,
    *,
    create_graph: bool,
    average_plant_terms_first: bool = False,
) -> Tensor:
    """Generator at independent frozen-policy draws, per draw [B,N,1]; callers average.

    With average_plant_terms_first the drift is averaged over the draws before
    contraction with the value derivatives, and the result has shape [B,1,1].
    """
    return policy_generator_terms(
        value,
        plant,
        controller,
        states,
        decision_draws,
        create_graph=create_graph,
        average_plant_terms_first=average_plant_terms_first,
    )[1]


def soft_reward_rate(
    states: Tensor,
    decision_draws: Tensor,
    log_probabilities: Tensor,
    controller: Controller,
    reward_rate: Callable[[Tensor, Tensor], Tensor],
    temperature_per_second: Tensor,
) -> Tensor:
    """Known reward rate minus alpha_c times log pi, per draw [B,N,1]; callers average."""
    actions, flat_states = _flat_decision_terms(controller, states, decision_draws)
    rates = reward_rate(flat_states, actions)
    return rates.unflatten(0, decision_draws.shape[:2]) - (
        temperature_per_second * log_probabilities
    )
