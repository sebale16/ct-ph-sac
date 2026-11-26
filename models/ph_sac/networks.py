"""Neural components of both algorithms; states are [...,D] and decisions [...,d].

d is the decision dimension, the length of the config's decision box: 3m for the
feedback laws, m for held torques (decision_mode "torque"). No component assumes either.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

import torch
from torch import nn

from common.ph_config import actor_log_std_init_setting

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import NetworkConfig
    from common.ph_types import DecisionSample, StateSpec

LOG_STD_INIT = -1.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
SQUASH_EPSILON = 1e-6


def _tanh_mlp(inputs: int, hidden: Sequence[int], outputs: int) -> nn.Sequential:
    sizes = (inputs, *hidden)
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
    layers.append(nn.Linear(sizes[-1], outputs))
    return nn.Sequential(*layers)


def _frozen(module: nn.Module) -> nn.Module:
    frozen = copy.deepcopy(module)
    frozen.requires_grad_(False)
    frozen.eval()
    return frozen


def _column(values: Tensor, states: Tensor) -> Tensor:
    """Accept scalar boundary callbacks [...] as well as columns [...,1]."""
    return values.unsqueeze(-1) if values.dim() == states.dim() - 1 else values


class StateFeatures(nn.Module):
    """Periodic positions as sine and cosine, other positions raw, velocities scaled.

    A position listed in the state spec's travel_position_scales is also fed raw, divided
    by its scale, after the velocities: the Acrobot elbow, whose unwrapped angle the
    reward reads but whose sine and cosine cannot tell one turn from the next.
    """

    def __init__(self, state_spec: StateSpec, config: NetworkConfig) -> None:
        super().__init__()
        positions = int(state_spec["position_dim"])
        periodic = sorted(set(state_spec["periodic_position_indices"]))
        other = [i for i in range(positions) if i not in periodic]
        self.register_buffer("_periodic", torch.as_tensor(periodic, dtype=torch.long))
        self.register_buffer("_other", torch.as_tensor(other, dtype=torch.long))
        self.position_dim = positions
        self.velocity_scale = float(config.get("velocity_feature_scale", 1.0))
        self.travel = tuple(
            (int(index), float(scale))
            for index, scale in state_spec.get("travel_position_scales", ())
        )
        self.output_dim = (
            2 * len(periodic)
            + len(other)
            + int(state_spec["velocity_dim"])
            + len(self.travel)
        )

    def forward(self, states: Tensor) -> Tensor:
        angles = states[..., self._periodic]
        features = torch.cat(
            (
                angles.sin(),
                angles.cos(),
                states[..., self._other],
                states[..., self.position_dim :] / self.velocity_scale,
            ),
            dim=-1,
        )
        if not self.travel:
            return features
        travel = [states[..., index : index + 1] / scale for index, scale in self.travel]
        return torch.cat((features, *travel), dim=-1)


class _DecisionScaling(nn.Module):
    """Map the decision box to [-1,1]; without a box, pass decisions through."""

    def __init__(self, low: Optional[Tensor], high: Optional[Tensor]) -> None:
        super().__init__()
        if (low is None) != (high is None):
            raise ValueError("Give both decision bounds or neither.")
        self.scaled = low is not None
        if self.scaled:
            self.register_buffer("low", torch.as_tensor(low).to(torch.get_default_dtype()))
            self.register_buffer("high", torch.as_tensor(high).to(torch.get_default_dtype()))

    def forward(self, decisions: Tensor) -> Tensor:
        if not self.scaled:
            return decisions
        return 2.0 * (decisions - self.low) / (self.high - self.low) - 1.0


class DecisionActor(nn.Module):
    """Tanh Gaussian decision policy with affine box map and state-free log std.

    Its log density, from sample and log_prob alike, is over the normalized box
    u = (kappa - centre) / h in [-1,1]^d, with h the box half-widths and d the box's
    dimension (3m feedback coordinates, or m torques in torque mode): the density
    every loss, target and temperature step charges. The physical-box density is
    this minus sum(log h); the constant Jacobian is left out so that the entropy
    charge and its target, -1 per coordinate, do not depend on the box's units.

    Every entry of the log standard deviation starts at LOG_STD_INIT, or at the network
    config's actor_log_std_init when given (common.ph_config.actor_log_std_init_setting).
    """

    def __init__(
        self,
        state_spec: StateSpec,
        decision_low: Tensor,
        decision_high: Tensor,
        config: NetworkConfig,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(decision_low).to(torch.get_default_dtype()).reshape(-1)
        high = torch.as_tensor(decision_high).to(torch.get_default_dtype()).reshape(-1)
        self.register_buffer("decision_low", low)
        self.register_buffer("decision_high", high)
        self.decision_dim = len(low)
        self.features = StateFeatures(state_spec, config)
        self.mean_network = _tanh_mlp(
            self.features.output_dim, config["hidden_sizes"], self.decision_dim
        )
        start = actor_log_std_init_setting(config)
        self.log_std = nn.Parameter(
            torch.full((self.decision_dim,), LOG_STD_INIT if start is None else start)
        )

    def forward(self, states: Tensor) -> Tensor:
        """Return the unconstrained Gaussian mean network output."""
        return self.mean_network(self.features(states))

    def log_standard_deviation(self) -> Tensor:
        """Return the learned state-independent vector of log standard deviations."""
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def _log_density(self, pre_tanh: Tensor, noise: Tensor, log_std: Tensor) -> Tensor:
        """Density of u = tanh(pre-tanh) on [-1,1]^d: Gaussian with the tanh Jacobian.

        No affine Jacobian: the density is over the normalized box, whatever its units.
        """
        normal = -0.5 * noise.square() - log_std - 0.5 * math.log(2.0 * math.pi)
        # log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)), stable for large |u|.
        squash = 2.0 * (
            math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh)
        )
        return (normal - squash).sum(dim=-1, keepdim=True)

    def sample(
        self,
        states: Tensor,
        *,
        sample_count: int = 1,
        noise: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> DecisionSample:
        """Return [B,N,d] decisions and their log densities over the normalized box."""
        mean = self.forward(states).unsqueeze(1)
        log_std = self.log_standard_deviation()
        shape = (mean.shape[0], sample_count, self.decision_dim)
        if deterministic:
            noise = mean.new_zeros(shape)
        elif noise is None:
            noise = torch.randn(shape, dtype=mean.dtype, device=mean.device)
        pre_tanh = mean + log_std.exp() * noise
        squashed = torch.tanh(pre_tanh)
        decisions = (
            self.decision_low
            + 0.5 * (squashed + 1.0) * (self.decision_high - self.decision_low)
        )
        return {
            "decisions": decisions,
            "log_probabilities": self._log_density(pre_tanh, noise, log_std),
        }

    def log_prob(self, states: Tensor, decisions: Tensor) -> Tensor:
        """Normalized-box log density of [...,d] decisions; the box map inverse is clamped."""
        squashed = (
            2.0
            * (decisions - self.decision_low)
            / (self.decision_high - self.decision_low)
            - 1.0
        ).clamp(-1.0 + SQUASH_EPSILON, 1.0 - SQUASH_EPSILON)
        pre_tanh = torch.atanh(squashed)
        mean = self.forward(states)
        if decisions.dim() == states.dim() + 1:
            mean = mean.unsqueeze(1)
        log_std = self.log_standard_deviation()
        noise = (pre_tanh - mean) * torch.exp(-log_std)
        return self._log_density(pre_tanh, noise, log_std)

    def frozen_copy(self) -> DecisionActor:
        return _frozen(self)


class SmoothValueHead(nn.Module):
    """Twice differentiable tanh value network, independent of the rate head."""

    def __init__(self, state_spec: StateSpec, config: NetworkConfig) -> None:
        super().__init__()
        self.features = StateFeatures(state_spec, config)
        self.network = _tanh_mlp(self.features.output_dim, config["hidden_sizes"], 1)
        self._lift: Optional[Callable[[Tensor], Tensor]] = None
        self._vanishing: Optional[Callable[[Tensor], Tensor]] = None

    def forward(self, states: Tensor) -> Tensor:
        """Represent H + B * network with B=0 on the task-failure boundary."""
        output = self.network(self.features(states))
        if self._lift is None:
            return output
        return (
            _column(self._lift(states), states)
            + _column(self._vanishing(states), states) * output
        )

    def configure_terminal_boundary(
        self,
        lift: Callable[[Tensor], Tensor],
        vanishing_factor: Callable[[Tensor], Tensor],
    ) -> None:
        """Fix H to the terminal payoff at the boundary and B to zero there."""
        self._lift = lift
        self._vanishing = vanishing_factor

    def boundary_lift(self, states: Tensor, terminal_payoffs: Tensor) -> Tensor:
        """Represent nonzero task-end payoffs through a fixed boundary lift."""
        if self._lift is None:
            return terminal_payoffs
        return _column(self._lift(states), states)

    def frozen_copy(self) -> SmoothValueHead:
        return _frozen(self)


class RateHead(nn.Module):
    """Separate order-one decision score fitted after each implicit value copy.

    With the optional decision box, inputs are scaled to [-1,1]; gains of order 100
    would otherwise saturate the tanh layers.
    """

    def __init__(
        self,
        state_spec: StateSpec,
        decision_dim: int,
        config: NetworkConfig,
        *,
        decision_low: Optional[Tensor] = None,
        decision_high: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.features = StateFeatures(state_spec, config)
        self.scaling = _DecisionScaling(decision_low, decision_high)
        self.network = _tanh_mlp(
            self.features.output_dim + decision_dim, config["hidden_sizes"], 1
        )

    def forward(self, states: Tensor, decisions: Tensor) -> Tensor:
        return self.network(
            torch.cat((self.features(states), self.scaling(decisions)), dim=-1)
        )

    def frozen_copy(self) -> RateHead:
        return _frozen(self)


class TwinQCritic(nn.Module):
    """Independent critics for the explicit single-level algorithm."""

    def __init__(
        self,
        state_spec: StateSpec,
        decision_dim: int,
        config: NetworkConfig,
        *,
        decision_low: Optional[Tensor] = None,
        decision_high: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.features = StateFeatures(state_spec, config)
        self.scaling = _DecisionScaling(decision_low, decision_high)
        inputs = self.features.output_dim + decision_dim
        self.first = _tanh_mlp(inputs, config["hidden_sizes"], 1)
        self.second = _tanh_mlp(inputs, config["hidden_sizes"], 1)

    def forward(self, states: Tensor, decisions: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = torch.cat((self.features(states), self.scaling(decisions)), dim=-1)
        return self.first(inputs), self.second(inputs)

    def minimum(self, states: Tensor, decisions: Tensor) -> Tensor:
        first, second = self.forward(states, decisions)
        return torch.min(first, second)

    def frozen_copy(self) -> TwinQCritic:
        return _frozen(self)
