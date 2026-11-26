"""Learned reward rate r_hat(s, u) for CT-PH-SAC: its trapezoid and end-state fits.

A replay row's reward is the discounted integral of the rate over its interval, so a
row that did not fail is fitted by the trapezoid
    r ~ Delta/2 * (r_hat(s, u_start) + exp(-lambda Delta) * r_hat(s', u_end)),
with u_start and u_end the controller's torques for the row's decision at its two
endpoints. Failed rows are skipped, so the failure payoff is not needed; truncated rows
are kept. The network reads the value head's state features and the torque divided by
the motor limit, with the value head's hidden sizes and tanh. Its scalar output is the
rate itself, or a sigmoid mapped onto the range a testbed declares
(TestbedSpec.reward_rate_bounds; every dm_control task but acrobot declares [0, 1]).

Under reward_observation "end_state" (common.ph_config.REWARD_OBSERVATION_KEY; every
CT-PH-SAC arm) a row's only reward datum is the rate rho_end at its end state s' with the
torque u_end there, so the network is regressed on it, mean((r_hat(s', u_end) -
rho_end)^2), on every row: the rate is a deterministic function of (s, u), so failed,
truncated and shortened rows are valid samples (end_state_fit_rows, end_state_reward_loss).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
from torch import nn

from models.ph_sac.networks import StateFeatures, _tanh_mlp

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import NetworkConfig
    from common.ph_types import StateSpec, TransitionBatch
    from environment.ph_sac.controller import Controller


class RewardRate(nn.Module):
    """r_hat(s, u): [...,D] states and [...,m] torques to a [...,1] reward rate.

    The tanh MLP's scalar output z is the rate itself (linear output) unless
    bounds=(low, high) is given; the rate is then low + (high - low) * sigmoid(z),
    strictly inside the range. The agents (algorithms.ph_base.PHAlgorithm.
    _learned_reward_rate: the implicit arm, and every arm under reward_observation
    "end_state") pass the testbed's
    TestbedSpec.reward_rate_bounds, (0, 1) for dm_control's tolerance rewards, and keep
    the linear output for a testbed that declares none (windy pendulum, Acrobot-XK,
    dm_control acrobot, lqr). The bound matters through the terminal lift
    V = H + B * network: on the reacher the true reward is an indicator in {0, 1}, a
    linear head went negative on much of the state space, and the wrist stop, where
    B = 0 and the payoff H is 0, became the value maximum. A sigmoid rather than a ReLU
    keeps a gradient on the many rows with reward 0, and its upper bound stops
    optimistic extrapolation at plant-rolled states. The bounds are plain attributes,
    not parameters or buffers, so the state dict is the same in both modes.
    """

    def __init__(
        self,
        state_spec: StateSpec,
        network: NetworkConfig,
        torque_limit: Sequence[float],
        *,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.features = StateFeatures(state_spec, network)
        self.network = _tanh_mlp(
            self.features.output_dim + len(torque_limit), network["hidden_sizes"], 1
        )
        self.register_buffer(
            "torque_limit",
            torch.as_tensor(torque_limit, dtype=torch.get_default_dtype()),
        )
        self._bounds: Optional[Tuple[float, float]] = None
        if bounds is not None:
            low, high = (float(value) for value in bounds)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(
                    f"Reward rate bounds must be finite with low < high, got {bounds!r}."
                )
            self._bounds = (low, high)

    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        """(low, high) of a sigmoid output, None for a linear one."""
        return self._bounds

    @property
    def bounded(self) -> bool:
        return self._bounds is not None

    @property
    def output(self) -> str:
        """The output mode for records: "linear" or "sigmoid(low, high)"."""
        if self._bounds is None:
            return "linear"
        return f"sigmoid({self._bounds[0]:g}, {self._bounds[1]:g})"

    def extra_repr(self) -> str:
        return f"output={self.output}"

    def forward(self, states: Tensor, torques: Tensor) -> Tensor:
        z = self.network(
            torch.cat((self.features(states), torques / self.torque_limit), dim=-1)
        )
        if self._bounds is None:
            return z
        low, high = self._bounds
        return low + (high - low) * torch.sigmoid(z)


def reward_fit_rows(
    controller: Controller, batch: TransitionBatch
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Rows that did not fail, with the controller's torques at both interval ends.

    Returns (states, start torques, next states, end torques, durations, rewards), or
    None if every row failed. The decision is resolved once at the start state.
    """
    keep = (batch["terminated"].reshape(-1) < 0.5) & (batch["durations"].reshape(-1) > 0.0)
    if not bool(keep.any()):
        return None
    states, next_states = batch["states"][keep], batch["next_states"][keep]
    with torch.no_grad():
        resolved = controller.resolve_decision(states, batch["decisions"][keep])
        start = controller.action(states, resolved)
        end = controller.action(next_states, resolved)
    return states, start, next_states, end, batch["durations"][keep], batch["rewards"][keep]


def reward_rate_loss(
    network: nn.Module,
    states: Tensor,
    start_torques: Tensor,
    next_states: Tensor,
    end_torques: Tensor,
    durations: Tensor,
    rewards: Tensor,
    *,
    discount_rate: float,
) -> Tensor:
    """mean(((Delta/2 (r_hat(s,u) + e^{-lambda Delta} r_hat(s',u')) - r) / Delta)^2)."""
    predicted = 0.5 * durations * (
        network(states, start_torques)
        + torch.exp(-discount_rate * durations) * network(next_states, end_torques)
    )
    return ((predicted - rewards) / durations).square().mean()


def end_state_fit_rows(
    controller: Controller, batch: TransitionBatch
) -> Tuple[Tensor, Tensor, Tensor]:
    """Every row's end state, end torque and end-state rate: (next states s' [B,D], end
    torques u_e [B,m], rho_end [B,1]).

    u_e is the controller's torque at s' for the row's decision resolved once at its start
    state (the held torque in torque mode). rho_end is the batch's end_rewards (a replay
    batch under reward_observation "end_state"). No row is skipped.
    """
    states, next_states = batch["states"], batch["next_states"]
    with torch.no_grad():
        resolved = controller.resolve_decision(states, batch["decisions"])
        end = controller.action(next_states, resolved)
    return next_states, end, batch["end_rewards"]


def end_state_reward_loss(
    network: nn.Module, next_states: Tensor, end_torques: Tensor, end_rates: Tensor
) -> Tensor:
    """mean((r_hat(s', u_e) - rho_end)^2), the end-state regression of the reward rate."""
    return (network(next_states, end_torques) - end_rates).square().mean()
