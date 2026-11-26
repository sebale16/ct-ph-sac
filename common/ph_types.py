"""Data contracts for CT-PH-SAC; arrays carry physical units."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Mapping, NotRequired, Optional, Sequence, Tuple, TypedDict

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor

# A testbed name: any name registered in environment/ph_sac/registry.py (in-house testbeds in
# environment/ph_sac/tasks.py, dm_control ports in environment/ph_sac/dmc/). The registry,
# not this alias, decides which names are valid.
TestbedName = str
Metrics = Dict[str, float]


class AnalysisContext(TypedDict):
    """Identify the exact candidate, policy, plant and objective in a bound."""

    outer_iteration: int
    candidate_id: str
    policy_id: str
    plant_id: str
    temperature_per_second: float
    policy_kind: Literal["stochastic", "deterministic_mean"]
    objective: Literal["soft", "reward_only"]


class BoundEvidence(TypedDict):
    """External justification for premises; a sampled maximum is not a uniform bound."""

    kind: Literal["sampled_diagnostic", "assumed_uniform", "certified_uniform"]
    domain: str
    justification: str
    failure_probability: Optional[float]


class StateSpec(TypedDict):
    """State is (observed positions, velocities), not momentum or sin/cos features.

    context_dim (default 0) counts episode-constant context coordinates, such as the
    reacher's target. They are the LAST context_dim entries of the position block, so
    position_dim = observed physical positions + context_dim. Their rows of the
    observation projection P are zero, hence their position rates P v and their drift
    are zero; they enter rewards and networks like any other raw position. Periodic
    and actuated indices refer to physical positions only.

    cyclic_position_indices (default empty) lists observed positions, context included,
    on which no force depends (the swimmer's root x, y and its target): the learned
    plant (PortHamiltonianPlant) leaves them out of its features, so its M, U and D
    cannot depend on them. Every other network still reads them.

    limited_position_ranges (default empty) lists (position index, low, high) of the
    coordinates the task limits with a soft one-sided constraint (dm_control's limited
    joints, kept as MuJoCo models them); the learned plant's limit port (network key
    limit_port) adds a learned one-sided spring and damper beyond each range.

    limit_port_init (optional, since 2026-11-10) lists, parallel to
    limited_position_ranges, the port's initial (stiffness, damping, width) per entry:
    the dm_control factories derive them from the compiled MuJoCo limit
    (environment/ph_sac/dmc/joint_limits.limit_port_init). Without it (configs and
    records written before) the port starts at the plant's LIMIT_*_INIT constants.
    """

    position_dim: int
    velocity_dim: int
    action_dim: int
    periodic_position_indices: Tuple[int, ...]
    actuated_position_indices: Tuple[int, ...]
    observation_projection: ndarray
    # Positions also fed to networks raw, divided by a scale: (index, scale) pairs.
    travel_position_scales: NotRequired[Tuple[Tuple[int, float], ...]]
    context_dim: NotRequired[int]
    cyclic_position_indices: NotRequired[Tuple[int, ...]]
    limited_position_ranges: NotRequired[Tuple[Tuple[int, float, float], ...]]
    limit_port_init: NotRequired[Tuple[Tuple[float, float, float], ...]]


def context_dim(spec: StateSpec) -> int:
    """Number of episode-constant context coordinates at the end of the position block."""
    return int(spec.get("context_dim", 0))


def context_position_indices(spec: StateSpec) -> Tuple[int, ...]:
    """State indices of the context coordinates (the last context_dim positions)."""
    positions = int(spec["position_dim"])
    return tuple(range(positions - context_dim(spec), positions))


def dynamic_state_indices(spec: StateSpec) -> Tuple[int, ...]:
    """State indices that can change during an episode: every index but the context."""
    context = set(context_position_indices(spec))
    size = int(spec["position_dim"]) + int(spec["velocity_dim"])
    return tuple(index for index in range(size) if index not in context)


class Transition(TypedDict):
    """One decision interval; next_state is the final observation before reset.

    limit_touched (since 2026-11-10; environments set it, hand-built transitions may omit
    it, meaning False): a limited coordinate (StateSpec limited_position_ranges) was past
    its range at a physics knot of the interval, its start included. near_limit (set by
    the CT-PH-SAC agents on tasks with limited ranges before storing the row): contact
    exclusion's start-state rule (models/ph_sac/rollouts.start_state_flags).
    end_reward_rate (since 2026-11-22; set by the CT-PH-SAC agents before storing the row
    under reward_observation "end_state" only, common.ph_config.REWARD_OBSERVATION_KEY):
    the task's reward rate at next_state with the torque there
    (environment/ph_sac/base.MujocoPHEnv.end_reward_rate).
    """

    state: ndarray
    decision: ndarray
    duration: float
    scheduled_duration: float
    reward: float
    next_state: ndarray
    terminated: bool
    truncated: bool
    time: float
    next_time: float
    limit_touched: NotRequired[bool]
    near_limit: NotRequired[bool]
    end_reward_rate: NotRequired[float]


class TransitionBatch(TypedDict):
    """Batched states/decisions [B,D]; scalar columns [B,1]; times in seconds.

    durations are elapsed, so a failure shortens them; horizons are the scheduled H_i.
    contact_flags (replay batches since 2026-11-10; hand-built batches may omit it,
    meaning no row flagged) is 1. for a row that contact exclusion flags (near_limit or
    limit_touched), else 0. rewards are the interval rewards; a replay under
    reward_observation "end_state" leaves them out and gives end_rewards instead, the rows'
    end-state reward rates rho(s', u_e) (Transition end_reward_rate), so that no learner can
    read an interval reward there.
    """

    states: Tensor
    decisions: Tensor
    durations: Tensor
    horizons: Tensor
    rewards: NotRequired[Tensor]
    next_states: Tensor
    terminated: Tensor
    truncated: Tensor
    contact_flags: NotRequired[Tensor]
    end_rewards: NotRequired[Tensor]


class DecisionSample(TypedDict):
    """Decision draws [B,N,d] and log densities [B,N,1] over the normalized box [-1,1]^d.

    d is 3m for feedback decisions and m for held torques (decision_mode "torque").
    """

    decisions: Tensor
    log_probabilities: Tensor


class SolverResult(TypedDict):
    """One inner solve: completed means all K steps ran, not that V is accurate."""

    inner_steps: int
    completed: bool


class GateResult(TypedDict):
    """One admission attempt: its status, last production batch and validation spending.

    The moments and interval are NaN and the counts zero if no batch could be scored.
    status is supported_improvement, evidence_against, inconclusive_at_validation_cap
    or rejected_nonfinite from the staged gate, where accepted holds only for
    supported_improvement; or admitted_without_test under plant_update "always", which
    scores nothing, always accepts and records only the transfer diagnostic J.
    """

    status: str
    inference_mode: str
    raw_mean: float
    variance_correction: float
    corrected_mean: float
    corrected_std: float
    interval_lower: float
    interval_upper: float
    transfer_diagnostic: float
    accepted: bool
    validation_rows: int
    paths_per_model: int
    stages: int
    real_rows_used: int
    real_rows_acquired: int
    simulated_paths_used: int
    pilot_variance_simulation: float
    pilot_variance_real: float
    stage_log: Sequence[Mapping[str, float]]


class RewardFitRecord(TypedDict):
    """A learned reward rate against the task's true rate; a sampled diagnostic only.

    rmse_taken compares at the replay rows' taken torques, rmse_policy_draws at fresh
    draws of the current policy from a private stream; the true-rate standard deviations
    give their scale. fit_loss is the last trapezoid loss ((predicted - r)/Delta)^2.
    reward_output is the head's output mode, "linear" or "sigmoid(low, high)" for a
    testbed that declares its reward range (RewardRate.output).

    Under reward_observation "end_state" the fit is the end-state regression: fit_loss is
    the last mean of (r_hat(s', u_e) - rho_end)^2, rmse_taken compares at the rows' end
    states and end torques (s', u_e), and the policy draws are taken at s'.
    """

    interactions: int
    steps: int
    fit_loss: float
    rmse_taken: float
    rmse_policy_draws: float
    true_rate_sd_taken: float
    true_rate_sd_draws: float
    reward_output: str
