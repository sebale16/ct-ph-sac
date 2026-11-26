"""Measurement contracts from the paper's evaluation and diagnostics sections."""

from __future__ import annotations

import math
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
)

import numpy as np
import torch
from scipy.stats import chi2, chisquare

from common.ph_config import DEFAULT_MODEL_INTEGRATOR
from common.ph_types import dynamic_state_indices

if TYPE_CHECKING:
    from algorithms.ph_performance import PerformanceBound
    from algorithms.ph_training import (
        OptimizerStepTrace,
        OriginalIterationBudget,
        OuterStepTrace,
        TrainingAccuracyBounds,
    )
    from common.ph_types import (
        GateResult,
        AnalysisContext,
        BoundEvidence,
        StateSpec,
        TransitionBatch,
    )
    from environment.ph_sac.base import MujocoPHEnv
    from models.ph_sac.plant import Plant


class EpisodeMetrics(TypedDict):
    """Reward-only deployment record; it is not a stochastic soft-return certificate.

    summarize_episode fills the required fields. protocol.evaluate_fixed_seeds adds the
    episode-level fields (protocol.EPISODE_LEVEL_KEYS, defined in its docstring); the
    reward-level ones are NaN when the environment exposes no undiscounted reward. The
    orbit diagnostics (protocol.LYAPUNOV_LEVEL_KEYS) appear only for an environment whose
    reward is the negative log of a Lyapunov function (env.lyapunov_ratio).
    """

    reset_seed: int
    policy_kind: Literal["stochastic", "deterministic_mean"]
    discounted_return: float
    failed: bool
    truncated: bool
    duration_seconds: float
    decisions: int
    physics_steps: int
    episode_return: NotRequired[float]
    survival_seconds: NotRequired[float]
    rate_executed: NotRequired[float]
    rate_horizon: NotRequired[float]
    upright_seconds: NotRequired[float]
    max_hold_seconds: NotRequired[float]
    mean_return_to_go: NotRequired[float]
    lyapunov_ratio_mean: NotRequired[float]
    lyapunov_ratio_final: NotRequired[float]
    orbit_fraction_1e_2: NotRequired[float]
    orbit_fraction_1e_3: NotRequired[float]


class TrainingCounters(TypedDict):
    """Resource accounting for comparing the different update schedules.

    The CT-SAC baseline and the explicit agent report exactly these keys; the implicit
    agent reports ImplicitTrainingCounters, which adds its reward steps, and so does the
    explicit agent under reward_observation "end_state" (one reward step per train step).
    """

    interactions: int
    simulated_seconds: float
    critic_gradient_steps: int
    value_gradient_steps: int
    rate_gradient_steps: int
    actor_gradient_steps: int
    temperature_gradient_steps: int
    plant_gradient_steps: int
    model_rollout_paths: int
    gate_simulated_paths: int
    gate_real_rows_acquired: int
    wall_seconds: float


class ImplicitTrainingCounters(TrainingCounters):
    """TrainingCounters plus the learned reward rate's steps.

    reward_gradient_steps counts the reward warm fit and one step per value step whose
    batch holds a row that did not fail; it stays 0 when reward_source is "known". Under
    reward_observation "end_state" no row is skipped (the end-state fit keeps every row),
    so it counts the warm fit and one step per value step; the explicit agent then counts
    one per train step, without a warm fit. plant_gradient_steps includes the plant warm
    fit.
    """

    reward_gradient_steps: int


class OptionalSolverFields(TypedDict, total=False):
    """Divergence monitors of the step-ablation arms only (protocol.step_ablation_arms;
    every other arm's records keep SolverMetrics' required keys alone). From the solve's
    K-th inner step (NaN after a budget exit): value_loss, that step's loss;
    residual_mean_square, mean_s (mean_i R_i)^2 over its batch before the step (for the
    explicit step, the squared distance of V to the averaged explicit target); value_abs_max
    and value_gradient_norm_max, max |V| and max |grad_s V| of the stepped value on its
    states. Then the last step's loss of the head fitted after the copy: rate_loss (rate
    head) or critic_loss (implicit_critic), NaN when no fit followed.

    The implicit CT-SAC arms (protocol.baseline_implicit_arms, algorithms/ct_sac_implicit.py;
    one record per hard copy of the critics) carry instead: interactions, the decision
    count at the copy; and from the solve's K-th step, before its gradient step,
    residual_u_statistic, the all-pairs loss per critic (an unbiased estimate of
    mean_s (E_draws R)^2, so possibly negative); q_policy_minus_stored, the batch mean of
    mean_i min_j Q_j(s, a_i) - min_j Q_j(s, a) (the critics at the frozen policy's draws
    minus at the stored actions); q_abs_max, max |Q| over every critic value that step
    evaluated."""

    value_loss: float
    residual_mean_square: float
    value_abs_max: float
    value_gradient_norm_max: float
    rate_loss: float
    critic_loss: float
    interactions: int
    residual_u_statistic: float
    q_policy_minus_stored: float
    q_abs_max: float


class SolverMetrics(OptionalSolverFields):
    """One copied solve: K inner steps unless the budget ended it; no status certifies accuracy."""

    outer_iteration: int
    inner_iterations: int
    status: Literal["completed", "budget_exit"]
    contraction_factor: float
    fixed_point_motion_input: float


class AdmissionMetrics(TypedDict):
    """One attempt: status, last production batch, spending, and return components.

    The corrected rows of each fresh batch drive the decision; J is computed only for
    supported improvement and is a sampled diagnostic. stage_log holds every stage.
    """

    admission_index: int
    outer_iteration: int
    status: str
    inference_mode: str
    accepted: bool
    corrected_mean: float
    corrected_standard_deviation: float
    raw_mean: float
    mean_variance_correction: float
    interval_lower: float
    interval_upper: float
    transfer_diagnostic: float
    admitted_mean_square: float
    admitted_monte_carlo_variance: float
    observed_failure_rate: float
    candidate_failure_rate: float
    admitted_failure_rate: float
    candidate_running_reward: float
    admitted_running_reward: float
    candidate_failure_payoff: float
    admitted_failure_payoff: float
    candidate_continuation: float
    admitted_continuation: float
    mean_scheduled_horizon: float
    validation_rows: int
    rollout_paths_per_row: int
    stages: int
    real_rows_used: int
    real_rows_acquired: int
    simulated_paths_used: int
    pilot_variance_simulation: float
    pilot_variance_real: float
    stage_log: Sequence[Mapping[str, object]]
    next_outer_iterations: int


class OracleMetrics(TypedDict):
    """True-plant comparisons on sampled states, with coverage stated explicitly."""

    interactions: int
    sampled_states: int
    drift_error_max: float
    diffusion_covariance_error_max: float
    value_direction_error_max: float
    value_direction_error_mean: float


class PlantCalibrationBin(TypedDict):
    """Duration-binned likelihood calibration and true-plant Monte Carlo checks."""

    interactions: int
    duration_lower_seconds: float
    duration_upper_seconds: float
    sample_count: int
    state_dimension: int
    mean_mahalanobis_square: float
    upper_tail_fraction: float
    endpoint_mean_error: Optional[float]
    endpoint_covariance_error: Optional[float]
    chi_square_goodness_of_fit_statistic: float
    chi_square_goodness_of_fit_pvalue: float
    duration_jacobian_norm_max: float
    saturated_fraction: float


class SampledTheoryDiagnostics(TypedDict):
    """Sample maxima and path lengths; never uniform theorem premises by themselves."""

    context: AnalysisContext
    evidence: BoundEvidence
    candidate_change_max: float
    raw_policy_residual_max: float
    evaluation_residual_max: Optional[float]
    rate_error_max: Optional[float]
    actor_gap_max: Optional[float]
    holding_consistency_max: Optional[float]
    online_admitted_lag_max: Optional[float]
    rate_target_bias_l2: Optional[float]
    value_update_path_length: float
    actor_update_path_length: float


class TheoryDiagnosticsRecord(TypedDict):
    """Keep contexts/evidence alongside optional conditional theorem accounting."""

    context: AnalysisContext
    solve_context: Optional[AnalysisContext]
    sampled: Sequence[SampledTheoryDiagnostics]
    optimizer_steps: Sequence[OptimizerStepTrace]
    training_accuracy: Optional[TrainingAccuracyBounds]
    outer_step: Optional[OuterStepTrace]
    original_iteration_budget: Optional[OriginalIterationBudget]
    performance_bound: Optional[PerformanceBound]


def summarize_episode(
    *,
    reset_seed: int,
    interval_rewards: Sequence[float],
    interval_start_times: Sequence[float],
    discount_rate: float,
    terminated: bool,
    truncated: bool,
    duration_seconds: float,
    physics_steps: int,
    policy_kind: Literal["stochastic", "deterministic_mean"],
) -> EpisodeMetrics:
    """Compose environment rewards/failure charge without adding an entropy term."""
    if len(interval_rewards) != len(interval_start_times):
        raise ValueError("Each interval reward needs one start time.")
    rewards = np.asarray(interval_rewards, dtype=np.float64)
    starts = np.asarray(interval_start_times, dtype=np.float64)
    return {
        "reset_seed": int(reset_seed),
        "policy_kind": policy_kind,
        "discounted_return": float(np.sum(np.exp(-discount_rate * starts) * rewards)),
        "failed": bool(terminated),
        "truncated": bool(truncated),
        "duration_seconds": float(duration_seconds),
        "decisions": len(rewards),
        "physics_steps": int(physics_steps),
    }


def measure_oracle_errors(
    learned_plant: Plant,
    oracle_plant: Plant,
    states: np.ndarray,
    actions: np.ndarray,
    value_gradients: np.ndarray,
    *,
    interactions: int,
) -> OracleMetrics:
    """Measure epsilon_b, epsilon_A, and drift error along the value gradient."""
    dtype = torch.get_default_dtype()
    states_t = torch.as_tensor(states, dtype=dtype)
    actions_t = torch.as_tensor(actions, dtype=dtype)
    gradients_t = torch.as_tensor(value_gradients, dtype=dtype)
    if states_t.ndim != 2 or len(states_t) == 0:
        raise ValueError("Oracle errors need a nonempty [B,D] state batch.")
    with torch.no_grad():
        drift_error = learned_plant.drift(states_t, actions_t) - oracle_plant.drift(
            states_t, actions_t
        )
        covariance_error = learned_plant.covariance(states_t) - oracle_plant.covariance(
            states_t
        )
    value_direction = (drift_error * gradients_t).sum(dim=-1).abs()
    return {
        "interactions": int(interactions),
        "sampled_states": len(states_t),
        "drift_error_max": float(drift_error.norm(dim=-1).max()),
        "diffusion_covariance_error_max": float(
            torch.linalg.matrix_norm(covariance_error, ord="fro").max()
        ),
        "value_direction_error_max": float(value_direction.max()),
        "value_direction_error_mean": float(value_direction.mean()),
    }


def measure_gate_diagnostics(
    *,
    admission_index: int,
    outer_iteration: int,
    result: GateResult,
    production: Optional[Mapping[str, object]],
    next_outer_iterations: int,
) -> AdmissionMetrics:
    """Record an attempt and, if a batch was scored, the fit terms of its last batch.

    production holds admitted_fit_terms as squares and variances, the scheduled horizons,
    both plants' return summaries and the observed failure rate; None means no batch.
    Stored rows simulated again never count as real data here.
    """
    nan = math.nan
    blank = {"failure_rate": nan, "running_reward": nan, "failure_payoff": nan, "continuation": nan}
    candidate = blank if production is None else production["candidate"]
    admitted = blank if production is None else production["admitted"]
    return {
        "admission_index": int(admission_index),
        "outer_iteration": int(outer_iteration),
        "status": result["status"],
        "inference_mode": result["inference_mode"],
        "accepted": bool(result["accepted"]),
        "corrected_mean": result["corrected_mean"],
        "corrected_standard_deviation": result["corrected_std"],
        "raw_mean": result["raw_mean"],
        "mean_variance_correction": result["variance_correction"],
        "interval_lower": result["interval_lower"],
        "interval_upper": result["interval_upper"],
        "transfer_diagnostic": result["transfer_diagnostic"],
        "admitted_mean_square": (
            nan if production is None else float(np.mean(production["squares"]))
        ),
        "admitted_monte_carlo_variance": (
            nan if production is None else float(np.mean(production["variances"]))
        ),
        "observed_failure_rate": (
            nan if production is None else float(production["observed_failure_rate"])
        ),
        "candidate_failure_rate": candidate["failure_rate"],
        "admitted_failure_rate": admitted["failure_rate"],
        "candidate_running_reward": candidate["running_reward"],
        "admitted_running_reward": admitted["running_reward"],
        "candidate_failure_payoff": candidate["failure_payoff"],
        "admitted_failure_payoff": admitted["failure_payoff"],
        "candidate_continuation": candidate["continuation"],
        "admitted_continuation": admitted["continuation"],
        "mean_scheduled_horizon": (
            nan if production is None else float(np.mean(production["horizons"]))
        ),
        "validation_rows": int(result["validation_rows"]),
        "rollout_paths_per_row": int(result["paths_per_model"]),
        "stages": int(result["stages"]),
        "real_rows_used": int(result["real_rows_used"]),
        "real_rows_acquired": int(result["real_rows_acquired"]),
        "simulated_paths_used": int(result["simulated_paths_used"]),
        "pilot_variance_simulation": result["pilot_variance_simulation"],
        "pilot_variance_real": result["pilot_variance_real"],
        "stage_log": result["stage_log"],
        "next_outer_iterations": int(next_outer_iterations),
    }


def _mahalanobis_squares(residuals: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """e^T Lambda^-1 e by Cholesky solves [n]; a singular row uses a pseudoinverse."""
    squares = np.empty(len(residuals))
    for row, (residual, covariance) in enumerate(zip(residuals, covariances)):
        try:
            factor = np.linalg.cholesky(covariance)
            whitened = np.linalg.solve(factor, residual)
            squares[row] = float(whitened @ whitened)
        except np.linalg.LinAlgError:
            squares[row] = float(residual @ np.linalg.pinv(covariance) @ residual)
    return squares


def _chi_square_fit(squares: np.ndarray, dimension: int) -> tuple[float, float]:
    """Goodness of fit against chi2_d in equal-probability cells; NaN below two cells."""
    cells = min(squares.size // 5, 20)
    if cells < 2:
        return math.nan, math.nan
    edges = chi2.ppf(np.linspace(0.0, 1.0, cells + 1), dimension)
    counts, _ = np.histogram(squares, bins=edges)
    result = chisquare(counts)
    return float(result.statistic), float(result.pvalue)


def _duration_bins(
    durations: np.ndarray, edges: Sequence[float]
) -> list[tuple[float, float, np.ndarray]]:
    """Half-open duration bins, closing the last; empty bins are skipped."""
    bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        last = index == len(edges) - 2
        mask = (durations >= lower) & (
            durations <= upper if last else durations < upper
        )
        if mask.any():
            bins.append((float(lower), float(upper), mask))
    return bins


def _wrap_residuals(residuals: np.ndarray, state_spec: Optional[StateSpec]) -> np.ndarray:
    if state_spec is None:
        return residuals
    wrapped = residuals.copy()
    for index in state_spec["periodic_position_indices"]:
        wrapped[..., index] = (
            np.mod(wrapped[..., index] + math.pi, 2.0 * math.pi) - math.pi
        )
    return wrapped


def _dynamic_indices(state_spec: Optional[StateSpec], size: int) -> np.ndarray:
    """State indices that move: all but an episode-constant context, which has no noise."""
    if state_spec is None:
        return np.arange(size)
    return np.asarray(dynamic_state_indices(state_spec), dtype=np.int64)


def calibrate_by_duration(
    transitions: TransitionBatch,
    predicted_endpoints: np.ndarray,
    predicted_covariances: np.ndarray,
    *,
    interactions: int,
    duration_bin_edges: Sequence[float],
    upper_tail_probability: float,
    duration_jacobian_norms: np.ndarray,
    saturated_fractions: np.ndarray,
    state_spec: Optional[StateSpec] = None,
    learns_diffusion: bool = True,
) -> Sequence[PlantCalibrationBin]:
    """Duration-binned Mahalanobis errors, tail behavior, and path diagnostics.

    state_spec, when given, wraps periodic position residuals as the plant fit does.
    These scores read the learned noise model; a drift-only plant
    (learns_diffusion=False) has none, so it gets no records.
    """
    if not learns_diffusion:
        return []
    durations = _to_numpy(transitions["durations"]).reshape(-1)
    observed = _to_numpy(transitions["next_states"])
    residuals = _wrap_residuals(
        observed - np.asarray(predicted_endpoints, dtype=np.float64), state_spec
    )
    covariances = np.asarray(predicted_covariances, dtype=np.float64)
    keep = _dynamic_indices(state_spec, observed.shape[-1])
    residuals = residuals[:, keep]
    covariances = covariances[:, keep[:, None], keep[None, :]]
    squares = _mahalanobis_squares(residuals, covariances)
    dimension = len(keep)
    threshold = chi2.isf(upper_tail_probability, dimension)
    jacobians = np.asarray(duration_jacobian_norms, dtype=np.float64).reshape(-1)
    saturation = np.asarray(saturated_fractions, dtype=np.float64).reshape(-1)

    records: list[PlantCalibrationBin] = []
    for lower, upper, mask in _duration_bins(durations, duration_bin_edges):
        statistic, pvalue = _chi_square_fit(squares[mask], dimension)
        records.append(
            {
                "interactions": int(interactions),
                "duration_lower_seconds": lower,
                "duration_upper_seconds": upper,
                "sample_count": int(mask.sum()),
                "state_dimension": int(dimension),
                "mean_mahalanobis_square": float(squares[mask].mean()),
                "upper_tail_fraction": float((squares[mask] > threshold).mean()),
                "endpoint_mean_error": None,
                "endpoint_covariance_error": None,
                "chi_square_goodness_of_fit_statistic": statistic,
                "chi_square_goodness_of_fit_pvalue": pvalue,
                "duration_jacobian_norm_max": float(jacobians[mask].max()),
                "saturated_fraction": float(saturation[mask].mean()),
            }
        )
    return records


def calibrate_against_true_paths(
    env_factory: Callable[[int], MujocoPHEnv],
    learned_plant: Plant,
    transitions: TransitionBatch,
    *,
    interactions: int,
    rollout_seeds: Sequence[int],
    paths_per_state: int,
    duration_bin_edges: Sequence[float],
    upper_tail_probability: float,
    model_integrator: str = DEFAULT_MODEL_INTEGRATOR,
) -> Sequence[PlantCalibrationBin]:
    """MuJoCo Monte Carlo endpoint moments and chi-square calibration checks.

    Every row is replayed paths_per_state times for each seed, from its start state
    with its decision held for its duration. Only surviving paths enter the moments,
    because the Gaussian model describes an unkilled endpoint; a state needs at least
    two surviving paths to contribute to the covariance error. A drift-only plant has
    no noise model to calibrate and gets no records. The model moments integrate the
    learned plant at the physics step with model_integrator (a config's
    model_integrator value).
    """
    from models.ph_sac.rollouts import integrate_endpoint_moments, path_jacobian_norms

    if paths_per_state < 1 or not rollout_seeds:
        raise ValueError("True-path calibration needs seeds and at least one path.")
    if not getattr(learned_plant, "learns_diffusion", True):
        return []
    states = _to_numpy(transitions["states"])
    decisions = _to_numpy(transitions["decisions"])
    durations = _to_numpy(transitions["durations"]).reshape(-1)

    probe = env_factory(int(rollout_seeds[0]))
    spec = probe.state_spec
    grid = {
        "maximum_step": probe.physics_dt,
        "physics_dt": probe.physics_dt,
        "controller_refresh_steps": probe.controller.refresh_steps,
        "integrator": model_integrator,
    }
    dtype = torch.get_default_dtype()
    states_t = torch.as_tensor(states, dtype=dtype)
    decisions_t = torch.as_tensor(decisions, dtype=dtype)
    durations_t = torch.as_tensor(durations, dtype=dtype).reshape(-1, 1)
    moments = integrate_endpoint_moments(
        learned_plant,
        probe.controller,
        states_t,
        decisions_t,
        durations_t,
        covariance_mode="propagated",
        create_graph=False,
        **grid,
    )
    jacobians = path_jacobian_norms(
        learned_plant, probe.controller, states_t, decisions_t, durations_t, **grid
    )
    means = moments["means"].detach().cpu().numpy().astype(np.float64)
    covariances = moments["covariances"].detach().cpu().numpy().astype(np.float64)
    keep = _dynamic_indices(spec, states.shape[-1])
    covariances = covariances[:, keep[:, None], keep[None, :]]
    jacobians = jacobians.detach().cpu().numpy().reshape(-1)
    saturation = moments["saturation_fractions"].detach().cpu().numpy().reshape(-1)

    endpoints: list[list[np.ndarray]] = [[] for _ in range(len(states))]
    for position, seed in enumerate(rollout_seeds):
        env = probe if position == 0 else env_factory(int(seed))
        env.reset(seed=int(seed), options={"state": states[0]})
        for row in range(len(states)):
            for _ in range(paths_per_state):
                env.reset(options={"state": states[row]})
                transition = env.step_dt(decisions[row], float(durations[row]))
                if not transition["terminated"]:
                    endpoints[row].append(np.asarray(transition["next_state"]))
        env.close()

    dimension = len(keep)
    threshold = chi2.isf(upper_tail_probability, dimension)
    records: list[PlantCalibrationBin] = []
    for lower, upper, mask in _duration_bins(durations, duration_bin_edges):
        squares: list[np.ndarray] = []
        mean_errors: list[float] = []
        covariance_errors: list[float] = []
        for row in np.flatnonzero(mask):
            if not endpoints[row]:
                continue
            residuals = _wrap_residuals(
                np.stack(endpoints[row]) - means[row], spec
            )[:, keep]
            squares.append(
                _mahalanobis_squares(
                    residuals, np.broadcast_to(covariances[row], (len(residuals),) + covariances[row].shape)
                )
            )
            mean_errors.append(float(np.linalg.norm(residuals.mean(axis=0))))
            if len(residuals) >= 2:
                sample = np.cov(residuals, rowvar=False)
                covariance_errors.append(
                    float(np.linalg.norm(sample - covariances[row], ord="fro"))
                )
        if not squares:
            continue
        pooled = np.concatenate(squares)
        statistic, pvalue = _chi_square_fit(pooled, dimension)
        records.append(
            {
                "interactions": int(interactions),
                "duration_lower_seconds": lower,
                "duration_upper_seconds": upper,
                "sample_count": int(pooled.size),
                "state_dimension": int(dimension),
                "mean_mahalanobis_square": float(pooled.mean()),
                "upper_tail_fraction": float((pooled > threshold).mean()),
                "endpoint_mean_error": float(np.mean(mean_errors)),
                "endpoint_covariance_error": (
                    float(np.mean(covariance_errors)) if covariance_errors else None
                ),
                "chi_square_goodness_of_fit_statistic": statistic,
                "chi_square_goodness_of_fit_pvalue": pvalue,
                "duration_jacobian_norm_max": float(jacobians[mask].max()),
                "saturated_fraction": float(saturation[mask].mean()),
            }
        )
    return records


def _to_numpy(values: object) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def _largest_magnitude(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _optional_largest(values: Optional[np.ndarray]) -> Optional[float]:
    return None if values is None else _largest_magnitude(np.asarray(values))


def _path_length(increments: Sequence[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(np.ravel(step)) for step in increments))


def summarize_sampled_theory_diagnostics(
    *,
    context: AnalysisContext,
    domain: str,
    sampling_description: str,
    candidate_changes: np.ndarray,
    raw_policy_residuals: np.ndarray,
    value_update_increments: Sequence[np.ndarray],
    actor_update_increments: Sequence[np.ndarray],
    evaluation_residuals: Optional[np.ndarray] = None,
    rate_errors: Optional[np.ndarray] = None,
    actor_gaps: Optional[np.ndarray] = None,
    holding_consistency_errors: Optional[np.ndarray] = None,
    online_admitted_generator_differences: Optional[np.ndarray] = None,
    rate_target_bias_samples: Optional[np.ndarray] = None,
    rate_target_sample_weights: Optional[np.ndarray] = None,
) -> SampledTheoryDiagnostics:
    """Sampled-only records; raw residuals are kept as measurements, never as a stop rule."""
    bias_l2: Optional[float] = None
    if rate_target_bias_samples is not None:
        bias = np.asarray(rate_target_bias_samples, dtype=np.float64).reshape(-1)
        weights = (
            np.ones_like(bias)
            if rate_target_sample_weights is None
            else np.asarray(rate_target_sample_weights, dtype=np.float64).reshape(-1)
        )
        if weights.shape != bias.shape or weights.sum() <= 0.0 or (weights < 0.0).any():
            raise ValueError("Rate-bias weights must be nonnegative and match the samples.")
        bias_l2 = float(np.sqrt(np.sum(weights * bias**2) / weights.sum()))
    return {
        "context": context,
        "evidence": {
            "kind": "sampled_diagnostic",
            "domain": domain,
            "justification": sampling_description,
            "failure_probability": None,
        },
        "candidate_change_max": _largest_magnitude(np.asarray(candidate_changes)),
        "raw_policy_residual_max": _largest_magnitude(np.asarray(raw_policy_residuals)),
        "evaluation_residual_max": _optional_largest(evaluation_residuals),
        "rate_error_max": _optional_largest(rate_errors),
        "actor_gap_max": _optional_largest(actor_gaps),
        "holding_consistency_max": _optional_largest(holding_consistency_errors),
        "online_admitted_lag_max": _optional_largest(
            online_admitted_generator_differences
        ),
        "rate_target_bias_l2": bias_l2,
        "value_update_path_length": _path_length(value_update_increments),
        "actor_update_path_length": _path_length(actor_update_increments),
    }


def _validate_context(context: AnalysisContext, where: str) -> None:
    if context["policy_kind"] == "deterministic_mean" and (
        context["temperature_per_second"] != 0.0 or context["objective"] != "reward_only"
    ):
        raise ValueError(
            f"{where}: a deterministic-mean policy needs alpha_c=0 and a reward-only "
            "objective with residuals recomputed for that deployed policy."
        )


def validate_theory_record_context(record: TheoryDiagnosticsRecord) -> None:
    """Preserve old/new actor contexts and require alpha_c=0 for mean-policy bounds."""
    context = record["context"]
    solve_context = record["solve_context"]
    _validate_context(context, "record context")
    if solve_context is not None:
        _validate_context(solve_context, "solve context")
        if solve_context["outer_iteration"] > context["outer_iteration"]:
            raise ValueError("A solve context cannot follow the context it precedes.")

    allowed = [context] if solve_context is None else [context, solve_context]
    for sampled in record["sampled"]:
        if sampled["evidence"]["kind"] != "sampled_diagnostic":
            raise ValueError("Sampled records must not claim uniform evidence.")
        if sampled["context"] not in allowed:
            raise ValueError("A sampled diagnostic refers to an unrelated context.")

    bound = record["performance_bound"]
    if bound is not None:
        if bound["context"] != context:
            raise ValueError("A performance bound must refer to the record's context.")
        if (
            bound["claim_kind"] == "certified"
            and bound["evidence"]["kind"] != "certified_uniform"
        ):
            raise ValueError("A certificate needs certified uniform evidence.")

    accuracy = record["training_accuracy"]
    if accuracy is not None:
        if accuracy["context"] != context:
            raise ValueError("Training accuracy must refer to the record's context.")
        if solve_context is not None and accuracy["residual_context"] != solve_context:
            raise ValueError("The raw solve residual must refer to the solve context.")

    outer = record["outer_step"]
    if outer is not None and solve_context is not None:
        if outer["solve_output_context"] != solve_context:
            raise ValueError("The outer step's solve output must match the solve context.")
