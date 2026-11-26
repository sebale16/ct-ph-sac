"""Paired full-return admission statistics, staged allocation, untested admission and spacing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, TypedDict

import torch

from common.ph_config import DEFAULT_MODEL_INTEGRATOR
from models.ph_sac.operators import policy_generator_terms
from models.ph_sac.rollouts import ReturnTerms, path_continuations, stochastic_rollout

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_types import GateResult, TransitionBatch
    from environment.ph_sac.controller import Controller
    from models.ph_sac.networks import DecisionActor, SmoothValueHead
    from models.ph_sac.plant import Plant
    from models.ph_sac.rollouts import ContactExclusion, ContactSubstepping


class ReturnSummary(TypedDict):
    """Means over rows and paths of one plant's simulated return and its parts."""

    failure_rate: float
    running_reward: float
    failure_payoff: float
    continuation: float


class PairedGateSamples(TypedDict):
    """Full returns: observed G_i [n,1], plants G_il [n,L] with L>=2, scheduled H_i [n,1]."""

    observed_returns: Tensor
    candidate_returns: Tensor
    admitted_returns: Tensor
    horizons: Tensor
    candidate_summary: ReturnSummary
    admitted_summary: ReturnSummary
    observed_failure_rate: float


class CorrectedGateRows(TypedDict):
    """Per-row D_raw, subtracted (S_phi^2-S_hat^2)/(L H^2), and D, each [n,1]."""

    raw_differences: Tensor
    variance_corrections: Tensor
    corrected_differences: Tensor


def paired_gate_samples(
    candidate: Plant,
    admitted: Plant,
    value: SmoothValueHead,
    controller: Controller,
    batch: TransitionBatch,
    *,
    failure_predicate: Callable[[Tensor], Tensor],
    return_terms: ReturnTerms,
    maximum_step: float,
    physics_dt: float,
    controller_refresh_steps: Optional[int],
    path_count: int,
    brownian_increments: Tensor,
    integrator: str = DEFAULT_MODEL_INTEGRATOR,
    contact: Optional[ContactSubstepping] = None,
    truncation: Optional[ContactExclusion] = None,
    skip_rows: Optional[Tensor] = None,
    observed_rewards: Optional[Tensor] = None,
) -> PairedGateSamples:
    """Share noise between plants; paths stay independent of the observed outcome, all rows used.

    The observed return G_i = r_i + e^(-lambda H_i) (1-d_i) V(s'_i) reads r_i from
    observed_rewards [n,1] when given (reward_observation "end_state": the observed ends'
    trapezoid of the learned rate over the elapsed interval plus the failure charge, so
    that it depends on neither plant), else the batch's interval rewards.

    Each failure charge appears once, in the observed r_i or in that model path's own
    return. Failed paths keep their running reward and payoff with zero continuation,
    with no survivor renormalization or observed-outcome fallback. Both plants'
    paths use the same substep map (integrator, see models/ph_sac/rollouts.py) and the
    same contact substepping (contact): the shared increments live on the fine grid, so
    a path keeps its noise whichever step each plant's detector chooses for it.

    Contact exclusion (truncation, skip_rows; None: off): a path that comes near a limit
    is truncated at its last clean knot t_c and bootstrapped there, G = (reward on
    [0, t_c]) + e^(-lambda t_c) V(s_(t_c)) (path_continuations); the rows of skip_rows
    (flagged gate rows) are not simulated, so both plants return V(s) on every path and
    the row's corrected difference D_i is exactly 0: it carries no evidence either way.
    """
    if path_count < 2:
        raise ValueError(f"The gate correction needs L >= 2 paths, got {path_count}.")
    if batch["states"].shape[0] < 2:
        raise ValueError("The gate needs n >= 2 validation rows.")
    if brownian_increments.shape[1] != path_count:
        raise ValueError("Brownian increments must hold path_count paths per row.")
    noise_dim = brownian_increments.shape[-1]
    if candidate.noise_dim != noise_dim or admitted.noise_dim != noise_dim:
        raise ValueError("Both plants must read the shared noise columns.")
    horizons = batch["horizons"].reshape(-1, 1)
    discount_rate = return_terms["discount_rate"]

    def simulated(plant: Plant) -> Tuple[Tensor, ReturnSummary]:
        paths = stochastic_rollout(
            plant,
            controller,
            batch["states"],
            batch["decisions"],
            horizons,
            failure_predicate,
            maximum_step=maximum_step,
            physics_dt=physics_dt,
            controller_refresh_steps=controller_refresh_steps,
            path_count=path_count,
            brownian_increments=brownian_increments,
            return_terms=return_terms,
            integrator=integrator,
            contact=contact,
            truncation=truncation,
            skip_rows=skip_rows,
        )
        continuation = path_continuations(value, paths, horizons, discount_rate)
        returns = paths["running_rewards"] + paths["failure_payoffs"] + continuation
        return returns.squeeze(-1), {
            "failure_rate": float((~paths["survived"]).double().mean()),
            "running_reward": float(paths["running_rewards"].mean()),
            "failure_payoff": float(paths["failure_payoffs"].mean()),
            "continuation": float(continuation.mean()),
        }

    with torch.no_grad():
        terminated = batch["terminated"].double().reshape(-1, 1)
        rewards = batch["rewards"] if observed_rewards is None else observed_rewards
        observed = rewards.double().reshape(-1, 1) + torch.exp(
            -discount_rate * horizons.double()
        ) * (1.0 - terminated) * value(batch["next_states"]).double()
    candidate_returns, candidate_summary = simulated(candidate)
    admitted_returns, admitted_summary = simulated(admitted)
    return {
        "observed_returns": observed,
        "candidate_returns": candidate_returns,
        "admitted_returns": admitted_returns,
        "horizons": horizons,
        "candidate_summary": candidate_summary,
        "admitted_summary": admitted_summary,
        "observed_failure_rate": float(terminated.mean()),
    }


def corrected_gate_rows(samples: PairedGateSamples) -> CorrectedGateRows:
    """D_i=e_phi^2-e_hat^2-(S_phi^2-S_hat^2)/(L H^2), with e=(G-G_hat)/H.

    Rows are double precision: 1/H^2 multiplies small return differences.
    """
    candidate = samples["candidate_returns"].double()
    admitted = samples["admitted_returns"].double()
    observed = samples["observed_returns"].double().reshape(-1, 1)
    rows, paths = candidate.shape
    if paths < 2:
        raise ValueError(f"The gate correction needs L >= 2 paths, got {paths}.")
    if rows < 2:
        raise ValueError(f"The gate needs n >= 2 validation rows, got {rows}.")
    horizons = samples["horizons"].reshape(-1, 1).double()

    candidate_error = (observed - candidate.mean(dim=1, keepdim=True)) / horizons
    admitted_error = (observed - admitted.mean(dim=1, keepdim=True)) / horizons
    raw = candidate_error.square() - admitted_error.square()
    corrections = (
        candidate.var(dim=1, unbiased=True, keepdim=True)
        - admitted.var(dim=1, unbiased=True, keepdim=True)
    ) / (paths * horizons.square())
    return {
        "raw_differences": raw,
        "variance_corrections": corrections,
        "corrected_differences": raw - corrections,
    }


def admitted_fit_terms(samples: PairedGateSamples) -> Tuple[Tensor, Tensor]:
    """Appendix diagnostics e_i(phi_hat)^2 and S_i^2(phi_hat)/(L H_i^2), each [n,1].

    Their difference removes only the model's Monte Carlo share; it keeps the
    variance of the observed full return.
    """
    admitted = samples["admitted_returns"].double()
    observed = samples["observed_returns"].double().reshape(-1, 1)
    horizons = samples["horizons"].reshape(-1, 1).double()
    squares = ((observed - admitted.mean(dim=1, keepdim=True)) / horizons).square()
    variances = admitted.var(dim=1, unbiased=True, keepdim=True) / (
        admitted.shape[1] * horizons.square()
    )
    return squares, variances


def transfer_diagnostic(
    candidate: Plant,
    admitted: Plant,
    value: SmoothValueHead,
    policy: DecisionActor,
    controller: Controller,
    states: Tensor,
    *,
    discount_rate: float,
    policy_draws: int,
) -> Tensor:
    """J=max_s |E_pi[(b_phi-b_hat).grad V + .5(A_phi-A_hat):Hess V]|/lambda, a 0-d tensor.

    Both plants read the same draws. J is a sampled score, not a bound on the jump.
    """
    with torch.no_grad():
        decisions = policy.sample(states, sample_count=policy_draws)["decisions"]
    _, candidate_generator = policy_generator_terms(
        value, candidate, controller, states, decisions, create_graph=False
    )
    _, admitted_generator = policy_generator_terms(
        value, admitted, controller, states, decisions, create_graph=False
    )
    difference = (candidate_generator - admitted_generator).mean(dim=1)
    return difference.abs().max() / discount_rate


class GateInterval(TypedDict):
    """I_r=[D_hat-w,D_hat+w] from one fresh production batch, with both row moments."""

    rows: int
    corrected_mean: float
    corrected_std: float
    width: float
    lower: float
    upper: float


def budgeted_gate_width(
    *,
    rows: int,
    look: int,
    gate_failure_budget: float,
    return_bound: float,
    min_horizon: float,
) -> float:
    """Hoeffding w=C sqrt(2 log(2/beta_g)/n), C=5M^2/H_min^2, beta_g=beta/(g(g+1)) over looks."""
    if look < 1:
        raise ValueError(f"Production looks count from one, got {look}.")
    budget = gate_failure_budget / (look * (look + 1))
    scale = 5.0 * return_bound**2 / min_horizon**2
    return scale * math.sqrt(2.0 * math.log(2.0 / budget) / rows)


def gate_interval(
    rows: CorrectedGateRows, *, margin_z: float, width: Optional[float] = None
) -> GateInterval:
    """Default width z s_D/sqrt(n) is an empirical margin; width overrides it."""
    corrected = rows["corrected_differences"].double().reshape(-1)
    count = corrected.numel()
    if count < 2:
        raise ValueError(f"The gate needs n >= 2 validation rows, got {count}.")
    mean = float(corrected.mean())
    deviation = float(corrected.std(unbiased=True))
    half = margin_z * deviation / math.sqrt(count) if width is None else float(width)
    return {
        "rows": count,
        "corrected_mean": mean,
        "corrected_std": deviation,
        "width": half,
        "lower": mean - half,
        "upper": mean + half,
    }


def gate_branch(
    interval: GateInterval,
) -> Literal["supported_improvement", "evidence_against", "inconclusive", "rejected_nonfinite"]:
    """Admit if sup I_r<0, keep the plant if inf I_r>0; a non-finite interval rejects."""
    if not (math.isfinite(interval["lower"]) and math.isfinite(interval["upper"])):
        return "rejected_nonfinite"
    if interval["upper"] < 0.0:
        return "supported_improvement"
    if interval["lower"] > 0.0:
        return "evidence_against"
    return "inconclusive"


def pilot_variance_components(pilot_rows: Tensor) -> Tuple[float, float]:
    """v_MC=mean_i W_i^2 and v_real=max(s^2_Dbar-v_MC/B,0) from D_ib [p,B]; allocation only."""
    rows, batches = pilot_rows.shape
    if rows < 2 or batches < 2:
        raise ValueError(f"The pilot needs p,B >= 2, got p={rows} and B={batches}.")
    pilot = pilot_rows.double()
    simulation = float(pilot.var(dim=1, unbiased=True).mean())
    across_rows = float(pilot.mean(dim=1).var(unbiased=True))
    return simulation, max(across_rows - simulation / batches, 0.0)


def gate_stage_paths(rows: int, paths: int, *, pilot_rows: int, pilot_batches: int) -> int:
    """A stage costs 2 L_r (B p_r + n_r) simulated paths, counting both plants."""
    return 2 * paths * (pilot_batches * pilot_rows + rows)


def next_gate_counts(
    rows: int,
    paths: int,
    *,
    simulation_variance: float,
    real_variance: float,
    fits: Callable[[int, int], bool],
) -> Optional[Tuple[int, int]]:
    """Prefer doubling L if v_MC>v_real, else n; try the other, or None if neither fits."""
    more_paths, more_rows = (rows, 2 * paths), (2 * rows, paths)
    order = (
        (more_paths, more_rows)
        if simulation_variance > real_variance
        else (more_rows, more_paths)
    )
    for candidate in order:
        if fits(*candidate):
            return candidate
    return None


def admission_spacing(*, discount_rate: float, resolvent_step: float, efolds: float) -> int:
    """M=ceil(n_e/log(1+lambda tau)) outer iterations after every admission attempt."""
    if discount_rate <= 0.0 or resolvent_step <= 0.0:
        raise ValueError("discount_rate and resolvent_step must be positive.")
    if efolds <= 0.0:
        raise ValueError("efolds must be positive.")
    return int(math.ceil(efolds / math.log1p(discount_rate * resolvent_step)))


def untested_admission_result(transfer: float) -> GateResult:
    """Record plant_update=always: the online plant is admitted with no test.

    Nothing is scored, no held-out row is read and no gate path is simulated; the sampled
    transfer score J is recorded and decides nothing, whatever its value.
    """
    nan = float("nan")
    return {
        "status": "admitted_without_test",
        "inference_mode": "untested",
        "raw_mean": nan,
        "variance_correction": nan,
        "corrected_mean": nan,
        "corrected_std": nan,
        "interval_lower": nan,
        "interval_upper": nan,
        "transfer_diagnostic": float(transfer),
        "accepted": True,
        "validation_rows": 0,
        "paths_per_model": 0,
        "stages": 0,
        "real_rows_used": 0,
        "real_rows_acquired": 0,
        "simulated_paths_used": 0,
        "pilot_variance_simulation": nan,
        "pilot_variance_real": nan,
        "stage_log": [],
    }
