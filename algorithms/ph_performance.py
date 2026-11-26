"""Conditional finite-error bounds from the revised paper; no new training steps."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

import numpy as np

from common.ph_time import clock_duration_pmf
from common.ph_types import AnalysisContext, BoundEvidence

if TYPE_CHECKING:
    from common.ph_config import ClockConfig


class ClockStatistics(TypedDict):
    """Moments of the fixed exogenous clock, with lambda>0 and gamma_Delta<1."""

    contraction: float
    mean_discount_weight: float
    second_duration_moment: float
    discounted_time_moment: float


class PerformancePremises(TypedDict):
    """Uniform errors for one W and policy, including boundary/integrability evidence."""

    context: AnalysisContext
    evidence: BoundEvidence
    discount_rate: float
    clock_contraction: float
    evaluation_residual: float
    rate_error: float
    actor_gap: float
    holding_error: float


class PerformanceBound(TypedDict):
    """Keep conditional/diagnostic conclusions distinct from justified certificates."""

    context: AnalysisContext
    evidence: BoundEvidence
    claim_kind: Literal["diagnostic", "conditional", "certified"]
    policy_loss_upper_bound: float
    soft_return_lower_bound: float
    reward_return_lower_bound: Optional[float]
    task_threshold: Optional[float]
    threshold_satisfied: Optional[bool]


def require_nonnegative(**values: float) -> None:
    """Reject negative or non-finite bound inputs, which make a bound meaningless."""
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value}.")


def require_positive(**values: float) -> None:
    """Reject nonpositive or non-finite rates, steps and scales."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")


def validate_analysis_context(context: AnalysisContext) -> None:
    """Identify one W/pi/model; the mean policy needs the reward-only alpha_c=0 version."""
    for field in ("candidate_id", "policy_id", "plant_id"):
        if not context[field]:
            raise ValueError(f"The analysis context needs a nonempty {field}.")
    require_nonnegative(temperature_per_second=context["temperature_per_second"])
    if context["policy_kind"] == "deterministic_mean" and (
        context["temperature_per_second"] != 0.0 or context["objective"] != "reward_only"
    ):
        raise ValueError(
            "A deterministic mean policy is certified only by the reward-only "
            "version with alpha_c=0."
        )


def clock_statistics(clock: ClockConfig, *, discount_rate: float) -> ClockStatistics:
    """gamma=E[exp(-lambda Delta)], wbar=(1-gamma)/lambda; no sampled proxy implied."""
    require_positive(discount_rate=discount_rate)
    steps, probabilities = clock_duration_pmf(clock)
    durations = steps * clock["physics_dt"]
    scaled = discount_rate * durations
    contraction = float(np.dot(probabilities, np.exp(-scaled)))
    # integral_0^Delta t exp(-lambda t) dt = (1-exp(-x)-x exp(-x))/lambda^2, x=lambda Delta.
    discounted_time = (-np.expm1(-scaled) - scaled * np.exp(-scaled)) / discount_rate**2
    return {
        "contraction": contraction,
        "mean_discount_weight": float(
            np.dot(probabilities, -np.expm1(-scaled) / discount_rate)
        ),
        "second_duration_moment": float(np.dot(probabilities, durations**2)),
        "discounted_time_moment": float(np.dot(probabilities, discounted_time)),
    }


def holding_consistency_bound(
    clock: ClockStatistics,
    *,
    generator_score_increment_bound: float,
    no_killing: bool,
    continuously_refreshed: bool,
    time_homogeneous: bool,
) -> float:
    """H*E[integral t exp(-lambda t)dt] <= H*E[Delta^2]/2, only under lemma hypotheses."""
    missing = [
        name
        for name, holds in (
            ("no_killing", no_killing),
            ("continuously_refreshed", continuously_refreshed),
            ("time_homogeneous", time_homogeneous),
        )
        if not holds
    ]
    if missing:
        raise ValueError(
            f"The holding bound needs its lemma hypotheses; missing {missing}. Failure "
            "boundaries and sampled controllers need a stopped or augmented estimate."
        )
    require_nonnegative(generator_score_increment_bound=generator_score_increment_bound)
    return generator_score_increment_bound * clock["discounted_time_moment"]


def coefficient_model_error_bound(
    *,
    reward_error: float,
    drift_error: float,
    covariance_error: float,
    value_gradient_bound: float,
    value_hessian_bound: float,
) -> float:
    """epsilon_rho + G_W*epsilon_b + .5*D_W*epsilon_A at the same candidate W."""
    require_nonnegative(
        reward_error=reward_error,
        drift_error=drift_error,
        covariance_error=covariance_error,
        value_gradient_bound=value_gradient_bound,
        value_hessian_bound=value_hessian_bound,
    )
    return (
        reward_error
        + value_gradient_bound * drift_error
        + 0.5 * value_hessian_bound * covariance_error
    )


def evaluation_residual_bound(
    *,
    value_change_bound: float,
    raw_resolvent_residual_bound: float,
    resolvent_step: float,
    model_error_bound: float,
) -> float:
    """b_W=(d_V+rho_R)/tau+m_W; rho_R is raw uniform residual, not solver RMS epsilon."""
    require_nonnegative(
        value_change_bound=value_change_bound,
        raw_resolvent_residual_bound=raw_resolvent_residual_bound,
        model_error_bound=model_error_bound,
    )
    require_positive(resolvent_step=resolvent_step)
    return (
        value_change_bound + raw_resolvent_residual_bound
    ) / resolvent_step + model_error_bound


def post_actor_evaluation_residual_bound(
    *,
    value_path_bound: float,
    old_policy_residual_bound: float,
    actor_hamiltonian_motion_bound: float,
    new_policy_model_error_bound: float,
    resolvent_step: float,
) -> float:
    """b_plus=(D_V+rho_R)/tau+S_pi+m_plus(W) after copy--rate--actor, fixed alpha."""
    require_nonnegative(
        value_path_bound=value_path_bound,
        old_policy_residual_bound=old_policy_residual_bound,
        actor_hamiltonian_motion_bound=actor_hamiltonian_motion_bound,
        new_policy_model_error_bound=new_policy_model_error_bound,
    )
    require_positive(resolvent_step=resolvent_step)
    return (
        (value_path_bound + old_policy_residual_bound) / resolvent_step
        + actor_hamiltonian_motion_bound
        + new_policy_model_error_bound
    )


def entropy_reward_correction(
    *,
    temperature_per_second: float,
    discount_rate: float,
    reference_mass: float,
) -> float:
    """c_ent=alpha_c*max(log m(K),0)/lambda for finite positive reference mass.

    m is the measure the policy density is taken against. DecisionActor's density is
    over the normalized box [-1,1]^D, so for it m(K)=2^D with D the decision dimension,
    whatever the box's physical units.
    """
    require_nonnegative(temperature_per_second=temperature_per_second)
    require_positive(discount_rate=discount_rate, reference_mass=reference_mass)
    return temperature_per_second * max(math.log(reference_mass), 0.0) / discount_rate


def validate_performance_premises(
    premises: PerformancePremises,
    *,
    claim_kind: Literal["diagnostic", "conditional", "certified"],
) -> None:
    """Require matched W/pi/model domains and evidence; mean-policy claims need alpha_c=0."""
    validate_analysis_context(premises["context"])
    require_positive(discount_rate=premises["discount_rate"])
    require_nonnegative(
        evaluation_residual=premises["evaluation_residual"],
        rate_error=premises["rate_error"],
        actor_gap=premises["actor_gap"],
        holding_error=premises["holding_error"],
    )
    if not 0.0 <= premises["clock_contraction"] < 1.0:
        raise ValueError("The clock contraction gamma_Delta must lie in [0, 1).")

    evidence = premises["evidence"]
    if claim_kind == "diagnostic":
        return
    if claim_kind == "conditional":
        accepted = ("assumed_uniform", "certified_uniform")
    elif claim_kind == "certified":
        accepted = ("certified_uniform",)
    else:
        raise ValueError(f"Unknown claim kind {claim_kind!r}.")
    if evidence["kind"] not in accepted:
        raise ValueError(
            f"A {claim_kind} claim needs {accepted} evidence, got {evidence['kind']!r}; "
            "sampled maxima are not uniform premises."
        )
    if not evidence["domain"] or not evidence["justification"]:
        raise ValueError(f"A {claim_kind} claim needs a stated domain and justification.")
    if claim_kind == "certified":
        probability = evidence["failure_probability"]
        if probability is None or not 0.0 <= probability < 1.0:
            raise ValueError("A certified claim needs a failure probability in [0, 1).")


def finite_error_performance_bound(
    premises: PerformancePremises,
    *,
    initial_value_expectation: float,
    entropy_correction: Optional[float] = None,
    task_threshold: Optional[float] = None,
    claim_kind: Literal["diagnostic", "conditional", "certified"] = "conditional",
) -> PerformanceBound:
    """Loss <= (zeta+2 eps_q+2 b_W)/lambda+2 d_W/(1-gamma); J>=mu(W)-b/lambda-d/(1-gamma)."""
    validate_performance_premises(premises, claim_kind=claim_kind)
    if not math.isfinite(initial_value_expectation):
        raise ValueError("The initial value expectation must be finite.")
    if entropy_correction is not None:
        require_nonnegative(entropy_correction=entropy_correction)

    context = premises["context"]
    if (
        context["objective"] == "reward_only"
        and context["policy_kind"] == "stochastic"
        and entropy_correction is None
    ):
        raise ValueError(
            "A stochastic reward-only claim needs the entropy correction c_ent."
        )

    rate = premises["discount_rate"]
    gap = 1.0 - premises["clock_contraction"]
    residual = premises["evaluation_residual"]
    holding = premises["holding_error"]
    policy_loss = (
        premises["actor_gap"] + 2.0 * premises["rate_error"] + 2.0 * residual
    ) / rate + 2.0 * holding / gap
    soft_lower = initial_value_expectation - residual / rate - holding / gap

    reward_lower: Optional[float] = None
    if entropy_correction is not None:
        reward_lower = soft_lower - entropy_correction
    elif context["objective"] == "reward_only":
        # The reward-only mean policy has alpha_c=0, so no entropy price remains.
        reward_lower = soft_lower

    threshold_lower = reward_lower if context["objective"] == "reward_only" else soft_lower
    return {
        "context": context,
        "evidence": premises["evidence"],
        "claim_kind": claim_kind,
        "policy_loss_upper_bound": policy_loss,
        "soft_return_lower_bound": soft_lower,
        "reward_return_lower_bound": reward_lower,
        "task_threshold": task_threshold,
        "threshold_satisfied": (
            None if task_threshold is None else bool(threshold_lower >= task_threshold)
        ),
    }


def validate_simultaneous_evidence(
    evidence: tuple[BoundEvidence, ...], *, failure_probability_budget: float
) -> BoundEvidence:
    """Require justified simultaneous uniform events; allocate a union bound, never infer one."""
    if not evidence:
        raise ValueError("Simultaneous evidence needs at least one premise.")
    if not 0.0 <= failure_probability_budget <= 1.0:
        raise ValueError("The failure-probability budget must lie in [0, 1].")

    total = 0.0
    has_probability = False
    for item in evidence:
        if item["kind"] not in ("assumed_uniform", "certified_uniform"):
            raise ValueError(
                f"Premise over {item['domain']!r} has {item['kind']!r} evidence; "
                "sampled diagnostics are never promoted to uniform premises."
            )
        if not item["domain"] or not item["justification"]:
            raise ValueError("Each uniform premise needs a domain and justification.")
        probability = item["failure_probability"]
        if probability is None:
            if item["kind"] == "certified_uniform":
                raise ValueError("A certified premise needs its failure probability.")
            continue
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Failure probabilities must lie in [0, 1].")
        total += probability
        has_probability = True
    if total > failure_probability_budget:
        raise ValueError(
            f"The union bound {total} exceeds the budget {failure_probability_budget}."
        )

    certified = all(item["kind"] == "certified_uniform" for item in evidence)
    return {
        "kind": "certified_uniform" if certified else "assumed_uniform",
        "domain": "; ".join(dict.fromkeys(item["domain"] for item in evidence)),
        "justification": "; ".join(
            dict.fromkeys(item["justification"] for item in evidence)
        ),
        "failure_probability": total if has_probability else None,
    }
