"""Optional accounting for actual updates and conditional training-accuracy theorems."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Sequence, TypedDict

import torch

from algorithms.ph_performance import (
    require_nonnegative,
    require_positive,
    validate_analysis_context,
)
from common.ph_types import AnalysisContext, BoundEvidence

if TYPE_CHECKING:
    from torch import Tensor

TrainingBlock = Literal["value", "rate", "actor", "model", "temperature"]

# Geometric radii searched by coverage_uniform_bound, as fractions of the largest radius.
_COVERAGE_GRID_POINTS = 4096
_COVERAGE_GRID_DECADES = 12.0
_EVIDENCE_ORDER = ("sampled_diagnostic", "assumed_uniform", "certified_uniform")


def _weakest_evidence(*evidence: BoundEvidence) -> BoundEvidence:
    """A derived bound is only as strong as its weakest input; nothing is promoted."""
    kind = min((item["kind"] for item in evidence), key=_EVIDENCE_ORDER.index)
    probabilities = [item["failure_probability"] for item in evidence]
    return {
        "kind": kind,
        "domain": "; ".join(dict.fromkeys(item["domain"] for item in evidence)),
        "justification": "; ".join(
            dict.fromkeys(item["justification"] for item in evidence)
        ),
        "failure_probability": (
            None if any(p is None for p in probabilities) else min(sum(probabilities), 1.0)
        ),
    }


def _flat_double(tensor: Tensor) -> Tensor:
    return tensor.detach().to(dtype=torch.float64).reshape(-1)


def _increment_norm_sum(steps: Sequence[OptimizerStepTrace]) -> float:
    return sum(
        float(torch.linalg.vector_norm(_flat_double(step["parameter_increment"])))
        for step in steps
    )


def _check_context_field(
    role: str,
    context: AnalysisContext,
    field: Literal["candidate_id", "policy_id", "plant_id"],
    expected: str,
) -> None:
    if context[field] != expected:
        raise ValueError(
            f"The {role} context has {field}={context[field]!r}, expected {expected!r}."
        )


class OptimizerStepTrace(TypedDict):
    """Record the step actually taken, including clipping/momentum and changing replay."""

    context: AnalysisContext
    block: TrainingBlock
    step_index: int
    empirical_risk_id: str
    replay_version: str
    target_version: str
    loss_value: float
    learning_rate: float
    parameter_increment: Tensor
    parameters_before: Optional[Tensor]
    empirical_gradient: Optional[Tensor]
    minibatch_gradient: Tensor
    clipped_gradient: Tensor
    adam_first_moment: Optional[Tensor]
    adam_preconditioner: Optional[Tensor]
    evidence: BoundEvidence


class CoveragePremises(TypedDict):
    """Domain-wide lower ball mass and regularity, supplied independently of replay RMS."""

    evidence: BoundEvidence
    lipschitz_constant: float
    lower_ball_mass: Callable[[float], float]
    maximum_radius: float


class BlockRiskBound(TypedDict):
    """Block-specific floor: rate A_q is normalized; model is chart infimum minus F_or.

    Excess is optimizer excess + twice uniform deviation. Only e_q, not A_q,
    is divided by Z_q; the separate wrapped/chart comparison is added once.
    """

    evidence: BoundEvidence
    optimization_excess: float
    statistical_deviation: float
    approximation_floor: float


class ModelIdentificationPremises(TypedDict):
    """Local identifiable output chart; never invert redundant PH/neural parameters."""

    evidence: BoundEvidence
    covariance_lower: float
    covariance_upper: float
    moment_jacobian_lower: float
    moment_jacobian_lipschitz: float
    coordinate_displacement: float
    generator_output_lipschitz: float
    reference_generator_error: float
    reference_moment_error: float
    wrapped_chart_risk_error: float
    admitted_generator_lag: float


class TrainingAccuracyBounds(TypedDict):
    """Keep rate/actor, raw solve residual and model-error contexts distinct."""

    context: AnalysisContext
    residual_context: AnalysisContext
    model_context: AnalysisContext
    evidence: BoundEvidence
    raw_resolvent_residual: float
    rate_error: float
    actor_gap: float
    model_error: float
    rate_target_bias: float
    admitted_model_lag: float


class OuterStepTrace(TypedDict):
    """Preserve solve-old-policy/new-value and copied-value/new-policy provenance."""

    input_context: AnalysisContext
    solve_output_context: AnalysisContext
    actor_output_context: AnalysisContext
    value_steps: Sequence[OptimizerStepTrace]
    rate_steps: Sequence[OptimizerStepTrace]
    actor_steps: Sequence[OptimizerStepTrace]
    model_steps: Sequence[OptimizerStepTrace]
    temperature_steps: Sequence[OptimizerStepTrace]
    value_increment_norm_sum: float
    actor_increment_norm_sum: float


def validate_training_lineage(
    trace: OuterStepTrace,
    accuracy: TrainingAccuracyBounds,
    *,
    purpose: Literal["post_actor_performance", "original_value_iteration"],
) -> None:
    """Check V_k/pi_k rate fits versus W/pi_plus, raw solve context and admitted-copy identity."""
    if purpose not in ("post_actor_performance", "original_value_iteration"):
        raise ValueError(f"Unknown lineage purpose {purpose!r}.")
    start = trace["input_context"]
    solved = trace["solve_output_context"]
    improved = trace["actor_output_context"]
    rate_actor = accuracy["context"]
    residual = accuracy["residual_context"]
    model = accuracy["model_context"]
    contexts = (start, solved, improved, rate_actor, residual, model)

    for context in contexts:
        validate_analysis_context(context)
    if len({context["outer_iteration"] for context in contexts}) != 1:
        raise ValueError("Every context in one lineage must share its outer iteration.")
    if len({(c["policy_kind"], c["objective"]) for c in contexts}) != 1:
        raise ValueError("Every context must share one policy kind and objective.")

    # The trace itself must be the old-policy solve followed by the actor step.
    plant = solved["plant_id"]
    _check_context_field("input", start, "plant_id", plant)
    _check_context_field("actor output", improved, "plant_id", plant)
    _check_context_field("input", start, "policy_id", solved["policy_id"])
    _check_context_field("actor output", improved, "candidate_id", solved["candidate_id"])
    if start["candidate_id"] == solved["candidate_id"]:
        raise ValueError("The solve must output a value V_(k+1) distinct from V_k.")
    if trace["actor_increment_norm_sum"] > 0.0 and (
        improved["policy_id"] == solved["policy_id"]
    ):
        raise ValueError("A moved actor needs a new policy identity pi_plus.")

    # The solve residual is always the old policy's, at V_(k+1), through the admitted copy.
    _check_context_field("residual", residual, "candidate_id", solved["candidate_id"])
    _check_context_field("residual", residual, "policy_id", solved["policy_id"])
    _check_context_field("residual", residual, "plant_id", plant)
    _check_context_field("model", model, "candidate_id", solved["candidate_id"])
    _check_context_field("model", model, "plant_id", plant)
    _check_context_field("rate/actor", rate_actor, "plant_id", plant)

    if purpose == "original_value_iteration":
        _check_context_field("model", model, "policy_id", solved["policy_id"])
        _check_context_field("rate/actor", rate_actor, "candidate_id", start["candidate_id"])
        _check_context_field("rate/actor", rate_actor, "policy_id", start["policy_id"])
        return

    _check_context_field("model", model, "policy_id", improved["policy_id"])
    _check_context_field("rate/actor", rate_actor, "candidate_id", solved["candidate_id"])
    _check_context_field("rate/actor", rate_actor, "policy_id", improved["policy_id"])
    temperatures = {context["temperature_per_second"] for context in contexts}
    if len(temperatures) != 1:
        raise ValueError(
            "Post-actor performance is stated at fixed temperature; automatic tuning "
            "needs temperature_drift_terms against a fixed target."
        )


class OriginalIterationBudget(TypedDict):
    """One B_k in the original outer-update bound, not a replacement training rule."""

    outer_iteration: int
    evidence: BoundEvidence
    semigroup_consistency: float
    resolvent_consistency: float
    greedy_error: float
    solve_model_error: float
    total: float


class IdealIterationPremises(TypedDict):
    """Every hypothesis of the any-tau exact theorem must be justified externally."""

    evidence: BoundEvidence
    fixed_correct_model_and_temperature: bool
    exact_generator_rate_and_statewise_greediness: bool
    exact_resolvent_steps: bool
    compact_state_domain: bool
    common_classical_generator_boundary_domains: bool
    uniformly_bounded_generator_coefficients: bool
    hamiltonian_continuous_in_c2: bool
    hjb_uniqueness_and_comparison: bool
    initial_value_below_optimum: bool
    initial_hamiltonian_nonnegative: bool
    iterates_precompact_in_c2: bool


def optimizer_descent_defect(
    empirical_gradient: Tensor,
    actual_increment: Tensor,
    *,
    smoothness: float,
    descent_coefficient: float,
) -> float:
    """z_t=[g_t.d_t+(L/2)||d_t||^2+a_t||g_t||^2]_+ for actual Adam/clipped increments."""
    require_nonnegative(smoothness=smoothness, descent_coefficient=descent_coefficient)
    gradient = _flat_double(empirical_gradient)
    increment = _flat_double(actual_increment)
    if gradient.shape != increment.shape:
        raise ValueError("The gradient and increment must share one parameter shape.")
    defect = (
        torch.dot(gradient, increment)
        + 0.5 * smoothness * increment.square().sum()
        + descent_coefficient * gradient.square().sum()
    )
    return max(float(defect), 0.0)


def empirical_objective_drift(
    *, uniform_risk_change: float, infimum_change: float
) -> float:
    """D_t includes replay/target risk drift and the corresponding change of infimum."""
    require_nonnegative(uniform_risk_change=uniform_risk_change)
    if not math.isfinite(infimum_change):
        raise ValueError("The change of the empirical infimum must be finite.")
    return uniform_risk_change + abs(infimum_change)


def adam_descent_bound(
    *,
    learning_rate: float,
    smoothness: float,
    pl_constant: float,
    preconditioner_lower: float,
    preconditioner_upper: float,
    moment_error_bound: float,
) -> tuple[float, float]:
    """Return a=eta*p/4 and z bound only if eta<=p/(4L*Pplus^2) and 2mu*a<=1."""
    require_positive(
        learning_rate=learning_rate,
        smoothness=smoothness,
        pl_constant=pl_constant,
        preconditioner_lower=preconditioner_lower,
        preconditioner_upper=preconditioner_upper,
    )
    require_nonnegative(moment_error_bound=moment_error_bound)
    if preconditioner_lower > preconditioner_upper:
        raise ValueError("The preconditioner bounds must be ordered.")
    if learning_rate > preconditioner_lower / (4.0 * smoothness * preconditioner_upper**2):
        raise ValueError("The Adam step needs eta <= p/(4 L Pplus^2).")
    coefficient = learning_rate * preconditioner_lower / 4.0
    if 2.0 * pl_constant * coefficient > 1.0:
        raise ValueError("The Adam step needs 2 mu a <= 1.")
    defect = (
        3.0
        * learning_rate
        * preconditioner_upper**2
        / (4.0 * preconditioner_lower)
        * moment_error_bound**2
    )
    return coefficient, defect


def momentum_error_bound(
    *,
    normalized_moment_weights: Sequence[float],
    minibatch_errors: Sequence[float],
    clipping_errors: Sequence[float],
    parameter_displacements: Sequence[float],
    gradient_objective_drifts: Sequence[float],
    smoothness: float,
    initial_moment_contribution: float,
) -> float:
    """Bound ||mhat-g|| with sampling, clipping, momentum lag, Gamma_jt and initial moment."""
    lengths = {
        len(normalized_moment_weights),
        len(minibatch_errors),
        len(clipping_errors),
        len(parameter_displacements),
        len(gradient_objective_drifts),
    }
    if len(lengths) != 1:
        raise ValueError("Every momentum-error sequence needs one entry per step.")
    require_nonnegative(
        smoothness=smoothness, initial_moment_contribution=initial_moment_contribution
    )
    weight_sum = math.fsum(normalized_moment_weights)
    if weight_sum > 1.0 + 1e-9 or (
        initial_moment_contribution == 0.0 and abs(weight_sum - 1.0) > 1e-9
    ):
        raise ValueError(
            "Moment weights must sum to one, or below one with an initial-moment term."
        )
    total = initial_moment_contribution
    for weight, minibatch, clipping, displacement, drift in zip(
        normalized_moment_weights,
        minibatch_errors,
        clipping_errors,
        parameter_displacements,
        gradient_objective_drifts,
    ):
        require_nonnegative(
            moment_weight=weight,
            minibatch_error=minibatch,
            clipping_error=clipping,
            parameter_displacement=displacement,
            gradient_objective_drift=drift,
        )
        total += weight * (minibatch + clipping + smoothness * displacement + drift)
    return total


def minibatch_gradient_error_bound(
    *, variance_bound: float, batch_size: int, failure_probability: float
) -> float:
    """Chebyshev sqrt(v/(B*delta)) for justified conditional minibatch-gradient moments."""
    require_nonnegative(variance_bound=variance_bound)
    if batch_size < 1:
        raise ValueError("The minibatch needs at least one row.")
    if not 0.0 < failure_probability <= 1.0:
        raise ValueError("The failure probability must lie in (0, 1].")
    return math.sqrt(variance_bound / (batch_size * failure_probability))


def optimization_excess_bound(
    *,
    initial_excess: float,
    pl_constant: float,
    descent_coefficients: Sequence[float],
    descent_defects: Sequence[float],
    objective_drifts: Sequence[float],
) -> float:
    """Propagate E_{t+1}<=(1-2mu*a_t)E_t+z_t+D_t to the actual finite stopping index."""
    require_nonnegative(initial_excess=initial_excess)
    require_positive(pl_constant=pl_constant)
    if not len(descent_coefficients) == len(descent_defects) == len(objective_drifts):
        raise ValueError("Each actual step needs a coefficient, defect and drift.")
    excess = initial_excess
    for coefficient, defect, drift in zip(
        descent_coefficients, descent_defects, objective_drifts
    ):
        factor = 2.0 * pl_constant * coefficient
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"Each step needs 0 < 2 mu a_t <= 1, got {factor}.")
        require_nonnegative(descent_defect=defect, objective_drift=drift)
        excess = (1.0 - factor) * excess + defect + drift
    return excess


def replay_risk_variance_bound(covariance_envelopes: Tensor) -> float:
    """V_n<=sum_ij c_ij/n^2; independent minibatch draws do not decorrelate stored rows."""
    envelopes = covariance_envelopes.detach().to(dtype=torch.float64)
    if envelopes.ndim != 2 or envelopes.shape[0] != envelopes.shape[1]:
        raise ValueError("Covariance envelopes must form a square [n,n] matrix.")
    if envelopes.shape[0] == 0:
        raise ValueError("Covariance envelopes need at least one replay row.")
    if not torch.isfinite(envelopes).all() or (envelopes < 0.0).any():
        raise ValueError("Covariance envelopes must be finite and nonnegative.")
    return float(envelopes.sum() / envelopes.shape[0] ** 2)


def uniform_risk_deviation_bound(
    *,
    variance_bound: float,
    cover_size: int,
    cover_radius: float,
    mean_loss_lipschitz_envelope: float,
    failure_probability: float,
) -> float:
    """sqrt(2 M(r) V_n/delta)+r h0(1+2/delta); requires a predeclared analysis class."""
    require_nonnegative(
        variance_bound=variance_bound,
        cover_radius=cover_radius,
        mean_loss_lipschitz_envelope=mean_loss_lipschitz_envelope,
    )
    if cover_size < 1:
        raise ValueError("The analysis class needs a cover of at least one point.")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("The failure probability must lie in (0, 1).")
    return math.sqrt(
        2.0 * cover_size * variance_bound / failure_probability
    ) + cover_radius * mean_loss_lipschitz_envelope * (1.0 + 2.0 / failure_probability)


def population_excess_bound(risk: BlockRiskBound) -> float:
    """e_j=mathfrak E_j+2*s_j; a single empirical all-pairs loss need not be positive."""
    require_nonnegative(
        optimization_excess=risk["optimization_excess"],
        statistical_deviation=risk["statistical_deviation"],
    )
    return risk["optimization_excess"] + 2.0 * risk["statistical_deviation"]


def coverage_uniform_bound(
    population_moment: float, *, power: Literal[1, 2], premises: CoveragePremises
) -> float:
    """U_p(z)=inf_{0<r<=r0}{L_f*r+(z/m_nu(r))^(1/p)}; no inferred Gaussian box coverage."""
    if power not in (1, 2):
        raise ValueError("Coverage uses a first or second moment.")
    require_nonnegative(
        population_moment=population_moment,
        lipschitz_constant=premises["lipschitz_constant"],
    )
    require_positive(maximum_radius=premises["maximum_radius"])
    best = math.inf
    for index in range(_COVERAGE_GRID_POINTS):
        fraction = index / (_COVERAGE_GRID_POINTS - 1)
        radius = premises["maximum_radius"] * 10.0 ** (-_COVERAGE_GRID_DECADES * fraction)
        mass = premises["lower_ball_mass"](radius)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"Coverage needs positive ball mass; m({radius})={mass}.")
        value = premises["lipschitz_constant"] * radius + (
            population_moment / mass
        ) ** (1.0 / power)
        best = min(best, value)
    return best


def training_accuracy_bounds(
    context: AnalysisContext,
    *,
    residual_context: AnalysisContext,
    model_context: AnalysisContext,
    value_risk: BlockRiskBound,
    rate_risk: BlockRiskBound,
    actor_risk: BlockRiskBound,
    model_risk: BlockRiskBound,
    coverage: Mapping[Literal["value", "rate", "actor"], CoveragePremises],
    rate_measure_mass: float,
    blended_target_bias: float,
    model_identification: ModelIdentificationPremises,
) -> TrainingAccuracyBounds:
    """Calibrate each frozen block at its stated context with floors, Z_q, B_q and coverage."""
    missing = {"value", "rate", "actor"} - set(coverage)
    if missing:
        raise ValueError(f"Missing coverage premises for {sorted(missing)}.")
    require_positive(rate_measure_mass=rate_measure_mass)
    require_nonnegative(
        blended_target_bias=blended_target_bias,
        value_floor=value_risk["approximation_floor"],
        rate_floor=rate_risk["approximation_floor"],
        actor_floor=actor_risk["approximation_floor"],
    )
    for context_item in (context, residual_context, model_context):
        validate_analysis_context(context_item)

    value_moment = value_risk["approximation_floor"] + population_excess_bound(value_risk)
    rate_root = (
        math.sqrt(
            rate_risk["approximation_floor"]
            + population_excess_bound(rate_risk) / rate_measure_mass
        )
        + blended_target_bias
    )
    actor_moment = actor_risk["approximation_floor"] + population_excess_bound(actor_risk)
    return {
        "context": context,
        "residual_context": residual_context,
        "model_context": model_context,
        "evidence": _weakest_evidence(
            value_risk["evidence"],
            rate_risk["evidence"],
            actor_risk["evidence"],
            model_risk["evidence"],
            coverage["value"]["evidence"],
            coverage["rate"]["evidence"],
            coverage["actor"]["evidence"],
            model_identification["evidence"],
        ),
        "raw_resolvent_residual": coverage_uniform_bound(
            value_moment, power=2, premises=coverage["value"]
        ),
        "rate_error": coverage_uniform_bound(
            rate_root**2, power=2, premises=coverage["rate"]
        ),
        "actor_gap": coverage_uniform_bound(
            actor_moment, power=1, premises=coverage["actor"]
        ),
        "model_error": model_generator_error_bound(model_risk, model_identification),
        "rate_target_bias": blended_target_bias,
        "admitted_model_lag": model_identification["admitted_generator_lag"],
    }


def model_generator_error_bound(
    model_risk: BlockRiskBound, premises: ModelIdentificationPremises
) -> float:
    """a_gen+(2 L_gen/kappa)(C_M sqrt(A_G+e_G)+a_M)+lag; A_G=risk.floor+2*chart_error."""
    lower = premises["covariance_lower"]
    upper = premises["covariance_upper"]
    jacobian_lower = premises["moment_jacobian_lower"]
    jacobian_lipschitz = premises["moment_jacobian_lipschitz"]
    require_positive(
        covariance_lower=lower,
        covariance_upper=upper,
        moment_jacobian_lower=jacobian_lower,
    )
    require_nonnegative(
        moment_jacobian_lipschitz=jacobian_lipschitz,
        coordinate_displacement=premises["coordinate_displacement"],
        generator_output_lipschitz=premises["generator_output_lipschitz"],
        reference_generator_error=premises["reference_generator_error"],
        reference_moment_error=premises["reference_moment_error"],
        wrapped_chart_risk_error=premises["wrapped_chart_risk_error"],
        admitted_generator_lag=premises["admitted_generator_lag"],
        risk_floor=model_risk["approximation_floor"],
    )
    if lower > upper:
        raise ValueError("The endpoint covariance bounds must satisfy a <= b.")
    if premises["coordinate_displacement"] * jacobian_lipschitz > jacobian_lower:
        raise ValueError(
            "Identification is local: the coordinate displacement exceeds kappa/L_id."
        )
    moment_constant = math.sqrt(max(2.0 * upper, 4.0 * upper**4 / lower**2))
    floor = model_risk["approximation_floor"] + 2.0 * premises["wrapped_chart_risk_error"]
    moment_error = (
        moment_constant * math.sqrt(floor + population_excess_bound(model_risk))
        + premises["reference_moment_error"]
    )
    return (
        premises["reference_generator_error"]
        + 2.0 * premises["generator_output_lipschitz"] / jacobian_lower * moment_error
        + premises["admitted_generator_lag"]
    )


def parameter_path_bounds(
    trace: OuterStepTrace,
    *,
    value_parameter_lipschitz: float,
    actor_hamiltonian_lipschitz: float,
) -> tuple[float, float]:
    """D_V=L_V*sum||d_V|| and S_pi=L_H*sum||d_pi|| for the actual warm-start sequence."""
    require_nonnegative(
        value_parameter_lipschitz=value_parameter_lipschitz,
        actor_hamiltonian_lipschitz=actor_hamiltonian_lipschitz,
        value_increment_norm_sum=trace["value_increment_norm_sum"],
        actor_increment_norm_sum=trace["actor_increment_norm_sum"],
    )
    # Step records may be absent; a recorded path can only lengthen the bound.
    value_path = max(
        trace["value_increment_norm_sum"], _increment_norm_sum(trace["value_steps"])
    )
    actor_path = max(
        trace["actor_increment_norm_sum"], _increment_norm_sum(trace["actor_steps"])
    )
    return value_parameter_lipschitz * value_path, actor_hamiltonian_lipschitz * actor_path


def temperature_drift_terms(
    *,
    target_temperature: float,
    solve_temperature: float,
    actor_temperature: float,
    entropy_price_bound: float,
) -> tuple[float, float]:
    """Add H_ent|alpha_solve-alpha_*| to model error and 2H_ent|alpha_actor-alpha_*| to gap."""
    require_nonnegative(
        target_temperature=target_temperature,
        solve_temperature=solve_temperature,
        actor_temperature=actor_temperature,
        entropy_price_bound=entropy_price_bound,
    )
    return (
        entropy_price_bound * abs(solve_temperature - target_temperature),
        2.0 * entropy_price_bound * abs(actor_temperature - target_temperature),
    )


def original_iteration_budget(
    *,
    outer_iteration: int,
    evidence: BoundEvidence,
    discount_rate: float,
    resolvent_step: float,
    semigroup_constant: float,
    semigroup_order: float,
    resolvent_constant: float,
    raw_residual_bound: float,
    input_value_rate_error: float,
    input_value_actor_gap: float,
    output_value_model_error: float,
) -> OriginalIterationBudget:
    """B_k=C_D*tau^p+beta*C_R*tau^2+tau*(zeta+2eps_q)+beta*(rho+tau*m), beta=1/(1+lambda tau)."""
    require_positive(discount_rate=discount_rate, resolvent_step=resolvent_step)
    require_nonnegative(
        semigroup_constant=semigroup_constant,
        resolvent_constant=resolvent_constant,
        raw_residual_bound=raw_residual_bound,
        input_value_rate_error=input_value_rate_error,
        input_value_actor_gap=input_value_actor_gap,
        output_value_model_error=output_value_model_error,
    )
    if not math.isfinite(semigroup_order) or semigroup_order <= 1.0:
        raise ValueError("The semigroup consistency order must exceed one.")
    beta = 1.0 / (1.0 + discount_rate * resolvent_step)
    semigroup = semigroup_constant * resolvent_step**semigroup_order
    resolvent = beta * resolvent_constant * resolvent_step**2
    greedy = resolvent_step * (input_value_actor_gap + 2.0 * input_value_rate_error)
    solve_model = beta * (raw_residual_bound + resolvent_step * output_value_model_error)
    return {
        "outer_iteration": outer_iteration,
        "evidence": evidence,
        "semigroup_consistency": semigroup,
        "resolvent_consistency": resolvent,
        "greedy_error": greedy,
        "solve_model_error": solve_model,
        "total": semigroup + resolvent + greedy + solve_model,
    }


def original_value_error_bound(
    initial_error: float,
    budgets: Sequence[OriginalIterationBudget],
    *,
    discount_rate: float,
    resolvent_step: float,
) -> float:
    """a^k E0+sum_{j<k}a^(k-1-j)B_j, a=exp(-lambda*tau), toward the fixed CT optimum."""
    require_nonnegative(initial_error=initial_error)
    require_positive(discount_rate=discount_rate, resolvent_step=resolvent_step)
    contraction = math.exp(-discount_rate * resolvent_step)
    error = initial_error
    for budget in budgets:
        require_nonnegative(budget_total=budget["total"])
        error = contraction * error + budget["total"]
    return error


def original_value_error_floor(
    *, budget_upper_bound: float, discount_rate: float, resolvent_step: float
) -> float:
    """sup B/(1-exp(-lambda*tau)); refinement also requires rho/tau->0 and other errors->0."""
    require_nonnegative(budget_upper_bound=budget_upper_bound)
    require_positive(discount_rate=discount_rate, resolvent_step=resolvent_step)
    return budget_upper_bound / -math.expm1(-discount_rate * resolvent_step)


def validate_ideal_iteration_premises(premises: IdealIterationPremises) -> None:
    """Validate supplied exact-theorem premises; no finite actor step is presumed greedy."""
    false_premises = [
        name for name, holds in premises.items() if name != "evidence" and not holds
    ]
    if premises["evidence"]["kind"] == "sampled_diagnostic":
        false_premises.append("evidence (sampled diagnostics do not justify the premises)")
    if false_premises:
        raise ValueError(f"The exact-iteration theorem lacks: {false_premises}.")


def fixed_clock_blended_rate_scale(
    *,
    duration: float,
    discount_rate: float,
    data_weight: float,
    rollout_weight: float,
    generator_weight: float,
) -> float:
    """One-state common-law counterexample c_eff in [w_lambda(Delta)/Delta,1); not a correction."""
    require_positive(duration=duration, discount_rate=discount_rate)
    require_nonnegative(
        data_weight=data_weight,
        rollout_weight=rollout_weight,
        generator_weight=generator_weight,
    )
    interval_weight = data_weight + rollout_weight
    total = interval_weight + generator_weight
    if total <= 0.0:
        raise ValueError("The blended rate loss needs a positive target weight.")
    scale = -math.expm1(-discount_rate * duration) / (discount_rate * duration)
    return (interval_weight * scale + generator_weight) / total
