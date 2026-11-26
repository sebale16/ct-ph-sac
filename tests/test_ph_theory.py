import math
import unittest

import numpy as np
import torch as th

from algorithms.ph_performance import (
    clock_statistics,
    coefficient_model_error_bound,
    entropy_reward_correction,
    evaluation_residual_bound,
    finite_error_performance_bound,
    holding_consistency_bound,
    post_actor_evaluation_residual_bound,
    validate_performance_premises,
    validate_simultaneous_evidence,
)
from algorithms.ph_training import (
    adam_descent_bound,
    coverage_uniform_bound,
    empirical_objective_drift,
    fixed_clock_blended_rate_scale,
    minibatch_gradient_error_bound,
    model_generator_error_bound,
    momentum_error_bound,
    optimization_excess_bound,
    optimizer_descent_defect,
    original_iteration_budget,
    original_value_error_bound,
    original_value_error_floor,
    parameter_path_bounds,
    population_excess_bound,
    replay_risk_variance_bound,
    temperature_drift_terms,
    training_accuracy_bounds,
    uniform_risk_deviation_bound,
    validate_ideal_iteration_premises,
    validate_training_lineage,
)


ASSUMED = {
    "kind": "assumed_uniform",
    "domain": "state box",
    "justification": "stated premise",
    "failure_probability": None,
}
CERTIFIED = {
    "kind": "certified_uniform",
    "domain": "state box",
    "justification": "covering argument",
    "failure_probability": 0.01,
}
SAMPLED = {
    "kind": "sampled_diagnostic",
    "domain": "replay states",
    "justification": "sample maximum",
    "failure_probability": None,
}


def make_context(
    candidate="V4",
    policy="pi4",
    plant="P1",
    temperature=0.2,
    kind="stochastic",
    objective="soft",
    outer=3,
):
    return {
        "outer_iteration": outer,
        "candidate_id": candidate,
        "policy_id": policy,
        "plant_id": plant,
        "temperature_per_second": temperature,
        "policy_kind": kind,
        "objective": objective,
    }


def make_premises(context=None, evidence=ASSUMED):
    return {
        "context": context or make_context(),
        "evidence": evidence,
        "discount_rate": 0.5,
        "clock_contraction": 0.9,
        "evaluation_residual": 0.1,
        "rate_error": 0.2,
        "actor_gap": 0.3,
        "holding_error": 0.01,
    }


def make_trace(actor_motion=1.0):
    return {
        "input_context": make_context(candidate="V3", policy="pi3"),
        "solve_output_context": make_context(candidate="V4", policy="pi3"),
        "actor_output_context": make_context(candidate="V4", policy="pi4"),
        "value_steps": (),
        "rate_steps": (),
        "actor_steps": (),
        "model_steps": (),
        "temperature_steps": (),
        "value_increment_norm_sum": 2.0,
        "actor_increment_norm_sum": actor_motion,
    }


def make_accuracy(context, residual, model):
    return {
        "context": context,
        "residual_context": residual,
        "model_context": model,
        "evidence": ASSUMED,
        "raw_resolvent_residual": 0.1,
        "rate_error": 0.1,
        "actor_gap": 0.1,
        "model_error": 0.1,
        "rate_target_bias": 0.0,
        "admitted_model_lag": 0.0,
    }


class TestClockStatistics(unittest.TestCase):
    def test_fixed_clock_matches_closed_form(self):
        clock = {
            "kind": "fixed",
            "physics_dt": 0.001,
            "reference_period": 0.01,
            "min_duration": 0.01,
            "max_duration": 0.01,
            "tail_probability": None,
            "tail_split": None,
            "beta_shape": None,
        }
        stats = clock_statistics(clock, discount_rate=0.5)
        gamma = math.exp(-0.005)
        self.assertAlmostEqual(stats["contraction"], gamma, places=14)
        self.assertAlmostEqual(stats["mean_discount_weight"], (1 - gamma) / 0.5, places=14)
        self.assertAlmostEqual(stats["second_duration_moment"], 1e-4, places=14)
        # A fine midpoint rule for the integral of t*exp(-lambda t) on [0, Delta].
        t = (np.arange(200000) + 0.5) * 0.01 / 200000
        integral = float(np.sum(t * np.exp(-0.5 * t)) * 0.01 / 200000)
        self.assertAlmostEqual(stats["discounted_time_moment"], integral, places=12)

    def test_two_tail_clock_moments(self):
        clock = {
            "kind": "two_tail_uniform",
            "physics_dt": 0.001,
            "reference_period": 0.01,
            "min_duration": 0.002,
            "max_duration": 0.018,
            "tail_probability": 0.8,
            "tail_split": 0.5,
            "beta_shape": None,
        }
        stats = clock_statistics(clock, discount_rate=0.1)
        durations = np.arange(2, 19) * 0.001
        weights = np.full(17, 0.2 / 15)
        weights[[0, -1]] = 0.4
        gamma = float(np.dot(weights, np.exp(-0.1 * durations)))
        self.assertAlmostEqual(stats["contraction"], gamma, places=13)
        self.assertAlmostEqual(
            stats["second_duration_moment"], float(np.dot(weights, durations**2)), places=13
        )
        self.assertAlmostEqual(stats["mean_discount_weight"], (1 - gamma) / 0.1, places=12)
        # The lemma's second inequality.
        self.assertLessEqual(
            stats["discounted_time_moment"], stats["second_duration_moment"] / 2
        )


class TestHoldingAndResiduals(unittest.TestCase):
    def setUp(self):
        self.clock = {
            "contraction": 0.99,
            "mean_discount_weight": 0.1,
            "second_duration_moment": 4e-4,
            "discounted_time_moment": 1.5e-4,
        }

    def test_holding_bound_value(self):
        bound = holding_consistency_bound(
            self.clock,
            generator_score_increment_bound=8.0,
            no_killing=True,
            continuously_refreshed=True,
            time_homogeneous=True,
        )
        self.assertAlmostEqual(bound, 1.2e-3)

    def test_holding_bound_requires_each_hypothesis(self):
        for flags in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.assertRaises(ValueError):
                holding_consistency_bound(
                    self.clock,
                    generator_score_increment_bound=8.0,
                    no_killing=flags[0],
                    continuously_refreshed=flags[1],
                    time_homogeneous=flags[2],
                )

    def test_coefficient_error_value(self):
        bound = coefficient_model_error_bound(
            reward_error=0.1,
            drift_error=0.2,
            covariance_error=0.4,
            value_gradient_bound=3.0,
            value_hessian_bound=5.0,
        )
        self.assertAlmostEqual(bound, 0.1 + 0.6 + 1.0)

    def test_evaluation_residual_value(self):
        bound = evaluation_residual_bound(
            value_change_bound=0.3,
            raw_resolvent_residual_bound=0.1,
            resolvent_step=2.0,
            model_error_bound=0.05,
        )
        self.assertAlmostEqual(bound, 0.25)

    def test_post_actor_residual_value(self):
        bound = post_actor_evaluation_residual_bound(
            value_path_bound=0.3,
            old_policy_residual_bound=0.1,
            actor_hamiltonian_motion_bound=0.2,
            new_policy_model_error_bound=0.05,
            resolvent_step=2.0,
        )
        self.assertAlmostEqual(bound, 0.45)

    def test_entropy_correction_value(self):
        value = entropy_reward_correction(
            temperature_per_second=0.2, discount_rate=0.5, reference_mass=math.e**2
        )
        self.assertAlmostEqual(value, 0.8)
        # A probability reference measure needs no correction.
        self.assertEqual(
            entropy_reward_correction(
                temperature_per_second=0.2, discount_rate=0.5, reference_mass=1.0
            ),
            0.0,
        )

    def test_meaningless_inputs_raise(self):
        with self.assertRaises(ValueError):
            evaluation_residual_bound(
                value_change_bound=-0.3,
                raw_resolvent_residual_bound=0.1,
                resolvent_step=2.0,
                model_error_bound=0.05,
            )
        with self.assertRaises(ValueError):
            evaluation_residual_bound(
                value_change_bound=0.3,
                raw_resolvent_residual_bound=math.nan,
                resolvent_step=2.0,
                model_error_bound=0.05,
            )
        with self.assertRaises(ValueError):
            entropy_reward_correction(
                temperature_per_second=0.2, discount_rate=0.5, reference_mass=0.0
            )


class TestPerformanceBound(unittest.TestCase):
    def test_policy_and_soft_return_values(self):
        bound = finite_error_performance_bound(
            make_premises(), initial_value_expectation=-2.0
        )
        # (0.3 + 0.4 + 0.2) / 0.5 + 0.02 / 0.1 = 1.8 + 0.2
        self.assertAlmostEqual(bound["policy_loss_upper_bound"], 2.0)
        # -2 - 0.1 / 0.5 - 0.01 / 0.1 = -2.3
        self.assertAlmostEqual(bound["soft_return_lower_bound"], -2.3)
        self.assertIsNone(bound["reward_return_lower_bound"])
        self.assertIsNone(bound["threshold_satisfied"])
        self.assertEqual(bound["claim_kind"], "conditional")

    def test_soft_threshold(self):
        passing = finite_error_performance_bound(
            make_premises(), initial_value_expectation=-2.0, task_threshold=-2.31
        )
        failing = finite_error_performance_bound(
            make_premises(), initial_value_expectation=-2.0, task_threshold=-2.2
        )
        self.assertTrue(passing["threshold_satisfied"])
        self.assertFalse(failing["threshold_satisfied"])

    def test_stochastic_reward_only_needs_entropy_correction(self):
        premises = make_premises(make_context(objective="reward_only"))
        with self.assertRaises(ValueError):
            finite_error_performance_bound(premises, initial_value_expectation=-2.0)
        bound = finite_error_performance_bound(
            premises,
            initial_value_expectation=-2.0,
            entropy_correction=0.5,
            task_threshold=-2.9,
        )
        self.assertAlmostEqual(bound["reward_return_lower_bound"], -2.8)
        self.assertTrue(bound["threshold_satisfied"])

    def test_mean_policy_needs_reward_only_zero_temperature(self):
        for temperature, objective in ((0.2, "reward_only"), (0.0, "soft")):
            premises = make_premises(
                make_context(
                    temperature=temperature, kind="deterministic_mean", objective=objective
                )
            )
            for claim in ("diagnostic", "conditional"):
                with self.assertRaises(ValueError):
                    validate_performance_premises(premises, claim_kind=claim)
        premises = make_premises(
            make_context(temperature=0.0, kind="deterministic_mean", objective="reward_only")
        )
        bound = finite_error_performance_bound(premises, initial_value_expectation=-2.0)
        self.assertAlmostEqual(bound["reward_return_lower_bound"], -2.3)

    def test_claims_require_matching_evidence(self):
        with self.assertRaises(ValueError):
            validate_performance_premises(
                make_premises(evidence=SAMPLED), claim_kind="conditional"
            )
        validate_performance_premises(
            make_premises(evidence=SAMPLED), claim_kind="diagnostic"
        )
        validate_performance_premises(make_premises(), claim_kind="conditional")
        with self.assertRaises(ValueError):
            validate_performance_premises(make_premises(), claim_kind="certified")
        validate_performance_premises(
            make_premises(evidence=CERTIFIED), claim_kind="certified"
        )
        for field, value in (("failure_probability", None), ("justification", "")):
            weak = {**CERTIFIED, field: value}
            with self.assertRaises(ValueError):
                validate_performance_premises(
                    make_premises(evidence=weak), claim_kind="certified"
                )

    def test_bound_inputs_are_checked(self):
        for field, value in (
            ("clock_contraction", 1.0),
            ("discount_rate", 0.0),
            ("rate_error", -1.0),
            ("holding_error", math.inf),
        ):
            with self.assertRaises(ValueError):
                validate_performance_premises(
                    {**make_premises(), field: value}, claim_kind="diagnostic"
                )

    def test_certificate_on_exact_interval_problem(self):
        """Corollary performance on a two-state fixed-interval kernel, solved exactly."""
        delta = 0.1
        rate = 0.5
        gamma = math.exp(-rate * delta)
        w = (1.0 - gamma) / rate
        # Two decisions with exact interval rewards and kernels; alpha_c = 0.
        rewards = np.array([[1.0, 0.5], [0.0, 0.2]]) * w
        kernels = np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.5, 0.5], [0.1, 0.9]],
            ]
        )

        def backup(values, policy):
            return np.array(
                [
                    rewards[s, policy[s]] + gamma * kernels[s, policy[s]] @ values
                    for s in range(2)
                ]
            )

        def policy_value(policy):
            kernel = np.array([kernels[s, policy[s]] for s in range(2)])
            reward = np.array([rewards[s, policy[s]] for s in range(2)])
            return np.linalg.solve(np.eye(2) - gamma * kernel, reward)

        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        optimal = np.max([policy_value(p) for p in policies], axis=0)
        policy = (1, 1)
        value = policy_value(policy)
        candidate = value + np.array([0.3, -0.2])

        # Defining the score as the interval rate makes d_W zero by construction.
        scores = np.array(
            [
                [(backup(candidate, (a, a))[s] - candidate[s]) / w for a in range(2)]
                for s in range(2)
            ]
        )
        residual = float(np.max(np.abs(scores[np.arange(2), policy])))
        gap = float(np.max(scores.max(axis=1) - scores[np.arange(2), policy]))
        premises = {
            **make_premises(
                make_context(temperature=0.0, kind="deterministic_mean", objective="reward_only")
            ),
            "discount_rate": rate,
            "clock_contraction": gamma,
            "evaluation_residual": residual,
            "rate_error": 0.0,
            "actor_gap": gap,
            "holding_error": 0.0,
        }
        mu = np.array([0.5, 0.5])
        bound = finite_error_performance_bound(
            premises, initial_value_expectation=float(mu @ candidate)
        )
        self.assertLessEqual(
            float(np.max(optimal - value)), bound["policy_loss_upper_bound"] + 1e-12
        )
        self.assertGreaterEqual(
            float(mu @ value), bound["soft_return_lower_bound"] - 1e-12
        )


class TestSimultaneousEvidence(unittest.TestCase):
    def test_union_bound_and_kind(self):
        second = {**CERTIFIED, "domain": "decision box", "failure_probability": 0.02}
        result = validate_simultaneous_evidence(
            (CERTIFIED, second), failure_probability_budget=0.05
        )
        self.assertEqual(result["kind"], "certified_uniform")
        self.assertAlmostEqual(result["failure_probability"], 0.03)
        self.assertIn("decision box", result["domain"])
        with self.assertRaises(ValueError):
            validate_simultaneous_evidence(
                (CERTIFIED, second), failure_probability_budget=0.02
            )

    def test_mixed_evidence_stays_assumed(self):
        result = validate_simultaneous_evidence(
            (CERTIFIED, ASSUMED), failure_probability_budget=0.05
        )
        self.assertEqual(result["kind"], "assumed_uniform")

    def test_sampled_is_never_promoted(self):
        with self.assertRaises(ValueError):
            validate_simultaneous_evidence(
                (CERTIFIED, SAMPLED), failure_probability_budget=1.0
            )
        with self.assertRaises(ValueError):
            validate_simultaneous_evidence(
                ({**CERTIFIED, "failure_probability": None},),
                failure_probability_budget=1.0,
            )
        with self.assertRaises(ValueError):
            validate_simultaneous_evidence((), failure_probability_budget=1.0)


class TestOptimizerBounds(unittest.TestCase):
    def test_descent_defect_values(self):
        g = th.tensor([1.0, 2.0])
        d = th.tensor([-0.1, 0.05])
        # g.d = 0, (L/2)||d||^2 = 2 * 0.0125, a||g||^2 = 0.01 * 5
        self.assertAlmostEqual(
            optimizer_descent_defect(g, d, smoothness=4.0, descent_coefficient=0.01), 0.075
        )
        # A strong descent direction clips to zero.
        self.assertEqual(
            optimizer_descent_defect(
                g, -0.1 * g, smoothness=1.0, descent_coefficient=0.01
            ),
            0.0,
        )
        with self.assertRaises(ValueError):
            optimizer_descent_defect(
                g, th.zeros(3), smoothness=1.0, descent_coefficient=0.01
            )

    def test_drift_value(self):
        self.assertAlmostEqual(
            empirical_objective_drift(uniform_risk_change=0.1, infimum_change=-0.2), 0.3
        )

    def test_adam_bound_values(self):
        a, z = adam_descent_bound(
            learning_rate=0.01,
            smoothness=2.0,
            pl_constant=1.0,
            preconditioner_lower=1.0,
            preconditioner_upper=2.0,
            moment_error_bound=0.5,
        )
        self.assertAlmostEqual(a, 0.0025)
        self.assertAlmostEqual(z, 3.0 * 0.01 * 4.0 / 4.0 * 0.25)

    def test_adam_bound_guards(self):
        common = {"smoothness": 2.0, "moment_error_bound": 0.5}
        # eta > p / (4 L Pplus^2) = 1 / 32
        with self.assertRaises(ValueError):
            adam_descent_bound(
                learning_rate=0.04,
                pl_constant=1.0,
                preconditioner_lower=1.0,
                preconditioner_upper=2.0,
                **common,
            )
        # 2 mu a > 1
        with self.assertRaises(ValueError):
            adam_descent_bound(
                learning_rate=0.01,
                pl_constant=1000.0,
                preconditioner_lower=1.0,
                preconditioner_upper=2.0,
                **common,
            )

    def test_momentum_error_value(self):
        bound = momentum_error_bound(
            normalized_moment_weights=[0.25, 0.75],
            minibatch_errors=[0.1, 0.2],
            clipping_errors=[0.0, 0.1],
            parameter_displacements=[1.0, 0.0],
            gradient_objective_drifts=[0.3, 0.1],
            smoothness=2.0,
            initial_moment_contribution=0.0,
        )
        self.assertAlmostEqual(bound, 0.25 * 2.4 + 0.75 * 0.4)
        with self.assertRaises(ValueError):
            momentum_error_bound(
                normalized_moment_weights=[0.25, 0.25],
                minibatch_errors=[0.1, 0.2],
                clipping_errors=[0.0, 0.1],
                parameter_displacements=[1.0, 0.0],
                gradient_objective_drifts=[0.3, 0.1],
                smoothness=2.0,
                initial_moment_contribution=0.0,
            )

    def test_minibatch_error_value(self):
        self.assertAlmostEqual(
            minibatch_gradient_error_bound(
                variance_bound=4.0, batch_size=16, failure_probability=0.25
            ),
            1.0,
        )

    def test_excess_recursion_value(self):
        # mu=1, a=0.25: E1=0.5*2+0.1+0.2=1.3; E2=0.5*1.3+0+0.05=0.7
        value = optimization_excess_bound(
            initial_excess=2.0,
            pl_constant=1.0,
            descent_coefficients=[0.25, 0.25],
            descent_defects=[0.1, 0.0],
            objective_drifts=[0.2, 0.05],
        )
        self.assertAlmostEqual(value, 0.7)
        with self.assertRaises(ValueError):
            optimization_excess_bound(
                initial_excess=2.0,
                pl_constant=1.0,
                descent_coefficients=[0.75],
                descent_defects=[0.1],
                objective_drifts=[0.0],
            )

    def test_excess_bounds_actual_adam_on_quadratic(self):
        """The pathwise bound holds for recorded Adam increments and changing offsets."""
        hessian = th.diag(th.tensor([1.0, 4.0], dtype=th.float64))
        mu, smoothness = 1.0, 4.0
        coefficient = 1e-3
        theta = th.nn.Parameter(th.tensor([2.0, -1.5], dtype=th.float64))
        optimizer = th.optim.Adam([theta], lr=0.01)

        def excess(values):
            return float(0.5 * values @ hessian @ values)

        initial = excess(theta.detach())
        coefficients, defects, drifts = [], [], []
        descended = False
        for step in range(400):
            before = theta.detach().clone()
            optimizer.zero_grad()
            loss = 0.5 * theta @ hessian @ theta + 0.1 * step
            loss.backward()
            gradient = theta.grad.detach().clone()
            optimizer.step()
            increment = theta.detach() - before
            coefficients.append(coefficient)
            defects.append(
                optimizer_descent_defect(
                    gradient, increment, smoothness=smoothness, descent_coefficient=coefficient
                )
            )
            # The offset moves risk and infimum together by 0.1 per step.
            drifts.append(empirical_objective_drift(uniform_risk_change=0.1, infimum_change=0.1))
            bound = optimization_excess_bound(
                initial_excess=initial,
                pl_constant=mu,
                descent_coefficients=coefficients,
                descent_defects=defects,
                objective_drifts=drifts,
            )
            self.assertLessEqual(excess(theta.detach()), bound + 1e-12)
            descended = descended or excess(theta.detach()) < initial
        self.assertTrue(descended)

        # Without the offset the same trace gives a tight-enough finite bound.
        bound = optimization_excess_bound(
            initial_excess=initial,
            pl_constant=mu,
            descent_coefficients=coefficients,
            descent_defects=defects,
            objective_drifts=[0.0] * len(defects),
        )
        self.assertLessEqual(excess(theta.detach()), bound + 1e-12)
        self.assertLess(bound, initial)


class TestStatistics(unittest.TestCase):
    def test_variance_bound_value(self):
        envelopes = th.tensor([[1.0, 0.5], [0.5, 2.0]])
        self.assertAlmostEqual(replay_risk_variance_bound(envelopes), 1.0)
        self.assertAlmostEqual(
            replay_risk_variance_bound(th.eye(100) * 3.0), 3.0 / 100, places=6
        )
        with self.assertRaises(ValueError):
            replay_risk_variance_bound(th.ones(2, 3))
        with self.assertRaises(ValueError):
            replay_risk_variance_bound(-th.ones(2, 2))

    def test_uniform_deviation_value(self):
        value = uniform_risk_deviation_bound(
            variance_bound=0.01,
            cover_size=50,
            cover_radius=0.1,
            mean_loss_lipschitz_envelope=2.0,
            failure_probability=0.5,
        )
        self.assertAlmostEqual(value, math.sqrt(2.0) + 0.1 * 2.0 * 5.0)

    def test_population_excess_value(self):
        risk = {
            "evidence": ASSUMED,
            "optimization_excess": 0.2,
            "statistical_deviation": 0.05,
            "approximation_floor": 1.0,
        }
        self.assertAlmostEqual(population_excess_bound(risk), 0.3)

    def test_coverage_matches_analytic_infimum(self):
        lipschitz, moment = 3.0, 0.04
        premises = {
            "evidence": ASSUMED,
            "lipschitz_constant": lipschitz,
            "lower_ball_mass": lambda r: r * r,
            "maximum_radius": 10.0,
        }
        # p=2: L r + sqrt(z) / r; p=1: L r + z / r^2.
        analytic_two = 2.0 * math.sqrt(lipschitz * math.sqrt(moment))
        value = coverage_uniform_bound(moment, power=2, premises=premises)
        self.assertGreaterEqual(value, analytic_two)
        self.assertLess(value, analytic_two * (1 + 1e-3))
        r = (2.0 * moment / lipschitz) ** (1.0 / 3.0)
        analytic_one = lipschitz * r + moment / r**2
        value = coverage_uniform_bound(moment, power=1, premises=premises)
        self.assertGreaterEqual(value, analytic_one)
        self.assertLess(value, analytic_one * (1 + 1e-3))

    def test_coverage_binds_at_largest_radius(self):
        premises = {
            "evidence": ASSUMED,
            "lipschitz_constant": 0.001,
            "lower_ball_mass": lambda r: r * r,
            "maximum_radius": 0.01,
        }
        value = coverage_uniform_bound(1.0, power=2, premises=premises)
        self.assertAlmostEqual(value, 0.001 * 0.01 + 100.0, places=9)

    def test_coverage_requires_positive_mass(self):
        premises = {
            "evidence": ASSUMED,
            "lipschitz_constant": 1.0,
            "lower_ball_mass": lambda r: 0.0,
            "maximum_radius": 1.0,
        }
        with self.assertRaises(ValueError):
            coverage_uniform_bound(1.0, power=2, premises=premises)
        with self.assertRaises(ValueError):
            coverage_uniform_bound(1.0, power=3, premises={**premises, "lower_ball_mass": lambda r: 1.0})


def make_identification(**overrides):
    premises = {
        "evidence": ASSUMED,
        "covariance_lower": 1.0,
        "covariance_upper": 2.0,
        "moment_jacobian_lower": 0.5,
        "moment_jacobian_lipschitz": 1.0,
        "coordinate_displacement": 0.25,
        "generator_output_lipschitz": 3.0,
        "reference_generator_error": 0.1,
        "reference_moment_error": 0.2,
        "wrapped_chart_risk_error": 0.002,
        "admitted_generator_lag": 0.05,
    }
    premises.update(overrides)
    return premises


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.model_risk = {
            "evidence": ASSUMED,
            "optimization_excess": 0.004,
            "statistical_deviation": 0.001,
            "approximation_floor": 0.01,
        }

    def test_model_error_value(self):
        # C_M=sqrt(max(4, 64))=8; A_G=0.014; e_G=0.006; sqrt(0.02).
        value = model_generator_error_bound(self.model_risk, make_identification())
        expected = 0.1 + 2.0 * 3.0 / 0.5 * (8.0 * math.sqrt(0.02) + 0.2) + 0.05
        self.assertAlmostEqual(value, expected)

    def test_model_error_constant_branch(self):
        # With a=b the mean term sets C_M when 2b exceeds 4b^2.
        premises = make_identification(covariance_lower=0.25, covariance_upper=0.25)
        value = model_generator_error_bound(self.model_risk, premises)
        expected = 0.1 + 12.0 * (math.sqrt(0.5) * math.sqrt(0.02) + 0.2) + 0.05
        self.assertAlmostEqual(value, expected)

    def test_identification_is_local(self):
        with self.assertRaises(ValueError):
            model_generator_error_bound(
                self.model_risk, make_identification(coordinate_displacement=0.6)
            )
        with self.assertRaises(ValueError):
            model_generator_error_bound(
                self.model_risk, make_identification(covariance_lower=3.0)
            )

    def test_accuracy_values_and_weakest_evidence(self):
        def risk(floor, optimization, statistical, evidence=ASSUMED):
            return {
                "evidence": evidence,
                "optimization_excess": optimization,
                "statistical_deviation": statistical,
                "approximation_floor": floor,
            }

        # Constant lower mass 1/4 and zero Lipschitz constant make U_p exact.
        coverage = {
            "evidence": ASSUMED,
            "lipschitz_constant": 0.0,
            "lower_ball_mass": lambda r: 0.25,
            "maximum_radius": 1.0,
        }
        context = make_context()
        residual = make_context(policy="pi3")
        model = make_context(plant="P1")
        accuracy = training_accuracy_bounds(
            context,
            residual_context=residual,
            model_context=model,
            value_risk=risk(0.01, 0.02, 0.005),
            rate_risk=risk(0.01, 0.06, 0.02),
            actor_risk=risk(0.02, 0.01, 0.01),
            model_risk=self.model_risk,
            coverage={"value": coverage, "rate": coverage, "actor": coverage},
            rate_measure_mass=2.0,
            blended_target_bias=0.1,
            model_identification=make_identification(),
        )
        # value: sqrt((0.01+0.03)/0.25)=0.4
        self.assertAlmostEqual(accuracy["raw_resolvent_residual"], 0.4)
        # rate: (sqrt(0.01+0.1/2)+0.1)/0.5; only e_q is divided by Z_q.
        self.assertAlmostEqual(accuracy["rate_error"], (math.sqrt(0.06) + 0.1) / 0.5)
        # actor: (0.02+0.03)/0.25
        self.assertAlmostEqual(accuracy["actor_gap"], 0.2)
        self.assertAlmostEqual(
            accuracy["model_error"],
            model_generator_error_bound(self.model_risk, make_identification()),
        )
        self.assertEqual(accuracy["rate_target_bias"], 0.1)
        self.assertEqual(accuracy["admitted_model_lag"], 0.05)
        self.assertEqual(accuracy["evidence"]["kind"], "assumed_uniform")
        self.assertIs(accuracy["residual_context"], residual)

        sampled = training_accuracy_bounds(
            context,
            residual_context=residual,
            model_context=model,
            value_risk=risk(0.01, 0.02, 0.005, SAMPLED),
            rate_risk=risk(0.01, 0.06, 0.02),
            actor_risk=risk(0.02, 0.01, 0.01),
            model_risk=self.model_risk,
            coverage={"value": coverage, "rate": coverage, "actor": coverage},
            rate_measure_mass=2.0,
            blended_target_bias=0.1,
            model_identification=make_identification(),
        )
        self.assertEqual(sampled["evidence"]["kind"], "sampled_diagnostic")

        with self.assertRaises(ValueError):
            training_accuracy_bounds(
                context,
                residual_context=residual,
                model_context=model,
                value_risk=risk(0.01, 0.02, 0.005),
                rate_risk=risk(0.01, 0.06, 0.02),
                actor_risk=risk(0.02, 0.01, 0.01),
                model_risk=self.model_risk,
                coverage={"value": coverage, "rate": coverage},
                rate_measure_mass=2.0,
                blended_target_bias=0.1,
                model_identification=make_identification(),
            )


class TestLineage(unittest.TestCase):
    def test_original_iteration_lineage(self):
        trace = make_trace()
        good = make_accuracy(
            trace["input_context"],
            trace["solve_output_context"],
            trace["solve_output_context"],
        )
        validate_training_lineage(trace, good, purpose="original_value_iteration")

        for bad in (
            # Rate/actor must be on V_k, pi_k.
            make_accuracy(
                trace["actor_output_context"],
                trace["solve_output_context"],
                trace["solve_output_context"],
            ),
            # The model error refers to V_(k+1), not V_k.
            make_accuracy(
                trace["input_context"],
                trace["solve_output_context"],
                trace["input_context"],
            ),
            # The solve residual is the old policy's.
            make_accuracy(
                trace["input_context"],
                trace["actor_output_context"],
                trace["solve_output_context"],
            ),
            # The solve's model error reads the admitted copy, not the online plant.
            make_accuracy(
                trace["input_context"],
                trace["solve_output_context"],
                make_context(candidate="V4", policy="pi3", plant="online"),
            ),
        ):
            with self.assertRaises(ValueError):
                validate_training_lineage(trace, bad, purpose="original_value_iteration")

    def test_post_actor_lineage(self):
        trace = make_trace()
        good = make_accuracy(
            trace["actor_output_context"],
            trace["solve_output_context"],
            trace["actor_output_context"],
        )
        validate_training_lineage(trace, good, purpose="post_actor_performance")

        for bad in (
            # Reusing V_k, pi_k for rate and actor.
            make_accuracy(
                trace["input_context"],
                trace["solve_output_context"],
                trace["actor_output_context"],
            ),
            # The candidate must be W=V_(k+1).
            make_accuracy(
                make_context(candidate="V3", policy="pi4"),
                trace["solve_output_context"],
                trace["actor_output_context"],
            ),
            # m_plus belongs to pi_plus.
            make_accuracy(
                trace["actor_output_context"],
                trace["solve_output_context"],
                trace["solve_output_context"],
            ),
        ):
            with self.assertRaises(ValueError):
                validate_training_lineage(trace, bad, purpose="post_actor_performance")

    def test_post_actor_requires_fixed_temperature(self):
        trace = make_trace()
        trace["actor_output_context"] = make_context(candidate="V4", temperature=0.3)
        accuracy = make_accuracy(
            trace["actor_output_context"],
            trace["solve_output_context"],
            trace["actor_output_context"],
        )
        with self.assertRaises(ValueError):
            validate_training_lineage(trace, accuracy, purpose="post_actor_performance")

    def test_trace_itself_must_be_consistent(self):
        # A moved actor cannot keep its policy identity.
        trace = make_trace()
        trace["actor_output_context"] = make_context(candidate="V4", policy="pi3")
        accuracy = make_accuracy(
            trace["input_context"],
            trace["solve_output_context"],
            trace["solve_output_context"],
        )
        with self.assertRaises(ValueError):
            validate_training_lineage(trace, accuracy, purpose="original_value_iteration")
        # An admission mid-step would break the admitted-copy identity.
        trace = make_trace()
        trace["input_context"] = make_context(candidate="V3", policy="pi3", plant="P0")
        with self.assertRaises(ValueError):
            validate_training_lineage(
                trace,
                make_accuracy(
                    trace["input_context"],
                    trace["solve_output_context"],
                    trace["solve_output_context"],
                ),
                purpose="original_value_iteration",
            )
        # Differing outer indices are rejected.
        trace = make_trace()
        trace["solve_output_context"] = make_context(candidate="V4", policy="pi3", outer=4)
        with self.assertRaises(ValueError):
            validate_training_lineage(
                trace,
                make_accuracy(
                    trace["input_context"],
                    trace["solve_output_context"],
                    trace["solve_output_context"],
                ),
                purpose="original_value_iteration",
            )
        with self.assertRaises(ValueError):
            validate_training_lineage(make_trace(), make_accuracy(*[make_context()] * 3), purpose="other")


class TestPathsAndTemperature(unittest.TestCase):
    def test_path_bound_values(self):
        self.assertEqual(
            parameter_path_bounds(
                make_trace(), value_parameter_lipschitz=3.0, actor_hamiltonian_lipschitz=0.5
            ),
            (6.0, 0.5),
        )

    def test_recorded_steps_only_lengthen_paths(self):
        trace = make_trace()
        trace["value_steps"] = (
            {"parameter_increment": th.tensor([3.0, 4.0])},
            {"parameter_increment": th.tensor([0.0, 5.0])},
        )
        value_path, _ = parameter_path_bounds(
            trace, value_parameter_lipschitz=1.0, actor_hamiltonian_lipschitz=1.0
        )
        self.assertAlmostEqual(value_path, 10.0)

    def test_temperature_drift_values(self):
        solve, actor = temperature_drift_terms(
            target_temperature=0.2,
            solve_temperature=0.3,
            actor_temperature=0.25,
            entropy_price_bound=2.0,
        )
        self.assertAlmostEqual(solve, 0.2)
        self.assertAlmostEqual(actor, 0.2)


class TestOriginalIteration(unittest.TestCase):
    def budget(self, index=0, rate=0.5, step=2.0):
        return original_iteration_budget(
            outer_iteration=index,
            evidence=ASSUMED,
            discount_rate=rate,
            resolvent_step=step,
            semigroup_constant=0.01,
            semigroup_order=2.0,
            resolvent_constant=0.02,
            raw_residual_bound=0.03,
            input_value_rate_error=0.04,
            input_value_actor_gap=0.05,
            output_value_model_error=0.06,
        )

    def test_budget_values(self):
        # lambda*tau=1, beta=1/2.
        budget = self.budget()
        self.assertAlmostEqual(budget["semigroup_consistency"], 0.04)
        self.assertAlmostEqual(budget["resolvent_consistency"], 0.04)
        self.assertAlmostEqual(budget["greedy_error"], 2.0 * (0.05 + 0.08))
        self.assertAlmostEqual(budget["solve_model_error"], 0.5 * (0.03 + 0.12))
        self.assertAlmostEqual(budget["total"], 0.04 + 0.04 + 0.26 + 0.075)
        with self.assertRaises(ValueError):
            original_iteration_budget(
                outer_iteration=0,
                evidence=ASSUMED,
                discount_rate=0.5,
                resolvent_step=2.0,
                semigroup_constant=0.01,
                semigroup_order=1.0,
                resolvent_constant=0.02,
                raw_residual_bound=0.03,
                input_value_rate_error=0.04,
                input_value_actor_gap=0.05,
                output_value_model_error=0.06,
            )

    def test_recurrence_value(self):
        a = math.exp(-1.0)
        budgets = [
            {"total": 0.1},
            {"total": 0.2},
            {"total": 0.05},
        ]
        expected = a**3 * 2.0 + a**2 * 0.1 + a * 0.2 + 0.05
        value = original_value_error_bound(
            2.0, budgets, discount_rate=0.5, resolvent_step=2.0
        )
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(
            original_value_error_bound(2.0, [], discount_rate=0.5, resolvent_step=2.0), 2.0
        )

    def test_floor_value(self):
        self.assertAlmostEqual(
            original_value_error_floor(
                budget_upper_bound=0.3, discount_rate=0.5, resolvent_step=2.0
            ),
            0.3 / (1.0 - math.exp(-1.0)),
        )

    def test_bound_upper_bounds_perturbed_contraction(self):
        rate, step = 0.5, 0.4
        contraction = math.exp(-rate * step)
        rng = np.random.default_rng(0)
        fixed_point = 1.7
        budgets = [self.budget(k, rate, step) for k in range(200)]
        initial_error = 3.0

        # Random perturbations bounded by each B_k stay below the recurrence.
        value = fixed_point + initial_error
        for k, budget in enumerate(budgets):
            noise = rng.uniform(-budget["total"], budget["total"])
            value = fixed_point + contraction * (value - fixed_point) + noise
            bound = original_value_error_bound(
                initial_error, budgets[: k + 1], discount_rate=rate, resolvent_step=step
            )
            self.assertLessEqual(abs(value - fixed_point), bound + 1e-12)

        # Worst-case same-sign perturbations attain the bound, and the floor is its limit.
        value = fixed_point + initial_error
        for budget in budgets:
            value = fixed_point + contraction * (value - fixed_point) + budget["total"]
        bound = original_value_error_bound(
            initial_error, budgets, discount_rate=rate, resolvent_step=step
        )
        self.assertAlmostEqual(abs(value - fixed_point), bound, places=10)
        self.assertAlmostEqual(
            bound,
            original_value_error_floor(
                budget_upper_bound=budgets[0]["total"],
                discount_rate=rate,
                resolvent_step=step,
            ),
            places=6,
        )


class TestIdealPremises(unittest.TestCase):
    NAMES = (
        "fixed_correct_model_and_temperature",
        "exact_generator_rate_and_statewise_greediness",
        "exact_resolvent_steps",
        "compact_state_domain",
        "common_classical_generator_boundary_domains",
        "uniformly_bounded_generator_coefficients",
        "hamiltonian_continuous_in_c2",
        "hjb_uniqueness_and_comparison",
        "initial_value_below_optimum",
        "initial_hamiltonian_nonnegative",
        "iterates_precompact_in_c2",
    )

    def test_all_premises_pass(self):
        validate_ideal_iteration_premises(
            {"evidence": ASSUMED, **{name: True for name in self.NAMES}}
        )

    def test_every_false_premise_is_named(self):
        premises = {"evidence": ASSUMED, **{name: True for name in self.NAMES}}
        premises["compact_state_domain"] = False
        premises["iterates_precompact_in_c2"] = False
        with self.assertRaises(ValueError) as caught:
            validate_ideal_iteration_premises(premises)
        message = str(caught.exception)
        self.assertIn("compact_state_domain", message)
        self.assertIn("iterates_precompact_in_c2", message)
        self.assertNotIn("exact_resolvent_steps", message)

    def test_sampled_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ideal_iteration_premises(
                {"evidence": SAMPLED, **{name: True for name in self.NAMES}}
            )


class TestRateCalibration(unittest.TestCase):
    def test_scale_value(self):
        duration, rate = 0.5, 2.0
        c_delta = (1.0 - math.exp(-1.0)) / (rate * duration)
        value = fixed_clock_blended_rate_scale(
            duration=duration,
            discount_rate=rate,
            data_weight=1.0,
            rollout_weight=2.0,
            generator_weight=1.0,
        )
        self.assertAlmostEqual(value, (3.0 * c_delta + 1.0) / 4.0)
        self.assertGreaterEqual(value, c_delta)
        self.assertLess(value, 1.0)

    def test_scale_endpoints_and_random_range(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            duration = float(rng.uniform(1e-3, 1.0))
            rate = float(rng.uniform(0.01, 5.0))
            data, rollout, generator = rng.uniform(0.0, 3.0, size=3)
            scale = fixed_clock_blended_rate_scale(
                duration=duration,
                discount_rate=rate,
                data_weight=float(data),
                rollout_weight=float(rollout),
                generator_weight=float(generator),
            )
            c_delta = (1.0 - math.exp(-rate * duration)) / (rate * duration)
            self.assertGreaterEqual(scale, c_delta - 1e-12)
            self.assertLess(scale, 1.0)
        # Only generator weight leaves the rate unscaled; only interval weight sits at c_Delta.
        self.assertEqual(
            fixed_clock_blended_rate_scale(
                duration=0.5,
                discount_rate=2.0,
                data_weight=0.0,
                rollout_weight=0.0,
                generator_weight=1.0,
            ),
            1.0,
        )
        self.assertAlmostEqual(
            fixed_clock_blended_rate_scale(
                duration=0.5,
                discount_rate=2.0,
                data_weight=1.0,
                rollout_weight=1.0,
                generator_weight=0.0,
            ),
            1.0 - math.exp(-1.0),
        )
        with self.assertRaises(ValueError):
            fixed_clock_blended_rate_scale(
                duration=0.5,
                discount_rate=2.0,
                data_weight=0.0,
                rollout_weight=0.0,
                generator_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
