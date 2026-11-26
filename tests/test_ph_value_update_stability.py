"""evaluations/ph_sac/value_update_stability.py: ground truth, mode factors, rules and ensemble.

Checks: the Lyapunov value of a linear policy solves the evaluation equation r + b . grad V
- lambda V = 0 with the testbed's own reward and oracle drift, the optimal policy's value is
the Riccati value and its greedy torque is K* x; the Van Loan interval maps reproduce a
quadrature and leave V^pi fixed under the rollout target; the threshold k_c and the three
amplification factors behave as eq. thresholds states, and the quadratic-subspace radius
equals the spectral radius of the matrix iteration it summarizes; the batched ensemble
equals SmoothValueHead member by member and its per-member clip equals clip_grad_norm_;
one outer update of every rule (Stage A) and a few steps of Stage B stay finite.
"""

import dataclasses
import math
import unittest

import numpy as np
import scipy.linalg
import torch

from evaluations.ph_sac import value_update_stability as stability

TASK = "lqr_2_1"


def _problem(**overrides):
    problem = stability.load_problem(TASK)
    if overrides:
        network = dict(problem.network)
        network.update(overrides.pop("network", {}))
        problem = dataclasses.replace(problem, network=network, **overrides)
    return problem


def _small_problem():
    """The task with a 16-16 value network and minibatches of 32, for fast rule steps."""
    return _problem(network={"hidden_sizes": [16, 16]}, batch_size=32)


class TestGroundTruth(unittest.TestCase):
    def setUp(self):
        self.problem = _problem()
        self.riccati, self.optimal = stability.riccati_solution(self.problem)
        self.states = np.random.default_rng(0).uniform(-2.0, 2.0, (64, self.problem.state_dim))

    def test_lyapunov_value_solves_the_evaluation_equation(self):
        """r + b . grad V^pi - lambda V^pi vanishes to 1e-8 for K*, 0.5 K* and a random gain."""
        random_gain = np.random.default_rng(1).normal(scale=0.3, size=self.optimal.shape)
        for gain in (self.optimal, 0.5 * self.optimal, random_gain):
            matrix = stability.policy_value_matrix(self.problem, gain)
            residual = stability.hjb_residual(self.problem, gain, matrix, self.states)
            self.assertLess(np.abs(residual).max(), 1e-8)

    def test_optimal_policy_value_is_the_riccati_value_and_its_greedy_torque_is_k_star(self):
        np.testing.assert_allclose(
            stability.policy_value_matrix(self.problem, self.optimal), self.riccati, atol=1e-8
        )
        policy = stability.GreedyPolicy(
            torch.as_tensor(self.problem.input_matrix), self.problem.control_cost, 1e9
        )
        _, gradients = stability.quadratic_value(self.riccati, self.problem.discount_rate, self.states)
        torques = policy(torch.as_tensor(self.states), torch.as_tensor(gradients)).numpy()
        np.testing.assert_allclose(torques, self.states @ self.optimal.T, atol=1e-10)

    def test_oracle_drift_is_the_linear_model(self):
        plant = stability.plant_functions(self.problem)
        torques = np.random.default_rng(2).normal(size=(64, self.problem.action_dim))
        drift = plant.drift(torch.as_tensor(self.states), torch.as_tensor(torques)).numpy()
        expected = self.states @ self.problem.state_matrix.T + torques @ self.problem.input_matrix.T
        np.testing.assert_allclose(drift, expected, atol=1e-12)

    def test_van_loan_weight_matches_quadrature(self):
        gain = 0.5 * self.optimal
        dynamics = stability.closed_loop(self.problem, gain)
        cost = self.problem.state_cost() + gain.T @ self.problem.control_weight() @ gain
        rate, step = self.problem.discount_rate, 0.7
        _, weight = stability.discounted_flow(dynamics, cost, rate, step)
        times = np.linspace(0.0, step, 2001)
        integrand = np.stack(
            [math.exp(-rate * t) * scipy.linalg.expm(dynamics.T * t) @ cost @ scipy.linalg.expm(dynamics * t)
             for t in times]
        )
        quadrature = scipy.integrate.simpson(integrand, x=times, axis=0)
        np.testing.assert_allclose(weight, quadrature, atol=1e-9)

    def test_rollout_target_leaves_the_policy_value_fixed(self):
        """int_0^T e^{-lambda s} r ds + e^{-lambda T} V^pi(x_T) = V^pi(x), the semigroup property."""
        for gain in (self.optimal, 0.5 * self.optimal):
            matrix = stability.policy_value_matrix(self.problem, gain)
            cost = self.problem.state_cost() + gain.T @ self.problem.control_weight() @ gain
            for step in (0.03, 0.1, 1.0):
                transition, weight = stability.discounted_flow(
                    stability.closed_loop(self.problem, gain), cost, self.problem.discount_rate, step
                )
                discount = math.exp(-self.problem.discount_rate * step)
                np.testing.assert_allclose(weight + discount * transition.T @ matrix @ transition, matrix, atol=1e-10)

    def test_held_torque_flow_matches_zero_order_hold(self):
        period = self.problem.reference_period
        transition, inputs = stability.zero_order_hold(self.problem, period)
        np.testing.assert_allclose(transition, scipy.linalg.expm(self.problem.state_matrix * period), atol=1e-12)
        times = np.linspace(0.0, period, 401)
        integral = scipy.integrate.simpson(
            np.stack([scipy.linalg.expm(self.problem.state_matrix * t) for t in times]), x=times, axis=0
        )
        np.testing.assert_allclose(inputs, integral @ self.problem.input_matrix, atol=1e-10)


class TestModeFactors(unittest.TestCase):
    rate = 0.5

    def test_threshold_formula(self):
        """|1 + eta T (i k_c - lambda)| = 1 at k_c = sqrt(lambda (2/(eta T) - lambda)), > 1 above."""
        for step in (0.01, 0.03, 0.1, 1.0):
            for eta in (1.0, 0.0125):
                threshold = stability.drift_threshold(step, eta, self.rate)
                self.assertAlmostEqual(threshold, math.sqrt(self.rate * (2.0 / (eta * step) - self.rate)))
                factor = lambda k: abs(stability.amplification("generator", 1j * k - self.rate, step, eta))
                self.assertAlmostEqual(factor(threshold), 1.0, places=12)
                self.assertGreater(factor(1.01 * threshold), 1.0)
                self.assertLess(factor(0.99 * threshold), 1.0)
                if eta * step * self.rate <= 0.05:
                    self.assertAlmostEqual(threshold, math.sqrt(2.0 * self.rate / (eta * step)), delta=0.02 * threshold)
        self.assertEqual(stability.drift_threshold(5.0, 1.0, self.rate), 0.0)

    def test_implicit_and_rollout_factors_stay_below_one(self):
        wavenumbers = np.logspace(-2, 4, 200)
        exponents = 1j * wavenumbers - self.rate
        for step in (0.03, 1.0):
            implicit = np.abs(stability.amplification("implicit", exponents, step))
            self.assertTrue(np.all(implicit < 1.0))
            self.assertTrue(np.all(np.diff(implicit) <= 0.0))
            np.testing.assert_allclose(implicit[-1] * step * wavenumbers[-1], 1.0, rtol=1e-3)
            for eta in (1.0, 0.0125):
                rollout = np.abs(stability.amplification("rollout", exponents, step, eta))
                self.assertTrue(np.all(rollout <= 1.0 - eta * (1.0 - math.exp(-self.rate * step)) + 1e-12))

    def test_quadratic_radius_is_the_radius_of_the_matrix_iteration(self):
        """E <- E + eta T (A^T E + E A - lambda E) on vec(E), with the constant mode 1 - eta T lambda."""
        problem = _problem()
        _, optimal = stability.riccati_solution(problem)
        closed = stability.closed_loop(problem, optimal)
        size = closed.shape[0]
        identity = np.eye(size * size)
        lyapunov = np.kron(np.eye(size), closed.T) + np.kron(closed.T, np.eye(size)) - self.rate * identity
        for step in (0.03, 1.0):
            for eta in (1.0, 0.0125):
                for rule, operator in (
                    ("generator", identity + eta * step * lyapunov),
                    ("rollout", (1.0 - eta) * identity + eta * scipy.linalg.expm(step * lyapunov)),
                ):
                    expected = max(
                        np.abs(np.linalg.eigvals(operator)).max(),
                        abs(stability.amplification(rule, -self.rate, step, eta)),
                    )
                    radius = stability.quadratic_spectral_radius(rule, closed, self.rate, step, eta)
                    self.assertAlmostEqual(radius, expected, places=9)
            implicit = np.linalg.inv(identity - step * lyapunov)
            self.assertAlmostEqual(
                stability.quadratic_spectral_radius("implicit", closed, self.rate, step, 1.0),
                max(np.abs(np.linalg.eigvals(implicit)).max(), 1.0 / (1.0 + self.rate * step)),
                places=9,
            )

    def test_paired_update_steps(self):
        self.assertEqual(stability.paired_update_steps(1.0), stability.UPDATE_STEPS)
        self.assertEqual(stability.paired_update_steps(0.0125), 1)
        with self.assertRaises(ValueError):
            stability.paired_update_steps(0.33)


class TestEnsembleAndRules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.problem = _small_problem()
        self.riccati, self.optimal = stability.riccati_solution(self.problem)
        self.box = stability.state_box(self.problem, [self.optimal], paths=20, duration=2.0)

    def test_ensemble_equals_smooth_value_head_member_by_member(self):
        heads = stability.value_heads(_problem(), [0, 1, 2])
        ensemble = stability.ValueEnsemble(heads)
        states = torch.randn(3, 17, self.problem.state_dim)
        outputs = ensemble(states)
        for index, head in enumerate(heads):
            torch.testing.assert_close(outputs[index], head(states[index]), atol=1e-5, rtol=1e-5)
        selected = ensemble.select([2, 0])
        torch.testing.assert_close(selected(states[:2]), torch.stack((heads[2](states[0]), heads[0](states[1]))),
                                   atol=1e-5, rtol=1e-5)

    def test_clip_per_member_equals_clip_grad_norm_on_each_member(self):
        heads = stability.value_heads(self.problem, [0, 1])
        ensemble = stability.ValueEnsemble(heads)
        states = 10.0 * torch.randn(2, 8, self.problem.state_dim)
        ensemble(states).square().sum().backward()
        for index, head in enumerate(heads):
            head(states[index]).square().sum().backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
        stability.clip_per_member(ensemble.parameters(), 0.5)
        for index, head in enumerate(heads):
            linears = [m for m in head.network if isinstance(m, torch.nn.Linear)]
            for layer, weight in zip(linears, ensemble.weights):
                torch.testing.assert_close(weight.grad[index], layer.weight.grad.T, atol=1e-5, rtol=1e-4)

    def _evaluation(self, gain, matrix):
        return stability.evaluation_set(self.problem, matrix, gain, self.box, seed=3)

    def test_one_outer_update_of_each_rule_is_finite(self):
        seeds = [0]
        data = stability.uniform_box_states(self.box, stability.DATA_SIZE, torch.Generator().manual_seed(1))[None]
        gain = self.optimal
        evaluation = self._evaluation(gain, stability.policy_value_matrix(self.problem, gain))
        policy = stability.LinearPolicy(torch.as_tensor(gain, dtype=torch.get_default_dtype()))
        for rule in stability.RULES:
            members = tuple(stability.stage_a_members(rule, (0.03, 1.0), (1.0, 0.0125), seeds))
            spec = stability.RunSpec("A", "K*", rule, members, stability.UPDATE_STEPS, stability.UPDATE_STEPS, gain)
            rows = stability.run_ensemble(self.problem, spec, seeds, evaluation, policy=policy, data=data,
                                          progress=lambda _: None)
            final = [row for row in rows if row["grad_steps"] == stability.UPDATE_STEPS]
            self.assertEqual(len(final), len(members))
            for row in final:
                self.assertEqual(row["diverged"], 0)
                self.assertGreaterEqual(row["outer_updates"], 1)
                for key in ("value_rms_error", "gradient_rms_error", "online_value_rms_error", "kappa_eff"):
                    self.assertTrue(math.isfinite(row[key]), (rule, key))

    def test_stage_b_steps_and_replay_refill_are_finite(self):
        seeds = [0]
        evaluation = self._evaluation(self.optimal, self.riccati)
        policy = stability.GreedyPolicy(
            torch.as_tensor(self.problem.input_matrix, dtype=torch.get_default_dtype()),
            self.problem.control_cost,
            self.problem.torque_limit,
        )
        for rule in stability.RULES:
            members = tuple(stability.stage_b_members(rule, self.problem, seeds, self.problem.reference_period))
            spec = stability.RunSpec("B", "greedy", rule, members, 40, 40, self.optimal, data_source="replay")
            rows = stability.run_ensemble(self.problem, spec, seeds, evaluation, policy=policy,
                                          progress=lambda _: None)
            for row in rows:
                self.assertTrue(math.isfinite(row["gradient_rms_error"]))
                self.assertTrue(math.isfinite(row["data_value_rms_error"]))
                self.assertLessEqual(row["max_abs_torque"], self.problem.torque_limit)
        box_data = stability.uniform_box_states(self.box, stability.DATA_SIZE, torch.Generator().manual_seed(1))[None]
        members = tuple(stability.stage_b_members("generator", self.problem, seeds, self.problem.reference_period))
        fixed = stability.LinearPolicy(torch.as_tensor(self.optimal, dtype=torch.get_default_dtype()))
        for source in ("fixed", "frozen replay"):
            spec = stability.RunSpec("B", "K*", "generator", members, 40, 40, self.optimal, data_source=source)
            rows = stability.run_ensemble(self.problem, spec, seeds, evaluation, policy=fixed, data=box_data,
                                          progress=lambda _: None)
            self.assertTrue(all(math.isfinite(row["max_abs_torque"]) for row in rows))
        trainer_target = stability.ValueEnsemble(stability.value_heads(self.problem, [0, 0])).frozen_copy()
        collector = stability.ReplayCollector(self.problem, 2, torch.tensor([0, 0]), 1)
        replay = collector.initial(trainer_target, policy)
        self.assertEqual(tuple(replay.shape), (2, stability.DATA_SIZE, self.problem.state_dim))
        before = replay.clone()
        collector.refill(replay, trainer_target, policy)
        fresh = stability.COLLECT_PATHS * stability.COLLECT_DECISIONS
        self.assertEqual(collector.pointer, fresh)
        self.assertFalse(torch.equal(replay[:, :fresh], before[:, :fresh]))
        torch.testing.assert_close(replay[:, fresh:], before[:, fresh:])
        self.assertTrue(torch.isfinite(replay).all())


if __name__ == "__main__":
    unittest.main()
