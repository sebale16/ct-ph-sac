import contextlib
import copy
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.run_ph_sac import apply_overrides, load_protocol, main, parse_args
from common.ph_config import decision_box_entropy_target, windy_pendulum_config
from environment.ph_sac.tasks import make_env, make_oracle_plant
from evaluations.ph_sac import diagnostics, protocol, reporting

CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac"


def smoke_protocol():
    return load_protocol(str(CONFIGS / "smoke_windy.json"))


class FakeAgent:
    """Holds a fixed decision and reports contract-shaped counters and records."""

    def __init__(self, env, decision):
        self.env = env
        self.decision = np.asarray(decision, dtype=np.float64)
        self.interactions = 0
        self.saved = None
        self.solver_records = [
            {
                "outer_iteration": 0,
                "inner_iterations": 10,
                "status": "completed",
                "contraction_factor": 0.9,
                "fixed_point_motion_input": 0.0,
            }
        ]
        self.admission_records = [
            {
                "admission_index": 0,
                "outer_iteration": 3,
                "accepted": True,
                "corrected_mean": -1.0,
                "corrected_standard_deviation": 0.5,
                "raw_mean": -0.9,
                "mean_variance_correction": 0.1,
                "status": "supported_improvement",
                "interval_lower": -1.5,
                "interval_upper": -0.5,
                "real_rows_used": 32,
                "real_rows_acquired": 0,
                "simulated_paths_used": 768,
                "transfer_diagnostic": 0.3,
                "admitted_mean_square": 0.2,
                "admitted_monte_carlo_variance": 0.1,
                "validation_rows": 16,
                "rollout_paths_per_row": 4,
                "next_outer_iterations": 32,
            }
        ]
        self.oracle_records = [
            {
                "interactions": 10,
                "sampled_states": 8,
                "drift_error_max": 1.0,
                "diffusion_covariance_error_max": 2.0,
                "value_direction_error_max": 3.0,
                "value_direction_error_mean": 1.5,
            }
        ]
        self.calibration_records = []

    def predict(self, state, deterministic=True):
        return self.decision

    def learn(self, total_interactions, callback=None):
        while self.interactions < total_interactions:
            self.interactions += 1
            if callback is not None:
                callback(self, {"interactions": self.interactions})
        return self

    def counters(self):
        return {
            "interactions": self.interactions,
            "simulated_seconds": self.interactions * 0.01,
            "critic_gradient_steps": self.interactions,
            "value_gradient_steps": self.interactions,
            "rate_gradient_steps": 0,
            "actor_gradient_steps": self.interactions,
            "temperature_gradient_steps": 0,
            "plant_gradient_steps": self.interactions,
            "reward_gradient_steps": self.interactions,
            "model_rollout_paths": 0,
            "wall_seconds": 1000.0,
        }

    def save(self, path, *, include_replay):
        self.saved = path
        self.saved_replay = include_replay
        Path(path).write_bytes(b"checkpoint")


class ArmTests(unittest.TestCase):
    def test_four_paper_arms_in_the_evaluation_matrix(self):
        # The default arm set, in role order: baseline, implicit, rollout, generator.
        names = [arm["name"] for arm in protocol.paper_arms()]
        self.assertEqual(
            names,
            [
                "ct_sac_torque",
                "implicit_torque",
                "explicit_rollout_torque",
                "explicit_generator_torque",
            ],
        )
        # The ablations, run only when named.
        self.assertEqual(
            {arm["name"] for arm in protocol.all_arms()} - set(names),
            {
                "implicit_learned_admissions",
                "explicit_rollout",
                "explicit_generator",
                "implicit_oracle",
                "implicit_averaged_draw",
                "implicit_no_rollout_rate",
                "implicit_polyak_plant",
                "implicit_gated_admissions",
                "implicit_fixed_gate_allocation",
                "implicit_explicit_step",
                "implicit_explicit_step_interval",
                "implicit_critic",
                "implicit_critic_head",
            },
        )

    def test_each_ablation_changes_one_field(self):
        arms = {arm["name"]: arm for arm in protocol.all_arms()}
        for reference_name, expected in (
            (
                "implicit_learned_admissions",
                {
                    "implicit_oracle": "plant_mode",
                    "implicit_averaged_draw": "residual_loss",
                    "implicit_no_rollout_rate": "rollout_rate_enabled",
                    "implicit_polyak_plant": "plant_mode",
                    "implicit_gated_admissions": "plant_mode",
                },
            ),
            # The fixed allocation ablates the staged gate, so it is one step from that arm.
            ("implicit_gated_admissions", {"implicit_fixed_gate_allocation": "gate_allocation"}),
        ):
            reference = arms[reference_name]
            for name, field in expected.items():
                changed = [
                    key
                    for key in reference
                    if key != "name" and arms[name][key] != reference[key]
                ]
                self.assertEqual(changed, [field], name)

    def test_arm_configs_apply_exactly_their_setting(self):
        proto = smoke_protocol()
        proto["arms"] = list(protocol.all_arms())
        matrix = {run["arm"]["name"]: run for run in protocol.build_run_matrix(proto)}
        base = protocol._implicit_arm_config(matrix["implicit_learned_admissions"])

        def changed(name):
            config = protocol._implicit_arm_config(matrix[name])
            found = []
            for section in ("solver", "rate", "admission"):
                for key in config[section]:
                    if config[section][key] != base[section][key]:
                        found.append(f"{section}.{key}")
            if config["plant_source"] != base["plant_source"]:
                found.append("plant_source")
            return sorted(found)

        self.assertEqual(base["admission"]["plant_update"], "always")
        self.assertEqual(changed("implicit_oracle"), ["plant_source"])
        self.assertEqual(changed("implicit_averaged_draw"), ["solver.residual_estimator"])
        self.assertEqual(changed("implicit_no_rollout_rate"), ["rate.rollout_weight"])
        self.assertEqual(
            changed("implicit_polyak_plant"),
            ["admission.plant_update", "admission.polyak_rate"],
        )
        self.assertEqual(changed("implicit_gated_admissions"), ["admission.plant_update"])
        self.assertEqual(
            protocol._implicit_arm_config(matrix["implicit_gated_admissions"])["admission"][
                "plant_update"
            ],
            "admission",
        )
        self.assertEqual(
            changed("implicit_fixed_gate_allocation"),
            ["admission.allocation", "admission.plant_update"],
        )
        self.assertEqual(
            protocol._implicit_arm_config(matrix["implicit_no_rollout_rate"])["rate"][
                "rollout_weight"
            ],
            0.0,
        )

    def test_arm_config_does_not_touch_the_protocol(self):
        proto = smoke_protocol()
        before = copy.deepcopy(proto["implicit_config"]["solver"])
        run = {
            "run_id": "x",
            "protocol": proto,
            "arm": {a["name"]: a for a in protocol.all_arms()}[
                "implicit_averaged_draw"
            ],
            "training_seed": 7,
        }
        config = protocol._implicit_arm_config(run)
        self.assertEqual(config["training"]["seed"], 7)
        self.assertEqual(proto["implicit_config"]["solver"], before)

    def test_run_matrix_expands_arms_and_seeds(self):
        proto = smoke_protocol()
        proto["training_seeds"] = [0, 1]
        matrix = protocol.build_run_matrix(proto)
        # The four paper arms by two seeds.
        self.assertEqual(len(matrix), 8)
        self.assertEqual(len({run["run_id"] for run in matrix}), 8)


class MatchedQuantityTests(unittest.TestCase):
    def assert_rejected(self, mutate):
        proto = smoke_protocol()
        proto = copy.deepcopy(proto)
        mutate(proto)
        with self.assertRaises(ValueError):
            protocol.validate_matched_quantities(proto)

    def test_valid_protocol_passes(self):
        protocol.validate_matched_quantities(smoke_protocol())

    def test_discount_mismatch(self):
        def mutate(p):
            p["implicit_config"]["testbed"]["discount_rate"] = 0.2

        self.assert_rejected(mutate)

    def test_clock_mismatch(self):
        def mutate(p):
            # Loaded protocols share one testbed object, so give this arm its own.
            p["explicit_config"]["testbed"] = copy.deepcopy(p["matched"]["testbed_config"])
            p["explicit_config"]["testbed"]["clock"]["tail_split"] = 0.25

        self.assert_rejected(mutate)

    def test_budget(self):
        def mutate(p):
            p["matched"]["interaction_budget"] = 0

        self.assert_rejected(mutate)

    def test_checkpoint_replay_must_be_a_flag(self):
        self.assertIs(smoke_protocol()["checkpoint_replay"], True)

        def mutate(p):
            p["checkpoint_replay"] = "false"

        self.assert_rejected(mutate)

    def test_temperature_mismatch(self):
        def mutate(p):
            p["explicit_config"]["temperature"]["initial_alpha_c"] = 0.5

        self.assert_rejected(mutate)

    def test_tuned_temperature_is_allowed_and_the_default(self):
        proto = smoke_protocol()
        for name in ("explicit_config", "implicit_config"):
            self.assertTrue(proto[name]["temperature"]["learn_temperature"])
        protocol.validate_matched_quantities(proto)
        fixed = copy.deepcopy(proto)
        for name in ("explicit_config", "implicit_config"):
            fixed[name]["temperature"] = dict(
                fixed[name]["temperature"], learn_temperature=False
            )
        protocol.validate_matched_quantities(fixed)

    def _own_temperatures(self, p):
        # Loaded protocols share one temperature object, so give each arm its own.
        for name in ("explicit_config", "implicit_config"):
            p[name]["temperature"] = dict(p[name]["temperature"])

    def test_entropy_target_mismatch(self):
        def mutate(p):
            self._own_temperatures(p)
            p["implicit_config"]["temperature"]["target_entropy"] += 1.0

        self.assert_rejected(mutate)

    def test_tuning_mismatch(self):
        def mutate(p):
            self._own_temperatures(p)
            p["implicit_config"]["temperature"]["learn_temperature"] = False

        self.assert_rejected(mutate)

    def test_initial_temperature_mismatch_across_ct_ph_sac_arms(self):
        def mutate(p):
            self._own_temperatures(p)
            p["implicit_config"]["temperature"]["initial_alpha_c"] = 0.2

        self.assert_rejected(mutate)

    def test_legacy_baseline_arm(self):
        def mutate(p):
            p["arms"][0]["decision_mode"] = "feedback"

        self.assert_rejected(mutate)

    def test_duplicate_reset_seeds(self):
        def mutate(p):
            p["matched"]["evaluation_reset_seeds"] = [1, 1]

        self.assert_rejected(mutate)

    def theory(self, **changes):
        config = {
            "policy_kind": "deterministic_mean",
            "objective": "reward_only",
            "every_outer_iterations": 1,
            "sampled_states": 8,
            "decision_draws": 2,
            "holding_paths": 2,
            "initial_state_law_id": "reset",
            "record_optimizer_steps": False,
            "record_training_calibration": False,
            "record_outer_budgets": False,
            "uniform_evidence": None,
            "task_return_threshold": None,
        }
        config.update(changes)
        return config

    def test_theory_mean_policy_needs_reward_only(self):
        protocol.validate_theory_diagnostics(self.theory())
        with self.assertRaises(ValueError):
            protocol.validate_theory_diagnostics(self.theory(objective="soft"))

    def test_theory_threshold_needs_uniform_evidence(self):
        with self.assertRaises(ValueError):
            protocol.validate_theory_diagnostics(self.theory(task_return_threshold=-1.0))
        sampled = {
            "kind": "sampled_diagnostic",
            "domain": "replay",
            "justification": "sampled",
            "failure_probability": None,
        }
        with self.assertRaises(ValueError):
            protocol.validate_theory_diagnostics(
                self.theory(uniform_evidence=sampled, task_return_threshold=-1.0)
            )

    def test_theory_policy_kind_must_match_evaluation(self):
        def mutate(p):
            p["theory_diagnostics"] = self.theory(
                policy_kind="stochastic", objective="soft"
            )

        self.assert_rejected(mutate)


class SummarizeEpisodeTests(unittest.TestCase):
    def test_hand_values(self):
        record = diagnostics.summarize_episode(
            reset_seed=3,
            interval_rewards=[-1.0, -2.0, 4.0],
            interval_start_times=[0.0, 1.0, 3.0],
            discount_rate=0.5,
            terminated=True,
            truncated=False,
            duration_seconds=3.5,
            physics_steps=3500,
            policy_kind="deterministic_mean",
        )
        expected = -1.0 - 2.0 * math.exp(-0.5) + 4.0 * math.exp(-1.5)
        self.assertAlmostEqual(record["discounted_return"], expected, places=12)
        self.assertTrue(record["failed"])
        self.assertFalse(record["truncated"])
        self.assertEqual(record["decisions"], 3)
        self.assertEqual(record["physics_steps"], 3500)


class GateDiagnosticTests(unittest.TestCase):
    RESULT = {
        "status": "evidence_against",
        "inference_mode": "empirical",
        "raw_mean": 4.0,
        "variance_correction": 1.0,
        "corrected_mean": 3.0,
        "corrected_std": 2.0,
        "interval_lower": 1.0,
        "interval_upper": 5.0,
        "transfer_diagnostic": math.nan,
        "accepted": False,
        "validation_rows": 4,
        "paths_per_model": 8,
        "stages": 2,
        "real_rows_used": 12,
        "real_rows_acquired": 3,
        "simulated_paths_used": 256,
        "pilot_variance_simulation": 0.5,
        "pilot_variance_real": 0.25,
        "stage_log": [{"rows": 4}],
    }

    def measure(self, production):
        return diagnostics.measure_gate_diagnostics(
            admission_index=2,
            outer_iteration=9,
            result=self.RESULT,
            production=production,
            next_outer_iterations=32,
        )

    def test_hand_values(self):
        record = self.measure(
            {
                "squares": np.array([1.0, 4.0, 9.0, 16.0]),
                "variances": np.array([1.0, 1.0, 1.0, 1.0]),
                "horizons": np.array([0.01, 0.02, 0.03, 0.04]),
                "candidate": {
                    "failure_rate": 0.5,
                    "running_reward": -0.1,
                    "failure_payoff": -5.0,
                    "continuation": -2.0,
                },
                "admitted": {
                    "failure_rate": 0.25,
                    "running_reward": -0.2,
                    "failure_payoff": -2.5,
                    "continuation": -3.0,
                },
                "observed_failure_rate": 0.125,
            }
        )
        self.assertEqual(record["status"], "evidence_against")
        self.assertAlmostEqual(record["corrected_mean"], 3.0)
        self.assertAlmostEqual(record["corrected_standard_deviation"], 2.0)
        self.assertAlmostEqual(record["interval_upper"], 5.0)
        self.assertFalse(record["accepted"])
        self.assertAlmostEqual(record["raw_mean"], 4.0)
        self.assertAlmostEqual(record["mean_variance_correction"], 1.0)
        self.assertAlmostEqual(record["admitted_mean_square"], 7.5)
        self.assertAlmostEqual(record["admitted_monte_carlo_variance"], 1.0)
        self.assertAlmostEqual(record["candidate_failure_rate"], 0.5)
        self.assertAlmostEqual(record["admitted_failure_payoff"], -2.5)
        self.assertAlmostEqual(record["observed_failure_rate"], 0.125)
        self.assertAlmostEqual(record["mean_scheduled_horizon"], 0.025)
        self.assertEqual(record["validation_rows"], 4)
        self.assertEqual(record["rollout_paths_per_row"], 8)
        self.assertEqual(record["real_rows_acquired"], 3)
        self.assertEqual(record["simulated_paths_used"], 256)
        self.assertEqual(record["stage_log"], [{"rows": 4}])
        self.assertEqual(record["next_outer_iterations"], 32)

    def test_an_attempt_with_no_batch_records_nan(self):
        record = self.measure(None)
        self.assertTrue(math.isnan(record["admitted_mean_square"]))
        self.assertTrue(math.isnan(record["candidate_failure_rate"]))


class OracleErrorTests(unittest.TestCase):
    def test_measured_against_a_known_difference(self):
        env = make_env(windy_config(), seed=0)
        oracle = make_oracle_plant(env)
        shift = torch.tensor([0.0, 3.0])

        class Shifted(type(oracle)):
            def drift(self, states, actions):
                return oracle.drift(states, actions) + shift

            def covariance(self, states):
                return oracle.covariance(states) + torch.eye(2) * 2.0

        learned = copy.copy(oracle)
        learned.__class__ = Shifted
        states = np.array([[0.1, 0.2], [-0.2, 0.0]])
        actions = np.array([[1.0], [2.0]])
        gradients = np.array([[0.0, 2.0], [0.0, -1.0]])
        record = diagnostics.measure_oracle_errors(
            learned, oracle, states, actions, gradients, interactions=5
        )
        self.assertAlmostEqual(record["drift_error_max"], 3.0, places=5)
        self.assertAlmostEqual(
            record["diffusion_covariance_error_max"], 2.0 * math.sqrt(2.0), places=5
        )
        self.assertAlmostEqual(record["value_direction_error_max"], 6.0, places=5)
        self.assertAlmostEqual(record["value_direction_error_mean"], 4.5, places=5)
        self.assertEqual(record["sampled_states"], 2)
        env.close()


def windy_config():
    return windy_pendulum_config(
        reward_parameters={
            "angle_weight": 10.0,
            "velocity_weight": 1.0,
            "velocity_scale": 2.0 * math.pi,
            "effort_weight": 0.01,
        },
        reset_parameters={"angle_radius": 0.3, "velocity_radius": 0.5},
        tail_probability=0.8,
        tail_split=0.5,
        clock_kind="two_tail_uniform",
    )


class CalibrationTests(unittest.TestCase):
    def gaussian_rows(self, scale=1.0, n=4000):
        rng = np.random.default_rng(0)
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        factor = np.linalg.cholesky(covariance)
        predicted = rng.normal(size=(n, 2))
        endpoints = predicted + rng.normal(size=(n, 2)) @ factor.T
        durations = rng.choice([0.002, 0.018], size=n).reshape(-1, 1)
        batch = {"next_states": endpoints, "durations": durations}
        return (
            batch,
            predicted,
            np.broadcast_to(covariance * scale, (n, 2, 2)),
            durations,
        )

    def calibrate(self, scale):
        batch, predicted, covariance, durations = self.gaussian_rows(scale)
        return diagnostics.calibrate_by_duration(
            batch,
            predicted,
            covariance,
            interactions=11,
            duration_bin_edges=[0.0, 0.01, 0.02],
            upper_tail_probability=0.05,
            duration_jacobian_norms=durations * 3.0,
            saturated_fractions=np.full(len(durations), 0.25),
        )

    def test_exact_gaussian_is_calibrated(self):
        records = self.calibrate(1.0)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["state_dimension"], 2)
            self.assertAlmostEqual(record["mean_mahalanobis_square"], 2.0, delta=0.15)
            self.assertAlmostEqual(record["upper_tail_fraction"], 0.05, delta=0.015)
            self.assertGreater(record["chi_square_goodness_of_fit_pvalue"], 0.001)
            self.assertAlmostEqual(record["saturated_fraction"], 0.25)
        self.assertAlmostEqual(records[1]["duration_jacobian_norm_max"], 0.054)
        self.assertEqual(
            sum(record["sample_count"] for record in records), 4000
        )

    def test_misscaled_covariance_is_detected(self):
        for record in self.calibrate(4.0):
            self.assertAlmostEqual(record["mean_mahalanobis_square"], 0.5, delta=0.05)
            self.assertLess(record["chi_square_goodness_of_fit_pvalue"], 1e-6)
            self.assertLess(record["upper_tail_fraction"], 0.01)

    def test_periodic_residual_is_wrapped_when_a_spec_is_given(self):
        batch = {
            "next_states": np.array([[math.pi - 0.05, 0.0]]),
            "durations": np.array([[0.01]]),
        }
        predicted = np.array([[-math.pi + 0.05, 0.0]])
        covariance = np.eye(2)[None]
        kwargs = dict(
            interactions=0,
            duration_bin_edges=[0.0, 0.02],
            upper_tail_probability=0.05,
            duration_jacobian_norms=np.zeros(1),
            saturated_fractions=np.zeros(1),
        )
        wrapped = diagnostics.calibrate_by_duration(
            batch, predicted, covariance, state_spec=windy_config()["state_spec"], **kwargs
        )[0]
        self.assertAlmostEqual(wrapped["mean_mahalanobis_square"], 0.01, places=6)
        self.assertTrue(math.isnan(wrapped["chi_square_goodness_of_fit_pvalue"]))

    def test_true_paths_match_oracle_moments_on_windy(self):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            config = windy_config()

            def factory(seed):
                return make_env(config, seed=seed)

            env = factory(0)
            oracle = make_oracle_plant(env)
            env.close()
            states = torch.tensor([[0.1, 0.2], [-0.05, 0.0]])
            decisions = torch.tensor([[0.0, 30.0, 5.0], [0.05, 20.0, 2.0]])
            durations = torch.tensor([[0.01], [0.018]])
            batch = {
                "states": states,
                "decisions": decisions,
                "durations": durations,
            }
            records = diagnostics.calibrate_against_true_paths(
                factory,
                oracle,
                batch,
                interactions=3,
                rollout_seeds=[1, 2],
                paths_per_state=1000,
                duration_bin_edges=[0.0, 0.015, 0.02],
                upper_tail_probability=0.05,
            )
        finally:
            torch.set_default_dtype(previous)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["sample_count"], 2000)
            self.assertAlmostEqual(record["mean_mahalanobis_square"], 2.0, delta=0.2)
            self.assertAlmostEqual(record["upper_tail_fraction"], 0.05, delta=0.02)
            self.assertLess(record["endpoint_mean_error"], 0.05)
            self.assertLess(record["endpoint_covariance_error"], 0.05)
            self.assertGreater(record["duration_jacobian_norm_max"], 0.0)


class SampledTheoryTests(unittest.TestCase):
    context = {
        "outer_iteration": 4,
        "candidate_id": "V5",
        "policy_id": "pi5",
        "plant_id": "phi1",
        "temperature_per_second": 0.01,
        "policy_kind": "stochastic",
        "objective": "soft",
    }

    def test_hand_values(self):
        record = diagnostics.summarize_sampled_theory_diagnostics(
            context=self.context,
            domain="replay states",
            sampling_description="256 replay states",
            candidate_changes=np.array([-3.0, 2.0]),
            raw_policy_residuals=np.array([0.5, -0.75]),
            value_update_increments=[np.array([3.0, 4.0]), np.array([0.0, 1.0])],
            actor_update_increments=[np.array([1.0, 0.0])],
            rate_target_bias_samples=np.array([1.0, 2.0]),
            rate_target_sample_weights=np.array([3.0, 1.0]),
        )
        self.assertEqual(record["evidence"]["kind"], "sampled_diagnostic")
        self.assertEqual(record["candidate_change_max"], 3.0)
        self.assertEqual(record["raw_policy_residual_max"], 0.75)
        self.assertEqual(record["value_update_path_length"], 6.0)
        self.assertEqual(record["actor_update_path_length"], 1.0)
        self.assertAlmostEqual(record["rate_target_bias_l2"], math.sqrt(7.0 / 4.0))
        self.assertIsNone(record["rate_error_max"])

    def record(self, **changes):
        record = {
            "context": self.context,
            "solve_context": None,
            "sampled": [],
            "optimizer_steps": [],
            "training_accuracy": None,
            "outer_step": None,
            "original_iteration_budget": None,
            "performance_bound": None,
        }
        record.update(changes)
        return record

    def test_mean_policy_context_needs_zero_temperature(self):
        diagnostics.validate_theory_record_context(self.record())
        bad = dict(self.context, policy_kind="deterministic_mean", objective="reward_only")
        with self.assertRaises(ValueError):
            diagnostics.validate_theory_record_context(self.record(context=bad))

    def test_certificate_needs_certified_evidence(self):
        bound = {
            "context": self.context,
            "evidence": {
                "kind": "sampled_diagnostic",
                "domain": "replay",
                "justification": "sampled",
                "failure_probability": None,
            },
            "claim_kind": "certified",
            "policy_loss_upper_bound": 1.0,
            "soft_return_lower_bound": -1.0,
            "reward_return_lower_bound": None,
            "task_threshold": None,
            "threshold_satisfied": None,
        }
        with self.assertRaises(ValueError):
            diagnostics.validate_theory_record_context(self.record(performance_bound=bound))

    def test_sampled_records_keep_their_contexts(self):
        other = dict(self.context, policy_id="pi9")
        sampled = diagnostics.summarize_sampled_theory_diagnostics(
            context=other,
            domain="d",
            sampling_description="s",
            candidate_changes=np.zeros(1),
            raw_policy_residuals=np.zeros(1),
            value_update_increments=[],
            actor_update_increments=[],
        )
        with self.assertRaises(ValueError):
            diagnostics.validate_theory_record_context(self.record(sampled=[sampled]))
        diagnostics.validate_theory_record_context(
            self.record(sampled=[sampled], solve_context=other)
        )


class FixedSeedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = windy_config()
        self.durations = []
        self.envs = []

    def factory(self, seed):
        env = make_env(self.config, seed=seed)
        original = env.step_dt
        sequence = []
        self.durations.append(sequence)
        self.envs.append(env)

        def spy(decision, duration):
            sequence.append(duration)
            return original(decision, duration)

        env.step_dt = spy
        return env

    def run_policy(self, decision, seeds=(5, 6)):
        decision = np.asarray(decision)
        return protocol.evaluate_fixed_seeds(
            lambda state, deterministic: decision,
            self.factory,
            seeds,
            discount_rate=self.config["discount_rate"],
            deterministic=True,
        )

    def test_reproducible(self):
        first = self.run_policy([0.0, 40.0, 8.0])
        second = self.run_policy([0.0, 40.0, 8.0])
        self.assertEqual(first, second)
        self.assertEqual(first[0]["policy_kind"], "deterministic_mean")
        self.assertEqual(first[0]["reset_seed"], 5)
        self.assertNotEqual(
            first[0]["discounted_return"], first[1]["discounted_return"]
        )

    def test_common_clock_and_gusts_across_policies(self):
        self.run_policy([0.0, 40.0, 8.0], seeds=(5,))
        self.run_policy([0.1, 60.0, 10.0], seeds=(5,))
        first, second = self.durations
        shared = min(len(first), len(second))
        self.assertGreater(shared, 10)
        self.assertEqual(first[:shared], second[:shared])

    def test_gust_stream_does_not_depend_on_the_decision(self):
        states = []
        for decision in ([0.0, 40.0, 8.0], [0.1, 60.0, 10.0]):
            env = make_env(self.config, seed=5)
            env.reset(seed=5)
            for _ in range(40):
                env.step_dt(np.asarray(decision), env.sample_duration())
            self.assertFalse(env.done)
            states.append(env._gust_rng.bit_generator.state)
            env.close()
        self.assertEqual(states[0], states[1])

    def test_return_is_the_reward_only_sum(self):
        env = make_env(self.config, seed=9)
        state, _ = env.reset(seed=9)
        total = 0.0
        decision = np.array([0.0, 40.0, 8.0])
        while not env.done:
            transition = env.step_dt(decision, env.sample_duration())
            total += math.exp(-0.1 * transition["time"]) * transition["reward"]
        env.close()
        record = protocol.evaluate_fixed_seeds(
            lambda state, deterministic: decision,
            lambda seed: make_env(self.config, seed=seed),
            [9],
            discount_rate=0.1,
            deterministic=False,
        )[0]
        self.assertAlmostEqual(record["discounted_return"], total, places=10)
        self.assertEqual(record["policy_kind"], "stochastic")


class ScriptedEnv:
    """Replays fixed intervals (duration, undiscounted reward, next angle) on windy."""

    def __init__(self, intervals, *, failed, discount_rate=0.5, horizon=4.0, report=True):
        self.intervals = intervals
        self.failed = failed
        self.discount_rate = discount_rate
        self.physics_dt = 0.001
        self.config = {"name": "windy_pendulum", "episode_duration": horizon}
        self.report = report
        self.done = False

    def reset(self, seed=None):
        self.index, self.time, self.done = 0, 0.0, False
        return np.array([0.0, 0.0]), {}

    def sample_duration(self):
        return self.intervals[self.index][0]

    def step_dt(self, decision, duration):
        length, integral, angle = self.intervals[self.index]
        if self.report:
            self.last_undiscounted_reward = integral
        start = self.time
        self.time += length
        self.index += 1
        last = self.index == len(self.intervals)
        self.done = last
        return {
            "state": np.array([0.0, 0.0]),
            "decision": decision,
            "duration": length,
            "scheduled_duration": length,
            "reward": integral,
            "next_state": np.array([angle, 0.0]),
            "terminated": last and self.failed,
            "truncated": last and not self.failed,
            "time": start,
            "next_time": self.time,
        }

    def terminal_payoff(self, state):
        return -1.0 / self.discount_rate

    def close(self):
        pass


def return_to_go_average(intervals, *, failed, discount_rate, horizon, payoff, points=200001):
    """Average G_t over [0, horizon] on a fine grid, with piecewise-constant rates."""
    edges = np.cumsum([0.0] + [length for length, _, _ in intervals])
    rates = [integral / length for length, integral, _ in intervals]
    end = edges[-1]
    grid = np.linspace(0.0, horizon, points)
    values = np.empty_like(grid)
    lam = discount_rate
    for i, t in enumerate(grid):
        if t >= end:
            values[i] = payoff if failed else 0.0
            continue
        total = 0.0
        for (a, b), rate in zip(zip(edges[:-1], edges[1:]), rates):
            lower = max(a, t)
            if b > lower:
                total += rate * (np.exp(-lam * (lower - t)) - np.exp(-lam * (b - t))) / lam
        if failed:
            total += np.exp(-lam * (end - t)) * payoff
        values[i] = total
    return float(np.trapezoid(values, grid) / horizon)


class EpisodeLevelMetricTests(unittest.TestCase):
    INTERVALS = [(1.0, -0.2, 0.1), (1.0, -0.4, 0.1), (0.5, -1.0, 1.0), (0.5, -0.1, 0.0)]

    def run_scripted(self, *, failed, report=True, horizon=4.0):
        return protocol.evaluate_fixed_seeds(
            lambda state, deterministic: np.zeros(3),
            lambda seed: ScriptedEnv(
                self.INTERVALS, failed=failed, report=report, horizon=horizon
            ),
            [3],
            discount_rate=0.5,
            deterministic=True,
        )[0]

    def test_a_failed_episode(self):
        record = self.run_scripted(failed=True)
        self.assertAlmostEqual(record["episode_return"], -1.7)
        self.assertAlmostEqual(record["survival_seconds"], 3.0)
        self.assertAlmostEqual(record["rate_executed"], -1.7 / 3.0)
        # The worst rate is lambda F = -1 over the second lost to the failure.
        self.assertAlmostEqual(record["rate_horizon"], (-1.7 - 1.0) / 4.0)
        self.assertAlmostEqual(record["upright_seconds"], 2.5)
        self.assertAlmostEqual(record["max_hold_seconds"], 2.0)
        expected = return_to_go_average(
            self.INTERVALS, failed=True, discount_rate=0.5, horizon=4.0, payoff=-2.0
        )
        self.assertAlmostEqual(record["mean_return_to_go"], expected, places=5)
        self.assertTrue(record["failed"])
        self.assertEqual(record["decisions"], 4)

    def test_a_truncated_episode(self):
        # Truncation ends the episode at its horizon, here 3 s.
        record = self.run_scripted(failed=False, horizon=3.0)
        self.assertAlmostEqual(record["rate_horizon"], -1.7 / 3.0)
        self.assertAlmostEqual(record["rate_executed"], -1.7 / 3.0)
        expected = return_to_go_average(
            self.INTERVALS, failed=False, discount_rate=0.5, horizon=3.0, payoff=-2.0
        )
        self.assertAlmostEqual(record["mean_return_to_go"], expected, places=5)

    def test_without_the_undiscounted_reward_the_reward_fields_are_nan(self):
        record = self.run_scripted(failed=True, report=False)
        for key in ("episode_return", "rate_executed", "rate_horizon", "mean_return_to_go"):
            self.assertTrue(math.isnan(record[key]), key)
        self.assertAlmostEqual(record["upright_seconds"], 2.5)
        self.assertAlmostEqual(record["survival_seconds"], 3.0)

    def test_summary(self):
        records = [self.run_scripted(failed=True), self.run_scripted(failed=False)]
        records[1] = dict(records[1], max_hold_seconds=0.5)
        summary = protocol.episode_level_summary(records)
        self.assertEqual(set(summary), set(protocol.EPISODE_LEVEL_KEYS) | {"capture_rate"})
        self.assertAlmostEqual(summary["capture_rate"], 0.5)
        self.assertAlmostEqual(summary["episode_return"], -1.7)
        self.assertAlmostEqual(summary["max_hold_seconds"], 1.25)

    def test_upright_per_testbed(self):
        upright = protocol.is_upright
        self.assertTrue(upright({"name": "windy_pendulum"}, np.array([2 * math.pi + 0.1, 0.0])))
        self.assertFalse(upright({"name": "windy_pendulum"}, np.array([0.3, 0.0])))
        self.assertTrue(upright({"name": "cartpole_swingup"}, np.array([1.0, 0.1, 0.0, 0.0])))
        self.assertFalse(upright({"name": "cartpole_swingup"}, np.array([0.0, math.pi, 0.0, 0.0])))
        self.assertTrue(upright({"name": "acrobot_xk"}, np.array([0.5 * math.pi, 0.1, 0.0, 0.0])))
        self.assertFalse(upright({"name": "acrobot_xk"}, np.array([-0.5 * math.pi, 0.0, 0.0, 0.0])))
        self.assertFalse(upright({"name": "acrobot_xk"}, np.array([0.5 * math.pi, 0.5, 0.0, 0.0])))
        with self.assertRaises(ValueError):
            upright({"name": "unknown"}, np.zeros(2))

    def test_real_episodes_carry_the_fields(self):
        config = windy_config()
        record = protocol.evaluate_fixed_seeds(
            lambda state, deterministic: np.array([0.0, 40.0, 8.0]),
            lambda seed: make_env(config, seed=seed),
            [5],
            discount_rate=config["discount_rate"],
            deterministic=True,
        )[0]
        for key in protocol.EPISODE_LEVEL_KEYS:
            self.assertIn(key, record)
        self.assertAlmostEqual(record["survival_seconds"], record["duration_seconds"])
        self.assertLessEqual(record["upright_seconds"], record["survival_seconds"] + 1e-9)
        env = make_env(config, seed=5)
        env.reset(seed=5)
        env.step_dt(np.array([0.0, 40.0, 8.0]), env.sample_duration())
        reported = getattr(env, "last_undiscounted_reward", None) is not None
        env.close()
        self.assertEqual(math.isnan(record["episode_return"]), not reported)


def fake_result(output, seed, arm_name="implicit_learned_admissions"):
    proto = smoke_protocol()
    proto["output_directory"] = str(output)
    # Feedback arms only: FakeAgent returns feedback decisions.
    arm = {a["name"]: a for a in protocol.all_arms()}[arm_name]
    proto["arms"] = [arm]
    env = make_env(proto["matched"]["testbed_config"], seed=seed)
    agent = FakeAgent(env, [0.0, 40.0, 8.0])
    run = protocol.build_run_matrix(proto)[0]
    run = dict(run, training_seed=seed)
    return run, agent, proto, env


class RunSeedTests(unittest.TestCase):
    def test_budget_checkpoints_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, agent, proto, env = fake_result(tmp, 0)
            original = protocol.make_training_agent
            protocol.make_training_agent = lambda r, e: agent
            try:
                result = protocol.run_seed(run)
            finally:
                protocol.make_training_agent = original
            env.close()
            evaluated = [c["interactions"] for c in result["evaluations"]]
            self.assertEqual(evaluated, [0, 500, 1000, 1500])
            self.assertEqual(agent.interactions, 1500)
            self.assertEqual(len(result["evaluations"][0]["episodes"]), 2)
            self.assertLess(result["counters"]["wall_seconds"], 1000.0)
            self.assertTrue(Path(result["checkpoint_paths"][0]).is_file())
            # The smoke protocol keeps the replay in its checkpoints.
            self.assertIs(agent.saved_replay, True)
            folder = (
                Path(tmp) / "windy_pendulum" / "implicit_learned_admissions" / "seed_0"
            )
            self.assertTrue((folder / reporting.RECORD_NAME).is_file())
            self.assertTrue((folder / "episodes.csv").is_file())
            self.assertEqual(len(result["solver"]), 1)
            self.assertEqual(len(result["admissions"]), 1)
            self.assertNotIn("reward_fit", result)

    def test_checkpoint_replay_false_saves_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, agent, proto, env = fake_result(tmp, 0)
            run = dict(run, protocol=dict(run["protocol"], checkpoint_replay=False))
            original = protocol.make_training_agent
            protocol.make_training_agent = lambda r, e: agent
            try:
                result = protocol.run_seed(run)
            finally:
                protocol.make_training_agent = original
            env.close()
            self.assertTrue(Path(result["checkpoint_paths"][0]).is_file())
            self.assertIs(agent.saved_replay, False)

    def test_reward_fit_records_reach_the_run_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, agent, proto, env = fake_result(tmp, 0)
            agent.reward_records = [{"steps": 3, "rmse_taken": 0.1}]
            original = protocol.make_training_agent
            protocol.make_training_agent = lambda r, e: agent
            try:
                result = protocol.run_seed(run)
            finally:
                protocol.make_training_agent = original
            env.close()
            self.assertEqual(result["reward_fit"], [{"steps": 3, "rmse_taken": 0.1}])

    def test_crash_keeps_checkpoint_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, agent, proto, env = fake_result(tmp, 0)
            learn = agent.learn

            def crashing_learn(total, callback=None):
                def crash(owner, metrics):
                    callback(owner, metrics)
                    if metrics["interactions"] == 700:
                        raise FloatingPointError("non-finite plant loss")

                return learn(total, callback=crash)

            agent.learn = crashing_learn
            original = protocol.make_training_agent
            protocol.make_training_agent = lambda r, e: agent
            try:
                with self.assertRaises(FloatingPointError):
                    protocol.run_seed(run)
            finally:
                protocol.make_training_agent = original
            env.close()
            folder = (
                Path(tmp) / "windy_pendulum" / "implicit_learned_admissions" / "seed_0"
            )
            self.assertTrue((folder / "checkpoint.pt").is_file())
            self.assertFalse((folder / "checkpoint.partial").exists())
            loaded = reporting.load_run_records([str(folder / reporting.RECORD_NAME)])[0]
            evaluated = [c["interactions"] for c in loaded["evaluations"]]
            self.assertEqual(evaluated, [0, 500])

    def run_results(self, output, arms_and_seeds):
        results = []
        original = protocol.make_training_agent
        try:
            for arm, seed in arms_and_seeds:
                run, agent, proto, env = fake_result(output, seed, arm)
                protocol.make_training_agent = lambda r, e, a=agent: a
                results.append(protocol.run_seed(run))
                env.close()
        finally:
            protocol.make_training_agent = original
        return proto, results

    def test_records_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, results = self.run_results(tmp, [("implicit_oracle", 1)])
            paths = reporting.write_run_records(results[0], os.path.join(tmp, "copy"))
            loaded = reporting.load_run_records(paths)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["evaluations"], results[0]["evaluations"])
            self.assertEqual(loaded[0]["solver"], results[0]["solver"])
            self.assertEqual(loaded[0]["counters"], results[0]["counters"])
            self.assertEqual(loaded[0]["run"]["arm"], results[0]["run"]["arm"])
            spec = loaded[0]["run"]["protocol"]["matched"]["testbed_config"]["controller"]
            self.assertIsInstance(spec["decision_low"], np.ndarray)
            with open(paths[0], encoding="utf-8") as handle:
                self.assertIn("units", json.load(handle))

    def test_summary_uses_seeds_as_the_replication_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, results = self.run_results(
                tmp,
                [
                    ("implicit_learned_admissions", 0),
                    ("implicit_learned_admissions", 1),
                    ("explicit_generator", 0),
                ],
            )
        results[0]["evaluations"][-1]["episodes"] = [
            dict(e, discounted_return=value, failed=flag)
            for e, value, flag in zip(
                results[0]["evaluations"][-1]["episodes"], (-1.0, -3.0), (True, False)
            )
        ]
        results[1]["evaluations"][-1]["episodes"] = [
            dict(e, discounted_return=-6.0, failed=False)
            for e in results[1]["evaluations"][-1]["episodes"]
        ]
        summaries = {s["arm"]: s for s in reporting.summarize_arms(results)}
        summary = summaries["implicit_learned_admissions"]
        self.assertEqual(summary["training_seed_count"], 2)
        self.assertEqual(summary["episode_count"], 4)
        self.assertAlmostEqual(summary["mean_discounted_return"], -4.0)
        self.assertAlmostEqual(
            summary["return_standard_error_across_training_seeds"],
            np.std([-2.0, -6.0], ddof=1) / math.sqrt(2.0),
        )
        self.assertAlmostEqual(summary["failure_rate"], 0.25)
        self.assertEqual(summary["interactions"], 1500)
        # Critic, value, actor, plant and reward steps: 1500 each.
        self.assertEqual(summary["gradient_steps"], 7500)
        self.assertEqual(summary["accepted_admissions"], 2)
        self.assertTrue(
            math.isnan(
                summaries["explicit_generator"][
                    "return_standard_error_across_training_seeds"
                ]
            )
        )

    def test_policy_kinds_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, results = self.run_results(tmp, [("implicit_oracle", 0)])
        episodes = results[0]["evaluations"][-1]["episodes"]
        episodes[0] = dict(episodes[0], policy_kind="stochastic")
        kinds = {s["policy_kind"] for s in reporting.summarize_arms(results)}
        self.assertEqual(kinds, {"stochastic", "deterministic_mean"})

    def test_reports_and_plots_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto, results = self.run_results(
                tmp, [("implicit_oracle", 0), ("explicit_rollout", 0)]
            )
            results[0]["plant_calibration"] = [
                {
                    "interactions": 10,
                    "duration_lower_seconds": 0.0,
                    "duration_upper_seconds": 0.01,
                    "sample_count": 5,
                    "state_dimension": 2,
                    "mean_mahalanobis_square": 2.1,
                    "upper_tail_fraction": 0.05,
                    "endpoint_mean_error": None,
                    "endpoint_covariance_error": None,
                    "chi_square_goodness_of_fit_statistic": 1.0,
                    "chi_square_goodness_of_fit_pvalue": 0.5,
                    "duration_jacobian_norm_max": 0.1,
                    "saturated_fraction": 0.0,
                }
            ]
            out = os.path.join(tmp, "report")
            for axis in ("interactions", "gradient_steps", "wall_seconds"):
                paths = reporting.plot_learning_curves(results, out, x_axis=axis)
                self.assertEqual(len(paths), 1)
                self.assertGreater(os.path.getsize(paths[0]), 1000)
            paths = reporting.plot_diagnostics(results, out)
            self.assertEqual(len(paths), 3)
            report = Path(reporting.write_comparison_report(proto, results, out)).read_text()
            self.assertIn("certified_uniform", report)
            self.assertIn("## Conditional bounds", report)
            self.assertIn("## Certificates", report)
            self.assertIn("implicit_oracle", report)
            self.assertIn("None recorded.", report)

    def theory_records(self):
        context = SampledTheoryTests.context
        sampled = diagnostics.summarize_sampled_theory_diagnostics(
            context=context,
            domain="d",
            sampling_description="s",
            candidate_changes=np.ones(1),
            raw_policy_residuals=np.ones(1),
            value_update_increments=[],
            actor_update_increments=[],
        )
        bound = {
            "context": context,
            "evidence": {
                "kind": "assumed_uniform",
                "domain": "all states",
                "justification": "assumed for illustration",
                "failure_probability": None,
            },
            "claim_kind": "conditional",
            "policy_loss_upper_bound": 12.5,
            "soft_return_lower_bound": -3.5,
            "reward_return_lower_bound": None,
            "task_threshold": None,
            "threshold_satisfied": None,
        }
        step = {
            "context": context,
            "parameter_increment": torch.tensor([3.0, 4.0]),
        }
        return [
            {
                "context": context,
                "solve_context": None,
                "sampled": [sampled],
                "optimizer_steps": [step],
                "training_accuracy": None,
                "outer_step": None,
                "original_iteration_budget": None,
                "performance_bound": bound,
            }
        ]

    def test_theory_records_and_figures(self):
        records = self.theory_records()
        with tempfile.TemporaryDirectory() as tmp:
            paths = reporting.write_theory_records(records, tmp)
            with open(paths[0], encoding="utf-8") as handle:
                stored = json.load(handle)["records"][0]
            increment = stored["optimizer_steps"][0]["parameter_increment"]
            self.assertAlmostEqual(increment["__tensor_summary__"]["norm"], 5.0)
            self.assertEqual(len(reporting.plot_theory_diagnostics(records, tmp)), 2)

    def test_report_separates_conditional_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto, results = self.run_results(tmp, [("implicit_oracle", 0)])
            results[0]["theory_records"] = self.theory_records()
            report = Path(reporting.write_comparison_report(proto, results, tmp)).read_text()
        conditional, certificates = report.split("## Certificates")
        self.assertIn("12.5", conditional)
        self.assertNotIn("12.5", certificates)
        self.assertIn("None recorded.", certificates)

    def test_report_only_rebuilds_from_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_results(tmp, [("implicit_oracle", 0)])
            path = str(CONFIGS / "smoke_windy.json")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                main(["--config", path, "--report-only", "--output", tmp, "--seeds", "0"])
            report = Path(tmp) / "windy_pendulum" / "report" / "comparison_windy_pendulum.md"
            self.assertTrue(report.is_file())
            self.assertIn(str(report), printed.getvalue())
            self.assertIn("implicit_oracle", report.read_text())

    def test_report_only_takes_every_arm_present_unless_named(self):
        """Without --arms, report-only rebuilds from every arm's records in the folder, not
        the default (torque) arm set: a folder of impedance-law runs reports them all. With
        --arms only those; a selection with no record is an error, never an empty report."""
        with tempfile.TemporaryDirectory() as tmp:
            self.run_results(tmp, [("implicit_learned_admissions", 0), ("explicit_rollout", 0)])
            path = str(CONFIGS / "smoke_windy.json")
            report = Path(tmp) / "windy_pendulum" / "report" / "comparison_windy_pendulum.md"
            base = ["--config", path, "--report-only", "--output", tmp, "--seeds", "0"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(base)
            text = report.read_text()
            for name in ("implicit_learned_admissions", "explicit_rollout"):
                self.assertIn(name, text)
            with contextlib.redirect_stdout(io.StringIO()):
                main([*base, "--arms", "explicit_rollout"])
            text = report.read_text()
            self.assertIn("explicit_rollout", text)
            self.assertNotIn("implicit_learned_admissions", text)
            for extra in (["--arms", "implicit_torque"], ["--seeds", "5"]):
                with self.subTest(selection=extra), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(FileNotFoundError, "implicit_learned_admissions"):
                        main([*base, *extra])

    def test_optimizer_steps_group_by_context(self):
        first = dict(SampledTheoryTests.context)
        second = dict(first, outer_iteration=5)
        records = protocol._optimizer_step_records(
            [{"context": first}, {"context": first}, {"context": second}]
        )
        self.assertEqual([len(r["optimizer_steps"]) for r in records], [2, 1])


class LoadProtocolTests(unittest.TestCase):
    def raw(self, name="smoke_windy.json"):
        with open(CONFIGS / name, encoding="utf-8") as handle:
            return json.load(handle)

    def load_raw(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(raw, handle)
            return load_protocol(path)

    def test_all_configs_load(self):
        for name, testbed, budget, polyak, seeds in (
            ("windy_pendulum.json", "windy_pendulum", 200000, 0.005, 32),
            ("windy_swingup_tq5.json", "windy_pendulum", 200000, 0.005, 32),
            ("windy_swingup_tq12.json", "windy_pendulum", 200000, 0.005, 32),
            ("acrobot_xk.json", "acrobot_xk", 800000, 0.0125, 32),
            ("cartpole_swingup.json", "cartpole_swingup", 500000, 0.0125, 32),
            ("smoke_windy.json", "windy_pendulum", 1500, 0.005, 2),
        ):
            proto = load_protocol(str(CONFIGS / name))
            self.assertEqual(proto["testbed"], testbed)
            self.assertEqual(proto["matched"]["interaction_budget"], budget)
            self.assertEqual(proto["explicit_config"]["polyak_rate"], polyak)
            self.assertEqual(
                [arm["name"] for arm in proto["arms"]],
                [arm["name"] for arm in protocol.paper_arms()],
            )
            self.assertEqual(len(proto["matched"]["evaluation_reset_seeds"]), seeds)
            self.assertTrue(self.raw(name)["_experiment_choices"])
        self.assertEqual(
            sorted(path.name for path in CONFIGS.glob("*.json")),
            sorted(
                [
                    "windy_pendulum.json",
                    "windy_swingup_tq5.json",
                    "windy_swingup_tq12.json",
                    "acrobot_xk.json",
                    "cartpole_swingup.json",
                    "smoke_windy.json",
                ]
            ),
        )

    def test_reference_run_defaults_in_the_configs(self):
        full = {
            # name: (target entropy, rate steps, actor steps, plant diffusion); the target
            # is -1 per coordinate of the normalized decision box, whatever its units.
            "windy_pendulum.json": (-3.0, 50, 10, "learned"),
            "windy_swingup_tq5.json": (-3.0, 50, 10, "learned"),
            "windy_swingup_tq12.json": (-3.0, 50, 10, "learned"),
            "acrobot_xk.json": (-3.0, 200, 40, "none"),
            "cartpole_swingup.json": (-3.0, 200, 40, "none"),
        }
        for name, (target, rate_steps, actor_steps, diffusion) in full.items():
            proto = load_protocol(str(CONFIGS / name))
            config = proto["implicit_config"]
            controller = proto["matched"]["testbed_config"]["controller"]
            self.assertEqual(
                config["temperature"],
                {"initial_alpha_c": 0.1, "learn_temperature": True, "target_entropy": target},
                name,
            )
            self.assertAlmostEqual(decision_box_entropy_target(controller), target, places=12)
            self.assertEqual(proto["matched"]["temperature_per_second"], 0.1)
            self.assertEqual(proto["training_seeds"], list(range(10)), name)
            self.assertEqual(config["rate"]["rate_steps"], rate_steps, name)
            self.assertEqual(config["rate"]["actor_steps"], actor_steps, name)
            self.assertEqual(config["network"]["diffusion"], diffusion, name)
            self.assertEqual(config["admission"]["plant_update"], "always", name)
            self.assertEqual(config["reward_source"], "learned", name)
            self.assertEqual(config["reward_warm_steps"], 2000, name)
            self.assertEqual(config["plant_warm_steps"], 2000, name)
            self.assertEqual(config["plant_fit_covariance"], "leading_order", name)
            self.assertEqual(config["temperature_every_rows"], 40, name)
            self.assertEqual(config["training"]["validation_fraction"], 0.05, name)
            choices = self.raw(name)["_experiment_choices"]
            for key in (
                "implicit.reward_source",
                "implicit.admission.plant_update",
                "network.diffusion",
                "reference_differences",
            ):
                self.assertIn(key, choices, name)
        smoke = load_protocol(str(CONFIGS / "smoke_windy.json"))["implicit_config"]
        self.assertEqual(smoke["admission"]["plant_update"], "always")
        self.assertEqual(smoke["reward_source"], "learned")

    def test_new_implicit_settings_are_required(self):
        for key in (
            "reward_source",
            "reward_warm_steps",
            "plant_warm_steps",
            "plant_fit_covariance",
            "temperature_every_rows",
        ):
            raw = self.raw()
            del raw["implicit"][key]
            with self.assertRaises(ValueError, msg=key) as caught:
                self.load_raw(raw)
            self.assertIn(key, str(caught.exception))
        raw = self.raw()
        raw["implicit"]["plant_fit_covariance"] = "exact"
        with self.assertRaises(ValueError):
            self.load_raw(raw)

    def test_swingup_configs_declare_the_wrapper_task(self):
        for name, limit in (("windy_swingup_tq5.json", 5.0), ("windy_swingup_tq12.json", 12.0)):
            raw = self.raw(name)
            self.assertEqual(raw["testbed_settings"]["task"], "swingup")
            testbed = load_protocol(str(CONFIGS / name))["matched"]["testbed_config"]
            controller = testbed["controller"]
            np.testing.assert_allclose(controller["decision_low"], [-math.pi, 0.0, 0.0])
            np.testing.assert_allclose(controller["decision_high"], [math.pi, 100.0, 20.0])
            np.testing.assert_allclose(controller["torque_low"], [-limit])
            np.testing.assert_allclose(controller["torque_high"], [limit])
            self.assertEqual(testbed["plant_parameters"]["wind_intensity"], 1.0)
            self.assertEqual(testbed["failure_limits"]["angle"], math.inf)
            self.assertEqual(testbed["failure_limits"]["velocity"], 4.0 * math.pi)
            self.assertEqual(testbed["reset_parameters"]["start_angle"], math.pi)
            env = make_env(testbed, seed=0)
            state, _ = env.reset(seed=0)
            self.assertLessEqual(abs(abs(state[0]) - math.pi), 0.3 + 1e-9)
            self.assertFalse(env.failed(np.array([math.pi / 2 + 0.5, 0.0])))
            env.close()
        balance = load_protocol(str(CONFIGS / "windy_pendulum.json"))["matched"]["testbed_config"]
        self.assertEqual(balance["failure_limits"]["angle"], math.pi / 2)
        self.assertEqual(balance["plant_parameters"]["wind_intensity"], 3.0)
        np.testing.assert_allclose(balance["controller"]["torque_high"], [12.0])

    def test_windy_task_declarations_are_checked(self):
        raw = self.raw("windy_swingup_tq5.json")
        del raw["testbed_settings"]["torque_limit"]
        with self.assertRaises(ValueError):
            self.load_raw(raw)
        raw = self.raw("windy_swingup_tq5.json")
        del raw["testbed_settings"]["reset_parameters"]["start_angle"]
        with self.assertRaises(ValueError):
            self.load_raw(raw)
        raw = self.raw()
        raw["testbed_settings"]["torque_limit"] = 5.0
        with self.assertRaises(ValueError):
            self.load_raw(raw)
        raw = self.raw()
        raw["testbed_settings"]["task"] = "spin"
        with self.assertRaises(ValueError):
            self.load_raw(raw)
        raw = self.raw()
        del raw["testbed_settings"]["task"]
        with self.assertRaises(ValueError) as caught:
            self.load_raw(raw)
        self.assertIn("task", str(caught.exception))

    def test_paper_defaults_in_the_full_configs(self):
        windy = load_protocol(str(CONFIGS / "windy_pendulum.json"))
        testbed = windy["matched"]["testbed_config"]
        self.assertEqual(testbed["physics_dt"], 0.001)
        self.assertEqual(testbed["discount_rate"], 0.1)
        config = windy["implicit_config"]
        self.assertEqual(config["model_step"], 0.001)
        self.assertEqual(config["solver"]["resolvent_step"], 1.0)
        self.assertEqual(config["admission"]["margin_z"], 1.645)
        self.assertEqual(config["admission"]["initial_rows"], 32)
        self.assertEqual(config["admission"]["initial_paths"], 4)
        self.assertEqual(config["admission"]["pilot_batches"], 2)
        self.assertEqual(config["admission"]["real_row_budget"], 512)
        self.assertEqual(config["admission"]["simulated_path_budget"], 65536)
        self.assertEqual(config["admission"]["efolds"], 3.0)
        self.assertEqual(config["network"]["mass_floor"], 0.01)
        self.assertEqual(config["network"]["diffusion_floor"], 0.001)
        self.assertEqual(config["network"]["plant_hidden_sizes"], (64, 64))
        self.assertEqual(config["training"]["learning_rate"], 3e-4)
        self.assertEqual(config["training"]["gradient_clip"], 10.0)
        self.assertEqual(config["training"]["batch_size"], 256)
        acrobot = load_protocol(str(CONFIGS / "acrobot_xk.json"))
        self.assertEqual(acrobot["matched"]["testbed_config"]["discount_rate"], 0.5)
        self.assertEqual(acrobot["implicit_config"]["network"]["damping_rank"], 2)

    def test_configs_build_their_environments(self):
        for name in ("windy_pendulum.json", "acrobot_xk.json"):
            proto = load_protocol(str(CONFIGS / name))
            env = make_env(proto["matched"]["testbed_config"], seed=0)
            state, _ = env.reset(seed=0)
            low = env.action_space.low
            transition = env.step_dt(np.asarray(low), env.sample_duration())
            self.assertEqual(transition["state"].shape, state.shape)
            env.close()

    def test_missing_keys_are_rejected(self):
        for path in (
            ("model_step",),
            ("testbed_settings", "tail_split"),
            ("network", "plant_hidden_sizes"),
            ("training", "learning_rate"),
            ("implicit", "solver", "inner_steps"),
            ("implicit", "rate", "rollout_weight"),
            ("protocol", "training_seeds"),
            ("explicit", "polyak_rate"),
        ):
            raw = self.raw()
            target = raw
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            with self.assertRaises(ValueError, msg=str(path)) as caught:
                self.load_raw(raw)
            self.assertIn(path[-1], str(caught.exception))

    def test_unknown_keys_are_rejected(self):
        raw = self.raw()
        raw["training"]["learning_rat"] = 1.0
        with self.assertRaises(ValueError):
            self.load_raw(raw)

    def test_inner_steps_set_k_in_every_config(self):
        for name, steps in (
            ("windy_pendulum.json", 400),
            ("windy_swingup_tq5.json", 400),
            ("windy_swingup_tq12.json", 400),
            ("acrobot_xk.json", 400),
            ("cartpole_swingup.json", 400),
            ("smoke_windy.json", 20),
        ):
            solver = load_protocol(str(CONFIGS / name))["implicit_config"]["solver"]
            self.assertEqual(
                solver,
                {
                    "resolvent_step": 1.0,
                    "policy_draws": 4,
                    "inner_steps": steps,
                    "residual_estimator": "all_pairs",
                },
                name,
            )
            choices = self.raw(name)["_experiment_choices"]
            self.assertIn("implicit.solver.inner_steps", choices)
            self.assertFalse(
                [key for key in choices if "value_accuracy" in key or "audit" in key]
            )

    def test_removed_stopping_keys_are_rejected(self):
        for key, value in (
            ("value_accuracy", 0.1),
            ("max_inner_steps", 400),
            ("stop_check_every", 10),
            ("stopping", "fixed_steps"),
            ("fixed_inner_steps", 200),
            ("value_audit", {"margin": 2.0}),
        ):
            raw = self.raw()
            raw["implicit"]["solver"][key] = value
            with self.assertRaises(ValueError, msg=key) as caught:
                self.load_raw(raw)
            self.assertIn(key, str(caught.exception))
        raw = self.raw()
        raw["protocol"]["ablation_inner_iterations"] = 200
        with self.assertRaises(ValueError) as caught:
            self.load_raw(raw)
        self.assertIn("ablation_inner_iterations", str(caught.exception))

    def test_invalid_values_are_rejected(self):
        raw = self.raw()
        raw["implicit"]["solver"]["policy_draws"] = 1
        with self.assertRaises(ValueError):
            self.load_raw(raw)
        raw = self.raw()
        raw["implicit"]["solver"]["inner_steps"] = 0
        with self.assertRaises(ValueError):
            self.load_raw(raw)

    def test_mismatched_temperature_is_rejected(self):
        raw = self.raw()
        raw["protocol"]["temperature_per_second"] = 0.2
        with self.assertRaises(ValueError):
            self.load_raw(raw)

    def test_command_line_overrides(self):
        path = str(CONFIGS / "smoke_windy.json")
        args = parse_args(
            [
                "--config",
                path,
                "--arms",
                "implicit_oracle",
                "explicit_rollout",
                "--seeds",
                "4",
                "5",
                "--output",
                "elsewhere",
                "--interactions",
                "77",
            ]
        )
        proto = apply_overrides(load_protocol(path), args)
        self.assertEqual(
            {arm["name"] for arm in proto["arms"]},
            {"implicit_oracle", "explicit_rollout"},
        )
        self.assertEqual(proto["training_seeds"], [4, 5])
        self.assertEqual(proto["output_directory"], "elsewhere")
        self.assertEqual(proto["matched"]["interaction_budget"], 77)


if __name__ == "__main__":
    unittest.main()
