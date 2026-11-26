import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch as th

from algorithms.ct_sac_implicit import (
    ImplicitTorqueCTSACBaseline,
    implicit_bracket,
    resolvent_residuals,
)
from algorithms.ph_baseline import (
    TorqueCTSACBaseline,
    _adapt_transition,
    end_state_reward_rate,
)
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import (
    CT_SAC_REWARD_SOURCES,
    CT_SAC_REWARD_UNITS,
    DEFAULT_CT_SAC_REWARD_SOURCE,
    DEFAULT_CT_SAC_REWARD_UNITS,
    check_ct_sac_reward_source,
    check_ct_sac_reward_units,
    ct_sac_reward_source_setting,
    ct_sac_reward_units_setting,
    windy_pendulum_config,
)
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.diagnostics import TrainingCounters
from evaluations.ph_sac.reporting import RECORD_NAME, load_run_records
from tests.test_ph_resume import ExactRestartMixin, run_captured

SMOKE_CONFIG = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac" / "smoke_windy.json"
)


def _testbed(**overrides):
    config = windy_pendulum_config(
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
    config.update(overrides)
    return config


def _training(**overrides):
    training = {
        "seed": 3,
        "device": "cpu",
        "replay_capacity": 2000,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "gradient_clip": 10.0,
        "warmup_episodes": 0,
        "validation_fraction": 0.0,
    }
    training.update(overrides)
    return training


def _baseline(
    testbed=None,
    training=None,
    *,
    alpha_c=0.0,
    learn_temperature=False,
    target_entropy=-3.0,
    **kwargs,
):
    testbed = testbed or _testbed()
    env = make_env(testbed, decision_mode="torque", seed=11)
    # The CT-PH-SAC arms' target, -1 per coordinate of their 3-dimensional normalized
    # decision box; the baseline ignores it and uses -1 per torque coordinate.
    temperature = {
        "initial_alpha_c": alpha_c,
        "learn_temperature": learn_temperature,
        "target_entropy": target_entropy,
    }
    kwargs.setdefault("polyak_rate", 0.005)
    kwargs.setdefault("expectation_samples", 1)
    kwargs.setdefault("net_arch", (32, 32))
    return TorqueCTSACBaseline(env, training or _training(), temperature, **kwargs)


def _spy_step_dt(env):
    """Record the wrapped environment's transitions without changing them."""
    transitions = []
    step_dt = env.step_dt

    def wrapped(decision, duration):
        transition = step_dt(decision, duration)
        transitions.append(transition)
        return transition

    env.step_dt = wrapped
    return transitions


class TestConversions(unittest.TestCase):
    def test_gamma_power_u_is_physical_discount(self):
        baseline = _baseline()
        algorithm = baseline.algorithm
        self.assertAlmostEqual(algorithm.time_rescale, 1.0 / 0.01)
        for duration in (0.002, 0.01, 0.018):
            u = duration * algorithm.time_rescale
            self.assertAlmostEqual(u, baseline.normalized_durations(duration))
            self.assertAlmostEqual(math.exp(-algorithm.beta * u), math.exp(-0.1 * duration))
            self.assertAlmostEqual(algorithm.gamma**u, math.exp(-0.1 * duration))

    def test_temperature_is_alpha_c_times_reference_period(self):
        # CT-SAC charges u alpha for an interval of u = Delta / T periods: alpha = alpha_c T.
        baseline = _baseline(alpha_c=0.3)
        self.assertAlmostEqual(baseline.env.config["reference_period"], 0.01)
        self.assertAlmostEqual(baseline.period_temperature(0.3), 0.3 * 0.01)
        self.assertAlmostEqual(baseline.algorithm.alpha, 0.3 * 0.01)
        self.assertIsNone(baseline.algorithm.log_alpha)

    def test_learned_temperature_starts_at_matched_value(self):
        baseline = _baseline(alpha_c=0.3, learn_temperature=True)
        self.assertIsNotNone(baseline.algorithm.alpha_optimizer)
        self.assertAlmostEqual(baseline.period_temperature(0.3), 0.3 * 0.01)
        self.assertAlmostEqual(
            baseline.algorithm.log_alpha.exp().item(),
            baseline.period_temperature(0.3),
            places=6,
        )

    def test_expectation_samples_other_than_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expectation_samples"):
            _baseline(expectation_samples=2)

    def test_feedback_environment_is_rejected(self):
        env = make_env(_testbed(), decision_mode="feedback", seed=0)
        temperature = {
            "initial_alpha_c": 0.0,
            "learn_temperature": False,
            "target_entropy": -1.0,
        }
        with self.assertRaisesRegex(ValueError, "torque-mode"):
            TorqueCTSACBaseline(
                env, _training(), temperature, polyak_rate=0.005, expectation_samples=1
            )

    def test_actor_and_critics_work_in_unit_actions(self):
        baseline = _baseline()
        actor = baseline.algorithm.model.actor
        adapter = baseline.algorithm.env
        self.assertEqual(adapter.action_space.low.tolist(), [-1.0])
        self.assertEqual(adapter.action_space.high.tolist(), [1.0])
        self.assertEqual(actor.action_low.tolist(), [-1.0])
        self.assertEqual(actor.action_high.tolist(), [1.0])
        self.assertEqual(actor._scale_action(th.tensor([[1.0]])).item(), 1.0)
        self.assertEqual(actor._scale_action(th.tensor([[-1.0]])).item(), -1.0)
        np.testing.assert_array_equal(baseline.torque_limit, [12.0])
        # The wrapped environment keeps its physical torque box.
        self.assertEqual(baseline.env.action_space.high.tolist(), [12.0])

    def test_motor_receives_unit_action_times_the_limit(self):
        baseline = _baseline()
        adapter = baseline.algorithm.env
        transitions = _spy_step_dt(baseline.env)
        adapter.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(20):
            u = rng.uniform(-1.0, 1.0, size=1)
            _, _, terminated, truncated, info, _ = adapter._step_physics(u, 0.01)
            self.assertIs(info["transition"], transitions[-1])
            np.testing.assert_allclose(transitions[-1]["decision"], 12.0 * u)
            if terminated or truncated:
                break
        adapter._step_physics(np.array([1.0]), 0.002)
        self.assertEqual(transitions[-1]["decision"].tolist(), [12.0])

    def test_predict_is_policy_action_times_the_limit(self):
        baseline = _baseline()
        states = np.array([[0.1, -0.2], [0.4, 1.0], [-0.3, 0.05]])
        policy = baseline.algorithm._policy_act(
            states.astype(np.float32), deterministic=True
        )
        torque = baseline.predict(states, deterministic=True)
        np.testing.assert_allclose(torque, 12.0 * np.asarray(policy, dtype=np.float64))
        self.assertTrue(np.all(np.abs(torque) <= 12.0))

    def test_target_entropy_is_auto_in_unit_actions(self):
        for target in (-3.0, -6.0, 2.52):
            baseline = _baseline(alpha_c=0.1, learn_temperature=True, target_entropy=target)
            self.assertEqual(baseline.algorithm.target_entropy, -1.0)
            self.assertIsNotNone(baseline.algorithm.alpha_optimizer)

    def test_temperature_is_tuned_at_every_gradient_step(self):
        baseline = _baseline(
            _testbed(episode_duration=0.05),
            _training(batch_size=4),
            alpha_c=0.1,
            learn_temperature=True,
        )
        start = baseline.algorithm.log_alpha.item()
        baseline.learn(20)
        counters = baseline.counters()
        self.assertGreater(counters["critic_gradient_steps"], 0)
        self.assertEqual(
            counters["temperature_gradient_steps"], counters["critic_gradient_steps"]
        )
        self.assertNotEqual(baseline.algorithm.log_alpha.item(), start)

    def test_asymmetric_motor_limit_is_rejected(self):
        testbed = _testbed()
        testbed["controller"] = dict(
            testbed["controller"],
            torque_low=np.array([-10.0]),
            torque_high=np.array([12.0]),
        )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            _baseline(testbed)

    def test_reward_rate_interior_and_failure_rows(self):
        baseline = _baseline()
        lam, T = 0.1, 0.01
        delta, r = 0.004, -0.0123
        F = -1.0 / lam
        charge = math.exp(-lam * delta) * F
        batch = {
            "durations": th.tensor([[delta], [delta]]),
            "rewards": th.tensor([[r], [r + charge]]),
            "terminated": th.tensor([[0.0], [1.0]]),
            "next_states": th.tensor([[0.1, 0.0], [1.6, 0.0]]),
        }
        converted = baseline.reward_rate(batch)
        u = delta / T
        self.assertAlmostEqual(converted[0].item(), r / u, places=5)
        self.assertAlmostEqual(converted[1].item(), r / u + charge, places=4)

    def test_constant_rate_maps_to_period_times_rate(self):
        baseline = _baseline()
        rate, delta = -0.7, 0.01
        interval = rate * (1.0 - math.exp(-0.1 * delta)) / 0.1
        adapted = baseline.adapt_transition(
            {
                "state": np.zeros(2),
                "decision": np.zeros(1),
                "duration": delta,
                "reward": interval,
                "next_state": np.zeros(2),
                "terminated": False,
                "truncated": False,
                "time": 0.0,
                "next_time": delta,
            }
        )
        # T*rate up to the fixed-interval factor c_Delta=w_lambda(Delta)/Delta.
        c_delta = (1.0 - math.exp(-0.1 * delta)) / (0.1 * delta)
        self.assertAlmostEqual(adapted["reward"], rate * 0.01 * c_delta, places=12)
        self.assertAlmostEqual(adapted["reward"], rate * 0.01, delta=4e-6)
        self.assertAlmostEqual(adapted["normalized_duration"], 1.0)


class TestCriticTarget(unittest.TestCase):
    def _target_for(self, transition, value, next_value):
        baseline = _baseline(training=_training(batch_size=1))
        algorithm = baseline.algorithm
        adapted = baseline.adapt_transition(transition)
        algorithm._store_transition(
            obs=np.asarray(transition["state"], dtype=np.float32),
            action=np.asarray(transition["decision"], dtype=np.float32),
            reward=adapted["reward"],
            done=adapted["terminated"] or adapted["truncated"],
            next_obs=np.asarray(transition["next_state"], dtype=np.float32),
            t=transition["time"],
            next_t=transition["next_time"],
            infos={"adapted": adapted},
        )

        start = th.as_tensor(transition["state"], dtype=th.float32)

        def target_min_q(obs, _actions):
            at_start = (obs == start).all(dim=-1, keepdim=True)
            return th.where(at_start, th.tensor(value), th.tensor(next_value))

        algorithm.model.target_min_q = target_min_q
        real_loss = th.nn.functional.mse_loss
        targets = []

        def capture(prediction, target):
            targets.append(target.detach().clone())
            return real_loss(prediction, target)

        with mock.patch("algorithms.ct_sac.F.mse_loss", side_effect=capture):
            algorithm.train(gradient_steps=1, batch_size=1)
        return targets[0].item()

    def test_interior_row_matches_hand_computation(self):
        lam, T, delta, r = 0.1, 0.01, 0.004, -0.0123
        value, next_value = -3.0, -3.5
        transition = {
            "state": np.array([0.1, 0.2]),
            "decision": np.array([1.0]),
            "duration": delta,
            "reward": r,
            "next_state": np.array([0.15, 0.3]),
            "terminated": False,
            "truncated": False,
            "time": 7.0,
            "next_time": 7.0 + delta,
        }
        u = delta / T
        expected = r / u + value + (math.exp(-lam * delta) * next_value - value) / u
        self.assertAlmostEqual(
            self._target_for(transition, value, next_value), expected, places=3
        )

    def test_truncated_row_still_bootstraps(self):
        lam, T, delta, r = 0.1, 0.01, 0.004, -0.0123
        value, next_value = -3.0, -3.5
        transition = {
            "state": np.array([0.1, 0.2]),
            "decision": np.array([1.0]),
            "duration": delta,
            "reward": r,
            "next_state": np.array([0.15, 0.3]),
            "terminated": False,
            "truncated": True,
            "time": 7.0,
            "next_time": 7.0 + delta,
        }
        u = delta / T
        expected = r / u + value + (math.exp(-lam * delta) * next_value - value) / u
        self.assertAlmostEqual(
            self._target_for(transition, value, next_value), expected, places=3
        )

    def test_failure_row_keeps_charge_in_value_units(self):
        lam, T, delta, interior = 0.1, 0.01, 0.004, -0.0123
        charge = math.exp(-lam * delta) * (-1.0 / lam)
        transition = {
            "state": np.array([1.5, 2.0]),
            "decision": np.array([12.0]),
            "duration": delta,
            "reward": interior + charge,
            "next_state": np.array([1.6, 2.1]),
            "terminated": True,
            "truncated": False,
            "time": 7.0,
            "next_time": 7.0 + delta,
        }
        expected = interior / (delta / T) + charge
        self.assertAlmostEqual(
            self._target_for(transition, -3.0, -3.5), expected, places=3
        )
        self.assertAlmostEqual(expected, -10.0 - 0.03, delta=0.01)


class TestStoredRows(unittest.TestCase):
    def _collect_only(self, testbed, interactions, action=None):
        baseline = _baseline(testbed, _training(warmup_episodes=10**6))
        transitions = _spy_step_dt(baseline.env)
        if action is not None:
            baseline.algorithm.env.action_space.sample = lambda: np.array([action])
        baseline.learn(interactions)
        return baseline, transitions

    def test_truncated_episode_stores_no_done_and_final_observation(self):
        baseline, transitions = self._collect_only(
            _testbed(episode_duration=0.05), 60
        )
        buffer = baseline.algorithm.replay_buffer
        self.assertEqual(buffer.size(), 60)
        truncated = [i for i, t in enumerate(transitions) if t["truncated"]]
        self.assertGreater(len(truncated), 2)
        self.assertEqual(baseline.counters()["critic_gradient_steps"], 0)
        for i, transition in enumerate(transitions):
            self.assertEqual(buffer.dones[i, 0], float(transition["terminated"]))
            np.testing.assert_allclose(
                buffer.next_observations[i, 0],
                transition["next_state"].astype(np.float32),
            )
            self.assertAlmostEqual(
                float(buffer.dt[i, 0]), transition["duration"], places=6
            )
        for i in truncated:
            self.assertEqual(buffer.dones[i, 0], 0.0)
            self.assertAlmostEqual(transitions[i]["next_time"], 0.05, places=9)

    def test_failure_stores_done_and_undiluted_charge(self):
        baseline, transitions = self._collect_only(_testbed(), 300, action=1.0)
        self.assertTrue(all(t["decision"].tolist() == [12.0] for t in transitions))
        buffer = baseline.algorithm.replay_buffer
        failed = [i for i, t in enumerate(transitions) if t["terminated"]]
        self.assertGreater(len(failed), 0)
        for i in failed:
            self.assertEqual(buffer.dones[i, 0], 1.0)
            self.assertLess(buffer.rewards[i, 0], -9.9)
            self.assertGreater(buffer.rewards[i, 0], -11.0)
            self.assertGreaterEqual(
                max(
                    abs(transitions[i]["next_state"][0]) / (math.pi / 2),
                    abs(transitions[i]["next_state"][1]) / (4 * math.pi),
                ),
                1.0,
            )
        interior = [i for i, t in enumerate(transitions) if not t["terminated"]]
        self.assertTrue(all(buffer.dones[i, 0] == 0.0 for i in interior))
        self.assertGreater(buffer.rewards[interior, 0].min(), -0.1)

    def test_warmup_episodes_delay_training_and_policy(self):
        baseline = _baseline(
            _testbed(episode_duration=0.05),
            _training(warmup_episodes=2, batch_size=4),
        )
        transitions = _spy_step_dt(baseline.env)
        baseline.learn(40)
        second_end = [i for i, t in enumerate(transitions) if t["truncated"]][1]
        self.assertEqual(baseline.algorithm.learning_starts, 0)
        self.assertEqual(
            baseline.counters()["critic_gradient_steps"], 40 - second_end
        )


class TestLearnAndPersistence(unittest.TestCase):
    def test_learn_counters_callback_and_resume(self):
        baseline = _baseline(alpha_c=0.1)
        transitions = _spy_step_dt(baseline.env)
        seen = []
        baseline.learn(300, callback=lambda agent, metrics: seen.append(dict(metrics)))
        counters = baseline.counters()
        self.assertEqual(set(counters), set(TrainingCounters.__annotations__))
        self.assertEqual(counters["interactions"], 300)
        self.assertEqual([m["interactions"] for m in seen], list(range(1, 301)))
        self.assertEqual(counters["critic_gradient_steps"], 300 - 31)
        self.assertEqual(counters["actor_gradient_steps"], 300 - 31)
        self.assertEqual(counters["temperature_gradient_steps"], 0)
        for key in (
            "value_gradient_steps",
            "rate_gradient_steps",
            "plant_gradient_steps",
            "model_rollout_paths",
        ):
            self.assertEqual(counters[key], 0)
        self.assertAlmostEqual(
            counters["simulated_seconds"], sum(t["duration"] for t in transitions)
        )
        self.assertGreater(counters["wall_seconds"], 0.0)
        for name in (
            "solver_records",
            "admission_records",
            "oracle_records",
            "calibration_records",
        ):
            self.assertEqual(getattr(baseline, name), [])

        last = transitions[-1]
        self.assertFalse(last["terminated"] or last["truncated"])
        baseline.learn(310)
        self.assertEqual(baseline.counters()["interactions"], 310)
        self.assertEqual(baseline.counters()["critic_gradient_steps"], 310 - 31)
        np.testing.assert_allclose(
            transitions[300]["state"], last["next_state"], atol=1e-5
        )

    def test_save_load_round_trip(self):
        baseline = _baseline(alpha_c=0.1, learn_temperature=True)
        baseline.learn(60)
        fresh = _baseline(alpha_c=0.1, learn_temperature=True)
        fresh_state = fresh.algorithm.log_alpha.item()
        states = np.array([[0.1, -0.2], [0.4, 1.0], [-0.3, 0.05]])
        before = baseline.predict(states, deterministic=True)
        self.assertFalse(np.allclose(fresh.predict(states, deterministic=True), before))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.pt"
            baseline.save(path)
            fresh.load(path)
        np.testing.assert_array_equal(fresh.predict(states, deterministic=True), before)
        self.assertEqual(fresh.counters()["interactions"], 60)
        self.assertEqual(
            fresh.counters()["critic_gradient_steps"],
            baseline.counters()["critic_gradient_steps"],
        )
        self.assertNotEqual(fresh.algorithm.log_alpha.item(), fresh_state)
        self.assertEqual(
            fresh.algorithm.log_alpha.item(), baseline.algorithm.log_alpha.item()
        )

    def test_save_accepts_the_replay_flag(self):
        """The protocol passes include_replay; this save holds no replay either way."""
        baseline = _baseline()
        baseline.learn(20)
        states = np.array([[0.1, -0.2], [0.4, 1.0]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.pt"
            baseline.save(path, include_replay=False)
            fresh = _baseline()
            fresh.load(path)
        self.assertEqual(fresh.counters()["interactions"], 20)
        np.testing.assert_array_equal(
            fresh.predict(states, deterministic=True),
            baseline.predict(states, deterministic=True),
        )

    def test_predict_returns_torque_inside_box(self):
        baseline = _baseline()
        action = baseline.predict(np.array([0.1, 0.0]), deterministic=False)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-12.0 <= action[0] <= 12.0)


# ------------------------------------------------------------------------- reward units

# The windy pendulum of _testbed: T = 10 ms, lambda = 0.1 /s, failure payoff -1 / lambda.
REFERENCE_PERIOD = 0.01
DISCOUNT_RATE = 0.1
# The sidecar keys of save before ct_sac_reward_units existed.
SIDECAR_KEYS = {
    "interactions", "simulated_seconds", "gradient_updates", "finished_episodes",
    "learning_starts", "wall_seconds", "alpha", "log_alpha",
}


def _interval(reward, *, terminated, next_state, duration=0.004):
    """One environment transition of the windy pendulum's shapes (2 states, 1 torque)."""
    return {
        "state": np.array([0.1, 0.2]),
        "decision": np.array([1.0]),
        "duration": duration,
        "reward": reward,
        "next_state": np.asarray(next_state, dtype=np.float64),
        "terminated": terminated,
        "truncated": False,
        "time": 7.0,
        "next_time": 7.0 + duration,
    }


def _smoke_runs(units=None, names=("ct_sac_torque", "implicit_torque")):
    """The smoke protocol's runs of the named arms, by name; units, when given, is the
    protocol's ct_sac_reward_units."""
    proto = run_ph_sac.load_protocol(str(SMOKE_CONFIG))
    if units is not None:
        proto["ct_sac_reward_units"] = units
    proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in names]
    P.validate_matched_quantities(proto)
    return {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}


class TestRewardUnits(unittest.TestCase):
    """ct_sac_reward_units (common.ph_config.CT_SAC_REWARD_UNITS): "reference_period"
    divides every reward, failure charge and the temperature the legacy learner sees by T;
    "second", the default, is the code before the setting."""

    def test_the_setting_and_its_validation(self):
        self.assertEqual(CT_SAC_REWARD_UNITS, ("second", "reference_period"))
        self.assertEqual(DEFAULT_CT_SAC_REWARD_UNITS, "second")
        for units in CT_SAC_REWARD_UNITS:
            self.assertEqual(check_ct_sac_reward_units(units), units)
            self.assertEqual(ct_sac_reward_units_setting({"ct_sac_reward_units": units}), units)
        self.assertEqual(ct_sac_reward_units_setting({}), "second")
        for bad in ("per_step", "Second", "", None, 1.0):
            with self.subTest(units=bad):
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                    check_ct_sac_reward_units(bad)
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                    ct_sac_reward_units_setting({"ct_sac_reward_units": bad})
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                    _baseline(reward_units=bad)
        # The protocol: validated when present, absent by default.
        proto = run_ph_sac.load_protocol(str(SMOKE_CONFIG))
        self.assertNotIn("ct_sac_reward_units", proto)
        for units in CT_SAC_REWARD_UNITS:
            proto["ct_sac_reward_units"] = units
            P.validate_matched_quantities(proto)
        proto["ct_sac_reward_units"] = "per_step"
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
            P.validate_matched_quantities(proto)

    def test_json_key_and_command_line(self):
        raw = json.loads(SMOKE_CONFIG.read_text())
        raw["protocol"]["ct_sac_reward_units"] = "reference_period"
        proto = run_ph_sac.protocol_from_config(copy.deepcopy(raw), "smoke")
        self.assertEqual(proto["ct_sac_reward_units"], "reference_period")
        raw["protocol"]["ct_sac_reward_units"] = "per_step"
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
            run_ph_sac.protocol_from_config(copy.deepcopy(raw), "smoke")
        path = str(SMOKE_CONFIG)
        self.assertNotIn("ct_sac_reward_units", run_ph_sac.load_protocol(path))
        args = run_ph_sac.parse_args(["--config", path])
        self.assertIsNone(args.ct_sac_reward_units)
        self.assertNotIn(
            "ct_sac_reward_units",
            run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args),
        )
        args = run_ph_sac.parse_args(
            ["--config", path, "--ct-sac-reward-units", "reference_period"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual(proto["ct_sac_reward_units"], "reference_period")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_ph_sac.parse_args(["--config", path, "--ct-sac-reward-units", "per_step"])

    def test_adapted_reward_is_the_default_divided_by_T(self):
        default = _baseline()
        scaled = _baseline(reward_units="reference_period")
        self.assertEqual(default.reward_units, "second")
        self.assertEqual(scaled.reward_units, "reference_period")
        self.assertEqual(scaled.algorithm.env.reward_units, "reference_period")
        T = scaled.reference_period
        self.assertEqual(T, REFERENCE_PERIOD)
        delta, interior = 0.004, -0.0123
        charge = math.exp(-DISCOUNT_RATE * delta) * (-1.0 / DISCOUNT_RATE)
        rows = {
            "interior": _interval(interior, terminated=False, next_state=[0.15, 0.3]),
            "failure": _interval(interior + charge, terminated=True, next_state=[1.6, 2.1]),
        }
        for name, transition in rows.items():
            with self.subTest(row=name):
                off = default.adapt_transition(transition)
                on = scaled.adapt_transition(transition)
                # Exactly the default's value divided by T; every other field the default's.
                self.assertEqual(on["reward"], off["reward"] / T)
                self.assertEqual(set(on), set(off))
                for key in off:
                    if key != "reward":
                        self.assertIs(type(on[key]), type(off[key]), key)
                        np.testing.assert_array_equal(on[key], off[key], err_msg=key)
        # By hand: about the rate, r / Delta, and the charge divided by T in a failed row.
        on = scaled.adapt_transition(rows["interior"])
        self.assertAlmostEqual(on["reward"], interior / delta, places=9)
        on = scaled.adapt_transition(rows["failure"])
        self.assertAlmostEqual(on["reward"], interior / delta + charge / T, places=8)

    def test_constant_rate_maps_to_the_rate(self):
        """A rate of 1 per second (here -0.7) is a reward of about 1 (-0.7) per period, up
        to the fixed-interval factor c_Delta = w_lambda(Delta) / Delta."""
        baseline = _baseline(reward_units="reference_period")
        rate, delta = -0.7, 0.01
        interval = rate * (1.0 - math.exp(-DISCOUNT_RATE * delta)) / DISCOUNT_RATE
        adapted = baseline.adapt_transition(
            _interval(interval, terminated=False, next_state=[0.0, 0.0], duration=delta)
        )
        c_delta = (1.0 - math.exp(-DISCOUNT_RATE * delta)) / (DISCOUNT_RATE * delta)
        self.assertAlmostEqual(adapted["reward"], rate * c_delta, places=12)
        self.assertAlmostEqual(adapted["reward"], rate, delta=4e-4)
        self.assertAlmostEqual(adapted["normalized_duration"], 1.0)

    def test_reward_rate_is_the_default_divided_by_T(self):
        default = _baseline()
        scaled = _baseline(reward_units="reference_period")
        T = scaled.reference_period
        delta, r = 0.004, -0.0123
        charge = math.exp(-DISCOUNT_RATE * delta) * (-1.0 / DISCOUNT_RATE)
        batch = {
            "durations": th.tensor([[delta], [delta], [0.018]]),
            "rewards": th.tensor([[r], [r + charge], [2.0 * r]]),
            "terminated": th.tensor([[0.0], [1.0], [0.0]]),
            "next_states": th.tensor([[0.1, 0.0], [1.6, 0.0], [0.2, 0.1]]),
        }
        off = default.reward_rate(batch)
        on = scaled.reward_rate(batch)
        self.assertEqual(on.shape, (3, 1))
        self.assertTrue(th.equal(on, off / T))
        self.assertAlmostEqual(on[0].item(), r / delta, places=4)
        self.assertAlmostEqual(on[1].item(), r / delta + charge / T, places=2)
        self.assertAlmostEqual(on[2].item(), 2.0 * r / 0.018, places=4)

    def test_the_legacy_replay_holds_the_divided_rewards(self):
        """The path that fills the legacy replay (the adapter's _step_physics): the same
        random decisions (full torque, so the pendulum fails) give the same rows, with every
        reward divided by T, the failure charge too."""
        stored = {}
        for units in CT_SAC_REWARD_UNITS:
            baseline = _baseline(
                _testbed(), _training(warmup_episodes=10**6), reward_units=units
            )
            transitions = _spy_step_dt(baseline.env)
            baseline.algorithm.env.action_space.sample = lambda: np.array([1.0])
            baseline.learn(300)
            self.assertEqual(baseline.counters()["critic_gradient_steps"], 0)
            stored[units] = (baseline.algorithm.replay_buffer, transitions)
        (off, transitions), (on, _) = stored["second"], stored["reference_period"]
        for name in ("observations", "next_observations", "actions", "dones", "dt"):
            np.testing.assert_array_equal(getattr(on, name)[:300], getattr(off, name)[:300], name)
        np.testing.assert_allclose(
            on.rewards[:300], off.rewards[:300] / REFERENCE_PERIOD, rtol=1e-6
        )
        failed = [i for i, t in enumerate(transitions) if t["terminated"]]
        self.assertGreater(len(failed), 0)
        for i in failed:
            # The default's undiluted charge, about -10, divided by T.
            self.assertLess(on.rewards[i, 0], -990.0)
            self.assertGreater(on.rewards[i, 0], -1100.0)

    def test_unit_temperature_is_ct_sacs_alpha_one(self):
        for learn in (False, True):
            with self.subTest(learn_temperature=learn):
                baseline = _baseline(
                    alpha_c=1.0, learn_temperature=learn, reward_units="reference_period"
                )
                algorithm = baseline.algorithm
                self.assertEqual(baseline.period_temperature(1.0), 1.0)
                self.assertEqual(algorithm.alpha, 1.0)
                if learn:
                    self.assertIsNotNone(algorithm.alpha_optimizer)
                    self.assertEqual(algorithm.log_alpha.item(), 0.0)
                    self.assertEqual(algorithm.log_alpha.exp().item(), 1.0)
                else:
                    self.assertIsNone(algorithm.log_alpha)
                    self.assertEqual(algorithm.alpha_tensor.item(), 1.0)
                diagnostics = baseline.policy_diagnostics()
                self.assertEqual(diagnostics["alpha_c"], 1.0)
                self.assertEqual(diagnostics["alpha_decision"], 1.0)

    def test_recorded_alpha_c_is_per_second_under_both_units(self):
        default = _baseline(alpha_c=0.3)
        scaled = _baseline(alpha_c=0.3, reward_units="reference_period")
        self.assertEqual(default.algorithm.alpha, 0.3 * REFERENCE_PERIOD)
        self.assertEqual(scaled.algorithm.alpha, 0.3)
        self.assertEqual(scaled.period_temperature(0.3), 0.3)
        off, on = default.policy_diagnostics(), scaled.policy_diagnostics()
        self.assertAlmostEqual(off["alpha_c"], 0.3, places=12)
        self.assertEqual(on["alpha_c"], 0.3)
        # alpha_decision is the learner's own alpha: alpha_c T, or alpha_c.
        self.assertEqual(off["alpha_decision"], default.algorithm.alpha)
        self.assertEqual(on["alpha_decision"], 0.3)

    def test_default_path_is_the_code_before_the_setting(self):
        """reward_units "second", given or not, builds the same learner and converts the
        same numbers; its resume state and sidecar hold no reward_units."""
        absent = _baseline(alpha_c=0.3, learn_temperature=True)
        given = _baseline(alpha_c=0.3, learn_temperature=True, reward_units="second")
        self.assertEqual(absent.reward_units, "second")
        self.assertEqual(absent.period_temperature(0.3), 0.3 * REFERENCE_PERIOD)
        self.assertEqual(given.period_temperature(0.3), absent.period_temperature(0.3))
        self.assertEqual(given.algorithm.alpha, absent.algorithm.alpha)
        self.assertEqual(given.algorithm.log_alpha.item(), absent.algorithm.log_alpha.item())
        transition = _interval(-0.0123, terminated=False, next_state=[0.15, 0.3])
        u = 0.004 / REFERENCE_PERIOD
        self.assertEqual(absent.adapt_transition(transition)["reward"], (-0.0123 - 0.0) / u + 0.0)
        self.assertEqual(
            given.adapt_transition(transition)["reward"],
            absent.adapt_transition(transition)["reward"],
        )
        absent.learn(5)
        self.assertNotIn("reward_units", absent.resume_state())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.pt"
            absent.save(path)
            sidecar = json.loads(path.with_name(path.name + ".counters.json").read_text())
        self.assertEqual(set(sidecar), SIDECAR_KEYS)

    def test_checkpoints_record_the_units_and_refuse_others(self):
        scaled = _baseline(alpha_c=0.1, learn_temperature=True, reward_units="reference_period")
        default = _baseline(alpha_c=0.1, learn_temperature=True)
        scaled.learn(40)
        default.learn(40)
        state = scaled.resume_state()
        self.assertEqual(state["reward_units"], "reference_period")
        self.assertNotIn("reward_units", default.resume_state())
        # Resume states: refused across units before anything is restored.
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
            _baseline(alpha_c=0.1, learn_temperature=True).load_resume_state(state)
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
            _baseline(
                alpha_c=0.1, learn_temperature=True, reward_units="reference_period"
            ).load_resume_state({})
        same = _baseline(alpha_c=0.1, learn_temperature=True, reward_units="reference_period")
        same.load_resume_state(state)
        self.assertEqual(same.counters()["interactions"], 40)
        self.assertEqual(same.resume_state()["reward_units"], "reference_period")
        # The sidecar of save: load and warm_start refuse a file of other units.
        with tempfile.TemporaryDirectory() as folder:
            scaled_path = Path(folder) / "scaled.pt"
            default_path = Path(folder) / "default.pt"
            scaled.save(scaled_path)
            default.save(default_path)
            sidecar = json.loads(
                scaled_path.with_name(scaled_path.name + ".counters.json").read_text()
            )
            self.assertEqual(set(sidecar), SIDECAR_KEYS | {"reward_units"})
            self.assertEqual(sidecar["reward_units"], "reference_period")
            for path, units in ((scaled_path, "second"), (default_path, "reference_period")):
                with self.subTest(source=path.name):
                    fresh = _baseline(alpha_c=0.1, learn_temperature=True, reward_units=units)
                    with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                        fresh.load(path)
                    self.assertEqual(fresh.counters()["interactions"], 0)
                    with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                        fresh.warm_start(path)
            loaded = _baseline(alpha_c=0.1, learn_temperature=True, reward_units="reference_period")
            loaded.load(scaled_path)
            self.assertEqual(loaded.counters()["interactions"], 40)
            self.assertEqual(loaded.algorithm.log_alpha.item(), scaled.algorithm.log_alpha.item())
            warm = _baseline(alpha_c=0.1, learn_temperature=True, reward_units="reference_period")
            record = warm.warm_start(scaled_path)
        self.assertEqual(record["alpha_decision"], warm.algorithm.alpha)
        self.assertEqual(record["alpha_c"], warm.algorithm.alpha)

    def test_every_ct_sac_arm_receives_the_units(self):
        """make_training_agent passes reward_units to every CT-SAC arm under
        "reference_period" and nothing under "second" or without the key."""
        names = tuple(arm["name"] for arm in P.selectable_arms() if arm["algorithm"] == "ct_sac")
        for name in ("ct_sac_torque", "ct_sac_released", "ct_sac_released_features",
                     "ct_sac_implicit", "ct_sac_implicit_schedule_features",
                     "ct_sac_implicit_released_features"):
            self.assertIn(name, names)
        captured = {}
        for units in (None, "second", "reference_period"):
            for name, run in _smoke_runs(units, names).items():
                with mock.patch("algorithms.ph_baseline.TorqueCTSACBaseline") as plain, \
                        mock.patch("algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline") as implicit:
                    P.make_training_agent(run, None)
                built = implicit if "ct_sac_implicit_step" in run["arm"] else plain
                captured[units, name] = built.call_args
        for name in names:
            with self.subTest(arm=name):
                (env, training, temperature), kwargs = captured[None, name]
                self.assertIsNone(env)
                self.assertNotIn("reward_units", kwargs)
                self.assertEqual(captured["second", name], captured[None, name])
                args, scaled = captured["reference_period", name]
                self.assertEqual(args, (env, training, temperature))
                self.assertEqual(scaled, dict(kwargs, reward_units="reference_period"))

    def test_fingerprints(self):
        """The key enters only a CT-SAC arm's fingerprint, and only off "second"; a resume
        across units is refused."""
        absent = _smoke_runs()
        second = _smoke_runs("second")
        scaled = _smoke_runs("reference_period")
        for name in ("ct_sac_torque", "implicit_torque"):
            with self.subTest(arm=name):
                self.assertEqual(R.run_fingerprint(second[name]), R.run_fingerprint(absent[name]))
                for runs in (second, scaled):
                    self.assertNotIn("ct_sac_reward_units", R.run_settings(runs[name])["protocol"])
                self.assertNotIn("ct_sac_reward_units", R.run_settings(absent[name])["resolved"])
        # A CT-PH-SAC arm does not read it.
        self.assertEqual(
            R.run_fingerprint(scaled["implicit_torque"]), R.run_fingerprint(absent["implicit_torque"])
        )
        self.assertNotIn("ct_sac_reward_units", R.run_settings(scaled["implicit_torque"])["resolved"])
        # A CT-SAC arm holds it, resolved.
        settings = R.run_settings(scaled["ct_sac_torque"])
        self.assertEqual(settings["resolved"]["ct_sac_reward_units"], "reference_period")
        self.assertNotEqual(
            R.run_fingerprint(scaled["ct_sac_torque"]), R.run_fingerprint(absent["ct_sac_torque"])
        )
        # A resumable checkpoint (or completion marker) of one is refused for the other.
        for stored_runs, runs in ((absent, scaled), (scaled, absent), (second, scaled)):
            run = stored_runs["ct_sac_torque"]
            stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
            with self.assertRaisesRegex(R.ResumeMismatch, "ct_sac_reward_units"):
                R.check_fingerprint(stored, runs["ct_sac_torque"], Path("resume.pt"))
        run = scaled["ct_sac_torque"]
        stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
        R.check_fingerprint(stored, copy.deepcopy(run), Path("resume.pt"))
        run = absent["ct_sac_torque"]
        stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
        R.check_fingerprint(stored, second["ct_sac_torque"], Path("resume.pt"))

    def test_implicit_ct_sac_takes_the_units(self):
        """The implicit CT-SAC arm reads the stored rewards and the learner's temperature,
        so it runs on the divided scale; its bracket and residual are linear in the rewards,
        the soft values and the critics, so dividing all by T divides them by T."""
        env = make_env(_testbed(), decision_mode="torque", seed=11)
        temperature = {"initial_alpha_c": 1.0, "learn_temperature": True, "target_entropy": -3.0}
        baseline = ImplicitTorqueCTSACBaseline(
            env,
            _training(),
            temperature,
            implicit={
                "resolvent_step": 1.0,
                "inner_steps": 5,
                "policy_draws": 3,
                "actor_steps": 2,
                "temperature_every_rows": 2.0,
            },
            actor_schedule="per_row",
            polyak_rate=0.005,
            expectation_samples=1,
            net_arch=(32, 32),
            reward_units="reference_period",
        )
        self.assertEqual(baseline.reward_units, "reference_period")
        self.assertEqual(baseline.algorithm.env.reward_units, "reference_period")
        self.assertEqual(baseline.algorithm.alpha, 1.0)
        self.assertEqual(baseline.algorithm.operator_alpha, 1.0)
        # c = tau / T, a ratio of times: the same under either units.
        self.assertAlmostEqual(baseline.algorithm.proximal_weight, 1.0 / REFERENCE_PERIOD)
        generator = th.Generator().manual_seed(3)
        rows, draws, critics = 4, 3, 2
        soft_here = th.randn(rows, draws, generator=generator, dtype=th.float64)
        soft_next = th.randn(rows, draws, generator=generator, dtype=th.float64)
        rewards = th.randn(rows, 1, generator=generator, dtype=th.float64)
        dones = th.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=th.float64)
        u = th.tensor([[0.2], [1.0], [1.8], [0.4]], dtype=th.float64)
        gamma_u = th.exp(-DISCOUNT_RATE * REFERENCE_PERIOD * u)
        stored = th.randn(critics, rows, 1, generator=generator, dtype=th.float64)
        anchors = th.randn(critics, rows, 1, generator=generator, dtype=th.float64)
        c = baseline.algorithm.proximal_weight
        T = REFERENCE_PERIOD
        bracket = implicit_bracket(soft_here, soft_next, rewards, dones, u, gamma_u)
        divided = implicit_bracket(soft_here / T, soft_next / T, rewards / T, dones, u, gamma_u)
        # float64 rounding only (the divided values are O(1e3) and O(1e5)).
        th.testing.assert_close(divided, bracket / T, rtol=1e-12, atol=1e-9)
        residuals = resolvent_residuals(stored, anchors, bracket, c)
        th.testing.assert_close(
            resolvent_residuals(stored / T, anchors / T, divided, c),
            residuals / T,
            rtol=1e-10,
            atol=1e-6,
        )


# ------------------------------------------------------------------------ reward source

# The testbeds of the reward-source tests, True where the task's reward rate reads the
# torque: the windy pendulum of _testbed (an effort cost; a failure set), the lqr-lqr_2_1
# table row (a control cost; no failure set) and the pendulum-swingup table row (a rate of
# the state alone; no failure set).
SOURCE_TESTBEDS = {"windy_pendulum": True, "lqr-lqr_2_1": True, "pendulum-swingup": False}


def _row_testbed(env_id):
    """The TestbedConfig of a table row (top mode), as tests/test_ph_baseline_features.py
    row_testbed builds it."""
    entry = table.find_ph_entry(env_id, "top")
    return run_ph_sac._build_testbed(entry.config["testbed"], entry.config["testbed_settings"])


def _source_testbed(name):
    """A SOURCE_TESTBEDS entry's TestbedConfig."""
    return _testbed() if name == "windy_pendulum" else _row_testbed(name)


def _random_transitions(env, torque_limit, count, seed=0):
    """count transitions of the torque-mode env on its own clock, each under a uniform
    random torque in the symmetric box; a finished episode is reset."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        if env.done:
            env.reset()
        torque = rng.uniform(-1.0, 1.0, size=torque_limit.shape) * torque_limit
        rows.append(env.step_dt(torque, env.sample_duration()))
    return rows


def _charge(env, transition):
    """The failure charge of a transition as _adapt_transition computes it: 0 unless it
    terminated, else exp(-lambda Delta) times the terminal payoff."""
    if not transition["terminated"]:
        return 0.0
    duration = float(transition["duration"])
    payoff = float(env.terminal_payoff(transition["next_state"]))
    return math.exp(-env.discount_rate * duration) * payoff


def _record_reward_rate(env):
    """Record every call of env.reward_rate as (state, torque, value) without changing it."""
    calls = []
    rate = env.reward_rate

    def recorded(state, torque):
        value = rate(state, torque)
        calls.append(
            (np.array(state, dtype=np.float64), np.array(torque, dtype=np.float64), value)
        )
        return value

    env.reward_rate = recorded
    return calls


def _source_runs(keys, names=("ct_sac_torque", "implicit_torque")):
    """The smoke protocol's runs of the named arms, by name, with the protocol keys given
    (e.g. {"ct_sac_reward_source": "end_state"}; {} for neither)."""
    proto = run_ph_sac.load_protocol(str(SMOKE_CONFIG))
    proto.update(keys)
    proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in names]
    P.validate_matched_quantities(proto)
    return {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}


def _row_protocol(env_id, output, budget, keys):
    """A tiny protocol of a table row (top mode) running ct_sac_torque alone: 1 s episodes,
    one warm-up episode, batch 16, replay 2000, evaluations every 40 decisions on one reset
    seed, training seed 0, and the protocol keys given, through the JSON path."""
    raw = copy.deepcopy(table.find_ph_entry(env_id, "top").config)
    raw["training"].update(batch_size=16, replay_capacity=2000, warmup_episodes=1)
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[0],
        evaluation_reset_seeds=[3],
        evaluation_every_interactions=40,
        output_directory=str(output),
    )
    raw["protocol"].update(keys)
    proto = run_ph_sac.protocol_from_config(raw, f"tiny {env_id}")
    for testbed in (
        proto["matched"]["testbed_config"],
        proto["explicit_config"]["testbed"],
        proto["implicit_config"]["testbed"],
    ):
        testbed["episode_duration"] = 1.0
    proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] == "ct_sac_torque"]
    P.validate_matched_quantities(proto)
    return proto


class TestRewardSource(unittest.TestCase):
    """ct_sac_reward_source (common.ph_config.CT_SAC_REWARD_SOURCES): "end_state" replaces
    the interval part (r - c) / u of the CT-SAC learner's reward by T rho(s', a), the task's
    reward rate at the interval's end state with the held torque, times T; "interval", the
    default, is the code before the setting. The failure charge and the reward units apply
    to either."""

    def test_the_setting_and_its_validation(self):
        self.assertEqual(CT_SAC_REWARD_SOURCES, ("interval", "end_state"))
        self.assertEqual(DEFAULT_CT_SAC_REWARD_SOURCE, "interval")
        for source in CT_SAC_REWARD_SOURCES:
            self.assertEqual(check_ct_sac_reward_source(source), source)
            self.assertEqual(
                ct_sac_reward_source_setting({"ct_sac_reward_source": source}), source
            )
        # A protocol without the key reads the default, whatever its units.
        self.assertEqual(ct_sac_reward_source_setting({}), "interval")
        self.assertEqual(
            ct_sac_reward_source_setting({"ct_sac_reward_units": "reference_period"}), "interval"
        )
        for bad in ("end", "End_state", "", None, 1.0, "second"):
            with self.subTest(source=bad):
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
                    check_ct_sac_reward_source(bad)
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
                    ct_sac_reward_source_setting({"ct_sac_reward_source": bad})
                with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
                    _baseline(reward_source=bad)
        # The protocol: validated when present, absent by default.
        proto = run_ph_sac.load_protocol(str(SMOKE_CONFIG))
        self.assertNotIn("ct_sac_reward_source", proto)
        for source in CT_SAC_REWARD_SOURCES:
            proto["ct_sac_reward_source"] = source
            P.validate_matched_quantities(proto)
        proto["ct_sac_reward_source"] = "end"
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
            P.validate_matched_quantities(proto)

    def test_json_key_and_command_line(self):
        raw = json.loads(SMOKE_CONFIG.read_text())
        raw["protocol"]["ct_sac_reward_source"] = "end_state"
        proto = run_ph_sac.protocol_from_config(copy.deepcopy(raw), "smoke")
        self.assertEqual(proto["ct_sac_reward_source"], "end_state")
        self.assertNotIn("ct_sac_reward_units", proto)
        raw["protocol"]["ct_sac_reward_source"] = "end"
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
            run_ph_sac.protocol_from_config(copy.deepcopy(raw), "smoke")
        path = str(SMOKE_CONFIG)
        self.assertNotIn("ct_sac_reward_source", run_ph_sac.load_protocol(path))
        args = run_ph_sac.parse_args(["--config", path])
        self.assertIsNone(args.ct_sac_reward_source)
        self.assertNotIn(
            "ct_sac_reward_source",
            run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args),
        )
        args = run_ph_sac.parse_args(
            ["--config", path, "--ct-sac-reward-source", "end_state"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual(proto["ct_sac_reward_source"], "end_state")
        self.assertNotIn("ct_sac_reward_units", proto)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_ph_sac.parse_args(["--config", path, "--ct-sac-reward-source", "end"])

    def test_adapted_reward_is_the_rate_at_the_end_state(self):
        """Transitions of the environment itself (random torques on its own clock) on three
        testbeds and, on the windy pendulum, hand-built interior and failure rows: the
        default gives exactly (r - c) / u + c, the code before the setting; "end_state"
        exactly T rho(s', a) + c, rho a direct call of the environment's reward_rate at
        next_state and the held torque (the decision); "reference_period" divides each by
        T, so "end_state" there is rho(s', a) itself in a row without failure."""
        for name, reads_torque in SOURCE_TESTBEDS.items():
            testbed = _source_testbed(name)
            default = _baseline(testbed)
            end = _baseline(testbed, reward_source="end_state")
            default_scaled = _baseline(testbed, reward_units="reference_period")
            end_scaled = _baseline(
                testbed, reward_units="reference_period", reward_source="end_state"
            )
            self.assertEqual(default.reward_source, "interval")
            self.assertEqual(end.reward_source, "end_state")
            self.assertEqual(end.algorithm.env.reward_source, "end_state")
            env = end.env
            T = float(env.config["reference_period"])
            rows = _random_transitions(env, end.torque_limit, 40)
            if name == "windy_pendulum":
                charge = math.exp(-DISCOUNT_RATE * 0.004) * (-1.0 / DISCOUNT_RATE)
                rows += [
                    _interval(-0.0123, terminated=False, next_state=[0.15, 0.3]),
                    _interval(-0.0123 + charge, terminated=True, next_state=[1.6, 2.1]),
                ]
            for index, transition in enumerate(rows):
                with self.subTest(testbed=name, row=index):
                    charge = _charge(env, transition)
                    u = float(transition["duration"]) / T
                    rho = env.reward_rate(transition["next_state"], transition["decision"])
                    off = default.adapt_transition(transition)
                    on = end.adapt_transition(transition)
                    # The default source: the formula of the code before the setting.
                    self.assertEqual(
                        off["reward"], (float(transition["reward"]) - charge) / u + charge
                    )
                    self.assertEqual(
                        default_scaled.adapt_transition(transition)["reward"], off["reward"] / T
                    )
                    # end_state: T rho(s', a) + c, and under reference_period that over T.
                    self.assertEqual(on["reward"], T * rho + charge)
                    on_scaled = end_scaled.adapt_transition(transition)["reward"]
                    self.assertEqual(on_scaled, (T * rho + charge) / T)
                    if not transition["terminated"]:
                        self.assertAlmostEqual(on_scaled, rho, delta=1e-12 * max(1.0, abs(rho)))
                    # The rate reads the held torque: a task whose rate reads the torque
                    # tells it from no torque; one whose rate does not, does not.
                    unforced = env.reward_rate(
                        transition["next_state"], np.zeros_like(transition["decision"])
                    )
                    if not reads_torque:
                        self.assertEqual(rho, unforced)
                    elif np.any(transition["decision"] != 0.0):
                        self.assertNotEqual(rho, unforced)
                    # Every other field is the default's.
                    self.assertEqual(set(on), set(off))
                    for key in off:
                        if key != "reward":
                            self.assertIs(type(on[key]), type(off[key]), key)
                            np.testing.assert_array_equal(on[key], off[key], err_msg=key)

    def test_the_rate_is_the_environments_last_knot(self):
        """Through the path that fills the legacy replay (the adapter's _step_physics): the
        end-state rate reads exactly the state and torque of the environment's own last
        trapezoid knot, and that torque is the decision the transition records, the unit
        action times the motor limit."""
        for name in SOURCE_TESTBEDS:
            with self.subTest(testbed=name):
                baseline = _baseline(_source_testbed(name), reward_source="end_state")
                env = baseline.env
                adapter = baseline.algorithm.env
                T = float(env.config["reference_period"])
                calls = _record_reward_rate(env)
                adapter.reset(seed=3)
                rng = np.random.default_rng(1)
                for _ in range(15):
                    u = rng.uniform(-1.0, 1.0, size=baseline.torque_limit.shape)
                    calls.clear()
                    _, reward, terminated, truncated, info, _ = adapter._step_physics(
                        u, env.sample_duration()
                    )
                    transition = info["transition"]
                    # Two knots per physics step in step_dt, then the adapter's one call.
                    self.assertEqual(len(calls), 2 * env.last_interval["physics_steps"] + 1)
                    knot_state, knot_torque, knot_value = calls[-2]
                    state, torque, value = calls[-1]
                    np.testing.assert_array_equal(state, knot_state)
                    np.testing.assert_array_equal(torque, knot_torque)
                    self.assertEqual(value, knot_value)
                    np.testing.assert_array_equal(state, transition["next_state"])
                    np.testing.assert_array_equal(torque, transition["decision"])
                    np.testing.assert_allclose(transition["decision"], u * baseline.torque_limit)
                    self.assertEqual(reward, T * knot_value + _charge(env, transition))
                    self.assertEqual(info["adapted"]["reward"], reward)
                    if terminated or truncated:
                        break

    def test_one_physics_step_agrees_and_a_long_interval_differs(self):
        """lqr-lqr_2_1 (a smooth quadratic rate), reference_period units: over one physics
        step the interval reward, the discounted trapezoid of its two knots, is within O(dt)
        of the end-state rate; over 300 steps from a stretched spring, as the rate climbs,
        they differ."""
        testbed = _row_testbed("lqr-lqr_2_1")
        interval = _baseline(testbed, reward_units="reference_period")
        end = _baseline(testbed, reward_units="reference_period", reward_source="end_state")
        env = interval.env
        dt = env.physics_dt
        lam = env.discount_rate
        # One physics step from a moving state under a held 5 N.
        start = np.array([0.3, -0.2, 0.5, -0.4])
        torque = np.array([5.0])
        env.reset(seed=0, options={"state": start})
        short = env.step_dt(torque, dt)
        self.assertEqual(env.last_interval["physics_steps"], 1)
        rho_start = env.reward_rate(start, torque)
        rho_end = env.reward_rate(short["next_state"], torque)
        a = interval.adapt_transition(short)["reward"]
        b = end.adapt_transition(short)["reward"]
        self.assertAlmostEqual(a, 0.5 * (rho_start + math.exp(-lam * dt) * rho_end), delta=1e-10)
        self.assertAlmostEqual(b, rho_end, delta=1e-12)
        self.assertGreater(abs(b), 0.1)
        self.assertLess(abs(a - b), 1e-3)
        # 300 physics steps from rest at q = (1, 0) without torque: the springs pull the
        # chain in and the rate 1 - |q|^2 / 2 rises from 0.5.
        stretched = np.array([1.0, 0.0, 0.0, 0.0])
        env.reset(seed=0, options={"state": stretched})
        long = env.step_dt(np.zeros(1), 300 * dt)
        self.assertEqual(env.last_interval["physics_steps"], 300)
        rho_start = env.reward_rate(stretched, np.zeros(1))
        rho_end = env.reward_rate(long["next_state"], np.zeros(1))
        a = interval.adapt_transition(long)["reward"]
        b = end.adapt_transition(long)["reward"]
        self.assertAlmostEqual(a, long["reward"] / long["duration"], delta=1e-12)
        self.assertAlmostEqual(b, rho_end, delta=1e-12)
        self.assertGreater(abs(rho_end - rho_start), 0.05)
        self.assertGreater(abs(a - b), 0.05)

    def test_a_feedback_decision_is_refused(self):
        """A feedback-mode decision is an impedance law, not the held torque."""
        env = make_env(_testbed(), decision_mode="feedback", seed=0)
        transition = _interval(-0.0123, terminated=False, next_state=[0.15, 0.3])
        with self.assertRaisesRegex(ValueError, "torque-mode"):
            end_state_reward_rate(env, transition)
        with self.assertRaisesRegex(ValueError, "torque-mode"):
            _adapt_transition(env, transition, "second", "end_state")
        # The default source never reads the decision.
        u = 0.004 / float(env.config["reference_period"])
        self.assertEqual(_adapt_transition(env, transition)["reward"], (-0.0123 - 0.0) / u + 0.0)

    def test_reward_rate_of_a_batch(self):
        """reward_rate converts a batch as adapt_transition converts one interval: under
        "end_state" T rho(s', a) + c from next_states and decisions (the held torques), with
        the task's tensor rate; the default reads the interval rewards, not the decisions."""
        delta, r = 0.004, -0.0123
        charge = math.exp(-DISCOUNT_RATE * delta) * (-1.0 / DISCOUNT_RATE)
        next_states = np.array([[0.1, 0.0], [1.6, 0.0], [0.2, 0.1]])
        torques = np.array([[1.0], [12.0], [-3.0]])
        charges = np.array([[0.0], [charge], [0.0]])
        batch = {
            "durations": th.tensor([[delta], [delta], [0.018]], dtype=th.float64),
            "rewards": th.tensor([[r], [r + charge], [2.0 * r]], dtype=th.float64),
            "terminated": th.tensor([[0.0], [1.0], [0.0]], dtype=th.float64),
            "next_states": th.as_tensor(next_states),
            "decisions": th.as_tensor(torques),
        }
        without_decisions = {k: v for k, v in batch.items() if k != "decisions"}
        for units in CT_SAC_REWARD_UNITS:
            with self.subTest(units=units):
                default = _baseline(reward_units=units)
                end = _baseline(reward_units=units, reward_source="end_state")
                T = end.reference_period
                self.assertEqual(T, REFERENCE_PERIOD)
                rates = np.array(
                    [[end.env.reward_rate(s, a)] for s, a in zip(next_states, torques)]
                )
                expected = T * rates + charges
                if units == "reference_period":
                    expected = expected / T
                converted = end.reward_rate(batch)
                self.assertEqual(converted.shape, (3, 1))
                th.testing.assert_close(
                    converted, th.as_tensor(expected), rtol=1e-10, atol=1e-12
                )
                self.assertTrue(
                    th.equal(default.reward_rate(batch), default.reward_rate(without_decisions))
                )
                self.assertFalse(th.allclose(default.reward_rate(batch), converted))

    def test_the_legacy_replay_holds_end_state_rewards(self):
        """The path that fills the legacy replay: the same random decisions (full torque, so
        the pendulum fails) give the same rows under either source, each reward
        T rho(s', a) + c under "end_state", the failure charge as the default's."""
        stored = {}
        for source in CT_SAC_REWARD_SOURCES:
            baseline = _baseline(
                _testbed(), _training(warmup_episodes=10**6), reward_source=source
            )
            transitions = _spy_step_dt(baseline.env)
            baseline.algorithm.env.action_space.sample = lambda: np.array([1.0])
            baseline.learn(300)
            self.assertEqual(baseline.counters()["critic_gradient_steps"], 0)
            stored[source] = (baseline, transitions)
        (default, _), (end, transitions) = stored["interval"], stored["end_state"]
        off, on = default.algorithm.replay_buffer, end.algorithm.replay_buffer
        for name in ("observations", "next_observations", "actions", "dones", "dt"):
            np.testing.assert_array_equal(getattr(on, name)[:300], getattr(off, name)[:300], name)
        env = end.env
        T = float(env.config["reference_period"])
        expected = []
        for transition in transitions:
            self.assertEqual(transition["decision"].tolist(), [12.0])
            rho = env.reward_rate(transition["next_state"], transition["decision"])
            expected.append(T * rho + _charge(env, transition))
        self.assertEqual(len(expected), 300)
        np.testing.assert_allclose(on.rewards[:300, 0], np.array(expected), rtol=1e-6, atol=1e-9)
        failed = [i for i, t in enumerate(transitions) if t["terminated"]]
        self.assertGreater(len(failed), 0)
        for i in failed:
            # The undiluted charge, about -10, as under the default source.
            self.assertLess(on.rewards[i, 0], -9.9)
            self.assertGreater(on.rewards[i, 0], -11.0)
        # The rate changes within an interval, so the two sources' rewards differ.
        self.assertTrue(np.any(on.rewards[:300, 0] != off.rewards[:300, 0]))

    def test_default_path_is_the_code_before_the_setting(self):
        """reward_source "interval", given or not, converts the same numbers; its resume state
        and sidecar hold no reward_source."""
        absent = _baseline(alpha_c=0.3, learn_temperature=True)
        given = _baseline(alpha_c=0.3, learn_temperature=True, reward_source="interval")
        self.assertEqual(absent.reward_source, "interval")
        self.assertEqual(absent.algorithm.env.reward_source, "interval")
        self.assertEqual(given.algorithm.alpha, absent.algorithm.alpha)
        charge = math.exp(-DISCOUNT_RATE * 0.004) * (-1.0 / DISCOUNT_RATE)
        for transition in (
            _interval(-0.0123, terminated=False, next_state=[0.15, 0.3]),
            _interval(-0.0123 + charge, terminated=True, next_state=[1.6, 2.1]),
        ):
            expected = absent.adapt_transition(transition)["reward"]
            self.assertEqual(given.adapt_transition(transition)["reward"], expected)
            # The module function without the argument, and with the defaults named.
            self.assertEqual(_adapt_transition(absent.env, transition)["reward"], expected)
            self.assertEqual(
                _adapt_transition(absent.env, transition, "second", "interval")["reward"],
                expected,
            )
        absent.learn(5)
        self.assertNotIn("reward_source", absent.resume_state())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.pt"
            absent.save(path)
            sidecar = json.loads(path.with_name(path.name + ".counters.json").read_text())
        self.assertEqual(set(sidecar), SIDECAR_KEYS)

    def test_checkpoints_record_the_source_and_refuse_the_other(self):
        end = _baseline(alpha_c=0.1, learn_temperature=True, reward_source="end_state")
        default = _baseline(alpha_c=0.1, learn_temperature=True)
        end.learn(40)
        default.learn(40)
        state = end.resume_state()
        self.assertEqual(state["reward_source"], "end_state")
        self.assertNotIn("reward_units", state)
        self.assertNotIn("reward_source", default.resume_state())
        # Resume states: refused across sources before anything is restored.
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
            _baseline(alpha_c=0.1, learn_temperature=True).load_resume_state(state)
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
            _baseline(
                alpha_c=0.1, learn_temperature=True, reward_source="end_state"
            ).load_resume_state({})
        same = _baseline(alpha_c=0.1, learn_temperature=True, reward_source="end_state")
        same.load_resume_state(state)
        self.assertEqual(same.counters()["interactions"], 40)
        self.assertEqual(same.resume_state()["reward_source"], "end_state")
        # The sidecar of save: load and warm_start refuse a file of the other source.
        with tempfile.TemporaryDirectory() as folder:
            end_path = Path(folder) / "end.pt"
            default_path = Path(folder) / "default.pt"
            end.save(end_path)
            default.save(default_path)
            sidecar = json.loads(end_path.with_name(end_path.name + ".counters.json").read_text())
            self.assertEqual(set(sidecar), SIDECAR_KEYS | {"reward_source"})
            self.assertEqual(sidecar["reward_source"], "end_state")
            for path, source in ((end_path, "interval"), (default_path, "end_state")):
                with self.subTest(source=path.name):
                    fresh = _baseline(alpha_c=0.1, learn_temperature=True, reward_source=source)
                    with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
                        fresh.load(path)
                    self.assertEqual(fresh.counters()["interactions"], 0)
                    with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
                        fresh.warm_start(path)
            # The units are checked as before, independently of the source.
            with self.assertRaisesRegex(ValueError, "ct_sac_reward_units"):
                _baseline(
                    alpha_c=0.1,
                    learn_temperature=True,
                    reward_units="reference_period",
                    reward_source="end_state",
                ).load(end_path)
            loaded = _baseline(alpha_c=0.1, learn_temperature=True, reward_source="end_state")
            loaded.load(end_path)
            self.assertEqual(loaded.counters()["interactions"], 40)
            self.assertEqual(loaded.algorithm.log_alpha.item(), end.algorithm.log_alpha.item())
            warm = _baseline(alpha_c=0.1, learn_temperature=True, reward_source="end_state")
            record = warm.warm_start(end_path)
        self.assertEqual(record["alpha_decision"], warm.algorithm.alpha)

    def test_every_ct_sac_arm_receives_the_source(self):
        """make_training_agent passes reward_source to every CT-SAC arm under "end_state"
        and nothing under "interval" or without the key."""
        names = tuple(arm["name"] for arm in P.selectable_arms() if arm["algorithm"] == "ct_sac")
        self.assertIn("ct_sac_torque", names)
        self.assertIn("ct_sac_implicit", names)
        captured = {}
        for source in (None, "interval", "end_state"):
            keys = {} if source is None else {"ct_sac_reward_source": source}
            for name, run in _source_runs(keys, names).items():
                with mock.patch("algorithms.ph_baseline.TorqueCTSACBaseline") as plain, \
                        mock.patch("algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline") as implicit:
                    P.make_training_agent(run, None)
                built = implicit if "ct_sac_implicit_step" in run["arm"] else plain
                captured[source, name] = built.call_args
        for name in names:
            with self.subTest(arm=name):
                (env, training, temperature), kwargs = captured[None, name]
                self.assertIsNone(env)
                self.assertNotIn("reward_source", kwargs)
                self.assertEqual(captured["interval", name], captured[None, name])
                args, end = captured["end_state", name]
                self.assertEqual(args, (env, training, temperature))
                self.assertEqual(end, dict(kwargs, reward_source="end_state"))

    def test_implicit_ct_sac_takes_the_source(self):
        env = make_env(_testbed(), decision_mode="torque", seed=11)
        temperature = {"initial_alpha_c": 1.0, "learn_temperature": True, "target_entropy": -3.0}
        baseline = ImplicitTorqueCTSACBaseline(
            env,
            _training(),
            temperature,
            implicit={
                "resolvent_step": 1.0,
                "inner_steps": 5,
                "policy_draws": 3,
                "actor_steps": 2,
                "temperature_every_rows": 2.0,
            },
            actor_schedule="per_row",
            polyak_rate=0.005,
            expectation_samples=1,
            net_arch=(32, 32),
            reward_source="end_state",
        )
        self.assertEqual(baseline.reward_source, "end_state")
        self.assertEqual(baseline.algorithm.env.reward_source, "end_state")
        transition = _interval(-0.0123, terminated=False, next_state=[0.15, 0.3])
        T = float(env.config["reference_period"])
        self.assertEqual(
            baseline.adapt_transition(transition)["reward"],
            T * env.reward_rate(transition["next_state"], transition["decision"]),
        )

    def test_fingerprints(self):
        """The key enters only a CT-SAC arm's fingerprint, and only off "interval"; a resume
        across sources is refused."""
        absent = _source_runs({})
        interval = _source_runs({"ct_sac_reward_source": "interval"})
        end = _source_runs({"ct_sac_reward_source": "end_state"})
        for name in ("ct_sac_torque", "implicit_torque"):
            with self.subTest(arm=name):
                self.assertEqual(R.run_settings(interval[name]), R.run_settings(absent[name]))
                self.assertEqual(R.run_fingerprint(interval[name]), R.run_fingerprint(absent[name]))
                for runs in (interval, end):
                    self.assertNotIn("ct_sac_reward_source", R.run_settings(runs[name])["protocol"])
                # Without the key (or at its default) the settings never mention it.
                self.assertNotIn("ct_sac_reward_source", json.dumps(R.run_settings(absent[name])))
        # A CT-PH-SAC arm does not read it.
        self.assertEqual(
            R.run_fingerprint(end["implicit_torque"]), R.run_fingerprint(absent["implicit_torque"])
        )
        self.assertNotIn("ct_sac_reward_source", R.run_settings(end["implicit_torque"])["resolved"])
        # A CT-SAC arm holds it, resolved.
        settings = R.run_settings(end["ct_sac_torque"])
        self.assertEqual(settings["resolved"]["ct_sac_reward_source"], "end_state")
        self.assertNotEqual(
            R.run_fingerprint(end["ct_sac_torque"]), R.run_fingerprint(absent["ct_sac_torque"])
        )
        # Independent of the reward units: each key changes the CT-SAC arm's fingerprint.
        units = _source_runs({"ct_sac_reward_units": "reference_period"})
        both = _source_runs(
            {"ct_sac_reward_units": "reference_period", "ct_sac_reward_source": "end_state"}
        )
        resolved = R.run_settings(both["ct_sac_torque"])["resolved"]
        self.assertEqual(resolved["ct_sac_reward_units"], "reference_period")
        self.assertEqual(resolved["ct_sac_reward_source"], "end_state")
        self.assertNotIn("ct_sac_reward_source", R.run_settings(units["ct_sac_torque"])["resolved"])
        prints = {R.run_fingerprint(runs["ct_sac_torque"]) for runs in (absent, end, units, both)}
        self.assertEqual(len(prints), 4)
        self.assertEqual(
            R.run_fingerprint(both["implicit_torque"]), R.run_fingerprint(absent["implicit_torque"])
        )
        # A resumable checkpoint (or completion marker) of one is refused for the other.
        for stored_runs, runs in ((absent, end), (end, absent), (interval, end)):
            run = stored_runs["ct_sac_torque"]
            stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
            with self.assertRaisesRegex(R.ResumeMismatch, "ct_sac_reward_source"):
                R.check_fingerprint(stored, runs["ct_sac_torque"], Path("resume.pt"))
        run = end["ct_sac_torque"]
        stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
        R.check_fingerprint(stored, copy.deepcopy(run), Path("resume.pt"))
        run = absent["ct_sac_torque"]
        stored = {"fingerprint": R.run_fingerprint(run), "settings": R.run_settings(run)}
        R.check_fingerprint(stored, interval["ct_sac_torque"], Path("resume.pt"))

    def test_a_tiny_run_records_the_source_and_stores_its_rewards(self):
        """ct_sac_torque on the pendulum-swingup row (no failure set) through run_seed: the
        legacy replay holds the adapter's reward of every interval (under end_state with
        reference_period units, rho(s', a) itself), and the run record's protocol and the
        checkpoint's counters sidecar hold the source only when it is set."""
        budget = 60
        for keys in (
            {},
            {"ct_sac_reward_source": "end_state", "ct_sac_reward_units": "reference_period"},
        ):
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as tmp:
                (run,) = P.build_run_matrix(_row_protocol("pendulum-swingup", tmp, budget, keys))
                agents, spied = [], []
                build = P.make_training_agent

                def capture(spec, env):
                    spied.append(_spy_step_dt(env))
                    agents.append(build(spec, env))
                    return agents[-1]

                with mock.patch.object(P, "make_training_agent", capture):
                    result = P.run_seed(run)
                (agent,), (rows,) = agents, spied
                self.assertEqual(result["counters"]["interactions"], budget)
                self.assertEqual(len(rows), budget)
                env = agent.env
                T = float(env.config["reference_period"])
                expected = []
                for transition in rows:
                    self.assertFalse(transition["terminated"])
                    if keys:
                        expected.append(
                            env.reward_rate(transition["next_state"], transition["decision"])
                        )
                    else:
                        expected.append(
                            float(transition["reward"]) / (float(transition["duration"]) / T)
                        )
                np.testing.assert_allclose(
                    agent.algorithm.replay_buffer.rewards[:budget, 0],
                    np.array(expected),
                    rtol=1e-6,
                    atol=1e-9,
                )
                folder = P.run_output_directory(run)
                (loaded,) = load_run_records([str(folder / RECORD_NAME)])
                sidecar = json.loads((folder / "checkpoint.pt.counters.json").read_text())
                if keys:
                    self.assertEqual(agent.reward_source, "end_state")
                    self.assertEqual(loaded["run"]["protocol"]["ct_sac_reward_source"], "end_state")
                    self.assertEqual(sidecar["reward_source"], "end_state")
                    self.assertEqual(sidecar["reward_units"], "reference_period")
                else:
                    self.assertEqual(agent.reward_source, "interval")
                    self.assertNotIn("ct_sac_reward_source", loaded["run"]["protocol"])
                    self.assertNotIn("reward_source", sidecar)


class TestRewardSourceRestart(ExactRestartMixin, unittest.TestCase):
    def test_ct_sac_torque_under_end_state(self):
        """ct_sac_torque under "end_state" on the pendulum-swingup row, stopped at 85
        (training from decision 50, after the one warm-up episode) and resumed in a
        disturbed fresh agent, ends bitwise where the uninterrupted run ends; its resume
        state and fingerprint hold the source, and the same run without the key refuses
        the checkpoint."""
        options = R.resume_options(every_interactions=0)
        keys = {"ct_sac_reward_source": "end_state"}
        budget, stop = 120, 85
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            (reference_run,) = P.build_run_matrix(
                _row_protocol("pendulum-swingup", first, budget, keys)
            )
            reference, ref_agent = run_captured(reference_run, options)
            (run,) = P.build_run_matrix(_row_protocol("pendulum-swingup", second, budget, keys))
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
            self.assertIsInstance(stopped, R.ResumableStop)
            folder = P.run_output_directory(run)
            payload = th.load(folder / R.RESUME_NAME, weights_only=False)
            (plain,) = P.build_run_matrix(_row_protocol("pendulum-swingup", second, budget, {}))
            with self.assertRaisesRegex(R.ResumeMismatch, "ct_sac_reward_source"):
                run_captured(plain, options)
            resumed, res_agent = run_captured(run, options, disturbed=True)
            self.assertFalse((folder / R.RESUME_NAME).exists())
            self.assertTrue((folder / R.COMPLETE_NAME).exists())
        self.assertGreaterEqual(payload["interactions"], stop)
        self.assertEqual(payload["agent"]["reward_source"], "end_state")
        self.assertEqual(payload["settings"]["resolved"]["ct_sac_reward_source"], "end_state")
        self.assertNotIn("ct_sac_reward_source", payload["settings"]["protocol"])
        self.assertEqual(reference["counters"]["interactions"], budget)
        self.assertGreater(reference["counters"]["critic_gradient_steps"], 0)
        self.assertEqual(ref_agent.reward_source, "end_state")
        self.assert_same_run(reference, ref_agent, resumed, res_agent)


if __name__ == "__main__":
    unittest.main()
