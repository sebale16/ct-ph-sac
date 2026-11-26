"""A run's discount rate in place of the row's (testbed_settings discount_rate).

common.ph_dmc._common.with_discount_rate, applied by benchmarks/run_ph_sac._build_testbed
on the testbeds whose TestbedSpec sets discount_setting. Without the setting every config
keeps the row's rate (0.5 /s); with it the config's discount_rate, the one rate every
consumer reads, is the setting's, after the clock settings and lqr's torque_limit. Checks
that it reaches the environment's interval reward, the CT-SAC baseline's period discount
and its per-interval target factor, both CT-PH-SAC agents' rate and the protocol's matched
rate (which the run record carries), and that the temperature conversion, alpha_c T,
reads the reference period instead. Building only, no training; a few seconds.
"""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_dmc._common import DMC_DISCOUNT_RATE, with_discount_rate
from common.ph_time import period_discount
from environment.ph_sac import tasks
from environment.ph_sac.registry import registered_testbeds, testbed_spec
from evaluations.ph_sac import protocol as P

# The tasks that take the setting, by table env_id, with each one's reference period (s).
REFERENCE_PERIODS = {
    "acrobot-swingup": 0.01,
    "acrobot-swingup_sparse": 0.01,
    "cartpole-balance": 0.01,
    "cartpole-balance_sparse": 0.01,
    "cartpole-swingup": 0.01,
    "cartpole-swingup_sparse": 0.01,
    "cartpole-two_poles": 0.01,
    "cartpole-three_poles": 0.01,
    "pendulum-swingup": 0.02,
    "point_mass-easy": 0.02,
    "lqr-lqr_2_1": 0.03,
    "lqr-lqr_6_2": 0.03,
}
RATE = 2.0
IRREGULAR = {"clock_kind": "two_tail_uniform", "tail_probability": 0.99, "tail_split": 0.9}


def row(env_id, mode="top"):
    return copy.deepcopy(table.find_ph_entry(env_id, mode).config)


def build(env_id, mode="top", **settings):
    raw = row(env_id, mode)
    raw["testbed_settings"].update(settings)
    return run_ph_sac._build_testbed(raw["testbed"], raw["testbed_settings"])


def without_rate(config):
    return {key: value for key, value in config.items() if key != "discount_rate"}


def tiny_protocol(env_id, arm_name, rate=None):
    """The top row's protocol with small networks, one seed and arm, the rate when given."""
    raw = row(env_id)
    if rate is not None:
        raw["testbed_settings"]["discount_rate"] = rate
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    raw["training"].update(batch_size=16, replay_capacity=2000, warmup_episodes=1)
    raw["protocol"].update(training_seeds=[0], evaluation_reset_seeds=[3])
    proto = run_ph_sac.protocol_from_config(raw, f"tiny discount {env_id}")
    proto["arms"] = [arm for arm in P.all_arms() if arm["name"] == arm_name]
    P.validate_matched_quantities(proto)
    (run,) = P.build_run_matrix(proto)
    return run


def build_agent(run):
    config = run["protocol"]["matched"]["testbed_config"]
    env = tasks.make_env(config, decision_mode=run["arm"]["decision_mode"], seed=0)
    return P.make_training_agent(run, env), env


class TestSetting(unittest.TestCase):
    def test_the_twelve_tasks_take_it(self):
        names = {table.find_ph_entry(env_id, "top").config["testbed"] for env_id in REFERENCE_PERIODS}
        self.assertEqual(len(names), 12)
        for name in registered_testbeds():
            with self.subTest(testbed=name):
                self.assertEqual(testbed_spec(name).discount_setting, name in names)

    def test_default_unchanged_and_override_replaces_only_the_rate(self):
        for env_id, period in REFERENCE_PERIODS.items():
            with self.subTest(env_id=env_id):
                nominal = build(env_id)
                self.assertEqual(nominal["discount_rate"], DMC_DISCOUNT_RATE)
                self.assertEqual(nominal["reference_period"], period)
                name = row(env_id)["testbed"]
                factory = testbed_spec(name).config_factory(**row(env_id)["testbed_settings"])
                self.assertTrue(P._same_settings(nominal, factory))
                changed = build(env_id, discount_rate=RATE)
                self.assertEqual(changed["discount_rate"], RATE)
                self.assertTrue(P._same_settings(without_rate(changed), without_rate(nominal)))

    def test_combines_with_the_clock_and_the_torque_limit(self):
        for env_id in REFERENCE_PERIODS:
            if env_id.startswith("lqr"):
                continue
            with self.subTest(env_id=env_id):
                irregular = build(env_id, "irregular")
                changed = build(env_id, "irregular", discount_rate=RATE)
                self.assertEqual(changed["clock"]["kind"], "two_tail_uniform")
                self.assertEqual(changed["discount_rate"], RATE)
                self.assertTrue(P._same_settings(without_rate(changed), without_rate(irregular)))
        for env_id in ("lqr-lqr_2_1", "lqr-lqr_6_2"):
            with self.subTest(env_id=env_id):
                changed = build(env_id, torque_limit=2.0, discount_rate=RATE, **IRREGULAR)
                self.assertEqual(changed["clock"]["kind"], "two_tail_uniform")
                self.assertEqual(float(changed["controller"]["torque_high"][0]), 2.0)
                self.assertEqual(changed["discount_rate"], RATE)

    def test_invalid_or_unsupported_raise(self):
        for rate in (0.0, -1.0, float("nan"), float("inf"), True, None, "2"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                build("cartpole-swingup", discount_rate=rate)
        with self.assertRaises(ValueError):
            with_discount_rate(build("cartpole-swingup"), 0.0)
        for env_id in ("reacher-easy", "reacher-hard"):
            with self.subTest(env_id=env_id), self.assertRaisesRegex(ValueError, "unknown settings"):
                build(env_id, discount_rate=RATE)

    def test_the_sparse_swing_ups_take_it_as_their_siblings(self):
        # acrobot-swingup_sparse and cartpole-swingup_sparse (since 2026-11-20): without the
        # key the factory's config; with it only the rate differs, as on the smooth task.
        for sparse, smooth in (
            ("acrobot-swingup_sparse", "acrobot-swingup"),
            ("cartpole-swingup_sparse", "cartpole-swingup"),
        ):
            with self.subTest(env_id=sparse):
                name = row(sparse)["testbed"]
                self.assertTrue(testbed_spec(name).discount_setting)
                self.assertEqual(
                    testbed_spec(name).discount_setting,
                    testbed_spec(row(smooth)["testbed"]).discount_setting,
                )
                nominal = build(sparse)
                factory = testbed_spec(name).config_factory(**row(sparse)["testbed_settings"])
                self.assertTrue(P._same_settings(nominal, factory))
                self.assertEqual(nominal["discount_rate"], DMC_DISCOUNT_RATE)
                changed = build(sparse, discount_rate=RATE)
                self.assertEqual(changed["discount_rate"], RATE)
                self.assertTrue(P._same_settings(without_rate(changed), without_rate(nominal)))
                for rate in (0.0, -1.0, float("nan"), True, "2"):
                    with self.assertRaises(ValueError):
                        build(sparse, discount_rate=rate)
                # Other unknown settings are still refused.
                with self.assertRaisesRegex(ValueError, "unknown settings"):
                    build(sparse, discount_rates=RATE)


class TestConsumers(unittest.TestCase):
    def test_protocol_and_record_carry_the_rate(self):
        run = tiny_protocol("acrobot-swingup", "implicit_learned_admissions", RATE)
        protocol = run["protocol"]
        self.assertEqual(protocol["matched"]["discount_rate"], RATE)
        for name in ("explicit_config", "implicit_config"):
            self.assertEqual(protocol[name]["testbed"]["discount_rate"], RATE)
        default = tiny_protocol("acrobot-swingup", "implicit_learned_admissions")
        self.assertEqual(default["protocol"]["matched"]["discount_rate"], DMC_DISCOUNT_RATE)

    def test_environment_discounts_the_interval_reward_at_the_rate(self):
        nominal_env = tasks.make_env(build("cartpole-swingup"), seed=0)
        env = tasks.make_env(build("cartpole-swingup", discount_rate=RATE), seed=0)
        self.assertEqual(env.discount_rate, RATE)
        start = np.array([0.1, 0.3, 0.0, 0.0])
        decision = np.array([0.0, 50.0, 10.0])
        rewards = []
        for each in (nominal_env, env):
            each.reset(seed=1, options={"state": start})
            transition = each.step_dt(decision, 0.03)
            rewards.append((transition["reward"], each.last_undiscounted_reward))
        (nominal_reward, nominal_plain), (reward, plain) = rewards
        self.assertEqual(plain, nominal_plain)
        # A positive rate discounted harder over the same 30 ms.
        self.assertGreater(plain, 0.0)
        self.assertLess(reward, nominal_reward)
        self.assertLess(reward, plain)

    def test_baseline_target_uses_the_rate_and_temperature_the_period(self):
        run = tiny_protocol("acrobot-swingup", "ct_sac_torque", RATE)
        agent, env = build_agent(run)
        algorithm = agent.algorithm
        period = float(env.config["reference_period"])
        self.assertEqual(agent.env.discount_rate, RATE)
        self.assertEqual(algorithm.gamma, period_discount(discount_rate=RATE, reference_period=period))
        # The per-interval factor of CT-SAC's target, gamma^(Delta/T), is exp(-rate Delta).
        for duration in (0.002, 0.01, 0.03):
            factor = math.exp(-algorithm.beta * duration * algorithm.time_rescale)
            self.assertAlmostEqual(factor, math.exp(-RATE * duration), places=12)
        # The temperature conversion reads the reference period, not the rate: alpha_c T.
        self.assertAlmostEqual(agent.period_temperature(1.0), period, places=15)

    def test_sparse_swing_up_protocols_and_agents_use_the_rate(self):
        for env_id in ("acrobot-swingup_sparse", "cartpole-swingup_sparse"):
            with self.subTest(env_id=env_id):
                run = tiny_protocol(env_id, "implicit_torque", RATE)
                self.assertEqual(run["protocol"]["matched"]["discount_rate"], RATE)
                agent, env = build_agent(run)
                self.assertEqual(env.discount_rate, RATE)
                self.assertEqual(agent.discount_rate, RATE)
                env.close()
                default = tiny_protocol(env_id, "implicit_torque")
                self.assertEqual(default["protocol"]["matched"]["discount_rate"], DMC_DISCOUNT_RATE)

    def test_ph_agent_uses_the_rate(self):
        run = tiny_protocol("acrobot-swingup", "implicit_learned_admissions", RATE)
        agent, env = build_agent(run)
        self.assertEqual(env.discount_rate, RATE)
        self.assertEqual(agent.discount_rate, RATE)
        tau = float(run["protocol"]["implicit_config"]["solver"]["resolvent_step"])
        self.assertEqual(agent.contraction, 1.0 / (1.0 + RATE * tau))
        self.assertAlmostEqual(
            agent.policy_diagnostics()["alpha_decision"],
            float(agent.alpha_c) * float(env.config["reference_period"]),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
