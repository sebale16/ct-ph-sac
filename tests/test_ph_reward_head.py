"""The learned reward head's output: a sigmoid onto a declared reward range, else linear.

On a testbed whose TestbedSpec declares reward_rate_bounds (every dm_control task but
acrobot and lqr: tolerance rewards in [0, 1]) the implicit agent's RewardRate maps a
sigmoid onto that range; elsewhere its output is linear. A negative learned reward on
the reacher made the payoff-0 wrist stop the value maximum.
"""

import copy
import math
import unittest

import torch

from environment.ph_sac.registry import testbed_spec
from models.ph_sac.reward import RewardRate
from tests.test_ph_algorithms import (
    collect,
    implicit_config,
    make_implicit,
    synthetic_reward_batch,
    windy,
)
from tests.test_ph_env import cartpole_config

NETWORK = {"hidden_sizes": (16, 16), "velocity_feature_scale": 2.0 * math.pi}
CARTPOLE_SPEC = cartpole_config()["state_spec"]
UNIT_INTERVAL = (
    "pendulum_swingup",
    "cartpole_swingup",
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup_sparse",
    "cartpole_two_poles",
    "cartpole_three_poles",
    "reacher_easy",
    "reacher_hard",
    "point_mass_easy",
)
LINEAR = ("lqr_2_1", "lqr_6_2", "windy_pendulum", "acrobot_xk")


def heads(bounds, *, seed=0):
    """A linear and a bounded head with the same initial parameters."""
    torch.manual_seed(seed)
    linear = RewardRate(CARTPOLE_SPEC, NETWORK, (10.0,))
    torch.manual_seed(seed)
    bounded = RewardRate(CARTPOLE_SPEC, NETWORK, (10.0,), bounds=bounds)
    return linear, bounded


def inputs(count=512, *, seed=1, scale=3.0):
    generator = torch.Generator().manual_seed(seed)
    states = scale * torch.randn(count, 4, generator=generator)
    torques = 10.0 * torch.randn(count, 1, generator=generator)
    return states, torques


class TestRewardRateOutput(unittest.TestCase):
    def test_default_is_linear(self):
        linear, _ = heads((0.0, 1.0))
        self.assertIsNone(linear.bounds)
        self.assertFalse(linear.bounded)
        self.assertEqual(linear.output, "linear")

    def test_bounded_outputs_lie_strictly_inside_the_bounds(self):
        for bounds in ((0.0, 1.0), (-2.0, 0.5)):
            _, bounded = heads(bounds)
            self.assertTrue(bounded.bounded)
            self.assertEqual(bounded.bounds, bounds)
            states, torques = inputs()
            with torch.no_grad():
                values = bounded(states, torques)
            self.assertEqual(values.shape, (512, 1))
            self.assertTrue(bool((values > bounds[0]).all() and (values < bounds[1]).all()))

    def test_bounded_is_the_sigmoid_of_the_linear_output(self):
        linear, bounded = heads((-2.0, 0.5))
        states, torques = inputs()
        with torch.no_grad():
            expected = -2.0 + 2.5 * torch.sigmoid(linear(states, torques))
            torch.testing.assert_close(bounded(states, torques), expected)

    def test_linear_output_is_unconstrained_in_sign(self):
        linear, bounded = heads((0.0, 1.0))
        last = [m for m in linear.network.modules() if isinstance(m, torch.nn.Linear)][-1]
        states, torques = inputs()
        with torch.no_grad():
            for bias in (-5.0, 5.0):
                last.bias.fill_(bias)
                values = linear(states, torques)
                self.assertTrue(bool((values < 0.0).all() if bias < 0 else (values > 1.0).all()))
                # The same parameters through the sigmoid stay inside (0, 1).
                bounded.load_state_dict(linear.state_dict())
                values = bounded(states, torques)
                self.assertTrue(bool((values > 0.0).all() and (values < 1.0).all()))

    def test_state_dicts_match_so_checkpoints_load_in_either_mode(self):
        linear, bounded = heads((0.0, 1.0))
        self.assertEqual(
            {k: v.shape for k, v in linear.state_dict().items()},
            {k: v.shape for k, v in bounded.state_dict().items()},
        )
        bounded.load_state_dict(linear.state_dict())
        self.assertTrue(bounded.bounded)

    def test_saturated_outputs_stay_in_the_closed_range_with_finite_gradients(self):
        _, bounded = heads((0.0, 1.0))
        with torch.no_grad():
            for p in bounded.parameters():
                p.mul_(50.0)
        states, torques = inputs()
        values = bounded(states, torques)
        self.assertTrue(bool((values >= 0.0).all() and (values <= 1.0).all()))
        values.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in bounded.parameters()))

    def test_bad_bounds_raise(self):
        for bounds in ((1.0, 0.0), (0.0, 0.0), (0.0, math.inf), (math.nan, 1.0)):
            with self.assertRaises(ValueError, msg=str(bounds)):
                RewardRate(CARTPOLE_SPEC, NETWORK, (10.0,), bounds=bounds)

    def test_deepcopy_keeps_the_bounds(self):
        _, bounded = heads((0.0, 1.0))
        self.assertEqual(copy.deepcopy(bounded).bounds, (0.0, 1.0))


class TestDeclaredRanges(unittest.TestCase):
    def test_tolerance_rewards_declare_the_unit_interval(self):
        for name in UNIT_INTERVAL:
            with self.subTest(name=name):
                declared = testbed_spec(name).reward_rate_bounds
                self.assertIsNotNone(declared)
                self.assertEqual(declared({"name": name}), (0.0, 1.0))

    def test_unbounded_and_in_house_rewards_declare_none(self):
        for name in LINEAR:
            with self.subTest(name=name):
                self.assertIsNone(testbed_spec(name).reward_rate_bounds)


class TestImplicitAgentHead(unittest.TestCase):
    def test_bounded_testbed_gets_a_sigmoid_head_and_frozen_copy(self):
        agent = make_implicit(implicit_config(cartpole_config()))
        self.assertEqual(agent.reward_net.bounds, (0.0, 1.0))
        collect(agent, 20)
        agent.freeze_operator()
        self.assertIsNot(agent._reward_frozen, agent.reward_net)
        self.assertEqual(agent._reward_frozen.bounds, (0.0, 1.0))
        generator = torch.Generator().manual_seed(2)
        states = 3.0 * torch.randn(256, agent.env.state_dim, generator=generator)
        torques = 20.0 * torch.randn(256, 1, generator=generator)
        with torch.no_grad():
            values = agent._reward_rate(states, torques)
        self.assertTrue(bool((values > 0.0).all() and (values < 1.0).all()))
        # A checkpoint round trip restores a bounded frozen copy.
        restored = make_implicit(implicit_config(cartpole_config()))
        restored.load_state_dict(agent.state_dict())
        self.assertEqual(restored._reward_frozen.bounds, (0.0, 1.0))
        with torch.no_grad():
            torch.testing.assert_close(restored._reward_rate(states, torques), values)

    def test_windy_keeps_a_linear_head(self):
        agent = make_implicit(implicit_config(windy()))
        self.assertIsNone(agent.reward_net.bounds)
        agent.freeze_operator()
        self.assertIsNone(agent._reward_frozen.bounds)

    def test_diagnostics_record_the_output_mode(self):
        for testbed, output in ((cartpole_config(), "sigmoid(0, 1)"), (windy(), "linear")):
            with self.subTest(output=output):
                agent = make_implicit(implicit_config(testbed))
                agent.reward_diagnostics_every = 2
                batch = synthetic_reward_batch(agent, rows=16)
                for _ in range(2):
                    agent.update_reward(batch)
                self.assertEqual(agent.reward_records[-1]["reward_output"], output)


if __name__ == "__main__":
    unittest.main()
