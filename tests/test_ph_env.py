import copy
import math
import unittest

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import acrobot_xk_config, cartpole_swingup_config, windy_pendulum_config
from common.ph_time import clock_duration_pmf
from environment.ph_sac.clock import DiscreteIntervalClock, duration_in_physics_steps
from environment.ph_sac.tasks import (
    AcrobotXKEnv,
    WindyPendulumEnv,
    failure_mask_tensor,
    make_env,
    make_oracle_plant,
    oracle_callbacks,
    reward_rate_tensor,
    state_spec,
    terminal_boundary_functions,
    terminal_payoff_tensor,
)
from models.ph_sac.rollouts import stochastic_rollout


def windy_config(**clock):
    options = dict(tail_probability=0.8, tail_split=0.5, clock_kind="two_tail_uniform")
    options.update(clock)
    return windy_pendulum_config(
        reward_parameters={
            "angle_weight": 10.0,
            "velocity_weight": 1.0,
            "velocity_scale": 2.0 * math.pi,
            "effort_weight": 0.01,
        },
        reset_parameters={"angle_radius": 0.3, "velocity_radius": 0.5},
        **options,
    )


def acrobot_config():
    plant = {
        "m1": 1.0,
        "m2": 1.0,
        "l1": 1.0,
        "l2": 2.0,
        "lc1": 0.5,
        "lc2": 1.0,
        "I1": 0.083,
        "I2": 0.33,
        "gravity": 9.8,
    }
    a1 = plant["I1"] + plant["lc1"] ** 2 + plant["l1"] ** 2
    a2 = plant["I2"] + plant["lc2"] ** 2
    a3 = plant["l1"] * plant["lc2"]
    b_total = (plant["lc1"] + plant["l1"] + plant["lc2"]) * plant["gravity"]
    span = 2.0 * b_total
    omega = math.sqrt(2.0 * span / (a1 + a2 + 2.0 * a3))
    return acrobot_xk_config(
        failure_limits={
            "elbow_angle": 4.0 * math.pi,
            "elbow_rate": 4.0 * math.pi,
            "shoulder_rate": 2.0 * omega,
        },
        reward_parameters={
            "energy_scale": span,
            "upright_energy": b_total,
            "lyapunov_k_p": 61.2,
            "reward_floor": 1e-6,
            "failure_rate": -1.0,
        },
        reset_parameters={"release_angle_low": 0.05, "release_angle_high": 0.5},
        plant_parameters=plant,
    )


def cartpole_config(damping=True):
    """The cart-pole swing-up; since 2026-11-10 its rail is dm_control's soft limit
    (cart_range and the limit_* keys) and it has no failure set."""
    return cartpole_swingup_config(
        failure_limits={},
        reward_parameters={"cart_margin": 2.0, "velocity_margin": 5.0, "failure_rate": 0.0},
        reset_parameters={
            "start_angle": math.pi,
            "position_noise": 0.01,
            "angle_noise": 0.01,
            "velocity_noise": 0.01,
        },
        plant_parameters={
            "cart_mass": 1.0,
            "pole_mass": 0.1,
            "pole_com": 0.5,
            "pole_inertia": 0.00942459,
            "pole_length": 1.0,
            "cart_damping": 5e-4 if damping else 0.0,
            "pole_damping": 2e-6 if damping else 0.0,
            "gravity": 9.81,
            "cart_range": 1.8,
            "limit_timeconst": 0.08,
            "limit_dampratio": 1.0,
            "limit_d0": 0.9,
            "limit_dmax": 0.95,
            "limit_width": 0.001,
            "limit_midpoint": 0.5,
            "limit_power": 2.0,
        },
    )


def random_states(config, rng, count):
    """Draw states strictly inside the failure limits (the cart-pole: inside its rail)."""
    limits = config["failure_limits"]
    if config["name"] == "windy_pendulum":
        high = np.array([limits["angle"], limits["velocity"]])
    elif config["name"] == "cartpole_swingup":
        high = np.array([config["plant_parameters"]["cart_range"], math.pi, 5.0, 15.0])
    else:
        high = np.array(
            [math.pi, limits["elbow_angle"], limits["shoulder_rate"], limits["elbow_rate"]]
        )
    return rng.uniform(-0.98 * high, 0.98 * high, size=(count, len(high)))


def central_decision(env):
    return 0.5 * (env.action_space.low + env.action_space.high)


class TestClock(unittest.TestCase):
    def test_empirical_law_matches_pmf(self):
        config = windy_config()["clock"]
        clock = DiscreteIntervalClock(config, seed=1)
        steps, probabilities = clock_duration_pmf(config)
        count = 200000
        draws = np.array([clock.sample_steps() for _ in range(count)])
        counts = np.array([(draws == step).sum() for step in steps]) / count
        standard = np.sqrt(probabilities * (1.0 - probabilities) / count)
        self.assertTrue((np.abs(counts - probabilities) <= 5.0 * standard + 1e-12).all())
        self.assertAlmostEqual(draws.mean() * 0.001, 0.01, delta=2e-4)

    def test_shorten_only_at_limit(self):
        clock = DiscreteIntervalClock(windy_config()["clock"], seed=2)
        draws = [clock.sample_steps(remaining_steps=5) for _ in range(500)]
        self.assertEqual(set(draws), {2, 3, 4, 5})
        with self.assertRaises(ValueError):
            clock.sample_steps(remaining_steps=0)

    def test_reseed_repeats_and_state_round_trip(self):
        clock = DiscreteIntervalClock(windy_config()["clock"], seed=2)
        first = [clock.sample_steps() for _ in range(30)]
        clock.reset(seed=2)
        self.assertEqual(first, [clock.sample_steps() for _ in range(30)])
        state = clock.state_dict()
        expected = [clock.sample_steps() for _ in range(30)]
        clock.load_state_dict(state)
        self.assertEqual(expected, [clock.sample_steps() for _ in range(30)])

    def test_fixed_clock(self):
        clock = DiscreteIntervalClock(acrobot_config()["clock"], seed=0)
        self.assertEqual({clock.sample_steps() for _ in range(20)}, {10})
        self.assertAlmostEqual(clock.sample_duration(), 0.01)

    def test_whole_step_durations(self):
        self.assertEqual(duration_in_physics_steps(0.018, 0.001), 18)
        with self.assertRaises(ValueError):
            duration_in_physics_steps(0.0105, 0.001)
        with self.assertRaises(ValueError):
            duration_in_physics_steps(0.0, 0.001)

    def test_expected_weight_matches_law(self):
        config = windy_config()["clock"]
        clock = DiscreteIntervalClock(config)
        steps, probabilities = clock_duration_pmf(config)
        weights = (1.0 - np.exp(-0.1 * steps * 0.001)) / 0.1
        self.assertAlmostEqual(
            clock.expected_discount_weight(0.1), float(probabilities @ weights), places=12
        )


class TestOracleAgainstMujoco(unittest.TestCase):
    def check(self, config):
        env = make_env(config, seed=0)
        rng = np.random.default_rng(0)
        drift, _ = oracle_callbacks(config)
        limit = float(config["controller"]["torque_high"][0])
        for state in random_states(config, rng, 200):
            torque = rng.uniform(-limit, limit, size=1)
            truth = env.mujoco_drift(state, torque)
            analytic = env.oracle_drift(state, torque)
            tensor = drift(
                torch.as_tensor(state[None], dtype=torch.float64),
                torch.as_tensor(torque[None], dtype=torch.float64),
            )[0].numpy()
            scale = np.maximum(np.abs(truth), 1.0)
            self.assertLess(np.max(np.abs(analytic - truth) / scale), 1e-6)
            self.assertLess(np.max(np.abs(tensor - truth) / scale), 1e-6)

    def test_windy_drift(self):
        self.check(windy_config())

    def test_acrobot_drift(self):
        self.check(acrobot_config())

    def test_cartpole_drift(self):
        self.check(cartpole_config())

    def test_mujoco_drift_leaves_episode_untouched(self):
        env = make_env(windy_config(), seed=0)
        env.reset(seed=0)
        before = env.observe()
        env.mujoco_drift(np.array([0.2, 0.1]), np.array([1.0]))
        np.testing.assert_array_equal(env.observe(), before)

    def test_callbacks_are_differentiable_with_torch_func(self):
        for config in (windy_config(), acrobot_config()):
            drift, diffusion = oracle_callbacks(config)
            rng = np.random.default_rng(1)
            states = torch.as_tensor(random_states(config, rng, 4), dtype=torch.float64)
            actions = torch.as_tensor(rng.uniform(-5, 5, (4, 1)), dtype=torch.float64)

            def single(state, action):
                return drift(state[None], action[None])[0]

            jacobian = torch.func.vmap(torch.func.jacfwd(single))(states, actions)
            epsilon = 1e-6
            for i in range(states.shape[1]):
                bump = torch.zeros_like(states)
                bump[:, i] = epsilon
                numeric = (drift(states + bump, actions) - drift(states - bump, actions)) / (
                    2.0 * epsilon
                )
                self.assertLess((jacobian[:, :, i] - numeric).abs().max().item(), 1e-6)
            self.assertEqual(diffusion(states).ndim, 3)
            sigma = torch.func.vmap(lambda s: diffusion(s[None])[0])(states)
            self.assertEqual(sigma.shape, diffusion(states).shape)

    def test_diffusion_shapes(self):
        states = torch.zeros(3, 2)
        _, diffusion = oracle_callbacks(windy_config())
        expected = torch.tensor([[0.0], [3.0]]).expand(3, 2, 1)
        self.assertTrue(torch.equal(diffusion(states), expected))
        states = torch.zeros(3, 4)
        _, diffusion = oracle_callbacks(acrobot_config())
        self.assertTrue(torch.equal(diffusion(states), torch.zeros(3, 4, 1)))

    def test_input_matrices(self):
        np.testing.assert_array_equal(
            make_env(windy_config()).oracle_input_matrix(), [[1.0]]
        )
        np.testing.assert_array_equal(
            make_env(acrobot_config()).oracle_input_matrix(), [[0.0], [1.0]]
        )

    def test_make_oracle_plant(self):
        for config in (windy_config(), acrobot_config()):
            env = make_env(config, seed=0)
            plant = make_oracle_plant(env)
            rng = np.random.default_rng(2)
            states = random_states(config, rng, 8)
            actions = rng.uniform(-5.0, 5.0, (8, 1))
            drift = plant.drift(
                torch.as_tensor(states, dtype=torch.get_default_dtype()),
                torch.as_tensor(actions, dtype=torch.get_default_dtype()),
            )
            truth = np.stack([env.mujoco_drift(s, a) for s, a in zip(states, actions)])
            self.assertLess(np.max(np.abs(drift.numpy() - truth) / (1.0 + np.abs(truth))), 1e-4)
            self.assertEqual(plant.state_spec, env.state_spec)
            self.assertEqual(
                plant.input_matrix().tolist(), env.oracle_input_matrix().tolist()
            )


class TestTaskTensors(unittest.TestCase):
    def check_agreement(self, config, *, failures=True):
        """failures: the task has a failure set (the cart-pole has none since 2026-11-10,
        so its mask stays all False beyond the rail too)."""
        env = make_env(config, seed=0)
        rng = np.random.default_rng(3)
        states = random_states(config, rng, 200)
        states[:50] *= 1.1  # Some rows fall outside the limits.
        torques = rng.uniform(-12.0, 12.0, (200, 1))
        rates = reward_rate_tensor(
            torch.as_tensor(states), torch.as_tensor(torques), config
        )
        self.assertEqual(rates.shape, (200, 1))
        mask = failure_mask_tensor(torch.as_tensor(states), config)
        self.assertEqual(mask.shape, (200,))
        if failures:
            self.assertGreater(mask.sum().item(), 0)
            self.assertLess(mask.sum().item(), 200)
        else:
            self.assertEqual(mask.sum().item(), 0)
        for i in range(200):
            self.assertAlmostEqual(
                rates[i, 0].item(), env.reward_rate(states[i], torques[i]), places=10
            )
            self.assertEqual(bool(mask[i]), env.failed(states[i]))
        payoff = terminal_payoff_tensor(torch.as_tensor(states), config)
        self.assertEqual(payoff.shape, (200, 1))
        self.assertAlmostEqual(payoff[0, 0].item(), env.terminal_payoff(states[0]), places=10)

    def test_windy_agreement(self):
        self.check_agreement(windy_config())

    def test_acrobot_agreement(self):
        self.check_agreement(acrobot_config())

    def test_cartpole_agreement(self):
        self.check_agreement(cartpole_config(), failures=False)

    def test_rates_bounded_inside_limits(self):
        config = windy_config()
        env = make_env(config)
        rng = np.random.default_rng(4)
        limit = float(config["controller"]["torque_high"][0])
        for state in random_states(config, rng, 500):
            rate = env.reward_rate(state, rng.uniform(-limit, limit, 1))
            self.assertLessEqual(rate, 0.0)
            self.assertGreaterEqual(rate, -1.0)

    def test_acrobot_rates_bounded_inside_limits(self):
        """r1 lies between -log(V_max / V_down), V_max the Lyapunov bound inside the limits, and -log(floor)."""
        config = acrobot_config()
        env = make_env(config)
        reward, limits = config["reward_parameters"], config["failure_limits"]
        plant = config["plant_parameters"]
        a1 = plant["I1"] + plant["lc1"] ** 2 + plant["l1"] ** 2
        a2 = plant["I2"] + plant["lc2"] ** 2
        a3 = plant["l1"] * plant["lc2"]
        shoulder_rate, elbow_rate = limits["shoulder_rate"], limits["elbow_rate"]
        kinetic = 0.5 * (
            (a1 + a2 + 2.0 * a3) * shoulder_rate**2
            + 2.0 * (a2 + a3) * shoulder_rate * elbow_rate
            + a2 * elbow_rate**2
        )
        potential = reward["upright_energy"]  # b1 + b2, the upright potential
        error = max(2.0 * potential, kinetic)
        largest = 0.5 * error**2 + 0.5 * reward["lyapunov_k_p"] * limits["elbow_angle"] ** 2
        lowest = -math.log(largest / (0.5 * reward["energy_scale"] ** 2))
        rng = np.random.default_rng(4)
        limit = float(config["controller"]["torque_high"][0])
        for state in random_states(config, rng, 500):
            rate = env.reward_rate(state, rng.uniform(-limit, limit, 1))
            self.assertLessEqual(rate, -math.log(reward["reward_floor"]))
            self.assertGreaterEqual(rate, lowest)

    def test_windy_reward_is_zero_only_at_upright_rest(self):
        env = make_env(windy_config())
        self.assertEqual(env.reward_rate(np.zeros(2), np.zeros(1)), 0.0)
        self.assertLess(env.reward_rate(np.array([0.1, 0.0]), np.zeros(1)), 0.0)
        limits = env.config["failure_limits"]
        corner = np.array([limits["angle"], limits["velocity"]])
        self.assertAlmostEqual(env.reward_rate(corner, np.array([12.0])), -1.0, places=12)

    def test_acrobot_reward_is_r1(self):
        """State-only, -log(floor) at upright rest, 0 at hanging rest, and reads the unwrapped elbow."""
        config = acrobot_config()
        env = make_env(config)
        top = -math.log(config["reward_parameters"]["reward_floor"])
        upright = np.array([math.pi / 2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(env.reward_rate(upright, np.array([3.0])), top, places=12)
        self.assertEqual(env.reward_rate(upright, np.array([3.0])), env.reward_rate(upright, np.zeros(1)))
        hanging = np.array([-math.pi / 2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(env.reward_rate(hanging, np.zeros(1)), 0.0, places=10)
        wound = np.array([math.pi / 2.0, 2.0 * math.pi, 0.0, 0.0])
        self.assertLess(env.reward_rate(wound, np.zeros(1)), 0.0)
        turned = np.array([math.pi / 2.0 + 2.0 * math.pi, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(env.reward_rate(turned, np.zeros(1)), top, places=12)

    def test_boundary_functions(self):
        windy = windy_config()
        lift, vanishing = terminal_boundary_functions(windy)
        limits = windy["failure_limits"]
        inside = torch.tensor([[0.3, 2.0], [0.0, 0.0], [-1.5, -12.0]])
        self.assertTrue(torch.all(vanishing(inside) > 0.0))
        self.assertEqual(vanishing(inside).shape, (3, 1))
        self.assertTrue(torch.equal(lift(inside), torch.full((3, 1), -10.0)))
        boundary = torch.tensor(
            [
                [limits["angle"], 1.0],
                [-limits["angle"], 1.0],
                [0.2, limits["velocity"]],
                [0.2, -limits["velocity"]],
            ],
            dtype=torch.float64,
        )
        self.assertLess(vanishing(boundary).abs().max().item(), 1e-12)

        acrobot = acrobot_config()
        lift, vanishing = terminal_boundary_functions(acrobot)
        limits = acrobot["failure_limits"]
        inside = torch.tensor([[3.0, 11.0, 5.0, 8.0], [-3.0, -11.0, -5.0, -8.0]])
        self.assertTrue(torch.all(vanishing(inside) > 0.0))
        self.assertTrue(torch.equal(lift(inside), torch.full((2, 1), -2.0)))
        boundary = torch.tensor(
            [
                [3.0, limits["elbow_angle"], 1.0, 1.0],
                [3.0, 1.0, limits["shoulder_rate"], 1.0],
                [3.0, 1.0, 1.0, -limits["elbow_rate"]],
            ],
            dtype=torch.float64,
        )
        self.assertLess(vanishing(boundary).abs().max().item(), 1e-12)
        # The elbow factor reads the unwrapped angle, not its sine.
        wound = torch.tensor([[0.0, 2.0 * math.pi, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        values = vanishing(wound)[:, 0]
        self.assertLess(values[0].item(), values[1].item())

    def test_state_spec_rejects_bad_projection(self):
        config = windy_config()
        self.assertIs(state_spec(config), config["state_spec"])
        config["state_spec"]["observation_projection"] = np.eye(2)
        with self.assertRaises(ValueError):
            state_spec(config)

    def test_missing_keys_raise(self):
        config = windy_pendulum_config(
            reward_parameters={},
            reset_parameters={},
            tail_probability=0.8,
            tail_split=0.5,
            clock_kind="two_tail_uniform",
        )
        with self.assertRaisesRegex(ValueError, "reward_parameters"):
            make_env(config)


class TestIntervalPhysics(unittest.TestCase):
    def test_gust_variance_matches_diffusion(self):
        """A held zero-torque velocity increment has variance sigma_w^2 Delta / I^2."""
        config = windy_config()
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        env.reset(seed=11)
        duration = 0.005
        increments = []
        for _ in range(6000):
            env.reset(options={"state": np.zeros(2)})
            transition = env.step_dt(np.zeros(1), duration)
            mean = env.oracle_drift(transition["state"], np.zeros(1))[1] * duration
            increments.append(transition["next_state"][1] - mean)
        increments = np.array(increments)
        expected = 9.0 * duration
        tolerance = 5.0 * expected * math.sqrt(2.0 / len(increments))
        self.assertAlmostEqual(increments.var(), expected, delta=tolerance)
        self.assertAlmostEqual(
            increments.mean(), 0.0, delta=5.0 * math.sqrt(expected / len(increments))
        )

    def test_gust_scales_with_inertia(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], mass=2.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        np.testing.assert_allclose(env.oracle_diffusion(np.zeros(2)), [[0.0], [1.5]])
        env.reset(seed=1)
        increments = []
        for _ in range(4000):
            env.reset(options={"state": np.zeros(2)})
            increments.append(env.step_dt(np.zeros(1), 0.005)["next_state"][1])
        expected = 1.5**2 * 0.005
        self.assertAlmostEqual(
            np.var(increments), expected, delta=5.0 * expected * math.sqrt(2.0 / 4000)
        )

    def test_gust_acts_outside_motor_clip(self):
        """A saturated motor shifts the increment mean and leaves the gust variance intact."""
        env = WindyPendulumEnv(windy_config(), decision_mode="torque", seed=0)
        env.reset(seed=2)
        duration = 0.005
        increments = []
        for _ in range(6000):
            env.reset(options={"state": np.zeros(2)})
            increments.append(env.step_dt(np.array([12.0]), duration)["next_state"][1])
        expected = 9.0 * duration
        self.assertAlmostEqual(
            np.var(increments), expected, delta=5.0 * expected * math.sqrt(2.0 / 6000)
        )
        self.assertAlmostEqual(
            np.mean(increments), 12.0 * duration, delta=5.0 * math.sqrt(expected / 6000)
        )

    def test_noise_free_step_is_semi_implicit_euler(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        start = np.array([0.1, 0.2])
        env.reset(seed=0, options={"state": start})
        torque = np.array([1.5])
        transition = env.step_dt(torque, 0.003)
        angle, speed = start
        for _ in range(3):
            speed = speed + 0.001 * (9.81 * math.sin(angle) + 1.5)
            angle = angle + 0.001 * speed
        self.assertAlmostEqual(transition["next_state"][0], angle, delta=1e-7)
        self.assertAlmostEqual(transition["next_state"][1], speed, delta=1e-7)

    def test_scheduled_duration_survives_a_failure(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        limit = config["failure_limits"]["angle"]
        env.reset(seed=0, options={"state": np.array([limit - 1e-4, 1.0])})
        transition = env.step_dt(np.array([0.0]), 0.01)
        self.assertTrue(transition["terminated"])
        self.assertLess(transition["duration"], 0.01)
        self.assertAlmostEqual(transition["scheduled_duration"], 0.01)

    def test_time_limit_shortens_the_schedule(self):
        config = windy_config()
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        env.reset(seed=0)
        env._steps_taken = env.remaining_steps + env._steps_taken - 3
        transition = env.step_dt(np.array([0.0]), 0.01)
        if transition["terminated"]:
            self.skipTest("The random start failed before the time limit.")
        self.assertAlmostEqual(transition["scheduled_duration"], 0.003)
        self.assertAlmostEqual(transition["duration"], 0.003)

    def test_model_path_return_matches_the_environment(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        limit = config["failure_limits"]["angle"]
        terms = {
            "reward_rate": lambda s, a: reward_rate_tensor(s, a, config),
            "terminal_payoff": lambda s: terminal_payoff_tensor(s, config),
            "discount_rate": config["discount_rate"],
        }
        for start, name in ((0.1, "survives"), (limit - 1e-3, "fails")):
            with self.subTest(name):
                env = WindyPendulumEnv(config, decision_mode="feedback", seed=0)
                state = np.array([start, 2.0])
                env.reset(seed=0, options={"state": state})
                decision = central_decision(env)
                transition = env.step_dt(decision, 0.01)
                self.assertEqual(transition["terminated"], name == "fails")
                dtype = torch.float64
                paths = stochastic_rollout(
                    make_oracle_plant(env),
                    env.controller,
                    torch.as_tensor(state, dtype=dtype).unsqueeze(0),
                    torch.as_tensor(decision, dtype=dtype).unsqueeze(0),
                    torch.tensor([[0.01]], dtype=dtype),
                    lambda s: failure_mask_tensor(s, config),
                    maximum_step=config["physics_dt"],
                    physics_dt=config["physics_dt"],
                    controller_refresh_steps=config["controller"]["refresh_steps"],
                    path_count=1,
                    brownian_increments=torch.zeros(1, 1, 10, 1, dtype=dtype),
                    return_terms=terms,
                    # The windy simulator's step (MuJoCo Euler, damping 0) is this map.
                    integrator="semi_implicit",
                )
                simulated = (paths["running_rewards"] + paths["failure_payoffs"]).item()
                self.assertAlmostEqual(simulated, transition["reward"], delta=2e-6)
                self.assertEqual(bool(paths["survived"].item()), name == "survives")

    def test_interval_reward_hand_computation(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        start = np.array([0.1, 0.2])
        env.reset(seed=0, options={"state": start})
        torque = np.array([1.5])
        transition = env.step_dt(torque, 0.002)

        reward = make_env(config)
        lam = 0.1
        h = 0.001
        angle, speed = start
        total = 0.0
        for j in range(2):
            before = reward.reward_rate(np.array([angle, speed]), torque)
            speed = speed + h * (9.81 * math.sin(angle) + 1.5)
            angle = angle + h * speed
            after = reward.reward_rate(np.array([angle, speed]), torque)
            total += 0.5 * h * (math.exp(-lam * j * h) * before + math.exp(-lam * (j + 1) * h) * after)
        self.assertAlmostEqual(transition["reward"], total, places=9)
        self.assertEqual(transition["duration"], 0.002)

    def test_trapezoid_with_jump_knots(self):
        env = make_env(windy_config())
        reward = env.integrate_interval_reward(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 0.5, 0.5, 1.0]),
            terminated=False,
            terminal_state=np.zeros(2),
        )
        lam = 0.1
        expected = 0.5 * 0.5 * (1.0 + 2.0 * math.exp(-lam * 0.5)) + 0.5 * 0.5 * (
            3.0 * math.exp(-lam * 0.5) + 4.0 * math.exp(-lam)
        )
        self.assertAlmostEqual(reward, expected, places=12)

    def test_failure_charged_once_and_interval_shortened(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        limit = config["failure_limits"]["velocity"]
        start = np.array([0.0, limit - 0.03])
        env.reset(seed=0, options={"state": start})
        torque = np.array([12.0])
        transition = env.step_dt(torque, 0.018)
        self.assertTrue(transition["terminated"])
        self.assertFalse(transition["truncated"])
        self.assertTrue(env.failed(transition["next_state"]))

        # The velocity gains about 12 mrad/s per physics step, so failure comes on step 3.
        scorer = make_env(config)
        lam, h = 0.1, 0.001
        angle, speed = start
        rates = 0.0
        steps = 0
        while abs(speed) < limit and abs(angle) < config["failure_limits"]["angle"]:
            before = scorer.reward_rate(np.array([angle, speed]), torque)
            speed = speed + h * (9.81 * math.sin(angle) + 12.0)
            angle = angle + h * speed
            after = scorer.reward_rate(np.array([angle, speed]), torque)
            rates += 0.5 * h * (
                math.exp(-lam * steps * h) * before + math.exp(-lam * (steps + 1) * h) * after
            )
            steps += 1
        self.assertEqual(steps, 3)
        self.assertAlmostEqual(transition["duration"], steps * h, places=12)
        self.assertAlmostEqual(
            transition["next_time"] - transition["time"], transition["duration"], places=12
        )
        charge = math.exp(-lam * steps * h) * (-1.0 / lam)
        self.assertAlmostEqual(transition["reward"], rates + charge, places=8)
        with self.assertRaises(RuntimeError):
            env.step_dt(np.array([0.0]), 0.002)

    def test_truncation_at_episode_end(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        config["episode_duration"] = 0.01
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        env.reset(seed=0, options={"state": np.zeros(2)})
        first = env.step_dt(np.zeros(1), 0.006)
        self.assertFalse(first["terminated"] or first["truncated"])
        last = env.step_dt(np.zeros(1), 0.018)
        self.assertTrue(last["truncated"])
        self.assertFalse(last["terminated"])
        self.assertAlmostEqual(last["duration"], 0.004, places=12)
        self.assertAlmostEqual(last["next_time"], 0.01, places=12)
        np.testing.assert_allclose(last["next_state"], env.observe())
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(1))

    def test_failure_at_episode_end_is_termination(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        config["episode_duration"] = 0.002
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        limit = config["failure_limits"]["velocity"]
        env.reset(seed=0, options={"state": np.array([0.0, limit - 1e-6])})
        last = env.step_dt(np.array([12.0]), 0.002)
        self.assertTrue(last["terminated"])
        self.assertFalse(last["truncated"])

    def test_non_whole_step_duration_raises(self):
        env = make_env(windy_config(), seed=0)
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step_dt(central_decision(env), 0.0105)

    def test_step_returns_transition_with_clock_duration(self):
        env = make_env(windy_config(), seed=0)
        env.reset(seed=0)
        allowed = set(range(2, 19))
        for _ in range(20):
            next_state, reward, terminated, truncated, info = env.step(central_decision(env))
            transition = info["transition"]
            self.assertIs(transition["next_state"], next_state)
            self.assertEqual(transition["reward"], reward)
            self.assertEqual(transition["terminated"], terminated)
            self.assertEqual(transition["truncated"], truncated)
            if terminated:
                break
            self.assertIn(round(transition["duration"] / 0.001), allowed)

    def test_feedback_holds_setpoint_and_refresh_cadence(self):
        # The sample-and-hold cadence of refresh_mode "held" (tests/test_ph_refresh_mode.py
        # covers the default "continuous" law).
        config = acrobot_config()
        config["controller"] = dict(config["controller"], refresh_steps=3, refresh_mode="held")
        env = AcrobotXKEnv(config, seed=0)
        env.reset(seed=0, options={"state": np.array([-1.4, 0.2, 0.0, 0.0])})
        torques = []
        physics_step = env._physics_step

        def recording(torque):
            torques.append(np.array(torque))
            physics_step(torque)

        env._physics_step = recording
        env.step_dt(np.array([0.5, 20.0, 1.0]), 0.007)
        # The offset resolves once against the elbow at the interval start.
        self.assertAlmostEqual(env._resolved_decision[0], 0.7, places=12)
        self.assertEqual(len(torques), 7)
        np.testing.assert_array_equal(torques[0], torques[1])
        np.testing.assert_array_equal(torques[0], torques[2])
        self.assertFalse(np.array_equal(torques[2], torques[3]))
        np.testing.assert_array_equal(torques[3], torques[5])
        self.assertAlmostEqual(float(torques[0][0]), -20.0 * (0.2 - 0.7) - 0.0, delta=1e-12)

    def test_torque_is_clipped_and_held(self):
        env = make_env(windy_config(), decision_mode="torque", seed=0)
        env.reset(seed=0, options={"state": np.array([0.0, 0.0])})
        seen = []
        physics_step = env._physics_step

        def recording(torque):
            seen.append(np.array(torque))
            physics_step(torque)

        env._physics_step = recording
        transition = env.step_dt(np.array([50.0]), 0.004)
        self.assertEqual(float(transition["decision"][0]), 12.0)
        for torque in seen:
            self.assertEqual(float(torque[0]), 12.0)

    def test_saturation_diagnostic(self):
        env = make_env(windy_config(), seed=0)
        env.reset(seed=0, options={"state": np.array([0.4, 0.0])})
        env.step_dt(np.array([-0.5, 100.0, 0.0]), 0.002)
        self.assertEqual(env.last_interval["physics_steps"], 2)
        self.assertEqual(env.last_interval["saturated_fraction"], 1.0)

    def test_observation_wraps_shoulder_but_not_elbow(self):
        config = acrobot_config()
        env = AcrobotXKEnv(config, seed=0)
        env.reset(seed=0, options={"state": np.array([3.5, 7.0, 0.0, 0.0])})
        observed = env.observe()
        self.assertAlmostEqual(observed[0], 3.5 - 2.0 * math.pi, places=12)
        self.assertEqual(observed[1], 7.0)
        wound = np.array([0.0, config["failure_limits"]["elbow_angle"], 0.0, 0.0])
        self.assertTrue(env.failed(wound))
        wound[1] -= 2.0 * math.pi
        self.assertFalse(env.failed(wound))

    def test_acrobot_rk4_conserves_energy_without_torque(self):
        env = make_env(acrobot_config(), decision_mode="torque", seed=0)
        self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_RK4)
        env.reset(seed=0, options={"state": np.array([-1.0, 0.1, 0.0, 0.0])})
        for _ in range(100):
            env.step_dt(np.zeros(1), 0.01)
        self.assertAlmostEqual(
            float(env.data.time), 1.0, places=9
        )
        grouped = env._grouped
        state = env.observe()
        m11, m12, m22 = env._mass_matrix(state[1])
        kinetic = 0.5 * (m11 * state[2] ** 2 + 2.0 * m12 * state[2] * state[3] + m22 * state[3] ** 2)
        potential = grouped[3] * math.sin(state[0]) + grouped[4] * math.sin(state[0] + state[1])
        start_potential = grouped[3] * math.sin(-1.0) + grouped[4] * math.sin(-0.9)
        self.assertAlmostEqual(kinetic + potential, start_potential, places=6)

    def test_windy_uses_euler(self):
        env = make_env(windy_config())
        self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
        self.assertEqual(env.model.opt.timestep, 0.001)

    def test_acrobot_release_reset(self):
        env = make_env(acrobot_config(), seed=4)
        for _ in range(100):
            state, _ = env.reset()
            displacement = abs(state[0] + math.pi / 2.0)
            self.assertGreaterEqual(displacement, 0.05)
            self.assertLessEqual(displacement, 0.5)
            self.assertEqual(state[1], 0.0)
            np.testing.assert_array_equal(state[2:], np.zeros(2))

    def test_windy_reset_radii(self):
        env = make_env(windy_config(), seed=4)
        for _ in range(100):
            state, _ = env.reset()
            self.assertLessEqual(abs(state[0]), 0.3)
            self.assertLessEqual(abs(state[1]), 0.5)


class TestReproducibility(unittest.TestCase):
    def rollout(self, env, count, decision_rng=None):
        rows = []
        for _ in range(count):
            decision = (
                env.action_space.sample()
                if decision_rng is None
                else decision_rng.uniform(env.action_space.low, env.action_space.high)
            )
            if env.done:
                env.reset()
            duration = env.sample_duration()
            transition = env.step_dt(decision, duration)
            rows.append(
                (
                    transition["next_state"].copy(),
                    transition["reward"],
                    transition["duration"],
                    transition["terminated"],
                )
            )
        return rows

    def assert_rows_equal(self, first, second):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])

    def test_snapshot_restore_replays_exactly(self):
        for config in (windy_config(), acrobot_config()):
            env = make_env(config, seed=0)
            env.reset(seed=5)
            self.rollout(env, 3, np.random.default_rng(0))
            snapshot = env.snapshot()
            first = self.rollout(env, 200, np.random.default_rng(1))
            env.restore(snapshot)
            second = self.rollout(env, 200, np.random.default_rng(1))
            self.assert_rows_equal(first, second)

    def test_snapshot_is_independent_of_later_steps(self):
        env = make_env(windy_config(), seed=0)
        env.reset(seed=5)
        snapshot = env.snapshot()
        saved = np.array(snapshot["qpos"])
        self.rollout(env, 5, np.random.default_rng(0))
        np.testing.assert_array_equal(snapshot["qpos"], saved)

    def test_same_seed_same_wind_clock_and_reset(self):
        """Different decisions on one seed see the same reset, clock and gust stream."""
        config = windy_config()
        first = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        second = WindyPendulumEnv(config, decision_mode="torque", seed=99)
        start_a, _ = first.reset(seed=21)
        start_b, _ = second.reset(seed=21)
        np.testing.assert_array_equal(start_a, start_b)
        for _ in range(30):
            duration_a = first.sample_duration()
            duration_b = second.sample_duration()
            self.assertEqual(duration_a, duration_b)
            first._apply_disturbance()
            second._apply_disturbance()
            self.assertEqual(float(first.data.qfrc_applied[0]), float(second.data.qfrc_applied[0]))

    def test_same_policy_and_seed_repeats_trajectory(self):
        config = windy_config()
        rows = []
        for _ in range(2):
            env = make_env(config, seed=0)
            env.reset(seed=3)
            rows.append(self.rollout(env, 100, np.random.default_rng(7)))
        self.assert_rows_equal(rows[0], rows[1])

    def test_different_seed_differs(self):
        config = windy_config()
        env = make_env(config, seed=0)
        env.reset(seed=3)
        first = self.rollout(env, 5, np.random.default_rng(7))
        env.reset(seed=4)
        second = self.rollout(env, 5, np.random.default_rng(7))
        self.assertFalse(np.array_equal(first[0][0], second[0][0]))

    def test_decision_modes_share_streams(self):
        config = windy_config()
        feedback = make_env(config, decision_mode="feedback")
        torque = make_env(config, decision_mode="torque")
        start_a, _ = feedback.reset(seed=8)
        start_b, _ = torque.reset(seed=8)
        np.testing.assert_array_equal(start_a, start_b)
        self.assertEqual(feedback.sample_duration(), torque.sample_duration())
        self.assertEqual(feedback.action_space.shape, (3,))
        self.assertEqual(torque.action_space.shape, (1,))


class TestCartpole(unittest.TestCase):
    def rate(self, state, force=0.0):
        return make_env(cartpole_config()).reward_rate(np.asarray(state, dtype=float), np.array([force]))

    def test_reward_is_dm_control_swingup(self):
        self.assertAlmostEqual(self.rate([0.0, 0.0, 0.0, 0.0]), 1.0, places=12)
        self.assertAlmostEqual(self.rate([0.0, math.pi, 0.0, 0.0]), 0.0, places=12)
        self.assertAlmostEqual(self.rate([0.0, 0.5 * math.pi, 0.0, 0.0]), 0.5, places=12)
        # At each margin dm_control's gaussian tolerance is 0.1, and full control scores 0.
        self.assertAlmostEqual(self.rate([2.0, 0.0, 0.0, 0.0]), 0.55, places=12)
        self.assertAlmostEqual(self.rate([0.0, 0.0, 0.0, 5.0]), 0.55, places=12)
        self.assertAlmostEqual(self.rate([0.0, 0.0, 0.0, 0.0], 10.0), 0.8, places=12)
        self.assertAlmostEqual(self.rate([0.0, 0.0, 0.0, 0.0], -5.0), 0.95, places=12)
        # The cart rate does not enter.
        self.assertAlmostEqual(self.rate([0.0, 0.0, 3.0, 0.0]), 1.0, places=12)

    def test_layout_and_input(self):
        env = make_env(cartpole_config())
        self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_RK4)
        self.assertEqual(env.wrapped_observation_indices, (1,))
        np.testing.assert_array_equal(env.oracle_input_matrix(), [[1.0], [0.0]])
        _, diffusion = oracle_callbacks(cartpole_config())
        self.assertTrue(torch.equal(diffusion(torch.zeros(3, 4)), torch.zeros(3, 4, 1)))
        env.reset(seed=0, options={"state": np.array([1.5, 3.5, 0.0, 0.0])})
        state = env.observe()
        self.assertEqual(state[0], 1.5)
        self.assertAlmostEqual(state[1], 3.5 - 2.0 * math.pi, places=12)

    def test_no_failure_or_lift_at_the_rail(self):
        """Since 2026-11-10 the rail is dm_control's soft limit: no failure, H = 0, B = 1."""
        config = cartpole_config()
        env = make_env(config)
        self.assertFalse(env.failed(np.array([1.8, 0.0, 0.0, 0.0])))
        self.assertFalse(env.failed(np.array([-1.85, 3.0, 0.0, 0.0])))
        self.assertFalse(env.failed(np.array([1.79, 0.0, 9.0, 30.0])))
        self.assertEqual(env.terminal_payoff(np.zeros(4)), 0.0)
        lift, vanishing = terminal_boundary_functions(config)
        states = torch.tensor([[1.8, 0.3, 1.0, 2.0], [-1.9, 3.0, 0.0, 0.0], [0.9, 1.0, 5.0, 9.0]])
        self.assertTrue(torch.equal(vanishing(states), torch.ones(3, 1)))
        self.assertTrue(torch.equal(lift(states), torch.zeros(3, 1)))
        self.assertEqual(env.model.jnt_limited.tolist(), [1, 0])
        np.testing.assert_array_equal(env.model.jnt_range[0], [-1.8, 1.8])

    def test_rk4_conserves_energy_without_force_or_damping(self):
        env = make_env(cartpole_config(damping=False), decision_mode="torque", seed=0)
        start = np.array([0.0, math.pi - 0.3, 0.0, 0.0])
        env.reset(seed=0, options={"state": start})
        for _ in range(100):
            env.step_dt(np.zeros(1), 0.01)

        def energy(state):
            _, angle, cart_rate, pole_rate = state
            kinetic = 0.5 * (
                1.1 * cart_rate**2
                + 2.0 * 0.05 * math.cos(angle) * cart_rate * pole_rate
                + (0.00942459 + 0.025) * pole_rate**2
            )
            return kinetic + 0.05 * 9.81 * math.cos(angle)

        self.assertAlmostEqual(energy(env.observe()), energy(start), places=6)

    def test_swingup_reset(self):
        env = make_env(cartpole_config(), seed=4)
        starts = np.array([env.reset()[0] for _ in range(400)])
        self.assertLess(abs(starts[:, 0].std() - 0.01), 0.002)
        self.assertLess(np.abs(np.abs(starts[:, 1]) - math.pi).max(), 0.06)
        self.assertLess(np.abs(starts[:, 2:]).max(), 0.06)


class TestUndiscountedIntervalReward(unittest.TestCase):
    """MujocoPHEnv.last_undiscounted_reward: the interval's trapezoids, undiscounted, no charge."""

    def test_hand_computation_with_jump_knots(self):
        env = make_env(windy_config())
        self.assertEqual(env.last_undiscounted_reward, 0.0)
        rates = np.array([1.0, 2.0, 3.0, 4.0])
        times = np.array([0.0, 0.5, 0.5, 1.0])
        discounted = env.integrate_interval_reward(
            rates, times, terminated=False, terminal_state=np.zeros(2)
        )
        self.assertAlmostEqual(env.last_undiscounted_reward, 0.5 * 0.5 * 3.0 + 0.5 * 0.5 * 7.0, places=12)
        lam = 0.1
        expected = 0.5 * 0.5 * (1.0 + 2.0 * math.exp(-lam * 0.5)) + 0.5 * 0.5 * (
            3.0 * math.exp(-lam * 0.5) + 4.0 * math.exp(-lam)
        )
        self.assertAlmostEqual(discounted, expected, places=12)

    def test_set_on_every_step_and_excludes_the_failure_charge(self):
        config = windy_config()
        config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=0.0)
        env = WindyPendulumEnv(config, decision_mode="torque", seed=0)
        env.reset(seed=0, options={"state": np.array([0.1, 0.2])})
        values = []
        for torque in (1.5, -3.0):
            transition = env.step_dt(np.array([torque]), 0.004)
            values.append(env.last_undiscounted_reward)
            # Over 4 ms the discount moves the integral by under lambda*Delta relatively.
            self.assertLess(values[-1], 0.0)
            self.assertAlmostEqual(
                transition["reward"] / values[-1], 1.0, delta=0.1 * transition["time"] + 1e-3
            )
        self.assertNotEqual(values[0], values[1])

        limit = config["failure_limits"]["velocity"]
        env.reset(seed=0, options={"state": np.array([0.0, limit - 0.03])})
        transition = env.step_dt(np.array([12.0]), 0.018)
        self.assertTrue(transition["terminated"])
        charge = math.exp(-0.1 * transition["duration"]) * (-1.0 / 0.1)
        self.assertAlmostEqual(
            transition["reward"] - charge, env.last_undiscounted_reward, delta=1e-5
        )
        self.assertGreater(env.last_undiscounted_reward, -0.01)

        env.reset(seed=1)
        self.assertEqual(env.last_undiscounted_reward, 0.0)

    def test_snapshot_round_trip(self):
        env = make_env(windy_config(), seed=2)
        env.reset(seed=2)
        env.step_dt(central_decision(env), 0.01)
        snapshot = env.snapshot()
        kept = env.last_undiscounted_reward
        env.step_dt(central_decision(env), 0.01)
        env.restore(snapshot)
        self.assertEqual(env.last_undiscounted_reward, kept)


def windy_swingup_config(torque_limit=5.0):
    """The swing-up as a config declares it: start at pi, sigma_w 1, no angle failure."""
    config = windy_config()
    config["reset_parameters"] = dict(config["reset_parameters"], start_angle=math.pi)
    config["plant_parameters"] = dict(config["plant_parameters"], wind_intensity=1.0)
    config["failure_limits"] = dict(config["failure_limits"], angle=math.inf)
    config["controller"] = dict(
        config["controller"],
        decision_low=np.array([-math.pi, 0.0, 0.0]),
        decision_high=np.array([math.pi, 100.0, 20.0]),
        torque_low=np.array([-torque_limit]),
        torque_high=np.array([torque_limit]),
    )
    return config


class TestWindySwingup(unittest.TestCase):
    """WindyPendulumEnv reads every swing-up setting from its config."""

    def test_reset_box_motor_and_wind_follow_the_config(self):
        config = windy_swingup_config(torque_limit=5.0)
        env = make_env(config, seed=4)
        for _ in range(50):
            state, _ = env.reset()
            self.assertLessEqual(math.pi - abs(state[0]), 0.3 + 1e-12)
            self.assertLessEqual(abs(state[1]), 0.5)
        np.testing.assert_allclose(env.action_space.low, [-math.pi, 0.0, 0.0])
        np.testing.assert_allclose(env.action_space.high, [math.pi, 100.0, 20.0])
        np.testing.assert_allclose(env.model.actuator_ctrlrange[0], [-5.0, 5.0])
        np.testing.assert_allclose(env.oracle_diffusion(np.zeros(2)), [[0.0], [1.0]])
        torque_env = make_env(config, decision_mode="torque", seed=0)
        np.testing.assert_allclose(torque_env.action_space.high, [5.0])
        # A strong gain against the upright setpoint saturates at the 5 N m motor.
        env.reset(seed=0, options={"state": np.array([1.0, 0.0])})
        env.step_dt(np.array([0.0, 100.0, 0.0]), 0.002)
        self.assertEqual(env.last_interval["saturated_fraction"], 1.0)

    def test_no_angle_failure_and_the_cost_bound_at_pi(self):
        config = windy_swingup_config()
        env = make_env(config, seed=0)
        velocity = config["failure_limits"]["velocity"]
        for angle in (0.0, math.pi / 2.0 + 0.1, -2.5, math.pi - 1e-3):
            self.assertFalse(env.failed(np.array([angle, 0.0])))
            self.assertTrue(env.failed(np.array([angle, velocity])))
        states = torch.tensor([[math.pi - 1e-3, 0.0], [2.0, -velocity], [-3.0, 1.0]])
        mask = failure_mask_tensor(states, config)
        self.assertEqual(mask.tolist(), [False, True, False])
        # The angle cost peaks at pi; with the torque limit the bound is 20 + 4 + 0.125.
        rate = env.reward_rate(np.array([math.pi, velocity]), np.array([5.0]))
        self.assertAlmostEqual(rate, -1.0, places=9)

        lift, vanishing = terminal_boundary_functions(config)
        inside = torch.tensor([[math.pi, 0.0], [0.0, 0.0], [-3.0, 6.0]], dtype=torch.float64)
        torch.testing.assert_close(vanishing(inside)[:, 0], 1.0 - (inside[:, 1] / velocity) ** 2)
        boundary = torch.tensor([[2.0, velocity], [-2.0, -velocity]], dtype=torch.float64)
        self.assertLess(vanishing(boundary).abs().max().item(), 1e-12)
        self.assertTrue(torch.equal(lift(inside), torch.full((3, 1), -10.0, dtype=torch.float64)))
        self.assertTrue(torch.isfinite(vanishing(inside)).all())


# The table rows of TestEndRewardRate: every dm_control family the runs use.
END_RATE_TESTBEDS = (
    "pendulum-swingup",
    "cartpole-swingup",
    "reacher-easy",
    "swimmer-swimmer6",
    "lqr-lqr_2_1",
    "acrobot-swingup",
    "point_mass-easy",
)
# Of those, the rows simulated with RK4 (the swimmer's physics is Euler).
RK4_TESTBEDS = tuple(name for name in END_RATE_TESTBEDS if not name.startswith("swimmer"))


def row_testbed(env_id):
    """The TestbedConfig of a table row (top mode), through the protocol loader."""
    entry = table.find_ph_entry(env_id, "top")
    return run_ph_sac._build_testbed(entry.config["testbed"], entry.config["testbed_settings"])


def record_reward_rate(env):
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


def assert_same_snapshot(case, left, right, path="snapshot"):
    """Nested environment snapshots agree exactly (arrays by value)."""
    if isinstance(left, dict):
        case.assertEqual(set(left), set(right), path)
        for key in left:
            assert_same_snapshot(case, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same_snapshot(case, a, b, f"{path}[{index}]")
    else:
        case.assertEqual(left, right, path)


class TestEndRewardRate(unittest.TestCase):
    """MujocoPHEnv.end_reward_rate, the end-state datum of reward_observation "end_state":
    reward_rate(s', u_e), u_e the held torque in torque mode, the law at s' in feedback
    mode; pure."""

    def test_torque_mode_is_the_last_knot_and_the_task_function(self):
        from algorithms.ph_baseline import end_state_reward_rate

        for env_id in END_RATE_TESTBEDS:
            with self.subTest(testbed=env_id):
                env = make_env(row_testbed(env_id), decision_mode="torque", seed=5)
                calls = record_reward_rate(env)
                env.reset(seed=2)
                rng = np.random.default_rng(0)
                low, high = env.action_space.low, env.action_space.high
                for _ in range(12):
                    if env.done:
                        env.reset()
                    calls.clear()
                    # Half a box beyond each side, so that step_dt clips some decisions.
                    decision = rng.uniform(low - 0.5 * (high - low), high + 0.5 * (high - low))
                    transition = env.step_dt(decision, env.sample_duration())
                    self.assertEqual(len(calls), 2 * env.last_interval["physics_steps"])
                    knot_state, knot_torque, knot_value = calls[-1]
                    np.testing.assert_array_equal(knot_state, transition["next_state"])
                    np.testing.assert_array_equal(knot_torque, transition["decision"])
                    calls.clear()
                    value = env.end_reward_rate(transition)
                    # One call: the rate at s' with the held torque, the last knot bitwise.
                    ((state, torque, direct),) = calls
                    self.assertIsInstance(value, float)
                    self.assertEqual(value, knot_value)
                    self.assertEqual(value, direct)
                    np.testing.assert_array_equal(state, transition["next_state"])
                    np.testing.assert_array_equal(torque, transition["decision"])
                    self.assertEqual(
                        value, env.reward_rate(transition["next_state"], transition["decision"])
                    )
                    # The CT-SAC baseline's end-state rate is the same definition.
                    self.assertEqual(value, end_state_reward_rate(env, transition))
                    # The task's tensor rate, the learners' oracle, agrees.
                    tensor = reward_rate_tensor(
                        torch.as_tensor(transition["next_state"][None], dtype=torch.float64),
                        torch.as_tensor(transition["decision"][None], dtype=torch.float64),
                        env.config,
                    )
                    self.assertAlmostEqual(
                        value, float(tensor[0, 0]), delta=1e-10 * max(1.0, abs(value))
                    )

    def test_feedback_mode_reads_the_law_at_the_end_state(self):
        for env_id in END_RATE_TESTBEDS:
            with self.subTest(testbed=env_id):
                env = make_env(row_testbed(env_id), decision_mode="feedback", seed=5)
                # Continuous refresh with RK4 physics: the law at every stage; the swimmer's
                # Euler physics runs the held path.
                self.assertEqual(env.stage_feedback, env_id in RK4_TESTBEDS)
                calls = record_reward_rate(env)
                env.reset(seed=2)
                rng = np.random.default_rng(1)
                for _ in range(10):
                    if env.done:
                        env.reset()
                    calls.clear()
                    decision = rng.uniform(env.action_space.low, env.action_space.high)
                    transition = env.step_dt(decision, env.sample_duration())
                    knot_state, knot_torque, knot_value = calls[-1]
                    calls.clear()
                    value = env.end_reward_rate(transition)
                    resolved = env.controller.resolve_decision_numpy(
                        transition["state"], transition["decision"]
                    )
                    law = env.controller.action_numpy(transition["next_state"], resolved)
                    ((state, torque, _),) = calls
                    np.testing.assert_array_equal(state, transition["next_state"])
                    np.testing.assert_array_equal(torque, law)
                    if env.stage_feedback:
                        # The law at s' is the torque of the environment's last knot.
                        np.testing.assert_array_equal(knot_torque, law)
                        self.assertEqual(value, knot_value)
        # Under Euler physics (the windy pendulum) the last knot reads the torque held over
        # the last physics step; the end-state rate still reads the law at s'.
        env = make_env(windy_config(), seed=3)
        self.assertFalse(env.stage_feedback)
        env.reset(seed=1)
        transition = env.step_dt(np.array([0.2, 60.0, 5.0]), 0.01)
        resolved = env.controller.resolve_decision_numpy(transition["state"], transition["decision"])
        law = env.controller.action_numpy(transition["next_state"], resolved)
        self.assertEqual(
            env.end_reward_rate(transition), env.reward_rate(transition["next_state"], law)
        )

    def test_terminated_and_shortened_rows(self):
        # A failed interval: s' is the first failing knot, and the rate is read there.
        env = make_env(windy_config(), decision_mode="torque", seed=0)
        env.reset(seed=1)
        calls = record_reward_rate(env)
        transition = None
        for _ in range(500):
            calls.clear()
            transition = env.step_dt(np.array([12.0]), env.sample_duration())
            if transition["terminated"] or transition["truncated"]:
                break
        self.assertTrue(transition["terminated"])
        self.assertTrue(env.failed(transition["next_state"]))
        knot_state, _, knot_value = calls[-1]
        np.testing.assert_array_equal(knot_state, transition["next_state"])
        calls.clear()
        value = env.end_reward_rate(transition)
        self.assertEqual(value, knot_value)
        self.assertTrue(math.isfinite(value))
        # The episode's shortened last interval (planned = min(requested, remaining)): the
        # rate at its final state, truncated.
        testbed = dict(row_testbed("pendulum-swingup"), episode_duration=0.05)
        env = make_env(testbed, decision_mode="torque", seed=0)
        env.reset(seed=3)
        first = env.step_dt(np.array([0.5]), 0.03)
        self.assertFalse(first["truncated"] or first["terminated"])
        calls = record_reward_rate(env)
        last = env.step_dt(np.array([0.5]), 0.03)
        self.assertTrue(last["truncated"])
        self.assertEqual(env.last_interval["physics_steps"], 20)
        self.assertAlmostEqual(last["duration"], 0.02, places=12)
        self.assertAlmostEqual(last["scheduled_duration"], 0.02, places=12)
        knot_state, _, knot_value = calls[-1]
        np.testing.assert_array_equal(knot_state, last["next_state"])
        calls.clear()
        self.assertEqual(env.end_reward_rate(last), knot_value)

    def test_reading_it_changes_no_stream(self):
        for env_id in ("reacher-easy", "pendulum-swingup"):
            for mode in ("torque", "feedback"):
                with self.subTest(testbed=env_id, mode=mode):
                    testbed = row_testbed(env_id)
                    env = make_env(testbed, decision_mode=mode, seed=4)
                    twin = make_env(testbed, decision_mode=mode, seed=4)
                    rng = np.random.default_rng(1)
                    decisions = rng.uniform(
                        env.action_space.low, env.action_space.high, size=(2, env.action_space.shape[0])
                    )
                    for each in (env, twin):
                        each.reset(seed=6)
                    transition = env.step_dt(decisions[0], env.sample_duration())
                    twin.step_dt(decisions[0], twin.sample_duration())
                    before, fields = env.snapshot(), copy.deepcopy(transition)
                    value = env.end_reward_rate(transition)
                    assert_same_snapshot(self, before, env.snapshot())
                    # The transition is read, not written.
                    self.assertEqual(set(transition), set(fields))
                    for key, stored in fields.items():
                        np.testing.assert_array_equal(transition[key], stored, err_msg=key)
                    self.assertEqual(env.end_reward_rate(transition), value)
                    # The episode goes on as in a twin that never read it.
                    following = env.step_dt(decisions[1], env.sample_duration())
                    expected = twin.step_dt(decisions[1], twin.sample_duration())
                    np.testing.assert_array_equal(following["next_state"], expected["next_state"])
                    self.assertEqual(following["reward"], expected["reward"])
                    self.assertEqual(following["duration"], expected["duration"])


if __name__ == "__main__":
    unittest.main()
