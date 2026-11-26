"""acrobot_swingup's rewards: Xin and Kaneda's Lyapunov reward and Zhang et al. (2023)'s.

The reward of the table row acrobot-swingup (environment/ph_sac/dmc/acrobot.py) is,
since the second change of 2026-11-08, the log-reciprocal of Xin and Kaneda's
energy-and-elbow Lyapunov function (reward kind "lyapunov"): the row's constants
against the plant's chain and Xin and Kaneda's bound on k_P, values at known states and
against a closed-form energy, numpy against tensor (and the registry), the gradient, and
the key and value checks. Zhang, Sathuluri and Zimmermann (2023), arXiv:2312.11311,
Eq. 4 (reward kind "zhang2023"), the row's reward before, stays supported and tested on
an explicit config with the rho and S the row carried, with the LQR terms
(environment/ph_sac/dmc/acrobot_lqr.py): the paper's constants, values at known states,
numpy against tensor, torch.func, the key checks, the upright linearization and the
Riccati solution S, the region-of-attraction level rho reproduced from the generator's
function, and the saturated LQR converging from the level set in the MuJoCo
environment. The checks that compare the plant, the reset and the trajectories with
dm_control are in tests/test_ph_dmc_pendulum_acrobot.py.
"""

import math
import unittest

import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from environment.ph_sac import tasks
from environment.ph_sac.dmc import acrobot as dmc_acrobot
from environment.ph_sac.dmc import acrobot_lqr
from environment.ph_sac.dmc.chain import (
    chain_drift_numpy,
    chain_energy_numpy,
    chain_potential_energy_numpy,
)
from environment.ph_sac.registry import testbed_spec

# Zhang et al. (2023), Table I, acrobot column, and the port's scale.
PAPER_CONSTANTS = {
    "q_angle": 10.0,
    "q_velocity": 0.2,
    "r_torque": 1e-4,
    "line_bonus": 500.0,
    "line_fraction": 0.8,
    "lqr_bonus": 1e4,
    "velocity_penalty": 1e4,
    "velocity_threshold": 8.0,
    "scale": 1e-4,
}
# The rho and S (upper triangle, row by row) that the acrobot-swingup row carried while
# its reward was Zhang et al.'s, as benchmarks/make_ct_ph_sac_table.py wrote them from
# acrobot_lqr.acrobot_lqr_reward_terms (recorded in common/ph_dmc/acrobot.py too).
ZHANG_LQR_TERMS = {
    "rho": 0.1468853616463693,
    "s11": 2460.646551158671,
    "s12": 682.4822519369063,
    "s13": 907.6439942361637,
    "s14": 282.0701102754566,
    "s22": 190.36706529392856,
    "s23": 251.85979713893715,
    "s24": 78.34549651978381,
    "s33": 334.8981452316153,
    "s34": 104.08074616907096,
    "s44": 32.36377880183468,
}
# dm_control's acrobot: g, link masses, lengths and centres of mass (capsules fromto 0..1).
GRAVITY, MASS, LENGTH, COM = 9.81, 1.0, 1.0, 0.5
# Acrobot-XK's floor; the row's reward_floor.
REWARD_FLOOR = 1e-6


def table_config(name="acrobot_swingup"):
    entry = table.find_ph_entry(name.replace("acrobot_", "acrobot-"), "top")
    return testbed_spec(name).config_factory(**entry.config["testbed_settings"])


def zhang_config():
    """acrobot_swingup with Zhang et al.'s reward: the row's other settings, the paper's
    constants and the recorded rho and S."""
    settings = dict(
        table.find_ph_entry("acrobot-swingup", "top").config["testbed_settings"],
        reward_parameters={**PAPER_CONSTANTS, **ZHANG_LQR_TERMS},
    )
    return testbed_spec("acrobot_swingup").config_factory(**settings)


def row_cost_to_go(config):
    return acrobot_lqr.cost_to_go_matrix(config["reward_parameters"])


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def grouped_constants(chain):
    """Xin and Kaneda's (a1, a2, a3, b1, b2) from the chain's link parameters."""
    (m1, m2), (c1, c2) = chain.link_masses, chain.link_com_distances
    (i1, i2), l1, g = chain.link_inertias, chain.link_lengths[0], chain.gravity
    a1 = i1 + m1 * c1**2 + m2 * l1**2
    a2 = i2 + m2 * c2**2
    a3 = m2 * l1 * c2
    return a1, a2, a3, g * (m1 * c1 + m2 * l1), g * m2 * c2


def closed_form_energy(chain, state):
    """The two-link energy in Xin and Kaneda's grouped form, angles from upright."""
    a1, a2, a3, b1, b2 = grouped_constants(chain)
    shoulder, elbow, shoulder_rate, elbow_rate = (float(x) for x in state)
    cosine = math.cos(elbow)
    kinetic = 0.5 * (
        (a1 + a2 + 2.0 * a3 * cosine) * shoulder_rate**2
        + 2.0 * (a2 + a3 * cosine) * shoulder_rate * elbow_rate
        + a2 * elbow_rate**2
    )
    return kinetic + b1 * math.cos(shoulder) + b2 * math.cos(shoulder + elbow)


def lyapunov_reference(chain, reward, state):
    """-log(max(V / V_down, floor)) from the closed-form energy and the wrapped elbow."""
    energy_error = closed_form_energy(chain, state) - reward["upright_energy"]
    lyapunov = 0.5 * energy_error**2 + 0.5 * reward["lyapunov_k_p"] * wrap(float(state[1])) ** 2
    return -math.log(max(lyapunov / (0.5 * reward["energy_scale"] ** 2), reward["reward_floor"]))


class TestLyapunovRow(unittest.TestCase):
    def test_row_carries_the_lyapunov_constants(self):
        config = table_config()
        reward = config["reward_parameters"]
        self.assertEqual(tuple(reward), dmc_acrobot.LYAPUNOV_REWARD_KEYS)
        self.assertEqual(
            dmc_acrobot.LYAPUNOV_REWARD_KEYS,
            ("energy_scale", "upright_energy", "lyapunov_k_p", "reward_floor"),
        )
        self.assertTrue(all(isinstance(value, float) for value in reward.values()))
        self.assertEqual(reward["reward_floor"], REWARD_FLOOR)
        self.assertEqual(dmc_acrobot.acrobot_reward_kind(reward), "lyapunov")
        self.assertEqual(tasks.make_env(config).reward_kind, "lyapunov")
        self.assertEqual(config["failure_limits"], {})
        # Unbounded below (logarithmically): no known range, a linear reward head.
        self.assertIsNone(testbed_spec("acrobot_swingup").reward_rate_bounds)

    def test_no_row_carries_zhang_keys(self):
        """Zhang et al.'s reward stays a reward kind of the port, but no table row uses it."""
        for entry in table.load_ph_table():
            reward = entry.config["testbed_settings"]["reward_parameters"]
            with self.subTest(row=(entry.env_id, entry.mode)):
                self.assertFalse(set(dmc_acrobot.ZHANG_REWARD_KEYS) & set(reward))

    def test_constants_match_the_chain(self):
        """upright_energy = E(upright rest) = b1 + b2, energy_scale = E(up) - E(down) = 2 (b1 + b2)."""
        config = table_config()
        reward = config["reward_parameters"]
        chain = dmc_acrobot.acrobot_chain(config)
        self.assertEqual(chain.link_masses, (MASS, MASS))
        self.assertEqual(chain.link_lengths, (LENGTH, LENGTH))
        np.testing.assert_allclose(chain.link_com_distances, (COM, COM), rtol=0.0, atol=1e-12)
        self.assertEqual(chain.gravity, GRAVITY)
        rest = np.zeros(2)
        upright = float(chain_energy_numpy(chain, rest, rest)[0])
        hanging = float(chain_energy_numpy(chain, np.array([math.pi, 0.0]), rest)[0])
        b1, b2 = GRAVITY * (MASS * COM + MASS * LENGTH), GRAVITY * MASS * COM
        self.assertAlmostEqual(b1, 14.715, places=12)
        self.assertAlmostEqual(b2, 4.905, places=12)
        self.assertAlmostEqual(reward["upright_energy"], upright, places=12)
        self.assertAlmostEqual(reward["upright_energy"], b1 + b2, places=12)
        self.assertAlmostEqual(reward["upright_energy"], 19.62, places=12)
        self.assertAlmostEqual(reward["energy_scale"], upright - hanging, places=12)
        self.assertAlmostEqual(reward["energy_scale"], 2.0 * (b1 + b2), places=12)
        self.assertAlmostEqual(reward["energy_scale"], 39.24, places=12)
        # The function the generator ran gives the row's numbers.
        terms = dmc_acrobot.acrobot_lyapunov_reward_terms(config)
        self.assertEqual(tuple(terms), dmc_acrobot.LYAPUNOV_REWARD_KEYS[:3])
        for key, value in terms.items():
            self.assertAlmostEqual(value / reward[key], 1.0, places=12, msg=key)

    def test_potential_is_zero_with_both_links_horizontal(self):
        """chain_potential_energy is g sum_j h_j cos(phi_j): zero horizontal, +-(b1 + b2) up/down."""
        chain = dmc_acrobot.acrobot_chain(table_config())
        for positions in ((0.5 * math.pi, 0.0), (-0.5 * math.pi, 0.0), (0.5 * math.pi, math.pi)):
            with self.subTest(positions=positions):
                potential = float(chain_potential_energy_numpy(chain, np.array(positions))[0])
                self.assertAlmostEqual(potential, 0.0, places=12)
        b1, b2 = (chain.gravity * chain.first_moments).tolist()
        for positions, expected in (((0.0, 0.0), b1 + b2), ((math.pi, 0.0), -(b1 + b2)),
                                    ((0.0, math.pi), b1 - b2), ((math.pi, math.pi), b2 - b1)):
            with self.subTest(positions=positions):
                potential = float(chain_potential_energy_numpy(chain, np.array(positions))[0])
                self.assertAlmostEqual(potential, expected, places=12)

    def test_k_p_is_just_above_xin_kanedas_bound(self):
        """lyapunov_k_p = 1.001 (2/pi) min(b1^2, b2^2) = 1.001 (2/pi) b2^2, as Acrobot-XK's ratio."""
        config = table_config()
        chain = dmc_acrobot.acrobot_chain(config)
        bound = dmc_acrobot.lyapunov_k_p_bound(chain)
        b2 = GRAVITY * MASS * COM
        self.assertAlmostEqual(bound, 2.0 / math.pi * b2**2, places=12)
        self.assertAlmostEqual(bound, 15.316451018885951, places=10)
        k_p = config["reward_parameters"]["lyapunov_k_p"]
        self.assertEqual(dmc_acrobot.LYAPUNOV_K_P_FACTOR, 1.001)
        self.assertAlmostEqual(k_p / bound, 1.001, places=12)
        self.assertGreater(k_p, bound)
        # Acrobot-XK's 61.2 against its bound (2/pi) min(14.7^2, 9.8^2) = 61.141.
        xk_bound = 2.0 / math.pi * min((1.5 * 9.8) ** 2, 9.8**2)
        self.assertAlmostEqual(xk_bound, 61.141, places=3)
        self.assertAlmostEqual(61.2 / xk_bound, 1.001, places=3)

    def test_settings_are_checked(self):
        config = table_config()
        reward = config["reward_parameters"]
        for key in dmc_acrobot.LYAPUNOV_REWARD_KEYS:
            with self.subTest(dropped=key):
                broken = {k: v for k, v in reward.items() if k != key}
                with self.assertRaisesRegex(ValueError, "missing required keys"):
                    tasks.make_env(dict(config, reward_parameters=broken))
        for key in ("energy_scale", "upright_energy", "lyapunov_k_p"):
            for value in (0.0, -1.0, math.inf, math.nan):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, f"{key} must be positive and finite"):
                        tasks.make_env(dict(config, reward_parameters=dict(reward, **{key: value})))
        for value in (0.0, -1e-6, 1.0, math.nan):
            with self.subTest(reward_floor=value):
                with self.assertRaisesRegex(ValueError, "reward_floor must lie in"):
                    tasks.make_env(dict(config, reward_parameters=dict(reward, reward_floor=value)))
        with self.assertRaisesRegex(ValueError, "mix"):
            tasks.make_env(dict(config, reward_parameters=dict(reward, margin=0.0)))
        with self.assertRaisesRegex(ValueError, "mix"):
            tasks.make_env(dict(config, reward_parameters=dict(reward, rho=0.1)))


class TestLyapunovValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = table_config()
        cls.reward = cls.config["reward_parameters"]
        cls.env = tasks.make_env(cls.config, seed=0)
        cls.chain = dmc_acrobot.acrobot_chain(cls.config)
        cls.down = 0.5 * cls.reward["energy_scale"] ** 2

    def rate(self, state, torque=0.0):
        return self.env.reward_rate(np.asarray(state, dtype=np.float64), np.array([torque]))

    def test_upright_at_rest_is_the_largest_value(self):
        """V = 0: the floor binds, r = -log(reward_floor) = 13.8155."""
        peak = -math.log(REWARD_FLOOR)
        self.assertAlmostEqual(peak, 13.815510557964274, places=12)
        self.assertEqual(self.rate([0.0, 0.0, 0.0, 0.0]), peak)
        # Whole turns change nothing (the energy is periodic, the elbow wrapped).
        self.assertEqual(self.rate([2.0 * np.pi, -2.0 * np.pi, 0.0, 0.0]), peak)
        # The floor binds wherever V <= reward_floor V_down, a small neighbourhood.
        self.assertEqual(self.rate([1e-4, -1e-4, 1e-4, 0.0]), peak)
        rng = np.random.default_rng(3)
        states = np.hstack(
            (rng.uniform(-np.pi, np.pi, (500, 2)), rng.uniform(-20.0, 20.0, (500, 2)))
        )
        rates = [self.rate(state) for state in states]
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertLessEqual(max(rates), peak)

    def test_hanging_at_rest_is_zero(self):
        """V = (E_down - E_up)^2 / 2 = V_down: r = -log 1 = 0."""
        for state in ([math.pi, 0.0, 0.0, 0.0], [-math.pi, 0.0, 0.0, 0.0], [3.0 * math.pi, 2.0 * math.pi, 0.0, 0.0]):
            with self.subTest(state=state):
                self.assertAlmostEqual(self.rate(state), 0.0, places=12)

    def test_matches_the_closed_form(self):
        """r on random states (whole turns included) against the grouped-form energy."""
        rng = np.random.default_rng(4)
        states = np.hstack(
            (rng.uniform(-np.pi, np.pi, (300, 2)), rng.uniform(-12.0, 12.0, (300, 2)))
        )
        states[:, :2] += 2.0 * np.pi * rng.integers(-1, 2, (300, 2))
        for state in states:
            energy = float(chain_energy_numpy(self.chain, state[:2], state[2:])[0])
            self.assertAlmostEqual(energy / closed_form_energy(self.chain, state), 1.0, places=12)
            self.assertAlmostEqual(
                self.rate(state), lyapunov_reference(self.chain, self.reward, state), places=10
            )
            self.assertAlmostEqual(
                dmc_acrobot.acrobot_lyapunov_reward_rate(self.config, state), self.rate(state),
                places=12,
            )

    def test_elbow_term(self):
        """Upright shoulder at rest, elbow bent by x: V = (b2 (cos x - 1))^2 / 2 + k_P x^2 / 2."""
        _, _, _, _, b2 = grouped_constants(self.chain)
        k_p = self.reward["lyapunov_k_p"]
        for elbow in (0.5, -1.0, 2.5):
            with self.subTest(elbow=elbow):
                energy_only = 0.5 * (b2 * (math.cos(elbow) - 1.0)) ** 2
                expected = -math.log((energy_only + 0.5 * k_p * elbow**2) / self.down)
                self.assertAlmostEqual(self.rate([0.0, elbow, 0.0, 0.0]), expected, places=12)
                self.assertLess(self.rate([0.0, elbow, 0.0, 0.0]), -math.log(energy_only / self.down))
                # The elbow is wrapped (Acrobot-XK's is not): a whole turn changes nothing.
                self.assertAlmostEqual(
                    self.rate([0.0, elbow + 2.0 * math.pi, 0.0, 0.0]), expected, places=12
                )
        # Continuous across the folded elbow theta_2 = +-pi, where the wrap jumps.
        self.assertAlmostEqual(
            self.rate([0.3, math.pi - 1e-9, 1.0, 0.0]), self.rate([0.3, -math.pi + 1e-9, 1.0, 0.0]),
            places=6,
        )

    def test_energy_term(self):
        """The elbow straight: V = (E - E_up)^2 / 2 alone; pumping energy from hanging pays."""
        for state in ([0.5 * math.pi, 0.0, 0.0, 0.0], [math.pi, 0.0, 3.0, 0.0], [1.0, 0.0, -2.0, 1.0]):
            with self.subTest(state=state):
                energy = closed_form_energy(self.chain, state)
                expected = -math.log(0.5 * (energy - self.reward["upright_energy"]) ** 2 / self.down)
                self.assertAlmostEqual(self.rate(state), expected, places=10)
        # Both links horizontal at rest: E = 0, V = V_down / 4, r = log 4.
        self.assertAlmostEqual(self.rate([0.5 * math.pi, 0.0, 0.0, 0.0]), math.log(4.0), places=12)
        self.assertGreater(self.rate([math.pi, 0.0, 3.0, 0.0]), self.rate([math.pi, 0.0, 0.0, 0.0]))
        # Far above the upright energy the reward is negative, but only logarithmically.
        self.assertLess(self.rate([math.pi, 0.0, 20.0, 0.0]), 0.0)
        self.assertGreater(self.rate([math.pi, 0.0, 20.0, 0.0]), -10.0)

    def test_state_only(self):
        for state in ([math.pi, 0.0, 0.0, 0.0], [0.4, -1.2, 3.0, -5.0]):
            for torque in (-2.0, 0.5, 2.0):
                with self.subTest(state=state, torque=torque):
                    self.assertEqual(self.rate(state, torque), self.rate(state))


class TestLyapunovTensor(unittest.TestCase):
    @staticmethod
    def states(count, rng):
        """Uniform angles (some whole turns) and rates up to 10 rad/s, near upright and hanging.

        The upright perturbations span 1e-4 to 1e-1, so the floor binds on some of them.
        """
        uniform = np.hstack(
            (rng.uniform(-np.pi, np.pi, (count, 2)), rng.uniform(-10.0, 10.0, (count, 2)))
        )
        uniform[:, :2] += 2.0 * np.pi * rng.integers(-1, 2, (count, 2))
        scales = 10.0 ** rng.uniform(-4.0, -1.0, (count, 1))
        upright = scales * rng.standard_normal((count, 4))
        hanging = np.array([math.pi, 0.0, 0.0, 0.0]) + 0.1 * rng.standard_normal((count, 4))
        return np.vstack((uniform, upright, hanging))

    def test_tensor_matches_numpy_and_the_registry(self):
        config = table_config()
        reward = config["reward_parameters"]
        env = tasks.make_env(config, seed=0)
        rng = np.random.default_rng(12)
        states = self.states(200, rng)
        torques = rng.uniform(-2.0, 2.0, (len(states), 1))
        states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
        tensor = tasks.reward_rate_tensor(states_t, torques_t, config)
        self.assertEqual(tensor.shape, (len(states), 1))
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertTrue(
            torch.equal(tensor, testbed_spec("acrobot_swingup").reward_rate_tensor(states_t, torques_t, config))
        )
        self.assertTrue(torch.equal(tensor, dmc_acrobot.acrobot_reward_rate_tensor(states_t, torques_t, config)))
        # State-only: other torques give the same tensor.
        self.assertTrue(torch.equal(tensor, tasks.reward_rate_tensor(states_t, torch.zeros_like(torques_t), config)))
        peak = -math.log(reward["reward_floor"])
        capped = 0
        for state, torque, value in zip(states, torques, tensor[:, 0].tolist()):
            self.assertAlmostEqual(value, env.reward_rate(state, torque), places=10)
            capped += abs(value - peak) < 1e-12
        self.assertGreater(capped, 20)
        self.assertLess(capped, 200)
        self.assertLessEqual(float(tensor.max()), peak + 1e-12)
        float32 = tasks.reward_rate_tensor(states_t.float(), torques_t.float(), config)
        self.assertEqual(float32.dtype, torch.float32)
        torch.testing.assert_close(float32.double(), tensor, rtol=1e-4, atol=1e-4)

    def test_gradient_matches_finite_differences(self):
        """vmap(grad) of the tensor rate against central differences of env.reward_rate."""
        config = table_config()
        env = tasks.make_env(config, seed=0)
        rng = np.random.default_rng(13)
        states = np.hstack((rng.uniform(-2.5, 2.5, (40, 2)), rng.uniform(-6.0, 6.0, (40, 2))))
        torques = rng.uniform(-2.0, 2.0, (40, 1))

        def single(state, torque):
            return tasks.reward_rate_tensor(state[None], torque[None], config)[0, 0]

        states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
        by_state = torch.func.vmap(torch.func.grad(single, argnums=0))(states_t, torques_t)
        by_torque = torch.func.vmap(torch.func.grad(single, argnums=1))(states_t, torques_t)
        self.assertTrue(torch.equal(by_torque, torch.zeros_like(by_torque)))
        epsilon = 1e-6
        for state, gradient in zip(states, by_state.numpy()):
            self.assertLess(env.reward_rate(state, np.zeros(1)), -math.log(REWARD_FLOOR))
            column = np.empty(4)
            for i in range(4):
                bump = np.zeros(4)
                bump[i] = epsilon
                column[i] = (
                    env.reward_rate(state + bump, np.zeros(1)) - env.reward_rate(state - bump, np.zeros(1))
                ) / (2.0 * epsilon)
            np.testing.assert_allclose(gradient, column, rtol=1e-5, atol=1e-6)
        # The graph exists: autograd reaches the states through the log, the energy and the wrap.
        leaf = states_t.clone().requires_grad_(True)
        rate = tasks.reward_rate_tensor(leaf, torques_t, config)
        self.assertTrue(rate.requires_grad)
        (gradient,) = torch.autograd.grad(rate.sum(), leaf)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(bool((gradient.abs().sum(dim=-1) > 0.0).all()))
        torch.testing.assert_close(gradient, by_state)


class TestRewardKind(unittest.TestCase):
    def test_kinds_of_the_key_sets(self):
        cases = (
            (dict.fromkeys(dmc_acrobot.LYAPUNOV_REWARD_KEYS, 1.0), "lyapunov"),
            (dict.fromkeys(dmc_acrobot.ZHANG_REWARD_KEYS, 1.0), "zhang2023"),
            (dict.fromkeys(dmc_acrobot.TOLERANCE_REWARD_KEYS, 1.0), "tolerance"),
            ({"reward_floor": 1e-6}, "lyapunov"),
            ({"rho": 0.1}, "zhang2023"),
            ({"margin": 0.0}, "tolerance"),
            ({}, "tolerance"),
        )
        for reward, kind in cases:
            with self.subTest(keys=tuple(reward)):
                self.assertEqual(dmc_acrobot.acrobot_reward_kind(reward), kind)
        self.assertEqual(tasks.make_env(table_config()).reward_kind, "lyapunov")
        self.assertEqual(tasks.make_env(table_config("acrobot_swingup_sparse")).reward_kind, "tolerance")
        self.assertEqual(tasks.make_env(zhang_config()).reward_kind, "zhang2023")

    def test_mixed_keys_are_rejected(self):
        sets = {
            "lyapunov": dmc_acrobot.LYAPUNOV_REWARD_KEYS,
            "zhang2023": dmc_acrobot.ZHANG_REWARD_KEYS,
            "tolerance": dmc_acrobot.TOLERANCE_REWARD_KEYS,
        }
        config = table_config()
        states = torch.zeros(2, 4, dtype=torch.float64)
        for first, second in (("lyapunov", "zhang2023"), ("lyapunov", "tolerance"), ("zhang2023", "tolerance")):
            reward = {**dict.fromkeys(sets[first], 1.0), sets[second][-1]: 1.0}
            with self.subTest(kinds=(first, second)):
                with self.assertRaisesRegex(ValueError, f"mix the keys of several rewards: {first}"):
                    dmc_acrobot.acrobot_reward_kind(reward)
                with self.assertRaisesRegex(ValueError, "mix"):
                    tasks.make_env(dict(config, reward_parameters=reward))
                with self.assertRaisesRegex(ValueError, "mix"):
                    tasks.reward_rate_tensor(states, torch.zeros(2, 1), dict(config, reward_parameters=reward))


class TestZhangSettings(unittest.TestCase):
    def test_config_carries_the_paper_constants_and_the_lqr_terms(self):
        config = zhang_config()
        reward = config["reward_parameters"]
        self.assertEqual(tuple(reward), dmc_acrobot.ZHANG_REWARD_KEYS)
        self.assertEqual(
            dmc_acrobot.ZHANG_REWARD_KEYS[-11:], ("rho",) + acrobot_lqr.COST_TO_GO_KEYS
        )
        for key, value in PAPER_CONSTANTS.items():
            self.assertEqual(reward[key], value, key)
        self.assertTrue(all(isinstance(value, float) for value in reward.values()))
        self.assertGreater(reward["rho"], 0.0)
        self.assertEqual(dmc_acrobot.acrobot_reward_kind(reward), "zhang2023")
        self.assertEqual(tasks.make_env(config).reward_kind, "zhang2023")
        # The reward is unbounded (the velocity terms): no known range for the reward head.
        self.assertIsNone(testbed_spec("acrobot_swingup").reward_rate_bounds)

    def test_cost_to_go_is_symmetric_positive_definite(self):
        cost_to_go = row_cost_to_go(zhang_config())
        np.testing.assert_array_equal(cost_to_go, cost_to_go.T)
        self.assertGreater(np.min(np.linalg.eigvalsh(cost_to_go)), 0.0)
        entries = acrobot_lqr.cost_to_go_entries(cost_to_go)
        self.assertEqual(tuple(entries), acrobot_lqr.COST_TO_GO_KEYS)
        np.testing.assert_array_equal(acrobot_lqr.cost_to_go_matrix(entries), cost_to_go)

    def test_settings_are_checked(self):
        config = zhang_config()
        reward = config["reward_parameters"]
        for key in ("q_angle", "rho", "s44"):
            with self.subTest(dropped=key):
                broken = {k: v for k, v in reward.items() if k != key}
                with self.assertRaisesRegex(ValueError, "missing required keys"):
                    tasks.make_env(dict(config, reward_parameters=broken))
        with self.assertRaisesRegex(ValueError, "mix"):
            tasks.make_env(dict(config, reward_parameters=dict(reward, margin=0.0)))
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            tasks.make_env(dict(config, reward_parameters=dict(reward, s11=-1.0)))
        with self.assertRaisesRegex(ValueError, "rho must be positive"):
            tasks.make_env(dict(config, reward_parameters=dict(reward, rho=0.0)))


class TestZhangValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = zhang_config()
        cls.reward = cls.config["reward_parameters"]
        cls.env = tasks.make_env(cls.config, seed=0)
        cls.cost_to_go = row_cost_to_go(cls.config)

    def rate(self, state, torque=0.0):
        return self.env.reward_rate(np.asarray(state, dtype=np.float64), np.array([torque]))

    def test_upright_at_rest(self):
        """Both bonuses, no cost: scale (line_bonus + lqr_bonus) = 1.05."""
        expected = self.reward["scale"] * (self.reward["line_bonus"] + self.reward["lqr_bonus"])
        self.assertAlmostEqual(expected, 1.05, places=12)
        self.assertAlmostEqual(self.rate([0.0, 0.0, 0.0, 0.0]), expected, places=12)
        # The angle differences are wrapped: whole turns change nothing.
        self.assertAlmostEqual(self.rate([2.0 * np.pi, -2.0 * np.pi, 0.0, 0.0]), expected, places=12)

    def test_hanging_at_rest(self):
        """theta_1 = pi, the elbow straight: h = -2 < 1.6, outside the LQR set, no rates."""
        expected = -self.reward["scale"] * self.reward["q_angle"] * math.pi**2
        self.assertAlmostEqual(self.rate([math.pi, 0.0, 0.0, 0.0]), expected, places=12)
        self.assertAlmostEqual(self.rate([-math.pi, 0.0, 0.0, 0.0]), expected, places=12)

    def test_torque_cost(self):
        scale, r = self.reward["scale"], self.reward["r_torque"]
        hanging = self.rate([math.pi, 0.0, 0.0, 0.0])
        for torque in (-2.0, 0.5, 2.0):
            with self.subTest(torque=torque):
                self.assertAlmostEqual(
                    self.rate([math.pi, 0.0, 0.0, 0.0], torque), hanging - scale * r * torque**2,
                    places=15,
                )

    def test_velocity_penalty(self):
        """Each rate of magnitude >= 8 rad/s adds -scale velocity_penalty."""
        scale = self.reward["scale"]
        penalty = scale * self.reward["velocity_penalty"]
        quadratic = scale * self.reward["q_velocity"]
        base = self.rate([math.pi, 0.0, 0.0, 0.0])
        for rates, count in (
            ((8.0, 0.0), 1),
            ((-8.0, 0.0), 1),
            ((0.0, 9.0), 1),
            ((8.5, -8.5), 2),
            ((7.99, 0.0), 0),
            ((0.0, -7.99), 0),
        ):
            with self.subTest(rates=rates):
                expected = base - quadratic * (rates[0] ** 2 + rates[1] ** 2) - count * penalty
                self.assertAlmostEqual(self.rate([math.pi, 0.0, *rates]), expected, places=12)

    def test_line_bonus(self):
        """The tip height cos theta_1 + cos(theta_1 + theta_2) >= 0.8 x 2 m, outside the LQR set."""
        scale, q = self.reward["scale"], self.reward["q_angle"]
        line = scale * self.reward["line_bonus"]
        for shoulder, elbow, on_line in (
            (0.3, 0.0, True),  # h = 1.911.
            (0.2, -0.5, True),  # h = 1.935.
            (0.7, 0.0, False),  # h = 1.530.
            (0.0, 1.0, False),  # h = 1.540.
            (0.0, math.pi, False),  # Folded: h = 0.
        ):
            with self.subTest(shoulder=shoulder, elbow=elbow):
                height = math.cos(shoulder) + math.cos(shoulder + elbow)
                self.assertEqual(height >= 1.6, on_line)
                error = np.array([shoulder, elbow, 0.0, 0.0])
                self.assertGreater(error @ self.cost_to_go @ error, self.reward["rho"])
                expected = -scale * q * (shoulder**2 + elbow**2) + line * on_line
                self.assertAlmostEqual(self.rate(error), expected, places=12)

    def test_lqr_bonus(self):
        """scale lqr_bonus inside e^T S e <= rho (with the line bonus near upright), not outside."""
        scale = self.reward["scale"]
        rho = self.reward["rho"]
        directions = acrobot_lqr.level_set_directions(self.cost_to_go, 50, 3)[:50]
        inside = outside = 0
        for direction in directions:
            for fraction in (0.5, 0.99, 1.01, 2.0):
                error = math.sqrt(fraction * rho) * direction
                with self.subTest(fraction=fraction):
                    height = math.cos(error[0]) + math.cos(error[0] + error[1])
                    smooth = -scale * (
                        self.reward["q_angle"] * (error[0] ** 2 + error[1] ** 2)
                        + self.reward["q_velocity"] * (error[2] ** 2 + error[3] ** 2)
                    )
                    self.assertLess(np.max(np.abs(error[2:])), 8.0)
                    bonus = scale * (
                        self.reward["line_bonus"] * (height >= 1.6)
                        + self.reward["lqr_bonus"] * (fraction < 1.0)
                    )
                    self.assertAlmostEqual(self.rate(error), smooth + bonus, places=12)
                    inside += fraction < 1.0
                    outside += fraction > 1.0
        self.assertEqual((inside, outside), (100, 100))


class TestZhangTensor(unittest.TestCase):
    def states(self, config, count, rng):
        """Uniform angles and rates up to 10 rad/s, and states around the LQR set and the line."""
        cost_to_go = row_cost_to_go(config)
        rho = config["reward_parameters"]["rho"]
        uniform = np.hstack(
            (rng.uniform(-np.pi, np.pi, (count, 2)), rng.uniform(-10.0, 10.0, (count, 2)))
        )
        directions = acrobot_lqr.level_set_directions(cost_to_go, count, int(rng.integers(1 << 30)))
        near = math.sqrt(rho) * directions[:count] * rng.uniform(0.2, 1.8, (count, 1))
        line = np.hstack(
            (rng.uniform(-0.8, 0.8, (count, 2)), rng.uniform(-10.0, 10.0, (count, 2)))
        )
        states = np.vstack((uniform, near, line))
        # Whole turns on some angles: the reward wraps them.
        states[:, :2] += 2.0 * np.pi * rng.integers(-1, 2, (len(states), 2))
        return states

    def test_tensor_matches_numpy_and_the_registry(self):
        config = zhang_config()
        reward = config["reward_parameters"]
        env = tasks.make_env(config, seed=0)
        cost_to_go = row_cost_to_go(config)
        rng = np.random.default_rng(10)
        states = self.states(config, 300, rng)
        torques = rng.uniform(-2.0, 2.0, (len(states), 1))
        states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
        tensor = tasks.reward_rate_tensor(states_t, torques_t, config)
        self.assertEqual(tensor.shape, (len(states), 1))
        self.assertTrue(
            torch.equal(tensor, testbed_spec("acrobot_swingup").reward_rate_tensor(states_t, torques_t, config))
        )
        self.assertTrue(torch.equal(tensor, dmc_acrobot.acrobot_reward_rate_tensor(states_t, torques_t, config)))
        counts = {"line": 0, "lqr": 0, "fast": 0}
        compared = 0
        for state, torque, value in zip(states, torques, tensor[:, 0].tolist()):
            error = acrobot_lqr.wrapped_errors(state)
            height = math.cos(state[0]) + math.cos(state[0] + state[1])
            level = float(error @ cost_to_go @ error)
            if (
                abs(height - 1.6) < 1e-9
                or abs(level - reward["rho"]) < 1e-9
                or np.min(np.abs(np.abs(state[2:]) - 8.0)) < 1e-9
            ):
                continue
            compared += 1
            counts["line"] += height >= 1.6
            counts["lqr"] += level <= reward["rho"]
            counts["fast"] += np.max(np.abs(state[2:])) >= 8.0
            self.assertAlmostEqual(value, env.reward_rate(state, torque), places=10)
            self.assertAlmostEqual(
                value, dmc_acrobot.acrobot_zhang_reward_rate(config, state, torque), places=10
            )
        self.assertGreater(compared, 850)
        for name, count in counts.items():
            self.assertGreater(count, 40, name)

    def test_torch_func_gradient_is_the_quadratic_part(self):
        """vmap(grad) works; the indicators are piecewise constant, so only the costs differentiate."""
        config = zhang_config()
        reward = config["reward_parameters"]
        rng = np.random.default_rng(11)
        states = torch.as_tensor(self.states(config, 40, rng))
        torques = torch.as_tensor(rng.uniform(-2.0, 2.0, (len(states), 1)))

        def single(state, torque):
            return tasks.reward_rate_tensor(state[None], torque[None], config)[0, 0]

        by_state = torch.func.vmap(torch.func.grad(single, argnums=0))(states, torques)
        by_torque = torch.func.vmap(torch.func.grad(single, argnums=1))(states, torques)
        scale = reward["scale"]
        angles = torch.atan2(torch.sin(states[:, :2]), torch.cos(states[:, :2]))
        expected = -2.0 * scale * torch.cat(
            (reward["q_angle"] * angles, reward["q_velocity"] * states[:, 2:]), dim=-1
        )
        torch.testing.assert_close(by_state, expected, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(by_torque, -2.0 * scale * reward["r_torque"] * torques, rtol=1e-12, atol=1e-18)
        # At the upright both bonuses are on and the costs are stationary: zero gradient.
        upright = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        rate = tasks.reward_rate_tensor(upright, torch.zeros(1, 1, dtype=torch.float64), config)
        self.assertAlmostEqual(rate.item(), 1.05, places=12)
        (gradient,) = torch.autograd.grad(rate.sum(), upright)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))


class TestLQR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = zhang_config()
        cls.chain = dmc_acrobot.acrobot_chain(cls.config)
        cls.cost_to_go, cls.gain = acrobot_lqr.upright_lqr(cls.chain)

    def test_linearization_matches_finite_differences(self):
        a, b = acrobot_lqr.upright_linearization(self.chain)
        epsilon = 1e-6
        origin, zero = np.zeros(4), np.zeros(1)
        for i in range(4):
            bump = np.zeros(4)
            bump[i] = epsilon
            column = (
                chain_drift_numpy(self.chain, origin + bump, zero)
                - chain_drift_numpy(self.chain, origin - bump, zero)
            ) / (2.0 * epsilon)
            np.testing.assert_allclose(a[:, i], column, atol=1e-8)
        column = (
            chain_drift_numpy(self.chain, origin, zero + epsilon)
            - chain_drift_numpy(self.chain, origin, zero - epsilon)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(b[:, 0], column, atol=1e-8)
        # The upright is an unstable equilibrium of the open loop.
        np.testing.assert_array_equal(chain_drift_numpy(self.chain, origin, zero), np.zeros(4))
        self.assertGreater(np.max(np.linalg.eigvals(a).real), 0.0)

    def test_riccati_solution(self):
        """S solves the CARE for the paper's Q_lqr, R_lqr; A - B K is Hurwitz; S was recorded."""
        a, b = acrobot_lqr.upright_linearization(self.chain)
        weights = np.diag(acrobot_lqr.LQR_STATE_WEIGHTS)
        control = acrobot_lqr.LQR_TORQUE_WEIGHT
        s, k = self.cost_to_go, self.gain
        self.assertEqual(acrobot_lqr.LQR_STATE_WEIGHTS, (0.97, 0.93, 0.39, 0.26))
        self.assertEqual(control, 0.11)
        residual = a.T @ s + s @ a - s @ b @ b.T @ s / control + weights
        self.assertLess(np.max(np.abs(residual)), 1e-8 * np.max(np.abs(s)))
        np.testing.assert_allclose(k, b.T @ s / control, rtol=1e-12)
        self.assertLess(np.max(np.linalg.eigvals(a - b @ k).real), 0.0)
        np.testing.assert_allclose(row_cost_to_go(self.config), s, rtol=1e-9, atol=0.0)

    def test_saturated_lqr_converges_from_the_level_set_in_the_env(self):
        """u = clip(-K e, -2, 2) every 1 ms in the MuJoCo env: starts on e^T S e = rho converge."""
        reward = self.config["reward_parameters"]
        rho = reward["rho"]
        cost_to_go = row_cost_to_go(self.config)
        env = tasks.make_env(self.config, decision_mode="torque", seed=0)
        starts = math.sqrt(rho) * acrobot_lqr.level_set_directions(cost_to_go, 4, 21)[:4]
        peak = reward["scale"] * (reward["line_bonus"] + reward["lqr_bonus"])
        for start in starts:
            with self.subTest(start=start.tolist()):
                self.assertAlmostEqual(float(start @ cost_to_go @ start), rho, places=12)
                state, _ = env.reset(options={"state": start})
                for _ in range(4000):
                    torque = np.clip(-(acrobot_lqr.wrapped_errors(state) @ self.gain.T), -2.0, 2.0)
                    state = env.step_dt(torque, 0.001)["next_state"]
                error = acrobot_lqr.wrapped_errors(state)
                self.assertLess(float(error @ cost_to_go @ error), 1e-4 * rho)
                self.assertAlmostEqual(env.reward_rate(state, np.zeros(1)), peak, places=4)

    def test_the_level_beyond_rho_fails_somewhere(self):
        """rho is not vacuous: at 3 rho some starts on the level set fall (chain rollouts)."""
        rho = self.config["reward_parameters"]["rho"]
        cost_to_go = row_cost_to_go(self.config)
        starts = math.sqrt(3.0 * rho) * acrobot_lqr.level_set_directions(cost_to_go, 64, 5)[:64]
        converged = acrobot_lqr.saturated_lqr_converges(
            self.chain, cost_to_go, self.gain, starts, torque_limit=2.0,
            threshold=acrobot_lqr.CONVERGENCE_FRACTION * 3.0 * rho,
        )
        self.assertGreater(int((~converged).sum()), 5)
        self.assertGreater(int(converged.sum()), 5)

    def test_rho_and_cost_to_go_reproduce_the_recorded_terms(self):
        """acrobot_lqr_reward_terms gives ZHANG_LQR_TERMS, which the generator wrote into the row.

        About 40 s: the bisection of the region of attraction with its rollouts.
        """
        terms = acrobot_lqr.acrobot_lqr_reward_terms(self.config)
        self.assertEqual(tuple(terms), ("rho",) + acrobot_lqr.COST_TO_GO_KEYS)
        self.assertEqual(tuple(terms), tuple(ZHANG_LQR_TERMS))
        for key, value in terms.items():
            self.assertAlmostEqual(value / ZHANG_LQR_TERMS[key], 1.0, places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
