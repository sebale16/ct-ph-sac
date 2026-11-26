"""The optional Lyapunov ("orbit") reward variants and their evaluation diagnostics.

Three variants of the negative-log Lyapunov reward of acrobot-swingup, each an extra row
of benchmarks/hyperparams/ct_ph_sac.csv (make_ct_ph_sac_table.VARIANTS) beside an
unchanged top row:
  A  cart n-pole, energy and cart (cartpole-two_poles / -three_poles, mode lyapunov_cart):
     V = (E - E_up)^2 / 2 + k_x x^2 / 2;
  B  A plus the straightness of the poles (mode lyapunov_cart_straight):
     + k_theta sum_{k>=2} wrap(theta_k)^2 / 2;
  C  acrobot, the Lyapunov reward plus the unactuated shoulder (acrobot-swingup, mode
     lyapunov_shoulder): V_C = V + k_1 wrap(theta_1)^2 / 2, normalized by V_C at hanging rest;
each r = -log(max(V / V_down, reward_floor)). Checked: 0 at hanging rest, -log(floor) at
upright rest, numpy against tensor (and the registry), finite gradients against finite
differences, the zero sets (A: any state with E = E_up and x = 0, B: only straight chains
among those, C: only the upright at rest where the top row's reward is maximal on its whole
orbit set), the scale rule of the weights, the key and value checks, the linear reward head,
the conservation of the chain energy E along unforced MuJoCo trajectories of the cart
n-pole (and E against MuJoCo's own kinetic and potential energy), the default rows
unchanged, and the evaluation's orbit diagnostics (V / V_down mean and final, time shares
below 1e-2 and 1e-3).
"""

import copy
import math
import sys
import unittest

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import load_protocol_from_table
from environment.ph_sac import tasks
from environment.ph_sac.dmc import acrobot as dmc_acrobot
from environment.ph_sac.dmc import cartpole as dmc_cartpole
from environment.ph_sac.dmc.chain import chain_energy_numpy, chain_potential_energy_numpy
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac import protocol

FLOOR = 1e-6
PEAK = -math.log(FLOOR)
GRAVITY = 9.81
CART_RANGE = 1.8
# env_id: (testbed, poles)
CARTPOLES = {"cartpole-two_poles": ("cartpole_two_poles", 2), "cartpole-three_poles": ("cartpole_three_poles", 3)}
CART_MODES = ("lyapunov_cart", "lyapunov_cart_straight")
VARIANT_ROWS = tuple((env_id, mode) for env_id in CARTPOLES for mode in CART_MODES) + (
    ("acrobot-swingup", "lyapunov_shoulder"),
)
NEW_KEYS = {"lyapunov_k_cart", "lyapunov_k_straight", "lyapunov_k_shoulder"}


def rows():
    return {(entry.env_id, entry.mode): entry for entry in table.load_ph_table()}


ROWS = rows()


def row_config(env_id, mode, **plant):
    """The factory's config of a table row, with plant_parameters overridden."""
    entry = ROWS[(env_id, mode)]
    settings = copy.deepcopy(entry.config["testbed_settings"])
    settings["plant_parameters"].update(plant)
    return testbed_spec(entry.config["testbed"]).config_factory(**settings)


def hanging(poles):
    state = np.zeros(2 * (1 + poles))
    state[1] = math.pi
    return state


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def at_energy(chain, positions, direction, energy):
    """(positions, velocities) with velocities along direction scaled to the given energy."""
    positions = np.asarray(positions, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    potential = float(chain_potential_energy_numpy(chain, positions)[0])
    kinetic = float(chain_energy_numpy(chain, positions, direction)[0]) - potential
    assert energy > potential and kinetic > 0.0
    return np.concatenate((positions, math.sqrt((energy - potential) / kinetic) * direction))


def cart_states(poles, rng, count):
    """Random cart n-pole states away from the floor and the wrap kinks, and near upright."""
    random = np.concatenate(
        (
            rng.uniform(-1.5, 1.5, (count, 1)),
            rng.uniform(-2.5, 2.5, (count, poles)),
            rng.uniform(-4.0, 4.0, (count, 1)),
            rng.uniform(-8.0, 8.0, (count, poles)),
        ),
        axis=1,
    )
    near = 10.0 ** rng.uniform(-5.0, -2.0, (count, 1)) * rng.standard_normal((count, 2 * (1 + poles)))
    return random, near


class TestRowsAndDefaults(unittest.TestCase):
    def test_default_rows_keep_their_rewards(self):
        """The top rows carry the rewards they had; no top row carries a new key."""
        smooth = {"cart_margin": 2.0, "velocity_margin": 5.0, "failure_rate": 0.0}
        for env_id, (name, poles) in CARTPOLES.items():
            with self.subTest(env_id=env_id):
                entry = ROWS[(env_id, "top")]
                self.assertEqual(entry.config["testbed_settings"]["reward_parameters"], smooth)
                self.assertEqual(entry.config["protocol"]["output_directory"], "out/ph_sac/" + env_id.replace("-", "_"))
                config = row_config(env_id, "top")
                self.assertEqual(dmc_cartpole.cartpole_reward_law(config), "smooth")
                self.assertEqual(testbed_spec(name).reward_rate_bounds(config), (0.0, 1.0))
                env = tasks.make_env(config, seed=0)
                self.assertEqual(env.reward_law, "smooth")
                self.assertIsNone(env.lyapunov_ratio(hanging(poles)))
        acrobot = ROWS[("acrobot-swingup", "top")]
        reward = acrobot.config["testbed_settings"]["reward_parameters"]
        self.assertEqual(tuple(reward), dmc_acrobot.LYAPUNOV_REWARD_KEYS)
        self.assertEqual(dmc_acrobot.acrobot_reward_kind(reward), "lyapunov")
        for key, entry in ROWS.items():
            if key[1] == "top":
                with self.subTest(row=key):
                    reward = entry.config["testbed_settings"].get("reward_parameters") or {}
                    self.assertFalse(NEW_KEYS & set(reward))

    def test_variant_rows_copy_their_top_rows_but_the_reward(self):
        for env_id, mode in VARIANT_ROWS:
            with self.subTest(env_id=env_id, mode=mode):
                variant, top = ROWS[(env_id, mode)], ROWS[(env_id, "top")]
                self.assertEqual(variant.task, top.task)
                left, right = copy.deepcopy(variant.config), copy.deepcopy(top.config)
                self.assertNotEqual(
                    left["testbed_settings"].pop("reward_parameters"),
                    right["testbed_settings"].pop("reward_parameters"),
                )
                self.assertEqual(left["protocol"].pop("output_directory"),
                                 right["protocol"].pop("output_directory") + "_" + mode)
                self.assertEqual(left, right)
                self.assertIn(f"--mode {mode}", variant.comment)
                built = load_protocol_from_table(env_id, mode)
                self.assertEqual(
                    built["matched"]["testbed_config"]["reward_parameters"],
                    variant.config["testbed_settings"]["reward_parameters"],
                )

    def test_kinds_laws_and_reward_heads(self):
        for env_id, (name, poles) in CARTPOLES.items():
            for mode in CART_MODES:
                with self.subTest(env_id=env_id, mode=mode):
                    config = row_config(env_id, mode)
                    reward = config["reward_parameters"]
                    keys = (dmc_cartpole.LYAPUNOV_STRAIGHT_KEYS if mode.endswith("straight")
                            else dmc_cartpole.LYAPUNOV_CART_KEYS)
                    self.assertEqual(tuple(reward), keys)
                    self.assertTrue(all(isinstance(value, float) for value in reward.values()))
                    self.assertEqual(dmc_cartpole.cartpole_reward_law(config), mode)
                    self.assertEqual(tasks.make_env(config).reward_law, mode)
                    # Unbounded below: the learned reward head stays linear.
                    self.assertIsNone(testbed_spec(name).reward_rate_bounds(config))
                    self.assertEqual(reward["reward_floor"], FLOOR)
                    self.assertEqual(reward["failure_rate"], 0.0)
        config = row_config("acrobot-swingup", "lyapunov_shoulder")
        reward = config["reward_parameters"]
        self.assertEqual(tuple(reward), dmc_acrobot.LYAPUNOV_SHOULDER_KEYS)
        self.assertEqual(dmc_acrobot.acrobot_reward_kind(reward), "lyapunov_shoulder")
        self.assertEqual(tasks.make_env(config).reward_kind, "lyapunov_shoulder")
        self.assertIsNone(testbed_spec("acrobot_swingup").reward_rate_bounds)
        # The shoulder key alone still selects the Lyapunov family; mixing families raises.
        self.assertEqual(dmc_acrobot.acrobot_reward_kind({"lyapunov_k_shoulder": 1.0}), "lyapunov_shoulder")
        with self.assertRaisesRegex(ValueError, "mix the keys of several rewards: lyapunov"):
            dmc_acrobot.acrobot_reward_kind({"lyapunov_k_shoulder": 1.0, "margin": 0.0})


class TestScaleRule(unittest.TestCase):
    def test_cart_weights(self):
        """upright_energy = g sum_j h_j, energy_scale twice that; the cart at the rail limit and
        one relative angle of pi each cost V_down."""
        sums = {2: 0.15 + 0.05, 3: 0.25 + 0.15 + 0.05}
        for env_id, (_, poles) in CARTPOLES.items():
            reward = row_config(env_id, "lyapunov_cart_straight")["reward_parameters"]
            with self.subTest(env_id=env_id):
                self.assertAlmostEqual(reward["upright_energy"], GRAVITY * sums[poles], places=12)
                self.assertAlmostEqual(reward["energy_scale"], 2.0 * GRAVITY * sums[poles], places=12)
                down = 0.5 * reward["energy_scale"] ** 2
                self.assertAlmostEqual(0.5 * reward["lyapunov_k_cart"] * CART_RANGE**2 / down, 1.0, places=12)
                self.assertAlmostEqual(0.5 * reward["lyapunov_k_straight"] * math.pi**2 / down, 1.0, places=12)
                plain = row_config(env_id, "lyapunov_cart")["reward_parameters"]
                for key in dmc_cartpole.LYAPUNOV_CART_KEYS:
                    self.assertEqual(plain[key], reward[key], key)
                terms = dmc_cartpole.cartpole_lyapunov_reward_terms(
                    row_config(env_id, "top")["plant_parameters"], poles, straight=True
                )
                for key, value in terms.items():
                    self.assertAlmostEqual(value / reward[key], 1.0, places=12, msg=key)
        # The values the docstrings and the report state.
        two = row_config("cartpole-two_poles", "lyapunov_cart_straight")["reward_parameters"]
        three = row_config("cartpole-three_poles", "lyapunov_cart_straight")["reward_parameters"]
        self.assertAlmostEqual(two["lyapunov_k_cart"], 3.924**2 / 3.24, places=10)
        self.assertAlmostEqual(two["lyapunov_k_straight"], 3.924**2 / math.pi**2, places=10)
        self.assertAlmostEqual(three["lyapunov_k_cart"], 8.829**2 / 3.24, places=10)
        self.assertAlmostEqual(three["lyapunov_k_straight"], 8.829**2 / math.pi**2, places=10)

    def test_shoulder_weight(self):
        """theta_1 = pi alone costs the energy term at hanging rest: k_1 = energy_scale^2 / pi^2."""
        reward = row_config("acrobot-swingup", "lyapunov_shoulder")["reward_parameters"]
        top = row_config("acrobot-swingup", "top")["reward_parameters"]
        for key in dmc_acrobot.LYAPUNOV_REWARD_KEYS:
            self.assertEqual(reward[key], top[key], key)
        self.assertAlmostEqual(
            0.5 * reward["lyapunov_k_shoulder"] * math.pi**2, 0.5 * reward["energy_scale"] ** 2, places=9
        )
        self.assertAlmostEqual(reward["lyapunov_k_shoulder"], 39.24**2 / math.pi**2, places=9)
        terms = dmc_acrobot.acrobot_lyapunov_shoulder_reward_terms(row_config("acrobot-swingup", "top"))
        self.assertEqual(tuple(terms), dmc_acrobot.LYAPUNOV_SHOULDER_KEYS[:4])
        for key, value in terms.items():
            self.assertAlmostEqual(value / reward[key], 1.0, places=12, msg=key)


class TestValues(unittest.TestCase):
    def cart_envs(self):
        for env_id, (_, poles) in CARTPOLES.items():
            for mode in CART_MODES:
                config = row_config(env_id, mode)
                yield env_id, mode, poles, config, tasks.make_env(config, seed=0)

    def test_hanging_rest_is_zero_and_upright_rest_the_peak(self):
        for env_id, mode, poles, config, env in self.cart_envs():
            with self.subTest(env_id=env_id, mode=mode):
                down = hanging(poles)
                self.assertAlmostEqual(env.reward_rate(down, np.zeros(1)), 0.0, places=12)
                self.assertAlmostEqual(env.lyapunov_ratio(down), 1.0, places=12)
                upright = np.zeros(2 * (1 + poles))
                self.assertEqual(env.reward_rate(upright, np.zeros(1)), PEAK)
                self.assertLess(env.lyapunov_ratio(upright), 1e-20)
                # Whole turns change nothing; the floor binds in a small neighbourhood.
                turned = upright.copy()
                turned[1 : 1 + poles] = 2.0 * math.pi
                self.assertEqual(env.reward_rate(turned, np.zeros(1)), PEAK)
                self.assertEqual(env.reward_rate(upright + 1e-6, np.zeros(1)), PEAK)
                # State-only.
                state = np.linspace(-0.7, 0.9, 2 * (1 + poles))
                self.assertEqual(env.reward_rate(state, np.array([7.0])), env.reward_rate(state, np.zeros(1)))
        for mode in ("top", "lyapunov_shoulder"):
            with self.subTest(acrobot=mode):
                env = tasks.make_env(row_config("acrobot-swingup", mode), seed=0)
                for state in ([math.pi, 0.0, 0.0, 0.0], [-math.pi, 0.0, 0.0, 0.0]):
                    self.assertAlmostEqual(env.reward_rate(np.array(state), np.zeros(1)), 0.0, places=12)
                    self.assertAlmostEqual(env.lyapunov_ratio(np.array(state)), 1.0, places=12)
                self.assertEqual(env.reward_rate(np.zeros(4), np.zeros(1)), PEAK)
                self.assertLess(env.lyapunov_ratio(np.zeros(4)), 1e-20)

    def test_cart_values_against_the_formula(self):
        rng = np.random.default_rng(1)
        for env_id, mode, poles, config, env in self.cart_envs():
            reward = config["reward_parameters"]
            chain = dmc_cartpole.cartpole_chain(config["plant_parameters"], poles)
            nv = 1 + poles
            states, _ = cart_states(poles, rng, 40)
            states[:, 1 : 1 + poles] += 2.0 * math.pi * rng.integers(-1, 2, (40, poles))
            with self.subTest(env_id=env_id, mode=mode):
                for state in states:
                    energy = float(chain_energy_numpy(chain, state[:nv], state[nv:])[0])
                    value = 0.5 * (energy - reward["upright_energy"]) ** 2 + 0.5 * reward["lyapunov_k_cart"] * state[0] ** 2
                    if mode.endswith("straight"):
                        value += 0.5 * reward["lyapunov_k_straight"] * sum(wrap(a) ** 2 for a in state[2 : 1 + poles])
                    expected = -math.log(max(value / (0.5 * reward["energy_scale"] ** 2), FLOOR))
                    self.assertAlmostEqual(env.reward_rate(state, np.zeros(1)), expected, places=10)

    def test_zero_sets(self):
        """A: zero wherever E = E_up and x = 0, bent or straight; B: only straight chains among
        those; neither with the cart off centre."""
        for env_id, (name, poles) in CARTPOLES.items():
            plain = tasks.make_env(row_config(env_id, "lyapunov_cart"), seed=0)
            straight = tasks.make_env(row_config(env_id, "lyapunov_cart_straight"), seed=0)
            reward = straight.config["reward_parameters"]
            chain = dmc_cartpole.cartpole_chain(straight.config["plant_parameters"], poles)
            up = reward["upright_energy"]
            bent = at_energy(chain, [0.0, 1.2, 0.7, -0.4][: 1 + poles], [0.3, -1.0, 2.0, 0.5][: 1 + poles], up)
            line = at_energy(chain, [0.0, 1.2] + [0.0] * (poles - 1), [-0.2, 1.5, 0.8, -0.3][: 1 + poles], up)
            with self.subTest(env_id=env_id):
                for state in (bent, line):
                    self.assertAlmostEqual(
                        float(chain_energy_numpy(chain, state[: 1 + poles], state[1 + poles :])[0]), up, places=12
                    )
                    self.assertEqual(plain.reward_rate(state, np.zeros(1)), PEAK)
                    self.assertLess(plain.lyapunov_ratio(state), 1e-20)
                self.assertEqual(straight.reward_rate(line, np.zeros(1)), PEAK)
                bends = sum(wrap(a) ** 2 for a in bent[2 : 1 + poles])
                ratio = 0.5 * reward["lyapunov_k_straight"] * bends / (0.5 * reward["energy_scale"] ** 2)
                self.assertAlmostEqual(straight.lyapunov_ratio(bent), ratio, places=12)
                self.assertAlmostEqual(ratio, bends / math.pi**2, places=12)
                self.assertLess(straight.reward_rate(bent, np.zeros(1)), PEAK - 1.0)
                for env in (plain, straight):
                    shifted = line.copy()
                    shifted[0] = 0.5
                    self.assertAlmostEqual(env.lyapunov_ratio(shifted), (0.5 / CART_RANGE) ** 2, places=12)
                    self.assertLess(env.reward_rate(shifted, np.zeros(1)), PEAK - 1.0)

    def test_shoulder_zero_set_is_the_upright(self):
        """On the top row's zero set {E = E_up, theta_2 = 0} the shoulder variant pays only at
        theta_1 = 0: V_C / V_C_down = k_1 theta_1^2 / 2 / energy_scale^2 = theta_1^2 / (2 pi^2)."""
        top = tasks.make_env(row_config("acrobot-swingup", "top"), seed=0)
        shoulder = tasks.make_env(row_config("acrobot-swingup", "lyapunov_shoulder"), seed=0)
        chain = dmc_acrobot.acrobot_chain(shoulder.config)
        up = shoulder.config["reward_parameters"]["upright_energy"]
        for angle in (0.4, 1.0, -2.0, 3.0):
            with self.subTest(theta_1=angle):
                state = at_energy(chain, [angle, 0.0], [1.0, -0.7], up)
                self.assertEqual(top.reward_rate(state, np.zeros(1)), PEAK)
                self.assertAlmostEqual(shoulder.lyapunov_ratio(state), angle**2 / (2.0 * math.pi**2), places=10)
                self.assertLess(shoulder.reward_rate(state, np.zeros(1)), PEAK - 1.0)
        self.assertEqual(shoulder.reward_rate(np.zeros(4), np.zeros(1)), PEAK)
        # The shoulder term is continuous across theta_1 = +-pi.
        self.assertAlmostEqual(
            shoulder.reward_rate(np.array([math.pi - 1e-9, 0.3, 1.0, 0.0]), np.zeros(1)),
            shoulder.reward_rate(np.array([-math.pi + 1e-9, 0.3, 1.0, 0.0]), np.zeros(1)),
            places=6,
        )
        # Everywhere at most the top row's reward shifted by the normalisation log 2.
        rng = np.random.default_rng(2)
        for state in np.hstack((rng.uniform(-3.0, 3.0, (50, 2)), rng.uniform(-6.0, 6.0, (50, 2)))):
            self.assertLessEqual(
                shoulder.reward_rate(state, np.zeros(1)), top.reward_rate(state, np.zeros(1)) + math.log(2.0) + 1e-9
            )

    def test_settings_are_checked(self):
        config = row_config("cartpole-two_poles", "lyapunov_cart_straight")
        reward = config["reward_parameters"]
        for key in dmc_cartpole.LYAPUNOV_STRAIGHT_KEYS:
            if key == "lyapunov_k_straight":
                continue
            with self.subTest(dropped=key):
                broken = {k: v for k, v in reward.items() if k != key}
                with self.assertRaisesRegex(ValueError, "missing required keys"):
                    tasks.make_env(dict(config, reward_parameters=broken))
        for key in ("energy_scale", "upright_energy", "lyapunov_k_cart", "lyapunov_k_straight"):
            for value in (0.0, -1.0, math.inf, math.nan):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, f"{key} must be positive and finite"):
                        tasks.make_env(dict(config, reward_parameters=dict(reward, **{key: value})))
        for value in (0.0, 1.0, math.nan):
            with self.subTest(reward_floor=value):
                with self.assertRaisesRegex(ValueError, "reward_floor must lie in"):
                    tasks.make_env(dict(config, reward_parameters=dict(reward, reward_floor=value)))
        for key in ("cart_margin", "cosine_bound"):
            with self.subTest(mixed=key):
                mixed = dict(reward, **{key: 1.0})
                with self.assertRaisesRegex(ValueError, "mix"):
                    tasks.make_env(dict(config, reward_parameters=mixed))
                with self.assertRaisesRegex(ValueError, "mix"):
                    tasks.reward_rate_tensor(torch.zeros(2, 6), torch.zeros(2, 1), dict(config, reward_parameters=mixed))
        acrobot = row_config("acrobot-swingup", "lyapunov_shoulder")
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(lyapunov_k_shoulder=value):
                with self.assertRaisesRegex(ValueError, "lyapunov_k_shoulder must be positive and finite"):
                    tasks.make_env(dict(acrobot, reward_parameters=dict(acrobot["reward_parameters"], lyapunov_k_shoulder=value)))
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            tasks.make_env(dict(acrobot, reward_parameters={"lyapunov_k_shoulder": 1.0, "reward_floor": FLOOR}))


class TestTensor(unittest.TestCase):
    def configs(self):
        for env_id, (_, poles) in CARTPOLES.items():
            for mode in CART_MODES:
                yield f"{env_id} {mode}", row_config(env_id, mode), 2 * (1 + poles)
        yield "acrobot lyapunov_shoulder", row_config("acrobot-swingup", "lyapunov_shoulder"), 4

    def states(self, size, rng):
        if size == 4:
            random = np.hstack((rng.uniform(-np.pi, np.pi, (150, 2)), rng.uniform(-10.0, 10.0, (150, 2))))
            near = 10.0 ** rng.uniform(-6.0, -1.0, (150, 1)) * rng.standard_normal((150, 4))
        else:
            random, near = cart_states(size // 2 - 1, rng, 150)
        states = np.vstack((random, near, np.vstack([hanging(size // 2 - 1)] * 3) if size > 4 else np.array([[math.pi, 0.0, 0.0, 0.0]])))
        return states

    def test_tensor_matches_numpy_and_the_registry(self):
        rng = np.random.default_rng(7)
        for label, config, size in self.configs():
            with self.subTest(label):
                env = tasks.make_env(config, seed=0)
                states = self.states(size, rng)
                torques = rng.uniform(-10.0, 10.0, (len(states), 1))
                states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
                tensor = tasks.reward_rate_tensor(states_t, torques_t, config)
                self.assertEqual(tensor.shape, (len(states), 1))
                self.assertEqual(tensor.dtype, torch.float64)
                spec = testbed_spec(config["name"])
                self.assertTrue(torch.equal(tensor, spec.reward_rate_tensor(states_t, torques_t, config)))
                self.assertTrue(torch.equal(tensor, tasks.reward_rate_tensor(states_t, torch.zeros_like(torques_t), config)))
                ratio_fn = (dmc_acrobot.acrobot_lyapunov_ratio_tensor if size == 4
                            else dmc_cartpole.cartpole_lyapunov_ratio_tensor)
                ratios = ratio_fn(states_t, config)[:, 0].tolist()
                capped = 0
                for state, torque, value, ratio in zip(states, torques, tensor[:, 0].tolist(), ratios):
                    self.assertAlmostEqual(value, env.reward_rate(state, torque), places=10)
                    if ratio > 1e-8:
                        # Below, E - E_up cancels to a few ulps and the relative gap grows.
                        self.assertAlmostEqual(ratio / env.lyapunov_ratio(state), 1.0, places=8)
                    capped += abs(value - PEAK) < 1e-12
                self.assertGreater(capped, 10)
                self.assertLess(capped, len(states) - 150)
                self.assertLessEqual(float(tensor.max()), PEAK + 1e-12)
                float32 = tasks.reward_rate_tensor(states_t.float(), torques_t.float(), config)
                self.assertEqual(float32.dtype, torch.float32)
                torch.testing.assert_close(float32.double(), tensor, rtol=1e-3, atol=1e-3)

    def test_gradients_are_finite_and_match_finite_differences(self):
        rng = np.random.default_rng(8)
        for label, config, size in self.configs():
            with self.subTest(label):
                env = tasks.make_env(config, seed=0)
                if size == 4:
                    states = np.hstack((rng.uniform(-2.5, 2.5, (30, 2)), rng.uniform(-6.0, 6.0, (30, 2))))
                else:
                    states, _ = cart_states(size // 2 - 1, rng, 30)
                torques = rng.uniform(-2.0, 2.0, (len(states), 1))

                def single(state, torque):
                    return tasks.reward_rate_tensor(state[None], torque[None], config)[0, 0]

                states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
                by_state = torch.func.vmap(torch.func.grad(single, argnums=0))(states_t, torques_t)
                self.assertTrue(torch.isfinite(by_state).all())
                epsilon = 1e-6
                for state, gradient in zip(states, by_state.numpy()):
                    self.assertLess(env.reward_rate(state, np.zeros(1)), PEAK)
                    column = np.empty(size)
                    for i in range(size):
                        bump = np.zeros(size)
                        bump[i] = epsilon
                        column[i] = (
                            env.reward_rate(state + bump, np.zeros(1)) - env.reward_rate(state - bump, np.zeros(1))
                        ) / (2.0 * epsilon)
                    np.testing.assert_allclose(gradient, column, rtol=1e-5, atol=1e-6)
                leaf = states_t.clone().requires_grad_(True)
                rate = tasks.reward_rate_tensor(leaf, torques_t, config)
                (gradient,) = torch.autograd.grad(rate.sum(), leaf)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue(bool((gradient.abs().sum(dim=-1) > 0.0).all()))


class TestEnergyConservation(unittest.TestCase):
    STARTS = {2: ([0.0, 2.0, -1.0], [0.0, 0.0, 0.0]), 3: ([0.0, 1.5, -0.8, 0.6], [0.0, 0.5, 0.0, -0.5])}

    def test_unforced_trajectories_conserve_the_chain_energy(self):
        """No damping, no force: E of the chain stays constant along MuJoCo's own 1 ms RK4
        trajectory (1 s, the cart well inside the rail), to 1e-5 of energy_scale (the
        measured drift is printed; dm_control's damping alone dissipates ~1e-4 of it)."""
        for env_id, (_, poles) in CARTPOLES.items():
            with self.subTest(env_id=env_id):
                config = row_config(env_id, "lyapunov_cart", cart_damping=0.0, pole_damping=0.0)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                chain = dmc_cartpole.cartpole_chain(config["plant_parameters"], poles)
                self.assertEqual(chain.cart_damping, 0.0)
                positions, velocities = self.STARTS[poles]
                state, _ = env.reset(options={"state": np.array(positions + velocities)})
                nv = 1 + poles

                def energy(s):
                    return float(chain_energy_numpy(chain, s[:nv], s[nv:])[0])

                start = energy(state)
                drift, travel, speed = 0.0, 0.0, 0.0
                for _ in range(100):
                    state = env.step_dt(np.zeros(1), 0.01)["next_state"]
                    drift = max(drift, abs(energy(state) - start))
                    travel = max(travel, abs(state[0]))
                    speed = max(speed, float(np.max(np.abs(state[nv:]))))
                scale = config["reward_parameters"]["energy_scale"]
                print(
                    f"\n{env_id}: unforced energy drift over 1 s {drift:.3e} J = "
                    f"{drift / scale:.3e} energy_scale (max |x| {travel:.3f} m, max rate {speed:.2f})",
                    file=sys.stderr,
                )
                self.assertLess(travel, 1.5)
                self.assertGreater(speed, 2.0)
                self.assertLess(drift, 1e-5 * scale)
                # The ratio the evaluation reports moves with x only (E is constant).
                self.assertTrue(math.isfinite(env.lyapunov_ratio(state)))

    def test_chain_energy_is_mujocos(self):
        """Kinetic energy 0.5 v^T M v with MuJoCo's mass matrix; potential differences equal
        MuJoCo's sum m g z over the bodies' centres of mass."""
        rng = np.random.default_rng(9)
        for env_id, (_, poles) in CARTPOLES.items():
            config = row_config(env_id, "lyapunov_cart")
            env = tasks.make_env(config, seed=0)
            model = env.model
            data = mujoco.MjData(model)
            chain = dmc_cartpole.cartpole_chain(config["plant_parameters"], poles)
            nv = 1 + poles
            gravity = -float(model.opt.gravity[2])

            def mujoco_potential(positions):
                data.qpos[:] = positions
                mujoco.mj_forward(model, data)
                return gravity * float(np.sum(model.body_mass * data.xipos[:, 2]))

            reference = mujoco_potential(np.zeros(nv))
            states, _ = cart_states(poles, rng, 20)
            with self.subTest(env_id=env_id):
                for state in states:
                    data.qpos[:] = state[:nv]
                    data.qvel[:] = state[nv:]
                    mujoco.mj_forward(model, data)
                    mass = np.zeros((model.nv, model.nv))
                    mujoco.mj_fullM(model, mass, data.qM)
                    kinetic = 0.5 * float(state[nv:] @ mass @ state[nv:])
                    potential = float(chain_potential_energy_numpy(chain, state[:nv])[0])
                    total = float(chain_energy_numpy(chain, state[:nv], state[nv:])[0])
                    self.assertAlmostEqual(total - potential, kinetic, delta=1e-9 * max(1.0, kinetic))
                    upright = float(chain_potential_energy_numpy(chain, np.zeros(nv))[0])
                    self.assertAlmostEqual(
                        potential - upright, mujoco_potential(state[:nv]) - reference, places=9
                    )


class ScriptedLyapunovEnv:
    """Replays intervals (duration, next V / V_down) on a windy-named config."""

    def __init__(self, intervals, ratios=True):
        self.intervals = intervals
        self.discount_rate = 0.5
        self.physics_dt = 0.001
        self.config = {"name": "windy_pendulum", "episode_duration": sum(d for d, _ in intervals)}
        self.done = False
        if ratios:
            self.lyapunov_ratio = lambda state: float(state[1])

    def reset(self, seed=None):
        self.index, self.time, self.done = 0, 0.0, False
        return np.array([1.0, 0.9]), {}

    def sample_duration(self):
        return self.intervals[self.index][0]

    def step_dt(self, decision, duration):
        length, ratio = self.intervals[self.index]
        self.last_undiscounted_reward = 0.0
        start = self.time
        self.time += length
        self.index += 1
        self.done = self.index == len(self.intervals)
        return {
            "state": np.array([1.0, 0.0]), "decision": decision, "duration": length,
            "scheduled_duration": length, "reward": 0.0, "next_state": np.array([1.0, ratio]),
            "terminated": False, "truncated": self.done, "time": start, "next_time": self.time,
        }

    def terminal_payoff(self, state):
        return 0.0

    def close(self):
        pass


class TestOrbitDiagnostics(unittest.TestCase):
    INTERVALS = [(1.0, 0.5), (1.0, 5e-3), (0.5, 5e-4), (0.5, 2e-2)]

    def evaluate(self, factory, seeds=(3,), decision=np.zeros(3)):
        return protocol.evaluate_fixed_seeds(
            lambda state, deterministic: decision, factory, list(seeds), discount_rate=0.5, deterministic=True
        )

    def test_time_weighted_shares(self):
        record = self.evaluate(lambda seed: ScriptedLyapunovEnv(self.INTERVALS))[0]
        self.assertAlmostEqual(record["lyapunov_ratio_mean"], (0.5 + 5e-3 + 0.5 * 5e-4 + 0.5 * 2e-2) / 3.0, places=12)
        self.assertEqual(record["lyapunov_ratio_final"], 2e-2)
        self.assertAlmostEqual(record["orbit_fraction_1e_2"], 1.5 / 3.0, places=12)
        self.assertAlmostEqual(record["orbit_fraction_1e_3"], 0.5 / 3.0, places=12)
        self.assertEqual(set(protocol.LYAPUNOV_LEVEL_KEYS) - set(record), set())
        # Without lyapunov_ratio (every other environment) the fields are absent.
        plain = self.evaluate(lambda seed: ScriptedLyapunovEnv(self.INTERVALS, ratios=False))[0]
        self.assertFalse(set(protocol.LYAPUNOV_LEVEL_KEYS) & set(plain))
        summary = protocol.episode_level_summary([record, record])
        self.assertAlmostEqual(summary["orbit_fraction_1e_2"], 0.5, places=12)
        self.assertEqual(set(protocol.episode_level_summary([plain])), set(protocol.EPISODE_LEVEL_KEYS) | {"capture_rate"})

    def test_real_episodes(self):
        """A held feedback law on the two-pole variant and on the acrobot rows; the smooth row
        reports no orbit fields."""
        for env_id, mode, decision in (
            ("cartpole-two_poles", "lyapunov_cart_straight", np.array([0.0, 50.0, 10.0])),
            ("acrobot-swingup", "top", np.array([0.0, 10.0, 2.0])),
            ("acrobot-swingup", "lyapunov_shoulder", np.array([0.5, 20.0, 1.0])),
        ):
            with self.subTest(env_id=env_id, mode=mode):
                config = row_config(env_id, mode)
                record = self.evaluate(lambda seed: tasks.make_env(config, seed=seed), seeds=(20000,), decision=decision)[0]
                self.assertTrue(0.0 <= record["orbit_fraction_1e_3"] <= record["orbit_fraction_1e_2"] <= 1.0)
                self.assertGreaterEqual(record["lyapunov_ratio_mean"], 0.0)
                self.assertTrue(math.isfinite(record["lyapunov_ratio_mean"]))
                self.assertGreaterEqual(record["lyapunov_ratio_final"], 0.0)
                self.assertAlmostEqual(record["survival_seconds"], 10.0)
        config = row_config("cartpole-two_poles", "top")
        record = self.evaluate(lambda seed: tasks.make_env(config, seed=seed), seeds=(20000,),
                               decision=np.array([0.0, 50.0, 10.0]))[0]
        self.assertFalse(set(protocol.LYAPUNOV_LEVEL_KEYS) & set(record))


if __name__ == "__main__":
    unittest.main()
