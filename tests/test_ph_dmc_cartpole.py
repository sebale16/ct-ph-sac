"""The dm_control cart-pole ports: balance, balance_sparse, swingup_sparse, two and three poles.

Each task is checked against dm_control's own cartpole domain (forward dynamics, reward,
1 s trajectories, reset law), its oracle against our MuJoCo model, its tensor task
functions against the environment, and its factory against its table row. Since
2026-11-10 the slider keeps dm_control's soft rail limit and no task has a failure set;
tests/test_ph_dmc_joint_limits.py checks the limit itself (compiled parameters, MuJoCo's
constraint rows and forces, the oracle and its RK4 map through the rail).

dm_control compiles the pole inertia to 0.0094245927672956; the table rounds it to
0.00942459 (relative 3e-7). The comparisons with dm_control at 1e-8 therefore use a
config with the compiled value read from dm_control's model; the table's config is checked
separately against the bound this rounding allows.
"""

import contextlib
import copy
import io
import math
import unittest

import mujoco
import numpy as np
import torch

import common.ph_config as ph_config
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import load_protocol_from_table, main
from common.ph_dmc._common import fixed_clock
from environment.ph_sac import tasks
from environment.ph_sac.base import validated_state_spec
from environment.ph_sac.dmc import cartpole as dmc_cartpole
from environment.ph_sac.dmc.chain import chain_drift_numpy
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac.protocol import UPRIGHT_TUBE, is_upright

# env_id: (testbed, dm_control task, poles, reset law, reward law)
TASKS = {
    "cartpole-balance": ("cartpole_balance", "balance", 1, "balance", "smooth"),
    "cartpole-balance_sparse": ("cartpole_balance_sparse", "balance_sparse", 1, "balance", "sparse"),
    "cartpole-swingup_sparse": ("cartpole_swingup_sparse", "swingup_sparse", 1, "swingup", "sparse"),
    "cartpole-two_poles": ("cartpole_two_poles", "two_poles", 2, "swingup", "smooth"),
    "cartpole-three_poles": ("cartpole_three_poles", "three_poles", 3, "swingup", "smooth"),
}
GEAR = 10.0
FORCE = 10.0
# The relative inertia rounding of the table, and the qacc and 1 s trajectory gaps it
# causes against dm_control (measured: 9e-8, 3e-6 and 8e-6 for one to three poles; 3e-7).
INERTIA_ROUNDING = 1e-6
TABLE_QACC_GAP = 5e-5
TRAJECTORY_TOLERANCE = 1e-6


def table_rows():
    return {entry.env_id: entry for entry in table.load_ph_table() if entry.mode == "top"}


ROWS = table_rows()


def table_config(env_id, **plant):
    """The factory's config from the row's settings, with plant_parameters overridden."""
    entry = ROWS[env_id]
    settings = copy.deepcopy(entry.config["testbed_settings"])
    settings["plant_parameters"].update(plant)
    return testbed_spec(TASKS[env_id][0]).config_factory(**settings)


def dm_load(env_id):
    from dm_control import suite

    return suite.load("cartpole", TASKS[env_id][1], task_kwargs={"random": 0})


def compiled_inertia(physics):
    """dm_control's compiled pole inertia about the hinge axis (identical for every pole)."""
    model = physics.model.ptr
    values = set()
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if name.startswith("pole"):
            values.add(float(model.body_inertia[index, 1]))
    assert len(values) == 1, values
    return values.pop()


def dm_config(env_id, physics):
    """The table's config with dm_control's compiled pole inertia."""
    return table_config(env_id, pole_inertia=compiled_inertia(physics))


def random_states(n_poles, rng, count, cart=1.5, cart_rate=5.0, hinge_rate=15.0):
    """(x, theta_1..n, x_dot, theta_dot_1..n), the angles wrapped, |x| < cart."""
    return np.concatenate(
        (
            rng.uniform(-cart, cart, (count, 1)),
            rng.uniform(-math.pi, math.pi, (count, n_poles)),
            rng.uniform(-cart_rate, cart_rate, (count, 1)),
            rng.uniform(-hinge_rate, hinge_rate, (count, n_poles)),
        ),
        axis=1,
    )


def set_dm_state(physics, state, force):
    """Write our state and force to dm_control's physics: ctrl = force / gear."""
    n = (len(state) - 2) // 2
    physics.data.qpos[:] = state[: n + 1]
    physics.data.qvel[:] = state[n + 1 :]
    physics.data.ctrl[:] = np.asarray(force, dtype=np.float64).reshape(-1) / GEAR
    physics.forward()


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def as_tensor(array):
    return torch.as_tensor(np.asarray(array, dtype=np.float64))


def wrapped_difference(ours, theirs, n_poles):
    difference = np.array(ours - theirs)
    for index in range(1, 1 + n_poles):
        difference[index] = math.atan2(math.sin(difference[index]), math.cos(difference[index]))
    return difference


class TestFactoriesAndTable(unittest.TestCase):
    def test_factories_reproduce_the_rows(self):
        for env_id, (name, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                entry = ROWS[env_id]
                config = table_config(env_id)
                self.assertEqual(config["name"], name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, config), [])
                ph_config._validate_testbed(config)
                spec = validated_state_spec(config)
                self.assertEqual((spec["position_dim"], spec["velocity_dim"]), (1 + n, 1 + n))
                self.assertEqual(tuple(spec["periodic_position_indices"]), tuple(range(1, 1 + n)))
                self.assertEqual(tuple(spec["actuated_position_indices"]), (0,))
                for key in ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters"):
                    self.assertEqual(config[key], entry.config["testbed_settings"][key])
                protocol = load_protocol_from_table(env_id)
                built = protocol["matched"]["testbed_config"]
                self.assertEqual(built["name"], name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, built), [])

    def test_registry_entries(self):
        for env_id, (name, _, _, _, reward_law) in TASKS.items():
            with self.subTest(env_id=env_id):
                spec = testbed_spec(name)
                self.assertTrue(spec.implemented)
                self.assertIs(spec.env_class, dmc_cartpole.CartpoleTaskEnv)
                self.assertEqual(spec.settings_keys, tuple(ROWS[env_id].config["testbed_settings"]))
                self.assertIs(spec.upright, dmc_cartpole.cartpole_poles_upright)
                # Every task dispatches on cartpole_reward_law; the row's config takes the
                # task's own law (the optional Lyapunov rewards: tests/test_ph_orbit_rewards.py).
                self.assertIs(spec.reward_rate_tensor, dmc_cartpole.cartpole_task_reward_rate_tensor)
                self.assertIs(spec.reward_rate_bounds, dmc_cartpole.cartpole_task_reward_rate_bounds)
                config = table_config(env_id)
                self.assertEqual(dmc_cartpole.cartpole_reward_law(config), reward_law)
                self.assertEqual(spec.reward_rate_bounds(config), (0.0, 1.0))
                expected = (
                    dmc_cartpole.cartpole_sparse_reward_rate_tensor
                    if reward_law == "sparse"
                    else dmc_cartpole.cartpole_smooth_reward_rate_tensor
                )
                self.assertIs(dmc_cartpole._LAW_REWARD_TENSORS[reward_law], expected)
                states = torch.as_tensor(random_states(TASKS[env_id][2], np.random.default_rng(5), 64))
                forces = torch.linspace(-12.0, 12.0, 64, dtype=torch.float64)[:, None]
                self.assertTrue(
                    torch.equal(
                        spec.reward_rate_tensor(states, forces, config), expected(states, forces, config)
                    )
                )

    def test_list_shows_the_rows_as_builds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--list"])
        status = {line.split()[0]: line for line in output.getvalue().splitlines()[1:]}
        for env_id in TASKS:
            self.assertTrue(status[env_id].endswith("builds"), status[env_id])

    def test_pole_count_is_checked(self):
        settings = copy.deepcopy(ROWS["cartpole-two_poles"].config["testbed_settings"])
        settings["plant_parameters"]["n_poles"] = 3
        with self.assertRaisesRegex(ValueError, "n_poles"):
            testbed_spec("cartpole_two_poles").config_factory(**settings)
        del settings["plant_parameters"]["n_poles"]
        with self.assertRaisesRegex(ValueError, "n_poles"):
            testbed_spec("cartpole_two_poles").config_factory(**settings)
        settings = copy.deepcopy(ROWS["cartpole-balance"].config["testbed_settings"])
        settings["plant_parameters"]["n_poles"] = 2
        with self.assertRaisesRegex(ValueError, "n_poles"):
            testbed_spec("cartpole_balance").config_factory(**settings)
        config = table_config("cartpole-three_poles")
        config["plant_parameters"] = dict(config["plant_parameters"], n_poles=2)
        with self.assertRaisesRegex(ValueError, "n_poles"):
            tasks.make_env(config)

    def test_missing_keys_raise(self):
        for env_id, section, key in (
            ("cartpole-balance", "reset_parameters", "angle_range"),
            ("cartpole-balance_sparse", "reward_parameters", "cosine_bound"),
            ("cartpole-two_poles", "reset_parameters", "upper_angle_noise"),
            ("cartpole-three_poles", "plant_parameters", "pole_com"),
        ):
            with self.subTest(env_id=env_id, key=key):
                config = table_config(env_id)
                config[section] = {k: v for k, v in config[section].items() if k != key}
                with self.assertRaisesRegex(ValueError, key):
                    tasks.make_env(config)


class TestDynamics(unittest.TestCase):
    def test_oracle_matches_our_mujoco(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                drift, diffusion = tasks.oracle_callbacks(config)
                rng = np.random.default_rng(0)
                states = random_states(n, rng, 200, cart=1.75)
                torques = rng.uniform(-FORCE, FORCE, (200, 1))
                tensor = drift(as_tensor(states), as_tensor(torques)).numpy()
                for state, torque, batched in zip(states, torques, tensor):
                    truth = env.mujoco_drift(state, torque)
                    self.assertLess(relative_error(env.oracle_drift(state, torque), truth), 1e-8)
                    self.assertLess(relative_error(batched, truth), 1e-8)
                zeros = torch.zeros(200, 2 + 2 * n, 1, dtype=torch.float64)
                self.assertTrue(torch.equal(diffusion(as_tensor(states)), zeros))
                np.testing.assert_array_equal(env.oracle_diffusion(states[0]), np.zeros((2 + 2 * n, 1)))

    def test_single_pole_closed_form_equals_the_chain(self):
        plant = table_config("cartpole-balance")["plant_parameters"]
        chain = dmc_cartpole.cartpole_chain(plant, 1)
        drift, _ = dmc_cartpole._cartpole_callbacks(plant)
        rng = np.random.default_rng(1)
        states = random_states(1, rng, 100)
        torques = rng.uniform(-FORCE, FORCE, (100, 1))
        expected = drift(as_tensor(states), as_tensor(torques)).numpy()
        self.assertLess(relative_error(chain_drift_numpy(chain, states, torques), expected), 1e-10)

    def test_our_mujoco_matches_dm_control_forward_dynamics(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                physics = dm_load(env_id).physics
                inertia = compiled_inertia(physics)
                table_inertia = table_config(env_id)["plant_parameters"]["pole_inertia"]
                self.assertLess(abs(table_inertia - inertia) / inertia, INERTIA_ROUNDING)
                exact = tasks.make_env(dm_config(env_id, physics), seed=0)
                rounded = tasks.make_env(table_config(env_id), seed=0)
                self.assertEqual(exact.model.nq, physics.model.nq)
                rng = np.random.default_rng(2)
                worst_rounded = 0.0
                for state in random_states(n, rng, 200):
                    force = rng.uniform(-FORCE, FORCE, 1)
                    set_dm_state(physics, state, force)
                    expected = np.array(physics.data.qacc)
                    self.assertLess(relative_error(exact.mujoco_drift(state, force)[1 + n :], expected), 1e-8)
                    self.assertLess(relative_error(exact.oracle_drift(state, force)[1 + n :], expected), 1e-8)
                    worst_rounded = max(
                        worst_rounded, relative_error(rounded.mujoco_drift(state, force)[1 + n :], expected)
                    )
                self.assertLess(worst_rounded, TABLE_QACC_GAP)

    def test_trajectories_match_dm_control(self):
        """1 s of open-loop control at dm_control's 10 ms RK4 step, from our reset."""
        for env_id, (_, _, n, _, _) in TASKS.items():
            for label in ("compiled", "table"):
                with self.subTest(env_id=env_id, inertia=label):
                    dm_env = dm_load(env_id)
                    physics = dm_env.physics
                    config = dm_config(env_id, physics) if label == "compiled" else table_config(env_id)
                    config = dict(config, physics_dt=0.01, clock=fixed_clock(0.01, 0.01))
                    env = tasks.make_env(config, decision_mode="torque", seed=3)
                    start, _ = env.reset(seed=3)
                    with physics.reset_context():
                        physics.data.qpos[:] = start[: n + 1]
                        physics.data.qvel[:] = start[n + 1 :]
                    controls = np.random.default_rng(4).uniform(-1.0, 1.0, 100)
                    worst = 0.0
                    for control in controls:
                        transition = env.step_dt(np.array([GEAR * control]), 0.01)
                        self.assertEqual(env.last_interval["physics_steps"], 1)
                        physics.data.ctrl[:] = control
                        physics.step()
                        theirs = np.concatenate((physics.data.qpos, physics.data.qvel))
                        difference = wrapped_difference(transition["next_state"], theirs, n)
                        worst = max(worst, float(np.abs(difference).max()))
                    self.assertFalse(transition["terminated"])
                    self.assertLess(abs(transition["next_state"][0]), 1.5)
                    self.assertLess(worst, 1e-12 if label == "compiled" else TRAJECTORY_TOLERANCE)

    def test_oracle_plant_and_torch_func(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                env = tasks.make_env(table_config(env_id), seed=0)
                rng = np.random.default_rng(5)
                states = random_states(n, rng, 6)
                actions = rng.uniform(-FORCE, FORCE, (6, 1))
                previous = torch.get_default_dtype()
                torch.set_default_dtype(torch.float64)
                try:
                    plant = tasks.make_oracle_plant(env)
                    values, by_state, by_action = plant.drift_jacobian(as_tensor(states), as_tensor(actions))
                finally:
                    torch.set_default_dtype(previous)
                truth = np.stack([env.oracle_drift(s, a) for s, a in zip(states, actions)])
                self.assertLess(relative_error(values.numpy(), truth), 1e-10)
                self.assertEqual(plant.state_spec, env.state_spec)
                np.testing.assert_array_equal(plant.input_matrix().numpy(), env.oracle_input_matrix())
                epsilon = 1e-6
                drift, _ = tasks.oracle_callbacks(env.config)
                for i in range(2 + 2 * n):
                    bump = np.zeros_like(states)
                    bump[:, i] = epsilon
                    numeric = (
                        drift(as_tensor(states + bump), as_tensor(actions))
                        - drift(as_tensor(states - bump), as_tensor(actions))
                    ) / (2.0 * epsilon)
                    self.assertLess((by_state[:, :, i] - numeric).abs().max().item(), 1e-5)
                numeric = (
                    drift(as_tensor(states), as_tensor(actions + epsilon))
                    - drift(as_tensor(states), as_tensor(actions - epsilon))
                ) / (2.0 * epsilon)
                self.assertLess((by_action[:, :, 0] - numeric).abs().max().item(), 1e-6)


def near_upright_states(n_poles, rng, count):
    """States around the sparse reward's bounds: |x| near 0.25, cos phi near 0.995."""
    states = random_states(n_poles, rng, count, cart=0.35, cart_rate=1.0, hinge_rate=3.0)
    states[:, 1 : 1 + n_poles] = rng.uniform(-0.15, 0.15, (count, n_poles))
    edge = math.acos(0.995)
    edges = [edge, -edge, edge + 1e-9, edge - 1e-9, -edge + 1e-9, 2.0 * math.pi - edge, 0.0]
    cart_edges = [0.25, -0.25, np.nextafter(0.25, 1.0), np.nextafter(-0.25, -1.0), 0.25 - 1e-9, 0.0]
    rows = []
    for angle in edges:
        for position in cart_edges:
            row = np.zeros(2 + 2 * n_poles)
            row[0], row[1] = position, angle
            rows.append(row)
    return np.concatenate((np.array(rows), states))


class TestRewards(unittest.TestCase):
    def check_against_dm_control(self, env_id, states, forces):
        name, _, n, _, reward_law = TASKS[env_id]
        config = table_config(env_id)
        env = tasks.make_env(config, seed=0)
        dm_env = dm_load(env_id)
        tensor = tasks.reward_rate_tensor(as_tensor(states), as_tensor(forces), config)
        self.assertEqual(tensor.shape, (len(states), 1))
        ours, theirs = [], []
        for state, force in zip(states, forces):
            set_dm_state(dm_env.physics, state, force)
            theirs.append(float(dm_env.task.get_reward(dm_env.physics)))
            ours.append(env.reward_rate(state, force))
        ours, theirs = np.array(ours), np.array(theirs)
        np.testing.assert_allclose(tensor[:, 0].numpy(), ours, rtol=0.0, atol=1e-12)
        if reward_law == "sparse":
            self.assertEqual(int(np.sum(ours != theirs)), 0)
            self.assertTrue(set(np.unique(ours)) <= {0.0, 1.0})
        else:
            np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-10)
        return ours

    def test_smooth_rewards_match_dm_control(self):
        for env_id in ("cartpole-balance", "cartpole-two_poles", "cartpole-three_poles"):
            with self.subTest(env_id=env_id):
                n = TASKS[env_id][2]
                rng = np.random.default_rng(6)
                states = np.concatenate(
                    (random_states(n, rng, 200, cart=3.0), near_upright_states(n, rng, 100))
                )
                forces = rng.uniform(-FORCE, FORCE, (len(states), 1))
                forces[:5] = [[0.0], [FORCE], [-FORCE], [0.5 * FORCE], [-0.3 * FORCE]]
                rewards = self.check_against_dm_control(env_id, states, forces)
                self.assertTrue(np.all((rewards >= 0.0) & (rewards <= 1.0)))

    def test_sparse_rewards_match_dm_control_at_the_edges(self):
        for env_id in ("cartpole-balance_sparse", "cartpole-swingup_sparse"):
            with self.subTest(env_id=env_id):
                rng = np.random.default_rng(7)
                states = np.concatenate((near_upright_states(1, rng, 400), random_states(1, rng, 100)))
                forces = rng.uniform(-FORCE, FORCE, (len(states), 1))
                rewards = self.check_against_dm_control(env_id, states, forces)
                self.assertGreater(rewards.sum(), 50)
                self.assertGreater((rewards == 0.0).sum(), 50)

    def test_reward_values(self):
        def rate(env_id, positions, force=0.0):
            n = TASKS[env_id][2]
            state = np.concatenate((positions, np.zeros(1 + n)))
            return tasks.make_env(table_config(env_id)).reward_rate(state, np.array([force]))

        self.assertAlmostEqual(rate("cartpole-balance", [0.0, 0.0]), 1.0, places=12)
        self.assertAlmostEqual(rate("cartpole-balance", [0.0, 0.0], FORCE), 0.8, places=12)
        self.assertAlmostEqual(rate("cartpole-balance", [2.0, 0.0]), 0.55, places=12)
        self.assertAlmostEqual(rate("cartpole-two_poles", [0.0, 0.0, 0.0]), 1.0, places=12)
        # The second pole at a right angle to the first: upright mean (1 + 1/2) / 2.
        self.assertAlmostEqual(rate("cartpole-two_poles", [0.0, 0.0, 0.5 * math.pi]), 0.75, places=12)
        # Folded back: the absolute angles are pi/2 and 0, not pi/2 and -pi/2.
        self.assertAlmostEqual(
            rate("cartpole-two_poles", [0.0, 0.5 * math.pi, -0.5 * math.pi]), 0.75, places=12
        )
        self.assertAlmostEqual(rate("cartpole-three_poles", [0.0, math.pi, 0.0, 0.0]), 0.0, places=12)
        self.assertEqual(rate("cartpole-balance_sparse", [0.25, 0.0]), 1.0)
        self.assertEqual(rate("cartpole-balance_sparse", [0.2501, 0.0]), 0.0)
        self.assertEqual(rate("cartpole-swingup_sparse", [0.0, 0.11]), 0.0)
        # The fastest hinge sets the velocity term; the cart rate does not enter.
        env = tasks.make_env(table_config("cartpole-three_poles"))
        state = np.zeros(8)
        state[4], state[6] = 9.0, 5.0
        self.assertAlmostEqual(env.reward_rate(state, np.zeros(1)), 0.55, places=12)

    def test_one_pole_smooth_reward_is_the_swingup_reward(self):
        config = table_config("cartpole-balance")
        rng = np.random.default_rng(8)
        states = as_tensor(random_states(1, rng, 100, cart=3.0))
        torques = as_tensor(rng.uniform(-12.0, 12.0, (100, 1)))
        torch.testing.assert_close(
            dmc_cartpole.cartpole_smooth_reward_rate_tensor(states, torques, config),
            dmc_cartpole.cartpole_swingup_reward_rate_tensor(states, torques, config),
            rtol=0.0,
            atol=1e-14,
        )


class TestTaskTensors(unittest.TestCase):
    def test_agreement_and_no_failure_set(self):
        """Rewards agree; since 2026-11-10 there is no failure set (the rail is dm_control's
        soft limit): the mask is all False, at and beyond the rail too, H = 0 and B = 1."""
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                self.assertEqual(config["failure_limits"], {})
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(9)
                states = random_states(n, rng, 200, cart=0.98 * 1.8)
                states[:50, 0] *= 1.2  # Some rows fall beyond the rail.
                states[50:70] = near_upright_states(n, rng, 20)[-20:]
                states[70:74, 0] = [1.8, -1.8, 1.8, -1.8]
                torques = rng.uniform(-12.0, 12.0, (200, 1))
                rates = tasks.reward_rate_tensor(as_tensor(states), as_tensor(torques), config)
                mask = tasks.failure_mask_tensor(as_tensor(states), config)
                self.assertEqual((rates.shape, mask.shape), ((200, 1), (200,)))
                self.assertEqual(mask.sum().item(), 0)
                self.assertGreater(int((np.abs(states[:, 0]) > 1.8).sum()), 0)
                for i in range(200):
                    self.assertAlmostEqual(
                        rates[i, 0].item(), env.reward_rate(states[i], torques[i]), places=10
                    )
                    self.assertFalse(env.failed(states[i]))
                payoff = tasks.terminal_payoff_tensor(as_tensor(states), config)
                self.assertEqual(payoff.shape, (200, 1))
                self.assertTrue(torch.equal(payoff, torch.zeros(200, 1, dtype=torch.float64)))
                self.assertEqual(env.terminal_payoff(states[0]), 0.0)

                lift, vanishing = tasks.terminal_boundary_functions(config)
                self.assertTrue(torch.equal(vanishing(as_tensor(states)), torch.ones(200, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(lift(as_tensor(states)), torch.zeros(200, 1, dtype=torch.float64)))


class TestResets(unittest.TestCase):
    def draw(self, env_id, count=400):
        n = TASKS[env_id][2]
        env = tasks.make_env(table_config(env_id), seed=10)
        ours = np.array([env.reset()[0] for _ in range(count)])
        dm_env = dm_load(env_id)
        theirs = []
        for _ in range(count):
            dm_env.reset()
            physics = dm_env.physics
            theirs.append(np.concatenate((physics.data.qpos, physics.data.qvel)))
        theirs = np.array(theirs)
        for samples in (ours, theirs):
            for index in range(1, 1 + n):
                samples[:, index] = np.arctan2(np.sin(samples[:, index]), np.cos(samples[:, index]))
        return ours, theirs

    def test_balance_reset_matches_dm_control(self):
        for env_id in ("cartpole-balance", "cartpole-balance_sparse"):
            with self.subTest(env_id=env_id):
                reset = table_config(env_id)["reset_parameters"]
                ours, theirs = self.draw(env_id)
                for samples in (ours, theirs):
                    self.assertLessEqual(np.abs(samples[:, 0]).max(), reset["position_range"])
                    self.assertGreater(np.abs(samples[:, 0]).max(), 0.95 * reset["position_range"])
                    self.assertLessEqual(np.abs(samples[:, 1]).max(), reset["angle_range"])
                    self.assertGreater(np.abs(samples[:, 1]).max(), 0.95 * reset["angle_range"])
                    uniform = np.array([reset["position_range"], reset["angle_range"]]) / math.sqrt(3.0)
                    np.testing.assert_allclose(samples[:, :2].std(axis=0), uniform, rtol=0.1)
                    np.testing.assert_allclose(samples[:, 2:].std(axis=0), reset["velocity_noise"], rtol=0.15)
                    self.assertLess(np.abs(samples.mean(axis=0)).max(), 0.01)
                np.testing.assert_allclose(ours.std(axis=0), theirs.std(axis=0), rtol=0.2)

    def test_swingup_reset_matches_dm_control(self):
        for env_id in ("cartpole-swingup_sparse", "cartpole-two_poles", "cartpole-three_poles"):
            with self.subTest(env_id=env_id):
                n = TASKS[env_id][2]
                reset = table_config(env_id)["reset_parameters"]
                ours, theirs = self.draw(env_id)
                expected = np.array(
                    [reset["position_noise"], reset["angle_noise"]]
                    + [reset.get("upper_angle_noise", 0.0)] * (n - 1)
                    + [reset["velocity_noise"]] * (1 + n)
                )
                for samples in (ours, theirs):
                    hanging = np.arctan2(np.sin(samples[:, 1] - math.pi), np.cos(samples[:, 1] - math.pi))
                    self.assertLess(np.abs(hanging).max(), 0.06)
                    centred = np.array(samples)
                    centred[:, 1] = hanging
                    np.testing.assert_allclose(centred.std(axis=0), expected, rtol=0.15)
                    self.assertLess(np.abs(centred.mean(axis=0) / expected).max(), 0.25)
                np.testing.assert_allclose(ours.std(axis=0), theirs.std(axis=0), rtol=0.2)

    def test_reset_is_seeded_and_takes_a_state(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                env = tasks.make_env(table_config(env_id))
                first, _ = env.reset(seed=11)
                again, _ = env.reset(seed=11)
                np.testing.assert_array_equal(first, again)
                state = np.zeros(2 + 2 * n)
                state[0], state[1:1 + n] = 0.4, 3.5
                observed, _ = env.reset(options={"state": state})
                self.assertEqual(observed[0], 0.4)
                np.testing.assert_allclose(observed[1 : 1 + n], 3.5 - 2.0 * math.pi, atol=1e-12)


class TestSmoke(unittest.TestCase):
    def test_layout(self):
        for env_id, (_, _, n, reset_law, reward_law) in TASKS.items():
            with self.subTest(env_id=env_id):
                env = tasks.make_env(table_config(env_id))
                self.assertEqual((env.n_poles, env.reset_law, env.reward_law), (n, reset_law, reward_law))
                self.assertEqual((env.model.nq, env.model.nv, env.model.nu), (1 + n, 1 + n, 1))
                self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_RK4)
                self.assertEqual(env.model.opt.timestep, 0.001)
                # dm_control's soft rail limit: the slider limited, the constraint solver on.
                self.assertFalse(env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
                self.assertEqual(env.model.jnt_limited.tolist(), [1] + [0] * n)
                np.testing.assert_array_equal(env.model.jnt_range[0], [-1.8, 1.8])
                self.assertEqual(env.wrapped_observation_indices, tuple(range(1, 1 + n)))
                np.testing.assert_array_equal(env.oracle_input_matrix(), [[1.0]] + [[0.0]] * n)
                np.testing.assert_array_equal(env.model.actuator_ctrlrange, [[-FORCE, FORCE]])

    def test_feedback_and_torque_episodes(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            for mode in ("feedback", "torque"):
                with self.subTest(env_id=env_id, mode=mode):
                    env = tasks.make_env(table_config(env_id), decision_mode=mode, seed=0)
                    start, info = env.reset(seed=0)
                    self.assertEqual(start.shape, (2 + 2 * n,))
                    decision = (
                        np.array([0.3, 40.0, 5.0]) if mode == "feedback" else np.array([3.0])
                    )
                    self.assertTrue(env.action_space.contains(decision))
                    for _ in range(5):
                        state, reward, terminated, truncated, info = env.step(decision)
                        self.assertTrue(np.isfinite(state).all())
                        self.assertTrue(math.isfinite(reward))
                        self.assertAlmostEqual(info["dt"], 0.01)
                        self.assertFalse(terminated or truncated)
                    self.assertAlmostEqual(env.time, 0.05)
                    self.assertGreater(state[0], start[0])  # Both push the cart toward +x.

    def test_episode_continues_past_the_rail(self):
        """Since 2026-11-10 the rail is dm_control's soft limit: the cart passes 1.8 m by a
        few centimetres, is pushed back, and the episode goes on."""
        env = tasks.make_env(table_config("cartpole-two_poles"), decision_mode="torque", seed=0)
        env.reset(options={"state": np.array([1.7, 0.0, 0.0, 3.0, 0.0, 0.0])})
        deepest = 0.0
        for _ in range(30):
            transition = env.step_dt(np.array([FORCE]), 0.01)
            self.assertFalse(transition["terminated"])
            deepest = max(deepest, transition["next_state"][0] - 1.8)
        self.assertGreater(deepest, 0.0)
        self.assertLess(deepest, 0.3)
        self.assertFalse(env.done)

    def test_upright_predicate(self):
        for env_id, (_, _, n, _, _) in TASKS.items():
            with self.subTest(env_id=env_id):
                config = table_config(env_id)

                def state(*angles):
                    return np.concatenate(([0.5], angles, np.full(1 + n, 3.0)))

                upright = dmc_cartpole.cartpole_poles_upright
                self.assertTrue(upright(config, state(*[0.0] * n), 0.2))
                self.assertTrue(is_upright(config, state(*[0.0] * n)))
                self.assertTrue(upright(config, state(*([2.0 * math.pi] + [0.0] * (n - 1))), 0.2))
                self.assertFalse(upright(config, state(*([math.pi] + [0.0] * (n - 1))), 0.2))
                self.assertFalse(is_upright(config, state(*([UPRIGHT_TUBE + 0.01] + [0.0] * (n - 1)))))
                if n > 1:
                    # Each relative angle within the tube, but the second pole's absolute
                    # angle 0.3 outside it; and a fold that leaves both poles upright.
                    self.assertFalse(upright(config, state(*([0.15, 0.15] + [0.0] * (n - 2))), 0.2))
                    self.assertTrue(upright(config, state(*([0.15, -0.15] + [0.0] * (n - 2))), 0.2))
                    self.assertFalse(upright(config, state(*([0.0] * (n - 1) + [math.pi])), 0.2))


if __name__ == "__main__":
    unittest.main()
