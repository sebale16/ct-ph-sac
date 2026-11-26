"""The dm_control ports reacher_easy, reacher_hard and point_mass_easy.

Each task is checked against its own MuJoCo model, dm_control's compiled model and
simulation, its tensor task functions and its table row (environment/ph_sac/dmc/README.md,
"Validation each port must pass"); the reacher's target is the episode-constant context.
Since 2026-11-10 the wrist and the slides keep dm_control's soft joint limits and neither
task has a failure set; tests/test_ph_dmc_joint_limits.py checks the limits themselves.
"""

import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_dmc._common import fixed_clock
from environment.ph_sac import tasks
from environment.ph_sac.dmc import reacher as dmc_reacher
from environment.ph_sac.registry import registered_testbeds, testbed_spec

ENV_IDS = {
    "reacher_easy": "reacher-easy",
    "reacher_hard": "reacher-hard",
    "point_mass_easy": "point_mass-easy",
}
DM_TASKS = {
    "reacher_easy": ("reacher", "easy"),
    "reacher_hard": ("reacher", "hard"),
    "point_mass_easy": ("point_mass", "easy"),
}
TESTBEDS = tuple(ENV_IDS)
REACHERS = ("reacher_easy", "reacher_hard")
# dm_control's control = our torque / gear.
GEAR = {"reacher": 0.05, "point_mass": 0.1}
# Keep the soft joint limits (wrist 160 deg, slides 0.29 m) inactive in these comparisons;
# tests/test_ph_dmc_joint_limits.py compares them with dm_control's.
WRIST_SAFE = math.radians(150.0)
POSITION_SAFE = 0.25


def is_reacher(name):
    return name in REACHERS


def table_entry(name):
    return table.find_ph_entry(ENV_IDS[name], "top")


def table_config(name):
    return testbed_spec(name).config_factory(**table_entry(name).config["testbed_settings"])


def euler_config(name):
    """The table config with dm_control's physics: 20 ms Euler steps."""
    config = copy.deepcopy(table_config(name))
    config["physics_dt"] = 0.02
    config["integrator"] = "Euler"
    config["clock"] = fixed_clock(0.02, 0.02)
    return config


def load_dm(name):
    from dm_control import suite

    domain, task = DM_TASKS[name]
    env = suite.load(domain, task, task_kwargs={"random": 0})
    env.reset()
    return env


def torque_bound(config):
    return float(config["controller"]["torque_high"][0])


def finger(config, states):
    plant = config["plant_parameters"]
    shoulder, absolute = states[..., 0], states[..., 0] + states[..., 1]
    return np.stack(
        (
            plant["arm_length"] * np.cos(shoulder) + plant["finger_offset"] * np.cos(absolute),
            plant["arm_length"] * np.sin(shoulder) + plant["finger_offset"] * np.sin(absolute),
        ),
        axis=-1,
    )


def random_states(config, rng, count):
    """States away from dm_control's stops; about half reach the target (reward > 0).

    The reacher's target is drawn on a disc of radius 0.25 m or near the finger; the
    point mass sits anywhere in +-0.25 m or near the origin.
    """
    near = rng.random(count) < 0.5
    if config["name"] in REACHERS:
        positions = np.stack(
            (rng.uniform(-np.pi, np.pi, count), rng.uniform(-WRIST_SAFE, WRIST_SAFE, count)), axis=-1
        )
        radius, angle = rng.uniform(0.0, 0.25, count), rng.uniform(0.0, 2.0 * np.pi, count)
        targets = np.stack((radius * np.sin(angle), radius * np.cos(angle)), axis=-1)
        reach = config["reward_parameters"]["reach_radius"]
        offset, direction = rng.uniform(0.0, 2.0 * reach, count), rng.uniform(0.0, 2.0 * np.pi, count)
        close = finger(config, positions) + offset[:, None] * np.stack((np.cos(direction), np.sin(direction)), axis=-1)
        targets[near] = close[near]
        velocities = rng.uniform(-5.0, 5.0, (count, 2))
        return np.concatenate((positions, targets, velocities), axis=-1)
    positions = rng.uniform(-POSITION_SAFE, POSITION_SAFE, (count, 2))
    positions[near] = rng.uniform(-0.04, 0.04, (int(near.sum()), 2))
    velocities = rng.uniform(-1.0, 1.0, (count, 2))
    return np.concatenate((positions, velocities), axis=-1)


def set_dm_state(dm_env, name, state, torque):
    """Write our state (the target into the target geom) and control to dm_control."""
    physics = dm_env.physics
    physics.data.qpos[:] = state[:2]
    physics.data.qvel[:] = state[-2:]
    physics.data.ctrl[:] = np.asarray(torque) / GEAR[DM_TASKS[name][0]]
    if is_reacher(name):
        physics.named.model.geom_pos["target", :2] = state[2:4]
    physics.forward()


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def wrap(angles):
    return np.mod(np.asarray(angles) + np.pi, 2.0 * np.pi) - np.pi


def ks_statistic(left, right):
    """Two-sample Kolmogorov-Smirnov statistic sup |F_left - F_right|."""
    left, right = np.sort(left), np.sort(right)
    points = np.concatenate((left, right))
    return float(
        np.max(
            np.abs(
                np.searchsorted(left, points, side="right") / len(left)
                - np.searchsorted(right, points, side="right") / len(right)
            )
        )
    )


class Float64:
    """Run a block with torch's default dtype set to float64."""

    def __enter__(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *args):
        torch.set_default_dtype(self.previous)


# ----------------------------------------------------------------------- oracle drift


class TestOracleDrift(unittest.TestCase):
    def test_oracle_matches_mujoco(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(0)
                states = random_states(config, rng, 200)
                limit = torque_bound(config)
                torques = rng.uniform(-limit, limit, (200, 2))
                drift, diffusion = tasks.oracle_callbacks(config)
                tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                for state, torque, from_tensor in zip(states, torques, tensor):
                    truth = env.mujoco_drift(state, torque)
                    analytic = env.oracle_drift(state, torque)
                    self.assertLess(relative_error(analytic, truth), 1e-8)
                    self.assertLess(relative_error(from_tensor, truth), 1e-8)
                    np.testing.assert_array_equal(truth[: env.position_dim], env.state_spec["observation_projection"] @ state[env.position_dim :])
                    if is_reacher(name):
                        for value in (truth, analytic, from_tensor):
                            np.testing.assert_array_equal(value[2:4], [0.0, 0.0])
                sigma = diffusion(torch.as_tensor(states))
                self.assertEqual(sigma.shape, (200, env.state_dim, 1))
                self.assertTrue(torch.equal(sigma, torch.zeros_like(sigma)))
                np.testing.assert_array_equal(env.oracle_diffusion(states[0]), np.zeros((env.state_dim, 1)))
                np.testing.assert_array_equal(env.oracle_input_matrix(), np.eye(2))

    def test_point_mass_oracle_is_the_linear_closed_form(self):
        config = table_config("point_mass_easy")
        drift, _ = tasks.oracle_callbacks(config)
        state, force = np.array([0.1, -0.2, 0.3, -0.4]), np.array([0.05, -0.02])
        expected = [0.3, -0.4, (0.05 - 0.3) / 0.3, (-0.02 + 0.4) / 0.3]
        np.testing.assert_allclose(tasks.make_env(config).oracle_drift(state, force), expected, rtol=1e-14)
        got = drift(torch.as_tensor(state[None]), torch.as_tensor(force[None]))[0].numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-14)

    def test_oracle_plant_drift_and_jacobian(self):
        """make_oracle_plant(env): the drift is oracle_drift and drift_jacobian is right."""
        for name in TESTBEDS:
            with self.subTest(testbed=name), Float64():
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                plant = tasks.make_oracle_plant(env)
                rng = np.random.default_rng(1)
                states = torch.as_tensor(random_states(config, rng, 6))
                limit = torque_bound(config)
                actions = torch.as_tensor(rng.uniform(-limit, limit, (6, 2)))
                expected = np.stack([env.oracle_drift(s, a) for s, a in zip(states.numpy(), actions.numpy())])
                np.testing.assert_allclose(plant.drift(states, actions).numpy(), expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(plant.state_spec, env.state_spec)
                np.testing.assert_array_equal(plant.input_matrix().numpy(), np.eye(2))
                values, by_state, by_action = plant.drift_jacobian(states, actions)
                torch.testing.assert_close(values, plant.drift(states, actions))
                for argument, jacobian, size in ((0, by_state, env.state_dim), (1, by_action, 2)):
                    epsilon = 1e-6 if argument == 0 else 1e-7
                    for i in range(size):
                        bump = torch.zeros(6, size, dtype=torch.float64)
                        bump[:, i] = epsilon
                        if argument == 0:
                            high, low = plant.drift(states + bump, actions), plant.drift(states - bump, actions)
                        else:
                            high, low = plant.drift(states, actions + bump), plant.drift(states, actions - bump)
                        numeric = (high - low) / (2.0 * epsilon)
                        scale = 1.0 + numeric.abs().max().item()
                        self.assertLess((jacobian[:, :, i] - numeric).abs().max().item() / scale, 1e-6)
                if is_reacher(name):
                    # The context has zero rate and does not move the dynamics.
                    self.assertEqual(by_state[:, 2:4].abs().max().item(), 0.0)
                    self.assertEqual(by_state[:, :, 2:4].abs().max().item(), 0.0)


# ----------------------------------------------------------------- against dm_control


class TestAgainstDmControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dm = {name: load_dm(name) for name in TESTBEDS}

    def test_forward_dynamics(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                dm_env = self.dm[name]
                rng = np.random.default_rng(2)
                limit = torque_bound(config)
                for state in random_states(config, rng, 200):
                    torque = rng.uniform(-limit, limit, 2)
                    set_dm_state(dm_env, name, state, torque)
                    expected = np.array(dm_env.physics.data.qacc)
                    got = env.mujoco_drift(state, torque)[env.position_dim :]
                    self.assertLess(relative_error(got, expected), 1e-8)

    def test_reward(self):
        """Tensor, environment and dm_control's task.get_reward at the same state and target."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                dm_env = self.dm[name]
                physics = dm_env.physics
                rng = np.random.default_rng(3)
                states = random_states(config, rng, 400)
                limit = torque_bound(config)
                # Beyond the motor limit the control factor reaches 0, as dm_control's.
                torques = rng.uniform(-1.2 * limit, 1.2 * limit, (400, 2))
                tensor = tasks.reward_rate_tensor(torch.as_tensor(states), torch.as_tensor(torques), config)
                self.assertEqual(tensor.shape, (400, 1))
                rewards = []
                for state, torque, from_tensor in zip(states, torques, tensor[:, 0].tolist()):
                    if is_reacher(name):
                        distance = np.linalg.norm(finger(config, state) - state[2:4])
                        # The indicator: keep away from its edge (none falls this close).
                        self.assertGreater(abs(distance - config["reward_parameters"]["reach_radius"]), 1e-9)
                    set_dm_state(dm_env, name, state, torque)
                    expected = float(dm_env.task.get_reward(physics))
                    got = env.reward_rate(state, torque)
                    self.assertLessEqual(abs(got - expected), 1e-10)
                    self.assertLessEqual(abs(from_tensor - expected), 1e-10)
                    rewards.append(expected)
                rewards = np.array(rewards)
                if is_reacher(name):
                    radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
                    self.assertAlmostEqual(radii, config["reward_parameters"]["reach_radius"], places=15)
                    self.assertTrue(set(rewards.tolist()) <= {0.0, 1.0})
                    self.assertGreater((rewards == 1.0).sum(), 50)
                    self.assertGreater((rewards == 0.0).sum(), 50)
                else:
                    self.assertGreater((rewards > 0.5).sum(), 20)
                    self.assertGreater((rewards < 1e-3).sum(), 20)

    def test_trajectory_with_dm_control_physics(self):
        """20 ms Euler steps under the same open-loop controls reproduce dm_control's env.step."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = euler_config(name)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
                dm_env = self.dm[name]
                physics = dm_env.physics
                gear = GEAR[DM_TASKS[name][0]]
                limit = torque_bound(config)
                rng = np.random.default_rng(4)
                for _ in range(3):
                    if is_reacher(name):
                        radius, angle = rng.uniform(0.05, 0.2), rng.uniform(0.0, 2.0 * np.pi)
                        start = np.array([
                            rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0),
                            radius * np.sin(angle), radius * np.cos(angle),
                            rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                        ])
                    else:
                        start = np.concatenate((rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.1, 0.1, 2)))
                    controls = rng.uniform(-0.5 * limit, 0.5 * limit, (50, 2))
                    env.reset(options={"state": start})
                    dm_env.reset()
                    with physics.reset_context():
                        physics.data.qpos[:] = start[:2]
                        physics.data.qvel[:] = start[-2:]
                        if is_reacher(name):
                            physics.named.model.geom_pos["target", :2] = start[2:4]
                    for torque in controls:
                        transition = env.step_dt(torque, 0.02)
                        timestep = dm_env.step(torque / gear)
                        self.assertEqual(env.last_interval["physics_steps"], 1)
                        self.assertFalse(transition["terminated"] or transition["truncated"])
                        np.testing.assert_allclose(env.data.qpos, physics.data.qpos, rtol=0, atol=1e-6)
                        np.testing.assert_allclose(env.data.qvel, physics.data.qvel, rtol=0, atol=1e-6)
                        state = transition["next_state"]
                        positions = np.array(physics.data.qpos)
                        if is_reacher(name):
                            positions = wrap(positions)
                            np.testing.assert_array_equal(state[2:4], start[2:4])
                            np.testing.assert_array_equal(env.context, start[2:4])
                            self.assertLess(abs(physics.data.qpos[1]), WRIST_SAFE)
                        else:
                            self.assertLess(np.max(np.abs(physics.data.qpos)), POSITION_SAFE)
                        np.testing.assert_allclose(state[:2], positions, rtol=0, atol=1e-6)
                        self.assertAlmostEqual(env.reward_rate(state, torque), timestep.reward, places=10)
                    self.assertAlmostEqual(env.time, 1.0, places=12)

    def test_reset_law(self):
        """Start states over 400 resets match dm_control's initialize_episode in law."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                ours = np.array([env.reset()[0] for _ in range(400)])
                dm_env = load_dm(name)
                physics = dm_env.physics
                theirs = []
                for _ in range(400):
                    dm_env.reset()
                    parts = [np.array(physics.data.qpos)]
                    if is_reacher(name):
                        parts.append(np.array(physics.named.model.geom_pos["target", :2]))
                    parts.append(np.array(physics.data.qvel))
                    theirs.append(np.concatenate(parts))
                theirs = np.array(theirs)
                np.testing.assert_array_equal(ours[:, -2:], 0.0)
                np.testing.assert_array_equal(theirs[:, -2:], 0.0)
                reset = config["reset_parameters"]
                if is_reacher(name):
                    bounds = [reset["shoulder_range"], reset["wrist_range"]]
                    for state in ours:
                        radius = np.hypot(state[2], state[3])
                        self.assertTrue(reset["target_radius_low"] <= radius <= reset["target_radius_high"])
                    columns = {
                        "shoulder": (ours[:, 0], theirs[:, 0]),
                        "wrist": (ours[:, 1], theirs[:, 1]),
                        "target_x": (ours[:, 2], theirs[:, 2]),
                        "target_y": (ours[:, 3], theirs[:, 3]),
                        "target_radius": (np.hypot(ours[:, 2], ours[:, 3]), np.hypot(theirs[:, 2], theirs[:, 3])),
                    }
                    self.assertFalse(any(env.failed(state) for state in ours))
                else:
                    bounds = [reset["position_range"]] * 2
                    columns = {"x": (ours[:, 0], theirs[:, 0]), "y": (ours[:, 1], theirs[:, 1])}
                for index, bound in enumerate(bounds):
                    self.assertLess(np.max(np.abs(ours[:, index])), bound)
                    self.assertGreater(np.max(np.abs(ours[:, index])), 0.95 * bound)
                for label, (left, right) in columns.items():
                    # sup |F - G| below the 0.1 % critical value 1.95 sqrt(2 / 400).
                    self.assertLess(ks_statistic(left, right), 0.138, label)

    def test_compiled_inertials_are_dm_controls(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                theirs = self.dm[name].physics.model.ptr
                for body in range(1, env.model.nbody):
                    label = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body)
                    other = mujoco.mj_name2id(theirs, mujoco.mjtObj.mjOBJ_BODY, label)
                    self.assertGreaterEqual(other, 0, label)
                    np.testing.assert_allclose(env.model.body_mass[body], theirs.body_mass[other], rtol=1e-12)
                    np.testing.assert_allclose(env.model.body_ipos[body], theirs.body_ipos[other], atol=1e-15)
                    np.testing.assert_allclose(env.model.body_inertia[body], theirs.body_inertia[other], rtol=1e-12)
                np.testing.assert_allclose(env.model.dof_damping, theirs.dof_damping, rtol=0)
                if is_reacher(name):
                    ours_chain = dmc_reacher.reacher_chain(env.model, config)
                    dm_chain = dmc_reacher.reacher_chain(self.dm[name].physics, config)
                    for field in ("link_masses", "link_com_distances", "link_inertias", "link_lengths", "joint_damping"):
                        np.testing.assert_allclose(getattr(ours_chain, field), getattr(dm_chain, field), rtol=1e-12)
                    self.assertEqual(ours_chain.gravity, 0.0)
                    np.testing.assert_allclose(ours_chain.input_matrix, np.eye(2))
                    np.testing.assert_allclose(dm_chain.input_matrix, 0.05 * np.eye(2))
                    # The welded finger is merged into the hand.
                    plant = config["plant_parameters"]
                    self.assertAlmostEqual(ours_chain.link_masses[1], 0.0356047 + 0.00418879, places=6)
                    self.assertEqual(ours_chain.link_lengths, (plant["arm_length"], plant["finger_offset"]))


# ------------------------------------------------------------------- task tensors


class TestTaskTensors(unittest.TestCase):
    def limit_states(self, config, rng, states):
        """Put 50 rows past the joint limits and four exactly on them."""
        states = states.copy()
        if config["name"] in REACHERS:
            limit = math.radians(config["plant_parameters"]["wrist_range_degrees"])
            states[:50, 1] = rng.choice([-1.0, 1.0], 50) * rng.uniform(limit, np.pi, 50)
            states[50:54, 1] = [limit, -limit, limit, -limit]
        else:
            limit = config["plant_parameters"]["slide_range"]
            axes = rng.integers(0, 2, 50)
            states[np.arange(50), axes] = rng.choice([-1.0, 1.0], 50) * rng.uniform(limit, 1.3 * limit, 50)
            states[50:54, :2] = [[limit, 0.0], [0.1, -limit], [-limit, limit], [0.0, limit]]
        return states

    def test_agreement_and_no_failure_set(self):
        """Rewards agree; since 2026-11-10 the limits are soft constraints and there is no
        failure set: the mask is all False past and on the limits, H = 0 and B = 1."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                self.assertEqual(config["failure_limits"], {})
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(5)
                states = self.limit_states(config, rng, random_states(config, rng, 200))
                limit = torque_bound(config)
                torques = rng.uniform(-1.2 * limit, 1.2 * limit, (200, 2))
                rates = tasks.reward_rate_tensor(torch.as_tensor(states), torch.as_tensor(torques), config)
                mask = tasks.failure_mask_tensor(torch.as_tensor(states), config)
                self.assertEqual(rates.shape, (200, 1))
                self.assertEqual(mask.shape, (200,))
                self.assertEqual(mask.sum().item(), 0)
                for i in range(200):
                    self.assertLessEqual(abs(rates[i, 0].item() - env.reward_rate(states[i], torques[i])), 1e-10)
                    self.assertFalse(env.failed(states[i]))
                payoff = tasks.terminal_payoff_tensor(torch.as_tensor(states), config)
                self.assertEqual(payoff.shape, (200, 1))
                self.assertEqual(env.terminal_payoff(states[0]), 0.0)
                self.assertTrue(torch.equal(payoff, torch.zeros(200, 1, dtype=torch.float64)))
                self.assertEqual(testbed_spec(name).terminal_payoff_value(config), 0.0)

                lift, vanishing = tasks.terminal_boundary_functions(config)
                tensor = torch.as_tensor(states)
                self.assertTrue(torch.equal(vanishing(tensor), torch.ones(200, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(lift(tensor), payoff))

    def test_rates_in_unit_interval(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                rng = np.random.default_rng(6)
                states = torch.as_tensor(random_states(config, rng, 500))
                torques = torch.as_tensor(rng.uniform(-1.0, 1.0, (500, 2)) * torque_bound(config))
                rates = tasks.reward_rate_tensor(states, torques, config)
                self.assertGreaterEqual(rates.min().item(), 0.0)
                self.assertLessEqual(rates.max().item(), 1.0)

    def test_point_mass_reward_formula(self):
        config = table_config("point_mass_easy")
        env = tasks.make_env(config)
        at_target = np.array([0.01, 0.0, 0.0, 0.0])
        self.assertEqual(env.reward_rate(at_target, np.zeros(2)), 1.0)
        self.assertAlmostEqual(env.reward_rate(at_target, np.array([0.1, 0.0])), 0.9, places=14)
        self.assertAlmostEqual(env.reward_rate(at_target, np.array([0.05, -0.05])), (4.0 + 0.75) / 5.0, places=14)
        at_margin = np.array([0.0, 0.03, 0.0, 0.0])
        self.assertAlmostEqual(env.reward_rate(at_margin, np.zeros(2)), 0.1, places=14)


# --------------------------------------------------------------------------- context


class TestReacherContext(unittest.TestCase):
    def target_geom(self, env):
        return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "target")

    def test_reset_draws_and_applies_the_target(self):
        for name in REACHERS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=3)
                state, _ = env.reset(seed=3)
                target = env.context
                self.assertEqual(env.target.tolist(), target.tolist())
                np.testing.assert_array_equal(state[2:4], target)
                self.assertTrue(0.05 <= np.hypot(*target) <= 0.2)
                geom = self.target_geom(env)
                np.testing.assert_array_equal(env.model.geom_pos[geom, :2], target)
                np.testing.assert_array_equal(env.data.geom_xpos[geom, :2], target)
                reach = config["reward_parameters"]["reach_radius"]
                self.assertAlmostEqual(env.model.geom_size[geom, 0], reach - 0.01, places=15)
                self.assertEqual(env.model.nq, 2)
                other, _ = env.reset()
                self.assertFalse(np.array_equal(other[2:4], target))
                again, _ = env.reset(seed=3)
                np.testing.assert_array_equal(again, state)

    def test_context_is_constant_over_an_episode(self):
        for name in REACHERS:
            with self.subTest(testbed=name):
                env = tasks.make_env(table_config(name), seed=4)
                state, _ = env.reset(seed=4)
                target = state[2:4].copy()
                rng = np.random.default_rng(4)
                for _ in range(40):
                    decision = rng.uniform(env.action_space.low, env.action_space.high)
                    transition = env.step_dt(decision, 0.02)
                    np.testing.assert_array_equal(transition["next_state"][2:4], target)
                    if env.done:
                        break
                np.testing.assert_array_equal(env.context, target)
                np.testing.assert_array_equal(env.data.geom_xpos[self.target_geom(env), :2], target)
                self.assertFalse(np.array_equal(env.observe()[:2], state[:2]))

    def test_explicit_state_and_snapshot_carry_the_context(self):
        env = tasks.make_env(table_config("reacher_easy"), seed=0)
        first = np.array([0.3, -0.4, 0.1, -0.05, 0.2, 0.0])
        state, _ = env.reset(options={"state": first})
        np.testing.assert_array_equal(state, first)
        snapshot = env.snapshot()
        env.step_dt(np.zeros(6), 0.02)
        env.reset(options={"state": np.array([0.0, 0.0, -0.15, 0.12, 0.0, 0.0])})
        geom = self.target_geom(env)
        np.testing.assert_array_equal(env.model.geom_pos[geom, :2], [-0.15, 0.12])
        env.restore(snapshot)
        np.testing.assert_array_equal(env.observe(), first)
        np.testing.assert_array_equal(env.model.geom_pos[geom, :2], [0.1, -0.05])
        np.testing.assert_array_equal(env.data.geom_xpos[geom, :2], [0.1, -0.05])
        with self.assertRaises(ValueError):
            env.reset(options={"state": np.zeros(4)})

    def test_drift_ignores_the_context(self):
        config = table_config("reacher_hard")
        env = tasks.make_env(config, seed=0)
        drift, _ = tasks.oracle_callbacks(config)
        state = np.array([0.7, -1.1, 0.12, 0.03, 2.0, -3.0])
        moved = state.copy()
        moved[2:4] = [-0.18, 0.02]
        torque = np.array([0.02, -0.04])
        for function in (env.mujoco_drift, env.oracle_drift):
            np.testing.assert_array_equal(function(state, torque), function(moved, torque))
            np.testing.assert_array_equal(function(state, torque)[2:4], [0.0, 0.0])
        both = drift(torch.as_tensor(np.stack((state, moved))), torch.as_tensor(np.stack((torque, torque))))
        self.assertTrue(torch.equal(both[0], both[1]))

    def test_reward_reads_the_context(self):
        for name in REACHERS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config)
                reach = config["reward_parameters"]["reach_radius"]
                angles = np.array([0.4, 0.9])
                tip = finger(config, angles)
                torque = np.zeros(2)
                cases = (
                    (tip, 1.0),
                    (tip + [0.0, 0.9 * reach], 1.0),
                    (tip + [1.1 * reach, 0.0], 0.0),
                    (-tip, 0.0),
                )
                for target, expected in cases:
                    state = np.concatenate((angles, target, [0.0, 0.0]))
                    self.assertEqual(env.reward_rate(state, torque), expected)
                    tensor = tasks.reward_rate_tensor(torch.as_tensor(state[None]), torch.as_tensor(torque[None]), config)
                    self.assertEqual(tensor.item(), expected)


# ---------------------------------------------------------------------------- table


class TestTableRows(unittest.TestCase):
    def test_rows_build_with_their_declared_task_cells(self):
        implemented = set(registered_testbeds(implemented_only=True))
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                entry = table_entry(name)
                spec = testbed_spec(name)
                self.assertTrue(spec.implemented)
                self.assertIn(name, implemented)
                self.assertIsNone(spec.upright)
                self.assertEqual(set(spec.settings_keys), set(entry.config["testbed_settings"]))
                config = table_config(name)
                self.assertEqual(config["name"], name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, config), [])
                protocol = run_ph_sac.load_protocol_from_table(ENV_IDS[name])
                built = protocol["matched"]["testbed_config"]
                self.assertEqual(built["name"], name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, built), [])
                self.assertEqual(built["state_spec"].get("context_dim", 0), 2 if is_reacher(name) else 0)

    def test_list_reads_builds(self):
        source = table.table_path(table.DEFAULT_HYPERPARAMS_DIR)
        with source.open("r", newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["env_id"] in ENV_IDS.values() and row["mode"] == "top"]
        self.assertEqual(len(rows), 3)
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / table.TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table.TABLE_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            lines = run_ph_sac.list_table(tmp)
        self.assertEqual(len(lines), 4)
        for line in lines[1:]:
            self.assertIn(line.split()[0], ENV_IDS.values())
            self.assertTrue(line.endswith("builds"), line)

    def test_factories_and_environments_check_their_settings(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                settings = table_entry(name).config["testbed_settings"]
                spec = testbed_spec(name)
                # failure_limits is empty since 2026-11-10 (no failure set).
                self.assertEqual(settings["failure_limits"], {})
                for section in ("plant_parameters", "reward_parameters", "reset_parameters"):
                    broken = copy.deepcopy(settings)
                    key = sorted(broken[section])[0]
                    del broken[section][key]
                    with self.assertRaisesRegex(ValueError, key):
                        spec.config_factory(**broken)
                    config = table_config(name)
                    del config[section][key]
                    with self.assertRaisesRegex(ValueError, key):
                        tasks.make_env(config)

    def test_reacher_masses_must_round_the_compiled_ones(self):
        config = table_config("reacher_easy")
        config["plant_parameters"] = dict(config["plant_parameters"], arm_mass=0.05)
        with self.assertRaisesRegex(ValueError, "arm_mass"):
            tasks.make_env(config)
        with self.assertRaisesRegex(ValueError, "arm_mass"):
            tasks.oracle_callbacks(config)


# ---------------------------------------------------------------------------- smoke


class TestSmokeEpisodes(unittest.TestCase):
    def test_feedback_episode(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                self.assertEqual(env.action_space.shape, (6,))
                state, info = env.reset(seed=11)
                self.assertFalse(env.failed(state))
                rng = np.random.default_rng(11)
                steps, total = 0, 0.0
                while not env.done:
                    decision = rng.uniform(env.action_space.low, env.action_space.high)
                    next_state, reward, terminated, truncated, info = env.step(decision)
                    self.assertAlmostEqual(info["dt"], 0.02, places=12)
                    self.assertTrue(np.isfinite(next_state).all())
                    self.assertTrue(np.isfinite(reward))
                    total += env.last_undiscounted_reward
                    steps += 1
                    # No failure set (soft joint limits): episodes end at the time limit.
                    self.assertFalse(terminated)
                self.assertTrue(truncated)
                self.assertEqual(steps, 1000)
                self.assertAlmostEqual(env.time, 20.0, places=9)
                self.assertGreaterEqual(total, 0.0)
                self.assertLessEqual(total, env.time + 1e-9)

    def test_torque_mode_and_resets(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                np.testing.assert_array_equal(env.action_space.high, config["controller"]["torque_high"])
                first, _ = env.reset(seed=21)
                rng = np.random.default_rng(21)
                for _ in range(25):
                    torque = rng.uniform(env.action_space.low, env.action_space.high)
                    next_state, _, terminated, truncated, _ = env.step(torque)
                    np.testing.assert_array_equal(env.data.ctrl, torque)
                    self.assertFalse(truncated)
                    if terminated:
                        break
                self.assertGreater(env.time, 0.0)
                second, _ = env.reset()
                self.assertFalse(np.array_equal(first, second))
                if is_reacher(name):
                    self.assertFalse(np.array_equal(first[2:4], second[2:4]))
                    np.testing.assert_array_equal(env.context, second[2:4])
                with Float64():
                    plant = tasks.make_oracle_plant(env)
                    states = np.stack((first, second))
                    actions = rng.uniform(env.action_space.low, env.action_space.high, (2, 2))
                    drift = plant.drift(torch.as_tensor(states), torch.as_tensor(actions)).numpy()
                expected = np.stack([env.oracle_drift(s, a) for s, a in zip(states, actions)])
                np.testing.assert_allclose(drift, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
