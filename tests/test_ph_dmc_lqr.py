"""dm_control's lqr ports (lqr_2_1, lqr_6_2) against dm_control and the porting contract.

Checks (environment/ph_sac/dmc/README.md, "Validation"): the factory against its table
row and dm_control's compiled stiffness draw; the analytic mass matrix against MuJoCo;
the oracle against our MuJoCo, and our MuJoCo against dm_control's forward dynamics,
reward and 30 ms Euler trajectory; the tensor task functions against the environment;
torch.func through make_oracle_plant; the reset law; smoke episodes; the optional irregular
clock (TestIrregularClock); and a diagnostic of the table's +-20 N motor limit against the
discounted Riccati optimum.

dm_control's spheres weigh 4.188790204786391 kg, which the table now writes exactly. The comparisons
with dm_control build our config with dm_control's compiled mass (dm_control_config).
"""

import contextlib
import copy
import io
import math
import unittest

import mujoco
import numpy as np
import scipy.linalg
import torch

import common.ph_config as ph_config
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import _build_testbed, load_protocol_from_table, main, protocol_from_config
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, dmc_clock, fixed_clock
from common.ph_dmc.lqr import (
    DM_CONTROL_SEED0_STIFFNESS,
    LQR_TORQUE_LIMIT,
    lqr_2_1_config,
    lqr_6_2_config,
)
from environment.ph_sac.base import validated_state_spec
from environment.ph_sac.dmc.lqr import (
    LQREnv,
    lqr_inverse_mass_matrix,
    lqr_linear_dynamics,
    lqr_mass_matrix,
    lqr_state_matrices,
)
from environment.ph_sac.registry import testbed_spec
from environment.ph_sac.tasks import (
    failure_mask_tensor,
    make_env,
    make_oracle_plant,
    oracle_callbacks,
    reward_rate_tensor,
    terminal_boundary_functions,
    terminal_payoff_tensor,
)
from models.ph_sac.rollouts import integration_grid

# (testbed, table env_id, n bodies, n actuated)
TASKS = (("lqr_2_1", "lqr-lqr_2_1", 2, 1), ("lqr_6_2", "lqr-lqr_6_2", 6, 2))
DM_CONTROL_MASS = 4.0 / 3.0 * math.pi * 0.1**3 * 1000.0


def table_settings(env_id):
    return copy.deepcopy(table.find_ph_entry(env_id, "top").config["testbed_settings"])


def table_config(testbed, env_id):
    """The config the table row builds (the compiled body mass, seed-0 stiffness)."""
    return testbed_spec(testbed).config_factory(**table_settings(env_id))


def load_dm_control(testbed):
    from dm_control import suite

    return suite.load("lqr", testbed, task_kwargs={"random": 0})


def dm_control_config(testbed, env_id, physics, **overrides):
    """The table config with dm_control's compiled mass and stiffness, then overrides."""
    settings = table_settings(env_id)
    model = physics.model.ptr
    settings["plant_parameters"]["body_mass"] = float(model.body_mass[1])
    settings["plant_parameters"]["stiffness"] = tuple(float(k) for k in model.jnt_stiffness)
    config = testbed_spec(testbed).config_factory(**settings)
    config.update(overrides)
    return config


def euler_config(config, physics_dt, duration):
    """dm_control's integration: config with an Euler physics step and fixed clock of physics_dt."""
    config = copy.deepcopy(config)
    config["physics_dt"] = physics_dt
    config["integrator"] = "Euler"
    config["reference_period"] = physics_dt
    config["episode_duration"] = duration
    config["clock"].update(
        physics_dt=physics_dt,
        reference_period=physics_dt,
        min_duration=physics_dt,
        max_duration=physics_dt,
    )
    return config


def random_states(n_bodies, rng, count, position=2.0, velocity=5.0):
    return np.concatenate(
        (
            rng.uniform(-position, position, (count, n_bodies)),
            rng.uniform(-velocity, velocity, (count, n_bodies)),
        ),
        axis=1,
    )


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def full_mass_matrix(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, matrix, data.qM)
    return matrix


class TestFactory(unittest.TestCase):
    def test_rows_build_the_declared_config(self):
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                entry = table.find_ph_entry(env_id, "top")
                settings = copy.deepcopy(entry.config["testbed_settings"])
                config = testbed_spec(testbed).config_factory(**settings)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, config), [])
                self.assertEqual(settings, entry.config["testbed_settings"])  # not mutated
                self.assertIs(validated_state_spec(config), config["state_spec"])
                ph_config._validate_testbed(config)
                plant = config["plant_parameters"]
                self.assertEqual(plant["stiffness"], DM_CONTROL_SEED0_STIFFNESS[:n_bodies])
                self.assertEqual(plant["body_mass"], DM_CONTROL_MASS)
                self.assertEqual(config["state_spec"]["periodic_position_indices"], ())
                self.assertEqual(
                    config["state_spec"]["actuated_position_indices"], tuple(range(n_actuated))
                )
                self.assertEqual(config["controller"]["torque_high"].tolist(), [20.0] * n_actuated)
                self.assertEqual(config["failure_limits"], {})

    def test_protocol_builds_and_list_reads_builds(self):
        for testbed, env_id, _, _ in TASKS:
            with self.subTest(testbed=testbed):
                protocol = load_protocol_from_table(env_id)
                self.assertEqual(protocol["matched"]["testbed_config"]["name"], testbed)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--list"])
        status = {line.split()[0]: line for line in output.getvalue().splitlines()[1:]}
        for _, env_id, _, _ in TASKS:
            self.assertTrue(status[env_id].endswith("builds"), status[env_id])

    def test_registration(self):
        for testbed, _, _, _ in TASKS:
            spec = testbed_spec(testbed)
            self.assertTrue(spec.implemented)
            self.assertIs(spec.env_class, LQREnv)
            self.assertIsNone(spec.upright)
            self.assertEqual(
                spec.settings_keys,
                ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters"),
            )
            # The motor limit and (TestIrregularClock) the irregular clock.
            self.assertEqual(spec.optional_settings_keys, ("torque_limit",) + CLOCK_SETTINGS_KEYS)

    def test_optional_torque_limit_rescales_the_box(self):
        """torque_limit (optional, since 2026-11-11, for the lqr_6_2 diagnostic) sets the
        motor and, by the box rule, stiffness up to 10 x limit / 1 m and damping up to 0.2
        s times that; without it every row keeps LQR_TORQUE_LIMIT (20 N: 200 and 40)."""
        for testbed, env_id, _, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                self.assertNotIn("torque_limit", table_settings(env_id))
                for limit, k_max in ((None, 200.0), (LQR_TORQUE_LIMIT, 200.0), (2.0, 20.0)):
                    settings = table_settings(env_id)
                    if limit is not None:
                        settings["torque_limit"] = limit
                    config = testbed_spec(testbed).config_factory(**settings)
                    controller = config["controller"]
                    expected = LQR_TORQUE_LIMIT if limit is None else limit
                    np.testing.assert_allclose(controller["torque_high"], [expected] * n_actuated)
                    np.testing.assert_allclose(controller["torque_low"], [-expected] * n_actuated)
                    np.testing.assert_allclose(
                        controller["decision_high"],
                        [1.0] * n_actuated + [k_max] * n_actuated + [0.2 * k_max] * n_actuated,
                    )
                    np.testing.assert_allclose(
                        controller["decision_low"],
                        [-1.0] * n_actuated + [0.0] * (2 * n_actuated),
                    )
                    ph_config._validate_testbed(config)
                    env = make_env(config, decision_mode="torque", seed=0)
                    np.testing.assert_allclose(env.action_space.high, [expected] * n_actuated)
                for bad in (0.0, -2.0, float("nan"), True):
                    settings = dict(table_settings(env_id), torque_limit=bad)
                    with self.assertRaisesRegex(ValueError, "torque_limit"):
                        testbed_spec(testbed).config_factory(**settings)

    def test_default_stiffness_is_dm_controls_seed_zero_draw(self):
        """The compiled draw, which RandomState(0) reproduces (stiffness, damping per body)."""
        for testbed, env_id, n_bodies, _ in TASKS:
            with self.subTest(testbed=testbed):
                model = load_dm_control(testbed).physics.model.ptr
                self.assertEqual(tuple(model.jnt_stiffness.tolist()), DM_CONTROL_SEED0_STIFFNESS[:n_bodies])
                random = np.random.RandomState(0)
                draws = []
                for _ in range(n_bodies):
                    draws.append(random.uniform(15, 25))
                    random.uniform(0, 0)
                self.assertEqual(tuple(draws), DM_CONTROL_SEED0_STIFFNESS[:n_bodies])
                ours = make_env(table_config(testbed, env_id)).model
                self.assertEqual(tuple(ours.jnt_stiffness.tolist()), DM_CONTROL_SEED0_STIFFNESS[:n_bodies])
                np.testing.assert_array_equal(ours.dof_damping, np.zeros(n_bodies))
                np.testing.assert_array_equal(model.dof_damping, np.zeros(n_bodies))

    def test_masses(self):
        """dm_control's sphere mass, which the table states exactly."""
        for testbed, env_id, _, _ in TASKS:
            physics = load_dm_control(testbed).physics
            dm_mass = physics.model.ptr.body_mass[1:]
            np.testing.assert_allclose(dm_mass, DM_CONTROL_MASS, rtol=1e-14)
            ours = make_env(dm_control_config(testbed, env_id, physics)).model
            np.testing.assert_array_equal(ours.body_mass, physics.model.ptr.body_mass)
            table_model = make_env(table_config(testbed, env_id)).model
            np.testing.assert_array_equal(table_model.body_mass[1:], DM_CONTROL_MASS)

    def test_explicit_stiffness_and_bad_settings(self):
        settings = table_settings("lqr-lqr_2_1")
        factory = testbed_spec("lqr_2_1").config_factory
        settings["plant_parameters"]["stiffness"] = [16.0, 24.0]
        config = factory(**settings)
        self.assertEqual(config["plant_parameters"]["stiffness"], (16.0, 24.0))
        self.assertEqual(make_env(config).model.jnt_stiffness.tolist(), [16.0, 24.0])
        # The form a table dict cell carries: "k1,k2" survives the cell round trip as text.
        plant = dict(table_settings("lqr-lqr_2_1")["plant_parameters"])
        plant["stiffness"] = ",".join(repr(k) for k in DM_CONTROL_SEED0_STIFFNESS[:2])
        column = next(c for c in table.CONFIG_COLUMNS if c.name == "env_plant_parameters")
        present, parsed = table.parse_cell(column.kind, table.format_cell(column, plant))
        self.assertTrue(present)
        self.assertEqual(parsed, plant)
        settings = table_settings("lqr-lqr_2_1")
        settings["plant_parameters"] = parsed
        self.assertEqual(factory(**settings)["plant_parameters"]["stiffness"], DM_CONTROL_SEED0_STIFFNESS[:2])
        bad = [
            ("plant_parameters", "stiffness", [16.0]),
            ("plant_parameters", "stiffness", [16.0, 26.0]),
            ("plant_parameters", "stiffness", "16.0,24.0,20.0"),
            ("plant_parameters", "n_bodies", 3),
            ("plant_parameters", "n_actuated", 2),
            ("failure_limits", "position", 1.0),
        ]
        for section, key, value in bad:
            with self.subTest(key=key, value=value):
                changed = table_settings("lqr-lqr_2_1")
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    factory(**changed)
        for section, key in (("plant_parameters", "body_mass"), ("reward_parameters", "control_cost"),
                             ("reset_parameters", "position_radius")):
            with self.subTest(missing=key):
                changed = table_settings("lqr-lqr_2_1")
                del changed[section][key]
                with self.assertRaisesRegex(ValueError, key):
                    factory(**changed)


class TestDynamics(unittest.TestCase):
    def test_mass_matrix(self):
        """M_ij = m (n - max(i, j)) against mj_fullM of ours and dm_control's; M^-1 closed form."""
        for testbed, env_id, n_bodies, _ in TASKS:
            with self.subTest(testbed=testbed):
                physics = load_dm_control(testbed).physics
                expected = lqr_mass_matrix(n_bodies, DM_CONTROL_MASS)
                self.assertLess(np.abs(full_mass_matrix(physics.model.ptr) - expected).max(), 1e-12)
                ours = make_env(dm_control_config(testbed, env_id, physics)).model
                self.assertLess(np.abs(full_mass_matrix(ours) - expected).max(), 1e-12)
                table_model = make_env(table_config(testbed, env_id)).model
                self.assertLess(
                    np.abs(full_mass_matrix(table_model) - lqr_mass_matrix(n_bodies, DM_CONTROL_MASS)).max(), 1e-12
                )
                inverse = lqr_inverse_mass_matrix(n_bodies, DM_CONTROL_MASS)
                np.testing.assert_allclose(
                    inverse @ lqr_mass_matrix(n_bodies, DM_CONTROL_MASS), np.eye(n_bodies), atol=1e-13
                )
                np.testing.assert_allclose(
                    inverse, np.linalg.inv(lqr_mass_matrix(n_bodies, DM_CONTROL_MASS)), atol=1e-13
                )

    def test_oracle_matches_mujoco(self):
        """env.oracle_drift and the tensor callbacks against env.mujoco_drift, 200 states."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                config = table_config(testbed, env_id)
                env = make_env(config, seed=0)
                drift, diffusion = oracle_callbacks(config)
                rng = np.random.default_rng(0)
                states = random_states(n_bodies, rng, 200)
                torques = rng.uniform(-LQR_TORQUE_LIMIT, LQR_TORQUE_LIMIT, (200, n_actuated))
                tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                for state, torque, row in zip(states, torques, tensor):
                    truth = env.mujoco_drift(state, torque)
                    self.assertLess(relative_error(env.oracle_drift(state, torque), truth), 1e-8)
                    self.assertLess(relative_error(row, truth), 1e-8)
                self.assertTrue(torch.equal(
                    diffusion(torch.as_tensor(states)), torch.zeros(200, 2 * n_bodies, 1, dtype=torch.float64)
                ))
                np.testing.assert_array_equal(env.oracle_diffusion(states[0]), np.zeros((2 * n_bodies, 1)))
                np.testing.assert_array_equal(env.oracle_input_matrix(), np.eye(n_bodies)[:, :n_actuated])

    def test_mujoco_matches_dm_control(self):
        """env.mujoco_drift against dm_control's qacc at the same state and control."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                physics = load_dm_control(testbed).physics
                env = make_env(dm_control_config(testbed, env_id, physics), seed=0)
                rng = np.random.default_rng(1)
                for state in random_states(n_bodies, rng, 200):
                    torque = rng.uniform(-LQR_TORQUE_LIMIT, LQR_TORQUE_LIMIT, n_actuated)
                    physics.data.qpos[:] = state[:n_bodies]
                    physics.data.qvel[:] = state[n_bodies:]
                    physics.data.ctrl[:] = torque  # gear 1
                    physics.forward()
                    ours = env.mujoco_drift(state, torque)
                    self.assertLess(relative_error(ours[n_bodies:], physics.data.qacc), 1e-8)
                    self.assertLess(relative_error(env.oracle_drift(state, torque)[n_bodies:], physics.data.qacc), 1e-8)

    def test_trajectory_matches_dm_control(self):
        """dm_control's 30 ms Euler steps against ours with the same physics, 200 steps (6 s)."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                dm_env = load_dm_control(testbed)
                physics = dm_env.physics
                self.assertEqual(physics.model.ptr.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
                self.assertEqual(physics.model.ptr.opt.timestep, 0.03)
                config = euler_config(dm_control_config(testbed, env_id, physics), 0.03, 6.0)
                env = make_env(config, decision_mode="torque", seed=0)
                rng = np.random.default_rng(2)
                start = random_states(n_bodies, rng, 1, position=1.0, velocity=1.0)[0]
                dm_env.reset()
                with physics.reset_context():
                    physics.data.qpos[:] = start[:n_bodies]
                    physics.data.qvel[:] = start[n_bodies:]
                observation, _ = env.reset(options={"state": start})
                np.testing.assert_array_equal(observation, start)
                controls = rng.uniform(-LQR_TORQUE_LIMIT, LQR_TORQUE_LIMIT, (200, n_actuated))
                for step, control in enumerate(controls):
                    time_step = dm_env.step(control)
                    transition = env.step_dt(control, 0.03)
                    expected = np.concatenate((physics.data.qpos, physics.data.qvel))
                    self.assertLess(np.abs(transition["next_state"] - expected).max(), 1e-6, step)
                    self.assertAlmostEqual(
                        env.reward_rate(transition["next_state"], control), time_step.reward, places=8
                    )
                    self.assertFalse(time_step.last())
                self.assertTrue(env.done)

    def test_torch_func_and_oracle_plant(self):
        """drift_jacobian is the constant (A, B), equals finite differences, drift equals the oracle."""
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            for testbed, env_id, n_bodies, n_actuated in TASKS:
                with self.subTest(testbed=testbed):
                    config = table_config(testbed, env_id)
                    env = make_env(config, seed=0)
                    plant = make_oracle_plant(env)
                    rng = np.random.default_rng(3)
                    states = torch.as_tensor(random_states(n_bodies, rng, 4))
                    actions = torch.as_tensor(rng.uniform(-20.0, 20.0, (4, n_actuated)))
                    values, by_state, by_action = plant.drift_jacobian(states, actions)
                    state_matrix, input_matrix = lqr_state_matrices(config)
                    for i in range(4):
                        np.testing.assert_allclose(by_state[i].numpy(), state_matrix, atol=1e-12)
                        np.testing.assert_allclose(by_action[i].numpy(), input_matrix, atol=1e-12)
                        np.testing.assert_allclose(
                            values[i].numpy(),
                            env.oracle_drift(states[i].numpy(), actions[i].numpy()),
                            atol=1e-12,
                        )
                    epsilon = 1e-6
                    for j in range(2 * n_bodies):
                        bump = torch.zeros_like(states)
                        bump[:, j] = epsilon
                        numeric = (plant.drift(states + bump, actions) - plant.drift(states - bump, actions)) / (
                            2.0 * epsilon
                        )
                        self.assertLess((by_state[:, :, j] - numeric).abs().max().item(), 1e-6)
                    np.testing.assert_allclose(
                        plant.drift(states, actions).numpy(),
                        np.stack([env.oracle_drift(s, a) for s, a in zip(states.numpy(), actions.numpy())]),
                        atol=1e-12,
                    )
                    self.assertEqual(plant.input_matrix().tolist(), env.oracle_input_matrix().tolist())
                    self.assertEqual(plant.state_spec, env.state_spec)
        finally:
            torch.set_default_dtype(previous)


class TestTask(unittest.TestCase):
    def test_reward_matches_dm_control(self):
        """Tensor, env and dm_control's get_reward at the same state and control."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                dm_env = load_dm_control(testbed)
                physics = dm_env.physics
                config = dm_control_config(testbed, env_id, physics)
                env = make_env(config, seed=0)
                rng = np.random.default_rng(4)
                states = random_states(n_bodies, rng, 200)
                torques = rng.uniform(-LQR_TORQUE_LIMIT, LQR_TORQUE_LIMIT, (200, n_actuated))
                rates = reward_rate_tensor(torch.as_tensor(states), torch.as_tensor(torques), config)
                self.assertEqual(rates.shape, (200, 1))
                for state, torque, rate in zip(states, torques, rates[:, 0].tolist()):
                    physics.data.qpos[:] = state[:n_bodies]
                    physics.data.qvel[:] = state[n_bodies:]
                    physics.data.ctrl[:] = torque
                    physics.forward()
                    expected = dm_env.task.get_reward(physics)
                    self.assertAlmostEqual(env.reward_rate(state, torque), expected, places=10)
                    self.assertAlmostEqual(rate, expected, places=10)
                # 1 at rest at the origin without control; negative far out.
                self.assertEqual(env.reward_rate(np.zeros(2 * n_bodies), np.zeros(n_actuated)), 1.0)
                self.assertLess(env.reward_rate(np.full(2 * n_bodies, 2.0), np.zeros(n_actuated)), 0.0)

    def test_tensor_agreement(self):
        """As check_agreement: reward, the empty failure set, payoff 0 and the lift H = 0, B = 1."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                config = table_config(testbed, env_id)
                env = make_env(config, seed=0)
                rng = np.random.default_rng(5)
                states = random_states(n_bodies, rng, 200, position=50.0, velocity=100.0)
                torques = rng.uniform(-30.0, 30.0, (200, n_actuated))
                tensor_states = torch.as_tensor(states)
                rates = reward_rate_tensor(tensor_states, torch.as_tensor(torques), config)
                mask = failure_mask_tensor(tensor_states, config)
                self.assertEqual(mask.shape, (200,))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertFalse(mask.any().item())
                payoff = terminal_payoff_tensor(tensor_states, config)
                self.assertTrue(torch.equal(payoff, torch.zeros(200, 1, dtype=torch.float64)))
                lift, vanishing = terminal_boundary_functions(config)
                self.assertTrue(torch.equal(lift(tensor_states), torch.zeros(200, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(vanishing(tensor_states), torch.ones(200, 1, dtype=torch.float64)))
                for i in range(200):
                    self.assertAlmostEqual(rates[i, 0].item(), env.reward_rate(states[i], torques[i]), places=10)
                    self.assertFalse(env.failed(states[i]))
                    self.assertEqual(env.terminal_payoff(states[i]), 0.0)

    def test_reset_law_matches_dm_control(self):
        """q uniform on the sphere of radius sqrt(2), velocities 0, in ours and dm_control's."""
        for testbed, env_id, n_bodies, _ in TASKS:
            with self.subTest(testbed=testbed):
                # dm_control's first reset is randn(n) normalized, after the 2n model draws.
                dm_env = load_dm_control(testbed)
                dm_env.reset()
                random = np.random.RandomState(0)
                random.uniform(size=2 * n_bodies)
                unit = random.randn(n_bodies)
                np.testing.assert_allclose(
                    dm_env.physics.data.qpos, math.sqrt(2.0) * unit / np.linalg.norm(unit), atol=1e-15
                )
                env = make_env(table_config(testbed, env_id), seed=0)
                ours, theirs = [], []
                for seed in range(400):
                    observation, _ = env.reset(seed=seed)
                    ours.append(observation)
                    dm_env.reset()
                    theirs.append(np.concatenate((dm_env.physics.data.qpos, dm_env.physics.data.qvel)))
                for samples in (np.array(ours), np.array(theirs)):
                    positions, velocities = samples[:, :n_bodies], samples[:, n_bodies:]
                    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), math.sqrt(2.0), rtol=1e-12)
                    np.testing.assert_array_equal(velocities, 0.0)
                    self.assertLess(np.abs(positions.mean(axis=0)).max(), 0.2)
                    np.testing.assert_allclose((positions**2).mean(axis=0), 2.0 / n_bodies, atol=0.15)
                    second = positions.T @ positions / len(positions)
                    self.assertLess(np.abs(second - np.diag(np.diag(second))).max(), 0.15)
                given = random_states(n_bodies, np.random.default_rng(6), 1)[0]
                np.testing.assert_array_equal(env.reset(options={"state": given})[0], given)

    def test_smoke_episodes(self):
        """A full 10 s feedback episode (333 decisions of 30 ms, one of 10 ms) and torque steps."""
        for testbed, env_id, n_bodies, n_actuated in TASKS:
            with self.subTest(testbed=testbed):
                config = table_config(testbed, env_id)
                env = make_env(config, seed=0)
                observation, _ = env.reset(seed=1)
                decision = 0.5 * (env.action_space.low + env.action_space.high)
                durations, done = [], False
                while not done:
                    observation, reward, terminated, truncated, info = env.step(decision)
                    self.assertFalse(terminated)
                    self.assertTrue(math.isfinite(reward))
                    durations.append(info["dt"])
                    done = terminated or truncated
                self.assertTrue(truncated)
                self.assertEqual(len(durations), 334)
                self.assertAlmostEqual(durations[-1], 0.01, places=12)
                self.assertAlmostEqual(env.time, 10.0, places=9)
                self.assertTrue(np.isfinite(observation).all())

                env = make_env(config, decision_mode="torque", seed=0)
                env.reset(seed=2)
                np.testing.assert_array_equal(env.action_space.high, [20.0] * n_actuated)
                rng = np.random.default_rng(7)
                for _ in range(20):
                    torque = rng.uniform(-20.0, 20.0, n_actuated)
                    observation, reward, terminated, truncated, _ = env.step(torque)
                    self.assertFalse(terminated or truncated)
                    self.assertTrue(np.isfinite(observation).all())
                # Torques beyond the limit are clipped to +-20 N.
                transition = env.step_dt(np.full(n_actuated, 50.0), 0.03)
                np.testing.assert_array_equal(transition["decision"], [20.0] * n_actuated)
                self.assertEqual(env.last_interval["saturated_fraction"], 1.0)


class TestIrregularClock(unittest.TestCase):
    """The optional clock settings CLOCK_SETTINGS_KEYS (common.ph_dmc._common.with_clock_settings),
    taken as the other dm_control ports take them, alone and with the optional torque_limit:
    the two-tailed clock on [T/5, 3T] = [6 ms, 90 ms] about the 30 ms reference period."""

    LAW = {"clock_kind": "two_tail_uniform", "tail_probability": 0.99, "tail_split": 0.9}
    MODULE_FACTORIES = {"lqr_2_1": lqr_2_1_config, "lqr_6_2": lqr_6_2_config}

    @staticmethod
    def settings(env_id, torque_limit=None, **clock):
        """The row's testbed_settings, plus torque_limit (unless None) and clock settings."""
        settings = table_settings(env_id)
        if torque_limit is not None:
            settings["torque_limit"] = torque_limit
        settings.update(clock)
        return settings

    def test_factories_take_the_clock_with_the_torque_limit(self):
        """Clock 6 to 90 ms; the box, motor and every other field are the fixed clock's (under
        2 N: setpoint +-1 m, stiffness 0-20 N/m, damping 0-4 N s/m). The run script's
        _build_testbed accepts the keys (it raised "unknown settings" before)."""
        for testbed, env_id, _, n_actuated in TASKS:
            factory = testbed_spec(testbed).config_factory
            for limit, k_max in ((None, 200.0), (2.0, 20.0)):
                with self.subTest(testbed=testbed, torque_limit=limit):
                    fixed = factory(**self.settings(env_id, limit))
                    self.assertEqual(fixed["clock"]["kind"], "fixed")
                    irregular = factory(**self.settings(env_id, limit, **self.LAW))
                    ph_config._validate_testbed(irregular)
                    clock = irregular["clock"]
                    self.assertEqual(clock, dmc_clock(0.03, 0.001, **self.LAW))
                    self.assertEqual(
                        (clock["kind"], clock["reference_period"], clock["physics_dt"]),
                        ("two_tail_uniform", 0.03, 0.001),
                    )
                    self.assertEqual((clock["min_duration"], clock["max_duration"]), (0.006, 0.09))
                    self.assertEqual(
                        (clock["tail_probability"], clock["tail_split"], clock["beta_shape"]),
                        (0.99, 0.9, None),
                    )
                    self.assertEqual(sorted(irregular), sorted(fixed))
                    for key in fixed:
                        if key not in ("clock", "state_spec", "controller"):
                            self.assertEqual(repr(irregular[key]), repr(fixed[key]), key)
                    for section in ("state_spec", "controller"):
                        self.assertEqual(sorted(irregular[section]), sorted(fixed[section]))
                        for key in fixed[section]:
                            np.testing.assert_array_equal(
                                irregular[section][key], fixed[section][key], err_msg=key
                            )
                    controller = irregular["controller"]
                    np.testing.assert_allclose(
                        controller["decision_high"],
                        [1.0] * n_actuated + [k_max] * n_actuated + [0.2 * k_max] * n_actuated,
                    )
                    np.testing.assert_allclose(
                        controller["decision_low"], [-1.0] * n_actuated + [0.0] * (2 * n_actuated)
                    )
                    expected_limit = LQR_TORQUE_LIMIT if limit is None else limit
                    np.testing.assert_allclose(controller["torque_high"], [expected_limit] * n_actuated)
                    built = _build_testbed(testbed, self.settings(env_id, limit, **self.LAW))
                    self.assertEqual(built["clock"], clock)
                    np.testing.assert_array_equal(
                        built["controller"]["decision_high"], controller["decision_high"]
                    )

    def test_protocol_builds_with_the_clock_and_its_model_grid(self):
        """The row's whole config with the clock settings and torque_limit 2 N passes
        validate_configuration at the row's 10 ms model step (no coarse step). Under the
        continuous refresh the model grid splits every interval the clock draws (6 to 90
        physics steps) into ceil(n / 10) equal substeps of at most 10 ms, as on the
        pendulum's irregular rows (10 ms model step, 4 ms minimum interval)."""
        for testbed, env_id, _, _ in TASKS:
            with self.subTest(testbed=testbed):
                raw = copy.deepcopy(table.find_ph_entry(env_id, "top").config)
                raw["testbed_settings"].update(self.LAW, torque_limit=2.0)
                protocol = protocol_from_config(raw, f"{env_id} with an irregular clock")
                config = protocol["matched"]["testbed_config"]
                self.assertEqual(config["name"], testbed)
                self.assertEqual(config["clock"], dmc_clock(0.03, 0.001, **self.LAW))
                self.assertEqual(config["controller"]["refresh_mode"], "continuous")
                for name in ("explicit_config", "implicit_config"):
                    self.assertEqual(protocol[name]["model_step"], 0.01)
                    self.assertIsNone(protocol[name].get("model_coarse_step"))
                lengths = list(range(6, 91))
                durations = torch.tensor([[n * 0.001] for n in lengths], dtype=torch.float64)
                counts, sizes = integration_grid(
                    durations, 0.01, physics_dt=0.001, controller_refresh_steps=None
                )
                self.assertEqual(counts.tolist(), [-(-n // 10) for n in lengths])
                for row, (n, count) in enumerate(zip(lengths, counts.tolist())):
                    self.assertLessEqual(float(sizes[row, :count].max()), 0.01 * (1.0 + 1e-12))
                    self.assertAlmostEqual(float(sizes[row].sum()), n * 0.001, places=12)

    def test_episode_under_the_irregular_clock(self):
        """A whole feedback episode (torque_limit 2 N, random decisions in the box) ends at
        10 s exactly; every interval is a whole number of physics steps, all but the last in
        [6 ms, 90 ms], the last (cut at the horizon) at most 90 ms; everything finite."""
        for testbed, env_id, _, _ in TASKS:
            with self.subTest(testbed=testbed):
                config = testbed_spec(testbed).config_factory(**self.settings(env_id, 2.0, **self.LAW))
                env = make_env(config, seed=0)
                np.testing.assert_array_equal(env.action_space.high, config["controller"]["decision_high"])
                observation, _ = env.reset(seed=1)
                rng = np.random.default_rng(10)
                steps, terminated, truncated = [], False, False
                while not env.done:
                    decision = rng.uniform(env.action_space.low, env.action_space.high)
                    observation, reward, terminated, truncated, info = env.step(decision)
                    self.assertFalse(terminated)
                    self.assertTrue(math.isfinite(reward))
                    self.assertTrue(np.isfinite(observation).all())
                    ratio = float(info["dt"]) / env.physics_dt
                    self.assertAlmostEqual(ratio, round(ratio), places=9)
                    steps.append(int(round(ratio)))
                self.assertTrue(truncated)
                self.assertEqual(sum(steps), 10000)
                self.assertEqual(env.remaining_steps, 0)
                self.assertAlmostEqual(env.time, 10.0, places=12)
                self.assertTrue(all(6 <= n <= 90 for n in steps[:-1]), sorted(set(steps[:-1])))
                self.assertTrue(1 <= steps[-1] <= 90, steps[-1])
                # Both ends of the law occur (0.891 and 0.099 per draw); the mean interval,
                # 0.4912 T, gives about twice the fixed clock's 334 decisions (about 680).
                self.assertEqual((min(steps[:-1]), max(steps[:-1])), (6, 90))
                self.assertGreater(len(steps), 334)

    def test_without_clock_settings_the_config_is_unchanged(self):
        """The registered factory without clock settings (or clock_kind None or "fixed")
        returns what the module-level factory returns: the fixed 30 ms clock, every row's."""
        for testbed, env_id, _, _ in TASKS:
            spec = testbed_spec(testbed)
            module_factory = self.MODULE_FACTORIES[testbed]
            for limit in (None, 2.0):
                expected = module_factory(**self.settings(env_id, limit))
                self.assertEqual(expected["clock"], fixed_clock(0.03, 0.001))
                for clock in ({}, {"clock_kind": None}, {"clock_kind": "fixed"}):
                    with self.subTest(testbed=testbed, torque_limit=limit, clock=clock):
                        config = spec.config_factory(**self.settings(env_id, limit, **clock))
                        self.assertEqual(sorted(config), sorted(expected))
                        for key in expected:
                            self.assertEqual(repr(config[key]), repr(expected[key]), key)
            for bad in (
                {"clock_kind": "fixed", "tail_probability": 0.99},
                {"clock_kind": "two_tail_uniform", "tail_probability": 0.99},
                {**self.LAW, "beta_shape": 2.0},
                {"clock_kind": "poisson", "tail_probability": 0.99, "tail_split": 0.9},
            ):
                with self.subTest(testbed=testbed, bad=bad), self.assertRaises(ValueError):
                    spec.config_factory(**self.settings(env_id, 2.0, **bad))


class TestRiccatiDiagnostic(unittest.TestCase):
    """The table's +-20 N limit against the discounted LQR optimum of the unconstrained task.

    Maximizing E int e^{-lambda t} (1 - |q|^2/2 - c |u|^2/2) dt is the discounted LQR with
    Q = diag(I, 0), R = c I: P solves the ARE of (A - lambda/2 I, B, Q, R), u = -R^-1 B^T P x,
    and the value is 1/lambda - x^T P x / 2.
    """

    def solve(self, config):
        state_matrix, input_matrix = lqr_state_matrices(config)
        n_bodies, n_actuated = (int(config["plant_parameters"][k]) for k in ("n_bodies", "n_actuated"))
        cost = np.diag([1.0] * n_bodies + [0.0] * n_bodies)
        control = config["reward_parameters"]["control_cost"] * np.eye(n_actuated)
        shifted = state_matrix - 0.5 * config["discount_rate"] * np.eye(2 * n_bodies)
        riccati = scipy.linalg.solve_continuous_are(shifted, input_matrix, cost, control)
        residual = (
            shifted.T @ riccati + riccati @ shifted + cost
            - riccati @ input_matrix @ np.linalg.solve(control, input_matrix.T @ riccati)
        )
        self.assertLess(np.abs(residual).max(), 1e-8)
        gain = np.linalg.solve(control, input_matrix.T @ riccati)
        return state_matrix, input_matrix, riccati, gain

    def test_optimal_torque_stays_within_the_limit(self):
        """Peak |u| of the optimal closed loop over 10 s from sampled resets: well under 20 N."""
        peaks = {}
        for testbed, env_id, n_bodies, _ in TASKS:
            with self.subTest(testbed=testbed):
                config = table_config(testbed, env_id)
                state_matrix, input_matrix, _, gain = self.solve(config)
                closed = state_matrix - input_matrix @ gain
                self.assertLess(np.linalg.eigvals(closed).real.max(), 0.0)
                transition = scipy.linalg.expm(closed * 0.01)
                env = make_env(config, seed=0)
                peak = 0.0
                for seed in range(20):
                    state, _ = env.reset(seed=seed)
                    for _ in range(1001):
                        peak = max(peak, float(np.abs(gain @ state).max()))
                        state = transition @ state
                peaks[testbed] = peak
                # At most sqrt(2) |gain row| (about 0.2 N) at the start; the peak comes later,
                # about 1.3 N for these resets (up to 1.6 N over others).
                self.assertLess(peak, 2.0)
                self.assertLess(peak, 0.1 * LQR_TORQUE_LIMIT)
        self.assertGreater(peaks["lqr_2_1"], 1.0)
        self.assertGreater(peaks["lqr_6_2"], 1.0)

    def test_closed_loop_return_matches_the_riccati_value(self):
        """Our env under u = -K x (1 ms hold) earns the Riccati value over a 3 s window."""
        for testbed, env_id, n_bodies, _ in TASKS:
            with self.subTest(testbed=testbed):
                config = table_config(testbed, env_id)
                _, _, riccati, gain = self.solve(config)
                env = make_env(config, decision_mode="torque", seed=0)
                state, _ = env.reset(seed=3)
                start = state.copy()
                rate = config["discount_rate"]
                total, horizon = 0.0, 3.0
                while env.time < horizon - 1e-9:
                    discount = math.exp(-rate * env.time)
                    transition = env.step_dt(-gain @ state, env.physics_dt)
                    total += discount * transition["reward"]
                    state = transition["next_state"]
                final = math.exp(-rate * horizon)
                expected = (1.0 - final) / rate - 0.5 * (start @ riccati @ start - final * state @ riccati @ state)
                self.assertLess(abs(total - expected), 1e-4 * (1.0 + abs(expected)))


if __name__ == "__main__":
    unittest.main()
