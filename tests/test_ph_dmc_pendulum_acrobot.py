"""The dm_control pendulum and acrobot ports against dm_control itself.

Testbeds pendulum_swingup, acrobot_swingup and acrobot_swingup_sparse, built from their
table rows. The checks follow environment/ph_sac/dmc/README.md ("Validation"): the oracle
drift against our MuJoCo model, our model's forward dynamics against dm_control's compiled
model, the reward against dm_control's task.get_reward, the empty failure set, trajectories
against dm_control's env.step at dm_control's physics step and integrator, the table rows,
the reset law, smoke episodes, the oracle plant and torch.func, and the upright predicate.
acrobot_swingup's reward is Xin and Kaneda's Lyapunov reward, not dm_control's, so the
reward checks against dm_control cover only DM_CONTROL_REWARDS;
tests/test_ph_dmc_acrobot_reward.py tests that reward (and Zhang et al.'s, its reward
before). The pendulum row's reward is dm_control's tolerance with margin 2 (dense since
2026-11-09) where dm_control's task uses margin 0, so the comparisons with dm_control's
task reward use suite_reward_config (the row with margin 0), and TestPendulumDenseReward
checks the row's margin 2 against dm_control's rewards.tolerance.
"""

import contextlib
import io
import math
import unittest

import mujoco
import numpy as np
import torch

import common.ph_config as ph_config
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import load_protocol_from_table, main
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, fixed_clock
from environment.ph_sac import tasks
from environment.ph_sac.base import validated_state_spec
from environment.ph_sac.dmc import acrobot as dmc_acrobot
from environment.ph_sac.dmc import pendulum as dmc_pendulum
from environment.ph_sac.dmc._common import (
    body_inertials,
    planar_chain_from_model,
    tolerance,
    unit_interval_bounds,
)
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac.protocol import UPRIGHT_TUBE, is_upright

TASKS = {
    "pendulum_swingup": dict(
        env_id="pendulum-swingup",
        domain="pendulum",
        task="swingup",
        gear=1.0,
        links=("pole",),
        dm_dt=0.02,
        dm_integrator="Euler",
        env_class=dmc_pendulum.PendulumSwingupEnv,
    ),
    "acrobot_swingup": dict(
        env_id="acrobot-swingup",
        domain="acrobot",
        task="swingup",
        gear=2.0,
        links=("upper_arm", "lower_arm"),
        dm_dt=0.01,
        dm_integrator="RK4",
        env_class=dmc_acrobot.AcrobotSwingupEnv,
    ),
    "acrobot_swingup_sparse": dict(
        env_id="acrobot-swingup_sparse",
        domain="acrobot",
        task="swingup_sparse",
        gear=2.0,
        links=("upper_arm", "lower_arm"),
        dm_dt=0.01,
        dm_integrator="RK4",
        env_class=dmc_acrobot.AcrobotSwingupEnv,
    ),
}
_INTEGRATORS = {"Euler": mujoco.mjtIntegrator.mjINT_EULER, "RK4": mujoco.mjtIntegrator.mjINT_RK4}
# The testbeds whose reward is dm_control's tolerance (in [0, 1]); acrobot_swingup's is Xin
# and Kaneda's Lyapunov reward. pendulum_swingup's row has margin 2, dm_control's task
# margin 0: suite_reward_config gives the latter.
DM_CONTROL_REWARDS = ("pendulum_swingup", "acrobot_swingup_sparse")
COSINE_8_DEG = math.cos(math.radians(8.0))
PENDULUM_ROW_MARGIN = 2.0


def table_row(name):
    return table.find_ph_entry(TASKS[name]["env_id"], "top")


def table_config(name):
    """The testbed's TestbedConfig from its table row's settings."""
    return testbed_spec(name).config_factory(**table_row(name).config["testbed_settings"])


def suite_reward_config(name):
    """table_config with dm_control's task reward: the pendulum row's margin set to 0."""
    config = table_config(name)
    if name == "pendulum_swingup":
        config = dict(config, reward_parameters=dict(config["reward_parameters"], margin=0.0))
    return config


def dm_env(name):
    from dm_control import suite

    info = TASKS[name]
    return suite.load(info["domain"], info["task"], task_kwargs={"random": 0})


def wrap(angles):
    return np.mod(np.asarray(angles) + np.pi, 2.0 * np.pi) - np.pi


def random_states(env, rng, count, rate_bound=8.0):
    """Wrapped angles uniform on [-pi, pi) and rates uniform on +-rate_bound."""
    nq = env.physical_position_dim
    return np.hstack(
        (rng.uniform(-np.pi, np.pi, (count, nq)), rng.uniform(-rate_bound, rate_bound, (count, nq)))
    )


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def reward_rate_peak(name, config):
    """The largest reward rate: 1 for dm_control's rewards, -log(reward_floor) for the
    Lyapunov reward, scale (r_line + r_lqr) for Zhang's."""
    if name in DM_CONTROL_REWARDS:
        return 1.0
    reward = config["reward_parameters"]
    kind = dmc_acrobot.acrobot_reward_kind(reward)
    if kind == "lyapunov":
        return -math.log(reward["reward_floor"])
    if kind == "zhang2023":
        return reward["scale"] * (reward["line_bonus"] + reward["lqr_bonus"])
    return 1.0


@contextlib.contextmanager
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class TestTableRows(unittest.TestCase):
    def test_factories_reproduce_the_rows(self):
        for name in TASKS:
            with self.subTest(testbed=name):
                spec = testbed_spec(name)
                self.assertTrue(spec.implemented)
                self.assertIs(spec.env_class, TASKS[name]["env_class"])
                entry = table_row(name)
                self.assertEqual(entry.config["testbed"], name)
                self.assertEqual(set(spec.settings_keys), set(entry.config["testbed_settings"]))
                # The irregular clocks (rows of mode irregular, since 2026-11-11).
                self.assertEqual(spec.optional_settings_keys, CLOCK_SETTINGS_KEYS)
                config = table_config(name)
                self.assertEqual(config["name"], name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, config), [])
                self.assertIs(validated_state_spec(config), config["state_spec"])
                ph_config._validate_testbed(config)
                self.assertEqual(config["failure_limits"], {})

    def test_acrobot_rows_differ_only_in_name_and_reward(self):
        dense, sparse = table_config("acrobot_swingup"), table_config("acrobot_swingup_sparse")
        self.assertEqual(tuple(dense["reward_parameters"]), dmc_acrobot.LYAPUNOV_REWARD_KEYS)
        self.assertEqual(sparse["reward_parameters"], {"target_radius": 0.2, "margin": 0.0})
        for key in dense:
            if key in ("name", "reward_parameters"):
                continue
            with self.subTest(key=key):
                left, right = dense[key], sparse[key]
                if isinstance(left, dict):
                    self.assertEqual(sorted(left), sorted(right))
                    for inner in left:
                        np.testing.assert_array_equal(left[inner], right[inner])
                else:
                    self.assertEqual(left, right)

    def test_protocols_build_from_the_table(self):
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                protocol = load_protocol_from_table(info["env_id"])
                self.assertEqual(protocol["matched"]["testbed_config"]["name"], name)

    def test_list_reads_builds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--list"])
        lines = {line.split()[0]: line for line in output.getvalue().splitlines()[1:]}
        for info in TASKS.values():
            with self.subTest(env_id=info["env_id"]):
                self.assertTrue(lines[info["env_id"]].endswith("builds"), lines[info["env_id"]])

    def test_missing_settings_raise(self):
        for name in TASKS:
            config = table_config(name)
            for section in ("plant_parameters", "reward_parameters", "reset_parameters"):
                with self.subTest(testbed=name, section=section):
                    broken = dict(config, **{section: dict(list(config[section].items())[1:])})
                    with self.assertRaisesRegex(ValueError, "missing required keys"):
                        tasks.make_env(broken)


class TestDynamics(unittest.TestCase):
    def test_oracle_matches_our_mujoco(self):
        """env.oracle_drift, the tensor oracle and env.mujoco_drift agree to 1e-8."""
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                drift, diffusion = tasks.oracle_callbacks(config)
                rng = np.random.default_rng(0)
                states = random_states(env, rng, 200)
                torques = rng.uniform(-info["gear"], info["gear"], (200, 1))
                tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                for state, torque, row in zip(states, torques, tensor):
                    truth = env.mujoco_drift(state, torque)
                    self.assertLess(relative_error(env.oracle_drift(state, torque), truth), 1e-8)
                    self.assertLess(relative_error(row, truth), 1e-8)
                self.assertEqual(diffusion(torch.as_tensor(states)).shape, (200, env.state_dim, 1))
                self.assertFalse(diffusion(torch.as_tensor(states)).any())
                np.testing.assert_array_equal(env.oracle_diffusion(states[0]), np.zeros((env.state_dim, 1)))

    def test_our_mujoco_matches_dm_control_forward_dynamics(self):
        """qacc at dm_control's (unwrapped) qpos equals ours at the wrapped state, 1e-8."""
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                env = tasks.make_env(table_config(name), seed=0)
                physics = dm_env(name).physics
                nq = env.physical_position_dim
                self.assertEqual((physics.model.nq, physics.model.nv, physics.model.nu), (nq, nq, 1))
                rng = np.random.default_rng(1)
                for _ in range(200):
                    qpos = rng.uniform(-3.0 * np.pi, 3.0 * np.pi, nq)
                    qvel = rng.uniform(-8.0, 8.0, nq)
                    control = rng.uniform(-1.0, 1.0, 1)
                    physics.data.qpos[:] = qpos
                    physics.data.qvel[:] = qvel
                    physics.data.ctrl[:] = control
                    physics.forward()
                    state = np.concatenate((wrap(qpos), qvel))
                    ours = env.mujoco_drift(state, control * info["gear"])
                    self.assertLess(relative_error(ours[nq:], physics.data.qacc), 1e-8)
                    np.testing.assert_array_equal(ours[:nq], qvel)

    def test_compiled_inertials_match_dm_control(self):
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                physics = dm_env(name).physics
                ours = body_inertials(env.model, info["links"])
                theirs = body_inertials(physics, info["links"])
                for link in info["links"]:
                    self.assertAlmostEqual(ours[link].mass, theirs[link].mass, places=12)
                    np.testing.assert_allclose(ours[link].com, theirs[link].com, atol=1e-12)
                    np.testing.assert_allclose(ours[link].inertia, theirs[link].inertia, atol=1e-12)
                if name == "pendulum_swingup":
                    chain = dmc_pendulum.pendulum_chain(config)
                    reference = planar_chain_from_model(
                        physics, info["links"], along=2, joint_axis=1, last_length=0.5
                    )
                else:
                    chain = dmc_acrobot.acrobot_chain(config)
                    reference = dmc_acrobot.acrobot_chain_from_model(physics, config)
                for field in (
                    "link_masses", "link_com_distances", "link_inertias", "link_lengths",
                    "joint_damping",
                ):
                    np.testing.assert_allclose(
                        getattr(chain, field), getattr(reference, field), rtol=1e-12, atol=1e-15
                    )
                self.assertAlmostEqual(chain.gravity, reference.gravity, places=12)
                # dm_control's input is the normalized control, ours the torque: G differs by the gear.
                np.testing.assert_allclose(chain.input_matrix * info["gear"], reference.input_matrix)
                np.testing.assert_array_equal(chain.input_matrix, env.oracle_input_matrix())
        inertias = dmc_acrobot.acrobot_chain(table_config("acrobot_swingup")).link_inertias
        np.testing.assert_allclose(inertias, (0.09557031, 0.09530366), atol=1e-8)
        pendulum = dmc_pendulum.pendulum_chain(table_config("pendulum_swingup"))
        self.assertAlmostEqual(pendulum.link_inertias[0], 0.001, places=15)

    def test_trajectories_match_dm_control_steps(self):
        """At dm_control's physics step and integrator, 1 s of open-loop control, 1e-6."""
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                config = suite_reward_config(name)
                dt = info["dm_dt"]
                config = dict(
                    config,
                    physics_dt=dt,
                    integrator=info["dm_integrator"],
                    clock=fixed_clock(config["reference_period"], dt),
                )
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                self.assertEqual(env.model.opt.integrator, _INTEGRATORS[info["dm_integrator"]])
                self.assertEqual(env.model.opt.timestep, dt)
                dm = dm_env(name)
                physics = dm.physics
                self.assertEqual(physics.model.opt.integrator, _INTEGRATORS[info["dm_integrator"]])
                self.assertEqual(physics.model.opt.timestep, dt)
                self.assertEqual(dm.control_timestep(), dt)
                nq = env.physical_position_dim
                rng = np.random.default_rng(2)
                qpos, qvel = rng.uniform(-np.pi, np.pi, nq), rng.uniform(-2.0, 2.0, nq)
                state, _ = env.reset(options={"state": np.concatenate((qpos, qvel))})
                dm.reset()
                with physics.reset_context():
                    physics.data.qpos[:] = qpos
                    physics.data.qvel[:] = qvel
                steps = round(1.0 / dt)
                controls = rng.uniform(-1.0, 1.0, steps)
                controls[:4] = (1.0, -1.0, 1.0, 0.0)
                worst = 0.0
                for control in controls:
                    transition = env.step_dt(np.array([control * info["gear"]]), dt)
                    time_step = dm.step(np.array([control]))
                    state = transition["next_state"]
                    worst = max(
                        worst,
                        float(np.max(np.abs(wrap(state[:nq] - physics.data.qpos)))),
                        float(np.max(np.abs(state[nq:] - physics.data.qvel))),
                    )
                    if name in DM_CONTROL_REWARDS:
                        self.assertAlmostEqual(
                            env.reward_rate(state, transition["decision"]), time_step.reward,
                            places=10,
                        )
                self.assertLess(worst, 1e-6)
                self.assertAlmostEqual(env.time, physics.data.time, places=12)
                self.assertGreater(np.max(np.abs(state - np.concatenate((qpos, qvel)))), 0.1)

    def test_oracle_plant_and_torch_func(self):
        """make_oracle_plant(env) drift equals oracle_drift; its Jacobians match differences."""
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(4)
                states = random_states(env, rng, 6, rate_bound=4.0)
                actions = rng.uniform(-info["gear"], info["gear"], (6, 1))
                with float64_default():
                    plant = tasks.make_oracle_plant(env)
                    states_t, actions_t = torch.as_tensor(states), torch.as_tensor(actions)
                    drift = plant.drift(states_t, actions_t).numpy()
                    values, by_state, by_action = plant.drift_jacobian(states_t, actions_t)
                for state, action, row in zip(states, actions, drift):
                    self.assertLess(relative_error(row, env.oracle_drift(state, action)), 1e-10)
                self.assertEqual(plant.state_spec, env.state_spec)
                self.assertEqual(plant.input_matrix().tolist(), env.oracle_input_matrix().tolist())
                np.testing.assert_allclose(values.numpy(), drift, rtol=1e-12, atol=1e-12)
                callback, _ = tasks.oracle_callbacks(config)
                epsilon = 1e-6
                for i in range(env.state_dim):
                    bump = torch.zeros_like(states_t)
                    bump[:, i] = epsilon
                    numeric = (callback(states_t + bump, actions_t) - callback(states_t - bump, actions_t)) / (
                        2.0 * epsilon
                    )
                    self.assertLess((by_state[:, :, i] - numeric).abs().max().item(), 1e-6)
                bump = torch.full_like(actions_t, epsilon)
                numeric = (callback(states_t, actions_t + bump) - callback(states_t, actions_t - bump)) / (
                    2.0 * epsilon
                )
                self.assertLess((by_action[:, :, 0] - numeric).abs().max().item(), 1e-6)


class TestReward(unittest.TestCase):
    def sample(self, name, rng):
        """States uniform on the circle and near upright, angles shifted by turns for dm_control.

        Returns (dm_control qpos, our wrapped states); none within 1e-9 of the indicator's edge.
        """
        nq = 1 if name == "pendulum_swingup" else 2
        near = math.radians(16.0) if name == "pendulum_swingup" else 0.25
        angles = np.vstack(
            (rng.uniform(-np.pi, np.pi, (150, nq)), rng.uniform(-near, near, (150, nq)))
        )
        rates = rng.uniform(-8.0, 8.0, angles.shape)
        qpos = angles + 2.0 * np.pi * rng.integers(-2, 3, angles.shape)
        return qpos, np.hstack((wrap(qpos), rates)), rates

    def test_reward_matches_dm_control(self):
        """reward_rate_tensor, env.reward_rate and dm_control's get_reward agree to 1e-10.

        The pendulum at dm_control's margin 0 (suite_reward_config), not the row's 2.
        """
        for name in DM_CONTROL_REWARDS:
            info = TASKS[name]
            with self.subTest(testbed=name):
                config = suite_reward_config(name)
                if name == "pendulum_swingup":
                    self.assertEqual(config["reward_parameters"]["margin"], 0.0)
                env = tasks.make_env(config, seed=0)
                dm = dm_env(name)
                physics = dm.physics
                rng = np.random.default_rng(5)
                qpos, states, rates = self.sample(name, rng)
                controls = rng.uniform(-1.0, 1.0, (len(states), 1))
                tensor = tasks.reward_rate_tensor(
                    torch.as_tensor(states), torch.as_tensor(controls * info["gear"]), config
                )
                self.assertEqual(tensor.shape, (len(states), 1))
                inside = outside = 0
                for row in range(len(states)):
                    physics.data.qpos[:] = qpos[row]
                    physics.data.qvel[:] = rates[row]
                    physics.data.ctrl[:] = controls[row]
                    physics.forward()
                    if name == "pendulum_swingup":
                        edge = physics.pole_vertical() - config["reward_parameters"]["cosine_bound"]
                    else:
                        edge = physics.to_target() - config["reward_parameters"]["target_radius"]
                        self.assertAlmostEqual(env.tip_distance(states[row]), physics.to_target(), places=12)
                    if abs(edge) < 1e-9:
                        continue
                    hit = edge > 0.0 if name == "pendulum_swingup" else edge < 0.0
                    inside += int(hit)
                    outside += int(not hit)
                    expected = float(dm.task.get_reward(physics))
                    ours = env.reward_rate(states[row], controls[row] * info["gear"])
                    self.assertAlmostEqual(ours, expected, places=10)
                    self.assertAlmostEqual(tensor[row, 0].item(), expected, places=10)
                    if hit:
                        self.assertEqual(expected, 1.0)
                self.assertGreater(inside, 20)
                self.assertGreater(outside, 20)

    def test_reward_ignores_the_torque_and_is_differentiable_at_the_target(self):
        for name in DM_CONTROL_REWARDS:
            info = TASKS[name]
            with self.subTest(testbed=name):
                config = table_config(name)
                states = torch.as_tensor(random_states(tasks.make_env(config), np.random.default_rng(6), 50))
                zero = tasks.reward_rate_tensor(states, torch.zeros(50, 1, dtype=torch.float64), config)
                full = tasks.reward_rate_tensor(states, torch.full((50, 1), info["gear"], dtype=torch.float64), config)
                self.assertTrue(torch.equal(zero, full))
                self.assertTrue(torch.equal(zero, testbed_spec(name).reward_rate_tensor(states, zero, config)))
                upright = torch.zeros(1, states.shape[1], dtype=torch.float64, requires_grad=True)
                rate = tasks.reward_rate_tensor(upright, torch.zeros(1, 1, dtype=torch.float64), config)
                self.assertEqual(rate.item(), 1.0)
                # The indicator rewards (margin 0) are piecewise constant, without a graph.
                self.assertEqual(rate.requires_grad, config["reward_parameters"]["margin"] > 0.0)
                if rate.requires_grad:
                    (gradient,) = torch.autograd.grad(rate.sum(), upright)
                    self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_tolerance_margin_decays_outside_the_target(self):
        """dm_control's acrobot/swingup reward (margin 1), which no row uses, stays available."""
        config = table_config("acrobot_swingup_sparse")
        config = dict(config, reward_parameters={"target_radius": 0.2, "margin": 1.0})
        env = tasks.make_env(config)
        self.assertEqual(env.reward_kind, "tolerance")
        # The upper arm up, the lower arm horizontal: the tip 1.414 m from the target.
        state = np.array([0.0, 0.5 * np.pi, 0.0, 0.0])
        distance = env.tip_distance(state)
        self.assertAlmostEqual(distance, math.sqrt(2.0), places=12)
        expected = math.exp(-0.5 * -2.0 * math.log(0.1) * (distance - 0.2) ** 2)
        self.assertAlmostEqual(env.reward_rate(state, np.zeros(1)), expected, places=12)
        sparse = tasks.make_env(table_config("acrobot_swingup_sparse"))
        self.assertEqual(sparse.reward_rate(state, np.zeros(1)), 0.0)


class TestPendulumDenseReward(unittest.TestCase):
    """The row's reward: dm_control's tolerance(cos theta, (cos 8 deg, 1), margin=2)."""

    def test_row_carries_margin_2(self):
        settings = table_row("pendulum_swingup").config["testbed_settings"]
        self.assertEqual(
            settings["reward_parameters"],
            {"cosine_bound": COSINE_8_DEG, "margin": PENDULUM_ROW_MARGIN},
        )
        self.assertEqual(table_config("pendulum_swingup")["reward_parameters"]["margin"], 2.0)
        self.assertIs(testbed_spec("pendulum_swingup").reward_rate_bounds, unit_interval_bounds)

    def test_matches_dm_control_tolerance(self):
        """env.reward_rate equals rewards.tolerance (1e-8); the tensor equals numpy (1e-12)."""
        from dm_control.utils import rewards

        config = table_config("pendulum_swingup")
        env = tasks.make_env(config, seed=0)
        rng = np.random.default_rng(11)
        angles = np.concatenate(
            (rng.uniform(-np.pi, np.pi, 300), rng.uniform(-0.3, 0.3, 100), [0.0, np.pi, -np.pi])
        )
        # Angles beyond one turn, as dm_control's unwrapped hinge reads them.
        angles = angles + 2.0 * np.pi * rng.integers(-2, 3, angles.shape)
        states = np.stack((angles, rng.uniform(-8.0, 8.0, angles.shape)), axis=1)
        torques = rng.uniform(-1.0, 1.0, (len(states), 1))
        expected = rewards.tolerance(
            np.cos(angles), bounds=(COSINE_8_DEG, 1.0), margin=2.0,
            sigmoid="gaussian", value_at_margin=0.1,
        )
        ours = np.array([env.reward_rate(state, torque) for state, torque in zip(states, torques)])
        np.testing.assert_allclose(ours, expected, rtol=0.0, atol=1e-8)
        tensor = tasks.reward_rate_tensor(torch.as_tensor(states), torch.as_tensor(torques), config)
        self.assertEqual(tensor.shape, (len(states), 1))
        np.testing.assert_allclose(tensor[:, 0].numpy(), ours, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            tolerance(np.cos(angles), (COSINE_8_DEG, 1.0), 2.0), expected, rtol=0.0, atol=1e-8
        )
        inside = np.cos(angles) >= COSINE_8_DEG
        self.assertGreater(inside.sum(), 20)
        self.assertTrue(np.all(ours[inside] == 1.0))
        self.assertTrue(np.all((ours[~inside] > 0.1) & (ours[~inside] < 1.0)))

    def test_reference_values(self):
        """1 within 8 deg, about 0.56 horizontal, about 0.1 hanging."""
        config = table_config("pendulum_swingup")
        env = tasks.make_env(config, seed=0)
        suite = tasks.make_env(suite_reward_config("pendulum_swingup"), seed=0)

        def rate(environment, angle):
            return environment.reward_rate(np.array([angle, 0.0]), np.zeros(1))

        for angle in (0.0, math.radians(7.9), -math.radians(7.9), 2.0 * np.pi):
            with self.subTest(angle=angle):
                self.assertEqual(rate(env, angle), 1.0)
                self.assertEqual(rate(suite, angle), 1.0)
        for angle in (0.5 * np.pi, -0.5 * np.pi):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(rate(env, angle), 0.56, delta=0.01)
                # dm_control's gaussian: 10^-(d / margin)^2 with d = cos 8 deg - cos theta.
                self.assertAlmostEqual(rate(env, angle), 10.0 ** -((COSINE_8_DEG / 2.0) ** 2), places=12)
                self.assertEqual(rate(suite, angle), 0.0)
        for angle in (np.pi, -np.pi):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(rate(env, angle), 0.1, delta=0.005)
                self.assertAlmostEqual(
                    rate(env, angle), 10.0 ** -(((COSINE_8_DEG + 1.0) / 2.0) ** 2), places=12
                )
                self.assertEqual(rate(suite, angle), 0.0)

    def test_gradient_outside_the_window(self):
        """Dense: autograd of the tensor reward matches central differences of the numpy one."""
        config = table_config("pendulum_swingup")
        env = tasks.make_env(config, seed=0)
        angles = torch.tensor([0.3, 1.0, 0.5 * np.pi, 2.5, -2.0], dtype=torch.float64)
        states = torch.stack((angles, torch.zeros_like(angles)), dim=1).requires_grad_(True)
        rate = tasks.reward_rate_tensor(states, torch.zeros(len(angles), 1, dtype=torch.float64), config)
        (gradient,) = torch.autograd.grad(rate.sum(), states)
        epsilon = 1e-6
        for row, angle in enumerate(angles.tolist()):
            numeric = (
                env.reward_rate(np.array([angle + epsilon, 0.0]), np.zeros(1))
                - env.reward_rate(np.array([angle - epsilon, 0.0]), np.zeros(1))
            ) / (2.0 * epsilon)
            self.assertNotEqual(gradient[row, 0].item(), 0.0)
            self.assertAlmostEqual(gradient[row, 0].item(), numeric, places=7)
            self.assertEqual(gradient[row, 1].item(), 0.0)


class TestNoFailureSet(unittest.TestCase):
    def test_mask_payoff_and_lift(self):
        """All-False mask (as env.failed), payoff 0 (as env.terminal_payoff), H = 0, B = 1."""
        for name in TASKS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(7)
                states = random_states(env, rng, 200)
                states[:50] *= 1e3  # Far outside any range a failure set could use.
                states_t = torch.as_tensor(states)
                mask = tasks.failure_mask_tensor(states_t, config)
                self.assertEqual(mask.shape, (200,))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertFalse(mask.any())
                for state in states:
                    self.assertFalse(env.failed(state))
                payoff = tasks.terminal_payoff_tensor(states_t, config)
                self.assertEqual(payoff.shape, (200, 1))
                self.assertFalse(payoff.any())
                self.assertEqual(env.terminal_payoff(states[0]), 0.0)
                self.assertEqual(testbed_spec(name).terminal_payoff_value(config), 0.0)
                lift, vanishing = tasks.terminal_boundary_functions(config)
                self.assertTrue(torch.equal(lift(states_t), torch.zeros(200, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(vanishing(states_t), torch.ones(200, 1, dtype=torch.float64)))


class TestEpisodes(unittest.TestCase):
    def test_reset_law_matches_dm_control(self):
        """Angles uniform on [-pi, pi), velocities 0, as dm_control's resets."""
        for name in TASKS:
            with self.subTest(testbed=name):
                env = tasks.make_env(table_config(name), seed=0)
                nq = env.physical_position_dim
                ours = np.array([env.reset(seed=seed)[0] for seed in range(400)])
                dm = dm_env(name)
                theirs = []
                for _ in range(400):
                    dm.reset()
                    theirs.append(np.concatenate((dm.physics.data.qpos, dm.physics.data.qvel)))
                theirs = np.array(theirs)
                deviation = math.pi / math.sqrt(3.0)
                for draws in (ours, theirs):
                    angles = draws[:, :nq]
                    self.assertTrue(np.all((angles >= -np.pi) & (angles < np.pi)))
                    self.assertTrue(np.all(draws[:, nq:] == 0.0))
                    self.assertLess(np.max(np.abs(angles.mean(axis=0))), 4.0 * deviation / math.sqrt(400))
                    np.testing.assert_allclose(angles.std(axis=0), deviation, rtol=0.1)
                    self.assertGreater(angles.max(), 0.95 * np.pi)
                    self.assertLess(angles.min(), -0.95 * np.pi)
                given = np.concatenate((np.full(nq, 2.0 * np.pi + 0.1), np.arange(1.0, nq + 1.0)))
                state, _ = env.reset(options={"state": given})
                np.testing.assert_allclose(state, np.concatenate((np.full(nq, 0.1), given[nq:])), atol=1e-12)

    def test_feedback_episode(self):
        """A whole episode of random decisions: 1000 decisions at the reference period."""
        for name in TASKS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                self.assertIsInstance(env, TASKS[name]["env_class"])
                np.testing.assert_array_equal(env.action_space.low, config["controller"]["decision_low"])
                np.testing.assert_array_equal(env.action_space.high, config["controller"]["decision_high"])
                state, _ = env.reset(seed=0)
                rng = np.random.default_rng(8)
                decisions = total = 0
                terminated = truncated = False
                peak = reward_rate_peak(name, config)
                while not env.done:
                    decision = rng.uniform(env.action_space.low, env.action_space.high)
                    state, reward, terminated, truncated, info = env.step(decision)
                    self.assertEqual(info["dt"], config["reference_period"])
                    self.assertTrue(np.isfinite(state).all())
                    self.assertTrue(np.all(np.abs(state[: env.physical_position_dim]) <= np.pi))
                    self.assertTrue(np.isfinite(reward))
                    if name in DM_CONTROL_REWARDS:
                        self.assertGreaterEqual(reward, 0.0)
                    total += env.last_undiscounted_reward
                    decisions += 1
                self.assertEqual(decisions, round(config["episode_duration"] / config["reference_period"]))
                self.assertTrue(truncated)
                self.assertFalse(terminated)
                self.assertAlmostEqual(env.time, config["episode_duration"], places=9)
                self.assertLessEqual(total, peak * config["episode_duration"] + 1e-9)

    def test_torque_episode(self):
        for name, info in TASKS.items():
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                np.testing.assert_array_equal(env.action_space.low, [-info["gear"]])
                np.testing.assert_array_equal(env.action_space.high, [info["gear"]])
                env.reset(seed=1)
                rng = np.random.default_rng(9)
                peak = reward_rate_peak(name, config)
                for _ in range(200):
                    state, reward, terminated, truncated, _ = env.step(rng.uniform(-3.0, 3.0, 1))
                    self.assertTrue(np.isfinite(state).all())
                    self.assertFalse(terminated or truncated)
                    self.assertLessEqual(reward, peak * config["reference_period"] + 1e-12)
                    if name in DM_CONTROL_REWARDS:
                        self.assertGreaterEqual(reward, 0.0)
                self.assertAlmostEqual(env.time, 200 * config["reference_period"], places=9)

    def test_upright(self):
        config = table_config("pendulum_swingup")
        spec = testbed_spec("pendulum_swingup")
        for angle, expected in ((0.1, True), (-0.19, True), (0.3, False), (2.0 * np.pi + 0.1, True), (np.pi, False)):
            with self.subTest(testbed="pendulum_swingup", angle=angle):
                state = np.array([angle, 5.0])
                self.assertEqual(spec.upright(config, state, UPRIGHT_TUBE), expected)
                self.assertEqual(is_upright(config, state), expected)
        for name in ("acrobot_swingup", "acrobot_swingup_sparse"):
            config = table_config(name)
            spec = testbed_spec(name)
            for shoulder, elbow, expected in (
                (0.0, 0.0, True),
                (0.1, -0.15, True),  # Lower link at -0.05.
                (0.15, 0.15, False),  # Lower link at 0.3.
                (0.19, -0.38, True),  # Lower link at -0.19.
                (0.1, 2.0 * np.pi - 0.1, True),
                (np.pi, 0.0, False),
                (0.0, np.pi, False),
            ):
                with self.subTest(testbed=name, shoulder=shoulder, elbow=elbow):
                    state = np.array([shoulder, elbow, 1.0, -1.0])
                    self.assertEqual(spec.upright(config, state, UPRIGHT_TUBE), expected)
                    self.assertEqual(is_upright(config, state), expected)


if __name__ == "__main__":
    unittest.main()
