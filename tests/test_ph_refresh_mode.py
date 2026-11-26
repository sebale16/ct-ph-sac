"""The controller's refresh_mode (common.ph_config.REFRESH_MODES).

"continuous" (the default since 2026-11-09) evaluates a feedback law from the current
state at every integrator stage: MuJoCo's control callback at the four RK4 stages of each
physics step in the environment, the RK4 stages of every substep in the model. "held" is
the sample-and-hold of every earlier run, torque refreshed each refresh_steps physics
steps and held through each step and its stages.

1. Config: the values, the default, validation; every factory and table row declares
   "continuous" (task_refresh_mode, checked against the factory); the top-level config
   key refresh_mode and run_ph_sac.py --refresh-mode replace it; both CT-PH-SAC arms
   share it; run records store it and records without it load as "held".
2. Environment: under RK4 physics the callback evaluates the law at the four stages of
   every physics step, only inside a continuous interval (never left installed, other
   environments and raw MuJoCo stepping unaffected, a previously installed callback kept
   and restored, also after an exception); "held" reproduces a hand-written
   sample-and-hold loop bitwise; Euler physics (the windy pendulum) and torque decisions
   agree bitwise in both modes; the reward's knots read the law at the knot's state.
3. Model: the continuous grid (no refresh points; a model step of k physics steps is
   exactly k physics steps), the stage torques of rk4_step, the closed-loop step
   Jacobians against central differences, the closed-loop term of path_jacobian_norms
   at every substep, the rollout's reward knots, and model steps above the physics step.
4. Held regression: in "held" mode, short training runs of four feedback arms and a
   torque arm on pendulum-swingup reproduce values computed with the code before the
   mode existed (cartpole-swingup until 2026-11-10, when its dynamics changed convention:
   dm_control's soft rail limit kept, no failure set).
"""

import argparse
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks.ph_hyperparams import load_ph_hyperparams_from_table
from benchmarks.run_ph_sac import (
    apply_overrides,
    load_protocol,
    load_protocol_from_table,
    parse_args,
    protocol_from_config,
)
from common.ph_config import (
    DEFAULT_REFRESH_MODE,
    LEGACY_REFRESH_MODE,
    REFRESH_MODES,
    check_refresh_mode,
    refresh_mode_setting,
    torque_decision_testbed,
    validate_configuration,
)
from environment.ph_sac import base as env_base
from environment.ph_sac import tasks
from environment.ph_sac.controller import FeedbackController, TorqueController
from environment.ph_sac.registry import registered_testbeds, testbed_spec
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac.reporting import (
    RECORD_NAME,
    load_run_records,
    record_refresh_mode,
)
from models.ph_sac.plant import PortHamiltonianPlant
from models.ph_sac.rollouts import (
    _action_state_jacobian,
    augmented_closed_loop_step,
    grid_refresh_steps,
    integrate_endpoint_moments,
    model_substep,
    path_jacobian_norms,
    refresh_aligned_grid,
    rk4_step,
    stage_feedback,
    stochastic_rollout,
)
from tests.test_ph_algorithms import explicit_config, windy
from tests.test_ph_model_integrator import (
    COUPLED_CONTROLLER,
    COUPLED_SPEC,
    PLANT_NETWORK,
    coupled_drift,
    coupled_plant,
    table_testbed,
)

CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac"


def with_mode(testbed, mode):
    """A shallow copy of testbed whose controller has refresh_mode mode."""
    return {**testbed, "controller": {**testbed["controller"], "refresh_mode": mode}}


def row_testbed(env_id, mode=None):
    entry = table.find_ph_entry(env_id, "top")
    testbed = testbed_spec(entry.config["testbed"]).config_factory(
        **entry.config["testbed_settings"]
    )
    return testbed if mode is None else with_mode(testbed, mode)


def continuous_controller():
    return FeedbackController(dict(COUPLED_CONTROLLER, refresh_mode="continuous"), COUPLED_SPEC)


def held_controller():
    return FeedbackController(dict(COUPLED_CONTROLLER, refresh_mode="held"), COUPLED_SPEC)


# ------------------------------------------------------------------------------ config


class TestRefreshModeConfig(unittest.TestCase):
    def test_values_default_and_validation(self):
        self.assertEqual(REFRESH_MODES, ("continuous", "held"))
        self.assertEqual(DEFAULT_REFRESH_MODE, "continuous")
        self.assertEqual(LEGACY_REFRESH_MODE, "held")
        self.assertEqual(refresh_mode_setting({}), "continuous")
        for mode in REFRESH_MODES:
            self.assertEqual(check_refresh_mode(mode), mode)
            self.assertEqual(refresh_mode_setting({"refresh_mode": mode}), mode)
        for bad in ("Continuous", "hold", "", None, 1):
            with self.assertRaisesRegex(ValueError, "refresh_mode", msg=repr(bad)):
                refresh_mode_setting({"refresh_mode": bad})
        config = explicit_config()
        config["testbed"] = with_mode(config["testbed"], "sampled")
        with self.assertRaisesRegex(ValueError, "refresh_mode"):
            validate_configuration(config)

    def test_every_factory_and_row_declares_continuous(self):
        for entry in table.load_ph_table():
            name = entry.config["testbed"]
            with self.subTest(row=(entry.env_id, entry.mode)):
                self.assertEqual(entry.task["refresh_mode"], "continuous")
                if name in registered_testbeds(implemented_only=True):
                    testbed = testbed_spec(name).config_factory(
                        **entry.config["testbed_settings"]
                    )
                    self.assertEqual(testbed["controller"]["refresh_mode"], "continuous")

    def test_declared_mode_is_checked_against_the_factory(self):
        testbed = row_testbed("cartpole-swingup")
        task = {"refresh_mode": "held"}
        self.assertEqual(
            table.declared_testbed_mismatches(task, testbed),
            ["task_refresh_mode: table 'held', factory 'continuous'"],
        )
        self.assertEqual(table.declared_testbed_mismatches(task, with_mode(testbed, "held")), [])

    def test_controllers_read_the_mode(self):
        testbed = row_testbed("cartpole-swingup")
        spec = testbed["state_spec"]
        for mode, continuous in (("continuous", True), ("held", False)):
            controller = FeedbackController(with_mode(testbed, mode)["controller"], spec)
            self.assertEqual((controller.refresh_mode, controller.continuous), (mode, continuous))
            self.assertEqual(stage_feedback(controller), continuous)
            self.assertEqual(grid_refresh_steps(controller, 1), None if continuous else 1)
            self.assertTrue(controller.refresh_due(0))
            self.assertEqual(controller.refresh_due(3), True)
            torque = TorqueController(
                torque_decision_testbed(with_mode(testbed, mode))["controller"], spec
            )
            self.assertEqual(torque.refresh_mode, mode)
            self.assertFalse(torque.continuous)
            self.assertFalse(stage_feedback(torque))
            self.assertEqual(grid_refresh_steps(torque, 1), None if continuous else 1)
        held = FeedbackController(dict(COUPLED_CONTROLLER, refresh_mode="held"), COUPLED_SPEC)
        self.assertEqual([held.refresh_due(step) for step in range(4)], [True, False, True, False])
        self.assertEqual(held.refresh_times(0.006, 0.001), (0.0, 0.002, 0.004))
        self.assertEqual(continuous_controller().refresh_times(0.006, 0.001), (0.0,))

    def test_json_key_and_command_line_replace_the_factory_mode(self):
        raw = json.loads((CONFIGS / "smoke_windy.json").read_text())
        for config_path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(config=config_path.name):
                choices = json.loads(config_path.read_text())["_experiment_choices"]
                self.assertIn("testbed.controller.refresh_mode", choices)
                protocol = load_protocol(str(config_path))
                self.assertEqual(
                    protocol["matched"]["testbed_config"]["controller"]["refresh_mode"],
                    "continuous",
                )
        held = protocol_from_config({**raw, "refresh_mode": "held"}, "held")
        for testbed in (
            held["matched"]["testbed_config"],
            held["explicit_config"]["testbed"],
            held["implicit_config"]["testbed"],
        ):
            self.assertEqual(testbed["controller"]["refresh_mode"], "held")
        with self.assertRaisesRegex(ValueError, "refresh_mode"):
            protocol_from_config({**raw, "refresh_mode": "sometimes"}, "bad")
        # A config that sets the key has no table row; the rows take the factory's mode.
        with self.assertRaises(ValueError):
            table.config_to_row({**raw, "refresh_mode": "held"}, env_id="x")
        args = parse_args(["--env-id", "cartpole-swingup", "--refresh-mode", "held"])
        overridden = apply_overrides(load_protocol_from_table("cartpole-swingup"), args)
        for testbed in (
            overridden["matched"]["testbed_config"],
            overridden["explicit_config"]["testbed"],
            overridden["implicit_config"]["testbed"],
        ):
            self.assertEqual(testbed["controller"]["refresh_mode"], "held")
        plain = apply_overrides(
            load_protocol_from_table("cartpole-swingup"), parse_args(["--env-id", "x"])
        )
        self.assertEqual(
            plain["matched"]["testbed_config"]["controller"]["refresh_mode"], "continuous"
        )

    def test_arms_share_the_mode(self):
        protocol = load_protocol(str(CONFIGS / "smoke_windy.json"))
        P.validate_matched_quantities(protocol)
        broken = copy.deepcopy(protocol)
        broken["implicit_config"]["testbed"] = with_mode(broken["implicit_config"]["testbed"], "held")
        with self.assertRaisesRegex(ValueError, "refresh_mode|matched testbed"):
            P.validate_matched_quantities(broken)

    def test_records_store_the_mode_and_old_records_load_held(self):
        protocol = load_protocol(str(CONFIGS / "smoke_windy.json"))
        result = {
            "run": {
                "protocol": protocol,
                "arm": {arm["name"]: arm for arm in P.all_arms()}["explicit_generator"],
                "training_seed": 0,
            },
            "evaluations": [],
            "counters": {},
            "solver": [],
            "admissions": [],
            "oracle": [],
            "plant_calibration": [],
            "checkpoint_paths": [],
            "decision_mode": "feedback",
        }
        self.assertEqual(record_refresh_mode(result), "held")
        self.assertEqual(record_refresh_mode({"refresh_mode": "continuous"}), "continuous")
        with tempfile.TemporaryDirectory() as directory:
            old = copy.deepcopy(result)
            for testbed in (
                old["run"]["protocol"]["matched"]["testbed_config"],
                old["run"]["protocol"]["explicit_config"]["testbed"],
                old["run"]["protocol"]["implicit_config"]["testbed"],
            ):
                testbed["controller"].pop("refresh_mode", None)
            new = dict(copy.deepcopy(result), refresh_mode="continuous")
            paths = []
            for name, record in (("old", old), ("new", new)):
                path = Path(directory) / name / RECORD_NAME
                path.parent.mkdir()
                from evaluations.ph_sac.reporting import _write_json

                _write_json({"units": {}, "result": record}, path)
                paths.append(str(path))
            loaded_old, loaded_new = load_run_records(paths)
        self.assertEqual(loaded_old["refresh_mode"], "held")
        self.assertEqual(
            loaded_old["run"]["protocol"]["matched"]["testbed_config"]["controller"]["refresh_mode"],
            "held",
        )
        self.assertEqual(loaded_new["refresh_mode"], "continuous")
        self.assertEqual(
            loaded_new["run"]["protocol"]["matched"]["testbed_config"]["controller"]["refresh_mode"],
            "continuous",
        )


# ------------------------------------------------------------------------- environment


def random_rows(env, rng, count, speed=2.0):
    """Interval-start states (reset law, random velocities) and box decisions."""
    rows = []
    for _ in range(count):
        env.reset(seed=int(rng.integers(1 << 30)))
        state = env.observe()
        state[env.position_dim :] = rng.uniform(-speed, speed, env.velocity_dim)
        rows.append((state, rng.uniform(env.action_space.low, env.action_space.high)))
    return rows


def held_reference_step(env, decision, duration):
    """The environment's step_dt before refresh_mode existed (refresh every physics step).

    Returns next state, reward, undiscounted reward and saturated fraction; no failure
    stop (the callers stay clear of the failure set).
    """
    decision = np.clip(np.asarray(decision, float), env.action_space.low, env.action_space.high)
    steps = int(round(duration / env.physics_dt))
    state = env.observe()
    resolved = env.controller.resolve_decision_numpy(state, decision)
    rates, times, saturated = np.empty(2 * steps), np.empty(2 * steps), 0
    for step in range(steps):
        torque = env.controller.action_numpy(state, resolved)
        rates[2 * step] = env.reward_rate(state, torque)
        times[2 * step] = step * env.physics_dt
        env.data.ctrl[:] = torque
        env._apply_disturbance()
        mujoco.mj_step(env.model, env.data)
        state = env.observe()
        rates[2 * step + 1] = env.reward_rate(state, torque)
        times[2 * step + 1] = (step + 1) * env.physics_dt
        saturated += int(np.any(torque <= env._torque_low) or np.any(torque >= env._torque_high))
    reward = env.integrate_interval_reward(rates, times, terminated=False, terminal_state=state)
    return state, reward, env.last_undiscounted_reward, saturated / steps


class TestContinuousEnvironment(unittest.TestCase):
    def tearDown(self):
        mujoco.set_mjcb_control(None)

    def test_callback_runs_at_every_rk4_stage_only_inside_steps(self):
        env = tasks.make_env(row_testbed("reacher-easy"), seed=0)
        self.assertTrue(env.stage_feedback)
        env.reset(seed=1)
        stages = []
        stage_control = env._stage_control

        def counting():
            stages.append((float(env.data.time), np.array(env.data.qpos[:2])))
            stage_control()

        env._stage_control = counting
        transition = env.step_dt(np.array([0.1, -0.1, 0.2, 0.2, 0.02, 0.02]), 0.02)
        self.assertFalse(transition["terminated"])
        self.assertEqual(len(stages), 4 * 20)
        # Stage times t, t + h/2, t + h/2, t + h of each physics step.
        times = np.array([time for time, _ in stages]).reshape(20, 4)
        offsets = times - times[:, :1]
        np.testing.assert_allclose(offsets, np.tile([0.0, 0.0005, 0.0005, 0.001], (20, 1)), atol=1e-12)
        self.assertIsNone(mujoco.get_mjcb_control())
        self.assertIsNone(env._stage_decision)
        self.assertEqual(transition["duration"], 0.02)
        # mj_forward outside a step never calls it.
        mujoco.mj_forward(env.model, env.data)
        env.mujoco_drift(env.observe(), np.zeros(2))
        self.assertEqual(len(stages), 80)

    def test_stage_torques_are_the_clipped_law_at_the_stage_state(self):
        env = tasks.make_env(row_testbed("cartpole-swingup"), seed=0)
        env.reset(seed=2)
        seen = []
        stage_control = env._stage_control

        def checking():
            stage_control()
            seen.append(
                (np.array(env.data.ctrl), env.controller.action_numpy(env.observe(), env._stage_decision))
            )

        env._stage_control = checking
        env.step_dt(np.array([0.05, 5.0, 0.5]), 0.01)  # inside the +-10 N clip
        self.assertEqual(len(seen), 40)
        for ctrl, law in seen:
            np.testing.assert_array_equal(ctrl, law)
        self.assertGreater(len({float(ctrl[0]) for ctrl, _ in seen}), 20)

    def test_held_mode_reproduces_sample_and_hold_bitwise(self):
        for env_id in ("cartpole-swingup", "reacher-easy", "lqr-lqr_6_2"):
            with self.subTest(env_id=env_id):
                env = tasks.make_env(row_testbed(env_id, "held"), seed=0)
                reference = tasks.make_env(row_testbed(env_id, "held"), seed=0)
                self.assertFalse(env.stage_feedback)
                rng = np.random.default_rng(4)
                for state, decision in random_rows(env, rng, 6, speed=0.5):
                    duration = env.config["reference_period"]
                    env.reset(options={"state": state})
                    reference.reset(options={"state": state})
                    transition = env.step_dt(decision, duration)
                    if transition["terminated"]:
                        continue
                    expected = held_reference_step(reference, decision, duration)
                    np.testing.assert_array_equal(transition["next_state"], expected[0])
                    self.assertEqual(transition["reward"], expected[1])
                    self.assertEqual(env.last_undiscounted_reward, expected[2])
                    self.assertEqual(env.last_interval["saturated_fraction"], expected[3])

    def test_continuous_differs_from_held_beyond_rounding(self):
        rng = np.random.default_rng(6)
        envs = {mode: tasks.make_env(row_testbed("acrobot-swingup", mode), seed=0) for mode in REFRESH_MODES}
        state, _ = random_rows(envs["held"], rng, 1, speed=1.0)[0]
        decision = np.array([0.1, 2.0, 0.5])  # inside the +-2 N m clip
        ends = {}
        for mode, env in envs.items():
            env.reset(options={"state": state})
            ends[mode] = env.step_dt(decision, 0.01)["next_state"]
        self.assertGreater(np.abs(ends["continuous"] - ends["held"]).max(), 1e-6)

    def test_euler_physics_and_torque_decisions_agree_bitwise(self):
        cases = (
            ("windy feedback", lambda mode: tasks.make_env(with_mode(windy(), mode), seed=3)),
            (
                "cartpole torque",
                lambda mode: tasks.make_env(
                    row_testbed("cartpole-swingup", mode), decision_mode="torque", seed=3
                ),
            ),
            (
                "windy torque",
                lambda mode: tasks.make_env(with_mode(windy(), mode), decision_mode="torque", seed=3),
            ),
        )
        for label, build in cases:
            with self.subTest(case=label):
                traces = {}
                for mode in REFRESH_MODES:
                    env = build(mode)
                    self.assertFalse(env.stage_feedback)
                    env.reset(seed=3)
                    rng = np.random.default_rng(9)
                    trace = []
                    for _ in range(150):
                        decision = rng.uniform(env.action_space.low, env.action_space.high)
                        next_state, reward, terminated, truncated, info = env.step(decision)
                        trace.append(
                            np.concatenate(
                                (next_state, [reward, info["dt"], info["saturated_fraction"],
                                              env.last_undiscounted_reward])
                            )
                        )
                        if terminated or truncated:
                            env.reset()
                    traces[mode] = np.array(trace)
                np.testing.assert_array_equal(traces["continuous"], traces["held"])

    def test_reward_knots_read_the_law_at_the_knot_state(self):
        env = tasks.make_env(row_testbed("cartpole-swingup"), seed=0)
        env.reset(seed=5)
        knots = []
        reward_rate = env.reward_rate

        def recording(state, torque):
            knots.append((np.array(state), np.array(torque)))
            return reward_rate(state, torque)

        env.reward_rate = recording
        decision = np.array([0.5, 100.0, 20.0])
        state = env.observe()
        resolved = env.controller.resolve_decision_numpy(state, decision)
        env.step_dt(decision, 0.01)
        self.assertEqual(len(knots), 20)
        for knot_state, torque in knots:
            np.testing.assert_array_equal(torque, env.controller.action_numpy(knot_state, resolved))
        # The end knot of a step and the start knot of the next are the same point.
        for step in range(9):
            np.testing.assert_array_equal(knots[2 * step + 1][0], knots[2 * step + 2][0])
            np.testing.assert_array_equal(knots[2 * step + 1][1], knots[2 * step + 2][1])

    def test_environments_and_raw_mujoco_do_not_interfere(self):
        testbed = row_testbed("reacher-easy")
        rng = np.random.default_rng(11)
        probe = tasks.make_env(testbed, seed=0)
        rows = random_rows(probe, rng, 6, speed=1.0)

        def run_alone(rows):
            env = tasks.make_env(testbed, seed=0)
            ends = []
            for state, decision in rows:
                env.reset(options={"state": state})
                ends.append(env.step_dt(decision, 0.02)["next_state"])
            return np.array(ends)

        alone_a, alone_b = run_alone(rows[:3]), run_alone(rows[3:])
        env_a, env_b = tasks.make_env(testbed, seed=0), tasks.make_env(testbed, seed=0)
        raw = mujoco.MjData(env_a.model)
        raw_reference = mujoco.MjData(env_a.model)
        foreign_calls = []

        def foreign(model, data):
            foreign_calls.append(data is raw)

        interleaved_a, interleaved_b = [], []
        for (state_a, decision_a), (state_b, decision_b) in zip(rows[:3], rows[3:]):
            env_a.reset(options={"state": state_a})
            env_b.reset(options={"state": state_b})
            # A callback installed by someone else: kept and restored around every step.
            mujoco.set_mjcb_control(foreign)
            interleaved_a.append(env_a.step_dt(decision_a, 0.02)["next_state"])
            self.assertIs(mujoco.get_mjcb_control(), foreign)
            # Raw stepping of another MjData sees only that callback.
            raw.ctrl[:] = 0.01
            mujoco.mj_step(env_a.model, raw)
            interleaved_b.append(env_b.step_dt(decision_b, 0.02)["next_state"])
            self.assertIs(mujoco.get_mjcb_control(), foreign)
            mujoco.set_mjcb_control(None)
        # The environments' own stages never reached it; raw's four stages per step did.
        self.assertEqual(foreign_calls, [True] * (3 * 4))
        np.testing.assert_array_equal(np.array(interleaved_a), alone_a)
        np.testing.assert_array_equal(np.array(interleaved_b), alone_b)
        mujoco.set_mjcb_control(None)
        raw_reference.ctrl[:] = 0.01
        for _ in range(3):
            mujoco.mj_step(env_a.model, raw_reference)
        np.testing.assert_array_equal(raw.qpos, raw_reference.qpos)
        np.testing.assert_array_equal(raw.qvel, raw_reference.qvel)

    def test_an_error_in_the_callback_leaves_nothing_installed(self):
        env = tasks.make_env(row_testbed("cartpole-swingup"), seed=0)
        env.reset(seed=0)

        def broken():
            raise RuntimeError("stage law failed")

        env._stage_control = broken
        with self.assertRaisesRegex(RuntimeError, "stage law failed"):
            env.step_dt(np.array([0.0, 10.0, 1.0]), 0.01)
        self.assertIsNone(mujoco.get_mjcb_control())
        self.assertIsNone(env._stage_decision)
        self.assertEqual(env_base._CALLBACK_USERS, 0)

    def test_snapshots_restore_continuous_intervals(self):
        env = tasks.make_env(row_testbed("acrobot-swingup"), seed=0)
        env.reset(seed=3)
        decision = np.array([0.2, 10.0, 2.0])
        snapshot = env.snapshot()
        first = env.step_dt(decision, 0.01)
        env.restore(snapshot)
        again = env.step_dt(decision, 0.01)
        np.testing.assert_array_equal(first["next_state"], again["next_state"])
        self.assertEqual(first["reward"], again["reward"])


# ------------------------------------------------------------------------------ model


def central_difference(function, states, epsilon=1e-6):
    """d function / d states [B,D_out,D] by central differences, row by row."""
    columns = []
    for index in range(states.shape[-1]):
        shift = torch.zeros_like(states)
        shift[:, index] = epsilon
        columns.append((function(states + shift) - function(states - shift)) / (2.0 * epsilon))
    return torch.stack(columns, dim=-1)


def learned_plant(spec, seed=4):
    torch.manual_seed(seed)
    network = dict(PLANT_NETWORK, damping_rank=int(spec["velocity_dim"]))
    plant = PortHamiltonianPlant(spec, network, learn_diffusion=True).double()
    with torch.no_grad():
        plant.mass_net.layers[-1].bias.fill_(-1.0)
    return plant


class TestContinuousModel(unittest.TestCase):
    def setUp(self):
        self.plant = coupled_plant()
        self.states = torch.tensor(
            [[0.4, -0.3, 1.0, -2.0], [2.0, 0.2, -1.0, 0.3], [-1.2, 0.8, 0.1, 0.4]],
            dtype=torch.float64,
        )
        # Weak springs, moderate damping: no stage reaches the +-4 clip.
        self.decisions = torch.tensor(
            [[0.1, 5.0, 0.3], [-0.2, 3.0, 0.5], [0.05, 8.0, 0.2]], dtype=torch.float64
        )

    def test_continuous_grid_has_no_refresh_points(self):
        durations = torch.tensor([[0.01], [0.03], [0.007]], dtype=torch.float64)
        counts, sizes, flags = refresh_aligned_grid(
            durations, 0.005, physics_dt=0.001, controller_refresh_steps=None
        )
        self.assertEqual(counts.tolist(), [2, 6, 2])
        self.assertTrue(torch.equal(sizes[0, :2], torch.full((2,), 5 * 0.001, dtype=torch.float64)))
        self.assertTrue(torch.equal(sizes[1], torch.full((6,), 5 * 0.001, dtype=torch.float64)))
        self.assertTrue(torch.equal(sizes[2, :2], torch.full((2,), 7 * 0.001 / 2, dtype=torch.float64)))
        self.assertEqual(flags.sum(dim=-1).tolist(), [1, 1, 1])
        self.assertTrue(flags[:, 0].all())
        # At the physics step the grid is the held grid of refresh_steps 1, exactly.
        one = refresh_aligned_grid(durations, 0.001, physics_dt=0.001, controller_refresh_steps=None)
        held = refresh_aligned_grid(durations, 0.001, physics_dt=0.001, controller_refresh_steps=1)
        self.assertTrue(torch.equal(one[0], held[0]) and torch.equal(one[1], held[1]))
        with self.assertRaisesRegex(ValueError, "physics_dt"):
            refresh_aligned_grid(durations, 0.001, controller_refresh_steps=1)
        with self.assertRaisesRegex(ValueError, "multiples"):
            refresh_aligned_grid(durations + 0.0004, 0.001, physics_dt=0.001)

    def test_rk4_stages_read_the_law_at_their_states(self):
        controller = continuous_controller()
        resolved = controller.resolve_decision(self.states, self.decisions)
        law = lambda points: controller.action(points, resolved)  # noqa: E731
        h = torch.tensor([0.002, 0.001, 0.005], dtype=torch.float64)[:, None]
        k1 = coupled_drift(self.states, law(self.states))
        point = self.states + 0.5 * h * k1
        k2 = coupled_drift(point, law(point))
        point = self.states + 0.5 * h * k2
        k3 = coupled_drift(point, law(point))
        point = self.states + h * k3
        k4 = coupled_drift(point, law(point))
        expected = self.states + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        for step in (rk4_step, lambda *a, **k: model_substep(*a, integrator="rk4", **k)):
            result = step(self.plant, self.states, law(self.states), h[:, 0], feedback=law)
            torch.testing.assert_close(result, expected, atol=1e-15, rtol=0.0)
        held = rk4_step(self.plant, self.states, law(self.states), h[:, 0])
        self.assertGreater((held - expected).abs().max().item(), 1e-9)
        # The semi-implicit map has one stage: feedback changes nothing.
        semi = model_substep(
            self.plant, self.states, law(self.states), h[:, 0], integrator="semi_implicit", feedback=law
        )
        torch.testing.assert_close(
            semi,
            model_substep(self.plant, self.states, law(self.states), h[:, 0], integrator="semi_implicit"),
            atol=0.0,
            rtol=0.0,
        )

    def check_jacobian(self, plant, controller, states, decisions, integrator):
        resolved = controller.resolve_decision(states, decisions)
        h = torch.full((states.shape[0],), 0.004, dtype=torch.float64)
        mask = torch.ones(states.shape[0], dtype=torch.bool)
        held = torch.zeros(states.shape[0], controller.action_dim, dtype=torch.float64)
        following, torque, jacobian = augmented_closed_loop_step(
            plant, controller, states, held, resolved, h, mask, integrator=integrator
        )
        law = lambda points: controller.action(points, resolved)  # noqa: E731

        def step(points):
            return model_substep(plant, points, law(points), h, integrator=integrator, feedback=law)

        torch.testing.assert_close(following, step(states), atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(torque, law(states), atol=0.0, rtol=0.0)
        size = states.shape[-1]
        numeric = central_difference(step, states)
        analytic = jacobian[:, :size, :size]
        error = (analytic - numeric).abs().max().item()
        self.assertLess(error, 1e-8 * max(1.0, numeric.abs().max().item()))
        if integrator == "rk4":
            self.assertEqual(jacobian[:, :size, size:].abs().max().item(), 0.0)
            self.assertEqual(jacobian[:, size:].abs().max().item(), 0.0)
        return analytic

    def test_closed_loop_step_jacobians_match_central_differences(self):
        controller = continuous_controller()
        for integrator in ("rk4", "semi_implicit"):
            with self.subTest(plant="analytic", integrator=integrator):
                self.check_jacobian(self.plant, controller, self.states, self.decisions, integrator)
        testbed = row_testbed("reacher-easy")
        spec = testbed["state_spec"]
        reacher = FeedbackController(testbed["controller"], spec)
        rng = np.random.default_rng(2)
        states = torch.as_tensor(
            np.concatenate(
                (rng.uniform(-1.0, 1.0, (4, 2)), rng.uniform(-0.2, 0.2, (4, 2)), rng.uniform(-0.3, 0.3, (4, 2))),
                axis=-1,
            )
        )
        decisions = torch.as_tensor(
            np.concatenate(
                (rng.uniform(-0.05, 0.05, (4, 2)), rng.uniform(0.0, 0.05, (4, 2)), rng.uniform(0.0, 0.02, (4, 2))),
                axis=-1,
            )
        )
        plant = learned_plant(spec)
        for integrator in ("rk4", "semi_implicit"):
            with self.subTest(plant="learned", integrator=integrator):
                self.check_jacobian(plant, reacher, states, decisions, integrator)

    def test_semi_implicit_continuous_step_is_the_refreshed_held_step(self):
        resolved = held_controller().resolve_decision(self.states, self.decisions)
        h = torch.full((3,), 0.002, dtype=torch.float64)
        mask = torch.ones(3, dtype=torch.bool)
        held = torch.zeros(3, 1, dtype=torch.float64)
        outputs = [
            augmented_closed_loop_step(
                self.plant, controller, self.states, held, resolved, h, mask, integrator="semi_implicit"
            )
            for controller in (continuous_controller(), held_controller())
        ]
        for left, right in zip(*outputs):
            self.assertTrue(torch.equal(left, right))

    def test_path_jacobian_norms_carry_the_law_at_every_substep(self):
        controller = continuous_controller()
        durations = torch.full((3, 1), 0.004, dtype=torch.float64)
        grid = {"maximum_step": 0.001, "physics_dt": 0.001, "controller_refresh_steps": 2}
        norms = path_jacobian_norms(
            self.plant, controller, self.states, self.decisions, durations, integrator="rk4", **grid
        )
        resolved = controller.resolve_decision(self.states, self.decisions)
        law = lambda points: controller.action(points, resolved)  # noqa: E731
        current, largest = self.states, torch.zeros(3, dtype=torch.float64)
        h = torch.full((3,), 0.001, dtype=torch.float64)
        for _ in range(4):
            _, by_state, by_action = self.plant.drift_jacobian(current, law(current))
            closed = by_state + by_action @ _action_state_jacobian(controller, current, resolved)
            largest = torch.max(largest, torch.linalg.matrix_norm(closed, ord=2))
            current = rk4_step(self.plant, current, law(current), h, feedback=law)
        torch.testing.assert_close(norms[:, 0], 0.004 * largest, atol=1e-14, rtol=1e-12)

    def test_rollout_rewards_read_the_law_at_the_knots(self):
        controller = continuous_controller()
        durations = torch.full((3, 1), 0.003, dtype=torch.float64)
        resolved = controller.resolve_decision(self.states, self.decisions)
        law = lambda points: controller.action(points, resolved)  # noqa: E731
        rate = lambda s, a: -(s[:, :2].square().sum(-1, keepdim=True)) - 0.5 * a.square()  # noqa: E731
        terms = {
            "reward_rate": rate,
            "terminal_payoff": lambda s: torch.zeros(s.shape[0], 1, dtype=s.dtype),
            "discount_rate": 0.5,
        }
        paths = stochastic_rollout(
            self.plant,
            controller,
            self.states,
            self.decisions,
            durations,
            lambda s: torch.zeros(s.shape[0], dtype=torch.bool),
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            brownian_increments=torch.zeros(3, 1, 3, 2, dtype=torch.float64),
            return_terms=terms,
            integrator="rk4",
        )
        current, running = self.states, torch.zeros(3, dtype=torch.float64)
        h = torch.full((3,), 0.001, dtype=torch.float64)
        for step in range(3):
            following = rk4_step(self.plant, current, law(current), h, feedback=law)
            before = math.exp(-0.5 * 0.001 * step) * rate(current, law(current))[:, 0]
            after = math.exp(-0.5 * 0.001 * (step + 1)) * rate(following, law(following))[:, 0]
            running = running + 0.0005 * (before + after)
            current = following
        torch.testing.assert_close(paths["endpoints"][:, 0], current, atol=1e-15, rtol=0.0)
        torch.testing.assert_close(paths["running_rewards"][:, 0, 0], running, atol=1e-15, rtol=1e-13)

    def test_model_step_above_the_physics_step(self):
        controller = continuous_controller()
        durations = torch.full((3, 1), 0.01, dtype=torch.float64)
        means = {}
        for step in (0.001, 0.005, 0.01):
            moments = integrate_endpoint_moments(
                self.plant,
                controller,
                self.states,
                self.decisions,
                durations,
                maximum_step=step,
                physics_dt=0.001,
                controller_refresh_steps=1,
                covariance_mode="propagated",
                create_graph=False,
                integrator="rk4",
            )
            self.assertEqual(moments["step_counts"].tolist(), [round(0.01 / step)] * 3)
            self.assertTrue(torch.isfinite(moments["covariances"]).all())
            means[step] = moments["means"]
        displacement = (means[0.001] - self.states).abs().max().item()
        self.assertLess((means[0.005] - means[0.001]).abs().max().item(), 1e-6 * displacement)
        self.assertLess((means[0.01] - means[0.001]).abs().max().item(), 1e-5 * displacement)
        # Under "held" a model step above the refresh period has no effect.
        held = [
            integrate_endpoint_moments(
                self.plant,
                held_controller(),
                self.states,
                self.decisions,
                durations,
                maximum_step=step,
                physics_dt=0.001,
                controller_refresh_steps=1,
                covariance_mode="leading_order",
                create_graph=False,
                integrator="rk4",
            )["means"]
            for step in (0.001, 0.005)
        ]
        self.assertTrue(torch.equal(held[0], held[1]))


# ------------------------------------------------------------------- held regression

HELD_ARMS = (
    "implicit_learned_admissions",
    "implicit_gated_admissions",
    "explicit_rollout",
    "explicit_generator",
    "implicit_torque",
)
HELD_DECISIONS = 150
# The row of the held regression: a task whose dynamics the soft-joint-limit change of
# 2026-11-10 left alone (until then cartpole-swingup).
HELD_ENV_ID = "pendulum-swingup"
# float32 training: bit-identical on one machine (the code at the change reproduced these
# exactly); another BLAS may round differently, so a relative tolerance as in
# tests/test_ph_torque_decisions.py.
HELD_TOLERANCE = {"rtol": 1e-7, "atol": 1e-9}


def held_protocol(env_id):
    """A tiny protocol of a table row with every controller in refresh_mode "held"."""
    raw = copy.deepcopy(load_ph_hyperparams_from_table(env_id, "top"))
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    raw["training"].update(batch_size=8, replay_capacity=5000, warmup_episodes=0)
    raw["explicit"].update(rollout_paths=3)
    implicit = raw["implicit"]
    implicit.update(reward_warm_steps=5, plant_warm_steps=5, temperature_every_rows=20)
    implicit["solver"].update(inner_steps=6, policy_draws=3)
    implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
    implicit["admission"].update(
        initial_rows=8, initial_paths=2, pilot_batches=2, real_row_budget=64,
        simulated_path_budget=4096, efolds=0.3,
    )
    raw["protocol"].update(
        interaction_budget=HELD_DECISIONS, training_seeds=[0], evaluation_reset_seeds=[1, 2],
        output_directory="/nonexistent",
    )
    proto = protocol_from_config(raw, env_id)
    for testbed in (
        proto["matched"]["testbed_config"],
        proto["explicit_config"]["testbed"],
        proto["implicit_config"]["testbed"],
    ):
        testbed["controller"]["refresh_mode"] = "held"
    proto["arms"] = list(P.all_arms())
    return proto


def held_training_summary(proto, arm_name):
    """Train one arm for HELD_DECISIONS; sums of its transitions and network parameters."""
    run = next(run for run in P.build_run_matrix(proto) if run["arm"]["name"] == arm_name)
    env = tasks.make_env(
        proto["matched"]["testbed_config"], decision_mode=run["arm"]["decision_mode"], seed=0
    )
    rows = []
    step_dt = env.step_dt

    def recording(decision, duration):
        transition = step_dt(decision, duration)
        rows.append(
            np.concatenate(
                (transition["state"], transition["decision"],
                 [transition["reward"], transition["duration"]], transition["next_state"])
            )
        )
        return transition

    env.step_dt = recording
    agent = P.make_training_agent(run, env)
    agent.learn(HELD_DECISIONS)
    rows = np.array(rows)
    summary = {"transitions": (float(rows.sum()), float(np.square(rows).sum()))}
    for name, module in sorted(agent._network_modules().items()):
        if module is None:
            continue
        values = torch.cat([p.detach().double().reshape(-1) for p in module.parameters()])
        summary[name] = (float(values.sum()), float(values.square().sum()))
    return summary


# pendulum-swingup (since 2026-11-10; cartpole-swingup until then, whose dynamics changed
# convention on 2026-11-10: dm_control's soft rail limit kept, no failure set, the plant's
# limit port on, so the old tree's cart-pole values no longer apply). Computed with the
# code before refresh_mode existed (2026-11-09; a copy of the source tree as it was, running
# held_training_summary above with the controller key ignored) and reproduced bitwise by
# the code of 2026-11-10 in held mode.
HELD_REFERENCE = {'explicit_generator': {'actor': (-0.6728759799152613, 15.873538370604116),
                        'critics': (3.7498672334477305, 25.4603736569261),
                        'online_plant': (-2.801415044697933, 23.4966137308003),
                        'target_critics': (3.6173508175706957, 25.51690384814754),
                        'target_plant': (-2.911629802954849, 23.005128486444804),
                        'target_value': (1.5228797756353742, 11.87872598591929),
                        'transitions': (620.4119992261053, 3656.375988331674),
                        'value': (1.5111124075338012, 11.859286644776247)},
 'explicit_rollout': {'actor': (-0.6700150077231228, 15.87612217699581),
                      'critics': (3.749811226152815, 25.460382550747667),
                      'online_plant': (-2.8014144033077173, 23.496610413149817),
                      'target_critics': (3.6173078962310683, 25.516924170875818),
                      'target_plant': (-2.911629105656175, 23.00512793325224),
                      'target_value': (1.5228927751086303, 11.878703838609605),
                      'transitions': (620.5871636690296, 3656.677922848483),
                      'value': (1.511160101450514, 11.859200296085787)},
 'implicit_gated_admissions': {'actor': (-1.6602716753259301, 14.866070263381157),
                               'admitted_plant': (7.871009481110377, 18.695811316471033),
                               'online_plant': (7.898494779277826, 18.732042856387938),
                               'rate': (4.800946555566043, 12.716742651563212),
                               'reward': (-1.4605366831528954, 12.611198567661875),
                               'reward_frozen': (-1.4545542911509983, 12.609142908707335),
                               'target_value': (-3.0846087897516554, 13.61567125062219),
                               'transitions': (270.0224103565706, 3392.462027851825),
                               'value': (-3.0846087897516554, 13.61567125062219)},
 'implicit_learned_admissions': {'actor': (-1.731437178561464, 14.814654640889954),
                                 'admitted_plant': (7.788949843030423, 19.325766064039847),
                                 'online_plant': (7.779731423128396, 19.37938015211675),
                                 'rate': (5.076195936708245, 12.717580339199515),
                                 'reward': (-1.7273970388341695, 13.273776221415096),
                                 'reward_frozen': (-1.716374090348836, 13.240425510509105),
                                 'target_value': (-3.2254896380800346, 14.113779515375029),
                                 'transitions': (397.88529623875286, 3410.41610291301),
                                 'value': (-3.2254896380800346, 14.113779515375029)},
 'implicit_torque': {'actor': (0.7000677337055095, 12.348694150396977),
                     'admitted_plant': (9.745467509841546, 21.56959050719987),
                     'online_plant': (9.751122523099184, 21.60887361657208),
                     'rate': (2.0884918649389874, 12.393594396242078),
                     'reward': (-1.8481224583520088, 13.365697545409951),
                     'reward_frozen': (-1.8358696723007597, 13.32939728159711),
                     'target_value': (-0.10295834063435905, 14.47119917289787),
                     'transitions': (32.61858976300887, 3086.9479689745585),
                     'value': (-0.10295834063435905, 14.47119917289787)}}


class TestHeldRegression(unittest.TestCase):
    def test_held_training_reproduces_the_earlier_code(self):
        proto = held_protocol(HELD_ENV_ID)
        for arm in HELD_ARMS:
            with self.subTest(arm=arm):
                summary = held_training_summary(proto, arm)
                expected = HELD_REFERENCE[arm]
                self.assertEqual(set(summary), set(expected))
                for name, values in expected.items():
                    np.testing.assert_allclose(summary[name], values, **HELD_TOLERANCE, err_msg=name)


if __name__ == "__main__":
    unittest.main()
