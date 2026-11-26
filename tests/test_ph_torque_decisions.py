"""Torque decisions: the paper arms' decisions are held torques.

decision_mode "torque" (common.ph_config.DECISION_MODES) replaces the impedance law by the
identity: a decision is the motor torque, held over the interval as the CT-SAC baseline
holds it. Since 2026-11-13 the paper arms (protocol.paper_arms) take torque decisions and
the impedance law is an ablation (protocol.impedance_decision_arms).

1. Config: torque_decision_testbed / torque_decision_config give a decision box of
   dimension m equal to the torque box, validation checks it, and the entropy target is
   -1 per torque coordinate (-m).
2. Controller and environment: TorqueController is the clipped identity, never
   refreshed after the interval start; a torque-mode environment carries the torque
   config, the identity controller and the torque action space.
3. Model integration: the torque is evaluated at the interval's first substep only and
   held through every substep and RK4 stage; the oracle plant under RK4 reproduces the
   torque-mode environment's endpoints to rounding.
4. Agents: short implicit and explicit runs through the protocol's run_seed complete
   with finite values and write run records carrying decision_mode "torque"; records
   written before the field existed load with their arm's mode, else "feedback".
5. Arms: implicit_torque, explicit_rollout_torque and explicit_generator_torque each
   change only decision_mode from their impedance-law counterpart, form the default arm
   set with ct_sac_torque, and --list labels them "paper"; the impedance-law arms run
   when --arms names them. The seven arm names of the runs made before the default
   changed keep their specs, configs and resume fingerprints (TestFixedArmNames).
6. Feedback arms unaffected: their configs, boxes and controllers are the protocol's,
   and short feedback training runs and the RK4 feedback model integration reproduce
   values computed with the code before the torque mode existed.
"""

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import get_args

import numpy as np
import torch

from benchmarks.ph_hyperparams import find_ph_entry
from benchmarks.run_ph_sac import (
    ABLATION_ARM_EXAMPLE,
    apply_overrides,
    list_arms,
    load_protocol,
    load_protocols,
    main,
    parse_args,
    protocol_from_config,
)
from common.ph_config import (
    DECISION_MODES,
    STEP_ABLATION_CHOICES,
    decision_box_entropy_target,
    decision_dim,
    decision_mode_setting,
    torque_decision_config,
    torque_decision_controller,
    torque_decision_testbed,
    validate_configuration,
)
from environment.ph_sac import tasks
from environment.ph_sac.controller import (
    FeedbackController,
    TorqueController,
    make_controller,
)
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac import protocol
from evaluations.ph_sac import resume
from evaluations.ph_sac.reporting import RECORD_NAME, load_run_records, record_decision_mode
from models.ph_sac.rollouts import (
    controller_refresh_flags,
    integrate_endpoint_moments,
    path_jacobian_norms,
    refresh_aligned_grid,
    stochastic_rollout,
    wrapped_endpoint_residual,
)
from tests.test_ph_algorithms import (
    acrobot,
    explicit_config,
    implicit_config,
    make_explicit,
    make_implicit,
    windy,
)
from tests.test_ph_model_integrator import (
    GRID,
    ORACLE_TASKS,
    ORACLE_TOLERANCE,
    REGRESSION_TOLERANCE,
    coupled_batch,
    coupled_controller,
    coupled_failure,
    coupled_increments,
    coupled_plant,
    coupled_return_terms,
    table_testbed,
)

CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac"
TORQUE_ARMS = {
    "implicit_torque": "implicit_learned_admissions",
    "explicit_rollout_torque": "explicit_rollout",
    "explicit_generator_torque": "explicit_generator",
}


def table_config(env_id):
    """The TestbedConfig of a table row (top mode)."""
    settings = find_ph_entry(env_id, "top").config["testbed_settings"]
    name = find_ph_entry(env_id, "top").config["testbed"]
    return testbed_spec(name).config_factory(**settings)


def smoke_protocol():
    return load_protocol(str(CONFIGS / "smoke_windy.json"))


def runs_by_arm(proto):
    return {run["arm"]["name"]: run for run in protocol.build_run_matrix(proto)}


def arm_config(run):
    if run["arm"]["algorithm"] == "implicit":
        return protocol._implicit_arm_config(run)
    return protocol._explicit_arm_config(run)


# ------------------------------------------------------------------------------ config


class TestTorqueDecisionConfig(unittest.TestCase):
    TESTBEDS = {
        "windy_pendulum": windy,
        "acrobot_xk": acrobot,
        "cartpole_swingup": lambda: table_config("cartpole-swingup"),
        "reacher_easy": lambda: table_config("reacher-easy"),
    }

    def test_decision_box_is_the_torque_box_of_dimension_m(self):
        for name, build in self.TESTBEDS.items():
            with self.subTest(testbed=name):
                testbed = build()
                before = copy.deepcopy(testbed["controller"])
                m = testbed["state_spec"]["action_dim"]
                converted = torque_decision_testbed(testbed)
                controller = converted["controller"]
                self.assertEqual(decision_mode_setting(controller), "torque")
                self.assertEqual(decision_dim(controller, m), m)
                np.testing.assert_array_equal(controller["decision_low"], before["torque_low"])
                np.testing.assert_array_equal(controller["decision_high"], before["torque_high"])
                self.assertEqual(controller["decision_low"].shape, (m,))
                for key in ("torque_low", "torque_high"):
                    np.testing.assert_array_equal(controller[key], before[key])
                for key in ("setpoint_mode", "setpoint_error", "refresh_steps"):
                    self.assertEqual(controller[key], before[key])
                # Everything but the controller is the same object; the input is untouched.
                for key in testbed:
                    if key != "controller":
                        self.assertIs(converted[key], testbed[key])
                self.assertNotIn("decision_mode", testbed["controller"])
                self.assertEqual(decision_mode_setting(testbed["controller"]), "feedback")
                self.assertEqual(decision_dim(testbed["controller"], m), 3 * m)
                np.testing.assert_array_equal(
                    testbed["controller"]["decision_low"], before["decision_low"]
                )
                validate_configuration(torque_decision_config(explicit_config(testbed)))

    def test_conversion_is_idempotent(self):
        config = torque_decision_config(implicit_config())
        again = torque_decision_config(config)
        self.assertEqual(again["temperature"], config["temperature"])
        np.testing.assert_array_equal(
            again["testbed"]["controller"]["decision_low"],
            config["testbed"]["controller"]["decision_low"],
        )
        controller = config["testbed"]["controller"]
        self.assertEqual(torque_decision_controller(controller), dict(controller))

    def test_validation_checks_the_torque_box(self):
        config = torque_decision_config(explicit_config())
        validate_configuration(config)
        controller = config["testbed"]["controller"]

        def rejected(changes, message):
            broken = copy.deepcopy(config)
            broken["testbed"]["controller"] = dict(controller, **changes)
            with self.assertRaisesRegex(ValueError, message):
                validate_configuration(broken)

        rejected({"decision_low": np.array([-12.0, 0.0, 0.0]),
                  "decision_high": np.array([12.0, 1.0, 1.0])}, "one torque per actuator")
        rejected({"decision_low": np.array([-6.0]), "decision_high": np.array([6.0])},
                 "must be the torque box")
        rejected({"decision_mode": "impedance"}, "decision_mode must be one of")
        self.assertEqual(DECISION_MODES, ("feedback", "torque"))

    def test_entropy_target_is_minus_one_per_torque_coordinate(self):
        for name, build in self.TESTBEDS.items():
            with self.subTest(testbed=name):
                testbed = build()
                m = testbed["state_spec"]["action_dim"]
                config = explicit_config(testbed)
                config["temperature"]["target_entropy"] = decision_box_entropy_target(
                    testbed["controller"]
                )
                self.assertEqual(config["temperature"]["target_entropy"], -3.0 * m)
                converted = torque_decision_config(config)
                self.assertEqual(converted["temperature"]["target_entropy"], -float(m))
                self.assertEqual(
                    decision_box_entropy_target(converted["testbed"]["controller"]), -float(m)
                )
                # The rest of the temperature and the input config are unchanged.
                self.assertEqual(config["temperature"]["target_entropy"], -3.0 * m)
                self.assertEqual(
                    {k: v for k, v in converted["temperature"].items() if k != "target_entropy"},
                    {k: v for k, v in config["temperature"].items() if k != "target_entropy"},
                )


# ----------------------------------------------------------- controller and environment


class TestTorqueController(unittest.TestCase):
    def setUp(self):
        testbed = table_config("reacher-easy")
        self.spec = testbed["state_spec"]
        self.config = torque_decision_controller(testbed["controller"])
        self.controller = make_controller(self.config, self.spec)
        self.limit = float(self.config["torque_high"][0])

    def test_action_is_the_clipped_identity_whatever_the_state(self):
        controller = self.controller
        self.assertIsInstance(controller, TorqueController)
        self.assertFalse(controller.state_feedback)
        self.assertEqual(controller.decision_dim, 2)
        generator = torch.Generator().manual_seed(0)
        states = torch.randn(64, 8, generator=generator, dtype=torch.float64)
        others = 10.0 * torch.randn(64, 8, generator=generator, dtype=torch.float64)
        decisions = 3.0 * self.limit * (
            2.0 * torch.rand(64, 2, generator=generator, dtype=torch.float64) - 1.0
        )
        resolved = controller.resolve_decision(states, decisions)
        self.assertIs(resolved, decisions)
        torques = controller.action(states, resolved)
        expected = decisions.clamp(-self.limit, self.limit)
        self.assertTrue(torch.equal(torques, expected))
        self.assertTrue(torch.equal(controller.action(others, resolved), expected))
        inside = decisions.clamp(-self.limit, self.limit)
        self.assertTrue(torch.equal(controller.action(states, inside), inside))
        np.testing.assert_array_equal(
            controller.action_numpy(states.numpy()[0], controller.resolve_decision_numpy(
                states.numpy()[0], decisions.numpy()[0])),
            expected.numpy()[0],
        )
        # Policy draws [B,N,m] at states [B,N,D], and one decision broadcast over states.
        draws = decisions.reshape(16, 4, 2)
        flat = controller.action(states.reshape(16, 4, 8), draws)
        self.assertTrue(torch.equal(flat, expected.reshape(16, 4, 2)))
        self.assertEqual(controller.action(states, decisions[:1]).shape, (64, 2))
        energy = controller.virtual_spring_energy(states, decisions)
        self.assertEqual(energy.shape, (64, 1))
        self.assertTrue(torch.equal(energy, torch.zeros_like(energy)))

    def test_the_torque_is_never_refreshed(self):
        controller = self.controller
        self.assertTrue(controller.refresh_due(0))
        self.assertFalse(any(controller.refresh_due(step) for step in range(1, 40)))
        self.assertEqual(controller.refresh_times(0.02, 0.001), (0.0,))
        self.assertEqual(controller.refresh_steps, self.config["refresh_steps"])

    def test_each_controller_requires_its_mode(self):
        feedback = table_config("reacher-easy")["controller"]
        self.assertIsInstance(make_controller(feedback, self.spec), FeedbackController)
        self.assertTrue(make_controller(feedback, self.spec).state_feedback)
        with self.assertRaisesRegex(ValueError, "feedback-decision"):
            FeedbackController(self.config, self.spec)
        with self.assertRaisesRegex(ValueError, "torque-decision"):
            TorqueController(feedback, self.spec)
        narrow = dict(self.config, decision_high=0.5 * self.config["torque_high"])
        with self.assertRaisesRegex(ValueError, "torque box"):
            TorqueController(narrow, self.spec)

    def test_torque_environment_carries_the_torque_config(self):
        testbed = table_config("cartpole-swingup")
        env = tasks.make_env(testbed, decision_mode="torque", seed=0)
        self.assertEqual(env.decision_mode, "torque")
        self.assertIsInstance(env.controller, TorqueController)
        self.assertEqual(decision_mode_setting(env.config["controller"]), "torque")
        np.testing.assert_array_equal(env.action_space.low, testbed["controller"]["torque_low"])
        np.testing.assert_array_equal(env.action_space.high, testbed["controller"]["torque_high"])
        np.testing.assert_array_equal(
            env.config["controller"]["decision_low"], env.action_space.low
        )
        self.assertNotIn("decision_mode", testbed["controller"])
        # A torque config builds the same environment and refuses feedback mode.
        again = tasks.make_env(env.config, decision_mode="torque", seed=0)
        np.testing.assert_array_equal(again.action_space.high, env.action_space.high)
        with self.assertRaisesRegex(ValueError, "decision_mode='torque'"):
            tasks.make_env(env.config, decision_mode="feedback", seed=0)
        # Feedback mode keeps the given config object and the impedance law.
        feedback = tasks.make_env(testbed, decision_mode="feedback", seed=0)
        self.assertIs(feedback.config, testbed)
        self.assertIsInstance(feedback.controller, FeedbackController)
        np.testing.assert_array_equal(
            feedback.action_space.low, testbed["controller"]["decision_low"]
        )

    def test_environment_holds_the_clipped_torque(self):
        env = tasks.make_env(windy(), decision_mode="torque", seed=0)
        env.reset(seed=0)
        torques = []
        physics_step = env._physics_step

        def spy(torque):
            torques.append(np.array(torque))
            physics_step(torque)

        env._physics_step = spy
        transition = env.step_dt(np.array([30.0]), 0.012)
        self.assertEqual(transition["decision"].tolist(), [12.0])
        self.assertEqual(len(torques), 12)
        self.assertTrue(all(t.tolist() == [12.0] for t in torques))

    def test_agents_require_the_environment_mode(self):
        config = explicit_config()
        torque_env = tasks.make_env(config["testbed"], decision_mode="torque", seed=0)
        from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
        from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC

        with self.assertRaisesRegex(ValueError, "torque_decision_config"):
            ExplicitCTPHSAC(torque_env, config)
        feedback_env = tasks.make_env(config["testbed"], decision_mode="feedback", seed=0)
        with self.assertRaisesRegex(ValueError, "takes torque decisions but the environment"):
            ImplicitCTPHSAC(feedback_env, torque_decision_config(implicit_config()))


# ------------------------------------------------------------------- model integration


class _Recorder:
    """Forward to a plant, recording the torques of every drift evaluation."""

    def __init__(self, plant):
        self.inner = plant
        self.seen = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def drift(self, states, actions):
        self.seen.append(actions.detach().clone())
        return self.inner.drift(states, actions)

    def drift_jacobian(self, states, actions):
        self.seen.append(actions.detach().clone())
        return self.inner.drift_jacobian(states, actions)


def torque_trials(name, *, seed=0, count=16):
    """Random states and torques in the box with the torque-mode environment's endpoints."""
    _, position_ranges, speed = ORACLE_TASKS[name]
    config = table_testbed(name)
    env = tasks.make_env(config, decision_mode="torque", seed=seed)
    plant = tasks.make_oracle_plant(env).double()
    rng = np.random.default_rng(seed)
    physical = len(position_ranges)
    velocities = env.state_spec["velocity_dim"]
    duration = config["reference_period"]
    rows = []
    for trial in range(count):
        env.reset(seed=seed + trial)
        state = env.observe()
        state[:physical] = rng.uniform(-1.0, 1.0, physical) * np.asarray(position_ranges)
        state[-velocities:] = rng.uniform(-1.0, 1.0, velocities) * speed
        # Half the rows hold a torque on a limit, the rest inside the box.
        torque = rng.uniform(env.action_space.low, env.action_space.high)
        if trial % 2:
            on_low = rng.uniform(size=torque.shape) < 0.5
            torque = np.where(on_low, env.action_space.low, env.action_space.high)
        env.reset(options={"state": state})
        state = env.observe()
        transition = env.step_dt(torque, duration)
        if transition["terminated"]:
            raise AssertionError(f"{name}: trial {trial} failed inside the interval.")
        rows.append((state, torque, transition["next_state"]))
    states, torques, observed = (
        torch.as_tensor(np.array(column), dtype=torch.float64) for column in zip(*rows)
    )
    durations = torch.full((count, 1), duration, dtype=torch.float64)
    return env, plant, states, torques, durations, observed


def grid(env):
    return {
        "maximum_step": env.physics_dt,
        "physics_dt": env.physics_dt,
        "controller_refresh_steps": env.controller.refresh_steps,
    }


class TestTorqueModelIntegration(unittest.TestCase):
    def test_oracle_rk4_reproduces_the_torque_environment(self):
        for name in ORACLE_TASKS:
            with self.subTest(testbed=name):
                env, plant, states, torques, durations, observed = torque_trials(name)
                self.assertIsInstance(env.controller, TorqueController)
                means = integrate_endpoint_moments(
                    plant, env.controller, states, torques, durations,
                    covariance_mode="leading_order", create_graph=False, integrator="rk4",
                    **grid(env),
                )["means"]
                residual = wrapped_endpoint_residual(observed, means, env.state_spec)
                self.assertLess(residual.abs().max().item(), ORACLE_TOLERANCE)
                steps = int(round(env.config["reference_period"] / env.physics_dt))
                noise = torch.zeros(states.shape[0], 2, steps, plant.noise_dim).double()
                paths = stochastic_rollout(
                    plant, env.controller, states, torques, durations,
                    lambda s: tasks.failure_mask_tensor(s, env.config), path_count=2,
                    brownian_increments=noise, integrator="rk4", **grid(env),
                )
                for path in range(2):
                    residual = wrapped_endpoint_residual(
                        observed, paths["endpoints"][:, path], env.state_spec
                    )
                    self.assertLess(residual.abs().max().item(), ORACLE_TOLERANCE)

    def test_torque_is_set_once_and_held_through_every_substep_and_stage(self):
        env, plant, states, torques, durations, _ = torque_trials("cartpole_swingup", count=6)
        decisions = 2.0 * torques  # half of them outside the box: the clip must hold
        expected = env.controller.action(states, decisions)
        controller = env.controller
        calls = []
        action = controller.action

        def counted(states_, decisions_):
            calls.append(states_.shape[0])
            return action(states_, decisions_)

        controller.action = counted
        try:
            for mode in ("leading_order", "propagated"):
                with self.subTest(covariance_mode=mode):
                    recorder = _Recorder(plant)
                    calls.clear()
                    moments = integrate_endpoint_moments(
                        recorder, controller, states, decisions, durations,
                        covariance_mode=mode, create_graph=False, integrator="rk4",
                        **grid(env),
                    )
                    self.assertEqual(len(calls), 1)
                    # Four RK4 stages in each of the interval's one-physics-step substeps.
                    steps = int(round(env.config["reference_period"] / env.physics_dt))
                    self.assertEqual(len(recorder.seen), 4 * steps)
                    for seen in recorder.seen:
                        self.assertTrue(torch.equal(seen, expected))
                    self.assertTrue(torch.isfinite(moments["covariances"]).all())
                    limit = float(env.config["controller"]["torque_high"][0])
                    saturated = (expected.abs() >= limit).double()
                    self.assertTrue(torch.equal(moments["saturation_fractions"], saturated))
            recorder = _Recorder(plant)
            calls.clear()
            stochastic_rollout(
                recorder, controller, states, decisions, durations,
                lambda s: tasks.failure_mask_tensor(s, env.config), path_count=2,
                integrator="rk4", **grid(env),
            )
            self.assertEqual(len(calls), 1)
            per_path = expected.repeat_interleave(2, 0)
            self.assertTrue(all(torch.equal(seen, per_path) for seen in recorder.seen))
            norms = path_jacobian_norms(
                plant, controller, states, decisions, durations, integrator="rk4", **grid(env)
            )
            self.assertTrue(torch.isfinite(norms).all())
        finally:
            controller.action = action

    def test_propagated_covariance_of_a_zero_torque_is_the_zero_gain_laws(self):
        """A held zero torque and a feedback law with zero gains drive the same path.

        The torque mode holds the torque as an augmented state with identity dynamics
        and no state feedback; with no noise on it, the physical covariance is the
        feedback law's, whose zero gains give zero torque and zero feedback Jacobian.
        """
        plant = coupled_plant()
        states, decisions, durations = coupled_batch()
        feedback = coupled_controller()
        torque = make_controller(
            torque_decision_controller(feedback.config), feedback.state_spec
        )
        zero_gains = decisions.clone()
        zero_gains[:, 1:] = 0.0
        held_zero = torch.zeros(states.shape[0], 1, dtype=torch.float64)
        for integrator in ("rk4", "semi_implicit"):
            with self.subTest(integrator=integrator):
                outputs = [
                    integrate_endpoint_moments(
                        plant, controller, states, chosen, durations,
                        covariance_mode="propagated", create_graph=False,
                        integrator=integrator, **GRID,
                    )
                    for controller, chosen in ((feedback, zero_gains), (torque, held_zero))
                ]
                for key in ("means", "covariances"):
                    np.testing.assert_allclose(
                        outputs[1][key].numpy(), outputs[0][key].numpy(),
                        rtol=1e-12, atol=1e-15, err_msg=key,
                    )
                held = torch.tensor([[4.0], [-1.5], [0.25]], dtype=torch.float64)
                moments = integrate_endpoint_moments(
                    plant, torque, states, held, durations, covariance_mode="propagated",
                    create_graph=False, integrator=integrator, **GRID,
                )
                self.assertTrue(torch.isfinite(moments["covariances"]).all())
                fractions = moments["saturation_fractions"].reshape(-1).tolist()
                self.assertEqual(fractions, [1.0, 0.0, 0.0])

    def test_refresh_flags_keep_the_first_substep_only(self):
        torque = tasks.make_env(windy(), decision_mode="torque", seed=0).controller
        feedback = tasks.make_env(windy(), decision_mode="feedback", seed=0).controller
        durations = torch.tensor([[0.002], [0.009], [0.018]], dtype=torch.float64)
        counts, sizes, flags = refresh_aligned_grid(
            durations, 0.001, physics_dt=0.001, controller_refresh_steps=1
        )
        self.assertIs(controller_refresh_flags(feedback, flags), flags)
        held = controller_refresh_flags(torque, flags)
        self.assertTrue(held[:, 0].all())
        self.assertFalse(held[:, 1:].any())
        self.assertTrue(flags[:, 1].all())  # the grid itself is the feedback arms'


# ------------------------------------------------------------------------------ agents

# Decisions of each short torque-arm run, warm fits and plant fits included.
RUN_BUDGET = 200


def tiny_protocol(output_directory, *, budget=RUN_BUDGET):
    """The cartpole-swingup row at the smallest settings of the algorithm tests."""
    raw = copy.deepcopy(find_ph_entry("cartpole-swingup", "top").config)
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    raw["training"].update(replay_capacity=5000, batch_size=8, warmup_episodes=0)
    raw["explicit"].update(rollout_paths=3)
    implicit = raw["implicit"]
    implicit.update(reward_warm_steps=5, plant_warm_steps=5)
    implicit["solver"].update(policy_draws=3, inner_steps=6)
    implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
    implicit["admission"].update(initial_rows=8, initial_paths=2)
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[0],
        evaluation_reset_seeds=[20000],
        evaluation_every_interactions=budget,
        output_directory=str(output_directory),
    )
    return protocol_from_config(raw, "tiny cartpole-swingup")


def finite_tree(test, value, where):
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point:
            test.assertTrue(torch.isfinite(value).all(), where)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(test, item, f"{where}.{key}")


class TestTorqueArmRuns(unittest.TestCase):
    """Each torque arm trains RUN_BUDGET decisions on cartpole-swingup through run_seed."""

    def run_arm(self, name, tmp):
        proto = tiny_protocol(tmp)
        proto["arms"] = [arm for arm in protocol.all_arms() if arm["name"] == name]
        protocol.validate_matched_quantities(proto)
        (run,) = protocol.build_run_matrix(proto)
        return run, protocol.run_seed(run)

    def check(self, name, *, algorithm):
        with tempfile.TemporaryDirectory() as tmp:
            run, result = self.run_arm(name, tmp)
            self.assertEqual(result["decision_mode"], "torque")
            self.assertEqual(result["run"]["arm"]["decision_mode"], "torque")
            self.assertEqual(result["counters"]["interactions"], RUN_BUDGET)
            self.assertEqual(
                [c["interactions"] for c in result["evaluations"]], [0, RUN_BUDGET]
            )
            for checkpoint in result["evaluations"]:
                for episode in checkpoint["episodes"]:
                    self.assertTrue(math.isfinite(episode["discounted_return"]))
                    self.assertTrue(math.isfinite(episode["episode_return"]))
            output = Path(tmp) / "cartpole_swingup" / name / "seed_0"
            text = (output / RECORD_NAME).read_text()
            self.assertEqual(json.loads(text)["result"]["decision_mode"], "torque")
            (loaded,) = load_run_records([str(output / RECORD_NAME)])
            self.assertEqual(loaded["decision_mode"], "torque")
            state = torch.load(output / "checkpoint.pt", weights_only=False)
            finite_tree(self, state["networks"], "networks")
            actor = state["networks"]["actor"]
            np.testing.assert_array_equal(actor["decision_low"].numpy(), [-10.0])
            np.testing.assert_array_equal(actor["decision_high"].numpy(), [10.0])
            config = state["config"]
            self.assertEqual(decision_mode_setting(config["testbed"]["controller"]), "torque")
            self.assertEqual(config["temperature"]["target_entropy"], -1.0)
            shared = run["protocol"][f"{algorithm}_config"]["temperature"]
            self.assertEqual(shared["target_entropy"], -3.0)
            self.assertTrue(math.isfinite(float(state["log_temperature"])))
            replay = state["replay"]["buffers"]["train"]["arrays"]
            self.assertEqual(replay["decisions"].shape[1], 1)
            self.assertTrue(np.all(np.abs(replay["decisions"]) <= 10.0))
            return state

    def test_implicit_torque_run(self):
        state = self.check("implicit_torque", algorithm="implicit")
        self.assertGreater(state["counters"]["value_gradient_steps"], 0)
        self.assertGreater(state["counters"]["rate_gradient_steps"], 0)
        self.assertGreater(state["counters"]["model_rollout_paths"], 0)
        self.assertEqual(state["networks"]["rate"]["scaling.low"].numpy().tolist(), [-10.0])

    def test_explicit_rollout_torque_run(self):
        state = self.check("explicit_rollout_torque", algorithm="explicit")
        self.assertGreater(state["counters"]["critic_gradient_steps"], 0)
        self.assertGreater(state["counters"]["model_rollout_paths"], 0)

    def test_explicit_generator_torque_run(self):
        state = self.check("explicit_generator_torque", algorithm="explicit")
        self.assertGreater(state["counters"]["critic_gradient_steps"], 0)
        self.assertEqual(state["counters"]["model_rollout_paths"], 0)

    def test_records_without_the_field_load_their_arms_mode(self):
        arms = {arm["name"]: arm for arm in protocol.all_arms()}
        cases = (
            ({}, "feedback"),
            ({"arm": {"name": "implicit_oracle"}}, "feedback"),
            ({"arm": dict(arms["implicit_oracle"])}, "feedback"),
            ({"arm": dict(arms["ct_sac_torque"])}, "torque"),
            ({"arm": dict(arms["implicit_torque"])}, "torque"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (run, expected) in enumerate(cases):
                with self.subTest(case=index):
                    path = Path(tmp) / f"record_{index}.json"
                    record = {"units": {}, "result": {"run": run, "evaluations": []}}
                    path.write_text(json.dumps(record))
                    (loaded,) = load_run_records([str(path)])
                    self.assertEqual(loaded["decision_mode"], expected)
                    self.assertEqual(record_decision_mode({"run": run}), expected)
            self.assertEqual(record_decision_mode({"decision_mode": "torque", "run": {}}), "torque")


# -------------------------------------------------------------------------------- arms


class TestTorqueArms(unittest.TestCase):
    def test_each_torque_arm_changes_only_the_decision_mode(self):
        arms = {arm["name"]: arm for arm in protocol.all_arms()}
        for name, reference in TORQUE_ARMS.items():
            with self.subTest(arm=name):
                changed = [
                    key
                    for key in arms[name]
                    if key != "name" and arms[name][key] != arms[reference][key]
                ]
                self.assertEqual(changed, ["decision_mode"])
                self.assertEqual(arms[name]["decision_mode"], "torque")
                self.assertEqual(arms[reference]["decision_mode"], "feedback")

    def test_default_arm_set_is_the_four_torque_arms(self):
        names = [arm["name"] for arm in protocol.paper_arms()]
        # Role order: baseline, implicit, rollout, generator.
        self.assertEqual(names, ["ct_sac_torque", *TORQUE_ARMS])
        self.assertEqual({arm["decision_mode"] for arm in protocol.paper_arms()}, {"torque"})
        self.assertEqual(
            [arm["name"] for arm in protocol.torque_decision_arms()], list(TORQUE_ARMS)
        )
        impedance = [arm["name"] for arm in protocol.impedance_decision_arms()]
        self.assertEqual(impedance, list(TORQUE_ARMS.values()))
        implicit = [arm["name"] for arm in protocol.implicit_ablation_arms()]
        self.assertEqual(
            implicit,
            ["implicit_oracle", "implicit_averaged_draw", "implicit_no_rollout_rate",
             "implicit_polyak_plant", "implicit_gated_admissions",
             "implicit_fixed_gate_allocation"],
        )
        self.assertEqual(
            {arm["decision_mode"] for arm in protocol.implicit_ablation_arms()}, {"feedback"}
        )
        # The value-step ablation arms (tests/test_ph_step_ablation.py) follow.
        step_ablation = [arm["name"] for arm in protocol.step_ablation_arms()]
        self.assertEqual(
            step_ablation,
            ["implicit_explicit_step", "implicit_explicit_step_interval", "implicit_critic",
             "implicit_critic_head"],
        )
        self.assertEqual(
            {arm["decision_mode"] for arm in protocol.step_ablation_arms()}, {"torque"}
        )
        everything = [arm["name"] for arm in protocol.all_arms()]
        self.assertEqual(everything, names + impedance + implicit + step_ablation)
        self.assertEqual(len(everything), 17)
        self.assertEqual(sorted(everything), sorted(get_args(protocol.ArmName)))
        # A protocol without --arms, from a JSON file or a table row, runs the paper arms.
        self.assertEqual([arm["name"] for arm in smoke_protocol()["arms"]], names)
        args = parse_args(["--env-id", "cartpole-swingup", "--seeds", "0"])
        (row,) = load_protocols(args)
        self.assertEqual([arm["name"] for arm in row["arms"]], names)

    def test_arms_option_selects_the_impedance_arms(self):
        path = str(CONFIGS / "smoke_windy.json")
        # Selected in all_arms() order: the baseline, then the impedance-law arms.
        selected = ["ct_sac_torque", *TORQUE_ARMS.values()]
        args = parse_args(["--config", path, "--arms", *reversed(selected), "--seeds", "2"])
        proto = apply_overrides(load_protocol(path), args)
        self.assertEqual([arm["name"] for arm in proto["arms"]], selected)
        self.assertEqual(
            [arm["decision_mode"] for arm in proto["arms"]], ["torque"] + ["feedback"] * 3
        )
        protocol.validate_matched_quantities(proto)
        self.assertEqual(
            sorted(run["run_id"] for run in protocol.build_run_matrix(proto)),
            sorted(f"windy_pendulum__{name}__seed2" for name in selected),
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--config", path, "--arms", "implicit_impedance_free"])

    def test_a_changed_torque_arm_is_rejected(self):
        proto = smoke_protocol()
        arm = dict({a["name"]: a for a in protocol.all_arms()}["implicit_torque"])
        arm["decision_mode"] = "feedback"
        proto["arms"] = [arm]
        with self.assertRaises(ValueError):
            protocol.validate_matched_quantities(proto)

    def test_torque_arm_configs_change_only_the_decisions(self):
        proto = smoke_protocol()
        proto["arms"] = list(protocol.all_arms())
        runs = runs_by_arm(proto)
        for name, reference in TORQUE_ARMS.items():
            with self.subTest(arm=name):
                torque, feedback = arm_config(runs[name]), arm_config(runs[reference])
                validate_configuration(torque)
                self.assertEqual(set(torque), set(feedback))
                for key in torque:
                    if key in ("testbed", "temperature"):
                        continue
                    self.assertTrue(protocol._same_settings(torque[key], feedback[key]), key)
                self.assertTrue(protocol._same_settings(
                    {k: v for k, v in torque["testbed"].items() if k != "controller"},
                    {k: v for k, v in feedback["testbed"].items() if k != "controller"},
                ))
                self.assertTrue(protocol._same_settings(
                    torque["testbed"]["controller"],
                    torque_decision_controller(feedback["testbed"]["controller"]),
                ))
                self.assertEqual(torque["temperature"]["target_entropy"], -1.0)
                self.assertEqual(feedback["temperature"]["target_entropy"], -3.0)
                self.assertEqual(
                    {k: v for k, v in torque["temperature"].items() if k != "target_entropy"},
                    {k: v for k, v in feedback["temperature"].items() if k != "target_entropy"},
                )
        # The protocol itself is untouched.
        self.assertNotIn("decision_mode", proto["implicit_config"]["testbed"]["controller"])
        self.assertEqual(proto["explicit_config"]["temperature"]["target_entropy"], -3.0)

    def test_list_shows_the_paper_and_impedance_arms_and_an_example(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--list"])
        lines = {line.split()[0]: line for line in output.getvalue().splitlines()}
        for name in ("ct_sac_torque", *TORQUE_ARMS):
            self.assertIn("torque", lines[name].split())
            self.assertEqual(lines[name].split()[-1], "paper")
            self.assertNotIn("ablation", lines[name])
        for name in TORQUE_ARMS.values():
            self.assertIn("feedback", lines[name].split())
            self.assertIn("impedance ablation (only with --arms)", lines[name])
        self.assertIn("implicit ablation", lines["implicit_oracle"])
        self.assertIn("step ablation", lines["implicit_critic"])
        self.assertIn("torque", lines["implicit_critic"].split())
        self.assertEqual(lines["example:"], f"example: {ABLATION_ARM_EXAMPLE}")
        self.assertEqual(list_arms()[-1], f"example: {ABLATION_ARM_EXAMPLE}")
        self.assertEqual(
            ABLATION_ARM_EXAMPLE,
            "python benchmarks/run_ph_sac.py --env-id cartpole-swingup --mode top "
            "--arms implicit_learned_admissions --seeds 0",
        )
        args = parse_args(ABLATION_ARM_EXAMPLE.split()[2:])
        self.assertEqual(args.arms, ["implicit_learned_admissions"])


# ------------------------------------------------------------------- fixed arm names

# The seven arm names of the runs made before torque decisions became the default arm set
# (2026-11-13), as evaluations/ph_sac/protocol.py declared them then (literal copies, not
# computed): every record stores its arm's spec (run.arm) and the resume fingerprint
# hashes it (resume.run_settings "arm"), so each name must resolve to exactly this.
FIXED_ARMS = {
    "ct_sac_torque": {
        "name": "ct_sac_torque", "algorithm": "ct_sac", "decision_mode": "torque",
        "target_mode": None, "plant_mode": "none", "residual_loss": None,
        "rollout_rate_enabled": False, "gate_allocation": None,
    },
    "implicit_learned_admissions": {
        "name": "implicit_learned_admissions", "algorithm": "implicit",
        "decision_mode": "feedback", "target_mode": None, "plant_mode": "admitted",
        "residual_loss": "all_pairs", "rollout_rate_enabled": True, "gate_allocation": "pilot",
    },
    "explicit_rollout": {
        "name": "explicit_rollout", "algorithm": "explicit", "decision_mode": "feedback",
        "target_mode": "rollout", "plant_mode": "polyak", "residual_loss": None,
        "rollout_rate_enabled": False, "gate_allocation": None,
    },
    "explicit_generator": {
        "name": "explicit_generator", "algorithm": "explicit", "decision_mode": "feedback",
        "target_mode": "generator", "plant_mode": "polyak", "residual_loss": None,
        "rollout_rate_enabled": False, "gate_allocation": None,
    },
    "implicit_torque": {
        "name": "implicit_torque", "algorithm": "implicit", "decision_mode": "torque",
        "target_mode": None, "plant_mode": "admitted", "residual_loss": "all_pairs",
        "rollout_rate_enabled": True, "gate_allocation": "pilot",
    },
    "explicit_rollout_torque": {
        "name": "explicit_rollout_torque", "algorithm": "explicit", "decision_mode": "torque",
        "target_mode": "rollout", "plant_mode": "polyak", "residual_loss": None,
        "rollout_rate_enabled": False, "gate_allocation": None,
    },
    "explicit_generator_torque": {
        "name": "explicit_generator_torque", "algorithm": "explicit",
        "decision_mode": "torque", "target_mode": "generator", "plant_mode": "polyak",
        "residual_loss": None, "rollout_rate_enabled": False, "gate_allocation": None,
    },
}
# Each spec's exact text inside the fingerprint's hashed JSON (resume.run_fingerprint:
# json.dumps of resume._canonical, sorted keys, compact separators), written out by hand.
FIXED_ARM_JSON = {
    "ct_sac_torque": (
        '{"algorithm":"ct_sac","decision_mode":"torque","gate_allocation":null,'
        '"name":"ct_sac_torque","plant_mode":"none","residual_loss":null,'
        '"rollout_rate_enabled":false,"target_mode":null}'
    ),
    "implicit_learned_admissions": (
        '{"algorithm":"implicit","decision_mode":"feedback","gate_allocation":"pilot",'
        '"name":"implicit_learned_admissions","plant_mode":"admitted",'
        '"residual_loss":"all_pairs","rollout_rate_enabled":true,"target_mode":null}'
    ),
    "explicit_rollout": (
        '{"algorithm":"explicit","decision_mode":"feedback","gate_allocation":null,'
        '"name":"explicit_rollout","plant_mode":"polyak","residual_loss":null,'
        '"rollout_rate_enabled":false,"target_mode":"rollout"}'
    ),
    "explicit_generator": (
        '{"algorithm":"explicit","decision_mode":"feedback","gate_allocation":null,'
        '"name":"explicit_generator","plant_mode":"polyak","residual_loss":null,'
        '"rollout_rate_enabled":false,"target_mode":"generator"}'
    ),
    "implicit_torque": (
        '{"algorithm":"implicit","decision_mode":"torque","gate_allocation":"pilot",'
        '"name":"implicit_torque","plant_mode":"admitted","residual_loss":"all_pairs",'
        '"rollout_rate_enabled":true,"target_mode":null}'
    ),
    "explicit_rollout_torque": (
        '{"algorithm":"explicit","decision_mode":"torque","gate_allocation":null,'
        '"name":"explicit_rollout_torque","plant_mode":"polyak","residual_loss":null,'
        '"rollout_rate_enabled":false,"target_mode":"rollout"}'
    ),
    "explicit_generator_torque": (
        '{"algorithm":"explicit","decision_mode":"torque","gate_allocation":null,'
        '"name":"explicit_generator_torque","plant_mode":"polyak","residual_loss":null,'
        '"rollout_rate_enabled":false,"target_mode":"generator"}'
    ),
}
# The default arm set before 2026-11-13 (protocol.paper_arms then), by name.
EARLIER_DEFAULT_ARMS = (
    "ct_sac_torque", "explicit_generator", "explicit_rollout", "implicit_oracle",
    "implicit_learned_admissions", "implicit_averaged_draw", "implicit_no_rollout_rate",
    "implicit_polyak_plant", "implicit_gated_admissions", "implicit_fixed_gate_allocation",
)


def compact_json(value):
    """The fingerprint's serialization (resume.run_fingerprint)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class TestFixedArmNames(unittest.TestCase):
    """The seven names keep their spec, config, records and resume fingerprint."""

    def test_specs_are_the_earlier_declarations(self):
        catalog = {arm["name"]: arm for arm in protocol.all_arms()}
        for name, expected in FIXED_ARMS.items():
            with self.subTest(arm=name):
                spec = catalog[name]
                self.assertEqual(sorted(spec), sorted(expected))
                for key, value in expected.items():
                    self.assertIs(type(spec[key]), type(value), key)
                    self.assertEqual(spec[key], value, key)
                self.assertEqual(dict(spec), expected)
                self.assertFalse(set(spec) & set(STEP_ABLATION_CHOICES))
                self.assertEqual(compact_json(resume._canonical(spec)), FIXED_ARM_JSON[name])
        # A protocol listing the earlier specs (as a stored protocol.json does) validates.
        proto = smoke_protocol()
        proto["arms"] = [copy.deepcopy(spec) for spec in FIXED_ARMS.values()]
        protocol.validate_matched_quantities(proto)

    def test_fingerprints_and_configs_are_the_earlier_ones(self):
        table_row = load_protocols(parse_args(["--env-id", "cartpole-swingup", "--seeds", "0"]))
        catalog = {arm["name"]: arm for arm in protocol.all_arms()}
        for label, proto in (("smoke_windy", smoke_protocol()), ("cartpole-swingup", table_row[0])):
            proto["arms"] = list(protocol.all_arms())
            protocol.validate_matched_quantities(proto)
            current = runs_by_arm(proto)
            for name, expected in FIXED_ARMS.items():
                with self.subTest(protocol=label, arm=name):
                    run = current[name]
                    earlier = dict(run, arm=copy.deepcopy(expected))
                    settings = resume.run_settings(run)
                    self.assertEqual(settings, resume.run_settings(earlier))
                    self.assertEqual(resume.run_fingerprint(run), resume.run_fingerprint(earlier))
                    self.assertEqual(compact_json(settings["arm"]), FIXED_ARM_JSON[name])
                    self.assertNotIn("step_ablation", settings["resolved"]["implicit_config"])
                    # The arm list is not fingerprinted: the earlier default set, the new
                    # one or this arm alone give one fingerprint.
                    for arms in (
                        [catalog[other] for other in EARLIER_DEFAULT_ARMS],
                        list(protocol.paper_arms()),
                        [expected],
                    ):
                        listed = dict(run, protocol=dict(run["protocol"], arms=arms))
                        self.assertEqual(
                            resume.run_fingerprint(listed), resume.run_fingerprint(run)
                        )
                    self.assertEqual(
                        run["run_id"], f"{proto['testbed']}__{name}__seed{run['training_seed']}"
                    )
                    self.assertEqual(
                        protocol.run_output_directory(run),
                        Path(proto["output_directory"]) / proto["testbed"] / name
                        / f"seed_{run['training_seed']}",
                    )
                    if expected["algorithm"] == "ct_sac":
                        continue
                    # The agent's config: the protocol's own box and entropy target for a
                    # feedback arm, torque_decision_config's for a torque arm.
                    config = arm_config(run)
                    shared = proto[f"{expected['algorithm']}_config"]
                    reference = (
                        shared if expected["decision_mode"] == "feedback"
                        else torque_decision_config(shared)
                    )
                    self.assertTrue(
                        protocol._same_settings(config["testbed"], reference["testbed"])
                    )
                    self.assertEqual(config["temperature"], reference["temperature"])
                    self.assertEqual(
                        decision_mode_setting(config["testbed"]["controller"]),
                        expected["decision_mode"],
                    )


# --------------------------------------------------------------- feedback arms unaffected

# Outputs of the code before the torque-decision mode existed (2026-11-09), from
# feedback_training_summary below on this machine: parameter sum and sum of squares of
# each network (parameters only, no buffers) and the integer counters, after
# learn(TRAINING_INTERACTIONS) with plant diagnostics every 40 interactions.
TRAINING_INTERACTIONS = 120
FEEDBACK_TRAINING = {
    "explicit_generator": (
        {
            # Regenerated 2026-11-12 on Vista after the generator continuation took each
            # row's own interval (this run uses an irregular clock); counters and seconds
            # are unchanged from before.
            "actor": (-2.4011352382367477, 17.955442738000762),
            "critics": (-3.134634875401389, 26.289285779046544),
            "value": (-1.3218723756726831, 13.789941418074429),
            "online_plant": (3.5795727858203463, 54.07381877760372),
        },
        {
            "interactions": 120,
            "critic_gradient_steps": 113,
            "value_gradient_steps": 113,
            "rate_gradient_steps": 0,
            "actor_gradient_steps": 113,
            "temperature_gradient_steps": 0,
            "plant_gradient_steps": 113,
            "model_rollout_paths": 0,
            "gate_simulated_paths": 0,
            "gate_real_rows_acquired": 0,
        },
        1.3190000000000006,
    ),
    "explicit_rollout": (
        {
            "actor": (-2.401126216049306, 17.955448518201187),
            "critics": (-3.1346271136717405, 26.289276017179482),
            "value": (-1.321872598840855, 13.789941358466281),
            "online_plant": (3.5795725212665275, 54.07381862271805),
        },
        {
            "interactions": 120,
            "critic_gradient_steps": 113,
            "value_gradient_steps": 113,
            "rate_gradient_steps": 0,
            "actor_gradient_steps": 113,
            "temperature_gradient_steps": 0,
            "plant_gradient_steps": 113,
            "model_rollout_paths": 2712,
            "gate_simulated_paths": 0,
            "gate_real_rows_acquired": 0,
        },
        1.3190000000000006,
    ),
    "implicit_gated": (
        {
            "actor": (-2.969527762848884, 17.369326786754456),
            "rate": (-5.296467133332044, 12.883742587520823),
            "value": (3.7471771952696145, 12.798439113688874),
            "online_plant": (-4.058631602340029, 51.10579479133564),
            "reward": (2.079895227099769, 12.55557728894572),
        },
        {
            "interactions": 120,
            "critic_gradient_steps": 0,
            "value_gradient_steps": 30,
            "rate_gradient_steps": 10,
            "actor_gradient_steps": 10,
            "temperature_gradient_steps": 0,
            "plant_gradient_steps": 30,
            "model_rollout_paths": 160,
            "gate_simulated_paths": 3904,
            "gate_real_rows_acquired": 82,
            "reward_gradient_steps": 30,
        },
        1.3190000000000006,
    ),
}
# Outputs of models/ph_sac/rollouts.py before the torque-decision mode existed on the
# coupled fixture of tests/test_ph_model_integrator.py with integrator "rk4":
# integrate_endpoint_moments (propagated; the leading-order means are the same, its
# covariance traces differ), path_jacobian_norms and stochastic_rollout with
# coupled_increments, coupled_return_terms and coupled_failure.
RK4_PROPAGATED_MEANS = (
    0.4029793149440111, -0.30588180411696453, 0.986203641945934, -1.921266848573169,
    2.9598052928294494, 0.2026606522973202, 5.961843559550494, 0.23195984702877448,
    -1.199078675288267, 0.8021894863671581, 0.16321056485626684, 0.22753444651742058,
)
RK4_PROPAGATED_COVARIANCE_TRACES = (
    0.0005759774766798034, 0.0011024483349660022, 0.0007641170223837246,
)
RK4_LEADING_ORDER_COVARIANCE_TRACES = (
    0.0005777728846594133, 0.0011210831169993195, 0.0008265687023616037,
)
RK4_SATURATION = (
    1.0, 0.0, 0.75,
)
RK4_JACOBIAN_NORMS = (
    0.03624585548399286, 0.35294066912375044, 1.0048641150862563,
)
RK4_ENDPOINTS = (
    0.40299495160244986, -0.30571611428127277, 0.9867478411131512, -1.9165851310974253,
    0.4036125755999623, -0.3054696458428919, 1.005449129058542, -1.906941203430775,
    2.9534533592238845, 0.20199595623444944, 5.954278319671064, 0.23239897680607927,
    2.9537444980163787, 0.20264481154538086, 5.962925552850307, 0.2408039960840249,
    -1.1980667819615016, 0.8040312518831926, 0.18634663658724976, 0.26834157478534343,
    -1.1978869721268208, 0.8015842700251878, 0.18924870199646449, 0.22070762732082166,
)
RK4_RUNNING_REWARDS = (
    -0.001237864168248414, -0.001238504915302149, -0.07776192182491717,
    -0.07776016423782427, -0.015667633135741724, -0.01559946236855848,
)
RK4_FAILURE_PAYOFFS = (
    0.0, 0.0, -0.9955101098295706, -0.9955101098295706, 0.0, 0.0,
)
# float32 training: bit-identical on one machine; another BLAS may round the float32
# steps differently. The explicit generator and rollout arms' actor sums differ by
# 9e-6 relative, so this tolerance still separates any change of the training path.
TRAINING_TOLERANCE = {"rtol": 1e-7, "atol": 1e-9}

def held_windy():
    """windy() with the sample-and-hold controller (refresh_mode "held") the values above
    were computed with; the model's RK4 then holds the torque through its stages."""
    testbed = windy()
    testbed["controller"] = dict(testbed["controller"], refresh_mode="held")
    return testbed


FEEDBACK_AGENTS = {
    "explicit_generator": lambda: make_explicit(
        explicit_config(held_windy(), continuation="generator")
    ),
    "explicit_rollout": lambda: make_explicit(
        explicit_config(held_windy(), continuation="rollout")
    ),
    "implicit_gated": lambda: make_implicit(implicit_config(held_windy())),
}


def feedback_training_summary(label):
    """Train one feedback agent of FEEDBACK_AGENTS; summarize its networks and counters."""
    agent = FEEDBACK_AGENTS[label]()
    agent.plant_diagnostics_every = 40
    agent.learn(TRAINING_INTERACTIONS)
    summary = {}
    for name, module in agent._network_modules().items():
        copies = name.startswith("target") or name in ("admitted_plant", "reward_frozen")
        if module is None or copies:
            continue
        values = torch.cat([p.detach().double().reshape(-1) for p in module.parameters()])
        summary[name] = (float(values.sum()), float(values.square().sum()))
    counters = {
        key: value for key, value in agent.counters().items()
        if key not in ("wall_seconds", "simulated_seconds")
    }
    return summary, counters, float(agent.counters()["simulated_seconds"])


def rk4_feedback_outputs():
    plant, controller = coupled_plant(), coupled_controller()
    states, decisions, durations = coupled_batch()
    propagated = integrate_endpoint_moments(
        plant, controller, states, decisions, durations,
        covariance_mode="propagated", create_graph=False, integrator="rk4", **GRID,
    )
    leading = integrate_endpoint_moments(
        plant, controller, states, decisions, durations,
        covariance_mode="leading_order", create_graph=False, integrator="rk4", **GRID,
    )
    norms = path_jacobian_norms(
        plant, controller, states, decisions, durations, integrator="rk4", **GRID
    )
    rollout = stochastic_rollout(
        plant, controller, states, decisions, durations, coupled_failure,
        brownian_increments=coupled_increments(), return_terms=coupled_return_terms(),
        integrator="rk4", **GRID,
    )
    trace = lambda m: torch.diagonal(m["covariances"], dim1=-2, dim2=-1).sum(-1)  # noqa: E731
    return {
        "means": propagated["means"],
        "leading_means": leading["means"],
        "propagated_traces": trace(propagated),
        "leading_traces": trace(leading),
        "saturation": propagated["saturation_fractions"],
        "norms": norms,
        "endpoints": rollout["endpoints"],
        "running": rollout["running_rewards"],
        "payoffs": rollout["failure_payoffs"],
    }


class TestFeedbackArmsUnaffected(unittest.TestCase):
    def test_feedback_arm_configs_keep_the_protocol_testbed(self):
        proto = smoke_protocol()
        # The impedance-law and implicit-setting ablations, nine arms.
        proto["arms"] = [arm for arm in protocol.all_arms() if arm["decision_mode"] == "feedback"]
        self.assertEqual(len(proto["arms"]), 9)
        testbed = proto["matched"]["testbed_config"]
        for name, run in runs_by_arm(proto).items():
            with self.subTest(arm=name):
                config = arm_config(run)
                self.assertEqual(run["arm"]["decision_mode"], "feedback")
                self.assertTrue(protocol._same_settings(config["testbed"], testbed))
                self.assertNotIn("decision_mode", config["testbed"]["controller"])
                self.assertEqual(np.size(config["testbed"]["controller"]["decision_low"]), 3)
                self.assertEqual(config["temperature"]["target_entropy"], -3.0)
        env = tasks.make_env(testbed, decision_mode="feedback", seed=0)
        self.assertIs(env.config, testbed)
        self.assertIsInstance(env.controller, FeedbackController)
        np.testing.assert_array_equal(env.action_space.low, [-0.5, 0.0, 0.0])
        np.testing.assert_array_equal(env.action_space.high, [0.5, 100.0, 20.0])

    def test_feedback_training_reproduces_the_earlier_code(self):
        for label, (networks, counters, seconds) in FEEDBACK_TRAINING.items():
            with self.subTest(agent=label):
                summary, observed_counters, observed_seconds = feedback_training_summary(label)
                self.assertEqual(observed_counters, counters)
                self.assertAlmostEqual(observed_seconds, seconds, places=12)
                self.assertEqual(set(summary), set(networks))
                for name, expected in networks.items():
                    np.testing.assert_allclose(
                        summary[name], expected, **TRAINING_TOLERANCE, err_msg=name
                    )

    def test_feedback_rk4_integration_reproduces_the_earlier_code(self):
        outputs = rk4_feedback_outputs()
        for key, expected in (
            ("means", RK4_PROPAGATED_MEANS),
            ("leading_means", RK4_PROPAGATED_MEANS),
            ("propagated_traces", RK4_PROPAGATED_COVARIANCE_TRACES),
            ("leading_traces", RK4_LEADING_ORDER_COVARIANCE_TRACES),
            ("saturation", RK4_SATURATION),
            ("norms", RK4_JACOBIAN_NORMS),
            ("endpoints", RK4_ENDPOINTS),
            ("running", RK4_RUNNING_REWARDS),
            ("payoffs", RK4_FAILURE_PAYOFFS),
        ):
            np.testing.assert_allclose(
                outputs[key].reshape(-1).numpy(), np.asarray(expected), **REGRESSION_TOLERANCE,
                err_msg=key,
            )


if __name__ == "__main__":
    unittest.main()
