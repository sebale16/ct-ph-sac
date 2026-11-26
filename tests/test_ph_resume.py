"""Exact restarts (evaluations/ph_sac/resume.py) and the plant-step schedule.

A run stopped at a safe point and continued in a fresh agent (built anew, its random
streams deliberately disturbed before the load) must end bitwise where the uninterrupted
run ends: every network, optimizer, replay row, counter, record, random stream and
evaluation return. Tiny configs of table rows with 1 s episodes (50 decisions of 20 ms):
pendulum-swingup (no joint limit, no context) and reacher-easy (the wrist's soft joint
limit and the target as context).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import random
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import NETWORK_COLUMNLESS_KEYS, plant_step_schedule, validate_configuration
from environment.ph_sac.tasks import make_env
import algorithms.ph_base as P_BASE
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.reporting import (
    load_run_records,
    record_plant_step_schedule,
    record_reward_observation,
)

# A short episode keeps warm-up (one episode) and evaluation (one episode) cheap.
EPISODE_SECONDS = 1.0
WARMUP_DECISIONS = 50
INNER_STEPS = 8


def tiny_run(env_id, arm_name, output, budget, *, schedule=None, learning_rate=None, mode="top",
             warmup_decisions=None, network=None, protocol=None):
    """A tiny protocol of a table row (mode: top, or irregular for its irregular clock),
    one arm and seed; schedule (D, k) or dense; the training key warmup_decisions when
    given (absent when None); network: extra network keys when given; protocol: extra
    protocol keys (a JSON config's "protocol" section) when given."""
    raw = copy.deepcopy(table.find_ph_entry(env_id, mode).config)
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    if network is not None:
        raw["network"].update(network)
    training = raw["training"]
    training.update(batch_size=16, replay_capacity=2000, warmup_episodes=1)
    for key in ("plant_step_dense_until", "plant_step_every"):
        training.pop(key, None)
    if schedule is not None:
        training.update(plant_step_dense_until=schedule[0], plant_step_every=schedule[1])
    if warmup_decisions is not None:
        training["warmup_decisions"] = warmup_decisions
    if learning_rate is not None:
        training["learning_rate"] = learning_rate
    raw["explicit"].update(rollout_paths=2)
    implicit = raw["implicit"]
    implicit.update(reward_warm_steps=5, plant_warm_steps=5, temperature_every_rows=4)
    implicit["solver"].update(inner_steps=INNER_STEPS, policy_draws=2)
    implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
    # efolds 0.5: an admission every M = ceil(0.5 / log 1.5) = 2 outer iterations.
    implicit["admission"].update(
        initial_rows=8, initial_paths=2, pilot_batches=2, real_row_budget=64,
        simulated_path_budget=4096, efolds=0.5,
    )
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[0],
        evaluation_reset_seeds=[3],
        evaluation_every_interactions=40,
        output_directory=str(output),
    )
    if protocol is not None:
        raw["protocol"].update(protocol)
    proto = run_ph_sac.protocol_from_config(raw, f"tiny {env_id}")
    for testbed in (
        proto["matched"]["testbed_config"],
        proto["explicit_config"]["testbed"],
        proto["implicit_config"]["testbed"],
    ):
        testbed["episode_duration"] = EPISODE_SECONDS
    proto["arms"] = [arm for arm in P.all_arms() if arm["name"] == arm_name]
    P.validate_matched_quantities(proto)
    (run,) = P.build_run_matrix(proto)
    return run


def disturb(agent, env):
    """Move every random stream away from where construction left it."""
    torch.manual_seed(4242)
    np.random.seed(4242)
    random.seed(4242)
    env.reset(seed=4242)
    if hasattr(agent, "replay"):
        agent._rng.random(7)
        agent._rollout_rng.random(3)
        agent._generator.manual_seed(4242)
        agent._predict_generator.manual_seed(4243)
        agent.replay._rng.random(5)
        for buffer in agent.replay.buffers.values():
            buffer._rng.random(5)
    else:
        agent.algorithm.env.action_space.seed(4242)


def run_captured(run, resume=None, *, disturbed=False, evaluate=None):
    """run_seed, also returning the agent it built (None if it built none).

    A ResumableStop is returned in place of the result.
    """
    agents = []
    build = P.make_training_agent

    def capture(spec, env):
        agent = build(spec, env)
        if disturbed:
            disturb(agent, env)
        agents.append(agent)
        return agent

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(P, "make_training_agent", capture))
        if evaluate is not None:
            stack.enter_context(mock.patch.object(P, "evaluate_fixed_seeds", evaluate))
        try:
            result = P.run_seed(run, resume)
        except R.ResumableStop as stop:
            result = stop
    return result, (agents[-1] if agents else None)


def without_wall(value):
    if isinstance(value, dict):
        return {k: without_wall(v) for k, v in value.items() if k != "wall_seconds"}
    if isinstance(value, (list, tuple)):
        return [without_wall(v) for v in value]
    return value


def differences(left, right, path="state", found=None):
    """Paths where two nested states differ (tensors and arrays exactly, dtype included)."""
    found = [] if found is None else found
    if isinstance(left, torch.Tensor):
        if not (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        ):
            found.append(path)
    elif isinstance(left, np.ndarray):
        if not (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        ):
            found.append(path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            found.append(f"{path} (keys)")
        else:
            for key in left:
                if key != "wall_seconds":
                    differences(left[key], right[key], f"{path}.{key}", found)
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            found.append(f"{path} (length)")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                differences(a, b, f"{path}[{index}]", found)
    elif isinstance(left, float) and isinstance(right, float) and np.isnan(left):
        if not np.isnan(right):
            found.append(path)
    elif left != right:
        found.append(path)
    return found


def network_hashes(agent):
    """SHA-256 of every network's parameters and buffers, by name."""
    if hasattr(agent, "_network_modules"):
        modules = {k: m for k, m in agent._network_modules().items() if m is not None}
    else:
        model = agent.algorithm.model
        modules = {
            "actor": model.actor,
            "critics": model.q_nets,
            "critic_targets": model.q_target_nets,
        }
    hashes = {}
    for name, module in sorted(modules.items()):
        digest = hashlib.sha256()
        for key, tensor in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def returns(result):
    return [
        episode["discounted_return"]
        for checkpoint in result["evaluations"]
        for episode in checkpoint["episodes"]
    ]


class ExactRestartMixin:
    def assert_same_run(self, reference, ref_agent, resumed, res_agent):
        self.assertEqual(network_hashes(ref_agent), network_hashes(res_agent))
        self.assertEqual(differences(ref_agent.resume_state(), res_agent.resume_state()), [])
        self.assertEqual(returns(reference), returns(resumed))
        for key in ("evaluations", "counters", "solver", "admissions", "oracle",
                    "plant_calibration"):
            self.assertEqual(
                differences(without_wall(reference[key]), without_wall(resumed[key]), key), []
            )
        self.assertEqual(reference.get("reward_fit"), resumed.get("reward_fit"))

    def check_exact_restart(self, env_id, arm, budget, stop, *, schedule=None, mode="top",
                            warmup_decisions=None, protocol=None):
        """Uninterrupted vs stopped at `stop` and resumed in a disturbed fresh agent."""
        options = R.resume_options(every_interactions=0)
        settings = {
            "schedule": schedule,
            "mode": mode,
            "warmup_decisions": warmup_decisions,
            "protocol": protocol,
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reference, ref_agent = run_captured(
                tiny_run(env_id, arm, first, budget, **settings), options
            )
            run = tiny_run(env_id, arm, second, budget, **settings)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
            self.assertIsInstance(stopped, R.ResumableStop)
            folder = P.run_output_directory(run)
            payload = torch.load(folder / R.RESUME_NAME, weights_only=False)
            at = payload["interactions"]
            self.assertGreaterEqual(at, stop)
            # The records on disk describe the stopped state, no later row.
            (on_disk,) = load_run_records([str(folder / "run_record.json")])
            self.assertEqual(on_disk["counters"]["interactions"], at)
            resumed, res_agent = run_captured(run, options, disturbed=True)
            self.assertFalse((folder / R.RESUME_NAME).exists())
            self.assertTrue((folder / R.COMPLETE_NAME).exists())
        self.assertEqual(reference["counters"]["interactions"], budget)
        self.assert_same_run(reference, ref_agent, resumed, res_agent)
        return reference, ref_agent, at, payload


class TestExactRestartPendulum(ExactRestartMixin, unittest.TestCase):
    """pendulum-swingup: limit-free, no context."""

    env_id = "pendulum-swingup"

    def test_implicit_learned_admissions(self):
        # The warm fit (decision 50), outer iterations of 8 steps with admissions every
        # second one, and the evaluations at 40 and 80 come before the stop at 110.
        reference, agent, at, payload = self.check_exact_restart(
            self.env_id, "implicit_learned_admissions", 160, 110
        )
        loop = payload["agent"]["loop"]
        self.assertTrue(loop["warm_fit_done"])
        self.assertGreater(loop["inner_steps"], 0)
        self.assertGreaterEqual(loop["admission_index"], 2)
        self.assertEqual(
            [c["interactions"] for c in payload["runner"]["evaluations"]], [0, 40, 80]
        )
        self.assertGreater(len(reference["admissions"]), 2)

    def test_implicit_stop_at_the_last_inner_step(self):
        # The K-th inner step's callback precedes the end of the outer iteration: the
        # checkpoint waits for the next safe point, the top of the next solve.
        _, _, at, payload = self.check_exact_restart(
            self.env_id, "implicit_learned_admissions", 100, WARMUP_DECISIONS + INNER_STEPS
        )
        self.assertEqual(at, WARMUP_DECISIONS + INNER_STEPS)
        self.assertEqual(payload["agent"]["loop"]["inner_steps"], 0)
        self.assertEqual(payload["agent"]["loop"]["outer_iteration"], 1)

    def test_stop_inside_warm_up(self):
        self.check_exact_restart(self.env_id, "implicit_learned_admissions", 80, 20)

    def test_explicit_rollout(self):
        self.check_exact_restart(self.env_id, "explicit_rollout", 120, 85)

    def test_explicit_generator(self):
        self.check_exact_restart(self.env_id, "explicit_generator", 120, 85)

    def test_ct_sac_baseline(self):
        self.check_exact_restart(self.env_id, "ct_sac_torque", 120, 85)

    def test_every_other_arm(self):
        # The paper's three torque-decision CT-PH-SAC arms (the default with ct_sac_torque),
        # the known-plant arm and the other implicit ablations; the value-step ablation
        # arms restart in tests/test_ph_step_ablation.py.
        covered = {
            "implicit_learned_admissions", "explicit_rollout", "explicit_generator",
            "ct_sac_torque", "implicit_gated_admissions",
        } | {arm["name"] for arm in P.step_ablation_arms()}
        self.assertEqual(len(P.step_ablation_arms()), 4)
        others = [arm["name"] for arm in P.all_arms() if arm["name"] not in covered]
        self.assertEqual(len(others), 8)
        self.assertEqual(
            others[:3], ["implicit_torque", "explicit_rollout_torque", "explicit_generator_torque"]
        )
        for arm in others:
            with self.subTest(arm=arm):
                self.check_exact_restart(self.env_id, arm, 100, 70)

    def test_gated_admissions_stopped_inside_an_attempt(self):
        # The first attempt (after the second outer iteration, decision 66) acquires
        # held-out rows; a stop requested then waits for the attempt to end.
        reference, _, at, _ = self.check_exact_restart(
            self.env_id, "implicit_gated_admissions", 130, 68
        )
        self.assertGreater(reference["counters"]["gate_real_rows_acquired"], 0)
        self.assertGreater(at, 68)

    def test_plant_schedule_position_survives_a_restart(self):
        # Dense through decision 60, then every 3rd row; stopped mid-cycle.
        schedule = (60, 3)
        for arm in ("implicit_learned_admissions", "explicit_rollout"):
            with self.subTest(arm=arm):
                reference, agent, at, payload = self.check_exact_restart(
                    self.env_id, arm, 130, 71, schedule=schedule
                )
                rows_after = payload["agent"]["plant_schedule"]["rows_after_dense"]
                self.assertEqual(rows_after, at - schedule[0])
                self.assertNotEqual(rows_after % schedule[1], 0)
                warm = 5 if arm.startswith("implicit") else 0
                expected = (
                    warm + (schedule[0] - WARMUP_DECISIONS) + (130 - schedule[0]) // schedule[1]
                )
                self.assertEqual(reference["counters"]["plant_gradient_steps"], expected)
                self.assertEqual(
                    reference["plant_step_schedule"], {"dense_until": 60, "every": 3}
                )


class TestExactRestartReacher(ExactRestartMixin, unittest.TestCase):
    """reacher-easy: the wrist's soft joint limit and the target as episode context. The
    four paper arms (torque decisions) and the three impedance-law arms."""

    env_id = "reacher-easy"

    def test_implicit_torque(self):
        self.check_exact_restart(self.env_id, "implicit_torque", 110, 85)

    def test_explicit_rollout_torque(self):
        self.check_exact_restart(self.env_id, "explicit_rollout_torque", 100, 75)

    def test_explicit_generator_torque(self):
        self.check_exact_restart(self.env_id, "explicit_generator_torque", 100, 75)

    def test_implicit_learned_admissions(self):
        self.check_exact_restart(self.env_id, "implicit_learned_admissions", 110, 85)

    def test_explicit_rollout(self):
        self.check_exact_restart(self.env_id, "explicit_rollout", 100, 75)

    def test_explicit_generator(self):
        self.check_exact_restart(self.env_id, "explicit_generator", 100, 75)

    def test_ct_sac_baseline(self):
        self.check_exact_restart(self.env_id, "ct_sac_torque", 100, 75)


class TestIrregularClockRows(ExactRestartMixin, unittest.TestCase):
    """The rows of mode irregular (the final run's clock: CT-SAC's law, 0.891 at T/5, 0.099 at
    3T, 0.01 between; since 2026-11-12): pendulum-swingup (model step 10 ms over intervals of
    4 to 60 ms) and cartpole-swingup (contact exclusion, coarse step 5 ms over 2 to 30 ms),
    every arm of the final run, stopped and resumed. The mean interval is 0.4912 T, so a 1 s
    warm-up episode has about 100 (pendulum) and 200 (cart-pole) decisions."""

    def irregular_durations(self, agent):
        """The stored training intervals in ms, which must hold both bounds."""
        buffer = agent.replay.buffers["train"] if hasattr(agent, "replay") else None
        if buffer is None:
            return None
        return np.round(buffer._arrays["durations"][: buffer.size(), 0] * 1000.0).astype(int)

    def test_pendulum(self):
        for arm in ("implicit_learned_admissions", "explicit_rollout", "explicit_generator",
                    "ct_sac_torque"):
            with self.subTest(arm=arm):
                reference, agent, _, _ = self.check_exact_restart(
                    "pendulum-swingup", arm, 200, 150, mode="irregular"
                )
                clock = reference["run"]["protocol"]["matched"]["testbed_config"]["clock"]
                self.assertEqual(clock["kind"], "two_tail_uniform")
                durations = self.irregular_durations(agent)
                if durations is not None:
                    # Both bounds occur (0.901^100 ~ 3e-5 that the 60 ms one would not);
                    # at most the upper bound, and the last interval of an episode may be
                    # shortened below the lower one.
                    self.assertLessEqual({4, 60}, set(durations.tolist()))
                    self.assertTrue(np.all((durations >= 1) & (durations <= 60)))

    def test_cartpole_under_exclusion(self):
        for arm in ("implicit_learned_admissions", "explicit_rollout", "explicit_generator",
                    "ct_sac_torque"):
            with self.subTest(arm=arm):
                reference, agent, _, _ = self.check_exact_restart(
                    "cartpole-swingup", arm, 320, 260, mode="irregular"
                )
                durations = self.irregular_durations(agent)
                if durations is not None:
                    self.assertEqual(agent.model_step, 0.005)
                    self.assertLessEqual({2, 30}, set(durations.tolist()))
                    self.assertTrue(np.all((durations >= 1) & (durations <= 30)))
                    self.assertEqual(reference["contact_handling"]["mode"], "exclude")


class TestPolicyDiagnostics(unittest.TestCase):
    """Every evaluation records the temperature and the policy's spread (since 2026-11-11)."""

    def test_every_evaluation_records_them(self):
        with tempfile.TemporaryDirectory() as output:
            for arm in ("implicit_learned_admissions", "explicit_rollout", "explicit_generator",
                        "ct_sac_torque"):
                with self.subTest(arm=arm):
                    run = tiny_run("pendulum-swingup", arm, output, 80, mode="irregular")
                    result, agent = run_captured(run)
                    self.assertEqual([c["interactions"] for c in result["evaluations"]], [0, 40, 80])
                    for point in result["evaluations"]:
                        self.assertEqual(set(point["policy"]), set(P.POLICY_DIAGNOSTIC_KEYS))
                    first, last = (result["evaluations"][i]["policy"] for i in (0, -1))
                    for value in (last["alpha_c"], last["alpha_decision"], last["log_std_mean"]):
                        self.assertTrue(np.isfinite(value))
                    self.assertGreater(last["alpha_decision"], 0.0)
                    # Draws nothing from the global stream; the same state gives the same values.
                    state = torch.get_rng_state()
                    again = agent.policy_diagnostics()
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    self.assertEqual(again, agent.policy_diagnostics())
                    if arm == "ct_sac_torque":
                        self.assertIsNone(last["entropy_estimate"])
                        self.assertEqual(last["entropy_rows"], 0)
                        self.assertEqual(again["alpha_decision"], agent.algorithm.alpha)
                    else:
                        self.assertIsNone(first["entropy_estimate"])
                        self.assertEqual(first["entropy_rows"], 0)
                        # exp(log 0.1) in float32, the agents' default dtype: 0.09999999403953552.
                        self.assertAlmostEqual(first["alpha_c"], 0.1, delta=1e-6)
                        self.assertEqual(
                            last["entropy_rows"],
                            min(P_BASE.POLICY_DIAGNOSTIC_ROWS, agent.replay.buffers["train"].size()),
                        )
                        self.assertGreater(last["entropy_rows"], 0)
                        self.assertTrue(np.isfinite(last["entropy_estimate"]))
                        self.assertEqual(again["alpha_c"], float(agent.alpha_c))
                        self.assertAlmostEqual(
                            again["log_std_mean"],
                            float(agent.actor.log_standard_deviation().mean()),
                            places=12,
                        )
                    # The record on disk carries them too.
                    path = P.run_output_directory(run) / "run_record.json"
                    (loaded,) = load_run_records([str(path)])
                    self.assertEqual(loaded["evaluations"][-1]["policy"], last)


class TestRestartMechanics(ExactRestartMixin, unittest.TestCase):
    env_id = "pendulum-swingup"
    arm = "explicit_rollout"

    def test_signal_writes_the_checkpoint_and_exits_75(self):
        """SIGUSR1 during the evaluation at 40: checkpoint at the next safe point, then
        run_ph_sac.main exits with RESUMABLE_EXIT_CODE; rerunning it finishes the run."""
        evaluate = P.evaluate_fixed_seeds
        calls = []

        def evaluate_and_signal(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                signal.raise_signal(signal.SIGUSR1)
            return evaluate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reference, _ = run_captured(tiny_run(self.env_id, self.arm, first, 90))
            run = tiny_run(self.env_id, self.arm, second, 90)
            protocol = copy.deepcopy(run["protocol"])
            argv = ["--config", "unused.json", "--resume"]
            with mock.patch.object(run_ph_sac, "load_protocols", lambda args: [protocol]), \
                    mock.patch.object(P, "evaluate_fixed_seeds", evaluate_and_signal), \
                    mock.patch.object(run_ph_sac, "write_reports", lambda p, r: []):
                with self.assertRaises(SystemExit) as caught:
                    run_ph_sac.main(argv)
            self.assertEqual(caught.exception.code, R.RESUMABLE_EXIT_CODE)
            # The handlers are gone again.
            self.assertIs(signal.getsignal(signal.SIGUSR1), signal.SIG_DFL)
            folder = P.run_output_directory(run)
            payload = torch.load(folder / R.RESUME_NAME, weights_only=False)
            self.assertEqual(payload["interactions"], 40)
            with mock.patch.object(run_ph_sac, "load_protocols", lambda args: [protocol]), \
                    mock.patch.object(run_ph_sac, "write_reports", lambda p, r: []):
                run_ph_sac.main(argv)
            (resumed,) = load_run_records([str(folder / "run_record.json")])
            self.assertTrue((folder / R.COMPLETE_NAME).exists())
        self.assertEqual(returns(reference), returns(resumed))
        self.assertEqual(
            without_wall(reference["counters"]), without_wall(resumed["counters"])
        )

    def test_atomic_write_keeps_the_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / R.RESUME_NAME
            R.atomic_torch_save({"interactions": 1, "rows": torch.arange(5)}, path)
            before = path.read_bytes()
            real_save = torch.save

            def failing_save(payload, handle):
                handle.write(b"partial checkpoint")
                raise OSError("disk full (simulated)")

            with mock.patch.object(R.torch, "save", failing_save):
                with self.assertRaises(OSError):
                    R.atomic_torch_save({"interactions": 2}, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(torch.load(path, weights_only=False)["interactions"], 1)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), [R.RESUME_NAME])
            self.assertIs(torch.save, real_save)

    def test_a_restart_with_other_settings_is_refused(self):
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, self.arm, output, 90)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=60))
            self.assertIsInstance(stopped, R.ResumableStop)
            changed = tiny_run(self.env_id, self.arm, output, 90, learning_rate=1e-3)
            with self.assertRaisesRegex(R.ResumeMismatch, "training.learning_rate"):
                run_captured(changed, R.resume_options(0))
            rescheduled = tiny_run(self.env_id, self.arm, output, 90, schedule=(50, 2))
            with self.assertRaisesRegex(R.ResumeMismatch, "plant_step"):
                run_captured(rescheduled, R.resume_options(0))
            convention = copy.deepcopy(run)
            for config in ("explicit_config", "implicit_config"):
                convention["protocol"][config]["model_integrator"] = "semi_implicit"
            with self.assertRaisesRegex(R.ResumeMismatch, "model_integrator"):
                run_captured(convention, R.resume_options(0))
            # The same settings in another output folder and arm list are the same run.
            same = copy.deepcopy(run)
            same["protocol"]["arms"] = list(P.all_arms())
            self.assertEqual(R.run_fingerprint(same), R.run_fingerprint(run))
            finished, _ = run_captured(run, R.resume_options(0))
            self.assertEqual(finished["counters"]["interactions"], 90)
            with self.assertRaisesRegex(R.ResumeMismatch, R.COMPLETE_NAME):
                run_captured(changed, R.resume_options(0))

    def test_a_finished_run_is_skipped(self):
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, self.arm, output, 60)
            first, _ = run_captured(run, R.resume_options(25))
            folder = P.run_output_directory(run)
            self.assertFalse((folder / R.RESUME_NAME).exists())
            marker = json.loads((folder / R.COMPLETE_NAME).read_text())
            self.assertEqual(marker["interactions"], 60)
            again, agent = run_captured(run, R.resume_options(25))
            self.assertIsNone(agent)
            self.assertEqual(returns(again), returns(first))

    def test_periodic_checkpoints_follow_the_cadence(self):
        written = []
        save = R.atomic_torch_save

        def record(payload, path):
            written.append(payload["interactions"])
            return save(payload, path)

        with tempfile.TemporaryDirectory() as output, \
                mock.patch.object(R, "atomic_torch_save", record):
            run_captured(tiny_run(self.env_id, self.arm, output, 90), R.resume_options(25))
        # Every 25 decisions and after the evaluations at 40 and 80.
        self.assertEqual(written, [25, 40, 50, 75, 80])

    def test_without_the_flag_nothing_changes(self):
        with tempfile.TemporaryDirectory() as plain_dir, tempfile.TemporaryDirectory() as other:
            plain, plain_agent = run_captured(tiny_run(self.env_id, self.arm, plain_dir, 90))
            folder = P.run_output_directory(tiny_run(self.env_id, self.arm, plain_dir, 90))
            self.assertFalse((folder / R.RESUME_NAME).exists())
            self.assertFalse((folder / R.COMPLETE_NAME).exists())
            self.assertIsNone(plain_agent._resume_hook)
            self.assertIs(signal.getsignal(signal.SIGUSR1), signal.SIG_DFL)
            restartable, agent = run_captured(
                tiny_run(self.env_id, self.arm, other, 90), R.resume_options(20)
            )
        # Writing resumable checkpoints draws from no stream: the same run either way.
        self.assert_same_run(plain, plain_agent, restartable, agent)
        args = run_ph_sac.parse_args(["--env-id", "cartpole-swingup"])
        self.assertIsNone(run_ph_sac.resume_settings(args))
        args = run_ph_sac.parse_args(["--env-id", "cartpole-swingup", "--resume"])
        self.assertEqual(
            run_ph_sac.resume_settings(args),
            {"every_interactions": R.DEFAULT_RESUME_EVERY, "stop_after_interactions": None},
        )
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            run_ph_sac.parse_args(
                ["--env-id", "cartpole-swingup", "--stop-after-interactions", "5"]
            )


class TestPlantStepSchedule(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(plant_step_schedule({}), (0, 1))
        self.assertEqual(
            plant_step_schedule({"plant_step_dense_until": 100000, "plant_step_every": 4}),
            (100000, 4),
        )
        for bad in (
            {"plant_step_every": 0},
            {"plant_step_every": True},
            {"plant_step_every": 2.0},
            {"plant_step_dense_until": -1},
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "plant_step"):
                plant_step_schedule(bad)
        config = tiny_run("pendulum-swingup", "explicit_rollout", "/nonexistent", 10)
        explicit = copy.deepcopy(config["protocol"]["explicit_config"])
        explicit["training"]["plant_step_every"] = 0
        with self.assertRaisesRegex(ValueError, "plant_step_every"):
            validate_configuration(explicit)

    def test_dense_schedule_is_the_default_bitwise(self):
        """Explicit keys (D, 1) run exactly as no keys: one plant step per row."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain, plain_agent = run_captured(
                tiny_run("pendulum-swingup", "implicit_learned_admissions", a, 80)
            )
            keyed, keyed_agent = run_captured(
                tiny_run("pendulum-swingup", "implicit_learned_admissions", b, 80,
                         schedule=(7, 1))
            )
        self.assertEqual(network_hashes(plain_agent), network_hashes(keyed_agent))
        self.assertEqual(returns(plain), returns(keyed))
        self.assertEqual(plain["counters"]["plant_gradient_steps"], 5 + 30)

    def test_every_k_from_the_start_takes_the_expected_steps(self):
        with tempfile.TemporaryDirectory() as output:
            for arm, warm in (("implicit_learned_admissions", 5), ("explicit_generator", 0)):
                with self.subTest(arm=arm):
                    result, agent = run_captured(
                        tiny_run("pendulum-swingup", arm, output, 90, schedule=(0, 4))
                    )
                    # Rows 51..90 are offered; every 4th of all rows after decision 0.
                    offered = 90 - WARMUP_DECISIONS
                    self.assertEqual(agent._plant_rows_after_dense, offered)
                    self.assertEqual(
                        result["counters"]["plant_gradient_steps"], warm + offered // 4
                    )

    def test_known_plant_arm_ignores_the_schedule(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            dense, dense_agent = run_captured(
                tiny_run("pendulum-swingup", "implicit_oracle", a, 80)
            )
            scheduled, scheduled_agent = run_captured(
                tiny_run("pendulum-swingup", "implicit_oracle", b, 80, schedule=(0, 4))
            )
        self.assertEqual(network_hashes(dense_agent), network_hashes(scheduled_agent))
        self.assertEqual(returns(dense), returns(scheduled))

    def test_records_store_the_schedule_and_old_records_load_dense(self):
        self.assertEqual(record_plant_step_schedule({}), {"dense_until": 0, "every": 1})
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run("pendulum-swingup", "ct_sac_torque", output, 60, schedule=(30, 2))
            result, _ = run_captured(run)
            self.assertEqual(result["plant_step_schedule"], {"dense_until": 30, "every": 2})
            path = P.run_output_directory(run) / "run_record.json"
            (loaded,) = load_run_records([str(path)])
            self.assertEqual(loaded["plant_step_schedule"], {"dense_until": 30, "every": 2})
            stored = json.loads(path.read_text())
            del stored["result"]["plant_step_schedule"]
            path.write_text(json.dumps(stored))
            (old,) = load_run_records([str(path)])
            self.assertEqual(old["plant_step_schedule"], {"dense_until": 0, "every": 1})

    def test_table_rows_carry_the_confirmed_schedule(self):
        """Dense to 100k decisions, then every 4th row: proposed 2026-11-10, confirmed
        2026-11-11 (make_ct_ph_sac_table.PLANT_STEP_SCHEDULE)."""
        rows = {(e.env_id, e.mode): e for e in table.load_ph_table()}
        in_house = {"windy_pendulum", "acrobot_xk"}
        for (env_id, mode), entry in rows.items():
            with self.subTest(row=(env_id, mode)):
                schedule = {
                    k: v for k, v in entry.config["training"].items() if k.startswith("plant_step")
                }
                if env_id.split("-")[0] in in_house:
                    self.assertEqual(schedule, {})
                else:
                    self.assertEqual(
                        schedule, {"plant_step_dense_until": 100000, "plant_step_every": 4}
                    )


class TestDecisionWarmUpRestart(ExactRestartMixin, unittest.TestCase):
    """The training key warmup_decisions on pendulum-swingup (1 s episodes of 50 decisions
    of 20 ms): a warm-up of 70 decisions spans the episode end at decision 50 and ends
    inside the second episode, which the policy continues."""

    env_id = "pendulum-swingup"
    WARMUP = 70

    def test_stop_inside_the_warm_up(self):
        # Stopped at 60: inside the warm-up, after its first episode end.
        for arm in ("implicit_torque", "explicit_rollout_torque", "explicit_generator_torque"):
            with self.subTest(arm=arm):
                reference, agent, at, payload = self.check_exact_restart(
                    self.env_id, arm, 130, 60, warmup_decisions=self.WARMUP
                )
                # The safe point that follows the stop request: the warm-up loop's top.
                self.assertEqual(at, 60)
                state = payload["agent"]
                self.assertIs(state["warmup_skipped"], False)
                self.assertEqual(state["warmup_episodes_done"], 0)
                steps = [
                    value for key, value in state["counters"].items()
                    if key.endswith("_gradient_steps")
                ]
                self.assertEqual(sum(steps), 0)
                self.assertEqual(agent.warmup_decisions, self.WARMUP)
                counters = reference["counters"]
                if arm.startswith("implicit"):
                    # The warm fit (5 plant steps) after decision 70, then a value step
                    # and a plant step on each of the 60 rows that follow.
                    self.assertEqual(counters["value_gradient_steps"], 130 - self.WARMUP)
                    self.assertEqual(counters["plant_gradient_steps"], 5 + 130 - self.WARMUP)
                else:
                    self.assertEqual(counters["critic_gradient_steps"], 130 - self.WARMUP)

    def test_fingerprint_holds_the_key_only_when_given(self):
        for arm in ("implicit_torque", "explicit_generator_torque", "ct_sac_torque"):
            with self.subTest(arm=arm):
                plain = tiny_run(self.env_id, arm, "/nonexistent", 100)
                keyed = tiny_run(
                    self.env_id, arm, "/nonexistent", 100, warmup_decisions=self.WARMUP
                )
                # Absent: no trace of the key in what the fingerprint hashes, so the
                # fingerprint is that of the code before the key.
                self.assertNotIn("warmup_decisions", json.dumps(R.run_settings(plain)))
                self.assertNotEqual(R.run_fingerprint(plain), R.run_fingerprint(keyed))
                self.assertEqual(
                    R._differences(R.run_settings(plain), R.run_settings(keyed)),
                    [
                        "protocol.explicit_config.training.warmup_decisions",
                        "protocol.implicit_config.training.warmup_decisions",
                    ],
                )

    def test_a_restart_across_the_key_is_refused(self):
        arm = "explicit_rollout_torque"
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, arm, output, 90)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=30))
            self.assertIsInstance(stopped, R.ResumableStop)
            keyed = tiny_run(self.env_id, arm, output, 90, warmup_decisions=self.WARMUP)
            with self.assertRaisesRegex(R.ResumeMismatch, "warmup_decisions"):
                run_captured(keyed, R.resume_options(0))

    def test_protocol_accepts_and_checks_the_key(self):
        run = tiny_run(
            self.env_id, "implicit_torque", "/nonexistent", 10, warmup_decisions=self.WARMUP
        )
        for name in ("explicit_config", "implicit_config"):
            self.assertEqual(run["protocol"][name]["training"]["warmup_decisions"], self.WARMUP)
        for bad in (-1, True, 2.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "warmup_decisions"):
                tiny_run(self.env_id, "implicit_torque", "/nonexistent", 10, warmup_decisions=bad)
        # Other unknown training keys are still refused.
        raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
        raw["training"]["warmup_decision"] = 10
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac.protocol_from_config(raw, "typo")
        # No table column: a config that sets it stays a JSON file.
        raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
        raw["training"]["warmup_decisions"] = self.WARMUP
        with self.assertRaisesRegex(ValueError, "training.warmup_decisions"):
            table.config_to_row(raw, env_id=self.env_id)

    def test_every_ct_ph_sac_arm_reads_the_key(self):
        # The paper's CT-PH-SAC arms, the impedance-law and implicit ablations and the
        # value-step ablation arms: every agent built on PHAlgorithm.
        arms = [arm for arm in P.all_arms() if arm["algorithm"] != "ct_sac"]
        names = {arm["name"] for arm in arms}
        self.assertTrue({arm["name"] for arm in P.step_ablation_arms()} <= names)
        for arm in arms:
            with self.subTest(arm=arm["name"]):
                run = tiny_run(
                    self.env_id, arm["name"], "/nonexistent", 10, warmup_decisions=self.WARMUP
                )
                testbed = run["protocol"]["matched"]["testbed_config"]
                env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                agent = P.make_training_agent(run, env)
                self.assertEqual(agent.warmup_decisions, self.WARMUP)
                env.close()


class TestNetworkStartKeys(unittest.TestCase):
    """The column-less network keys actor_log_std_init and input_matrix_start
    (common.ph_config.NETWORK_COLUMNLESS_KEYS) through the JSON loader, the arms and the
    resume fingerprint: absent and stated at the default compare equal, any other value
    must match. Building only, apart from two stops inside the warm-up (one run without
    the keys, one with input_matrix_start "zero")."""

    env_id = "pendulum-swingup"
    KEYS = {"actor_log_std_init": -0.5, "input_matrix_start": "random"}
    DEFAULTS = {"actor_log_std_init": -1.0, "input_matrix_start": "true"}
    ZERO = {"input_matrix_start": "zero"}

    def test_fingerprint_holds_the_keys_only_off_their_defaults(self):
        for arm in ("implicit_torque", "explicit_generator_torque", "ct_sac_torque"):
            with self.subTest(arm=arm):
                plain = tiny_run(self.env_id, arm, "/nonexistent", 100)
                text = json.dumps(R.run_settings(plain))
                for key in NETWORK_COLUMNLESS_KEYS:
                    self.assertNotIn(key, text)
                for stated in (self.DEFAULTS, {"actor_log_std_init": -1}):
                    run = tiny_run(self.env_id, arm, "/nonexistent", 100, network=stated)
                    self.assertEqual(R.run_fingerprint(run), R.run_fingerprint(plain))
                for key, value in self.KEYS.items():
                    keyed = tiny_run(self.env_id, arm, "/nonexistent", 100, network={key: value})
                    self.assertNotEqual(R.run_fingerprint(keyed), R.run_fingerprint(plain))
                    self.assertEqual(
                        R._differences(R.run_settings(plain), R.run_settings(keyed)),
                        [
                            f"protocol.explicit_config.network.{key}",
                            f"protocol.implicit_config.network.{key}",
                        ],
                    )
                # One value however written, and only that value.
                fingerprints = [
                    R.run_fingerprint(
                        tiny_run(self.env_id, arm, "/nonexistent", 100, network={"actor_log_std_init": v})
                    )
                    for v in (0, 0.0, -0.25)
                ]
                self.assertEqual(fingerprints[0], fingerprints[1])
                self.assertNotEqual(fingerprints[0], fingerprints[2])

    def test_a_restart_across_a_key_is_refused_and_across_a_stated_default_is_not(self):
        arm = "explicit_rollout_torque"
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, arm, output, 90)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=30))
            self.assertIsInstance(stopped, R.ResumableStop)
            path = P.run_output_directory(run) / R.RESUME_NAME
            for key, value in self.KEYS.items():
                keyed = tiny_run(self.env_id, arm, output, 90, network={key: value})
                with self.subTest(key=key), self.assertRaisesRegex(R.ResumeMismatch, key):
                    R.load_resume_payload(path, keyed)
            stated = tiny_run(self.env_id, arm, output, 90, network=self.DEFAULTS)
            payload = R.load_resume_payload(path, stated)
            self.assertGreaterEqual(payload["interactions"], 30)

    def test_loader_accepts_and_checks_the_keys_and_the_table_has_no_column(self):
        run = tiny_run(self.env_id, "implicit_torque", "/nonexistent", 10, network=self.KEYS)
        for name in ("explicit_config", "implicit_config"):
            for key, value in self.KEYS.items():
                self.assertEqual(run["protocol"][name]["network"][key], value)
        for key, bad in (
            ("actor_log_std_init", 5.0),
            ("actor_log_std_init", "x"),
            ("input_matrix_start", "learned"),
        ):
            with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, key):
                tiny_run(self.env_id, "implicit_torque", "/nonexistent", 10, network={key: bad})
        # Other unknown network keys are still refused.
        raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
        raw["network"]["input_matrix_starts"] = "random"
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac.protocol_from_config(raw, "typo")
        # No table column: a config that sets one stays a JSON file.
        for key, value in self.KEYS.items():
            raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
            raw["network"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"network.{key}"):
                table.config_to_row(raw, env_id=self.env_id)

    def test_every_arm_reads_the_keys_and_the_ct_sac_arms_ignore_them(self):
        # The CT-PH-SAC arms (the paper's, the impedance-law and implicit ablations, the
        # value-step ablations) start their actor and plant from the keys; the known-plant
        # arm refuses "random"; the CT-SAC arms, with or without the state features that
        # read the network config, build the same networks as without the keys.
        feature_arm = next(
            arm for arm in P.baseline_feature_arms() if arm["name"] == "ct_sac_features"
        )
        for arm in [*P.all_arms(), feature_arm]:
            with self.subTest(arm=arm["name"]):
                source = "ct_sac_torque" if arm is feature_arm else arm["name"]
                keyed = dict(
                    tiny_run(self.env_id, source, "/nonexistent", 10, network=self.KEYS), arm=arm
                )
                testbed = keyed["protocol"]["matched"]["testbed_config"]
                env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    if arm["plant_mode"] == "oracle":
                        with self.assertRaisesRegex(ValueError, "input_matrix_start"):
                            P.make_training_agent(keyed, env)
                        env.close()
                        continue
                    agent = P.make_training_agent(keyed, env)
                if arm["algorithm"] == "ct_sac":
                    self.assertNotIn("[input-start]", printed.getvalue())
                    plain_run = dict(tiny_run(self.env_id, source, "/nonexistent", 10), arm=arm)
                    plain_env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                    plain = P.make_training_agent(plain_run, plain_env)
                    self.assertEqual(network_hashes(agent), network_hashes(plain))
                    plain_env.close()
                else:
                    self.assertEqual(
                        agent.actor.log_std.detach().tolist(), [-0.5] * agent.actor.decision_dim
                    )
                    # pendulum-swingup's G is 1x1; training seed 0 (float32 default dtype).
                    self.assertEqual(
                        agent.online_plant.force_matrix.detach().tolist(), [[-0.936945915222168]]
                    )
                    self.assertIn("[input-start] random seed 0 true ", printed.getvalue())
                env.close()

    def test_fingerprint_of_the_zero_start_differs_from_the_default_and_the_random_start(self):
        random_start = {"input_matrix_start": "random"}
        for arm in ("implicit_torque", "explicit_generator_torque", "ct_sac_torque"):
            with self.subTest(arm=arm):
                plain = tiny_run(self.env_id, arm, "/nonexistent", 100)
                stated = tiny_run(
                    self.env_id, arm, "/nonexistent", 100, network={"input_matrix_start": "true"}
                )
                drawn = tiny_run(self.env_id, arm, "/nonexistent", 100, network=random_start)
                zero = tiny_run(self.env_id, arm, "/nonexistent", 100, network=self.ZERO)
                for name in ("explicit_config", "implicit_config"):
                    network = zero["protocol"][name]["network"]
                    self.assertEqual(network["input_matrix_start"], "zero")
                self.assertNotEqual(R.run_fingerprint(zero), R.run_fingerprint(plain))
                self.assertNotEqual(R.run_fingerprint(zero), R.run_fingerprint(stated))
                self.assertNotEqual(R.run_fingerprint(zero), R.run_fingerprint(drawn))
                for other in (plain, drawn):
                    self.assertEqual(
                        R._differences(R.run_settings(other), R.run_settings(zero)),
                        [
                            "protocol.explicit_config.network.input_matrix_start",
                            "protocol.implicit_config.network.input_matrix_start",
                        ],
                    )

    def test_a_restart_of_a_zero_start_run_needs_the_zero_start(self):
        arm = "explicit_rollout_torque"
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, arm, output, 90, network=self.ZERO)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=30))
            self.assertIsInstance(stopped, R.ResumableStop)
            self.assertIn("[input-start] zero seed 0 true ", printed.getvalue())
            path = P.run_output_directory(run) / R.RESUME_NAME
            for label, network in (
                ("absent", None),
                ("true", {"input_matrix_start": "true"}),
                ("random", {"input_matrix_start": "random"}),
            ):
                other = tiny_run(self.env_id, arm, output, 90, network=network)
                with self.subTest(start=label), self.assertRaisesRegex(
                    R.ResumeMismatch, "input_matrix_start"
                ):
                    R.load_resume_payload(path, other)
            # The same start, from a config built anew, resumes.
            again = tiny_run(self.env_id, arm, output, 90, network=self.ZERO)
            payload = R.load_resume_payload(path, again)
            self.assertGreaterEqual(payload["interactions"], 30)

    def test_every_arm_reads_the_zero_start_and_the_ct_sac_arms_ignore_it(self):
        # As test_every_arm_reads_the_keys_and_the_ct_sac_arms_ignore_them, for "zero": the
        # CT-PH-SAC arms start their plant's G at exactly zero and print it, the known-plant
        # arm refuses it, the CT-SAC arms build the same networks as without the key.
        feature_arm = next(
            arm for arm in P.baseline_feature_arms() if arm["name"] == "ct_sac_features"
        )
        for arm in [*P.all_arms(), feature_arm]:
            with self.subTest(arm=arm["name"]):
                source = "ct_sac_torque" if arm is feature_arm else arm["name"]
                keyed = dict(
                    tiny_run(self.env_id, source, "/nonexistent", 10, network=self.ZERO), arm=arm
                )
                testbed = keyed["protocol"]["matched"]["testbed_config"]
                env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    if arm["plant_mode"] == "oracle":
                        with self.assertRaisesRegex(ValueError, "input_matrix_start 'zero'"):
                            P.make_training_agent(keyed, env)
                        env.close()
                        continue
                    agent = P.make_training_agent(keyed, env)
                if arm["algorithm"] == "ct_sac":
                    self.assertNotIn("[input-start]", printed.getvalue())
                    plain_run = dict(tiny_run(self.env_id, source, "/nonexistent", 10), arm=arm)
                    plain_env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                    plain = P.make_training_agent(plain_run, plain_env)
                    self.assertEqual(network_hashes(agent), network_hashes(plain))
                    plain_env.close()
                else:
                    start = agent.online_plant.force_matrix.detach()
                    self.assertGreater(start.numel(), 0)
                    self.assertTrue(torch.equal(start, torch.zeros_like(start)))
                    lines = [
                        line
                        for line in printed.getvalue().splitlines()
                        if line.startswith("[input-start]")
                    ]
                    self.assertEqual(len(lines), 1)
                    self.assertTrue(lines[0].startswith("[input-start] zero seed 0 true "))
                    self.assertTrue(lines[0].endswith(f" start {start.tolist()}"))
                env.close()


class TestRewardObservation(ExactRestartMixin, unittest.TestCase):
    """The protocol key reward_observation (common.ph_config.REWARD_OBSERVATION_KEY) through
    the JSON loader, the command line, the records and the resume fingerprint: absent and
    "interval" fingerprint alike; "end_state" enters a CT-PH-SAC arm's settings as resolved
    reward_observation and a CT-SAC arm's as its ct_sac_reward_source; a run under it
    restarts exactly, and a restart across it is refused."""

    env_id = "pendulum-swingup"
    END_STATE = {"reward_observation": "end_state"}
    CT_PH_SAC_ARMS = (
        "implicit_torque", "explicit_rollout_torque", "explicit_generator_torque",
        "implicit_learned_admissions", "implicit_critic",
    )

    def test_fingerprint_holds_the_key_only_off_its_default(self):
        for arm in (*self.CT_PH_SAC_ARMS, "ct_sac_torque"):
            with self.subTest(arm=arm):
                plain = tiny_run(self.env_id, arm, "/nonexistent", 100)
                stated = tiny_run(
                    self.env_id, arm, "/nonexistent", 100,
                    protocol={"reward_observation": "interval"},
                )
                keyed = tiny_run(self.env_id, arm, "/nonexistent", 100, protocol=self.END_STATE)
                self.assertNotIn("reward_observation", json.dumps(R.run_settings(plain)))
                self.assertEqual(R.run_settings(stated), R.run_settings(plain))
                self.assertEqual(R.run_fingerprint(stated), R.run_fingerprint(plain))
                self.assertNotIn("reward_observation", R.run_settings(keyed)["protocol"])
                self.assertNotEqual(R.run_fingerprint(keyed), R.run_fingerprint(plain))
                changed = R._differences(R.run_settings(plain), R.run_settings(keyed))
                if arm == "ct_sac_torque":
                    self.assertEqual(changed, ["resolved.ct_sac_reward_source"])
                    self.assertEqual(
                        R.run_settings(keyed)["resolved"]["ct_sac_reward_source"], "end_state"
                    )
                else:
                    self.assertEqual(changed, ["resolved.reward_observation"])
                    self.assertEqual(
                        R.run_settings(keyed)["resolved"]["reward_observation"], "end_state"
                    )

    def test_ct_sac_fingerprint_is_its_end_state_sources(self):
        source = {"ct_sac_reward_source": "end_state"}
        both = {**source, **self.END_STATE}
        runs = {
            name: tiny_run(self.env_id, "ct_sac_torque", "/nonexistent", 100, protocol=keys)
            for name, keys in (("plain", None), ("source", source), ("key", self.END_STATE),
                               ("both", both))
        }
        prints = {name: R.run_fingerprint(run) for name, run in runs.items()}
        self.assertEqual(prints["key"], prints["source"])
        self.assertEqual(prints["both"], prints["source"])
        self.assertNotEqual(prints["key"], prints["plain"])
        # A CT-PH-SAC arm reads ct_sac_reward_source not at all, reward_observation as its own.
        plain = tiny_run(self.env_id, "implicit_torque", "/nonexistent", 100)
        sourced = tiny_run(self.env_id, "implicit_torque", "/nonexistent", 100, protocol=source)
        keyed = tiny_run(self.env_id, "implicit_torque", "/nonexistent", 100, protocol=both)
        self.assertEqual(R.run_fingerprint(sourced), R.run_fingerprint(plain))
        self.assertNotEqual(R.run_fingerprint(keyed), R.run_fingerprint(plain))
        self.assertNotIn("ct_sac_reward_source", R.run_settings(keyed)["resolved"])
        # A contradiction is refused when the protocol is built.
        with self.assertRaisesRegex(ValueError, "contradicts"):
            tiny_run(
                self.env_id, "ct_sac_torque", "/nonexistent", 100,
                protocol={**self.END_STATE, "ct_sac_reward_source": "interval"},
            )

    def test_a_restart_across_the_mode_is_refused(self):
        arm = "explicit_rollout_torque"
        with tempfile.TemporaryDirectory() as plain_dir, tempfile.TemporaryDirectory() as keyed_dir:
            plain = tiny_run(self.env_id, arm, plain_dir, 90)
            keyed = tiny_run(self.env_id, arm, keyed_dir, 90, protocol=self.END_STATE)
            for run in (plain, keyed):
                stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=30))
                self.assertIsInstance(stopped, R.ResumableStop)
            plain_path = P.run_output_directory(plain) / R.RESUME_NAME
            keyed_path = P.run_output_directory(keyed) / R.RESUME_NAME
            # Each run's checkpoint, read for the other mode's run: the fingerprint names it.
            for path, other_dir, keys in (
                (plain_path, plain_dir, self.END_STATE),
                (keyed_path, keyed_dir, None),
            ):
                other = tiny_run(self.env_id, arm, other_dir, 90, protocol=keys)
                with self.subTest(path=str(path)), self.assertRaisesRegex(
                    R.ResumeMismatch, "reward_observation"
                ):
                    R.load_resume_payload(path, other)
            # "interval" stated is the plain run's mode.
            stated = tiny_run(
                self.env_id, arm, plain_dir, 90, protocol={"reward_observation": "interval"}
            )
            self.assertGreaterEqual(R.load_resume_payload(plain_path, stated)["interactions"], 30)
            # A checkpoint loaded past the fingerprint meets the agents' own guard.
            payload = torch.load(keyed_path, weights_only=False)
            testbed = plain["protocol"]["matched"]["testbed_config"]
            env = make_env(testbed, decision_mode="torque", seed=0)
            agent = P.make_training_agent(plain, env)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                agent.load_resume_state(payload["agent"])
            env.close()

    def test_exact_restart_under_end_state(self):
        for env_id, budget, stop in (("pendulum-swingup", 110, 85), ("reacher-easy", 100, 75)):
            for arm in ("implicit_torque", "explicit_rollout_torque", "explicit_generator_torque"):
                with self.subTest(testbed=env_id, arm=arm):
                    reference, agent, _, payload = self.check_exact_restart(
                        env_id, arm, budget, stop, protocol=self.END_STATE
                    )
                    self.assertEqual(agent.reward_observation, "end_state")
                    self.assertEqual(agent.config["reward_observation"], "end_state")
                    self.assertEqual(reference["reward_observation"], "end_state")
                    self.assertIsInstance(reference["reward_fit"], list)
                    counters = reference["counters"]
                    self.assertGreater(counters["reward_gradient_steps"], 0)
                    if arm.startswith("explicit"):
                        # One reward step per train step, no warm fit.
                        self.assertEqual(
                            counters["reward_gradient_steps"], counters["critic_gradient_steps"]
                        )
                    arrays = payload["agent"]["replay"]["buffers"]["train"]["arrays"]
                    self.assertIn("end_rewards", arrays)
                    networks = payload["agent"]["networks"]
                    self.assertIn("reward", networks)
                    if arm.startswith("explicit"):
                        self.assertIn("target_reward", networks)
                        self.assertIn("reward_diagnostic_rng", payload["agent"]["loop"])

    def test_records_store_the_mode_and_old_records_load_interval(self):
        self.assertEqual(record_reward_observation({}), "interval")
        self.assertEqual(record_reward_observation({"reward_observation": "end_state"}), "end_state")
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(
                self.env_id, "explicit_generator_torque", output, 60, protocol=self.END_STATE
            )
            result, agent = run_captured(run)
            self.assertEqual(result["reward_observation"], "end_state")
            self.assertEqual(result["reward_fit"], list(agent.reward_records))
            path = P.run_output_directory(run) / "run_record.json"
            stored = json.loads(path.read_text())
            self.assertEqual(stored["result"]["reward_observation"], "end_state")
            (loaded,) = load_run_records([str(path)])
            self.assertEqual(loaded["reward_observation"], "end_state")
            del stored["result"]["reward_observation"]
            path.write_text(json.dumps(stored))
            (old,) = load_run_records([str(path)])
            self.assertEqual(old["reward_observation"], "interval")
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(self.env_id, "explicit_generator_torque", output, 60)
            result, _ = run_captured(run)
            self.assertNotIn("reward_observation", result)
            self.assertNotIn("reward_fit", result)
            path = P.run_output_directory(run) / "run_record.json"
            self.assertNotIn("reward_observation", json.loads(path.read_text())["result"])
            (loaded,) = load_run_records([str(path)])
            self.assertEqual(loaded["reward_observation"], "interval")

    def test_command_line_and_json_key(self):
        raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
        raw["protocol"]["reward_observation"] = "end_state"
        proto = run_ph_sac.protocol_from_config(copy.deepcopy(raw), "keyed")
        self.assertEqual(proto["reward_observation"], "end_state")
        for name in ("explicit_config", "implicit_config"):
            self.assertNotIn("reward_observation", proto[name])
        for bad in ("end", "End_state", 1):
            raw["protocol"]["reward_observation"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "reward_observation"):
                run_ph_sac.protocol_from_config(copy.deepcopy(raw), "bad")
        raw["protocol"].update(reward_observation="end_state", ct_sac_reward_source="interval")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            run_ph_sac.protocol_from_config(copy.deepcopy(raw), "contradiction")
        # The explicit and implicit sections may not set it (no such setting).
        for section in ("explicit", "implicit"):
            raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
            raw[section]["reward_observation"] = "end_state"
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "unknown settings"):
                run_ph_sac.protocol_from_config(raw, "section")
        # No table column: a config that sets it stays a JSON file.
        raw = copy.deepcopy(table.find_ph_entry(self.env_id, "top").config)
        raw["protocol"]["reward_observation"] = "end_state"
        with self.assertRaisesRegex(ValueError, "protocol.reward_observation"):
            table.config_to_row(raw, env_id=self.env_id)
        # The command line.
        args = run_ph_sac.parse_args(["--env-id", self.env_id])
        self.assertIsNone(args.reward_observation)
        self.assertNotIn("reward_observation", run_ph_sac.load_protocols(args)[0])
        args = run_ph_sac.parse_args(["--env-id", self.env_id, "--reward-observation", "end_state"])
        self.assertEqual(run_ph_sac.load_protocols(args)[0]["reward_observation"], "end_state")
        args = run_ph_sac.parse_args(
            ["--env-id", self.env_id, "--reward-observation", "end_state",
             "--ct-sac-reward-source", "interval"]
        )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            run_ph_sac.load_protocols(args)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_ph_sac.parse_args(["--env-id", self.env_id, "--reward-observation", "end"])


if __name__ == "__main__":
    unittest.main()
