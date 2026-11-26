"""CT-SAC with the CT-PH-SAC networks' inputs: the arms ct_sac_features,
ct_sac_released_features and ct_sac_released_timeout_features
(evaluations/ph_sac/protocol.baseline_feature_arms).

1. Inputs per task: on the ten paper tasks (their table rows, top mode) ct_sac_torque's
   actor and critics read the state itself (periodic angles wrapped to [-pi, pi),
   velocities unscaled); with state_features they read models.ph_sac.networks.
   StateFeatures of it, built from the row's network config: the dimension and values of
   the CT-PH-SAC networks' input, and the hand-written map (sine and cosine of the
   periodic positions, the other positions raw, the velocities over the scale).
2. The wrap seam: two states just either side of a periodic angle's seam (+-pi) are about
   2 pi apart as raw inputs and within 2 sin(eps) as features, directly and through the
   environment's own wrap.
3. Arms: each is its source arm's spec (ct_sac_torque, ct_sac_released,
   ct_sac_released_timeout) plus ct_sac_inputs "state_features", outside paper_arms() and
   all_arms(); --arms and --list; their agents get the explicit config's network as
   state_features, and their replay holds feature rows; the source arms' constructor
   arguments, observations and fingerprints are unchanged.
4. Runs: a tiny run of each arm through run_protocol, --report-only, and exact restarts
   (--resume) in training.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import get_args
from unittest import mock

import numpy as np
import torch

from algorithms.ph_baseline import TorqueCTSACBaseline
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.reporting import RECORD_NAME, load_run_records
from models.ph_sac.networks import StateFeatures
from tests.test_ph_baseline import _spy_step_dt
from tests.test_ph_resume import ExactRestartMixin, run_captured
from tests.test_ph_torque_decisions import FIXED_ARM_JSON, FIXED_ARMS, compact_json

CONFIG = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac" / "smoke_windy.json"
# Each feature arm and the arm whose spec it extends.
SOURCES = {
    "ct_sac_features": "ct_sac_torque",
    "ct_sac_released_features": "ct_sac_released",
    "ct_sac_released_timeout_features": "ct_sac_released_timeout",
}
FEATURE_ARMS = tuple(SOURCES)
# The ten paper tasks (table rows, top mode): the state dimension (positions, then
# velocities; no context), the periodic positions and the feature dimension
# 2 x periodic + other positions + velocities.
TASKS = {
    "pendulum-swingup": (2, (0,), 3),
    "acrobot-swingup": (4, (0, 1), 6),
    "cartpole-balance": (4, (1,), 5),
    "cartpole-balance_sparse": (4, (1,), 5),
    "cartpole-swingup": (4, (1,), 5),
    "cartpole-two_poles": (6, (1, 2), 8),
    "cartpole-three_poles": (8, (1, 2, 3), 11),
    "point_mass-easy": (4, (), 4),
    "lqr-lqr_2_1": (4, (), 4),
    "lqr-lqr_6_2": (12, (), 12),
}
# The pendulum-swingup row cut to 1 s episodes (50 decisions of 20 ms, no failure set).
EPISODE_SECONDS = 1.0


def row_testbed(env_id):
    """The TestbedConfig and network config of a table row (top mode)."""
    entry = table.find_ph_entry(env_id, "top")
    testbed = run_ph_sac._build_testbed(entry.config["testbed"], entry.config["testbed_settings"])
    return testbed, dict(entry.config["network"])


def row_baseline(env_id, *, features):
    """A small CT-SAC baseline on a table row's torque environment; with features, its
    networks read StateFeatures with the row's network config."""
    testbed, network = row_testbed(env_id)
    env = make_env(testbed, decision_mode="torque", seed=5)
    training = {
        "seed": 0,
        "device": "cpu",
        "replay_capacity": 1000,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "gradient_clip": 10.0,
        "warmup_episodes": 0,
        "validation_fraction": 0.0,
    }
    temperature = {"initial_alpha_c": 0.0, "learn_temperature": False, "target_entropy": -1.0}
    baseline = TorqueCTSACBaseline(
        env,
        training,
        temperature,
        polyak_rate=0.005,
        expectation_samples=1,
        net_arch=(16, 16),
        state_features=network if features else None,
    )
    return baseline, network


def feature_map(spec, network, states):
    """The CT-PH-SAC networks' input of states [..., D], in float64."""
    features = StateFeatures(spec, network)
    with torch.no_grad():
        return features(torch.as_tensor(np.asarray(states, dtype=np.float64))).numpy()


def smoke_protocol():
    return run_ph_sac.load_protocol(str(CONFIG))


def feature_protocol(output, budget, arms=FEATURE_ARMS):
    """The pendulum-swingup row with 1 s episodes, one warm-up episode (for the arms that
    warm up in episodes), batch 16, evaluations every 40 decisions on one reset seed, the
    named arms and training seed 0."""
    raw = copy.deepcopy(table.find_ph_entry("pendulum-swingup", "top").config)
    raw["training"].update(batch_size=16, warmup_episodes=1)
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[0],
        evaluation_reset_seeds=[3],
        evaluation_every_interactions=40,
        output_directory=str(output),
    )
    proto = run_ph_sac.protocol_from_config(raw, "tiny pendulum-swingup")
    for testbed in (
        proto["matched"]["testbed_config"],
        proto["explicit_config"]["testbed"],
        proto["implicit_config"]["testbed"],
    ):
        testbed["episode_duration"] = EPISODE_SECONDS
    proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in arms]
    P.validate_matched_quantities(proto)
    return proto


def feature_run(name, output, budget):
    (run,) = P.build_run_matrix(feature_protocol(output, budget, (name,)))
    return run


# ------------------------------------------------------------------------- inputs per task


class TestInputsPerTask(unittest.TestCase):
    def test_raw_and_feature_inputs(self):
        for env_id, (raw_dim, periodic, feature_dim) in TASKS.items():
            with self.subTest(task=env_id):
                raw, network = row_baseline(env_id, features=False)
                mapped, _ = row_baseline(env_id, features=True)
                spec = raw.env.state_spec
                self.assertEqual(tuple(spec["periodic_position_indices"]), periodic)
                self.assertEqual(spec.get("context_dim", 0), 0)
                self.assertNotIn("travel_position_scales", spec)
                self.assertEqual(spec["position_dim"] + spec["velocity_dim"], raw_dim)
                actions = int(spec["action_dim"])

                # ct_sac_torque's input: the state itself, the same object, wrapped.
                adapter = raw.algorithm.env
                self.assertIsNone(raw.input_features)
                self.assertIs(adapter.observation_space, raw.env.observation_space)
                self.assertEqual(adapter.observation_space.shape, (raw_dim,))
                state = raw.env.observe()
                self.assertIs(adapter.network_input(state), state)
                np.testing.assert_array_equal(adapter._last_obs, state.astype(np.float32))
                for index in periodic:
                    self.assertTrue(-math.pi <= state[index] < math.pi)
                model = raw.algorithm.model
                self.assertEqual(model.q_nets[0][0].in_features, raw_dim + actions)
                self.assertEqual(model.actor.body[0].in_features, raw_dim)

                # With state_features: the CT-PH-SAC networks' StateFeatures of the state.
                features = StateFeatures(spec, network)
                self.assertEqual(features.output_dim, feature_dim)
                self.assertEqual(
                    features.velocity_scale, float(network.get("velocity_feature_scale", 1.0))
                )
                adapter = mapped.algorithm.env
                self.assertEqual(adapter.observation_space.shape, (feature_dim,))
                state = mapped.env.observe()
                np.testing.assert_array_equal(
                    adapter._last_obs, feature_map(spec, network, state).astype(np.float32)
                )
                model = mapped.algorithm.model
                self.assertEqual(model.q_nets[0][0].in_features, feature_dim + actions)
                self.assertEqual(model.actor.body[0].in_features, feature_dim)

    def test_feature_values_on_sample_states(self):
        rng = np.random.default_rng(7)
        for env_id, (raw_dim, periodic, feature_dim) in TASKS.items():
            with self.subTest(task=env_id):
                baseline, network = row_baseline(env_id, features=True)
                spec = baseline.env.state_spec
                states = rng.normal(size=(6, raw_dim))
                states[:, list(periodic)] = rng.uniform(-math.pi, math.pi, (6, len(periodic)))
                mapped = baseline.algorithm.env.network_input(states)
                self.assertEqual(mapped.shape, (6, feature_dim))
                np.testing.assert_array_equal(mapped, feature_map(spec, network, states))
                # By hand: sine and cosine of the periodic positions, the other positions
                # raw, the velocities over the network's scale.
                positions = int(spec["position_dim"])
                other = [i for i in range(positions) if i not in periodic]
                scale = float(network.get("velocity_feature_scale", 1.0))
                by_hand = np.concatenate(
                    (
                        np.sin(states[:, list(periodic)]),
                        np.cos(states[:, list(periodic)]),
                        states[:, other],
                        states[:, positions:] / scale,
                    ),
                    axis=1,
                )
                np.testing.assert_allclose(mapped, by_hand, rtol=1e-12, atol=1e-15)
                # A single state maps as a row of the batch.
                np.testing.assert_array_equal(
                    baseline.algorithm.env.network_input(states[0]), mapped[0]
                )

    def test_the_wrap_seam(self):
        eps = 1e-3
        for env_id, (raw_dim, periodic, _) in TASKS.items():
            if not periodic:
                continue
            baseline, _ = row_baseline(env_id, features=True)
            base = np.linspace(0.1, 0.4, raw_dim)
            for index in periodic:
                with self.subTest(task=env_id, index=index):
                    below = base.copy()
                    below[index] = math.pi - eps
                    # pi + eps as the environment observes it, wrapped to [-pi, pi).
                    above = base.copy()
                    above[index] = -math.pi + eps
                    self.assertAlmostEqual(
                        np.abs(below - above).max(), 2.0 * math.pi - 2.0 * eps, places=9
                    )
                    adapter = baseline.algorithm.env
                    gap = np.abs(adapter.network_input(below) - adapter.network_input(above))
                    self.assertAlmostEqual(gap.max(), 2.0 * math.sin(eps), places=12)

    def test_the_seam_through_the_environment(self):
        """States at pi + eps and pi - eps set through reset: the environment wraps the
        first to -pi + eps, so the raw inputs straddle the seam; the features do not."""
        eps = 1e-3
        cases = (
            ("pendulum-swingup", 0, [math.pi + eps, 0.3], [math.pi - eps, 0.3]),
            (
                "cartpole-swingup",
                1,
                [0.2, math.pi + eps, -0.1, 0.3],
                [0.2, math.pi - eps, -0.1, 0.3],
            ),
        )
        for env_id, index, above, below in cases:
            with self.subTest(task=env_id):
                observed = {}
                for features in (False, True):
                    baseline, _ = row_baseline(env_id, features=features)
                    adapter = baseline.algorithm.env
                    observed[features] = [
                        np.asarray(
                            adapter.reset(options={"state": np.array(state)})[0],
                            dtype=np.float64,
                        )
                        for state in (above, below)
                    ]
                raw_above, raw_below = observed[False]
                self.assertAlmostEqual(raw_above[index], -math.pi + eps, places=12)
                self.assertAlmostEqual(raw_below[index], math.pi - eps, places=12)
                self.assertGreater(np.abs(raw_above - raw_below).max(), 2.0 * math.pi - 3.0 * eps)
                feature_above, feature_below = observed[True]
                self.assertLess(np.abs(feature_above - feature_below).max(), 3.0 * eps)


# ---------------------------------------------------------------------------------- arms


class TestFeatureArms(unittest.TestCase):
    def test_declarations(self):
        arms = {arm["name"]: arm for arm in P.baseline_feature_arms()}
        self.assertEqual(tuple(arms), FEATURE_ARMS)
        self.assertEqual(get_args(P.FeatureBaselineArmName), FEATURE_ARMS)
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        for name, source in SOURCES.items():
            with self.subTest(arm=name):
                spec = dict(arms[name])
                self.assertEqual(spec.pop("ct_sac_inputs"), "state_features")
                self.assertEqual(spec.pop("name"), name)
                expected = dict(catalog[source])
                expected.pop("name")
                self.assertEqual(spec, expected)
                self.assertNotIn("ct_sac_inputs", catalog[source])
                self.assertEqual(catalog[name], arms[name])
        self.assertEqual(
            {k: v for k, v in catalog["ct_sac_torque"].items()}, FIXED_ARMS["ct_sac_torque"]
        )
        everything = {arm["name"] for arm in P.all_arms()}
        self.assertFalse(set(FEATURE_ARMS) & everything)
        self.assertFalse(set(FEATURE_ARMS) & {arm["name"] for arm in P.paper_arms()})
        self.assertEqual(len(P.all_arms()), 17)
        names = [arm["name"] for arm in P.selectable_arms()]
        self.assertEqual([n for n in names if n in FEATURE_ARMS], list(FEATURE_ARMS))
        # Only the feature arms carry the field: these, then the implicit CT-SAC arms with
        # feature inputs (protocol.baseline_implicit_feature_arms, tests/test_ct_sac_implicit.py).
        carriers = [arm["name"] for arm in P.selectable_arms() if "ct_sac_inputs" in arm]
        self.assertEqual(
            carriers,
            list(FEATURE_ARMS)
            + [arm["name"] for arm in P.baseline_implicit_feature_arms()],
        )

    def test_arms_option_and_list(self):
        path = str(CONFIG)
        args = run_ph_sac.parse_args(
            ["--config", path, "--arms", *reversed(FEATURE_ARMS), "--seeds", "0"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual([arm["name"] for arm in proto["arms"]], list(FEATURE_ARMS))
        P.validate_matched_quantities(proto)
        lines = run_ph_sac.list_arms()
        listed = {line.split()[0]: line for line in lines[1:-1]}
        for name in FEATURE_ARMS:
            with self.subTest(arm=name):
                self.assertEqual(listed[name].split()[1:3], ["torque", "ct_sac"])
                self.assertTrue(listed[name].endswith("feature baseline (only with --arms)"))
        # A protocol without --arms runs the paper arms only.
        self.assertEqual(
            [arm["name"] for arm in smoke_protocol()["arms"]],
            [arm["name"] for arm in P.paper_arms()],
        )

    def test_changed_specs_are_rejected(self):
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        changed = []
        arm = copy.deepcopy(catalog["ct_sac_features"])
        del arm["ct_sac_inputs"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_torque"])
        arm["ct_sac_inputs"] = "state_features"
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_released_features"])
        arm["ct_sac_inputs"] = "raw"
        changed.append(arm)
        proto = smoke_protocol()
        for index, arm in enumerate(changed):
            with self.subTest(case=index):
                proto["arms"] = [arm]
                with self.assertRaises(ValueError):
                    P.validate_matched_quantities(proto)


class TestFeatureAgents(unittest.TestCase):
    """make_training_agent on the smoke protocol (the windy pendulum: one periodic angle,
    velocity_feature_scale 2 pi)."""

    NAMES = ("ct_sac_torque", "ct_sac_released", "ct_sac_released_timeout", *FEATURE_ARMS)

    @classmethod
    def setUpClass(cls):
        proto = smoke_protocol()
        proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in cls.NAMES]
        P.validate_matched_quantities(proto)
        cls.protocol = proto
        cls.runs = {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}

    def build(self, name):
        run = self.runs[name]
        env = make_env(
            self.protocol["matched"]["testbed_config"],
            decision_mode="torque",
            seed=run["training_seed"],
        )
        return P.make_training_agent(run, env)

    def test_constructor_arguments(self):
        explicit = self.protocol["explicit_config"]
        captured = {}
        for name, run in self.runs.items():
            with mock.patch("algorithms.ph_baseline.TorqueCTSACBaseline") as built:
                P.make_training_agent(run, None)
                captured[name] = built.call_args
        released_kwargs = {
            "polyak_rate": 0.02,
            "expectation_samples": 1,
            "warmup_decisions": 10000,
        }
        expected_kwargs = {
            "ct_sac_torque": {"polyak_rate": explicit["polyak_rate"], "expectation_samples": 1},
            "ct_sac_released": dict(released_kwargs, time_limit="bootstrap"),
            "ct_sac_released_timeout": dict(released_kwargs, time_limit="terminal"),
        }
        for name, kwargs in expected_kwargs.items():
            with self.subTest(arm=name):
                # The source arms: exactly the arguments of the code before the feature arms.
                self.assertEqual(captured[name][1], kwargs)
        for name, source in SOURCES.items():
            with self.subTest(arm=name):
                (env, training, temperature), kwargs = captured[name]
                self.assertEqual(kwargs, dict(expected_kwargs[source], state_features=explicit["network"]))
                self.assertEqual(training, captured[source][0][1])
                self.assertEqual(temperature, captured[source][0][2])
        self.assertEqual(explicit["network"]["velocity_feature_scale"], 2.0 * math.pi)

    def test_agents_inputs_and_replay(self):
        spec = None
        for name in ("ct_sac_torque", "ct_sac_features", "ct_sac_released_timeout_features"):
            with self.subTest(arm=name):
                agent = self.build(name)
                spec = agent.env.state_spec
                network = self.protocol["explicit_config"]["network"]
                transitions = _spy_step_dt(agent.env)
                agent.learn(30)
                buffer = agent.algorithm.replay_buffer
                self.assertEqual(buffer.size(), 30)
                states = np.stack([t["state"] for t in transitions])
                next_states = np.stack([t["next_state"] for t in transitions])
                if name == "ct_sac_torque":
                    self.assertIsNone(agent.input_features)
                    self.assertEqual(agent.algorithm.env.observation_space.shape, (2,))
                    expected, expected_next = states, next_states
                else:
                    self.assertEqual(agent.algorithm.env.observation_space.shape, (3,))
                    self.assertEqual(agent.input_features.velocity_scale, 2.0 * math.pi)
                    expected = feature_map(spec, network, states)
                    expected_next = feature_map(spec, network, next_states)
                np.testing.assert_array_equal(
                    buffer.observations[:30, 0], expected.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    buffer.next_observations[:30, 0], expected_next.astype(np.float32)
                )
                # predict maps the states it is given the same way.
                probe = np.array([[math.pi - 1e-3, 0.4], [-math.pi + 1e-3, 0.4], [0.2, -1.0]])
                inputs = probe if name == "ct_sac_torque" else feature_map(spec, network, probe)
                policy = agent.algorithm._policy_act(
                    inputs.astype(np.float32), deterministic=True
                )
                np.testing.assert_array_equal(
                    agent.predict(probe, deterministic=True),
                    np.asarray(policy, dtype=np.float64) * agent.torque_limit,
                )


class TestFingerprints(unittest.TestCase):
    def test_existing_fingerprints_unchanged(self):
        proto = smoke_protocol()
        proto["arms"] = list(P.selectable_arms())
        P.validate_matched_quantities(proto)
        runs = {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        sources = dict(FIXED_ARMS)
        sources.update({name: catalog[name] for name in ("ct_sac_released", "ct_sac_released_timeout")})
        for name, expected in sources.items():
            with self.subTest(arm=name):
                run = runs[name]
                settings = R.run_settings(run)
                self.assertNotIn("ct_sac_inputs", settings["arm"])
                if name in FIXED_ARM_JSON:
                    self.assertEqual(compact_json(settings["arm"]), FIXED_ARM_JSON[name])
                alone = dict(
                    run,
                    protocol=dict(run["protocol"], arms=[copy.deepcopy(expected)]),
                    arm=copy.deepcopy(expected),
                )
                self.assertEqual(R.run_fingerprint(run), R.run_fingerprint(alone))
        for name, source in SOURCES.items():
            with self.subTest(arm=name):
                settings = R.run_settings(runs[name])
                self.assertEqual(settings["arm"]["ct_sac_inputs"], "state_features")
                self.assertNotEqual(
                    R.run_fingerprint(runs[name]), R.run_fingerprint(runs[source])
                )
                without = {k: v for k, v in settings["arm"].items() if k not in ("name", "ct_sac_inputs")}
                source_arm = {
                    k: v for k, v in R.run_settings(runs[source])["arm"].items() if k != "name"
                }
                self.assertEqual(without, source_arm)


# ---------------------------------------------------------------------------------- runs


class TestFeatureRuns(unittest.TestCase):
    def test_tiny_runs_and_report_only(self):
        """60 decisions of each arm: ct_sac_features warms up for its one 50-decision
        episode and trains on decisions 50 to 60; the released arms are still inside their
        10000-decision warm-up."""
        budget = 60
        declared = {arm["name"]: arm for arm in P.baseline_feature_arms()}
        with tempfile.TemporaryDirectory() as tmp:
            proto = feature_protocol(tmp, budget)
            testbed = proto["testbed"]
            results = P.run_protocol(proto)
            self.assertEqual([r["run"]["arm"]["name"] for r in results], list(FEATURE_ARMS))
            for result in results:
                name = result["run"]["arm"]["name"]
                with self.subTest(arm=name):
                    counters = result["counters"]
                    self.assertEqual(counters["interactions"], budget)
                    steps = budget - 49 if name == "ct_sac_features" else 0
                    self.assertEqual(counters["critic_gradient_steps"], steps)
                    self.assertEqual(
                        [c["interactions"] for c in result["evaluations"]], [0, 40, budget]
                    )
                    for checkpoint in result["evaluations"]:
                        for episode in checkpoint["episodes"]:
                            self.assertTrue(math.isfinite(episode["discounted_return"]))
                    folder = Path(tmp) / testbed / name / "seed_0"
                    (loaded,) = load_run_records([str(folder / RECORD_NAME)])
                    self.assertEqual(loaded["run"]["arm"], declared[name])
                    self.assertTrue((folder / "checkpoint.pt").exists())
            stored = json.loads((Path(tmp) / testbed / "protocol.json").read_text())
            self.assertEqual(stored["protocol"]["arms"], list(P.baseline_feature_arms()))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_ph_sac.main(
                    ["--env-id", "pendulum-swingup", "--seeds", "0", "--output", tmp,
                     "--report-only", "--arms", *FEATURE_ARMS]
                )
            report = Path(tmp) / testbed / "report" / f"comparison_{testbed}.md"
            self.assertIn(str(report), output.getvalue().splitlines())
            text = report.read_text()
            for name in FEATURE_ARMS:
                self.assertIn(f"| {name} |", text)
            args = run_ph_sac.parse_args(
                ["--env-id", "pendulum-swingup", "--seeds", "0", "--output", tmp,
                 "--arms", "ct_sac_features"]
            )
            (row,) = run_ph_sac.load_protocols(args)
            selected = run_ph_sac._existing_results(row, args.arms)
            self.assertEqual([r["run"]["arm"]["name"] for r in selected], ["ct_sac_features"])


class TestFeatureRestart(ExactRestartMixin, unittest.TestCase):
    """Exact restarts (--resume) in training: ct_sac_features after its one warm-up
    episode (50 decisions), ct_sac_released_timeout_features with the warm-up shortened to
    30 decisions (RELEASED_CT_SAC_SETTINGS patched)."""

    BUDGET = 120

    def check(self, name, stop, warmup):
        options = R.resume_options(every_interactions=0)
        with mock.patch.dict(P.RELEASED_CT_SAC_SETTINGS, warmup_decisions=30):
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                reference, ref_agent = run_captured(feature_run(name, first, self.BUDGET), options)
                run = feature_run(name, second, self.BUDGET)
                stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
                self.assertIsInstance(stopped, R.ResumableStop)
                folder = P.run_output_directory(run)
                payload = torch.load(folder / R.RESUME_NAME, weights_only=False)
                resumed, res_agent = run_captured(run, options, disturbed=True)
                self.assertFalse((folder / R.RESUME_NAME).exists())
                self.assertTrue((folder / R.COMPLETE_NAME).exists())
                again, rebuilt = run_captured(run, options)
                self.assertIsNone(rebuilt)
                self.assertEqual(again["counters"]["interactions"], self.BUDGET)
        self.assertGreaterEqual(payload["interactions"], stop)
        self.assertEqual(payload["settings"]["arm"]["ct_sac_inputs"], "state_features")
        replay = payload["agent"]["replay"]["arrays"]
        self.assertEqual(tuple(replay["observations"].shape[1:]), (1, 3))
        self.assertEqual(
            reference["counters"]["critic_gradient_steps"], self.BUDGET - warmup + 1
        )
        self.assert_same_run(reference, ref_agent, resumed, res_agent)

    def test_ct_sac_features(self):
        # The warm-up episode ends at decision 50, which trains: steps on decisions 50-120.
        self.check("ct_sac_features", 85, 50)

    def test_ct_sac_released_timeout_features(self):
        self.check("ct_sac_released_timeout_features", 85, 30)


if __name__ == "__main__":
    unittest.main()
