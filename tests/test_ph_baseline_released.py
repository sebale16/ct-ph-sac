"""CT-SAC with its authors' released hyperparameters: the arms ct_sac_released and
ct_sac_released_timeout (evaluations/ph_sac/protocol.baseline_released_arms).

1. Arms: both are ct_sac_torque's spec under their own names plus ct_sac_settings (step
   size 7.3e-4, replay 300000, warm-up 10000 decisions, target rate 0.02; the time limit
   bootstrapped or terminal), outside paper_arms() and all_arms(); --arms selects them,
   --list labels them, a changed spec is refused and a stored protocol validates.
2. Agents: make_training_agent gives them the released values and everything else of
   ct_sac_torque (the same networks at initialization, batch, gamma, temperature,
   entropy target, one gradient step per decision).
3. Existing arms unchanged: the specs and fingerprints that TestFixedArmNames pins
   (tests/test_ph_torque_decisions.py), and ct_sac_torque's agent built with the same
   arguments as before.
4. Warm-up in decisions: exactly warmup_decisions uniform random decisions, counted across
   episodes and learn calls, then training from the decision that reaches the count.
5. Time limit: ct_sac_released stores done = terminated (the critic target bootstraps
   through a truncation), ct_sac_released_timeout done = terminated or truncated (the
   target there is the reward alone).
6. Runs: a tiny run of each arm through run_protocol, then --report-only; an exact
   restart (--resume) of each with the warm-up shortened so that training is reached.
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

from algorithms.ph_baseline import _NO_LEGACY_LOG, TIME_LIMIT_MODES
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_time import period_discount
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.reporting import RECORD_NAME, load_run_records, write_protocol_manifest
from tests.test_ph_baseline import _baseline, _spy_step_dt, _testbed, _training
from tests.test_ph_resume import ExactRestartMixin, run_captured
from tests.test_ph_torque_decisions import FIXED_ARM_JSON, FIXED_ARMS, compact_json

CONFIG = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac" / "smoke_windy.json"
RELEASED = ("ct_sac_released", "ct_sac_released_timeout")
TIME_LIMITS = {"ct_sac_released": "bootstrap", "ct_sac_released_timeout": "terminal"}
# benchmarks/hyperparams/ct_sac.csv, cheetah-run and walker-run rows (algo_learning_rate,
# algo_buffer_size, algo_learning_starts, algo_tau), written out by hand.
RELEASED_VALUES = {
    "learning_rate": 0.00073,
    "replay_capacity": 300000,
    "warmup_decisions": 10000,
    "polyak_rate": 0.02,
}
# The pendulum-swingup row (fixed 20 ms clock, no failure set) cut to 1 s episodes:
# 50 decisions each, so the time limit ends an episode at decisions 50 and 100.
EPISODE_SECONDS = 1.0


def smoke_protocol():
    return run_ph_sac.load_protocol(str(CONFIG))


def released_protocol(output, budget, arms=RELEASED):
    """The pendulum-swingup table row with 1 s episodes, batch 16, evaluations every 40
    decisions on one reset seed, the named arms and training seed 0."""
    raw = copy.deepcopy(table.find_ph_entry("pendulum-swingup", "top").config)
    raw["training"].update(batch_size=16)
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


def released_run(name, output, budget):
    (run,) = P.build_run_matrix(released_protocol(output, budget, (name,)))
    return run


def watch_decisions(baseline):
    """Record ("random" or "policy", decisions before it) for every action the legacy
    loop draws: a uniform draw from the action space or the policy's."""
    algorithm = baseline.algorithm
    drawn = []
    sample = algorithm.env.action_space.sample
    policy = algorithm._policy_act

    def random_action(*args, **kwargs):
        drawn.append(("random", algorithm.num_timesteps))
        return sample(*args, **kwargs)

    def policy_action(obs, deterministic=False):
        drawn.append(("policy", algorithm.num_timesteps))
        return policy(obs, deterministic=deterministic)

    algorithm.env.action_space.sample = random_action
    algorithm._policy_act = policy_action
    return drawn


# ---------------------------------------------------------------------------------- arms


class TestReleasedArms(unittest.TestCase):
    def test_declarations(self):
        arms = {arm["name"]: arm for arm in P.baseline_released_arms()}
        self.assertEqual(tuple(arms), RELEASED)
        self.assertEqual(get_args(P.ReleasedBaselineArmName), RELEASED)
        self.assertEqual(P.RELEASED_CT_SAC_SETTINGS, RELEASED_VALUES)
        baseline = {k: v for k, v in FIXED_ARMS["ct_sac_torque"].items() if k != "name"}
        for name, arm in arms.items():
            with self.subTest(arm=name):
                rest = {k: v for k, v in arm.items() if k not in ("name", "ct_sac_settings")}
                self.assertEqual(rest, baseline)
                settings = arm["ct_sac_settings"]
                self.assertEqual(settings, {**RELEASED_VALUES, "time_limit": TIME_LIMITS[name]})
                self.assertIs(type(settings["replay_capacity"]), int)
                self.assertIs(type(settings["warmup_decisions"]), int)
                self.assertIn(settings["time_limit"], TIME_LIMIT_MODES)
        # Every call builds fresh specs: changing one leaves the declaration alone.
        P.baseline_released_arms()[0]["ct_sac_settings"]["learning_rate"] = 1.0
        self.assertEqual(
            P.baseline_released_arms()[0]["ct_sac_settings"]["learning_rate"], 0.00073
        )

    def test_outside_the_paper_and_ablation_sets(self):
        paper = [arm["name"] for arm in P.paper_arms()]
        self.assertEqual(
            paper,
            ["ct_sac_torque", "implicit_torque", "explicit_rollout_torque",
             "explicit_generator_torque"],
        )
        everything = [arm["name"] for arm in P.all_arms()]
        self.assertEqual(len(everything), 17)
        self.assertEqual(sorted(everything), sorted(get_args(P.ArmName)))
        self.assertFalse(set(RELEASED) & set(everything))
        # Then the implicit CT-SAC arms (protocol.baseline_implicit_arms,
        # tests/test_ct_sac_implicit.py).
        implicit = [arm["name"] for arm in P.baseline_implicit_arms()]
        # Then the feature-input CT-SAC arms (protocol.baseline_feature_arms,
        # tests/test_ph_baseline_features.py).
        features = [arm["name"] for arm in P.baseline_feature_arms()]
        # Then the implicit CT-SAC arms with feature inputs
        # (protocol.baseline_implicit_feature_arms, tests/test_ct_sac_implicit.py).
        implicit_features = [arm["name"] for arm in P.baseline_implicit_feature_arms()]
        self.assertEqual(
            [arm["name"] for arm in P.selectable_arms()],
            everything + list(RELEASED) + implicit + features + implicit_features,
        )
        # A protocol without --arms runs the paper arms only.
        self.assertEqual([arm["name"] for arm in smoke_protocol()["arms"]], paper)

    def test_arms_option_and_list(self):
        path = str(CONFIG)
        args = run_ph_sac.parse_args(
            ["--config", path, "--arms", *reversed(RELEASED), "ct_sac_torque", "--seeds", "1"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        # In selectable_arms() order.
        selected = ["ct_sac_torque", *RELEASED]
        self.assertEqual([arm["name"] for arm in proto["arms"]], selected)
        P.validate_matched_quantities(proto)
        self.assertEqual(
            sorted(run["run_id"] for run in P.build_run_matrix(proto)),
            sorted(f"windy_pendulum__{name}__seed1" for name in selected),
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_ph_sac.parse_args(["--config", path, "--arms", "ct_sac_released_typo"])
        lines = run_ph_sac.list_arms()
        listed = {line.split()[0]: line for line in lines[1:-1]}
        self.assertEqual(list(listed), [arm["name"] for arm in P.selectable_arms()])
        for name in RELEASED:
            with self.subTest(arm=name):
                self.assertEqual(listed[name].split()[1:3], ["torque", "ct_sac"])
                self.assertTrue(listed[name].endswith("released baseline (only with --arms)"))
        self.assertEqual(listed["ct_sac_torque"].split()[-1], "paper")
        self.assertTrue(lines[-1].startswith("example: "))

    def test_changed_specs_are_rejected(self):
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        changed = []
        arm = copy.deepcopy(catalog["ct_sac_released"])
        arm["ct_sac_settings"]["learning_rate"] = 3e-4
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_released_timeout"])
        arm["ct_sac_settings"]["time_limit"] = "bootstrap"
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_released"])
        del arm["ct_sac_settings"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_torque"])
        arm["ct_sac_settings"] = copy.deepcopy(catalog["ct_sac_released"]["ct_sac_settings"])
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_released"])
        arm["name"] = "ct_sac_released_v2"
        changed.append(arm)
        proto = smoke_protocol()
        for index, arm in enumerate(changed):
            with self.subTest(case=index):
                proto["arms"] = [arm]
                with self.assertRaises(ValueError):
                    P.validate_matched_quantities(proto)

    def test_a_stored_protocol_validates(self):
        """protocol.json (reporting.write_protocol_manifest) stores the arms as JSON."""
        proto = smoke_protocol()
        proto["arms"] = list(P.baseline_released_arms())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            write_protocol_manifest(proto, str(path))
            stored = json.loads(path.read_text())["protocol"]["arms"]
        self.assertEqual(stored, list(P.baseline_released_arms()))
        proto["arms"] = stored
        P.validate_matched_quantities(proto)


# -------------------------------------------------------------------------------- agents


class TestReleasedAgents(unittest.TestCase):
    """The agents make_training_agent builds on the smoke protocol (lr 3e-4, replay 20000,
    one warm-up episode, Polyak rate 0.005, batch 64, a tuned temperature)."""

    @classmethod
    def setUpClass(cls):
        proto = smoke_protocol()
        proto["arms"] = [
            arm for arm in P.selectable_arms() if arm["name"] in ("ct_sac_torque", *RELEASED)
        ]
        P.validate_matched_quantities(proto)
        cls.protocol = proto
        cls.runs = {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}
        cls.agents = {}
        for name, run in cls.runs.items():
            env = make_env(
                proto["matched"]["testbed_config"],
                decision_mode="torque",
                seed=run["training_seed"],
            )
            cls.agents[name] = P.make_training_agent(run, env)

    def test_released_values(self):
        explicit = self.protocol["explicit_config"]
        for name in RELEASED:
            with self.subTest(arm=name):
                agent = self.agents[name]
                algorithm = agent.algorithm
                for optimizer in (
                    algorithm.actor_optimizer,
                    algorithm.critic_optimizer,
                    algorithm.alpha_optimizer,
                ):
                    self.assertEqual(
                        [group["lr"] for group in optimizer.param_groups],
                        [0.00073] * len(optimizer.param_groups),
                    )
                self.assertEqual(algorithm.lr_schedule(1.0), 0.00073)
                self.assertEqual(algorithm.lr_schedule(0.0), 0.00073)
                self.assertEqual(algorithm.buffer_size, 300000)
                self.assertEqual(algorithm.replay_buffer.observations.shape[0], 300000)
                self.assertEqual(algorithm.learning_starts, 10000)
                self.assertEqual(algorithm.warmup_decisions, 10000)
                self.assertEqual(algorithm.tau, 0.02)
                self.assertEqual(agent.time_limit, TIME_LIMITS[name])
                self.assertEqual(
                    algorithm.terminal_time_limit, name == "ct_sac_released_timeout"
                )
                # Batch from the config, as ct_sac_torque (256 in the paper's configs).
                self.assertEqual(algorithm.batch_size, explicit["training"]["batch_size"])

    def test_everything_else_is_ct_sac_torques(self):
        reference = self.agents["ct_sac_torque"]
        ref = reference.algorithm
        testbed = self.protocol["matched"]["testbed_config"]
        for name in RELEASED:
            with self.subTest(arm=name):
                agent = self.agents[name]
                algorithm = agent.algorithm
                for key in (
                    "gamma", "beta", "time_rescale", "batch_size", "train_freq",
                    "gradient_steps", "num_expectation_samples", "target_entropy", "alpha",
                ):
                    self.assertEqual(getattr(algorithm, key), getattr(ref, key), key)
                self.assertEqual(
                    algorithm.gamma,
                    period_discount(
                        discount_rate=testbed["discount_rate"],
                        reference_period=testbed["reference_period"],
                    ),
                )
                self.assertEqual((algorithm.train_freq, algorithm.gradient_steps), (1, 1))
                self.assertEqual(algorithm.num_expectation_samples, 1)
                self.assertEqual(algorithm.target_entropy, -1.0)
                self.assertEqual(algorithm.log_alpha.item(), ref.log_alpha.item())
                self.assertAlmostEqual(
                    algorithm.alpha,
                    self.protocol["matched"]["temperature_per_second"]
                    * testbed["reference_period"],
                )
                np.testing.assert_array_equal(agent.torque_limit, reference.torque_limit)
                self.assertEqual(agent.policy_diagnostics()["log_std_mean"], -3.0)
                # The same networks at initialization: the run's seed, the same architecture.
                for part in ("actor", "q_nets", "q_target_nets"):
                    mine = getattr(algorithm.model, part).state_dict()
                    theirs = getattr(ref.model, part).state_dict()
                    self.assertEqual(list(mine), list(theirs), part)
                    for key in mine:
                        self.assertTrue(torch.equal(mine[key], theirs[key]), f"{part}.{key}")

    def test_ct_sac_torque_keeps_the_explicit_configs_values(self):
        explicit = self.protocol["explicit_config"]
        agent = self.agents["ct_sac_torque"]
        algorithm = agent.algorithm
        self.assertEqual(algorithm.lr_schedule(1.0), explicit["training"]["learning_rate"])
        self.assertEqual(algorithm.buffer_size, explicit["training"]["replay_capacity"])
        self.assertEqual(algorithm.tau, explicit["polyak_rate"])
        self.assertEqual(algorithm.warmup_episodes, explicit["training"]["warmup_episodes"])
        self.assertEqual(algorithm.warmup_episodes, 1)
        self.assertIsNone(algorithm.warmup_decisions)
        # Held until the first episode ends.
        self.assertEqual(algorithm.learning_starts, _NO_LEGACY_LOG)
        self.assertFalse(algorithm.terminal_time_limit)
        self.assertEqual(agent.time_limit, "bootstrap")

    def test_constructor_arguments(self):
        explicit = self.protocol["explicit_config"]
        for name, run in self.runs.items():
            with self.subTest(arm=name), mock.patch(
                "algorithms.ph_baseline.TorqueCTSACBaseline"
            ) as built:
                P.make_training_agent(run, None)
                (env, training, temperature), kwargs = built.call_args
                self.assertIsNone(env)
                self.assertEqual(temperature, explicit["temperature"])
                expected = dict(explicit["training"], seed=run["training_seed"])
                if name == "ct_sac_torque":
                    # Exactly the arguments of the code before the released arms.
                    self.assertEqual(training, expected)
                    self.assertEqual(
                        kwargs,
                        {"polyak_rate": explicit["polyak_rate"], "expectation_samples": 1},
                    )
                else:
                    expected.update(learning_rate=0.00073, replay_capacity=300000)
                    self.assertEqual(training, expected)
                    self.assertEqual(
                        kwargs,
                        {
                            "polyak_rate": 0.02,
                            "expectation_samples": 1,
                            "warmup_decisions": 10000,
                            "time_limit": TIME_LIMITS[name],
                        },
                    )
        # The protocol's own config is untouched.
        self.assertEqual(explicit["training"]["learning_rate"], 0.0003)
        self.assertEqual(explicit["training"]["replay_capacity"], 20000)
        self.assertEqual(explicit["polyak_rate"], 0.005)


class TestTrainingKeyWarmupDecisions(unittest.TestCase):
    """The training key warmup_decisions (common.ph_config.warmup_decisions_setting) on the
    smoke protocol: every CT-SAC arm without ct_sac_settings takes it as warmup_decisions,
    the arms with them keep their own 10000, and nothing else changes."""

    WARMUP = 123

    def protocol(self, arms):
        proto = smoke_protocol()
        for name in ("explicit_config", "implicit_config"):
            proto[name]["training"]["warmup_decisions"] = self.WARMUP
        proto["arms"] = list(arms)
        P.validate_matched_quantities(proto)
        return proto

    def test_constructor_arguments(self):
        arms = [arm for arm in P.selectable_arms() if arm["algorithm"] == "ct_sac"]
        self.assertTrue(any("ct_sac_settings" in arm for arm in arms))
        self.assertTrue(any("ct_sac_implicit_step" in arm for arm in arms))
        plain = smoke_protocol()
        plain["arms"] = list(arms)
        plain_runs = {run["arm"]["name"]: run for run in P.build_run_matrix(plain)}
        for run in P.build_run_matrix(self.protocol(arms)):
            arm = run["arm"]
            target = (
                "algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline"
                if "ct_sac_implicit_step" in arm
                else "algorithms.ph_baseline.TorqueCTSACBaseline"
            )
            with self.subTest(arm=arm["name"]):
                with mock.patch(target) as built:
                    P.make_training_agent(run, None)
                _, kwargs = built.call_args
                with mock.patch(target) as built:
                    P.make_training_agent(plain_runs[arm["name"]], None)
                _, plain_kwargs = built.call_args
                settings = arm.get("ct_sac_settings")
                if settings is None:
                    self.assertEqual(kwargs, {**plain_kwargs, "warmup_decisions": self.WARMUP})
                else:
                    self.assertEqual(settings["warmup_decisions"], 10000)
                    self.assertEqual(kwargs, plain_kwargs)

    def test_ct_sac_torque_warms_up_in_decisions(self):
        (run,) = P.build_run_matrix(
            self.protocol(arm for arm in P.selectable_arms() if arm["name"] == "ct_sac_torque")
        )
        env = make_env(
            run["protocol"]["matched"]["testbed_config"],
            decision_mode="torque",
            seed=run["training_seed"],
        )
        algorithm = P.make_training_agent(run, env).algorithm
        self.assertEqual(algorithm.warmup_decisions, self.WARMUP)
        self.assertEqual(algorithm.learning_starts, self.WARMUP)


# ------------------------------------------------------------------ existing arms unchanged


class TestExistingArmsUnchanged(unittest.TestCase):
    def test_fixed_specs_and_fingerprints(self):
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        proto = smoke_protocol()
        proto["arms"] = list(P.selectable_arms())
        P.validate_matched_quantities(proto)
        runs = {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}
        for name, expected in FIXED_ARMS.items():
            with self.subTest(arm=name):
                spec = catalog[name]
                self.assertEqual(dict(spec), expected)
                self.assertNotIn("ct_sac_settings", spec)
                self.assertEqual(compact_json(R._canonical(spec)), FIXED_ARM_JSON[name])
                run = runs[name]
                settings = R.run_settings(run)
                self.assertEqual(compact_json(settings["arm"]), FIXED_ARM_JSON[name])
                # The released arms in the protocol's arm list leave the fingerprint alone.
                alone = dict(
                    run,
                    protocol=dict(run["protocol"], arms=[copy.deepcopy(expected)]),
                    arm=copy.deepcopy(expected),
                )
                self.assertEqual(settings, R.run_settings(alone))
                self.assertEqual(R.run_fingerprint(run), R.run_fingerprint(alone))
        # The released arms have fingerprints of their own, holding their settings.
        prints = {name: R.run_fingerprint(runs[name]) for name in ("ct_sac_torque", *RELEASED)}
        self.assertEqual(len(set(prints.values())), 3)
        baseline = {
            k: v for k, v in json.loads(FIXED_ARM_JSON["ct_sac_torque"]).items() if k != "name"
        }
        for name in RELEASED:
            with self.subTest(arm=name):
                arm = R.run_settings(runs[name])["arm"]
                self.assertEqual(
                    {k: v for k, v in arm.items() if k not in ("name", "ct_sac_settings")},
                    baseline,
                )
                self.assertEqual(
                    arm["ct_sac_settings"],
                    {
                        "learning_rate": {"float": (0.00073).hex()},
                        "polyak_rate": {"float": (0.02).hex()},
                        "replay_capacity": 300000,
                        "time_limit": TIME_LIMITS[name],
                        "warmup_decisions": 10000,
                    },
                )


# ------------------------------------------------------------------------------- warm-up


class TestDecisionWarmUp(unittest.TestCase):
    """The windy pendulum with 50 ms episodes (about five decisions each)."""

    def test_exactly_the_warm_up_decisions_are_random(self):
        warmup, budget = 40, 100
        baseline = _baseline(
            _testbed(episode_duration=0.05),
            _training(batch_size=8, warmup_episodes=2),
            warmup_decisions=warmup,
        )
        transitions = _spy_step_dt(baseline.env)
        drawn = watch_decisions(baseline)
        baseline.learn(budget)
        self.assertEqual(
            drawn,
            [("random", n) for n in range(warmup)]
            + [("policy", n) for n in range(warmup, budget)],
        )
        # More episodes than warmup_episodes ended inside the warm-up: that count is ignored.
        ends = sum(bool(t["terminated"] or t["truncated"]) for t in transitions[:warmup])
        self.assertGreater(ends, 2)
        self.assertEqual(baseline.algorithm.learning_starts, warmup)
        # One gradient step on every decision from the one that brings the count to 40.
        counters = baseline.counters()
        self.assertEqual(counters["critic_gradient_steps"], budget - warmup + 1)
        self.assertEqual(counters["actor_gradient_steps"], budget - warmup + 1)

    def test_the_count_runs_across_learn_calls(self):
        warmup = 40
        baseline = _baseline(
            _testbed(episode_duration=0.05), _training(batch_size=8), warmup_decisions=warmup
        )
        drawn = watch_decisions(baseline)
        baseline.learn(25)
        self.assertEqual(baseline.counters()["critic_gradient_steps"], 0)
        baseline.learn(70)
        self.assertEqual(baseline.counters()["interactions"], 70)
        self.assertEqual([n for _, n in drawn], list(range(70)))
        self.assertEqual([kind for kind, _ in drawn], ["random"] * warmup + ["policy"] * 30)
        self.assertEqual(baseline.counters()["critic_gradient_steps"], 70 - warmup + 1)

    def test_zero_warm_up_acts_and_trains_at_once(self):
        baseline = _baseline(
            _testbed(episode_duration=0.05),
            _training(batch_size=8, warmup_episodes=3),
            warmup_decisions=0,
        )
        drawn = watch_decisions(baseline)
        baseline.learn(30)
        self.assertEqual({kind for kind, _ in drawn}, {"policy"})
        # Training once the replay holds a batch: decisions 8 to 30.
        self.assertEqual(baseline.counters()["critic_gradient_steps"], 30 - 8 + 1)
        self.assertEqual(baseline.algorithm.learning_starts, 0)

    def test_defaults_are_the_episode_warm_up_and_bootstrap(self):
        baseline = _baseline(_testbed(), _training(warmup_episodes=2))
        self.assertIsNone(baseline.algorithm.warmup_decisions)
        self.assertEqual(baseline.algorithm.learning_starts, _NO_LEGACY_LOG)
        self.assertFalse(baseline.algorithm.terminal_time_limit)
        self.assertEqual(baseline.time_limit, "bootstrap")

    def test_invalid_settings_are_rejected(self):
        for value in (-1, 2.5, True):
            with self.subTest(warmup_decisions=value):
                with self.assertRaisesRegex(ValueError, "warmup_decisions"):
                    _baseline(warmup_decisions=value)
        with self.assertRaisesRegex(ValueError, "time_limit"):
            _baseline(time_limit="truncate")


# ---------------------------------------------------------------------------- time limit


class TestTimeLimit(unittest.TestCase):
    def collect(self, time_limit, interactions=60):
        """Random actions only (an unreachable warm-up) on 50 ms episodes."""
        baseline = _baseline(
            _testbed(episode_duration=0.05),
            _training(),
            warmup_decisions=10**6,
            time_limit=time_limit,
        )
        transitions = _spy_step_dt(baseline.env)
        baseline.learn(interactions)
        return baseline, transitions

    def test_stored_done_flags(self):
        stored = {}
        for time_limit in TIME_LIMIT_MODES:
            with self.subTest(time_limit=time_limit):
                baseline, transitions = self.collect(time_limit)
                buffer = baseline.algorithm.replay_buffer
                self.assertEqual(buffer.size(), 60)
                self.assertEqual(baseline.counters()["critic_gradient_steps"], 0)
                truncated = [i for i, t in enumerate(transitions) if t["truncated"]]
                self.assertGreater(len(truncated), 2)
                for i, transition in enumerate(transitions):
                    expected = transition["terminated"] or (
                        time_limit == "terminal" and transition["truncated"]
                    )
                    self.assertEqual(buffer.dones[i, 0], float(expected), i)
                stored[time_limit] = (buffer, transitions)
        # The same rows otherwise (the same seeds and random actions): only the flag differs.
        (bootstrap, first), (terminal, _) = stored["bootstrap"], stored["terminal"]
        for name in ("observations", "next_observations", "actions", "rewards", "dt"):
            np.testing.assert_array_equal(
                getattr(bootstrap, name)[:60], getattr(terminal, name)[:60], err_msg=name
            )
        differ = np.flatnonzero(bootstrap.dones[:60, 0] != terminal.dones[:60, 0]).tolist()
        self.assertEqual(differ, [i for i, t in enumerate(first) if t["truncated"]])

    def target(self, time_limit, transition, value, next_value):
        """The critic target of one stored row, with the target critics replaced by value
        at the row's state and next_value elsewhere (tests/test_ph_baseline.py
        TestCriticTarget); also the stored done flag."""
        baseline = _baseline(training=_training(batch_size=1), time_limit=time_limit)
        algorithm = baseline.algorithm
        adapted = baseline.adapt_transition(transition)
        algorithm._store_transition(
            obs=np.asarray(transition["state"], dtype=np.float32),
            action=np.asarray(transition["decision"], dtype=np.float32),
            reward=adapted["reward"],
            # The legacy loop's own flag; the adapter stores its convention's instead.
            done=adapted["terminated"] or adapted["truncated"],
            next_obs=np.asarray(transition["next_state"], dtype=np.float32),
            t=transition["time"],
            next_t=transition["next_time"],
            infos={"adapted": adapted},
        )
        start = torch.as_tensor(transition["state"], dtype=torch.float32)

        def target_min_q(obs, _actions):
            at_start = (obs == start).all(dim=-1, keepdim=True)
            return torch.where(at_start, torch.tensor(value), torch.tensor(next_value))

        algorithm.model.target_min_q = target_min_q
        real_loss = torch.nn.functional.mse_loss
        targets = []

        def capture(prediction, target):
            targets.append(target.detach().clone())
            return real_loss(prediction, target)

        with mock.patch("algorithms.ct_sac.F.mse_loss", side_effect=capture):
            algorithm.train(gradient_steps=1, batch_size=1)
        return targets[0].item(), float(algorithm.replay_buffer.dones[0, 0])

    def test_critic_target_at_a_truncation(self):
        lam, T, delta, r = 0.1, 0.01, 0.004, -0.0123
        value, next_value = -3.0, -3.5
        transition = {
            "state": np.array([0.1, 0.2]),
            "decision": np.array([1.0]),
            "duration": delta,
            "reward": r,
            "next_state": np.array([0.15, 0.3]),
            "terminated": False,
            "truncated": True,
            "time": 7.0,
            "next_time": 7.0 + delta,
        }
        u = delta / T
        bootstrapped = r / u + value + (math.exp(-lam * delta) * next_value - value) / u
        target, done = self.target("bootstrap", transition, value, next_value)
        self.assertEqual(done, 0.0)
        self.assertAlmostEqual(target, bootstrapped, places=3)
        target, done = self.target("terminal", transition, value, next_value)
        self.assertEqual(done, 1.0)
        # CT-SAC's own convention: the reward alone (T times the rate), no continuation.
        self.assertAlmostEqual(target, r / u, places=5)
        # An interior row bootstraps under either convention.
        interior = dict(transition, truncated=False)
        for time_limit in TIME_LIMIT_MODES:
            with self.subTest(time_limit=time_limit):
                target, done = self.target(time_limit, interior, value, next_value)
                self.assertEqual(done, 0.0)
                self.assertAlmostEqual(target, bootstrapped, places=3)


# ---------------------------------------------------------------------------------- runs


class TestReleasedRuns(unittest.TestCase):
    def test_tiny_runs_and_report_only(self):
        """60 decisions of each arm, all inside the 10000-decision warm-up, then the
        reports rebuilt from their records."""
        budget = 60
        released = {arm["name"]: arm for arm in P.baseline_released_arms()}
        with tempfile.TemporaryDirectory() as tmp:
            proto = released_protocol(tmp, budget)
            testbed = proto["testbed"]
            results = P.run_protocol(proto)
            self.assertEqual([r["run"]["arm"]["name"] for r in results], list(RELEASED))
            for result in results:
                name = result["run"]["arm"]["name"]
                with self.subTest(arm=name):
                    counters = result["counters"]
                    self.assertEqual(counters["interactions"], budget)
                    self.assertEqual(counters["critic_gradient_steps"], 0)
                    self.assertEqual(
                        [c["interactions"] for c in result["evaluations"]], [0, 40, budget]
                    )
                    for checkpoint in result["evaluations"]:
                        for episode in checkpoint["episodes"]:
                            self.assertTrue(math.isfinite(episode["discounted_return"]))
                    self.assertEqual(result["decision_mode"], "torque")
                    folder = Path(tmp) / testbed / name / "seed_0"
                    (loaded,) = load_run_records([str(folder / RECORD_NAME)])
                    self.assertEqual(loaded["run"]["arm"], released[name])
                    sidecar = json.loads((folder / "checkpoint.pt.counters.json").read_text())
                    self.assertEqual(sidecar["learning_starts"], 10000)
                    self.assertEqual(sidecar["interactions"], budget)
            stored = json.loads((Path(tmp) / testbed / "protocol.json").read_text())
            self.assertEqual(stored["protocol"]["arms"], list(P.baseline_released_arms()))

            # --report-only finds both arms' records.
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_ph_sac.main(
                    ["--env-id", "pendulum-swingup", "--seeds", "0", "--output", tmp,
                     "--report-only", "--arms", *RELEASED]
                )
            report = Path(tmp) / testbed / "report" / f"comparison_{testbed}.md"
            self.assertIn(str(report), output.getvalue().splitlines())
            text = report.read_text()
            for name in RELEASED:
                self.assertIn(f"| {name} |", text)
            # One named arm, or every arm found when --arms names none.
            args = run_ph_sac.parse_args(
                ["--env-id", "pendulum-swingup", "--seeds", "0", "--output", tmp,
                 "--arms", "ct_sac_released_timeout"]
            )
            (row,) = run_ph_sac.load_protocols(args)
            selected = run_ph_sac._existing_results(row, args.arms)
            self.assertEqual(
                [r["run"]["arm"]["name"] for r in selected], ["ct_sac_released_timeout"]
            )
            everything = run_ph_sac._existing_results(row)
            self.assertEqual(sorted(r["run"]["arm"]["name"] for r in everything), list(RELEASED))


class TestReleasedRestart(ExactRestartMixin, unittest.TestCase):
    """Exact restarts (--resume) with the warm-up shortened to WARMUP decisions
    (RELEASED_CT_SAC_SETTINGS patched), so that a stop at 85 falls in training."""

    WARMUP = 30
    BUDGET = 120

    def check(self, name, stop):
        options = R.resume_options(every_interactions=0)
        with mock.patch.dict(P.RELEASED_CT_SAC_SETTINGS, warmup_decisions=self.WARMUP):
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                reference, ref_agent = run_captured(
                    released_run(name, first, self.BUDGET), options
                )
                run = released_run(name, second, self.BUDGET)
                stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
                self.assertIsInstance(stopped, R.ResumableStop)
                folder = P.run_output_directory(run)
                payload = torch.load(folder / R.RESUME_NAME, weights_only=False)
                resumed, res_agent = run_captured(run, options, disturbed=True)
                self.assertFalse((folder / R.RESUME_NAME).exists())
                self.assertTrue((folder / R.COMPLETE_NAME).exists())
                # Submitted again, the finished run returns its stored record.
                again, rebuilt = run_captured(run, options)
                self.assertIsNone(rebuilt)
                self.assertEqual(again["counters"]["interactions"], self.BUDGET)
        self.assertGreaterEqual(payload["interactions"], stop)
        self.assertEqual(payload["agent"]["algorithm"]["learning_starts"], self.WARMUP)
        settings = payload["settings"]["arm"]["ct_sac_settings"]
        self.assertEqual(settings["warmup_decisions"], self.WARMUP)
        self.assertEqual(settings["time_limit"], TIME_LIMITS[name])
        self.assertEqual(reference["counters"]["interactions"], self.BUDGET)
        self.assertEqual(
            reference["counters"]["critic_gradient_steps"], self.BUDGET - self.WARMUP + 1
        )
        self.assert_same_run(reference, ref_agent, resumed, res_agent)
        # The time limit ends the 50-decision episodes at rows 49 and 99; the task has no
        # failure set, so only the timeout arm stores a done.
        dones = ref_agent.algorithm.replay_buffer.dones[: self.BUDGET, 0]
        expected = [49, 99] if TIME_LIMITS[name] == "terminal" else []
        self.assertEqual(np.flatnonzero(dones).tolist(), expected)
        return reference, payload

    def test_ct_sac_released(self):
        self.check("ct_sac_released", 85)

    def test_ct_sac_released_timeout(self):
        self.check("ct_sac_released_timeout", 85)

    def test_stop_inside_the_warm_up(self):
        _, payload = self.check("ct_sac_released_timeout", 20)
        self.assertLess(payload["interactions"], self.WARMUP)


if __name__ == "__main__":
    unittest.main()
