"""CT-SAC with the implicit value step: the arms ct_sac_implicit, ct_sac_implicit_schedule
and ct_sac_implicit_released (evaluations/ph_sac/protocol.baseline_implicit_arms,
algorithms/ct_sac_implicit.py).

1. Arms and defaults: ct_sac_torque's spec under their own names plus ct_sac_implicit_step
   (and ct_sac_released's ct_sac_settings for the released one), outside paper_arms() and
   all_arms(); --arms selects them, --list labels them, a changed spec is refused. tau, K, N
   (1 s, 400, 4) and the outer schedule's 40 actor steps and 40 rows per temperature step
   are the explicit config's optional ct_sac_implicit section; make_training_agent passes
   them with everything else of ct_sac_torque.
2. Existing arms unchanged: no fixed arm's spec or resolved settings carries anything new,
   and a protocol listing the new arms leaves every fixed arm's fingerprint alone; the
   implicit arms' fingerprints hold the resolved section.
3. The residual: c = 0 gives Q - Qbar; the bracket and residual by hand; the all-pairs loss
   is the mean over i != i' of R_i R_i'; the gradient reaches the next-state value (and not
   on a row that failed); the draws come from the frozen actor.
4. Solves: the anchors (target critics) and the frozen actor are hard copies made exactly
   every K training rows; under the outer schedule the actor moves only at the copies; one
   solver record per copy with finite diagnostics; the counters.
5. Checkpoints: resume_state round trip of the frozen copies and the counters (and the same
   continuation), and an exact restart of each arm through run_seed (--resume).
6. The parent CT-SAC update (N_e = 1) is unchanged: the hook builds _PHCTSAC with no extra
   argument, and one update's loss matches a hand computation.
7. Feature inputs: ct_sac_implicit_features, ct_sac_implicit_schedule_features and
   ct_sac_implicit_released_features (protocol.baseline_implicit_feature_arms) are their
   implicit arm's spec plus ct_sac_inputs "state_features"; their agents get the explicit
   config's network as state_features (the released one also ct_sac_implicit_released's
   released settings); every network (critics, anchors, actor, frozen actor) takes the
   feature dimension; the replay holds the features of the states, mapped once; one
   residual step runs on feature rows (a synthetic batch, and the training loop through
   one copy; for the released one a synthetic batch on the agent make_training_agent
   builds); listing the released one leaves every other arm's fingerprint alone.

selectable_arms() and --list (also pinned by tests/test_ph_baseline_released.py and
tests/test_ph_baseline_features.py): all_arms(), then the released, implicit, feature and
implicit-feature CT-SAC arms, in that order.
"""

from __future__ import annotations

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

from algorithms.ct_sac_implicit import (
    ImplicitTorqueCTSACBaseline,
    _ImplicitPHCTSAC,
    implicit_bracket,
    implicit_loss,
    resolvent_residuals,
)
from algorithms.ph_baseline import TorqueCTSACBaseline, _PHCTSAC
from algorithms.ph_resolvent import all_pairs_loss
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.buffers import ReplayBatch
from common.ph_config import (
    CT_SAC_ACTOR_SCHEDULES,
    CT_SAC_IMPLICIT_DEFAULTS,
    CT_SAC_IMPLICIT_KEY,
    ct_sac_implicit_setting,
)
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.diagnostics import SolverMetrics, TrainingCounters
from tests.test_ph_baseline import _baseline, _spy_step_dt, _testbed, _training
from tests.test_ph_resume import ExactRestartMixin, differences, run_captured
from tests.test_ph_torque_decisions import FIXED_ARM_JSON, FIXED_ARMS, compact_json

CONFIG = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac" / "smoke_windy.json"
IMPLICIT = ("ct_sac_implicit", "ct_sac_implicit_schedule", "ct_sac_implicit_released")
SCHEDULES = {
    "ct_sac_implicit": "per_row",
    "ct_sac_implicit_schedule": "outer",
    "ct_sac_implicit_released": "per_row",
}
# Each implicit feature arm and the implicit arm whose spec it extends.
FEATURE_SOURCES = {
    "ct_sac_implicit_features": "ct_sac_implicit",
    "ct_sac_implicit_schedule_features": "ct_sac_implicit_schedule",
    "ct_sac_implicit_released_features": "ct_sac_implicit_released",
}
IMPLICIT_FEATURES = tuple(FEATURE_SOURCES)
# The other feature arms (tests/test_ph_baseline_features.py), in declaration order.
FEATURES = ("ct_sac_features", "ct_sac_released_features", "ct_sac_released_timeout_features")
# The windy pendulum's state (angle, velocity) as network input: sin, cos, velocity / scale.
FEATURE_DIM = 3
VELOCITY_SCALE = 2.0 * math.pi
# Small settings for the direct agents below: K = 5 rows per solve, N = 3 draws, and for the
# outer schedule 2 actor steps per copy and one temperature step per 2 rows.
SMALL = {
    "resolvent_step": 1.0,
    "inner_steps": 5,
    "policy_draws": 3,
    "actor_steps": 2,
    "temperature_every_rows": 2.0,
}
# The windy pendulum of tests/test_ph_baseline.py: T = 10 ms, lambda = 0.1 /s.
REFERENCE_PERIOD = 0.01
DISCOUNT_RATE = 0.1


def smoke_protocol():
    return run_ph_sac.load_protocol(str(CONFIG))


def _implicit(testbed=None, training=None, *, schedule="per_row", alpha_c=0.1,
              learn_temperature=True, implicit=None, **kwargs):
    """An ImplicitTorqueCTSACBaseline on the windy pendulum with SMALL settings, 32-32
    networks and a tuned temperature (alpha_c 0.1) unless told otherwise."""
    testbed = testbed or _testbed()
    env = make_env(testbed, decision_mode="torque", seed=11)
    temperature = {
        "initial_alpha_c": alpha_c,
        "learn_temperature": learn_temperature,
        "target_entropy": -3.0,
    }
    kwargs.setdefault("polyak_rate", 0.005)
    kwargs.setdefault("expectation_samples", 1)
    kwargs.setdefault("net_arch", (32, 32))
    return ImplicitTorqueCTSACBaseline(
        env,
        training or _training(),
        temperature,
        implicit=dict(SMALL, **(implicit or {})),
        actor_schedule=schedule,
        **kwargs,
    )


def synthetic_batch():
    """Four rows of the windy pendulum's shapes (2 states, 1 action); row 1 failed."""
    generator = torch.Generator().manual_seed(5)
    obs = torch.randn(4, 2, generator=generator)
    next_obs = obs + 0.05 * torch.randn(4, 2, generator=generator)
    actions = torch.rand(4, 1, generator=generator) * 2.0 - 1.0
    rewards = 0.01 * torch.randn(4, 1, generator=generator)
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    dt = torch.tensor([[0.002], [0.01], [0.018], [0.004]])
    return ReplayBatch(
        observations=obs,
        actions=actions,
        next_observations=next_obs,
        rewards=rewards,
        dones=dones,
        t=torch.zeros(4, 1),
        next_t=dt.clone(),
        dt=dt,
    )


def states(module):
    return {key: tensor.detach().clone() for key, tensor in module.state_dict().items()}


def same(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def implicit_protocol(output, budget, arms):
    """The pendulum-swingup table row with 1 s episodes (50 decisions of 20 ms), batch 16,
    one warm-up episode, K = 8, N = 2, 2 actor steps per copy and one temperature step per
    4 rows (outer schedule), evaluations every 40 decisions on one reset seed."""
    raw = copy.deepcopy(table.find_ph_entry("pendulum-swingup", "top").config)
    raw["training"].update(batch_size=16, replay_capacity=2000, warmup_episodes=1)
    raw["explicit"][CT_SAC_IMPLICIT_KEY] = {
        "inner_steps": 8,
        "policy_draws": 2,
        "actor_steps": 2,
        "temperature_every_rows": 4,
    }
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
        testbed["episode_duration"] = 1.0
    proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in arms]
    P.validate_matched_quantities(proto)
    return proto


# ---------------------------------------------------------------------------------- arms


class TestArms(unittest.TestCase):
    def test_declarations(self):
        arms = {arm["name"]: arm for arm in P.baseline_implicit_arms()}
        self.assertEqual(tuple(arms), IMPLICIT)
        self.assertEqual(get_args(P.ImplicitBaselineArmName), IMPLICIT)
        baseline = {k: v for k, v in FIXED_ARMS["ct_sac_torque"].items() if k != "name"}
        for name, arm in arms.items():
            with self.subTest(arm=name):
                rest = {
                    k: v
                    for k, v in arm.items()
                    if k not in ("name", "ct_sac_implicit_step", "ct_sac_settings")
                }
                self.assertEqual(rest, baseline)
                self.assertEqual(arm["ct_sac_implicit_step"], {"actor_schedule": SCHEDULES[name]})
                self.assertIn(arm["ct_sac_implicit_step"]["actor_schedule"], CT_SAC_ACTOR_SCHEDULES)
        self.assertNotIn("ct_sac_settings", arms["ct_sac_implicit"])
        self.assertNotIn("ct_sac_settings", arms["ct_sac_implicit_schedule"])
        released = {arm["name"]: arm for arm in P.baseline_released_arms()}
        self.assertEqual(
            arms["ct_sac_implicit_released"]["ct_sac_settings"],
            released["ct_sac_released"]["ct_sac_settings"],
        )
        self.assertEqual(
            arms["ct_sac_implicit_released"]["ct_sac_settings"],
            {**P.RELEASED_CT_SAC_SETTINGS, "time_limit": "bootstrap"},
        )
        # Every call builds fresh specs: changing one leaves the declaration alone.
        P.baseline_implicit_arms()[0]["ct_sac_implicit_step"]["actor_schedule"] = "outer"
        self.assertEqual(
            P.baseline_implicit_arms()[0]["ct_sac_implicit_step"]["actor_schedule"], "per_row"
        )

    def test_selected_only_by_name(self):
        paper = [arm["name"] for arm in P.paper_arms()]
        everything = [arm["name"] for arm in P.all_arms()]
        self.assertFalse(set(IMPLICIT) & set(everything))
        self.assertFalse(set(IMPLICIT) & set(get_args(P.ArmName)))
        # The exact selectable_arms() order (as tests/test_ph_baseline_released.py pins it):
        # all_arms(), the released, implicit, feature and implicit-feature CT-SAC arms.
        released = ("ct_sac_released", "ct_sac_released_timeout")
        self.assertEqual(
            [arm["name"] for arm in P.selectable_arms()],
            everything + list(released) + list(IMPLICIT) + list(FEATURES)
            + list(IMPLICIT_FEATURES),
        )
        self.assertEqual([arm["name"] for arm in smoke_protocol()["arms"]], paper)
        path = str(CONFIG)
        args = run_ph_sac.parse_args(
            ["--config", path, "--arms", *reversed(IMPLICIT), "--seeds", "2"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual([arm["name"] for arm in proto["arms"]], list(IMPLICIT))
        P.validate_matched_quantities(proto)
        self.assertEqual(
            sorted(run["run_id"] for run in P.build_run_matrix(proto)),
            sorted(f"windy_pendulum__{name}__seed2" for name in IMPLICIT),
        )
        lines = run_ph_sac.list_arms()
        listed = {line.split()[0]: line for line in lines[1:-1]}
        self.assertEqual(list(listed), [arm["name"] for arm in P.selectable_arms()])
        self.assertEqual(len(lines), len(P.selectable_arms()) + 2)
        labels = {
            **{name: "implicit baseline" for name in IMPLICIT},
            **{name: "feature baseline" for name in FEATURES},
            **{name: "implicit feature baseline" for name in IMPLICIT_FEATURES},
        }
        for name, label in labels.items():
            with self.subTest(arm=name):
                self.assertEqual(listed[name].split()[1:3], ["torque", "ct_sac"])
                # The whole set column, so "implicit baseline" cannot match the tail of
                # "implicit feature baseline".
                self.assertEqual(
                    listed[name].split(None, 3)[3], f"{label} (only with --arms)"
                )

    def test_changed_specs_are_rejected(self):
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        changed = []
        arm = copy.deepcopy(catalog["ct_sac_implicit"])
        arm["ct_sac_implicit_step"]["actor_schedule"] = "outer"
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit"])
        del arm["ct_sac_implicit_step"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_torque"])
        arm["ct_sac_implicit_step"] = {"actor_schedule": "per_row"}
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit_released"])
        del arm["ct_sac_settings"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit_features"])
        del arm["ct_sac_inputs"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit"])
        arm["ct_sac_inputs"] = "state_features"
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit_schedule_features"])
        arm["ct_sac_implicit_step"]["actor_schedule"] = "per_row"
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit_released_features"])
        del arm["ct_sac_settings"]
        changed.append(arm)
        arm = copy.deepcopy(catalog["ct_sac_implicit_released"])
        arm["ct_sac_inputs"] = "state_features"
        changed.append(arm)
        proto = smoke_protocol()
        for index, arm in enumerate(changed):
            with self.subTest(case=index):
                proto["arms"] = [arm]
                with self.assertRaises(ValueError):
                    P.validate_matched_quantities(proto)


class TestSettings(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(CT_SAC_IMPLICIT_KEY, "ct_sac_implicit")
        self.assertEqual(
            dict(CT_SAC_IMPLICIT_DEFAULTS),
            {
                "resolvent_step": 1.0,
                "inner_steps": 400,
                "policy_draws": 4,
                "actor_steps": 40,
                "temperature_every_rows": 40,
            },
        )
        resolved = ct_sac_implicit_setting({})
        self.assertEqual(
            resolved,
            {
                "resolvent_step": 1.0,
                "inner_steps": 400,
                "policy_draws": 4,
                "actor_steps": 40,
                "temperature_every_rows": 40.0,
            },
        )
        for key in ("inner_steps", "policy_draws", "actor_steps"):
            self.assertIs(type(resolved[key]), int, key)
        for key in ("resolvent_step", "temperature_every_rows"):
            self.assertIs(type(resolved[key]), float, key)
        # The configs carry no section: the defaults.
        explicit = smoke_protocol()["explicit_config"]
        self.assertNotIn(CT_SAC_IMPLICIT_KEY, explicit)
        self.assertEqual(ct_sac_implicit_setting(explicit), resolved)
        self.assertEqual(ct_sac_implicit_setting({CT_SAC_IMPLICIT_KEY: None}), resolved)
        # A partial section overrides only its keys.
        partial = ct_sac_implicit_setting({CT_SAC_IMPLICIT_KEY: {"inner_steps": 8}})
        self.assertEqual(partial, dict(resolved, inner_steps=8))
        # The defaults are implicit CT-PH-SAC's values in the table's rows.
        row = table.find_ph_entry("pendulum-swingup", "top").config["implicit"]
        self.assertEqual(row["solver"]["resolvent_step"], resolved["resolvent_step"])
        self.assertEqual(row["solver"]["inner_steps"], resolved["inner_steps"])
        self.assertEqual(row["solver"]["policy_draws"], resolved["policy_draws"])
        self.assertEqual(row["rate"]["actor_steps"], resolved["actor_steps"])
        self.assertEqual(row["temperature_every_rows"], resolved["temperature_every_rows"])

    def test_invalid_settings_are_rejected(self):
        for bad in (
            {"inner_steps": 0},
            {"inner_steps": 2.5},
            {"policy_draws": 1},
            {"policy_draws": True},
            {"resolvent_step": 0.0},
            {"resolvent_step": -1.0},
            {"resolvent_step": float("nan")},
            {"actor_steps": -1},
            {"temperature_every_rows": 0},
            {"inner_step": 8},
        ):
            with self.subTest(section=bad):
                with self.assertRaises(ValueError):
                    ct_sac_implicit_setting({CT_SAC_IMPLICIT_KEY: bad})
        with self.assertRaises(ValueError):
            ct_sac_implicit_setting({CT_SAC_IMPLICIT_KEY: [1]})
        # The learner checks what it receives too.
        for bad in ({"policy_draws": 1}, {"inner_steps": 0}, {"resolvent_step": 0.0}):
            with self.subTest(learner=bad), self.assertRaises(ValueError):
                _implicit(implicit=bad)
        with self.assertRaisesRegex(ValueError, "actor_schedule"):
            _implicit(schedule="sometimes")

    def test_json_explicit_section(self):
        raw = json.loads(CONFIG.read_text())
        raw["explicit"][CT_SAC_IMPLICIT_KEY] = {"inner_steps": 8, "policy_draws": 2}
        proto = run_ph_sac.protocol_from_config(raw, "x")
        self.assertEqual(
            proto["explicit_config"][CT_SAC_IMPLICIT_KEY], {"inner_steps": 8, "policy_draws": 2}
        )
        self.assertNotIn(CT_SAC_IMPLICIT_KEY, proto["implicit_config"])
        raw["explicit"][CT_SAC_IMPLICIT_KEY] = {"inner_step": 8}
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac.protocol_from_config(raw, "x")
        raw["explicit"][CT_SAC_IMPLICIT_KEY] = {"policy_draws": 1}
        with self.assertRaises(ValueError):
            run_ph_sac.protocol_from_config(raw, "x")
        del raw["explicit"][CT_SAC_IMPLICIT_KEY]
        raw["explicit"]["ct_sac_implicit_typo"] = {}
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac.protocol_from_config(raw, "x")


class TestAgents(unittest.TestCase):
    """The agents make_training_agent builds on the smoke protocol (windy pendulum, T 10 ms,
    lr 3e-4, replay 20000, one warm-up episode, Polyak rate 0.005, a tuned temperature)."""

    @classmethod
    def setUpClass(cls):
        proto = smoke_protocol()
        proto["arms"] = [
            arm for arm in P.selectable_arms() if arm["name"] in ("ct_sac_torque", *IMPLICIT)
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

    def test_implicit_settings(self):
        testbed = self.protocol["matched"]["testbed_config"]
        T, lam = testbed["reference_period"], testbed["discount_rate"]
        for name in IMPLICIT:
            with self.subTest(arm=name):
                agent = self.agents[name]
                algorithm = agent.algorithm
                self.assertIs(type(agent), ImplicitTorqueCTSACBaseline)
                self.assertIs(type(algorithm), _ImplicitPHCTSAC)
                self.assertEqual(algorithm.resolvent_step, 1.0)
                self.assertEqual(algorithm.steps_per_solve, 400)
                self.assertEqual(algorithm.policy_draws, 4)
                self.assertEqual(algorithm.actor_steps_per_solve, 40)
                self.assertEqual(algorithm.temperature_every_rows, 40.0)
                self.assertEqual(algorithm.actor_schedule, SCHEDULES[name])
                self.assertAlmostEqual(algorithm.proximal_weight, 1.0 / T)
                self.assertAlmostEqual(algorithm.discount_rate, lam)
                self.assertAlmostEqual(algorithm.contraction, 1.0 / (1.0 + lam))
                self.assertIs(agent.solver_records, algorithm.solver_records)
                self.assertEqual(agent.solver_records, [])
                self.assertEqual(algorithm.inner_steps, 0)
        self.assertIs(type(self.agents["ct_sac_torque"]), TorqueCTSACBaseline)
        self.assertIs(type(self.agents["ct_sac_torque"].algorithm), _PHCTSAC)

    def test_everything_else_is_ct_sac_torques(self):
        reference = self.agents["ct_sac_torque"].algorithm
        for name in ("ct_sac_implicit", "ct_sac_implicit_schedule"):
            with self.subTest(arm=name):
                algorithm = self.agents[name].algorithm
                for key in (
                    "gamma", "beta", "time_rescale", "batch_size", "train_freq",
                    "gradient_steps", "num_expectation_samples", "target_entropy", "alpha",
                    "tau", "buffer_size", "learning_starts", "warmup_episodes",
                    "warmup_decisions", "terminal_time_limit",
                ):
                    self.assertEqual(getattr(algorithm, key), getattr(reference, key), key)
                self.assertEqual(algorithm.lr_schedule(1.0), reference.lr_schedule(1.0))
                self.assertEqual(algorithm.log_alpha.item(), reference.log_alpha.item())
                # The same networks at initialization (the run's seed and architecture);
                # the anchors are the critics and the frozen actor is the actor.
                for part in ("actor", "q_nets", "q_target_nets"):
                    self.assertTrue(
                        same(
                            states(getattr(algorithm.model, part)),
                            states(getattr(reference.model, part)),
                        ),
                        part,
                    )
                self.assertTrue(
                    same(states(algorithm.model.q_target_nets), states(algorithm.model.q_nets))
                )
                self.assertTrue(same(states(algorithm.frozen_actor), states(algorithm.model.actor)))
                self.assertTrue(
                    all(not p.requires_grad for p in algorithm.frozen_actor.parameters())
                )
                self.assertAlmostEqual(
                    algorithm.operator_alpha, float(reference.log_alpha.exp().item()), places=7
                )
        released = self.agents["ct_sac_implicit_released"].algorithm
        self.assertEqual(released.lr_schedule(1.0), 0.00073)
        self.assertEqual(released.buffer_size, 300000)
        self.assertEqual(released.learning_starts, 10000)
        self.assertEqual(released.warmup_decisions, 10000)
        self.assertEqual(released.tau, 0.02)
        self.assertFalse(released.terminal_time_limit)

    def test_constructor_arguments(self):
        explicit = self.protocol["explicit_config"]
        defaults = ct_sac_implicit_setting({})
        for name in IMPLICIT:
            with self.subTest(arm=name), mock.patch(
                "algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline"
            ) as built:
                P.make_training_agent(self.runs[name], None)
                (env, training, temperature), kwargs = built.call_args
                self.assertIsNone(env)
                self.assertEqual(temperature, explicit["temperature"])
                expected = dict(explicit["training"], seed=self.runs[name]["training_seed"])
                arguments = {
                    "polyak_rate": explicit["polyak_rate"],
                    "expectation_samples": 1,
                    "implicit": defaults,
                    "actor_schedule": SCHEDULES[name],
                }
                if name == "ct_sac_implicit_released":
                    expected.update(learning_rate=0.00073, replay_capacity=300000)
                    arguments.update(
                        polyak_rate=0.02, warmup_decisions=10000, time_limit="bootstrap"
                    )
                self.assertEqual(training, expected)
                self.assertEqual(kwargs, arguments)


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
                self.assertNotIn("ct_sac_implicit_step", spec)
                run = runs[name]
                settings = R.run_settings(run)
                self.assertEqual(compact_json(settings["arm"]), FIXED_ARM_JSON[name])
                self.assertNotIn(CT_SAC_IMPLICIT_KEY, settings["resolved"]["explicit_config"])
                self.assertNotIn(CT_SAC_IMPLICIT_KEY, settings["protocol"]["explicit_config"])
                # The implicit arms in the protocol's arm list leave the fingerprint alone.
                alone = dict(
                    run,
                    protocol=dict(run["protocol"], arms=[copy.deepcopy(expected)]),
                    arm=copy.deepcopy(expected),
                )
                self.assertEqual(settings, R.run_settings(alone))
                self.assertEqual(R.run_fingerprint(run), R.run_fingerprint(alone))
        # The implicit arms (with feature inputs too) hold the resolved section; each has a
        # fingerprint of its own.
        implicit = (*IMPLICIT, *IMPLICIT_FEATURES)
        for name in implicit:
            with self.subTest(arm=name):
                resolved = R.run_settings(runs[name])["resolved"]["explicit_config"]
                self.assertEqual(
                    resolved[CT_SAC_IMPLICIT_KEY], R._canonical(ct_sac_implicit_setting({}))
                )
        # The other feature arms and the released arms hold nothing of it.
        for name in (*FEATURES, "ct_sac_released", "ct_sac_released_timeout"):
            with self.subTest(arm=name):
                resolved = R.run_settings(runs[name])["resolved"]["explicit_config"]
                self.assertNotIn(CT_SAC_IMPLICIT_KEY, resolved)
        named = ("ct_sac_torque", *implicit)
        prints = {name: R.run_fingerprint(runs[name]) for name in named}
        self.assertEqual(len(set(prints.values())), len(named))
        # A changed default is detected by the implicit arms only.
        with mock.patch.dict(CT_SAC_IMPLICIT_DEFAULTS, inner_steps=8):
            after = {name: R.run_fingerprint(runs[name]) for name in named}
        self.assertEqual(after["ct_sac_torque"], prints["ct_sac_torque"])
        for name in implicit:
            self.assertNotEqual(after[name], prints[name], name)


# ------------------------------------------------------------------------ feature inputs


class TestImplicitFeatureArms(unittest.TestCase):
    """ct_sac_implicit_features, ct_sac_implicit_schedule_features and
    ct_sac_implicit_released_features on the smoke protocol (the windy pendulum: one
    periodic angle, velocity_feature_scale 2 pi, 3 features)."""

    @classmethod
    def setUpClass(cls):
        proto = smoke_protocol()
        names = (*FEATURE_SOURCES.values(), *IMPLICIT_FEATURES)
        proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in names]
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

    def test_declarations(self):
        arms = {arm["name"]: arm for arm in P.baseline_implicit_feature_arms()}
        self.assertEqual(tuple(arms), IMPLICIT_FEATURES)
        self.assertEqual(get_args(P.ImplicitFeatureBaselineArmName), IMPLICIT_FEATURES)
        catalog = {arm["name"]: arm for arm in P.selectable_arms()}
        for name, source in FEATURE_SOURCES.items():
            with self.subTest(arm=name):
                spec = dict(arms[name])
                self.assertEqual(spec.pop("ct_sac_inputs"), "state_features")
                self.assertEqual(spec.pop("name"), name)
                expected = dict(catalog[source])
                expected.pop("name")
                self.assertEqual(spec, expected)
                self.assertNotIn("ct_sac_inputs", catalog[source])
                self.assertEqual(catalog[name], arms[name])
        # The released one carries ct_sac_implicit_released's settings, the others none.
        self.assertEqual(
            arms["ct_sac_implicit_released_features"]["ct_sac_settings"],
            {**P.RELEASED_CT_SAC_SETTINGS, "time_limit": "bootstrap"},
        )
        self.assertNotIn("ct_sac_settings", arms["ct_sac_implicit_features"])
        self.assertNotIn("ct_sac_settings", arms["ct_sac_implicit_schedule_features"])
        self.assertFalse(set(IMPLICIT_FEATURES) & {arm["name"] for arm in P.all_arms()})
        self.assertFalse(set(IMPLICIT_FEATURES) & {arm["name"] for arm in P.paper_arms()})
        # Every call builds fresh specs.
        P.baseline_implicit_feature_arms()[0]["ct_sac_implicit_step"]["actor_schedule"] = "outer"
        self.assertEqual(
            P.baseline_implicit_feature_arms()[0]["ct_sac_implicit_step"]["actor_schedule"],
            "per_row",
        )
        self.assertEqual(
            P.baseline_implicit_arms()[0]["ct_sac_implicit_step"]["actor_schedule"], "per_row"
        )
        # --arms selects them.
        path = str(CONFIG)
        args = run_ph_sac.parse_args(
            ["--config", path, "--arms", *reversed(IMPLICIT_FEATURES), "--seeds", "0"]
        )
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual([arm["name"] for arm in proto["arms"]], list(IMPLICIT_FEATURES))
        P.validate_matched_quantities(proto)

    def test_constructor_arguments(self):
        explicit = self.protocol["explicit_config"]
        captured = {}
        for name in self.runs:
            with mock.patch("algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline") as built:
                P.make_training_agent(self.runs[name], None)
                captured[name] = built.call_args
        for name, source in FEATURE_SOURCES.items():
            with self.subTest(arm=name):
                (_, training, temperature), kwargs = captured[name]
                (_, source_training, source_temperature), source_kwargs = captured[source]
                self.assertNotIn("state_features", source_kwargs)
                self.assertEqual(kwargs, dict(source_kwargs, state_features=explicit["network"]))
                self.assertEqual(training, source_training)
                self.assertEqual(temperature, source_temperature)
                self.assertEqual(kwargs["actor_schedule"], SCHEDULES[source])
                self.assertEqual(kwargs["implicit"], ct_sac_implicit_setting({}))
        # The released one, in full: ct_sac_implicit_released's released settings (step
        # size, replay, warm-up in decisions, target rate, time limit) plus the input map.
        name = "ct_sac_implicit_released_features"
        (_, training, temperature), kwargs = captured[name]
        self.assertEqual(
            training,
            dict(
                explicit["training"],
                seed=self.runs[name]["training_seed"],
                learning_rate=0.00073,
                replay_capacity=300000,
            ),
        )
        self.assertEqual(temperature, explicit["temperature"])
        self.assertEqual(
            kwargs,
            {
                "polyak_rate": 0.02,
                "expectation_samples": 1,
                "implicit": ct_sac_implicit_setting({}),
                "actor_schedule": "per_row",
                "warmup_decisions": 10000,
                "time_limit": "bootstrap",
                "state_features": explicit["network"],
            },
        )
        self.assertEqual(explicit["network"]["velocity_feature_scale"], VELOCITY_SCALE)

    def test_networks_take_the_feature_dimension(self):
        for name in IMPLICIT_FEATURES:
            with self.subTest(arm=name):
                agent = self.build(name)
                algorithm = agent.algorithm
                self.assertIs(type(agent), ImplicitTorqueCTSACBaseline)
                self.assertEqual(agent.input_features.output_dim, FEATURE_DIM)
                self.assertEqual(agent.input_features.velocity_scale, VELOCITY_SCALE)
                self.assertEqual(algorithm.env.observation_space.shape, (FEATURE_DIM,))
                actions = int(agent.env.state_spec["action_dim"])
                model = algorithm.model
                for critics in (model.q_nets, model.q_target_nets):
                    for critic in critics:
                        self.assertEqual(critic[0].in_features, FEATURE_DIM + actions)
                for actor in (model.actor, algorithm.frozen_actor):
                    self.assertEqual(actor.body[0].in_features, FEATURE_DIM)
                self.assertEqual(algorithm.replay_buffer.observations.shape[-1], FEATURE_DIM)
                self.assertEqual(algorithm.replay_buffer.next_observations.shape[-1], FEATURE_DIM)
        # The source arms read the state (2 coordinates).
        source = self.build("ct_sac_implicit")
        self.assertIsNone(source.input_features)
        self.assertEqual(source.algorithm.model.actor.body[0].in_features, 2)

    def test_one_residual_step_on_feature_rows(self):
        """A synthetic batch of feature rows (the features of random states) through the
        residual and one critic step, then the training loop through one copy."""
        agent = _implicit(
            _testbed(episode_duration=0.05),
            _training(batch_size=8),
            state_features={"velocity_feature_scale": VELOCITY_SCALE},
        )
        algorithm = agent.algorithm
        adapter = algorithm.env
        self.assertEqual(algorithm.model.actor.body[0].in_features, FEATURE_DIM)
        rng = np.random.default_rng(3)
        # (Not named states: that is the module's state-dict helper, used below.)
        start_states = np.stack(
            [rng.uniform(-math.pi, math.pi, 4), rng.normal(size=4)], axis=1
        )
        end_states = start_states + 0.01 * rng.normal(size=(4, 2))
        dt = torch.tensor([[0.002], [0.01], [0.018], [0.004]])
        batch = ReplayBatch(
            observations=torch.as_tensor(
                adapter.network_input(start_states), dtype=torch.float32
            ),
            actions=torch.as_tensor(rng.uniform(-1.0, 1.0, (4, 1)), dtype=torch.float32),
            next_observations=torch.as_tensor(
                adapter.network_input(end_states), dtype=torch.float32
            ),
            rewards=torch.as_tensor(0.01 * rng.normal(size=(4, 1)), dtype=torch.float32),
            dones=torch.zeros(4, 1),
            t=torch.zeros(4, 1),
            next_t=dt.clone(),
            dt=dt,
        )
        self.assertEqual(tuple(batch.observations.shape), (4, FEATURE_DIM))
        terms = algorithm.implicit_terms(batch)
        self.assertEqual(tuple(terms["residuals"].shape), (2, 4, SMALL["policy_draws"]))
        loss = implicit_loss(terms["residuals"])
        self.assertTrue(math.isfinite(loss.item()))
        before = states(algorithm.model.q_nets)
        algorithm.critic_optimizer.zero_grad()
        loss.backward()
        algorithm.critic_optimizer.step()
        after = states(algorithm.model.q_nets)
        self.assertFalse(same(before, after))
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in after.values()))

        # The training loop: the replay holds the features of the states, mapped once; one
        # copy after K training rows; predict maps the states it is given once.
        transitions = _spy_step_dt(agent.env)
        budget = 8 + SMALL["inner_steps"] - 1
        agent.learn(budget)
        buffer = algorithm.replay_buffer
        self.assertEqual(buffer.size(), budget)
        observed = np.stack([t["state"] for t in transitions])
        observed_next = np.stack([t["next_state"] for t in transitions])
        np.testing.assert_array_equal(
            buffer.observations[:budget, 0], adapter.network_input(observed).astype(np.float32)
        )
        np.testing.assert_array_equal(
            buffer.next_observations[:budget, 0],
            adapter.network_input(observed_next).astype(np.float32),
        )
        self.assertEqual(len(agent.solver_records), 1)
        self.assertEqual(algorithm.inner_steps, 0)
        self.assertTrue(same(states(algorithm.model.q_target_nets), states(algorithm.model.q_nets)))
        probe = np.array([[math.pi - 1e-3, 0.4], [-math.pi + 1e-3, 0.4], [0.2, -1.0]])
        policy = algorithm._policy_act(
            adapter.network_input(probe).astype(np.float32), deterministic=True
        )
        np.testing.assert_array_equal(
            agent.predict(probe, deterministic=True),
            np.asarray(policy, dtype=np.float64) * agent.torque_limit,
        )

    def test_released_agent_and_residual_step_on_feature_rows(self):
        """ct_sac_implicit_released_features as make_training_agent builds it: the learner
        holds ct_sac_implicit_released's released settings, then one residual and critic
        step on a synthetic batch of feature rows (its 10000-decision warm-up keeps a short
        training loop from reaching a step)."""
        agent = self.build("ct_sac_implicit_released_features")
        algorithm = agent.algorithm
        self.assertIs(type(agent), ImplicitTorqueCTSACBaseline)
        self.assertIs(type(algorithm), _ImplicitPHCTSAC)
        self.assertEqual(algorithm.actor_schedule, "per_row")
        self.assertEqual(algorithm.lr_schedule(1.0), 0.00073)
        self.assertEqual(algorithm.buffer_size, 300000)
        self.assertEqual(algorithm.learning_starts, 10000)
        self.assertEqual(algorithm.warmup_decisions, 10000)
        self.assertEqual(algorithm.tau, 0.02)
        self.assertFalse(algorithm.terminal_time_limit)
        self.assertEqual(agent.input_features.output_dim, FEATURE_DIM)
        adapter = algorithm.env
        rng = np.random.default_rng(4)
        start_states = np.stack(
            [rng.uniform(-math.pi, math.pi, 4), rng.normal(size=4)], axis=1
        )
        end_states = start_states + 0.01 * rng.normal(size=(4, 2))
        dt = torch.tensor([[0.002], [0.01], [0.018], [0.004]])
        batch = ReplayBatch(
            observations=torch.as_tensor(
                adapter.network_input(start_states), dtype=torch.float32
            ),
            actions=torch.as_tensor(rng.uniform(-1.0, 1.0, (4, 1)), dtype=torch.float32),
            next_observations=torch.as_tensor(
                adapter.network_input(end_states), dtype=torch.float32
            ),
            rewards=torch.as_tensor(0.01 * rng.normal(size=(4, 1)), dtype=torch.float32),
            dones=torch.zeros(4, 1),
            t=torch.zeros(4, 1),
            next_t=dt.clone(),
            dt=dt,
        )
        self.assertEqual(tuple(batch.observations.shape), (4, FEATURE_DIM))
        terms = algorithm.implicit_terms(batch)
        self.assertEqual(tuple(terms["residuals"].shape), (2, 4, algorithm.policy_draws))
        loss = implicit_loss(terms["residuals"])
        self.assertTrue(math.isfinite(loss.item()))
        before = states(algorithm.model.q_nets)
        algorithm.critic_optimizer.zero_grad()
        loss.backward()
        algorithm.critic_optimizer.step()
        after = states(algorithm.model.q_nets)
        self.assertFalse(same(before, after))
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in after.values()))

    def test_listing_the_released_arm_leaves_the_other_arms_alone(self):
        """With every selectable arm in the protocol, each other arm's run settings and
        fingerprint are those of a protocol listing that arm alone; the new arm's run
        settings are ct_sac_implicit_released's plus ct_sac_inputs and its own name."""
        new = "ct_sac_implicit_released_features"
        proto = smoke_protocol()
        proto["arms"] = list(P.selectable_arms())
        P.validate_matched_quantities(proto)
        runs = {run["arm"]["name"]: run for run in P.build_run_matrix(proto)}
        self.assertIn(new, runs)
        for arm in P.selectable_arms():
            if arm["name"] == new:
                continue
            with self.subTest(arm=arm["name"]):
                run = runs[arm["name"]]
                alone = dict(
                    run,
                    protocol=dict(run["protocol"], arms=[copy.deepcopy(arm)]),
                    arm=copy.deepcopy(arm),
                )
                self.assertEqual(R.run_settings(run), R.run_settings(alone))
                self.assertEqual(R.run_fingerprint(run), R.run_fingerprint(alone))
        settings = R.run_settings(runs[new])
        source = R.run_settings(runs["ct_sac_implicit_released"])
        self.assertEqual(settings["arm"]["ct_sac_inputs"], "state_features")
        self.assertEqual(
            {k: v for k, v in settings["arm"].items() if k not in ("name", "ct_sac_inputs")},
            {k: v for k, v in source["arm"].items() if k != "name"},
        )
        self.assertEqual(
            {k: v for k, v in settings.items() if k != "arm"},
            {k: v for k, v in source.items() if k != "arm"},
        )
        self.assertNotEqual(
            R.run_fingerprint(runs[new]), R.run_fingerprint(runs["ct_sac_implicit_released"])
        )


# ------------------------------------------------------------------------------ residual


class TestResidual(unittest.TestCase):
    def test_zero_step_is_q_minus_anchor(self):
        generator = torch.Generator().manual_seed(0)
        stored = torch.randn(2, 4, 1, generator=generator)
        anchors = torch.randn(2, 4, 1, generator=generator)
        bracket = torch.randn(4, 3, generator=generator)
        residuals = resolvent_residuals(stored, anchors, bracket, 0.0)
        self.assertEqual(tuple(residuals.shape), (2, 4, 3))
        self.assertTrue(torch.equal(residuals, (stored - anchors).expand(2, 4, 3)))

    def test_bracket_and_residual_by_hand(self):
        soft_here = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        soft_next = torch.tensor([[0.5, 1.5], [2.0, 4.0]])
        rewards = torch.tensor([[0.1], [0.2]])
        dones = torch.tensor([[0.0], [1.0]])
        u = torch.tensor([[0.5], [2.0]])
        gamma_u = torch.tensor([[0.9], [0.8]])
        bracket = implicit_bracket(soft_here, soft_next, rewards, dones, u, gamma_u)
        expected = [
            [0.1 + 1.0 + (0.9 * 0.5 - 1.0) / 0.5, 0.1 + 2.0 + (0.9 * 1.5 - 2.0) / 0.5],
            [0.2, 0.2],
        ]
        for b in range(2):
            for i in range(2):
                self.assertAlmostEqual(bracket[b, i].item(), expected[b][i], places=5)
        stored = torch.tensor([[[1.0], [2.0]], [[1.5], [2.5]]])
        anchors = torch.tensor([[[0.5], [1.0]], [[0.25], [3.0]]])
        residuals = resolvent_residuals(stored, anchors, bracket, 2.0)
        for j in range(2):
            for b in range(2):
                for i in range(2):
                    hand = 3.0 * stored[j, b, 0].item() - anchors[j, b, 0].item() - 2.0 * expected[b][i]
                    self.assertAlmostEqual(residuals[j, b, i].item(), hand, places=4)

    def test_all_pairs_is_the_mean_over_distinct_pairs(self):
        generator = torch.Generator().manual_seed(1)
        residuals = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
        total = 0.0
        for j in range(2):
            rows = []
            for b in range(5):
                values = residuals[j, b].tolist()
                pairs = [
                    values[i] * values[k] for i in range(4) for k in range(4) if i != k
                ]
                rows.append(sum(pairs) / len(pairs))
            manual = sum(rows) / len(rows)
            self.assertAlmostEqual(all_pairs_loss(residuals[j]).item(), manual, places=12)
            total += manual
        self.assertAlmostEqual(implicit_loss(residuals).item(), total, places=12)


class TestLearnerTerms(unittest.TestCase):
    def setUp(self):
        self.agent = _implicit()
        self.algorithm = self.agent.algorithm
        self.batch = synthetic_batch()

    def test_terms_by_hand(self):
        algorithm, batch = self.algorithm, self.batch
        terms = algorithm.implicit_terms(batch)
        rows, draws = 4, SMALL["policy_draws"]
        self.assertEqual(tuple(terms["residuals"].shape), (2, rows, draws))
        self.assertAlmostEqual(algorithm.proximal_weight, 1.0 / REFERENCE_PERIOD)
        with torch.no_grad():
            anchors = torch.stack(algorithm.model.target_q_values(batch.observations, batch.actions))
            stored = torch.stack(algorithm.model.q_values(batch.observations, batch.actions))
        self.assertTrue(torch.equal(terms["anchors"], anchors))
        torch.testing.assert_close(terms["stored"].detach(), stored)
        # Right after construction the anchors are the critics: c = 0 would give ~0.
        torch.testing.assert_close(terms["stored"].detach(), terms["anchors"])
        c = algorithm.proximal_weight
        alpha = algorithm.operator_alpha
        beta = algorithm.beta
        q_here = terms["q_here"].detach()
        q_next = terms["q_next"].detach()
        for b in range(rows):
            u = batch.dt[b, 0].item() / REFERENCE_PERIOD
            gamma_u = math.exp(-beta * u)
            r = batch.rewards[b, 0].item()
            d = batch.dones[b, 0].item()
            for i in range(draws):
                here = min(q_here[0, b, i].item(), q_here[1, b, i].item()) - alpha * terms[
                    "log_here"
                ][b, i].item()
                there = min(q_next[0, b, i].item(), q_next[1, b, i].item()) - alpha * terms[
                    "log_next"
                ][b, i].item()
                bracket = r + (1.0 - d) * (here + (gamma_u * there - here) / (u + 1e-8))
                self.assertAlmostEqual(
                    terms["bracket"][b, i].item(), bracket,
                    delta=1e-4 * max(1.0, abs(bracket)),
                )
                for j in range(2):
                    hand = (1.0 + c) * terms["stored"][j, b, 0].item() - terms["anchors"][
                        j, b, 0
                    ].item() - c * bracket
                    self.assertAlmostEqual(
                        terms["residuals"][j, b, i].item(), hand,
                        delta=1e-3 * max(1.0, abs(hand)),
                    )

    def test_zero_step_reduces_to_q_minus_anchor(self):
        self.algorithm.proximal_weight = 0.0
        terms = self.algorithm.implicit_terms(self.batch)
        expected = (terms["stored"] - terms["anchors"]).expand_as(terms["residuals"])
        self.assertTrue(torch.equal(terms["residuals"], expected))

    def test_gradient_reaches_the_next_state_value(self):
        algorithm, batch = self.algorithm, self.batch
        terms = algorithm.implicit_terms(batch)
        loss = implicit_loss(terms["residuals"])
        grad_next, grad_here = torch.autograd.grad(
            loss, [terms["q_next"], terms["q_here"]], retain_graph=True
        )
        for name, gradient in (("next", grad_next), ("here", grad_here)):
            per_row = gradient.abs().sum(dim=(0, 2))
            for b in range(4):
                with self.subTest(path=name, row=b):
                    if batch.dones[b, 0].item() == 1.0:
                        # A failed row's target is its reward alone.
                        self.assertEqual(per_row[b].item(), 0.0)
                    else:
                        self.assertGreater(per_row[b].item(), 0.0)
        # The critics' gradient differs from the one with the next-state values held fixed.
        parameters = list(algorithm.model.critic_parameters)
        full = torch.autograd.grad(loss, parameters, retain_graph=True)
        alpha = algorithm.operator_alpha
        held_next = terms["q_next"].detach().min(dim=0).values - alpha * terms["log_next"]
        u = batch.dt * algorithm.time_rescale
        bracket = implicit_bracket(
            terms["soft_here"], held_next, batch.rewards, batch.dones, u,
            torch.exp(-algorithm.beta * u),
        )
        held = resolvent_residuals(
            terms["stored"], terms["anchors"], bracket, algorithm.proximal_weight
        )
        semi = torch.autograd.grad(implicit_loss(held), parameters)
        self.assertTrue(
            any(not torch.allclose(a, b) for a, b in zip(full, semi)),
            "detaching the next-state value must change the critics' gradient",
        )

    def test_draws_come_from_the_frozen_actor(self):
        algorithm = self.algorithm
        torch.manual_seed(7)
        first = algorithm.implicit_terms(self.batch)["residuals"].detach()
        with torch.no_grad():
            for parameter in algorithm.model.actor.parameters():
                parameter.add_(0.5)
        torch.manual_seed(7)
        second = algorithm.implicit_terms(self.batch)["residuals"].detach()
        self.assertTrue(torch.equal(first, second))
        with torch.no_grad():
            for parameter in algorithm.frozen_actor.parameters():
                parameter.add_(0.5)
        torch.manual_seed(7)
        third = algorithm.implicit_terms(self.batch)["residuals"].detach()
        self.assertFalse(torch.equal(first, third))


# -------------------------------------------------------------------------------- solves


class TestSolves(unittest.TestCase):
    @staticmethod
    def watch(agent):
        """After every train call: the update count, the solve's position and the networks."""
        algorithm = agent.algorithm
        log = []
        original = algorithm.train

        def train(gradient_steps, batch_size):
            original(gradient_steps, batch_size)
            model = algorithm.model
            log.append(
                {
                    "updates": algorithm.gradient_updates,
                    "inner": algorithm.inner_steps,
                    "critics": states(model.q_nets),
                    "targets": states(model.q_target_nets),
                    "actor": states(model.actor),
                    "frozen": states(algorithm.frozen_actor),
                }
            )

        algorithm.train = train
        return log

    def test_hard_copy_exactly_every_k_rows(self):
        K = SMALL["inner_steps"]
        for schedule in CT_SAC_ACTOR_SCHEDULES:
            with self.subTest(schedule=schedule):
                agent = _implicit(
                    _testbed(episode_duration=0.05), _training(batch_size=8), schedule=schedule
                )
                algorithm = agent.algorithm
                last = {"critics": states(algorithm.model.q_nets), "actor": states(algorithm.model.actor)}
                self.assertTrue(same(states(algorithm.model.q_target_nets), last["critics"]))
                self.assertTrue(same(states(algorithm.frozen_actor), last["actor"]))
                log = self.watch(agent)
                agent.learn(40)
                # Training from the decision at which the replay holds a batch: 8 to 40.
                self.assertEqual(len(log), 40 - 8 + 1)
                self.assertEqual([entry["updates"] for entry in log], list(range(1, len(log) + 1)))
                for entry in log:
                    g = entry["updates"]
                    if g % K == 0:
                        self.assertEqual(entry["inner"], 0, g)
                        self.assertTrue(same(entry["targets"], entry["critics"]), g)
                        self.assertTrue(same(entry["frozen"], entry["actor"]), g)
                        last = {"critics": entry["critics"], "actor": entry["actor"]}
                    else:
                        self.assertEqual(entry["inner"], g % K, g)
                        self.assertTrue(same(entry["targets"], last["critics"]), g)
                        self.assertFalse(same(entry["targets"], entry["critics"]), g)
                        self.assertTrue(same(entry["frozen"], last["actor"]), g)
                        if schedule == "outer":
                            # The policy is frozen through the solve.
                            self.assertTrue(same(entry["actor"], last["actor"]), g)
                        else:
                            self.assertFalse(same(entry["actor"], last["actor"]), g)
                solves = len(log) // K
                records = agent.solver_records
                self.assertEqual(len(records), solves)
                for index, record in enumerate(records):
                    self.assertLessEqual(set(SolverMetrics.__required_keys__), set(record))
                    self.assertEqual(record["outer_iteration"], index)
                    self.assertEqual(record["inner_iterations"], K)
                    self.assertEqual(record["status"], "completed")
                    self.assertAlmostEqual(
                        record["contraction_factor"], 1.0 / (1.0 + DISCOUNT_RATE * 1.0)
                    )
                    self.assertEqual(record["fixed_point_motion_input"], 0.0)
                    for key in ("residual_u_statistic", "q_policy_minus_stored", "q_abs_max"):
                        self.assertTrue(math.isfinite(record[key]), key)
                    self.assertGreaterEqual(record["q_abs_max"], 0.0)
                    # The copy after the K-th row of the solve: decision 8 + K (index + 1) - 1.
                    self.assertEqual(record["interactions"], 8 + K * (index + 1) - 1)
                counters = agent.counters()
                self.assertEqual(set(counters), set(TrainingCounters.__annotations__))
                self.assertEqual(counters["critic_gradient_steps"], len(log))
                self.assertEqual(counters["value_gradient_steps"], 0)
                if schedule == "per_row":
                    self.assertEqual(counters["actor_gradient_steps"], len(log))
                    self.assertEqual(counters["temperature_gradient_steps"], len(log))
                else:
                    self.assertEqual(
                        counters["actor_gradient_steps"], solves * SMALL["actor_steps"]
                    )
                    self.assertEqual(
                        counters["temperature_gradient_steps"],
                        int(math.floor(solves * K / SMALL["temperature_every_rows"])),
                    )

    def test_anchors_never_follow_by_polyak(self):
        """polyak_rate is unused: with rate 1 the anchors still wait for the K-th row."""
        agent = _implicit(_testbed(episode_duration=0.05), _training(batch_size=8), polyak_rate=1.0)
        log = self.watch(agent)
        agent.learn(8 + SMALL["inner_steps"] - 2)
        self.assertGreater(len(log), 0)
        for entry in log:
            self.assertFalse(same(entry["targets"], entry["critics"]))


# --------------------------------------------------------------------------- checkpoints


class TestCheckpoint(unittest.TestCase):
    def test_resume_state_round_trip(self):
        K = SMALL["inner_steps"]
        for schedule in CT_SAC_ACTOR_SCHEDULES:
            with self.subTest(schedule=schedule):
                build = lambda: _implicit(  # noqa: E731
                    _testbed(episode_duration=0.05), _training(batch_size=8), schedule=schedule
                )
                agent = build()
                # 2K + 3 training rows (decisions 8 to 2K + 10): two copies, three rows in.
                agent.learn(2 * K + 10)
                # Through torch.save and torch.load, as the resume path writes and reads it
                # (resume.atomic_torch_save, load_resume_payload). Handed over in memory, the
                # state would alias the agent's live tensors: resume_state() returns the
                # optimizers' own state_dict()s, and Optimizer.load_state_dict keeps tensors
                # it need not cast, so both agents' Adam moments and step counts would be
                # one set of tensors that the agent's further steps move for both.
                saved = io.BytesIO()
                torch.save(agent.resume_state(), saved)
                saved.seek(0)
                state = torch.load(saved, weights_only=False)
                implicit = state["implicit_ct_sac"]
                self.assertEqual(implicit["inner_steps"], 3)
                self.assertEqual(implicit["outer_iterations"], 2)
                self.assertEqual(len(implicit["solver_records"]), 2)
                self.assertTrue(same(implicit["frozen_actor"], states(agent.algorithm.frozen_actor)))
                fresh = build()
                fresh.load_resume_state(state)
                self.assertEqual(differences(agent.resume_state(), fresh.resume_state()), [])
                # The two agents share no optimizer tensor (the premise of the continuation
                # check below).
                for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
                    pointers = []
                    for owner in (agent, fresh):
                        optimizer = getattr(owner.algorithm, name)
                        pointers.append(
                            {
                                tensor.data_ptr()
                                for slot in optimizer.state.values()
                                for tensor in slot.values()
                                if torch.is_tensor(tensor)
                            }
                        )
                    self.assertTrue(pointers[0], name)
                    self.assertFalse(pointers[0] & pointers[1], name)
                self.assertIs(fresh.solver_records, fresh.algorithm.solver_records)
                self.assertEqual(fresh.solver_records, agent.solver_records)
                for part in ("frozen_actor",):
                    self.assertTrue(
                        same(states(getattr(fresh.algorithm, part)), states(getattr(agent.algorithm, part)))
                    )
                self.assertTrue(
                    same(
                        states(fresh.algorithm.model.q_target_nets),
                        states(agent.algorithm.model.q_target_nets),
                    )
                )
                self.assertEqual(fresh.algorithm.operator_alpha, agent.algorithm.operator_alpha)
                ours = {k: v for k, v in agent.counters().items() if k != "wall_seconds"}
                theirs = {k: v for k, v in fresh.counters().items() if k != "wall_seconds"}
                self.assertEqual(ours, theirs)
                # From the same global streams both continue to the same state.
                streams = R.capture_global_rng()
                agent.learn(3 * K + 12)
                R.restore_global_rng(streams)
                fresh.learn(3 * K + 12)
                self.assertEqual(differences(agent.resume_state(), fresh.resume_state()), [])


class TestExactRestart(ExactRestartMixin, unittest.TestCase):
    """Uninterrupted against stopped at decision 85 and resumed in a disturbed fresh agent
    (implicit_protocol: training from decision 50, a copy every 8 rows)."""

    BUDGET = 120

    def check(self, name, stop):
        options = R.resume_options(every_interactions=0)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            (reference_run,) = P.build_run_matrix(implicit_protocol(first, self.BUDGET, (name,)))
            reference, ref_agent = run_captured(reference_run, options)
            (run,) = P.build_run_matrix(implicit_protocol(second, self.BUDGET, (name,)))
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
            self.assertIsInstance(stopped, R.ResumableStop)
            folder = P.run_output_directory(run)
            payload = torch.load(folder / R.RESUME_NAME, weights_only=False)
            resumed, res_agent = run_captured(run, options, disturbed=True)
            self.assertFalse((folder / R.RESUME_NAME).exists())
            self.assertTrue((folder / R.COMPLETE_NAME).exists())
        self.assertGreaterEqual(payload["interactions"], stop)
        self.assertEqual(reference["counters"]["interactions"], self.BUDGET)
        self.assert_same_run(reference, ref_agent, resumed, res_agent)
        implicit = payload["agent"]["implicit_ct_sac"]
        self.assertIn(implicit["inner_steps"], range(8))
        self.assertGreaterEqual(len(implicit["solver_records"]), 1)
        self.assertGreater(len(reference["solver"]), len(implicit["solver_records"]))
        resolved = payload["settings"]["resolved"]["explicit_config"][CT_SAC_IMPLICIT_KEY]
        self.assertEqual(resolved["inner_steps"], 8)
        self.assertEqual(resolved["policy_draws"], 2)
        self.assertEqual(
            payload["settings"]["arm"]["ct_sac_implicit_step"],
            {"actor_schedule": SCHEDULES[name]},
        )
        return reference

    def test_ct_sac_implicit(self):
        self.check("ct_sac_implicit", 85)

    def test_ct_sac_implicit_schedule(self):
        self.check("ct_sac_implicit_schedule", 85)


# ------------------------------------------------------------------- the parent CT-SAC


class TestParentUnchanged(unittest.TestCase):
    def test_the_hook_builds_the_legacy_learner_as_before(self):
        baseline = _baseline()
        self.assertIs(TorqueCTSACBaseline._legacy_class, _PHCTSAC)
        self.assertIs(type(baseline.algorithm), _PHCTSAC)
        self.assertEqual(baseline._legacy_kwargs(), {})
        self.assertEqual(baseline.algorithm.num_expectation_samples, 1)
        self.assertFalse(hasattr(baseline.algorithm, "frozen_actor"))
        self.assertIs(ImplicitTorqueCTSACBaseline._legacy_class, _ImplicitPHCTSAC)

    def test_one_update_matches_a_hand_computation(self):
        """One CT-SAC update on one stored row (N_e = 1, alpha 0) with the target critics
        replaced by value at the row's state and next_value elsewhere, as
        tests/test_ph_baseline.py TestCriticTarget: the critic loss is
        sum_i (Q_i(s, a) - y)^2 with y = r/u + value + (e^(-lambda Delta) next_value - value)/u."""
        lam, T, delta, r = 0.1, 0.01, 0.004, -0.0123
        value, next_value = -3.0, -3.5
        transition = {
            "state": np.array([0.1, 0.2]),
            "decision": np.array([1.0]),
            "duration": delta,
            "reward": r,
            "next_state": np.array([0.15, 0.3]),
            "terminated": False,
            "truncated": False,
            "time": 7.0,
            "next_time": 7.0 + delta,
        }
        baseline = _baseline(training=_training(batch_size=1))
        algorithm = baseline.algorithm
        adapted = baseline.adapt_transition(transition)
        algorithm._store_transition(
            obs=np.asarray(transition["state"], dtype=np.float32),
            action=np.asarray(transition["decision"], dtype=np.float32),
            reward=adapted["reward"],
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
        stored_action = torch.as_tensor(algorithm.replay_buffer.actions[0], dtype=torch.float32)
        with torch.no_grad():
            before = [
                q.item()
                for q in algorithm.model.q_values(start.view(1, -1), stored_action.view(1, -1))
            ]
        real_loss = torch.nn.functional.mse_loss
        captured = []

        def capture(prediction, target):
            loss = real_loss(prediction, target)
            captured.append((prediction.detach().clone(), target.detach().clone(), loss.item()))
            return loss

        with mock.patch("algorithms.ct_sac.F.mse_loss", side_effect=capture):
            algorithm.train(gradient_steps=1, batch_size=1)
        u = delta / T
        target = r / u + value + (math.exp(-lam * delta) * next_value - value) / u
        self.assertEqual(len(captured), 2)
        for (prediction, captured_target, _), q in zip(captured, before):
            self.assertAlmostEqual(captured_target.item(), target, places=3)
            self.assertAlmostEqual(prediction.item(), q, places=5)
        total = sum(loss for *_, loss in captured)
        hand = sum((q - target) ** 2 for q in before)
        self.assertTrue(math.isclose(total, hand, rel_tol=1e-4), (total, hand))
        # The parent took exactly one step and keeps its Polyak target update.
        self.assertEqual(algorithm.gradient_updates, 1)


if __name__ == "__main__":
    unittest.main()
