"""reward_observation "end_state" (common.ph_config.REWARD_OBSERVATION_KEY, since 2026-11-22).

Under "end_state" the only reward datum of a transition any learner may use is the task's
rate at the interval's end state with the torque there (rho_end, MujocoPHEnv.end_reward_rate):
the replay hands out end_rewards and no rewards, the CT-SAC arms run ct_sac_reward_source
"end_state", and every CT-PH-SAC arm fits a reward-rate network to rho_end and integrates it
over each interval (the generator target's left rectangle, the rollout target's path
trapezoids, the model-free targets' observed-ends trapezoid, each plus the failure charge).
"interval" (absent: every earlier run) is the code before the key, bitwise. Windy-pendulum
agents of tests/test_ph_algorithms.py, and tiny protocols of table rows
(tests/test_ph_resume.py tiny_run) for the paper and ablation arms.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import algorithms.implicit_ct_ph_sac as implicit_module
import algorithms.ph_base as base_module
from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC
from algorithms.ph_resolvent import excluded_resolvent_residuals, observed_interval_residuals
from algorithms.ph_targets import (
    data_rate_target,
    failure_charge,
    generator_continuation,
    generator_rate_target,
    observed_continuation,
    observed_end_reward,
    rollout_interval_reward,
    start_rate_reward,
)
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import (
    DEFAULT_REWARD_OBSERVATION,
    IMPLICIT_DEFAULTS,
    REWARD_OBSERVATION_KEY,
    REWARD_OBSERVATIONS,
    check_reward_observation,
    effective_ct_sac_reward_source,
    reward_observation_setting,
    validate_configuration,
)
from common.ph_replay import ReplayPartitions
from environment.ph_sac.registry import testbed_spec
from environment.ph_sac.tasks import (
    failure_mask_tensor,
    make_env,
    make_oracle_plant,
    reward_rate_tensor,
    terminal_payoff_tensor,
)
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.diagnostics import ImplicitTrainingCounters, TrainingCounters
from models.ph_sac.reward import (
    RewardRate,
    end_state_fit_rows,
    end_state_reward_loss,
    reward_fit_rows,
)
from models.ph_sac.rollouts import (
    grid_refresh_steps,
    sample_brownian_increments,
    stochastic_rollout,
    survivor_continuation,
)
from tests.test_ph_algorithms import (
    all_parameters,
    assert_same_state,
    explicit_config,
    implicit_config,
    make_explicit,
    make_implicit,
    same,
    snapshot,
    spy_transitions,
    streams,
    synthetic_reward_batch,
    windy,
)
from tests.test_ph_contact_substep import COARSE, cartpole
from tests.test_ph_replay import INTERVAL_ARRAYS, INTERVAL_BATCH
from tests.test_ph_resume import differences, tiny_run

END_STATE = {"reward_observation": "end_state"}


@contextlib.contextmanager
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def end_state_agent(kind, *, seed=11, **overrides):
    """A windy-pendulum agent (feedback decisions; test_ph_algorithms' configs with the
    overrides) under reward_observation "end_state"."""
    maker = explicit_config if kind == "explicit" else implicit_config
    config = maker(reward_observation="end_state", **overrides)
    env = make_env(config["testbed"], seed=seed)
    agent_class = ExplicitCTPHSAC if kind == "explicit" else ImplicitCTPHSAC
    return agent_class(env, config)


def exclusion_agent(kind, **overrides):
    """A cart-pole agent (feedback decisions) under contact exclusion and "end_state"."""
    maker = explicit_config if kind == "explicit" else implicit_config
    config = maker(
        cartpole(), model_coarse_step=COARSE, contact_handling="exclude",
        reward_observation="end_state", **overrides,
    )
    env = make_env(config["testbed"], seed=11)
    agent_class = ExplicitCTPHSAC if kind == "explicit" else ImplicitCTPHSAC
    return agent_class(env, config)


def end_state_batch(agent, rows=16, *, seed=0, failed=(), truncated=()):
    """synthetic_reward_batch's rows as a replay under "end_state" gives them: end_rewards
    (the task's rate at s' with the end torque) in place of rewards."""
    batch = synthetic_reward_batch(agent, rows, seed=seed, failed=failed, truncated=truncated)
    del batch["rewards"]
    resolved = agent.controller.resolve_decision(batch["states"], batch["decisions"])
    end = agent.controller.action(batch["next_states"], resolved)
    batch["end_rewards"] = reward_rate_tensor(batch["next_states"], end, agent.testbed)
    return batch


def fill(agent, rows):
    """rows of uniform decisions (the warm-up's) into the agent's replay, no update."""
    for _ in range(rows):
        agent._collect(lambda state: agent._uniform_decision())


def flagged(batch, rows):
    """batch with contact_flags set on the given row indices only, and that mask [B]."""
    batch = dict(batch)
    flags = torch.zeros(batch["states"].shape[0], 1, dtype=batch["states"].dtype)
    flags[list(rows)] = 1.0
    batch["contact_flags"] = flags
    return batch, flags[:, 0] > 0.5


def run_agent(env_id, arm, budget=10, *, protocol=END_STATE, **settings):
    """The agent make_training_agent builds for a tiny run of a table row, and the run."""
    run = tiny_run(env_id, arm, "/nonexistent", budget, protocol=protocol, **settings)
    testbed = run["protocol"]["matched"]["testbed_config"]
    env = make_env(testbed, decision_mode=run["arm"]["decision_mode"], seed=0)
    return P.make_training_agent(run, env), run


def row_testbed(env_id):
    entry = table.find_ph_entry(env_id, "top")
    return run_ph_sac._build_testbed(entry.config["testbed"], entry.config["testbed_settings"])


def end_torques(agent, batch):
    """u_e: the controller's torques at the batch's next states (decisions resolved at s)."""
    with torch.no_grad():
        resolved = agent.controller.resolve_decision(batch["states"], batch["decisions"])
        return agent.controller.action(batch["next_states"], resolved)


def start_torques(agent, batch):
    with torch.no_grad():
        resolved = agent.controller.resolve_decision(batch["states"], batch["decisions"])
        return agent.controller.action(batch["states"], resolved)


def charges(agent, batch):
    """c = d e^(-lambda Delta) F(s'), [B,1], computed by hand."""
    return (
        batch["terminated"]
        * torch.exp(-agent.discount_rate * batch["durations"])
        * terminal_payoff_tensor(batch["next_states"], agent.testbed)
    )


def trapezoids(agent, batch, rate):
    """Delta/2 [rate(s, u_0) + e^(-lambda Delta) rate(s', u_e)] + c, [B,1], by hand."""
    durations = batch["durations"]
    with torch.no_grad():
        start = rate(batch["states"], start_torques(agent, batch))
        end = rate(batch["next_states"], end_torques(agent, batch))
    discount = torch.exp(-agent.discount_rate * durations)
    return 0.5 * durations * (start + discount * end) + charges(agent, batch)


# ------------------------------------------------------------------------- the setting


class TestSetting(unittest.TestCase):
    def test_constants_and_defaults(self):
        self.assertEqual(REWARD_OBSERVATION_KEY, "reward_observation")
        self.assertEqual(REWARD_OBSERVATIONS, ("interval", "end_state"))
        self.assertEqual(DEFAULT_REWARD_OBSERVATION, "interval")
        for value in REWARD_OBSERVATIONS:
            self.assertEqual(check_reward_observation(value), value)
            self.assertEqual(reward_observation_setting({"reward_observation": value}), value)
        self.assertEqual(reward_observation_setting({}), "interval")
        for bad in ("end", "End_state", "", None, 1.0, "known"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "reward_observation"):
                    check_reward_observation(bad)
                with self.assertRaisesRegex(ValueError, "reward_observation"):
                    reward_observation_setting({"reward_observation": bad})
        # Not an implicit default: every implicit fingerprint holds IMPLICIT_DEFAULTS' keys.
        self.assertNotIn("reward_observation", IMPLICIT_DEFAULTS)
        run = tiny_run("pendulum-swingup", "implicit_torque", "/nonexistent", 10)
        self.assertNotIn("reward_observation", run["protocol"])
        for agent in (make_explicit(), make_implicit()):
            self.assertEqual(agent.reward_observation, "interval")
            self.assertFalse(agent.end_state)
            self.assertEqual(agent.replay.reward_observation, "interval")
        for kind in ("explicit", "implicit"):
            agent = end_state_agent(kind)
            self.assertTrue(agent.end_state)
            self.assertEqual(agent.replay.reward_observation, "end_state")
            for buffer in agent.replay.buffers.values():
                self.assertEqual(buffer.reward_observation, "end_state")
        config = explicit_config()
        config["reward_observation"] = "end"
        with self.assertRaisesRegex(ValueError, "reward_observation"):
            validate_configuration(config)
        # A replay passed in must hold the config's datum.
        config = implicit_config(reward_observation="end_state")
        env = make_env(config["testbed"], seed=11)
        replay = ReplayPartitions(capacity=100, validation_fraction=0.1, device="cpu", seed=3)
        with self.assertRaisesRegex(ValueError, "reward_observation"):
            ImplicitCTPHSAC(env, config, replay=replay)

    def test_ct_sac_source_follows_and_a_contradiction_raises(self):
        cases = (
            ({}, "interval"),
            ({"ct_sac_reward_source": "interval"}, "interval"),
            ({"ct_sac_reward_source": "end_state"}, "end_state"),
            ({"reward_observation": "interval", "ct_sac_reward_source": "end_state"}, "end_state"),
            ({"reward_observation": "end_state"}, "end_state"),
            ({"reward_observation": "end_state", "ct_sac_reward_source": "end_state"}, "end_state"),
        )
        for protocol, expected in cases:
            with self.subTest(protocol=protocol):
                self.assertEqual(effective_ct_sac_reward_source(protocol), expected)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            effective_ct_sac_reward_source(
                {"reward_observation": "end_state", "ct_sac_reward_source": "interval"}
            )
        with self.assertRaisesRegex(ValueError, "ct_sac_reward_source"):
            effective_ct_sac_reward_source(
                {"reward_observation": "end_state", "ct_sac_reward_source": "end"}
            )
        run = tiny_run("pendulum-swingup", "ct_sac_torque", "/nonexistent", 10, protocol=END_STATE)
        proto = copy.deepcopy(run["protocol"])
        P.validate_matched_quantities(proto)
        proto["ct_sac_reward_source"] = "interval"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            P.validate_matched_quantities(proto)

    def test_known_reward_is_refused_under_end_state(self):
        with self.assertRaisesRegex(ValueError, "learned"):
            implicit_config(reward_source="known", reward_observation="end_state")
        config = implicit_config(reward_source="known")
        config["reward_observation"] = "end_state"
        env = make_env(config["testbed"], seed=11)
        with self.assertRaisesRegex(ValueError, "learned"):
            ImplicitCTPHSAC(env, config)
        run = tiny_run("pendulum-swingup", "implicit_torque", "/nonexistent", 10, protocol=END_STATE)
        proto = copy.deepcopy(run["protocol"])
        proto["implicit_config"]["reward_source"] = "known"
        with self.assertRaisesRegex(ValueError, "learned"):
            P.validate_matched_quantities(proto)
        # Under "interval" the known reward stays an option.
        self.assertIsNone(make_implicit(implicit_config(reward_source="known")).reward_net)

    def test_arm_configs_may_not_carry_the_key(self):
        run = tiny_run("pendulum-swingup", "implicit_torque", "/nonexistent", 10, protocol=END_STATE)
        for name in ("explicit_config", "implicit_config"):
            self.assertNotIn("reward_observation", run["protocol"][name])
            proto = copy.deepcopy(run["protocol"])
            proto[name]["reward_observation"] = "end_state"
            with self.subTest(config=name), self.assertRaisesRegex(ValueError, "protocol key"):
                P.validate_matched_quantities(proto)
        # The arm builders write it into a CT-PH-SAC arm's config, only off the default.
        for arm in ("implicit_torque", "explicit_generator_torque", "explicit_rollout_torque",
                    "implicit_learned_admissions", "implicit_critic"):
            with self.subTest(arm=arm):
                keyed = tiny_run("pendulum-swingup", arm, "/nonexistent", 10, protocol=END_STATE)
                plain = tiny_run("pendulum-swingup", arm, "/nonexistent", 10)
                stated = tiny_run(
                    "pendulum-swingup", arm, "/nonexistent", 10,
                    protocol={"reward_observation": "interval"},
                )
                implicit = keyed["arm"]["algorithm"] == "implicit"
                build = P._implicit_arm_config if implicit else P._explicit_arm_config
                self.assertEqual(build(keyed)["reward_observation"], "end_state")
                self.assertNotIn("reward_observation", build(plain))
                self.assertNotIn("reward_observation", build(stated))


# ------------------------------------------------------------------- the end-state fit


class TestEndStateFit(unittest.TestCase):
    def test_every_row_is_fitted_at_its_end_state(self):
        agent = end_state_agent("implicit")
        batch = end_state_batch(agent, 8, failed=(0, 3, 5), truncated=(1, 6))
        next_states, torques, rates = end_state_fit_rows(agent.controller, batch)
        self.assertEqual(next_states.shape[0], 8)
        self.assertTrue(torch.equal(next_states, batch["next_states"]))
        self.assertTrue(torch.equal(torques, end_torques(agent, batch)))
        self.assertIs(rates, batch["end_rewards"])
        # The interval fit would read the interval rewards, which the batch lacks.
        with self.assertRaises(KeyError):
            reward_fit_rows(agent.controller, batch)
        # A batch of failed rows alone still takes a step (the interval fit takes none).
        for kind in ("implicit", "explicit"):
            with self.subTest(kind=kind):
                agent = end_state_agent(kind)
                failed = end_state_batch(agent, 4, failed=range(4))
                metrics = agent.update_reward(failed)
                self.assertIn("reward_fit_loss", metrics)
                self.assertEqual(agent.counters()["reward_gradient_steps"], 1)

    def test_loss_is_the_end_state_regression(self):
        agent = end_state_agent("implicit")
        batch = end_state_batch(agent, 16, failed=(2,), truncated=(5,))
        net = copy.deepcopy(agent.reward_net)
        end = end_torques(agent, batch)
        with torch.no_grad():
            hand = ((net(batch["next_states"], end) - batch["end_rewards"]) ** 2).mean()
            computed = end_state_reward_loss(net, batch["next_states"], end, batch["end_rewards"])
        torch.testing.assert_close(computed, hand)
        metrics = agent.update_reward(batch)
        self.assertAlmostEqual(metrics["reward_fit_loss"], float(hand), places=6)
        self.assertEqual(agent.last_reward_fit_loss, metrics["reward_fit_loss"])
        # The regression fits the true rate on a synthetic batch.
        batch = end_state_batch(agent, 128)
        losses = [agent.update_reward(batch)["reward_fit_loss"] for _ in range(300)]
        self.assertLess(losses[-1], 0.2 * losses[0])
        self.assertEqual(agent.counters()["reward_gradient_steps"], 301)

    def test_update_reward_switches_and_counts(self):
        def spies():
            return (
                mock.patch.object(
                    base_module, "end_state_reward_loss", wraps=base_module.end_state_reward_loss
                ),
                mock.patch.object(
                    implicit_module, "reward_rate_loss", wraps=implicit_module.reward_rate_loss
                ),
            )

        for kind in ("implicit", "explicit"):
            with self.subTest(kind=kind):
                agent = end_state_agent(kind)
                batch = end_state_batch(agent, 8)
                end_spy, trapezoid_spy = spies()
                with end_spy as end, trapezoid_spy as trapezoid:
                    for _ in range(3):
                        metrics = agent.update_reward(batch)
                self.assertEqual((end.call_count, trapezoid.call_count), (3, 0))
                self.assertEqual(agent.counters()["reward_gradient_steps"], 3)
                self.assertEqual(agent.last_reward_fit_loss, metrics["reward_fit_loss"])
        plain = make_implicit()
        end_spy, trapezoid_spy = spies()
        with end_spy as end, trapezoid_spy as trapezoid:
            plain.update_reward(synthetic_reward_batch(plain, 8))
        self.assertEqual((end.call_count, trapezoid.call_count), (0, 1))
        # The explicit agent without the key has no reward step.
        self.assertEqual(make_explicit().update_reward(end_state_batch(plain, 8)), {})
        # One reward step per value step (no warm fit in these configs), as under "interval".
        agent = end_state_agent("implicit")
        agent.learn(40)
        counters = agent.counters()
        self.assertGreater(counters["value_gradient_steps"], 0)
        self.assertEqual(counters["reward_gradient_steps"], counters["value_gradient_steps"])

    def test_warm_fit_uses_the_end_state_loss(self):
        agent = end_state_agent("implicit", plant_warm_steps=0, reward_warm_steps=7)
        fill(agent, 40)
        reward = snapshot(agent.reward_net)
        with mock.patch.object(
            base_module, "end_state_reward_loss", wraps=base_module.end_state_reward_loss
        ) as end, mock.patch.object(
            implicit_module, "reward_rate_loss", wraps=implicit_module.reward_rate_loss
        ) as trapezoid:
            agent.warm_fit()
        self.assertEqual((end.call_count, trapezoid.call_count), (7, 0))
        counters = agent.counters()
        self.assertEqual(counters["reward_gradient_steps"], 7)
        self.assertEqual(counters["value_gradient_steps"], 0)
        self.assertFalse(same(agent.reward_net, reward))
        # Each warm step read a replay batch of end-state rates.
        for call in end.call_args_list:
            self.assertEqual(call.args[3].shape, (agent.training["batch_size"], 1))

    def test_diagnostics_read_end_states_and_no_stream(self):
        for kind in ("implicit", "explicit"):
            with self.subTest(kind=kind):
                agent = end_state_agent(kind)
                agent.reward_diagnostics_every = 3
                batch = end_state_batch(agent, 16)
                global_state = torch.get_rng_state()
                policy_state = agent._generator.get_state()
                with mock.patch.object(
                    agent, "reward_fit_diagnostics", wraps=agent.reward_fit_diagnostics
                ) as diagnostics:
                    for _ in range(7):
                        agent.update_reward(batch)
                self.assertTrue(torch.equal(torch.get_rng_state(), global_state))
                self.assertTrue(torch.equal(agent._generator.get_state(), policy_state))
                self.assertEqual([r["steps"] for r in agent.reward_records], [3, 6])
                self.assertEqual(diagnostics.call_count, 2)
                states, torques = diagnostics.call_args.args
                self.assertTrue(torch.equal(states, batch["next_states"]))
                self.assertTrue(torch.equal(torques, end_torques(agent, batch)))
                record = agent.reward_records[-1]
                for key in ("rmse_taken", "rmse_policy_draws", "true_rate_sd_taken"):
                    self.assertTrue(math.isfinite(record[key]) and record[key] >= 0.0, key)
                self.assertEqual(record["reward_output"], agent.reward_net.output)


# ------------------------------------------------------- the explicit arms' reward network


class TestExplicitRewardNetwork(unittest.TestCase):
    def test_built_only_under_end_state_with_registry_bounds(self):
        plain = make_explicit()
        for name in ("reward_net", "target_reward", "reward_optimizer", "reward_records",
                     "_reward_log_generator"):
            self.assertIsNone(getattr(plain, name), name)
        for name in ("reward", "target_reward"):
            self.assertIsNone(plain._network_modules()[name])
        self.assertIsNone(plain._optimizers()["reward"])
        self.assertNotIn("reward_gradient_steps", plain.counters())
        agent = end_state_agent("explicit")
        self.assertIsInstance(agent.reward_net, RewardRate)
        self.assertIsInstance(agent.target_reward, RewardRate)
        self.assertIsNot(agent.target_reward, agent.reward_net)
        self.assertTrue(same(agent.target_reward, snapshot(agent.reward_net)))
        self.assertTrue(all(p.requires_grad for p in agent.reward_net.parameters()))
        self.assertFalse(any(p.requires_grad for p in agent.target_reward.parameters()))
        self.assertIs(agent._network_modules()["reward"], agent.reward_net)
        self.assertIs(agent._network_modules()["target_reward"], agent.target_reward)
        self.assertIs(agent._optimizers()["reward"], agent.reward_optimizer)
        self.assertEqual(agent.counters()["reward_gradient_steps"], 0)
        self.assertEqual(agent.reward_records, [])
        # The windy pendulum declares no reward range: a linear head.
        self.assertIsNone(agent.reward_net.bounds)
        # The implicit agent's network: the same builder and seed (both configs' seed 3).
        implicit = end_state_agent("implicit")
        self.assertTrue(same(agent.reward_net, snapshot(implicit.reward_net)))
        # A testbed that declares its range (pendulum-swingup) gets the sigmoid head.
        bounded, run = run_agent("pendulum-swingup", "explicit_generator_torque")
        testbed = run["protocol"]["matched"]["testbed_config"]
        declared = testbed_spec(testbed["name"]).reward_rate_bounds
        self.assertIsNotNone(declared)
        expected = tuple(float(value) for value in declared(testbed))
        self.assertEqual(bounded.reward_net.bounds, expected)
        self.assertEqual(bounded.target_reward.bounds, expected)

    def test_other_initializations_unchanged(self):
        for continuation in ("generator", "rollout"):
            with self.subTest(continuation=continuation):
                plain = make_explicit(explicit_config(continuation=continuation))
                plain_streams = streams(plain)
                keyed = end_state_agent("explicit", continuation=continuation)
                self.assertEqual(streams(keyed), plain_streams)
                for name in ("actor", "critics", "target_critics", "value", "target_value",
                             "online_plant", "target_plant"):
                    self.assertTrue(
                        same(keyed._network_modules()[name], snapshot(plain._network_modules()[name])),
                        name,
                    )
                states = torch.zeros(3, 2)
                np.testing.assert_array_equal(
                    plain.predict(states[0].numpy()), keyed.predict(states[0].numpy())
                )

    def test_one_reward_step_per_train_step_on_the_shared_batch(self):
        agent = end_state_agent("explicit")
        fill(agent, 20)
        sampled, calls = [], []
        real_sample = agent.replay.sample

        def sample(partition, size):
            batch = real_sample(partition, size)
            sampled.append(batch)
            return batch

        agent.replay.sample = sample
        for name in ("update_plant", "update_reward", "update_value"):
            real = getattr(agent, name)

            def recording(batch, _name=name, _real=real):
                calls.append((_name, batch))
                return _real(batch)

            setattr(agent, name, recording)
        agent.train(3, 8)
        self.assertEqual(len(sampled), 3)
        self.assertEqual(
            [name for name, _ in calls], ["update_plant", "update_reward", "update_value"] * 3
        )
        for index, (_, batch) in enumerate(calls):
            self.assertIs(batch, sampled[index // 3])
        counters = agent.counters()
        self.assertEqual(counters["reward_gradient_steps"], 3)
        self.assertEqual(counters["critic_gradient_steps"], 3)
        # Through learn: one reward step per train step, from the first full batch on.
        agent = end_state_agent("explicit")
        agent.learn(30)
        counters = agent.counters()
        self.assertEqual(counters["critic_gradient_steps"], 30 - 8 + 1)
        self.assertEqual(counters["reward_gradient_steps"], counters["critic_gradient_steps"])

    def test_target_reward_follows_by_polyak(self):
        agent = end_state_agent("explicit")
        fill(agent, 20)
        target = snapshot(agent.target_reward)
        agent.update_reward(agent.replay.sample("train", 8))
        online = snapshot(agent.reward_net)
        self.assertFalse(same(agent.reward_net, target))
        agent.update_targets()
        tau = agent.polyak_rate
        for moved, before, fast in zip(agent.target_reward.parameters(), target, online):
            torch.testing.assert_close(moved, (1.0 - tau) * before + tau * fast)
        # The critic target reads the target copy, not the online network.
        batch = agent.replay.sample("train", 8)
        expected = agent.build_critic_target(batch)
        with torch.no_grad():
            for parameter in agent.reward_net.parameters():
                parameter.fill_(float("nan"))
        self.assertTrue(torch.equal(agent.build_critic_target(batch), expected))

    def test_counters_and_records_only_under_end_state(self):
        plain = make_explicit()
        plain.learn(30)
        self.assertEqual(set(plain.counters()), set(TrainingCounters.__required_keys__))
        self.assertIsNone(plain.reward_records)
        state = plain.state_dict(include_replay=False)
        self.assertEqual(set(state["loop"]), {"next_clock"})
        self.assertFalse({"reward", "target_reward"} & set(state["networks"]))
        self.assertNotIn("reward", state["optimizers"])
        agent = end_state_agent("explicit")
        agent.reward_diagnostics_every = 5
        agent.learn(30)
        counters = agent.counters()
        self.assertEqual(set(counters), set(ImplicitTrainingCounters.__required_keys__))
        steps = int(counters["reward_gradient_steps"])
        self.assertEqual(steps, counters["critic_gradient_steps"])
        self.assertEqual([r["steps"] for r in agent.reward_records], list(range(5, steps + 1, 5)))
        state = agent.state_dict(include_replay=False)
        self.assertEqual(
            set(state["loop"]),
            {"next_clock", "reward_records", "last_reward_fit_loss", "reward_diagnostic_rng"},
        )
        self.assertEqual(state["loop"]["reward_records"], agent.reward_records)
        self.assertLessEqual({"reward", "target_reward"}, set(state["networks"]))
        self.assertIn("reward", state["optimizers"])
        self.assertEqual(state["config"]["reward_observation"], "end_state")

    def test_checkpoint_round_trip_and_mode_guard(self):
        reference = end_state_agent("explicit")
        reference.reward_diagnostics_every = 4
        reference.learn(30)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            reference.save(path)
            restored = end_state_agent("explicit")
            restored.reward_diagnostics_every = 4
            restored.load(path)
            self.assertIs(restored.replay_restored, True)
            for p, q in zip(all_parameters(reference), all_parameters(restored)):
                self.assertTrue(torch.equal(p, q))
            optimizers = restored._optimizers()
            for name, optimizer in reference._optimizers().items():
                if optimizer is not None:
                    assert_same_state(
                        self, optimizer.state_dict(), optimizers[name].state_dict(), name
                    )
            counts, other = reference.counters(), restored.counters()
            counts.pop("wall_seconds"), other.pop("wall_seconds")
            self.assertEqual(counts, other)
            self.assertEqual(restored.reward_records, reference.reward_records)
            self.assertEqual(restored.last_reward_fit_loss, reference.last_reward_fit_loss)
            self.assertTrue(
                torch.equal(
                    restored._reward_log_generator.get_state(),
                    reference._reward_log_generator.get_state(),
                )
            )
            # Both continue identically (the global stream is the process's, so each
            # continuation starts from the saved one).
            global_state = torch.get_rng_state()
            reference.learn(40)
            torch.set_rng_state(global_state)
            restored.learn(40)
            for p, q in zip(all_parameters(reference), all_parameters(restored)):
                self.assertTrue(torch.equal(p, q))
            self.assertEqual(restored.reward_records, reference.reward_records)
            # A checkpoint of the other mode is refused, before anything is loaded.
            plain = make_explicit()
            before = all_parameters(plain)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                plain.load(path)
            for p, q in zip(before, all_parameters(plain)):
                self.assertTrue(torch.equal(p, q))
            plain_path = Path(folder) / "plain.pt"
            make_explicit().save(plain_path)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                end_state_agent("explicit").load(plain_path)
            implicit = make_implicit()
            implicit_path = Path(folder) / "implicit.pt"
            end_state_agent("implicit").save(implicit_path)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                implicit.load(implicit_path)


# --------------------------------------------------------------------------- quadratures


class TestQuadratures(unittest.TestCase):
    def test_generator_target_is_the_first_order_rectangle(self):
        agent = end_state_agent("explicit")
        fill(agent, 40)
        batch = agent.replay.sample("train", 16)
        self.assertNotIn("rewards", batch)
        target = agent.build_critic_target(batch)
        states, decisions, durations = batch["states"], batch["decisions"], batch["durations"]
        lam = agent.discount_rate
        with torch.no_grad():
            rewards = durations * agent.target_reward(states, start_torques(agent, batch))
        rewards = rewards + charges(agent, batch)
        continuation = generator_continuation(
            agent.target_plant, agent.target_value, agent.controller, states, decisions,
            durations, discount_rate=lam,
        )
        expected = rewards + (1.0 - batch["terminated"]) * continuation.detach()
        torch.testing.assert_close(target, expected)
        # y - V = Delta (r_hat + G V - lambda V) on rows that did not fail: Delta times the
        # generator rate target of the same value, plant and rate.
        live = batch["terminated"][:, 0] < 0.5
        self.assertTrue(bool(live.any()))
        rate = generator_rate_target(
            agent.target_plant, agent.target_value, agent.controller, states, decisions,
            agent.target_reward, discount_rate=lam,
        ).detach()
        with torch.no_grad():
            values = agent.target_value(states)
        torch.testing.assert_close(
            (target - values)[live], (durations * rate)[live], rtol=1e-4, atol=1e-5
        )
        # The implicit critic head's target: the same rectangle of the frozen reward copy.
        critic = end_state_agent("implicit", decision_head="critic", value_operator="model")
        fill(critic, 40)
        critic.freeze_operator()
        with mock.patch.object(
            implicit_module, "generator_critic_target", wraps=implicit_module.generator_critic_target
        ) as spy:
            critic.fit_critic(1)
        args, kwargs = spy.call_args
        used = args[3]
        with torch.no_grad():
            rectangle = used["durations"] * critic._reward_rate(
                used["states"], start_torques(critic, used)
            )
        torch.testing.assert_close(kwargs["rewards"], rectangle + charges(critic, used))

    def test_rollout_target_integrates_the_learned_rate_on_the_path_knots(self):
        agent = end_state_agent("explicit", continuation="rollout")
        fill(agent, 40)
        batch = agent.replay.sample("train", 12)
        rollout_state = copy.deepcopy(agent._rollout_rng.bit_generator.state)
        target = agent.build_critic_target(batch)
        agent._rollout_rng.bit_generator.state = rollout_state
        seed = agent._rollout_seed()
        noise = sample_brownian_increments(
            batch["durations"],
            path_count=agent.rollout_paths,
            noise_dim=agent.target_plant.noise_dim,
            maximum_step=agent.model_step,
            physics_dt=agent.physics_dt,
            controller_refresh_steps=agent.refresh_steps,
            seed=seed,
        )
        grid = dict(
            maximum_step=agent.model_step,
            physics_dt=agent.physics_dt,
            controller_refresh_steps=agent.refresh_steps,
            path_count=agent.rollout_paths,
            brownian_increments=noise,
            integrator=agent.model_integrator,
            contact=agent.contact,
        )
        terms = {
            "reward_rate": agent.target_reward,
            "terminal_payoff": lambda s: terminal_payoff_tensor(s, agent.testbed),
            "discount_rate": agent.discount_rate,
        }
        args = (
            agent.target_plant, agent.controller, batch["states"], batch["decisions"],
            batch["durations"], agent._failure_predicate,
        )
        paths = stochastic_rollout(*args, return_terms=terms, **grid)
        # The return terms move no path: the endpoints of the paths without them.
        plain = stochastic_rollout(*args, **grid)
        self.assertTrue(torch.equal(paths["endpoints"], plain["endpoints"]))
        self.assertTrue(torch.equal(paths["survived"], plain["survived"]))
        running = paths["running_rewards"].double()
        survived = paths["survived"].double()
        counts = survived.sum(dim=1)
        mean = (running * survived).sum(dim=1) / counts.clamp_min(1.0)
        observed = trapezoids(agent, batch, agent.target_reward) - charges(agent, batch)
        rewards = torch.where(counts > 0.0, mean, observed.double()).to(target.dtype)
        rewards = rewards + charges(agent, batch)
        continuation = survivor_continuation(
            agent.target_value, paths, batch["next_states"], batch["durations"], agent.discount_rate
        )
        expected = rewards + (1.0 - batch["terminated"]) * continuation
        torch.testing.assert_close(target, expected)
        # One path's running reward is the discounted trapezoid of the rate on its knots:
        # with a single substep (the first knot at s, the second at the endpoint) it is the
        # observed-ends rule of the path's own endpoint.
        row = {key: value[:1] for key, value in batch.items()}
        row["durations"] = torch.full((1, 1), agent.model_step)
        single_noise = torch.zeros(1, 1, 1, agent.target_plant.noise_dim)
        single = stochastic_rollout(
            agent.target_plant, agent.controller, row["states"], row["decisions"],
            row["durations"], agent._failure_predicate, maximum_step=agent.model_step,
            physics_dt=agent.physics_dt, controller_refresh_steps=agent.refresh_steps,
            path_count=1, brownian_increments=single_noise, return_terms=terms,
            integrator=agent.model_integrator,
        )
        if bool(single["survived"].all()):
            row["next_states"] = single["endpoints"][:, 0]
            with torch.no_grad():
                expected_single = 0.5 * agent.model_step * (
                    agent.target_reward(row["states"], start_torques(agent, row))
                    + math.exp(-agent.discount_rate * agent.model_step)
                    * agent.target_reward(row["next_states"], end_torques(agent, row))
                )
            torch.testing.assert_close(
                single["running_rewards"][:, 0].to(expected_single.dtype), expected_single,
                rtol=1e-5, atol=1e-7,
            )

    def test_rollout_reward_uses_survivor_weights_and_fallback(self):
        paths = {
            "running_rewards": torch.tensor(
                [[[1.0], [3.0], [5.0]], [[2.0], [4.0], [6.0]], [[7.0], [8.0], [9.0]],
                 [[1.0], [1.0], [1.0]]],
                dtype=torch.float64,
            ),
            "survived": torch.tensor(
                [[[True], [False], [True]], [[False], [False], [False]],
                 [[True], [True], [True]], [[True], [True], [True]]]
            ),
        }
        observed = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        result = rollout_interval_reward(paths, observed)
        self.assertEqual(result.shape, (4, 1))
        self.assertEqual(result.dtype, observed.dtype)
        # Row 0: the survivors' mean; row 1: no survivor, the observed fallback.
        torch.testing.assert_close(result, torch.tensor([[3.0], [20.0], [8.0], [1.0]]))
        # A truncated path (contact exclusion) takes the observed value in its place.
        paths["truncated"] = torch.tensor(
            [[[False], [False], [False]], [[False], [False], [False]],
             [[False], [True], [False]], [[True], [True], [True]]]
        )
        result = rollout_interval_reward(paths, observed)
        torch.testing.assert_close(
            result, torch.tensor([[3.0], [20.0], [(7.0 + 30.0 + 9.0) / 3.0], [40.0]])
        )
        # Forced failure on every path: the agent's rollout reward is the observed-ends
        # trapezoid plus the charge. From 1.565 rad at 12.4 rad/s the angle passes the
        # windy limit pi/2 within the first 1 ms substep.
        agent = end_state_agent("explicit", continuation="rollout")
        fill(agent, 10)
        batch = {key: value.clone() for key, value in agent.replay.sample("train", 4).items()}
        batch["states"][:, 0] = 1.565
        batch["states"][:, 1] = 12.4
        noise = sample_brownian_increments(
            batch["durations"], path_count=3, noise_dim=agent.target_plant.noise_dim,
            maximum_step=agent.model_step, physics_dt=agent.physics_dt,
            controller_refresh_steps=agent.refresh_steps, seed=5,
        )
        paths = stochastic_rollout(
            agent.target_plant, agent.controller, batch["states"], batch["decisions"],
            batch["durations"], agent._failure_predicate, maximum_step=agent.model_step,
            physics_dt=agent.physics_dt, controller_refresh_steps=agent.refresh_steps,
            path_count=3, brownian_increments=noise,
            return_terms=agent._reward_return_terms(agent.target_reward),
            integrator=agent.model_integrator,
        )
        self.assertFalse(bool(paths["survived"].any()))
        torch.testing.assert_close(
            agent._rollout_rewards(batch, paths, agent.target_reward),
            agent._observed_end_rewards(batch, agent.target_reward),
        )

    def test_data_target_is_the_observed_end_trapezoid(self):
        agent = end_state_agent("implicit")
        fill(agent, 40)
        agent.freeze_operator()
        batch = agent.replay.sample("train", 16)
        model_batch = agent.replay.sample("train", 8)
        targets = agent._rate_targets(
            batch, model_batch["states"], model_batch["decisions"], model_batch=model_batch
        )
        lam = agent.discount_rate
        rewards = trapezoids(agent, batch, agent._reward_rate)
        with torch.no_grad():
            continuation = torch.exp(-lam * batch["durations"]) * agent.target_value(
                batch["next_states"]
            )
            expected = (
                rewards + (1.0 - batch["terminated"]) * continuation
                - agent.target_value(batch["states"])
            ) / batch["durations"]
        torch.testing.assert_close(targets["data"], expected)
        # implicit_no_rollout_rate (rollout weight 0): the data target serves as both.
        rate = dict(implicit_config()["rate"], rollout_weight=0.0)
        zero = end_state_agent("implicit", rate=rate)
        fill(zero, 40)
        zero.freeze_operator()
        batch = zero.replay.sample("train", 16)
        targets = zero._rate_targets(batch, batch["states"], batch["decisions"])
        self.assertIs(targets["rollout"], targets["data"])
        expected = data_rate_target(
            zero.target_value, batch, discount_rate=zero.discount_rate,
            rewards=trapezoids(zero, batch, zero._reward_rate),
        )
        torch.testing.assert_close(targets["data"], expected)

    def test_flagged_rows_use_the_observed_end_trapezoid(self):
        rows = (0, 2, 5, 9)
        # The explicit generator and rollout targets: r_obs + c + (1-d) e^(-lambda Delta)
        # Vbar(s') at flagged rows.
        for continuation in ("generator", "rollout"):
            with self.subTest(continuation=continuation):
                agent = exclusion_agent("explicit", continuation=continuation)
                self.assertTrue(agent.exclusion_active)
                fill(agent, 30)
                batch, flags = flagged(agent.replay.sample("train", 12), rows)
                target = agent.build_critic_target(batch)
                observed = trapezoids(agent, batch, agent.target_reward) + (
                    1.0 - batch["terminated"]
                ) * observed_continuation(agent.target_value, batch, discount_rate=agent.discount_rate)
                torch.testing.assert_close(target[flags], observed[flags])
                if continuation == "generator":
                    with torch.no_grad():
                        rectangle = batch["durations"] * agent.target_reward(
                            batch["states"], start_torques(agent, batch)
                        )
                    generated = generator_continuation(
                        agent.target_plant, agent.target_value, agent.controller,
                        batch["states"], batch["decisions"], batch["durations"],
                        discount_rate=agent.discount_rate,
                    ).detach()
                    expected = rectangle + charges(agent, batch) + (
                        1.0 - batch["terminated"]
                    ) * generated
                    torch.testing.assert_close(target[~flags], expected[~flags])
        # The implicit rate targets: the data target with r_obs + c, which flagged rows'
        # rollout target and flagged model points' generator target take.
        agent = exclusion_agent("implicit")
        fill(agent, 40)
        operator = agent.freeze_operator()
        batch, flags = flagged(agent.replay.sample("train", 12), rows)
        model_batch, model_flags = flagged(agent.replay.sample("train", 8), (1, 4))
        targets = agent._rate_targets(
            batch, model_batch["states"], model_batch["decisions"], model_batch=model_batch
        )
        lam = agent.discount_rate
        data = data_rate_target(
            agent.target_value, batch, discount_rate=lam,
            rewards=trapezoids(agent, batch, agent._reward_rate),
        )
        torch.testing.assert_close(targets["data"], data)
        torch.testing.assert_close(targets["rollout"][flags], data[flags])
        model_data = data_rate_target(
            agent.target_value, model_batch, discount_rate=lam,
            rewards=trapezoids(agent, model_batch, agent._reward_rate),
        )
        torch.testing.assert_close(targets["generator"][model_flags], model_data[model_flags])
        # The value step's observed residual at flagged rows reads r_obs + c of the
        # operator's frozen reward copy.
        with mock.patch.object(
            implicit_module, "excluded_resolvent_residuals",
            wraps=implicit_module.excluded_resolvent_residuals,
        ) as spy:
            agent.update_value(batch, operator)
        rewards = spy.call_args.kwargs["rewards"]
        torch.testing.assert_close(rewards, trapezoids(agent, batch, operator["reward_rate"]))
        index = torch.nonzero(flags).squeeze(-1)
        with torch.no_grad():
            sample = agent._policy_sample(
                batch["states"], sample_count=int(agent.solver["policy_draws"]),
                actor=operator["policy"],
            )
        draws = {key: value.index_select(0, index) for key, value in sample.items()}
        selected = {key: value.index_select(0, index) for key, value in batch.items()}
        direct = observed_interval_residuals(
            agent.value, operator, selected, draws, create_graph=False,
            rewards=rewards.index_select(0, index),
        )
        as_column = dict(selected, rewards=rewards.index_select(0, index))
        torch.testing.assert_close(
            direct,
            observed_interval_residuals(agent.value, operator, as_column, draws, create_graph=False),
        )
        combined = excluded_resolvent_residuals(
            agent.value, operator, agent.controller, batch, sample, flags,
            create_graph=False, rewards=rewards,
        )
        torch.testing.assert_close(combined.index_select(0, index), direct)
        # The critic head's target: r_obs + c at flagged rows, the rectangle + c elsewhere.
        critic = exclusion_agent("implicit", decision_head="critic", value_operator="model")
        fill(critic, 40)
        critic.freeze_operator()
        batch, flags = flagged(critic.replay.sample("train", 12), rows)
        rewards = critic._generator_rewards(batch, critic._reward_rate, flags)
        torch.testing.assert_close(
            rewards[flags], trapezoids(critic, batch, critic._reward_rate)[flags]
        )
        with torch.no_grad():
            rectangle = batch["durations"] * critic._reward_rate(
                batch["states"], start_torques(critic, batch)
            )
        torch.testing.assert_close(
            rewards[~flags], (rectangle + charges(critic, batch))[~flags]
        )
        with mock.patch.object(
            implicit_module, "generator_critic_target", wraps=implicit_module.generator_critic_target
        ) as spy:
            critic.fit_critic(1)
        args, kwargs = spy.call_args
        torch.testing.assert_close(
            kwargs["rewards"], critic._generator_rewards(args[3], critic._reward_rate, kwargs["flags"])
        )

    def test_gate_observed_return_uses_the_learned_trapezoid(self):
        agent = end_state_agent("implicit")
        self.assertEqual(agent.admission["plant_update"], "admission")
        fill(agent, 40)
        agent.freeze_operator()
        batch = agent.replay.sample("train", 6)
        with mock.patch.object(
            implicit_module, "paired_gate_samples", wraps=implicit_module.paired_gate_samples
        ) as spy:
            samples = agent._gate_samples(batch, 2)
        kwargs = spy.call_args.kwargs
        rewards = trapezoids(agent, batch, agent._reward_rate)
        torch.testing.assert_close(kwargs["observed_rewards"], rewards)
        with torch.no_grad():
            expected = rewards.double() + torch.exp(
                -agent.discount_rate * batch["horizons"].double()
            ) * (1.0 - batch["terminated"].double()) * agent.target_value(
                batch["next_states"]
            ).double()
        torch.testing.assert_close(samples["observed_returns"], expected)
        # The plants' paths integrate the same frozen rate.
        self.assertEqual(kwargs["return_terms"]["reward_rate"], agent._reward_rate)
        # The gate reads no interval reward: the batch has none.
        self.assertNotIn("rewards", batch)

    def test_failure_charge_is_the_task_payoff(self):
        agent = end_state_agent("explicit")
        lam = agent.discount_rate
        batch = {
            "states": torch.zeros(3, 2),
            "decisions": torch.tensor([[0.0, 10.0, 1.0]] * 3),
            "durations": torch.tensor([[0.004], [0.01], [0.018]]),
            "next_states": torch.tensor([[1.6, 0.0], [0.1, 0.0], [0.2, 13.0]]),
            "terminated": torch.tensor([[1.0], [0.0], [1.0]]),
            "truncated": torch.zeros(3, 1),
        }
        payoff = -1.0 / lam
        expected = torch.tensor(
            [[math.exp(-lam * 0.004) * payoff], [0.0], [math.exp(-lam * 0.018) * payoff]]
        )
        torch.testing.assert_close(agent._failure_charges(batch), expected)
        self.assertEqual(float(agent._failure_charges(batch)[1, 0]), 0.0)
        torch.testing.assert_close(
            failure_charge(batch, lambda s: terminal_payoff_tensor(s, agent.testbed), discount_rate=lam),
            expected,
        )
        # It is the charge the environment adds to a failed interval's reward.
        env = make_env(windy(), seed=0)
        rates, times = np.array([0.1, -0.2, -0.3]), np.array([0.0, 0.001, 0.002])
        failing = np.array([1.6, 0.0])
        added = env.integrate_interval_reward(
            rates, times, terminated=True, terminal_state=failing
        ) - env.integrate_interval_reward(rates, times, terminated=False, terminal_state=failing)
        row = {
            "durations": torch.tensor([[0.002]], dtype=torch.float64),
            "next_states": torch.as_tensor(failing[None], dtype=torch.float64),
            "terminated": torch.ones(1, 1, dtype=torch.float64),
        }
        charge = failure_charge(
            row, lambda s: terminal_payoff_tensor(s, env.config), discount_rate=lam
        )
        self.assertAlmostEqual(float(charge[0, 0]), added, places=12)
        # Every end-state reward adds it once.
        start = start_rate_reward(agent.controller, batch, agent.target_reward)
        torch.testing.assert_close(
            agent._generator_rewards(batch, agent.target_reward) - start, expected
        )
        observed = observed_end_reward(
            agent.controller, batch, agent.target_reward, discount_rate=lam
        )
        torch.testing.assert_close(
            agent._observed_end_rewards(batch, agent.target_reward) - observed, expected
        )

    def test_quadrature_orders_against_the_simulator(self):
        """lqr-lqr_2_1 with the known plant and r_hat the task's own rate: against the
        simulator's interval reward (discounted trapezoids on the 1 ms physics knots), the
        rectangle's error grows as Delta^2 (its first-order bias Delta^2 (rho' - lambda
        rho) / 2), while the observed-ends trapezoid and the model paths' trapezoids on the
        same 1 ms knots are far more accurate."""
        with float64_default():
            env = make_env(row_testbed("lqr-lqr_2_1"), decision_mode="torque", seed=0)
            oracle = make_oracle_plant(env)
            config = env.config
            lam = float(env.discount_rate)

            def rate(states, torques):
                return reward_rate_tensor(states, torques, config)

            terms = {
                "reward_rate": rate,
                "terminal_payoff": lambda s: terminal_payoff_tensor(s, config),
                "discount_rate": lam,
            }
            refresh = grid_refresh_steps(env.controller, int(config["controller"]["refresh_steps"]))
            start = np.array([0.6, 0.0, 2.0, 0.0])
            torque = np.array([5.0])
            errors = {"rectangle": [], "trapezoid": [], "rollout": []}
            for steps in (4, 8, 16, 32):
                env.reset(seed=0, options={"state": start})
                transition = env.step_dt(torque, steps * env.physics_dt)
                self.assertEqual(env.last_interval["physics_steps"], steps)
                self.assertFalse(transition["terminated"])
                batch = {
                    "states": torch.as_tensor(transition["state"][None]),
                    "decisions": torch.as_tensor(transition["decision"][None]),
                    "durations": torch.tensor([[float(transition["duration"])]]),
                    "next_states": torch.as_tensor(transition["next_state"][None]),
                    "terminated": torch.zeros(1, 1),
                }
                simulated = float(transition["reward"])
                rectangle = float(start_rate_reward(env.controller, batch, rate))
                observed = observed_end_reward(env.controller, batch, rate, discount_rate=lam)
                noise = sample_brownian_increments(
                    batch["durations"], path_count=2, noise_dim=oracle.noise_dim,
                    maximum_step=env.physics_dt, physics_dt=env.physics_dt,
                    controller_refresh_steps=refresh, seed=0,
                )
                paths = stochastic_rollout(
                    oracle, env.controller, batch["states"], batch["decisions"],
                    batch["durations"], lambda s: failure_mask_tensor(s, config),
                    maximum_step=env.physics_dt, physics_dt=env.physics_dt,
                    controller_refresh_steps=refresh, path_count=2, brownian_increments=noise,
                    return_terms=terms, integrator="rk4",
                )
                rollout = float(rollout_interval_reward(paths, observed))
                errors["rectangle"].append(abs(rectangle - simulated))
                errors["trapezoid"].append(abs(float(observed) - simulated))
                errors["rollout"].append(abs(rollout - simulated))
            rectangles = errors["rectangle"]
            for small, large in zip(rectangles, rectangles[1:]):
                self.assertGreater(large / small, 3.0)
                self.assertLess(large / small, 5.0)
            for rectangle, trapezoid, rollout in zip(
                rectangles, errors["trapezoid"], errors["rollout"]
            ):
                self.assertLess(trapezoid, 0.1 * rectangle)
                self.assertLess(rollout, 1e-2 * rectangle)


# ------------------------------------------------------------- no learner reads the reward


# Every CT-PH-SAC update path: the explicit generator, rollout and td targets (td is
# explicit_generator_torque's config with the td continuation), the implicit arm, both
# critic arms, both explicit-step arms, a gated arm and the arm without rollout rates.
POISON_ARMS = (
    "explicit_generator_torque", "explicit_rollout_torque", "explicit_td",
    "implicit_torque", "implicit_critic_head", "implicit_critic", "implicit_explicit_step",
    "implicit_explicit_step_interval", "implicit_gated_admissions", "implicit_no_rollout_rate",
)


def poison(env):
    """Make every interval reward the environment returns NaN (its physics untouched)."""
    step_dt = env.step_dt

    def poisoned(decision, duration):
        transition = step_dt(decision, duration)
        transition["reward"] = float("nan")
        return transition

    env.step_dt = poisoned


def trained(env_id, arm, budget, *, poisoned, protocol=END_STATE):
    """The agent, its state_dict and parameters after learn(budget) on a tiny run of the row
    (a warm-up of 40 decisions); built and trained before the next agent is built, so each
    starts from its own seeded global stream."""
    source = "explicit_generator_torque" if arm == "explicit_td" else arm
    run = tiny_run(
        env_id, source, "/nonexistent", budget, warmup_decisions=40, protocol=protocol
    )
    testbed = run["protocol"]["matched"]["testbed_config"]
    env = make_env(testbed, decision_mode=run["arm"]["decision_mode"], seed=0)
    if poisoned:
        poison(env)
    if arm == "explicit_td":
        torch.manual_seed(run["training_seed"])
        np.random.seed(run["training_seed"])
        config = P._explicit_arm_config(run)
        config["continuation"] = "td"
        agent = ExplicitCTPHSAC(env, config)
    else:
        agent = P.make_training_agent(run, env)
    agent.learn(budget)
    state = agent.state_dict()
    parameters = all_parameters(agent)
    env.close()
    return agent, state, parameters


def without_interval_rewards(state):
    """A state_dict without the replay's stored interval rewards."""
    state = dict(state)
    replay = copy.deepcopy(state["replay"])
    for buffer in replay["buffers"].values():
        if buffer["arrays"] is not None:
            buffer["arrays"].pop("rewards")
    state["replay"] = replay
    return state


def finite(parameters):
    return all(bool(torch.isfinite(p).all()) for p in parameters)


class TestNoLearnerReadsTheIntervalReward(unittest.TestCase):
    def test_poisoned_interval_rewards_change_nothing(self):
        """Under "end_state" the interval rewards (all NaN in the poisoned run) reach no
        learner: with and without contact exclusion (cartpole-swingup, reacher-easy), every
        arm's state after training is the clean run's bitwise, but for the stored rewards."""
        for env_id in ("cartpole-swingup", "reacher-easy"):
            for arm in POISON_ARMS:
                with self.subTest(testbed=env_id, arm=arm):
                    clean_agent, clean, _ = trained(env_id, arm, 60, poisoned=False)
                    dirty_agent, dirty, parameters = trained(env_id, arm, 60, poisoned=True)
                    self.assertEqual(clean_agent.exclusion_active, env_id == "cartpole-swingup")
                    self.assertTrue(dirty_agent.end_state)
                    self.assertGreater(clean["counters"]["reward_gradient_steps"], 0)
                    self.assertTrue(finite(parameters))
                    train = dirty_agent.replay.buffers["train"]
                    self.assertTrue(np.isnan(train._arrays["rewards"][: train.size()]).all())
                    self.assertEqual(
                        differences(without_interval_rewards(clean), without_interval_rewards(dirty)),
                        [],
                    )

    def test_the_poison_shows_under_interval(self):
        for arm in ("explicit_generator_torque", "implicit_torque"):
            with self.subTest(arm=arm):
                _, clean, _ = trained("reacher-easy", arm, 60, poisoned=False, protocol=None)
                try:
                    _, dirty, parameters = trained(
                        "reacher-easy", arm, 60, poisoned=True, protocol=None
                    )
                except ValueError as error:
                    # The NaN rewards made the networks NaN; their next decision is NaN and
                    # the environment refuses it before learn returns: the poison showed.
                    self.assertIn("Decision must be finite", str(error))
                else:
                    self.assertFalse(finite(parameters))
                    self.assertNotEqual(
                        differences(
                            without_interval_rewards(clean), without_interval_rewards(dirty)
                        ),
                        [],
                    )

    def test_end_state_batches_hold_no_interval_reward(self):
        for kind in ("explicit", "implicit"):
            with self.subTest(kind=kind):
                agent = end_state_agent(kind)
                transitions = spy_transitions(agent)
                fill(agent, 30)
                self.assertEqual(len(transitions), 30)
                for transition in transitions:
                    self.assertEqual(
                        transition["end_reward_rate"], agent.env.end_reward_rate(transition)
                    )
                stored = np.concatenate(
                    [
                        buffer._arrays["end_rewards"][: buffer.size(), 0]
                        for buffer in agent.replay.buffers.values()
                        if buffer._arrays is not None
                    ]
                )
                np.testing.assert_array_equal(
                    np.sort(stored), np.sort([t["end_reward_rate"] for t in transitions])
                )
                if kind == "explicit":
                    train = agent.replay.buffers["train"]
                    np.testing.assert_array_equal(
                        train._arrays["end_rewards"][:30, 0],
                        [t["end_reward_rate"] for t in transitions],
                    )
                    np.testing.assert_array_equal(
                        train._arrays["rewards"][:30, 0], [t["reward"] for t in transitions]
                    )
                batch = agent.replay.sample("train", 8)
                self.assertNotIn("rewards", batch)
                self.assertEqual(batch["end_rewards"].shape, (8, 1))


# ------------------------------------------------------------------------- unchanged


def run_spied(run):
    """run_seed, with the agent it built and every transition its training env returned."""
    agents, transitions = [], []
    build = P.make_training_agent

    def capture(spec, env):
        step_dt = env.step_dt

        def spied(decision, duration):
            transition = step_dt(decision, duration)
            transitions.append(transition)
            return transition

        env.step_dt = spied
        agents.append(build(spec, env))
        return agents[-1]

    with mock.patch.object(P, "make_training_agent", capture):
        result = P.run_seed(run)
    return result, agents[-1], transitions


class TestUnchanged(unittest.TestCase):
    def test_plant_fit_identical_across_modes(self):
        """On the same uniform-decision replay the plant fit is the same in both modes,
        bitwise, and every random stream stays aligned: the reward draws from none."""
        for continuation in ("generator", "rollout"):
            with self.subTest(continuation=continuation):
                results = []
                for keyed in (False, True):
                    agent = (
                        end_state_agent("explicit", continuation=continuation)
                        if keyed
                        else make_explicit(explicit_config(continuation=continuation))
                    )
                    fill(agent, 30)
                    agent.train(6, 8)
                    rows = {
                        name: array.copy()
                        for name, array in agent.replay.buffers["train"]._arrays.items()
                    }
                    results.append(
                        (snapshot(agent.online_plant), snapshot(agent.target_plant),
                         streams(agent), rows)
                    )
                (plant, target, plain_streams, plain_rows), keyed_result = results
                keyed_plant, keyed_target, keyed_streams, keyed_rows = keyed_result
                for p, q in zip(plant + target, keyed_plant + keyed_target):
                    self.assertTrue(torch.equal(p, q))
                self.assertEqual(plain_streams, keyed_streams)
                self.assertEqual(set(keyed_rows) - set(plain_rows), {"end_rewards"})
                for name, array in plain_rows.items():
                    np.testing.assert_array_equal(array, keyed_rows[name], err_msg=name)
        results = []
        for keyed in (False, True):
            agent = end_state_agent("implicit") if keyed else make_implicit()
            fill(agent, 40)
            for _ in range(5):
                agent.update_online_plant(agent.replay.sample("train", 8))
            results.append((snapshot(agent.online_plant), streams(agent)))
        (plant, plain_streams), (keyed_plant, keyed_streams) = results
        for p, q in zip(plant, keyed_plant):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(plain_streams, keyed_streams)

    def test_evaluation_ignores_the_mode(self):
        for arm in ("implicit_torque", "explicit_rollout_torque", "explicit_generator_torque",
                    "implicit_learned_admissions"):
            with self.subTest(arm=arm):
                records = []
                for keys in (None, END_STATE):
                    run = tiny_run("pendulum-swingup", arm, "/nonexistent", 10, protocol=keys)
                    testbed = run["protocol"]["matched"]["testbed_config"]
                    mode = run["arm"]["decision_mode"]
                    env = make_env(testbed, decision_mode=mode, seed=0)
                    agent = P.make_training_agent(run, env)
                    records.append(
                        P.evaluate_fixed_seeds(
                            agent.predict,
                            lambda seed: make_env(testbed, decision_mode=mode, seed=seed),
                            [3, 4],
                            discount_rate=run["protocol"]["matched"]["discount_rate"],
                            deterministic=run["protocol"]["deterministic_evaluation"],
                        )
                    )
                    env.close()
                self.assertEqual(records[0], records[1])

    def test_default_mode_writes_no_new_key(self):
        for arm in ("implicit_torque", "explicit_generator_torque", "explicit_rollout_torque"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as output:
                run = tiny_run("pendulum-swingup", arm, output, 60)
                result, agent, transitions = run_spied(run)
                self.assertEqual(agent.reward_observation, "interval")
                # Transitions, replay columns and batches.
                self.assertEqual(len(transitions), 60)
                for transition in transitions:
                    self.assertNotIn("end_reward_rate", transition)
                for buffer in agent.replay.buffers.values():
                    if buffer._arrays is not None:
                        self.assertEqual(list(buffer._arrays), list(INTERVAL_ARRAYS))
                self.assertEqual(list(agent.replay.sample("train", 4)), list(INTERVAL_BATCH))
                # Arm config and resume settings.
                self.assertNotIn("reward_observation", agent.config)
                self.assertNotIn("reward_observation", json.dumps(R.run_settings(run)))
                # Records, on disk too.
                self.assertNotIn("reward_observation", result)
                stored = json.loads((P.run_output_directory(run) / "run_record.json").read_text())
                self.assertNotIn("reward_observation", stored["result"])
                state = agent.state_dict(include_replay=False)
                if arm.startswith("explicit"):
                    self.assertNotIn("reward_fit", result)
                    self.assertNotIn("reward_gradient_steps", result["counters"])
                    self.assertEqual(
                        set(state["networks"]),
                        {"actor", "critics", "target_critics", "value", "target_value",
                         "online_plant", "target_plant"},
                    )
                    self.assertNotIn("reward", state["optimizers"])
                    self.assertEqual(set(state["loop"]), {"next_clock"})
                else:
                    self.assertIn("reward_fit", result)
                    self.assertIn("reward_gradient_steps", result["counters"])
                    expected = set(agent._LOOP_FIELDS + agent._LATER_LOOP_FIELDS) | {
                        "gate_rng", "reward_records", "reward_diagnostic_rng",
                    }
                    self.assertEqual(set(state["loop"]), expected)

    def test_every_ct_sac_arm_receives_end_state(self):
        """make_training_agent passes reward_source "end_state" to every CT-SAC arm under
        reward_observation "end_state", exactly as under ct_sac_reward_source "end_state"."""
        names = tuple(arm["name"] for arm in P.selectable_arms() if arm["algorithm"] == "ct_sac")
        self.assertIn("ct_sac_torque", names)
        self.assertIn("ct_sac_implicit", names)
        base = tiny_run("pendulum-swingup", "ct_sac_torque", "/nonexistent", 10)
        captured = {}
        for label, keys in (
            ("plain", {}),
            ("key", END_STATE),
            ("both", {**END_STATE, "ct_sac_reward_source": "end_state"}),
            ("source", {"ct_sac_reward_source": "end_state"}),
        ):
            proto = copy.deepcopy(base["protocol"])
            proto.update(keys)
            proto["arms"] = [arm for arm in P.selectable_arms() if arm["name"] in names]
            P.validate_matched_quantities(proto)
            for run in P.build_run_matrix(proto):
                with mock.patch("algorithms.ph_baseline.TorqueCTSACBaseline") as plain, \
                        mock.patch("algorithms.ct_sac_implicit.ImplicitTorqueCTSACBaseline") as implicit:
                    P.make_training_agent(run, None)
                built = implicit if "ct_sac_implicit_step" in run["arm"] else plain
                captured[label, run["arm"]["name"]] = built.call_args
        for name in names:
            with self.subTest(arm=name):
                args, kwargs = captured["plain", name]
                self.assertNotIn("reward_source", kwargs)
                for label in ("key", "both", "source"):
                    keyed_args, keyed_kwargs = captured[label, name]
                    self.assertEqual(keyed_args, args)
                    self.assertEqual(keyed_kwargs, dict(kwargs, reward_source="end_state"))


if __name__ == "__main__":
    unittest.main()
