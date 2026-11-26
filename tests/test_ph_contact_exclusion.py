"""Contact exclusion (config key contact_handling "exclude", since 2026-11-10).

common.ph_config's CONTACT_HANDLINGS block and models/ph_sac/rollouts.ContactExclusion:
on a task with joint limits a replay row is flagged when its start state is within a
reach margin of a limit (start_state_flags) or the environment recorded a limit touch
(limit_touched); flagged rows leave the plant fit, and every model use at them falls back
to the observed transition; model paths that come near a limit are truncated.

1. Off: "port" is the absent key, bitwise; a rollout without truncation carries no
   truncated column and equals the plain one.
2. Flags: the start-state rule exactly (margin scaling with velocity, interval and the
   acceleration allowance; numpy and torch agree); the environment's limit_touched; the
   agents store both in the replay in either mode; the replay stores and restores them
   (and reads older states and files as unflagged).
3. Plant fit: flagged rows are ignored exactly (the step equals the step on the unflagged
   sub-batch); an all-flagged batch takes no step and gives no NaN.
4. Fallback targets on tiny examples, against hand-computed formulas: the implicit value
   residual, the implicit rate-head targets, the explicit generator and rollout critic
   targets; flagged gate rows give D = 0.
5. Truncation: equals a manual truncated rollout (the plain rollout over the clean part).
6. Plumbing: validation, the coarse model step, records, fingerprint, CLI, table cell.
7. Exact restart with exclusion on (cartpole-swingup, tiny, forced contact).
"""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from algorithms.ph_admission import corrected_gate_rows, paired_gate_samples
from algorithms.ph_resolvent import (
    excluded_resolvent_residuals,
    observed_interval_residuals,
    resolvent_residuals,
)
from algorithms.ph_targets import (
    data_rate_target,
    generator_continuation,
)
from benchmarks import ph_hyperparams as table
from benchmarks import ph_plant_schedule_drift as harness
from benchmarks import run_ph_sac
from common.ph_config import (
    EXCLUSION_ACCELERATION_FACTOR,
    contact_exclusion_setting,
    contact_handling_setting,
    contact_substepping_setting,
    exclusion_model_step,
)
from common.ph_replay import PHReplayBuffer
from environment.ph_sac import tasks
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import reporting
from evaluations.ph_sac import resume as R
from models.ph_sac import rollouts
from models.ph_sac.rollouts import (
    batch_contact_flags,
    contact_exclusion,
    contact_flags,
    start_state_flag_numpy,
    start_state_flags,
    stochastic_rollout,
    subset_batch,
)
from tests.test_ph_algorithms import collect, explicit_config, implicit_config, make_explicit, make_implicit
from tests.test_ph_contact_substep import COARSE, FINE, INTERVAL, Float64, cartpole
from tests.test_ph_resume import ExactRestartMixin, run_captured

RAIL = 1.8


def exclude_implicit(**overrides):
    return make_implicit(
        implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="exclude", **overrides)
    )


def exclude_explicit(continuation, **overrides):
    return make_explicit(
        explicit_config(cartpole(), continuation=continuation, model_coarse_step=COARSE,
                        contact_handling="exclude", **overrides)
    )


def flagged(batch, rows):
    """batch with contact_flags set on the given row indices only."""
    batch = dict(batch)
    flags = torch.zeros(batch["states"].shape[0], 1, dtype=batch["states"].dtype)
    flags[list(rows)] = 1.0
    batch["contact_flags"] = flags
    return batch


def transition(i, *, touched=False, near=False, with_columns=True):
    data = {
        "state": np.array([0.1 * i, 0.0, 0.0, 0.0]),
        "decision": np.zeros(3),
        "duration": 0.01,
        "scheduled_duration": 0.01,
        "reward": 0.5,
        "next_state": np.array([0.1 * i + 0.01, 0.0, 0.0, 0.0]),
        "terminated": False,
        "truncated": False,
        "time": 0.01 * i,
        "next_time": 0.01 * (i + 1),
    }
    if with_columns:
        data["limit_touched"] = touched
        data["near_limit"] = near
    return data


# ------------------------------------------------------------------------------ 1. off


class TestOff(unittest.TestCase):
    def test_port_is_the_absent_key(self):
        self.assertEqual(contact_handling_setting({}), "port")
        self.assertEqual(contact_handling_setting({"contact_handling": "port"}), "port")
        with self.assertRaisesRegex(ValueError, "contact_handling"):
            contact_handling_setting({"contact_handling": "drop"})
        plain = make_implicit(implicit_config(cartpole(), model_coarse_step=COARSE))
        keyed = make_implicit(implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="port"))
        for agent in (plain, keyed):
            self.assertFalse(agent.exclusion_active)
            self.assertIsInstance(agent.exclusion, rollouts.ContactExclusion)
            self.assertIsNotNone(agent.contact)
            self.assertEqual(agent.model_step, FINE)
            collect(agent, 30)
        for name in ("train", "gate"):
            a, b = plain.replay.buffers[name], keyed.replay.buffers[name]
            for field in a._arrays:
                np.testing.assert_array_equal(a._arrays[field], b._arrays[field], field)
        batch = plain.replay.sample("train", 8)
        keyed.replay.sample("train", 8)
        self.assertEqual(plain.update_online_plant(batch), keyed.update_online_plant(batch))
        for p, q in zip(plain.online_plant.parameters(), keyed.online_plant.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_rollout_without_truncation_is_plain(self):
        with Float64():
            env = tasks.make_env(cartpole(), seed=0)
            oracle = tasks.make_oracle_plant(env).double()
            states = torch.tensor([[0.0, 0.3, 0.5, 0.0], [1.7, 0.0, 4.0, 0.0]], dtype=torch.float64)
            decisions = torch.tensor([[0.2, 10.0, 2.0], [0.5, 50.0, 5.0]], dtype=torch.float64)
            durations = torch.full((2, 1), INTERVAL, dtype=torch.float64)
            noise = torch.zeros(2, 2, 2, oracle.noise_dim, dtype=torch.float64)
            kwargs = dict(maximum_step=COARSE, physics_dt=FINE, controller_refresh_steps=None,
                          path_count=2, brownian_increments=noise, integrator="rk4")
            fail = lambda s: tasks.failure_mask_tensor(s, env.config)  # noqa: E731
            plain = stochastic_rollout(oracle, env.controller, states, decisions, durations, fail, **kwargs)
            self.assertNotIn("truncated", plain)
            again = stochastic_rollout(oracle, env.controller, states, decisions, durations, fail,
                                       truncation=None, skip_rows=None, **kwargs)
            for key in ("endpoints", "survived", "durations_used", "running_rewards"):
                self.assertTrue(torch.equal(plain[key], again[key]), key)


# ---------------------------------------------------------------------------- 2. flags


class TestFlags(unittest.TestCase):
    def rule(self, accelerations=(10.0, 0.0)):
        return contact_exclusion(cartpole()["state_spec"], np.asarray(accelerations), active=True,
                                 model_step=COARSE)

    def test_start_state_rule_exactly(self):
        rule = self.rule()
        self.assertEqual(rule.positions, (0,))
        self.assertEqual(rule.velocities, (2,))
        self.assertAlmostEqual(rule.margins[0], 0.01 * 2 * RAIL)
        self.assertAlmostEqual(rule.accelerations[0], EXCLUSION_ACCELERATION_FACTOR * 10.0)

        def margin(speed, delta):
            return 0.036 + 1.5 * abs(speed) * delta + 0.5 * 20.0 * delta * delta

        cases = []
        for speed, delta in ((0.0, 0.01), (2.0, 0.01), (1.0, 0.02), (-3.0, 0.03)):
            edge = RAIL - margin(speed, delta)
            for side in (1.0, -1.0):
                cases.append((side * (edge - 1e-6), speed, delta, False))
                cases.append((side * (edge + 1e-6), speed, delta, True))
        cases.append((RAIL + 0.01, 0.0, 0.01, True))
        cases.append((-RAIL - 0.01, 0.0, 0.01, True))
        cases.append((float("nan"), 0.0, 0.01, True))
        for x, speed, delta, expected in cases:
            state = np.array([x, 0.4, speed, -2.0])
            with self.subTest(x=x, speed=speed, delta=delta):
                self.assertEqual(start_state_flag_numpy(rule, state, delta), expected)
                tensor = start_state_flags(rule, torch.as_tensor(state).unsqueeze(0),
                                           torch.tensor([[delta]], dtype=torch.float64))
                self.assertEqual(bool(tensor[0]), expected)

    def test_margin_grows_with_speed_interval_and_acceleration(self):
        state = np.array([1.745, 0.0, 0.0, 0.0])
        rule = self.rule()
        self.assertFalse(start_state_flag_numpy(rule, state, 0.01))
        self.assertTrue(start_state_flag_numpy(rule, np.array([1.745, 0.0, 2.0, 0.0]), 0.01))
        moving = np.array([1.745, 0.0, 1.0, 0.0])
        self.assertFalse(start_state_flag_numpy(rule, moving, 0.01))
        self.assertTrue(start_state_flag_numpy(rule, moving, 0.02))
        self.assertTrue(start_state_flag_numpy(self.rule((400.0, 0.0)), state, 0.01))
        # The speed of another coordinate does not matter.
        self.assertFalse(start_state_flag_numpy(rule, np.array([1.745, 0.0, 0.0, 50.0]), 0.01))

    def test_agent_rule_reads_the_motor_bound(self):
        agent = exclude_implicit()
        bounds = agent.env.motor_acceleration_bounds()
        self.assertEqual(bounds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(bounds)) and bounds[0] > 0.0)
        self.assertAlmostEqual(agent.exclusion.accelerations[0], EXCLUSION_ACCELERATION_FACTOR * bounds[0])
        print("cartpole exclusion rule:", agent.exclusion.record())

    def test_environment_records_limit_touches(self):
        env = tasks.make_env(cartpole(), seed=0)
        rest = np.zeros(3)
        for state, expected in (
            ([0.0, 0.0, 0.0, 0.0], False),
            ([1.79, 0.0, 3.0, 0.0], True),
            ([-1.79, 0.0, -3.0, 0.0], True),
            ([RAIL + 0.01, 0.0, 0.0, 0.0], True),
            ([1.7, 0.0, 0.0, 0.0], False),
        ):
            with self.subTest(state=state):
                env.reset(options={"state": np.array(state)})
                self.assertIs(env.step_dt(rest, INTERVAL)["limit_touched"], expected)
        windy = tasks.make_env(implicit_config()["testbed"], seed=0)
        windy.reset(seed=0)
        self.assertIs(windy.step_dt(np.array([0.0, 1.0, 1.0]), 0.004)["limit_touched"], False)

    def test_agents_store_both_columns_in_either_mode(self):
        for agent in (exclude_implicit(), make_implicit(implicit_config(cartpole()))):
            env = agent.env
            for state in ([1.79, 0.0, 3.0, 0.0], [1.78, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
                agent._state, _ = env.reset(options={"state": np.array(state)})
                agent._first_reset = False
                agent.collect_transition()
            columns = {}
            for name in ("limit_touched", "near_limit"):
                columns[name] = np.concatenate(
                    [b.contact_columns()[name] for b in agent.replay.buffers.values()]
                )
            self.assertEqual(int(columns["limit_touched"].sum()), 1)
            self.assertEqual(int(columns["near_limit"].sum()), 2)

    def test_replay_stores_and_restores_the_flags(self):
        buffer = PHReplayBuffer(20, seed=0)
        for i in range(6):
            buffer.add(transition(i, touched=i in (1, 4), near=i in (2, 4)))
        buffer.add(transition(6, with_columns=False))
        batch = buffer.batch(np.arange(7))
        np.testing.assert_array_equal(batch["contact_flags"].reshape(-1).numpy(), [0, 1, 1, 0, 1, 0, 0])
        restored = PHReplayBuffer(5, seed=1)
        restored.load_state_dict(buffer.state_dict())
        for name in ("limit_touched", "near_limit"):
            np.testing.assert_array_equal(restored._arrays[name], buffer._arrays[name])
        # A state written before the columns existed: no row flagged.
        old = copy.deepcopy(buffer.state_dict())
        for name in ("limit_touched", "near_limit"):
            del old["arrays"][name]
        legacy = PHReplayBuffer(5, seed=1)
        legacy.load_state_dict(old)
        self.assertFalse(legacy._arrays["limit_touched"].any())
        self.assertEqual(float(legacy.batch(np.arange(7))["contact_flags"].sum()), 0.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "replay.npz"
            buffer.save(path)
            loaded = PHReplayBuffer(5, seed=2)
            loaded.load(path)
            np.testing.assert_array_equal(loaded._arrays["near_limit"], buffer._arrays["near_limit"])
        self.assertTrue(torch.equal(batch_contact_flags(batch), batch["contact_flags"].reshape(-1) > 0.5))
        plain = {k: v for k, v in batch.items() if k != "contact_flags"}
        self.assertFalse(batch_contact_flags(plain).any())


# ------------------------------------------------------------------------- 3. plant fit


class TestPlantFit(unittest.TestCase):
    def test_flagged_rows_are_ignored_exactly(self):
        for build in (exclude_implicit, lambda: exclude_explicit("rollout")):
            first, second = build(), build()
            collect(first, 40)
            batch = first.replay.sample("train", 8)
            marked = flagged(batch, (0, 3, 5))
            keep = ~batch_contact_flags(marked)
            sub = {k: v for k, v in subset_batch(marked, keep).items() if k != "contact_flags"}
            update = "update_online_plant" if hasattr(first, "update_online_plant") else "update_plant"
            loss_a = getattr(first, update)(marked)["plant_loss"]
            loss_b = getattr(second, update)(sub)["plant_loss"]
            self.assertEqual(loss_a, loss_b)
            for p, q in zip(first.online_plant.parameters(), second.online_plant.parameters()):
                self.assertTrue(torch.equal(p, q))

    def test_all_flagged_batch_takes_no_step(self):
        agent = exclude_implicit()
        collect(agent, 40)
        batch = flagged(agent.replay.sample("train", 8), range(8))
        before = [p.detach().clone() for p in agent.online_plant.parameters()]
        steps = agent._counters["plant_gradient_steps"]
        metrics = agent.update_online_plant(batch)
        self.assertEqual(metrics, {"plant_loss": 0.0, "plant_rows": 0.0})
        self.assertFalse(math.isnan(metrics["plant_loss"]))
        self.assertEqual(agent._counters["plant_gradient_steps"], steps)
        for p, q in zip(before, agent.online_plant.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_exclusion_runs_the_free_mechanism_at_the_coarse_step(self):
        agent = exclude_implicit()
        self.assertTrue(agent.exclusion_active)
        self.assertIsNone(agent.contact)
        self.assertEqual(agent.model_step, COARSE)
        self.assertFalse(agent.online_plant.limit_port)
        with self.assertRaisesRegex(ValueError, "limit_port"):
            implicit_config(cartpole(), contact_handling="exclude",
                            network={**implicit_config(cartpole())["network"], "limit_port": True})
        with self.assertRaisesRegex(ValueError, "limited_position_ranges"):
            implicit_config(contact_handling="exclude")


# ------------------------------------------------------------------- 4. fallback targets


class TestFallbackTargets(unittest.TestCase):
    def test_implicit_value_residual(self):
        agent = exclude_implicit()
        collect(agent, 40)
        batch = flagged(agent.replay.sample("train", 8), (1, 2, 6))
        operator = agent.freeze_operator()
        sample = agent._policy_sample(batch["states"], sample_count=3)
        flags = batch_contact_flags(batch)
        residuals = excluded_resolvent_residuals(
            agent.value, operator, agent.controller, batch, sample, flags, create_graph=True
        )
        self.assertEqual(tuple(residuals.shape), (8, 3, 1))
        with torch.no_grad():
            tau, lam = float(operator["resolvent_step"]), float(operator["discount_rate"])
            alpha = float(operator["temperature_per_second"])
            for row in range(8):
                if not flags[row]:
                    continue
                v = float(agent.value(batch["states"][row:row + 1]))
                v_next = float(agent.value(batch["next_states"][row:row + 1]))
                v_k = float(operator["target_value"](batch["states"][row:row + 1]))
                delta = float(batch["durations"][row])
                reward = float(batch["rewards"][row])
                live = 1.0 - float(batch["terminated"][row])
                weight = (1.0 - math.exp(-lam * delta)) / lam
                for draw in range(3):
                    log_p = float(sample["log_probabilities"][row, draw])
                    expected = ((1.0 + tau / delta) * v - v_k
                                - (tau / delta) * (reward + live * math.exp(-lam * delta) * v_next)
                                + tau * alpha * (weight / delta) * log_p)
                    self.assertAlmostEqual(float(residuals[row, draw]), expected,
                                           delta=1e-4 * max(1.0, abs(expected)))
        kept = torch.nonzero(~flags).squeeze(-1)
        modelled = resolvent_residuals(
            agent.value, operator, agent.controller, batch["states"][kept],
            {k: v[kept] for k, v in sample.items()}, create_graph=False,
        )
        torch.testing.assert_close(residuals[kept].detach(), modelled.detach(), rtol=1e-6, atol=1e-6)
        residuals.square().mean().backward()
        for parameter in agent.value.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        # With no row flagged it is resolvent_residuals of the batch.
        clear = flagged(batch, ())
        torch.testing.assert_close(
            excluded_resolvent_residuals(agent.value, operator, agent.controller, clear, sample,
                                         batch_contact_flags(clear), create_graph=False).detach(),
            resolvent_residuals(agent.value, operator, agent.controller, batch["states"], sample,
                                create_graph=False).detach(), rtol=0, atol=0)
        observed = observed_interval_residuals(agent.value, operator, subset_batch(batch, flags),
                                               {k: v[flags] for k, v in sample.items()}, create_graph=False)
        torch.testing.assert_close(residuals[flags].detach(), observed, rtol=1e-6, atol=1e-6)

    def test_implicit_rate_targets(self):
        agent = exclude_implicit()
        collect(agent, 40)
        agent.warm_fit()
        batch = flagged(agent.replay.sample("train", 8), (0, 4))
        model_batch = flagged(agent.replay.sample("train", 8), (2, 3, 7))
        model_decisions = agent._policy_sample(model_batch["states"])["decisions"][:, 0].detach()
        targets = agent._rate_targets(batch, model_batch["states"], model_decisions, model_batch=model_batch)
        data = data_rate_target(agent.target_value, batch, discount_rate=agent.discount_rate)
        torch.testing.assert_close(targets["data"], data, rtol=0, atol=0)
        for row in (0, 4):
            self.assertEqual(float(targets["rollout"][row]), float(data[row]))
            floor = min(agent.rate_config["replay_weight_max"], 1.0 / agent.rate_config["noise_variance_floor"])
            self.assertAlmostEqual(float(targets["data_weights"][row]), floor, places=6)
        observed = data_rate_target(agent.target_value, model_batch, discount_rate=agent.discount_rate)
        for row in (2, 3, 7):
            self.assertEqual(float(targets["generator"][row]), float(observed[row]))
        self.assertTrue(all(torch.isfinite(t).all() for t in targets.values()))
        # fit_rate_head runs with flagged rows in the replay.
        agent.fit_rate_head(1)

    def test_explicit_critic_targets(self):
        for continuation in ("generator", "rollout"):
            with self.subTest(continuation=continuation):
                agent = exclude_explicit(continuation)
                collect(agent, 40)
                batch = flagged(agent.replay.sample("train", 8), (1, 5))
                targets = agent.build_critic_target(batch)
                with torch.no_grad():
                    for row in (1, 5):
                        expected = (
                            float(batch["rewards"][row])
                            + (1.0 - float(batch["terminated"][row]))
                            * math.exp(-agent.discount_rate * float(batch["durations"][row]))
                            * float(agent.target_value(batch["next_states"][row:row + 1]))
                        )
                        self.assertAlmostEqual(float(targets[row]), expected,
                                               delta=1e-5 * max(1.0, abs(expected)))
                    if continuation == "generator":
                        rows = torch.tensor([0, 2, 3, 4, 6, 7])
                        sub = subset_batch(batch, rows)
                        unflagged = sub["rewards"] + (1.0 - sub["terminated"]) * generator_continuation(
                            agent.target_plant, agent.target_value, agent.controller, sub["states"],
                            sub["decisions"], sub["durations"], discount_rate=agent.discount_rate)
                        torch.testing.assert_close(targets[rows], unflagged, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.isfinite(targets).all())
                agent.train(1, 8)

    def test_flagged_gate_rows_carry_no_evidence(self):
        agent = exclude_implicit()
        collect(agent, 60)
        batch = flagged(agent.replay.sample("train", 6), (0, 3))
        other = copy.deepcopy(agent.online_plant)
        with torch.no_grad():
            for parameter in other.parameters():
                parameter.add_(0.05)
        samples = paired_gate_samples(
            other, agent.admitted_plant, agent.target_value, agent.controller, batch,
            failure_predicate=agent._failure_predicate,
            return_terms={"reward_rate": agent._reward_rate,
                          "terminal_payoff": lambda s: tasks.terminal_payoff_tensor(s, agent.testbed),
                          "discount_rate": agent.discount_rate},
            maximum_step=agent.model_step, physics_dt=agent.physics_dt, controller_refresh_steps=None,
            path_count=2, brownian_increments=agent._gate_noise(batch, 2), integrator="rk4",
            truncation=agent.exclusion, skip_rows=batch_contact_flags(batch),
        )
        rows = corrected_gate_rows(samples)["corrected_differences"].reshape(-1)
        self.assertEqual(float(rows[0]), 0.0)
        self.assertEqual(float(rows[3]), 0.0)
        with torch.no_grad():
            start = agent.target_value(batch["states"][:1]).double()
        torch.testing.assert_close(samples["candidate_returns"][0], start.reshape(-1).expand(2),
                                   rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------ 5. truncation


class TestTruncation(unittest.TestCase):
    def test_truncation_equals_the_manual_truncated_rollout(self):
        with Float64():
            env = tasks.make_env(cartpole(), seed=0)
            config = env.config
            oracle = tasks.make_oracle_plant(env).double()
            rule = contact_exclusion(config["state_spec"], np.zeros(2), active=True, model_step=COARSE)
            # Free; clean first coarse substep then near the rail; starting within the margin.
            states = torch.tensor([[0.0, 0.3, 0.5, 0.0], [1.694, 0.0, 4.0, 0.0],
                                   [1.79, 0.0, 0.0, 0.0]], dtype=torch.float64)
            decisions = torch.tensor([[0.0, 0.0, 0.0]] * 3, dtype=torch.float64)
            durations = torch.full((3, 1), INTERVAL, dtype=torch.float64)
            fail = lambda s: tasks.failure_mask_tensor(s, config)  # noqa: E731
            terms = {"reward_rate": lambda s, u: tasks.reward_rate_tensor(s, u, config),
                     "terminal_payoff": lambda s: tasks.terminal_payoff_tensor(s, config),
                     "discount_rate": float(config["discount_rate"])}

            def roll(rows, duration, **extra):
                count = len(rows)
                noise = torch.zeros(count, 1, int(round(duration / COARSE)), oracle.noise_dim,
                                    dtype=torch.float64)
                return stochastic_rollout(
                    oracle, env.controller, states[rows], decisions[rows],
                    torch.full((count, 1), duration, dtype=torch.float64), fail,
                    maximum_step=COARSE, physics_dt=FINE, controller_refresh_steps=None, path_count=1,
                    brownian_increments=noise, return_terms=terms, integrator="rk4", **extra)

            truncated = roll([0, 1, 2], INTERVAL, truncation=rule)
            self.assertEqual(truncated["truncated"].reshape(-1).tolist(), [False, True, True])
            self.assertTrue(truncated["survived"].all())
            torch.testing.assert_close(truncated["durations_used"].reshape(-1),
                                       torch.tensor([INTERVAL, COARSE, 0.0], dtype=torch.float64))
            # The detector's verdict on the plain path's substeps.
            plain = roll([1], COARSE)
            first = plain["endpoints"][0, 0].unsqueeze(0)
            self.assertFalse(bool(contact_flags(rule, states[1:2], first, torch.tensor([COARSE]))))
            full = roll([1], INTERVAL)
            self.assertTrue(bool(contact_flags(rule, first, full["endpoints"][0, 0].unsqueeze(0),
                                               torch.tensor([COARSE]))))
            whole = roll([0], INTERVAL)
            for key in ("endpoints", "running_rewards", "durations_used"):
                torch.testing.assert_close(truncated[key][0], whole[key][0], rtol=0, atol=1e-13)
                torch.testing.assert_close(truncated[key][1], plain[key][0], rtol=0, atol=1e-13)
            torch.testing.assert_close(truncated["endpoints"][2, 0], states[2])
            self.assertEqual(float(truncated["running_rewards"][2]), 0.0)
            skipped = roll([0, 1, 2], INTERVAL, truncation=rule, skip_rows=torch.tensor([True, False, False]))
            torch.testing.assert_close(skipped["endpoints"][0, 0], states[0])
            self.assertEqual(skipped["truncated"].reshape(-1).tolist(), [True, True, True])


# -------------------------------------------------------------------------- 6. plumbing


class TestPlumbing(unittest.TestCase):
    def test_settings_records_fingerprint_cli_and_cell(self):
        config = implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="exclude")
        self.assertIsNone(contact_substepping_setting(config))
        self.assertEqual(exclusion_model_step(config), COARSE)
        self.assertEqual(exclusion_model_step(implicit_config(cartpole(), model_coarse_step=COARSE)), FINE)
        setting = contact_exclusion_setting(config)
        self.assertEqual(setting["mode"], "exclude")
        self.assertEqual(setting["model_step"], COARSE)
        self.assertIsNone(contact_exclusion_setting(implicit_config(cartpole())))
        self.assertIsNone(reporting.record_contact_handling({}))
        # CLI on a table row: exclude turns the port off; the fingerprint carries it. The
        # cartpole-swingup row is under exclusion itself since 2026-11-11; --contact-handling
        # port turns it back into the port convention (limit_port on, no key).
        args = run_ph_sac.parse_args(["--env-id", "cartpole-swingup", "--contact-handling", "exclude",
                                      "--arms", "implicit_learned_admissions", "--seeds", "0"])
        (protocol,) = run_ph_sac.load_protocols(args)
        (plain,) = run_ph_sac.load_protocols(run_ph_sac.parse_args(
            ["--env-id", "cartpole-swingup", "--arms", "implicit_learned_admissions", "--seeds", "0"]))
        (ported,) = run_ph_sac.load_protocols(run_ph_sac.parse_args(
            ["--env-id", "cartpole-swingup", "--contact-handling", "port",
             "--arms", "implicit_learned_admissions", "--seeds", "0"]))
        for name in ("explicit_config", "implicit_config"):
            self.assertEqual(protocol[name]["contact_handling"], "exclude")
            self.assertFalse(protocol[name]["network"]["limit_port"])
            self.assertEqual(plain[name]["contact_handling"], "exclude")
            self.assertFalse(plain[name]["network"].get("limit_port", False))
            self.assertNotIn("contact_handling", ported[name])
            self.assertIs(ported[name]["network"]["limit_port"], True)
        (run,) = P.build_run_matrix(protocol)
        (run_plain,) = P.build_run_matrix(plain)
        (run_off,) = P.build_run_matrix(ported)
        self.assertEqual(R.run_settings(run)["resolved"]["implicit_config"]["contact_handling"]["mode"],
                         "exclude")
        self.assertNotIn("contact_handling", R.run_settings(run_off)["resolved"]["implicit_config"])
        self.assertNotEqual(R.run_fingerprint(run), R.run_fingerprint(run_off))
        self.assertEqual(P._arm_contact_handling(run)["mode"], "exclude")
        self.assertEqual(P._arm_contact_handling(run_plain)["mode"], "exclude")
        self.assertIsNone(P._arm_contact_handling(run_off))
        mixed = copy.deepcopy(protocol)
        mixed["explicit_config"].pop("contact_handling")
        with self.assertRaisesRegex(ValueError, "contact_handling"):
            P.validate_matched_quantities(mixed)
        # The table cell: "exclude" round-trips (the row's own, since 2026-11-11); empty is
        # "port" (absent), the reacher and swimmer rows'.
        entry = table.find_ph_entry("cartpole-swingup", "top")
        self.assertEqual(entry.config["contact_handling"], "exclude")
        self.assertNotIn("limit_port", entry.config["network"])
        self.assertNotIn("contact_handling", table.find_ph_entry("reacher-easy", "top").config)
        body = copy.deepcopy(entry.config)
        row = table.config_to_row(body, env_id="cartpole-swingup", mode="top", task=entry.task)
        self.assertEqual(row["model_contact_handling"], "exclude")
        self.assertEqual(table.parse_ph_row(row).config["contact_handling"], "exclude")
        run_ph_sac.protocol_from_config(table.parse_ph_row(row).config, "x")
        self.assertEqual(row["algo_training_plant_diagnostics_every"], "")
        body.pop("contact_handling")
        body["network"]["limit_port"] = True
        row = table.config_to_row(body, env_id="cartpole-swingup", mode="top", task=entry.task)
        self.assertEqual(row["model_contact_handling"], "")
        self.assertNotIn("contact_handling", table.parse_ph_row(row).config)
        run_ph_sac.protocol_from_config(table.parse_ph_row(row).config, "x")

    def test_harness_configs(self):
        with tempfile.TemporaryDirectory() as out:
            args = harness.parse_args(["point_mass-easy", "100000,4", out, "--experiment",
                                       "--contact-handling", "exclude", "--seed", "2"])
            run = harness.experiment_run(args)
            config = run["protocol"]["implicit_config"]
            self.assertEqual(config["contact_handling"], "exclude")
            self.assertEqual(config["training"]["plant_step_every"], 4)
            self.assertEqual(config["training"]["plant_diagnostics_every"], 5000)
            self.assertEqual(run["protocol"]["matched"]["interaction_budget"], 200_000)
            self.assertEqual(exclusion_model_step(config), 0.01)
            dense = harness.experiment_run(harness.parse_args(
                ["point_mass-easy", "dense", out, "--experiment", "--seed", "2"]))
            self.assertNotIn("plant_step_every", dense["protocol"]["implicit_config"]["training"])
            self.assertTrue(dense["protocol"]["implicit_config"]["network"]["limit_port"])
            self.assertIsNotNone(contact_substepping_setting(dense["protocol"]["implicit_config"]))


# ------------------------------------------------------------------- 7. exact restart


def forced_build(probability=0.5):
    build = P.make_training_agent

    def make(spec, env):
        harness.force_contact(env, probability)
        return build(spec, env)

    return make


class TestExactRestartWithExclusion(ExactRestartMixin, unittest.TestCase):
    def tiny(self, arm, output, budget):
        args = harness.parse_args(["cartpole-swingup", "dense", str(output), "--experiment", "--tiny",
                                   "--contact-handling", "exclude", "--arm", arm, "--budget",
                                   str(budget), "--plant-diagnostics-every", "25"])
        run = harness.experiment_run(args)
        run["protocol"]["evaluation_every_interactions"] = 40
        return run

    def check(self, arm, budget, stop):
        options = R.resume_options(every_interactions=0)
        with mock.patch.object(P, "make_training_agent", forced_build()), \
                tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reference, ref_agent = run_captured(self.tiny(arm, first, budget), options)
            run = self.tiny(arm, second, budget)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=stop))
            self.assertIsInstance(stopped, R.ResumableStop)
            payload = torch.load(P.run_output_directory(run) / R.RESUME_NAME, weights_only=False)
            resumed, res_agent = run_captured(run, options, disturbed=True)
        self.assertEqual(reference["counters"]["interactions"], budget)
        self.assert_same_run(reference, ref_agent, resumed, res_agent)
        self.assertEqual(reference["contact_handling"]["mode"], "exclude")
        columns = [b.contact_columns() for b in ref_agent.replay.buffers.values()]
        touched = sum(int(c["limit_touched"].sum()) for c in columns)
        near = sum(int(c["near_limit"].sum()) for c in columns)
        print(f"{arm}: replay rows {sum(c['near_limit'].size for c in columns)}, touched {touched}, "
              f"near {near}, stopped at {payload['interactions']}")
        self.assertGreater(touched, 0)
        self.assertGreater(near, 0)
        for name in ("limit_touched", "near_limit"):
            for left, right in zip(ref_agent.replay.buffers.values(), res_agent.replay.buffers.values()):
                # The explicit arm holds no rows out: its gate buffer stays empty.
                self.assertEqual(left._arrays is None, right._arrays is None)
                if left._arrays is not None:
                    np.testing.assert_array_equal(left._arrays[name], right._arrays[name])

    def test_implicit_learned_admissions(self):
        self.check("implicit_learned_admissions", 260, 190)

    def test_explicit_rollout(self):
        self.check("explicit_rollout", 200, 150)


if __name__ == "__main__":
    unittest.main()
