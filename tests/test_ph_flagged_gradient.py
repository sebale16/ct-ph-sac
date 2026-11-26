"""The value gradient at contact exclusion's flagged rows (implicit config key
flagged_value_gradient, common.ph_config.FLAGGED_VALUE_GRADIENT_KEY, since 2026-11-23).

Under contact_handling "exclude" the implicit value residual at a flagged row is the
observed interval's (algorithms/ph_resolvent.observed_interval_residuals), with the online
value V at s and s'. "full" (absent: every earlier run) keeps the gradient through both;
"stop_successor" detaches V(s') after the same evaluation, so the residuals keep their
values and the gradient reaches the value's parameters through V(s) alone.

1. Function level, in both reward data (interval rewards and reward_observation
   "end_state", whose flagged rows pass rewards=): the default equals a reference written
   out here with the full gradient; "stop_successor" keeps every residual value bitwise
   and its gradient equals the reference with V(s') replaced by a constant tensor of its
   values, and differs from the full gradient; unflagged rows are untouched.
2. Agent level: update_value of agents with the key absent, "full" and "stop_successor"
   (same seed, same batch): bitwise the same step on a batch without flagged rows; on a
   flagged batch the same loss and, without the key or with "full", the same gradient;
   "stop_successor" another gradient. Every implicit arm reads the key; the arms with the
   default value update pass it to the residual, the explicit-step arms and implicit_critic
   never form the online-value residual.
3. Validation, the JSON loader (written into the implicit config only off "full"), the
   command line's --contact-handling port and the table (no column).
4. Records and resume (cartpole-swingup, the table row under contact exclusion): the
   fingerprint holds the key only off its default; a restart across it is refused; a run
   under it with forced contacts restarts exactly; records store it only then and old
   records load "full".

Comparisons with a reference written out in the test use torch.testing.assert_close with
a tolerance of TOLERANCE times the reference's largest magnitude (rounding only: the same
arithmetic in a possibly different order); comparisons of one code path under two settings
that must not differ use torch.equal.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import tempfile
import unittest
from unittest import mock

import torch

import algorithms.implicit_ct_ph_sac as implicit_module
from algorithms.ph_resolvent import (
    all_pairs_loss,
    excluded_resolvent_residuals,
    observed_interval_residuals,
    resolvent_residuals,
)
from benchmarks import ph_hyperparams as table
from benchmarks import ph_plant_schedule_drift as harness
from benchmarks import run_ph_sac
from common.ph_config import (
    DEFAULT_FLAGGED_VALUE_GRADIENT,
    FLAGGED_VALUE_GRADIENT_KEY,
    FLAGGED_VALUE_GRADIENTS,
    flagged_value_gradient_setting,
)
from common.ph_time import discount_weight
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.reporting import load_run_records, record_flagged_value_gradient
from models.ph_sac.rollouts import batch_contact_flags, subset_batch
from tests.test_ph_algorithms import collect, implicit_config, make_implicit
from tests.test_ph_contact_substep import COARSE, cartpole
from tests.test_ph_resume import ExactRestartMixin, run_captured, tiny_run

KEY = FLAGGED_VALUE_GRADIENT_KEY
STOP = "stop_successor"
# The table row under contact exclusion (contact_handling "exclude" since 2026-11-11).
ENV_ID = "cartpole-swingup"
# Rounding tolerance, relative to the reference's largest magnitude.
TOLERANCE = 1e-5
# The flagged rows of the function-level batches (8 rows).
ROWS = (1, 2, 6)
# The implicit arms whose value step trains on observed_interval_residuals at flagged rows
# (value_step "implicit", value_operator "model"); the other implicit arms are the
# explicit-step arms and implicit_critic.
DEFAULT_UPDATE_ARMS = (
    "implicit_torque",
    "implicit_learned_admissions",
    "implicit_oracle",
    "implicit_averaged_draw",
    "implicit_no_rollout_rate",
    "implicit_polyak_plant",
    "implicit_gated_admissions",
    "implicit_fixed_gate_allocation",
    "implicit_critic_head",
)
OTHER_IMPLICIT_ARMS = (
    "implicit_explicit_step",
    "implicit_explicit_step_interval",
    "implicit_critic",
)


def exclusion_agent(gradient=None, *, end_state=False):
    """A cart-pole implicit agent (feedback decisions) under contact exclusion, with
    flagged_value_gradient gradient (absent when None), under reward_observation "end_state"
    when end_state."""
    keys = {} if gradient is None else {KEY: gradient}
    if end_state:
        keys["reward_observation"] = "end_state"
    return make_implicit(
        implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="exclude", **keys)
    )


def flagged(batch, rows):
    """batch with contact_flags set on the given row indices only."""
    batch = dict(batch)
    flags = torch.zeros(batch["states"].shape[0], 1, dtype=batch["states"].dtype)
    flags[list(rows)] = 1.0
    batch["contact_flags"] = flags
    return batch


def fill(agent, rows):
    """rows of uniform decisions into the agent's replay, no update."""
    for _ in range(rows):
        agent._collect(lambda state: agent._uniform_decision())


def gradients(loss, module):
    """d loss / d parameters of module, zeros for a parameter the loss does not reach."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    # retain_graph: several losses in one test share a residual graph (rows of one tensor).
    found = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, found)]


def scale(tensors):
    """The largest magnitude over tensors (1 if every entry is zero)."""
    largest = max(float(t.detach().abs().max()) for t in tensors)
    return largest if largest > 0.0 else 1.0


def largest_difference(left, right):
    return max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(left, right))


def reference_residuals(value, operator, batch, sample, *, successor=None, rewards=None):
    """observed_interval_residuals' formula written out, per draw [B,N,1]:
        R_i = (1 + tau/Delta) V(s) - V_k(s) - (tau/Delta) [r + (1-d) e^(-lambda Delta) V(s')]
              + tau alpha_c (w_lambda(Delta)/Delta) log pi_i,
    V(s') the value at the next states with its gradient (the full residual gradient), or
    successor, a constant tensor, when given. rewards [B,1] (None: the batch's interval
    rewards) stand for r."""
    tau = float(operator["resolvent_step"])
    lam = float(operator["discount_rate"])
    durations = batch["durations"].reshape(-1, 1)
    here = value(batch["states"])
    there = value(batch["next_states"]) if successor is None else successor
    with torch.no_grad():
        anchor = operator["target_value"](batch["states"])
        alpha = torch.as_tensor(operator["temperature_per_second"]).detach()
        reward = (batch["rewards"] if rewards is None else rewards).reshape(-1, 1)
        live = 1.0 - batch["terminated"].reshape(-1, 1)
        discount = torch.exp(-lam * durations)
        weight = discount_weight(durations, discount_rate=lam) / durations
    ratio = tau / durations
    base = (1.0 + ratio) * here - anchor - ratio * (reward + live * discount * there)
    charge = (tau * alpha * weight).unsqueeze(1) * sample["log_probabilities"].detach()
    return base.unsqueeze(1) + charge


@contextlib.contextmanager
def implicit_key(value):
    """Every table row read inside the block carries the implicit key KEY = value, as a JSON
    config's "implicit" section would (tests/test_ph_resume.tiny_run builds its protocol
    from the row with run_ph_sac.protocol_from_config)."""
    find = table.find_ph_entry

    def keyed(env_id, mode="top", *args, **kwargs):
        entry = find(env_id, mode, *args, **kwargs)
        config = copy.deepcopy(entry.config)
        config["implicit"][KEY] = value
        return dataclasses.replace(entry, config=config)

    with mock.patch.object(table, "find_ph_entry", keyed):
        yield


def keyed_run(arm, output, budget, value=STOP):
    """tests/test_ph_resume.tiny_run of ENV_ID with the implicit key KEY = value."""
    with implicit_key(value):
        return tiny_run(ENV_ID, arm, output, budget)


def forced_build(probability):
    """make_training_agent with every training episode starting at a limit with this
    probability (benchmarks/ph_plant_schedule_drift.force_contact), as
    tests/test_ph_contact_exclusion.py's restart test."""
    build = P.make_training_agent

    def make(spec, env):
        harness.force_contact(env, probability)
        return build(spec, env)

    return make


# ------------------------------------------------------------------- 1. function level


class TestObservedResidual(unittest.TestCase):
    def rows(self, end_state):
        """An exclusion agent after 40 rows, its operator, and the flagged rows ROWS of a
        batch of 8: their sub-batch, the operator policy's draws there and, under
        "end_state", their rewards (the observed ends' trapezoid plus the charge, as
        update_value passes them); also the whole batch, its flags and draws."""
        agent = exclusion_agent(end_state=end_state)
        collect(agent, 40)
        operator = agent.freeze_operator()
        batch = flagged(agent.replay.sample("train", 8), ROWS)
        flags = batch_contact_flags(batch)
        with torch.no_grad():
            sample = agent._policy_sample(
                batch["states"],
                sample_count=int(agent.solver["policy_draws"]),
                actor=operator["policy"],
            )
        rewards = None
        if end_state:
            self.assertNotIn("rewards", batch)
            rewards = agent._observed_end_rewards(batch, operator["reward_rate"])
        index = torch.nonzero(flags).squeeze(-1)
        sub = subset_batch(batch, index)
        draws = {
            key: sample[key].index_select(0, index) for key in ("decisions", "log_probabilities")
        }
        sub_rewards = None if rewards is None else rewards.index_select(0, index)
        # Every flagged row moved (s' != s) and did not terminate: V(s') enters its residual.
        self.assertTrue(bool((sub["states"] != sub["next_states"]).any(dim=-1).all()))
        self.assertEqual(float(sub["terminated"].sum()), 0.0)
        return {
            "agent": agent, "operator": operator, "batch": batch, "flags": flags,
            "sample": sample, "rewards": rewards, "sub": sub, "draws": draws,
            "sub_rewards": sub_rewards,
        }

    def assert_matches(self, actual, expected):
        """Equal to rounding: within TOLERANCE of expected's largest magnitude."""
        actual = actual if isinstance(actual, list) else [actual]
        expected = expected if isinstance(expected, list) else [expected]
        bound = TOLERANCE * scale(expected)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.detach(), e.detach(), rtol=0.0, atol=bound)

    def test_default_is_the_full_gradient(self):
        for end_state in (False, True):
            with self.subTest(end_state=end_state):
                case = self.rows(end_state)
                agent, operator = case["agent"], case["operator"]
                arguments = (agent.value, operator, case["sub"], case["draws"])
                absent = observed_interval_residuals(
                    *arguments, create_graph=True, rewards=case["sub_rewards"]
                )
                stated = observed_interval_residuals(
                    *arguments, create_graph=True, rewards=case["sub_rewards"],
                    stop_successor=False,
                )
                reference = reference_residuals(*arguments, rewards=case["sub_rewards"])
                self.assertEqual(tuple(absent.shape), (len(ROWS), 3, 1))
                self.assertTrue(torch.equal(absent.detach(), stated.detach()))
                absent_gradient = gradients(all_pairs_loss(absent), agent.value)
                stated_gradient = gradients(all_pairs_loss(stated), agent.value)
                for a, b in zip(absent_gradient, stated_gradient):
                    self.assertTrue(torch.equal(a, b))
                self.assert_matches(absent, reference)
                self.assert_matches(
                    absent_gradient, gradients(all_pairs_loss(reference), agent.value)
                )

    def test_stop_successor_cuts_only_the_successor_gradient(self):
        for end_state in (False, True):
            with self.subTest(end_state=end_state):
                case = self.rows(end_state)
                agent, operator, sub = case["agent"], case["operator"], case["sub"]
                arguments = (agent.value, operator, sub, case["draws"])
                full = observed_interval_residuals(
                    *arguments, create_graph=True, rewards=case["sub_rewards"]
                )
                stop = observed_interval_residuals(
                    *arguments, create_graph=True, rewards=case["sub_rewards"],
                    stop_successor=True,
                )
                # The residual values do not change.
                self.assertTrue(torch.equal(stop.detach(), full.detach()))
                # The gradient is the one with V(s') a constant holding its values.
                constant = agent.value(sub["next_states"]).detach()
                reference = reference_residuals(
                    *arguments, successor=constant, rewards=case["sub_rewards"]
                )
                self.assert_matches(stop, reference)
                stop_gradient = gradients(all_pairs_loss(stop), agent.value)
                self.assert_matches(
                    stop_gradient, gradients(all_pairs_loss(reference), agent.value)
                )
                # And not the full gradient.
                full_gradient = gradients(all_pairs_loss(full), agent.value)
                self.assertGreater(
                    largest_difference(stop_gradient, full_gradient),
                    TOLERANCE * scale(full_gradient),
                )

    def test_unflagged_rows_are_untouched(self):
        for end_state in (False, True):
            with self.subTest(end_state=end_state):
                case = self.rows(end_state)
                agent, operator, flags = case["agent"], case["operator"], case["flags"]
                arguments = (agent.value, operator, agent.controller, case["batch"], case["sample"])
                full = excluded_resolvent_residuals(
                    *arguments, flags, create_graph=True, rewards=case["rewards"]
                )
                stop = excluded_resolvent_residuals(
                    *arguments, flags, create_graph=True, rewards=case["rewards"],
                    stop_successor=True,
                )
                self.assertTrue(torch.equal(stop.detach(), full.detach()))
                kept = ~flags
                self.assertGreater(int(kept.sum()), 0)
                kept_full = gradients(all_pairs_loss(full[kept]), agent.value)
                kept_stop = gradients(all_pairs_loss(stop[kept]), agent.value)
                for a, b in zip(kept_full, kept_stop):
                    self.assertTrue(torch.equal(a, b))
                flagged_full = gradients(all_pairs_loss(full[flags]), agent.value)
                flagged_stop = gradients(all_pairs_loss(stop[flags]), agent.value)
                self.assertGreater(
                    largest_difference(flagged_full, flagged_stop),
                    TOLERANCE * scale(flagged_full),
                )
                # With no row flagged the setting leaves resolvent_residuals of the batch.
                clear = torch.zeros_like(flags)
                none = excluded_resolvent_residuals(
                    *arguments, clear, create_graph=True, rewards=case["rewards"],
                    stop_successor=True,
                )
                plain = resolvent_residuals(
                    agent.value, operator, agent.controller, case["batch"]["states"],
                    case["sample"], create_graph=True,
                )
                self.assertTrue(torch.equal(none.detach(), plain.detach()))
                for a, b in zip(
                    gradients(all_pairs_loss(none), agent.value),
                    gradients(all_pairs_loss(plain), agent.value),
                ):
                    self.assertTrue(torch.equal(a, b))


# ---------------------------------------------------------------------- 2. agent level


def value_step(agent, batch):
    """update_value on batch under a fresh operator: the loss, the value's (clipped)
    gradients the step used, and its parameters after the step."""
    operator = agent.freeze_operator()
    loss = agent.update_value(batch, operator)["value_loss"]
    parameters = list(agent.value.parameters())
    grads = [
        torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters
    ]
    return loss, grads, [p.detach().clone() for p in parameters]


class TestValueStep(unittest.TestCase):
    def agents(self, end_state):
        """Agents with the key absent, "full" and "stop_successor", the same seed and the
        same 40 rows, and one batch of 8 that each replay draws alike."""
        agents = {
            label: exclusion_agent(gradient, end_state=end_state)
            for label, gradient in (("absent", None), ("full", "full"), ("stop", STOP))
        }
        self.assertEqual(agents["absent"].flagged_value_gradient, DEFAULT_FLAGGED_VALUE_GRADIENT)
        self.assertEqual(agents["full"].flagged_value_gradient, "full")
        self.assertEqual(agents["stop"].flagged_value_gradient, STOP)
        for agent in agents.values():
            collect(agent, 40)
        batches = [agent.replay.sample("train", 8) for agent in agents.values()]
        for other in batches[1:]:
            for key, column in batches[0].items():
                self.assertTrue(torch.equal(column, other[key]), key)
        return agents, batches[0]

    def test_unflagged_batch_steps_alike(self):
        for end_state in (False, True):
            with self.subTest(end_state=end_state):
                agents, batch = self.agents(end_state)
                steps = {label: value_step(agent, flagged(batch, ())) for label, agent in agents.items()}
                for label in ("full", "stop"):
                    self.assertEqual(steps[label][0], steps["absent"][0])
                    for part in (1, 2):
                        for a, b in zip(steps[label][part], steps["absent"][part]):
                            self.assertTrue(torch.equal(a, b), label)

    def test_flagged_batch_keeps_the_loss_and_changes_the_gradient(self):
        for end_state in (False, True):
            with self.subTest(end_state=end_state):
                agents, batch = self.agents(end_state)
                marked = flagged(batch, ROWS)
                with mock.patch.object(
                    implicit_module, "excluded_resolvent_residuals",
                    wraps=implicit_module.excluded_resolvent_residuals,
                ) as spy:
                    steps = {label: value_step(agent, marked) for label, agent in agents.items()}
                self.assertEqual(
                    [call.kwargs["stop_successor"] for call in spy.call_args_list],
                    [False, False, True],
                )
                if end_state:
                    self.assertTrue(
                        all(call.kwargs["rewards"] is not None for call in spy.call_args_list)
                    )
                # Absent and "full": the same step, bitwise.
                self.assertEqual(steps["full"][0], steps["absent"][0])
                for part in (1, 2):
                    for a, b in zip(steps["full"][part], steps["absent"][part]):
                        self.assertTrue(torch.equal(a, b))
                # "stop_successor": the same residuals (loss), another gradient.
                self.assertEqual(steps["stop"][0], steps["absent"][0])
                self.assertGreater(
                    largest_difference(steps["stop"][1], steps["absent"][1]),
                    TOLERANCE * scale(steps["absent"][1]),
                )

    def test_every_implicit_arm_reads_the_key(self):
        implicit = [arm for arm in P.all_arms() if arm["algorithm"] == "implicit"]
        self.assertEqual(
            sorted(arm["name"] for arm in implicit),
            sorted(DEFAULT_UPDATE_ARMS + OTHER_IMPLICIT_ARMS),
        )
        for arm in implicit:
            with self.subTest(arm=arm["name"]):
                run = keyed_run(arm["name"], "/nonexistent", 10)
                self.assertEqual(run["protocol"]["implicit_config"][KEY], STOP)
                testbed = run["protocol"]["matched"]["testbed_config"]
                env = make_env(testbed, decision_mode=arm["decision_mode"], seed=0)
                agent = P.make_training_agent(run, env)
                self.assertEqual(agent.flagged_value_gradient, STOP)
                self.assertTrue(agent.exclusion_active)
                fill(agent, 40)
                batch = flagged(agent.replay.sample("train", 16), (0, 3, 7))
                with mock.patch.object(
                    implicit_module, "excluded_resolvent_residuals",
                    wraps=implicit_module.excluded_resolvent_residuals,
                ) as spy:
                    agent.update_value(batch, agent.freeze_operator())
                if arm["name"] in DEFAULT_UPDATE_ARMS:
                    self.assertEqual(spy.call_count, 1)
                    self.assertIs(spy.call_args.kwargs["stop_successor"], True)
                    self.assertIs(spy.call_args.kwargs["create_graph"], True)
                else:
                    # The explicit step's flagged rate reads V_k(s') (ph_resolvent.
                    # operator_anchor_rates, no gradient); implicit_critic reads no observed
                    # interval. Neither forms the online-value residual.
                    spy.assert_not_called()
                env.close()


# ----------------------------------------------------------------------- 3. validation


class TestValidation(unittest.TestCase):
    def test_setting_and_config_checks(self):
        self.assertEqual(FLAGGED_VALUE_GRADIENTS, ("full", STOP))
        self.assertEqual(flagged_value_gradient_setting({}), "full")
        for value in FLAGGED_VALUE_GRADIENTS:
            self.assertEqual(flagged_value_gradient_setting({KEY: value}), value)
            implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="exclude",
                            **{KEY: value})
        for bad in ("stop", "Stop_successor", "", None, True, 1):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, KEY):
                    flagged_value_gradient_setting({KEY: bad})
                with self.assertRaisesRegex(ValueError, KEY):
                    implicit_config(cartpole(), model_coarse_step=COARSE,
                                    contact_handling="exclude", **{KEY: bad})
        # "stop_successor" acts at flagged rows only: refused without contact exclusion.
        for testbed in (cartpole(), None):
            with self.assertRaisesRegex(ValueError, "contact_handling"):
                implicit_config(testbed, **{KEY: STOP})
            implicit_config(testbed, **{KEY: "full"})

    def test_json_loader_command_line_and_table(self):
        raw = copy.deepcopy(table.find_ph_entry(ENV_ID, "top").config)
        self.assertEqual(raw["contact_handling"], "exclude")
        plain = run_ph_sac.protocol_from_config(copy.deepcopy(raw), "plain")
        self.assertNotIn(KEY, plain["implicit_config"])
        keyed = copy.deepcopy(raw)
        keyed["implicit"][KEY] = STOP
        proto = run_ph_sac.protocol_from_config(copy.deepcopy(keyed), "keyed")
        self.assertEqual(proto["implicit_config"][KEY], STOP)
        self.assertNotIn(KEY, proto["explicit_config"])
        self.assertNotIn(KEY, proto)
        # "full" stated is the absent key.
        stated = copy.deepcopy(raw)
        stated["implicit"][KEY] = "full"
        self.assertNotIn(KEY, run_ph_sac.protocol_from_config(stated, "stated")["implicit_config"])
        for bad in ("stop", None, True):
            broken = copy.deepcopy(raw)
            broken["implicit"][KEY] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, KEY):
                run_ph_sac.protocol_from_config(broken, "bad")
        # Only the implicit section takes it.
        for where in ("explicit", "protocol", None):
            misplaced = copy.deepcopy(raw)
            (misplaced if where is None else misplaced[where])[KEY] = STOP
            with self.subTest(where=where), self.assertRaisesRegex(ValueError, "unknown settings"):
                run_ph_sac.protocol_from_config(misplaced, "misplaced")
        # A row without contact exclusion refuses "stop_successor" and takes "full".
        port = copy.deepcopy(table.find_ph_entry("pendulum-swingup", "top").config)
        self.assertNotIn("contact_handling", port)
        port["implicit"][KEY] = STOP
        with self.assertRaisesRegex(ValueError, "contact_handling"):
            run_ph_sac.protocol_from_config(copy.deepcopy(port), "port")
        port["implicit"][KEY] = "full"
        self.assertNotIn(KEY, run_ph_sac.protocol_from_config(port, "port")["implicit_config"])
        # --contact-handling port on a protocol with the key is refused.
        args = run_ph_sac.parse_args(["--env-id", ENV_ID, "--contact-handling", "port"])
        with self.assertRaisesRegex(ValueError, "contact_handling"):
            run_ph_sac.apply_overrides(proto, args)
        # No table column: a config that sets it stays a JSON file.
        with self.assertRaisesRegex(ValueError, "implicit.flagged_value_gradient"):
            table.config_to_row(keyed, env_id=ENV_ID)


# -------------------------------------------------------------- 4. records and resume


class TestRecordsAndResume(ExactRestartMixin, unittest.TestCase):
    ARMS = ("implicit_torque", "explicit_rollout_torque", "ct_sac_torque")

    def test_fingerprint_holds_the_key_only_off_its_default(self):
        for arm in self.ARMS:
            with self.subTest(arm=arm):
                plain = tiny_run(ENV_ID, arm, "/nonexistent", 100)
                stated = keyed_run(arm, "/nonexistent", 100, "full")
                keyed = keyed_run(arm, "/nonexistent", 100)
                self.assertNotIn(KEY, json.dumps(R.run_settings(plain)))
                self.assertEqual(R.run_settings(stated), R.run_settings(plain))
                self.assertEqual(R.run_fingerprint(stated), R.run_fingerprint(plain))
                self.assertNotEqual(R.run_fingerprint(keyed), R.run_fingerprint(plain))
                self.assertEqual(
                    R._differences(R.run_settings(plain), R.run_settings(keyed)),
                    [f"protocol.implicit_config.{KEY}"],
                )

    def test_a_restart_across_the_key_is_refused(self):
        arm = "implicit_torque"
        with tempfile.TemporaryDirectory() as plain_dir, tempfile.TemporaryDirectory() as keyed_dir:
            plain = tiny_run(ENV_ID, arm, plain_dir, 90)
            keyed = keyed_run(arm, keyed_dir, 90)
            for run in (plain, keyed):
                stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=30))
                self.assertIsInstance(stopped, R.ResumableStop)
            plain_path = P.run_output_directory(plain) / R.RESUME_NAME
            keyed_path = P.run_output_directory(keyed) / R.RESUME_NAME
            for path, other in (
                (plain_path, keyed_run(arm, plain_dir, 90)),
                (keyed_path, tiny_run(ENV_ID, arm, keyed_dir, 90)),
            ):
                with self.subTest(path=str(path)), self.assertRaisesRegex(R.ResumeMismatch, KEY):
                    R.load_resume_payload(path, other)
            # "full" stated is the plain run; the key again is the keyed run.
            stated = keyed_run(arm, plain_dir, 90, "full")
            self.assertGreaterEqual(R.load_resume_payload(plain_path, stated)["interactions"], 30)
            again = keyed_run(arm, keyed_dir, 90)
            self.assertGreaterEqual(R.load_resume_payload(keyed_path, again)["interactions"], 30)

    def test_exact_restart_under_the_key(self):
        # Every training episode starts at the rail (force_contact probability 1), so the
        # replay holds flagged rows from the first decision on.
        with implicit_key(STOP), mock.patch.object(P, "make_training_agent", forced_build(1.0)):
            reference, agent, _, payload = self.check_exact_restart(
                ENV_ID, "implicit_torque", 150, 125
            )
        self.assertEqual(agent.flagged_value_gradient, STOP)
        self.assertEqual(agent.config[KEY], STOP)
        self.assertEqual(reference[KEY], STOP)
        self.assertEqual(reference["run"]["protocol"]["implicit_config"][KEY], STOP)
        self.assertEqual(reference["contact_handling"]["mode"], "exclude")
        self.assertEqual(payload["settings"]["protocol"]["implicit_config"][KEY], STOP)
        self.assertGreater(reference["counters"]["value_gradient_steps"], 0)
        near = sum(
            int(buffer.contact_columns()["near_limit"].sum())
            for buffer in agent.replay.buffers.values()
        )
        self.assertGreater(near, 0)

    def test_records_store_the_key_and_old_records_load_full(self):
        self.assertEqual(record_flagged_value_gradient({}), "full")
        self.assertEqual(record_flagged_value_gradient({KEY: STOP}), STOP)
        with tempfile.TemporaryDirectory() as output:
            run = keyed_run("implicit_torque", output, 60)
            result, agent = run_captured(run)
            self.assertEqual(agent.flagged_value_gradient, STOP)
            self.assertEqual(result[KEY], STOP)
            path = P.run_output_directory(run) / "run_record.json"
            stored = json.loads(path.read_text())
            self.assertEqual(stored["result"][KEY], STOP)
            self.assertEqual(stored["result"]["run"]["protocol"]["implicit_config"][KEY], STOP)
            (loaded,) = load_run_records([str(path)])
            self.assertEqual(loaded[KEY], STOP)
            del stored["result"][KEY]
            path.write_text(json.dumps(stored))
            (old,) = load_run_records([str(path)])
            self.assertEqual(old[KEY], "full")
        # Under the same protocol the explicit and CT-SAC arms record nothing new.
        for arm in ("explicit_rollout_torque", "ct_sac_torque"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as output:
                result, _ = run_captured(keyed_run(arm, output, 60))
                self.assertNotIn(KEY, result)
        # Without the key: nothing new in the record, and it loads "full".
        with tempfile.TemporaryDirectory() as output:
            run = tiny_run(ENV_ID, "implicit_torque", output, 60)
            result, agent = run_captured(run)
            self.assertEqual(agent.flagged_value_gradient, "full")
            self.assertNotIn(KEY, result)
            path = P.run_output_directory(run) / "run_record.json"
            stored = json.loads(path.read_text())["result"]
            self.assertNotIn(KEY, stored)
            self.assertNotIn(KEY, stored["run"]["protocol"]["implicit_config"])
            (loaded,) = load_run_records([str(path)])
            self.assertEqual(loaded[KEY], "full")


if __name__ == "__main__":
    unittest.main()
