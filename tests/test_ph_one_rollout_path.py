"""One model path per transition: rollout_paths 1 (JSON explicit.rollout_paths, the rollout
arms' critic target, and implicit.rate.rollout_paths, the rate head's rollout target).

With a drift-only plant (network diffusion "none": a zero diffusion column, so the noise
adds exactly nothing and every path of a row is the same deterministic integration) the
targets at L = 1 are those at L = 4:

1. The reductions, exactly: survivor_continuation and rollout_interval_reward of four
   copies of one path equal those of the path itself, with surviving, failed (the
   observed-endpoint fallback) and truncated paths (the observed endpoint). The sum of four
   equal floating-point numbers is exact in any order and the division by four too, and
   the value here is elementwise plain arithmetic, so torch.equal.
2. The agents' targets on a cart-pole under contact exclusion (flagged rows set), on a
   cart-pole under contact substepping, and on the windy pendulum (a failure set), in both
   reward data: the implicit rate targets and the explicit rollout critic target. The
   path-free targets and the flagged rows: torch.equal. The rolled-out rows: the same
   arithmetic, but the networks evaluate B rows instead of 4B, and a CPU matrix kernel may
   round a row differently with the batch size; torch.equal may hold but is not
   guaranteed, so assert_close within RELATIVE of the largest magnitude (the rate
   targets compared times Delta, in value units). One seed is drawn from the rollout
   stream per call whatever L; model_rollout_paths counts B L.
3. Validation: L = 1 passes both configs' checks and the JSON loader; the gate's
   admission.initial_paths still needs 2 (the arms here admit without a test).
"""

from __future__ import annotations

import copy
import unittest

import torch

from algorithms.ph_targets import rollout_interval_reward
from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import validate_configuration
from models.ph_sac.rollouts import batch_contact_flags, survivor_continuation
from tests.test_ph_algorithms import (
    collect,
    explicit_config,
    implicit_config,
    make_explicit,
    make_implicit,
    windy,
)
from tests.test_ph_contact_substep import COARSE, cartpole

# Rounding tolerance of the rolled-out rows, relative to the largest magnitude (at least 1).
# A matrix kernel that rounds a row differently with the batch size (B against 4B rows, or
# one moving path against four near an interval's end) is expected to change only the last
# float32 digits (an expectation, not measured here); a wrong path count would change the
# targets by O(1).
RELATIVE = 1e-4
ROWS = 8


def setups():
    """(name, testbed, config keys, flagged batch rows, flagged model-point rows)."""
    return (
        ("exclusion", cartpole(), {"model_coarse_step": COARSE, "contact_handling": "exclude"},
         (1, 5), (2,)),
        ("substepping", cartpole(), {"model_coarse_step": COARSE}, (), ()),
        ("failure_set", windy(), {}, (), ()),
    )


def drift_only(config):
    """config's network with diffusion "none" (the drift-only plant)."""
    return {**config["network"], "diffusion": "none"}


def flagged(batch, rows):
    """batch with contact_flags set on the given row indices only."""
    batch = dict(batch)
    flags = torch.zeros(batch["states"].shape[0], 1, dtype=batch["states"].dtype)
    flags[list(rows)] = 1.0
    batch["contact_flags"] = flags
    return batch


def plain_value(states):
    """A value of plain arithmetic only (no matrix product, no transcendental function), so
    each row's number is the same whatever batch it is evaluated in."""
    return states[..., :1] * 0.5 + states[..., 1:2] * states[..., 1:2] - 0.25


def assert_rounding_equal(case, one, four):
    """one (L = 1) equals four (L = 4) to rounding: within RELATIVE of four's largest
    magnitude (at least 1)."""
    if four.numel() == 0:
        case.assertEqual(one.numel(), 0)
        return
    bound = RELATIVE * max(1.0, float(four.abs().max()))
    torch.testing.assert_close(one, four, rtol=0.0, atol=bound)


def keys(overrides, end_state):
    keys = dict(overrides)
    if end_state:
        keys["reward_observation"] = "end_state"
    return keys


def check_drift_only(case, agent, plant, states):
    """The plant has one noise column and its diffusion is exactly zero at states."""
    case.assertEqual(plant.noise_dim, 1)
    with torch.no_grad():
        case.assertEqual(int(torch.count_nonzero(plant.diffusion(states))), 0)


# ---------------------------------------------------------------------- 1. reductions


class TestReductions(unittest.TestCase):
    def test_four_copies_of_a_path_reduce_to_the_path(self):
        generator = torch.Generator().manual_seed(0)
        rows, dim, lam = 6, 3, 0.5
        endpoints = torch.randn(rows, 1, dim, generator=generator)
        survived = torch.tensor([True, True, False, True, False, True]).reshape(rows, 1, 1)
        truncated = torch.tensor([False, True, False, False, True, False]).reshape(rows, 1, 1)
        running = torch.randn(rows, 1, 1, generator=generator, dtype=torch.float64)
        observed_next = torch.randn(rows, dim, generator=generator)
        observed_reward = torch.randn(rows, 1, generator=generator)
        durations = torch.full((rows, 1), 0.01)
        for truncation in (False, True):
            with self.subTest(truncation=truncation):
                one = {"endpoints": endpoints, "survived": survived, "running_rewards": running}
                if truncation:
                    one["truncated"] = truncated
                four = {key: value.repeat(1, 4, 1) for key, value in one.items()}
                self.assertEqual(tuple(four["endpoints"].shape), (rows, 4, dim))
                continuation = survivor_continuation(
                    plain_value, one, observed_next, durations, lam
                )
                self.assertTrue(
                    torch.equal(
                        continuation,
                        survivor_continuation(plain_value, four, observed_next, durations, lam),
                    )
                )
                reward = rollout_interval_reward(one, observed_reward)
                self.assertTrue(torch.equal(reward, rollout_interval_reward(four, observed_reward)))
                # What one path gives: its own continuation and reward if it survived (a
                # truncated one the observed endpoint's), the observed ones if it failed.
                alive = survived[:, 0]
                cut = truncated[:, 0] if truncation else torch.zeros_like(alive)
                fallback = plain_value(observed_next)
                own = torch.where(cut, fallback, plain_value(endpoints[:, 0]))
                torch.testing.assert_close(
                    continuation, torch.exp(-lam * durations) * torch.where(alive, own, fallback)
                )
                observed = observed_reward.double()
                own_reward = torch.where(cut, observed, running[:, 0])
                torch.testing.assert_close(
                    reward, torch.where(alive, own_reward, observed).to(observed_reward.dtype)
                )


# -------------------------------------------------------------------------- 2. agents


class TestAgentTargets(unittest.TestCase):
    def test_implicit_rate_targets(self):
        for name, testbed, overrides, rows, model_rows in setups():
            for end_state in (False, True):
                with self.subTest(setup=name, end_state=end_state):
                    base = implicit_config(testbed)
                    config = implicit_config(
                        testbed,
                        network=drift_only(base),
                        rate={**base["rate"], "rollout_paths": 4},
                        **keys(overrides, end_state),
                    )
                    agent = make_implicit(config)
                    collect(agent, 40)
                    batch = flagged(agent.replay.sample("train", ROWS), rows)
                    model_batch = flagged(agent.replay.sample("train", ROWS), model_rows)
                    check_drift_only(self, agent, agent.admitted_plant, batch["states"])
                    with torch.no_grad():
                        decisions = agent._policy_sample(model_batch["states"])["decisions"][:, 0]
                    start = copy.deepcopy(agent._rollout_rng.bit_generator.state)
                    results = {}
                    for paths in (4, 1):
                        agent._rollout_rng.bit_generator.state = copy.deepcopy(start)
                        agent.rate_config = {**agent.rate_config, "rollout_paths": paths}
                        before = int(agent._counters["model_rollout_paths"])
                        targets = agent._rate_targets(
                            batch, model_batch["states"], decisions, model_batch=model_batch
                        )
                        results[paths] = (
                            targets,
                            int(agent._counters["model_rollout_paths"]) - before,
                            copy.deepcopy(agent._rollout_rng.bit_generator.state),
                        )
                    four, one = results[4][0], results[1][0]
                    self.assertEqual((results[4][1], results[1][1]), (4 * ROWS, ROWS))
                    self.assertEqual(results[1][2], results[4][2])
                    self.assertNotEqual(results[1][2], start)
                    for key in ("data", "generator", "data_weights"):
                        self.assertTrue(torch.equal(one[key], four[key]), key)
                    if agent.exclusion_active:
                        flags = batch_contact_flags(batch)
                        self.assertEqual(int(flags.sum()), len(rows))
                    else:
                        flags = torch.zeros(ROWS, dtype=torch.bool)
                    # Flagged rows: the data target, not rolled out.
                    self.assertTrue(torch.equal(one["rollout"][flags], four["rollout"][flags]))
                    self.assertTrue(torch.equal(four["rollout"][flags], four["data"][flags]))
                    rolled = ~flags
                    durations = batch["durations"][rolled]
                    assert_rounding_equal(
                        self, one["rollout"][rolled] * durations, four["rollout"][rolled] * durations
                    )
                    self.assertTrue(all(torch.isfinite(t).all() for t in one.values()))

    def test_explicit_rollout_critic_target(self):
        for name, testbed, overrides, rows, _ in setups():
            for end_state in (False, True):
                with self.subTest(setup=name, end_state=end_state):
                    base = explicit_config(testbed)
                    config = explicit_config(
                        testbed, continuation="rollout", network=drift_only(base),
                        **keys(overrides, end_state),
                    )
                    config["rollout_paths"] = 4
                    validate_configuration(config)
                    agent = make_explicit(config)
                    self.assertEqual(agent.rollout_paths, 4)
                    collect(agent, 40)
                    batch = flagged(agent.replay.sample("train", ROWS), rows)
                    check_drift_only(self, agent, agent.target_plant, batch["states"])
                    start = copy.deepcopy(agent._rollout_rng.bit_generator.state)
                    results = {}
                    for paths in (4, 1):
                        agent._rollout_rng.bit_generator.state = copy.deepcopy(start)
                        agent.rollout_paths = paths
                        before = int(agent._counters["model_rollout_paths"])
                        target = agent.build_critic_target(batch)
                        results[paths] = (
                            target,
                            int(agent._counters["model_rollout_paths"]) - before,
                            copy.deepcopy(agent._rollout_rng.bit_generator.state),
                        )
                    four, one = results[4][0], results[1][0]
                    self.assertEqual((results[4][1], results[1][1]), (4 * ROWS, ROWS))
                    self.assertEqual(results[1][2], results[4][2])
                    if agent.exclusion_active:
                        flags = batch_contact_flags(batch)
                        self.assertEqual(int(flags.sum()), len(rows))
                    else:
                        flags = torch.zeros(ROWS, dtype=torch.bool)
                    # Flagged rows: the observed continuation, not rolled out.
                    self.assertTrue(torch.equal(one[flags], four[flags]))
                    assert_rounding_equal(self, one[~flags], four[~flags])
                    self.assertTrue(bool(torch.isfinite(one).all()))


# ---------------------------------------------------------------------- 3. validation


class TestValidation(unittest.TestCase):
    def test_one_path_passes_and_the_gate_keeps_two(self):
        base = implicit_config(windy())
        implicit = implicit_config(windy(), rate={**base["rate"], "rollout_paths": 1})
        self.assertEqual(make_implicit(implicit).rate_config["rollout_paths"], 1)
        with self.assertRaisesRegex(ValueError, "Rate rollouts"):
            implicit_config(windy(), rate={**base["rate"], "rollout_paths": 0})
        # The gate's paths (admission.initial_paths) are another count, still at least 2.
        with self.assertRaisesRegex(ValueError, "L >= 2"):
            implicit_config(windy(), admission={**base["admission"], "initial_paths": 1})
        explicit = explicit_config(windy(), continuation="rollout")
        explicit["rollout_paths"] = 1
        validate_configuration(explicit)
        self.assertEqual(make_explicit(explicit).rollout_paths, 1)
        explicit["rollout_paths"] = 0
        with self.assertRaisesRegex(ValueError, "at least one path"):
            validate_configuration(explicit)
        # A JSON config's two keys: explicit.rollout_paths and implicit.rate.rollout_paths.
        raw = copy.deepcopy(table.find_ph_entry("cartpole-swingup", "top").config)
        raw["explicit"]["rollout_paths"] = 1
        raw["implicit"]["rate"]["rollout_paths"] = 1
        protocol = run_ph_sac.protocol_from_config(raw, "one path")
        self.assertEqual(protocol["explicit_config"]["rollout_paths"], 1)
        self.assertEqual(protocol["implicit_config"]["rate"]["rollout_paths"], 1)
        self.assertGreaterEqual(protocol["implicit_config"]["admission"]["initial_paths"], 2)


if __name__ == "__main__":
    unittest.main()
