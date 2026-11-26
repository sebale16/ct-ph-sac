"""Contact-triggered fine substepping (config key model_coarse_step, since 2026-11-10).

models/ph_sac/rollouts.ContactSubstepping: on a task with joint limits every model
integration of a decision interval runs every row at the coarse step, flags the rows
whose coarse path comes near a limit (contact_flags) and integrates those again at the
model step, whose results replace the coarse ones.

1. Composition, bitwise: with no row flagged the result is the plain coarse
   integration's, with every row flagged the plain fine integration's, and a mixed batch
   is the row-wise composition (coarse rows of the full-batch coarse pass, fine rows of
   the fine pass over the flagged rows in order), for the endpoint moments (leading-order
   and propagated covariance), the path Jacobian norms and the stochastic rollouts (noise,
   rewards and payoffs included); repeated calls agree bitwise.
2. Detector: the margin rule exactly; a fast approach that crosses the margin within one
   coarse step, a bounce off the limit inside one step, a knot past the range and a
   non-finite knot are flagged; a state resting just inside the range beyond the margin
   is not; over fast approaches toward the rail no row whose 1 ms path goes past the
   range is left unflagged.
3. Gradients: the plant-fit loss through a mixed batch, drift-only and with a learned
   diffusion (propagated covariance), against central differences; the limit port's
   parameters get their gradient from the fine rows.
4. Noise: coarse increments are the sums of the fine increments they span; grids that
   do not nest are rejected.
5. Plumbing: the setting and its validation, both arms pass their ContactSubstepping to
   every model integration, refresh_mode "held" turns it off, protocols, run records,
   the resume fingerprint and the table cell.
"""

import copy
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import algorithms.explicit_ct_ph_sac as explicit_module
import algorithms.implicit_ct_ph_sac as implicit_module
import algorithms.ph_base as base_module
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import protocol_from_config
from common.ph_config import (
    CONTACT_MARGIN_FRACTION,
    CONTACT_VELOCITY_FACTOR,
    contact_substepping_setting,
    model_coarse_step_setting,
)
from environment.ph_sac import tasks
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import reporting
from evaluations.ph_sac import resume as R
from models.ph_sac import rollouts
from models.ph_sac.losses import plant_likelihood_loss
from models.ph_sac.plant import PortHamiltonianPlant
from models.ph_sac.rollouts import (
    contact_flags,
    contact_substepping,
    integrate_endpoint_moments,
    model_contact_substepping,
    path_jacobian_norms,
    sample_brownian_increments,
    stochastic_rollout,
)
from tests.test_ph_algorithms import (
    collect,
    explicit_config,
    implicit_config,
    make_explicit,
    make_implicit,
    windy,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "benchmarks" / "configs" / "ph_sac"
FINE, COARSE, INTERVAL = 0.001, 0.005, 0.01
RAIL = 1.8
MOMENT_KEYS = ("means", "covariances", "step_counts", "step_sizes", "saturation_fractions")
PATH_KEYS = (
    "endpoints", "survived", "failure_times", "durations_used", "running_rewards",
    "failure_payoffs",
)
NETWORK = {
    "hidden_sizes": (16, 16),
    "plant_hidden_sizes": (8, 8),
    "activation": "tanh",
    "mass_floor": 0.01,
    "diffusion_floor": 1e-3,
    "damping_rank": 2,
    "velocity_feature_scale": 2.0 * math.pi,
    "limit_port": True,
}


class Float64:
    def __enter__(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *args):
        torch.set_default_dtype(self.previous)


def table_testbed(env_id, testbed):
    entry = table.find_ph_entry(env_id, "top")
    return testbed_spec(testbed).config_factory(**copy.deepcopy(entry.config["testbed_settings"]))


def cartpole():
    return table_testbed("cartpole-swingup", "cartpole_swingup")


def free_states(rng, count):
    """Cart near the middle of the rail at moderate speed: never near a limit."""
    return np.stack(
        (
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(-math.pi, math.pi, count),
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(-3.0, 3.0, count),
        ),
        axis=-1,
    )


def touching_states(rng, count):
    """Cart within 2 cm of an end of the rail or up to 2 cm past it (the first row 1 cm
    past it), moving outward."""
    side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    offsets = rng.uniform(-0.02, 0.02, count)
    if count:
        offsets[0] = 0.01
    return np.stack(
        (
            side * (RAIL + offsets),
            rng.uniform(-math.pi, math.pi, count),
            side * rng.uniform(1.0, 3.0, count),
            rng.uniform(-3.0, 3.0, count),
        ),
        axis=-1,
    )


def box_decisions(rng, count, *, gentle=False):
    """Decisions in the cart-pole box (offset +-1, stiffness 0-100, damping 0-20); gentle
    ones (stiffness and damping below 1) keep the force inside the clip, so a loss of the
    endpoints is smooth in the parameters."""
    scale = 0.01 if gentle else 1.0
    return np.stack(
        (
            rng.uniform(-1.0, 1.0, count),
            scale * rng.uniform(0.0, 100.0, count),
            scale * rng.uniform(0.0, 20.0, count),
        ),
        axis=-1,
    )


class Setup:
    """The cart-pole (rail limit at +-1.8 m), its continuous controller, the oracle and
    learned plants, and the scheme at 5 ms over 1 ms."""

    def __init__(self, seed=0):
        self.config = cartpole()
        self.env = tasks.make_env(self.config, seed=0)
        self.controller = self.env.controller
        self.spec = self.config["state_spec"]
        self.contact = contact_substepping(self.spec, COARSE, FINE)
        self.oracle = tasks.make_oracle_plant(self.env).double()
        self.rng = np.random.default_rng(seed)

    def learned(self, learn_diffusion, seed=4):
        torch.manual_seed(seed)
        return PortHamiltonianPlant(self.spec, NETWORK, learn_diffusion=learn_diffusion).double()

    def rows(self, free, touching, *, gentle=False):
        states = np.concatenate((free_states(self.rng, free), touching_states(self.rng, touching)))
        order = self.rng.permutation(len(states))
        decisions = box_decisions(self.rng, len(states), gentle=gentle)
        durations = torch.full((len(states), 1), INTERVAL, dtype=torch.float64)
        expected = np.concatenate((np.zeros(free, bool), np.ones(touching, bool)))[order]
        return (
            torch.as_tensor(states[order]),
            torch.as_tensor(decisions),
            durations,
            torch.as_tensor(expected),
        )

    def moments(self, plant, states, decisions, durations, *, step, mode="leading_order",
                create_graph=False, contact=None):
        return integrate_endpoint_moments(
            plant, self.controller, states, decisions, durations,
            maximum_step=step, physics_dt=FINE, controller_refresh_steps=None,
            covariance_mode=mode, create_graph=create_graph, integrator="rk4", contact=contact,
        )

    def norms(self, plant, states, decisions, durations, *, step, contact=None):
        return path_jacobian_norms(
            plant, self.controller, states, decisions, durations,
            maximum_step=step, physics_dt=FINE, controller_refresh_steps=None,
            integrator="rk4", contact=contact,
        )

    def return_terms(self):
        config = self.config
        return {
            "reward_rate": lambda s, u: tasks.reward_rate_tensor(s, u, config),
            "terminal_payoff": lambda s: tasks.terminal_payoff_tensor(s, config),
            "discount_rate": float(config["discount_rate"]),
        }

    def rollout(self, plant, states, decisions, durations, noise, *, step, contact=None):
        config = self.config
        return stochastic_rollout(
            plant, self.controller, states, decisions, durations,
            lambda s: tasks.failure_mask_tensor(s, config),
            maximum_step=step, physics_dt=FINE, controller_refresh_steps=None,
            path_count=noise.shape[1], brownian_increments=noise,
            return_terms=self.return_terms(), integrator="rk4", contact=contact,
        )


def scatter(kept_values, flagged_values, flagged):
    """The row-wise composition the scheme promises, built independently of it."""
    rows = flagged.shape[0]
    shape = (rows,) + tuple(kept_values.shape[1:])
    if kept_values.dim() == 2 and flagged_values.dim() == 2:
        width = max(kept_values.shape[1], flagged_values.shape[1])
        kept_values = torch.nn.functional.pad(kept_values, (0, width - kept_values.shape[1]))
        flagged_values = torch.nn.functional.pad(flagged_values, (0, width - flagged_values.shape[1]))
        shape = (rows, width)
    out = torch.zeros(shape, dtype=kept_values.dtype)
    out[~flagged] = kept_values
    out[flagged] = flagged_values
    return out


# ------------------------------------------------------------------ 1. composition


class TestComposition(unittest.TestCase):
    def test_off_is_the_plain_integration(self):
        with Float64():
            setup = Setup(1)
            states, decisions, durations, _ = setup.rows(4, 4)
            plain = setup.moments(setup.oracle, states, decisions, durations, step=FINE)
            self.assertNotIn("contact_rows", plain)
            again = setup.moments(setup.oracle, states, decisions, durations, step=FINE, contact=None)
            for key in MOMENT_KEYS:
                self.assertTrue(torch.equal(plain[key], again[key]), key)

    def test_no_row_flagged_is_the_plain_coarse_integration(self):
        with Float64():
            setup = Setup(2)
            states, decisions, durations, _ = setup.rows(12, 0)
            for plant, mode in ((setup.oracle, "leading_order"), (setup.learned(True), "propagated"),
                                (setup.learned(False), "leading_order")):
                with self.subTest(plant=type(plant).__name__, mode=mode):
                    got = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode,
                                        contact=setup.contact)
                    self.assertFalse(got["contact_rows"].any())
                    plain = setup.moments(plant, states, decisions, durations, step=COARSE, mode=mode)
                    for key in MOMENT_KEYS:
                        self.assertTrue(torch.equal(got[key], plain[key]), key)
                    self.assertEqual(int(got["step_counts"][0]), round(INTERVAL / COARSE))
            norms = setup.norms(setup.oracle, states, decisions, durations, step=FINE, contact=setup.contact)
            plain = setup.norms(setup.oracle, states, decisions, durations, step=COARSE)
            self.assertTrue(torch.equal(norms, plain))

    def test_every_row_flagged_is_the_plain_fine_integration(self):
        with Float64():
            setup = Setup(3)
            states, decisions, durations, _ = setup.rows(0, 10)
            for plant, mode in ((setup.oracle, "leading_order"), (setup.learned(True), "propagated")):
                with self.subTest(plant=type(plant).__name__, mode=mode):
                    got = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode,
                                        contact=setup.contact)
                    self.assertTrue(got["contact_rows"].all())
                    plain = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode)
                    for key in MOMENT_KEYS:
                        self.assertTrue(torch.equal(got[key], plain[key]), key)
            norms = setup.norms(setup.oracle, states, decisions, durations, step=FINE, contact=setup.contact)
            self.assertTrue(torch.equal(norms, setup.norms(setup.oracle, states, decisions, durations, step=FINE)))

    def test_a_mixed_batch_is_the_row_wise_composition(self):
        with Float64():
            setup = Setup(4)
            states, decisions, durations, expected = setup.rows(9, 7)
            for plant, mode in ((setup.oracle, "leading_order"), (setup.learned(True), "propagated")):
                with self.subTest(plant=type(plant).__name__, mode=mode):
                    got = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode,
                                        contact=setup.contact)
                    flagged = got["contact_rows"]
                    self.assertTrue(torch.equal(flagged, expected))
                    coarse = setup.moments(plant, states, decisions, durations, step=COARSE, mode=mode)
                    fine = setup.moments(plant, states[flagged], decisions[flagged], durations[flagged],
                                         step=FINE, mode=mode)
                    for key in MOMENT_KEYS:
                        composed = scatter(coarse[key][~flagged], fine[key], flagged)
                        self.assertTrue(torch.equal(got[key], composed.to(got[key].dtype)), key)
                    # And the fine rows agree with a fine pass over the whole batch (the same
                    # arithmetic on another batch: to rounding, not necessarily bitwise).
                    whole = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode)
                    torch.testing.assert_close(got["means"][flagged], whole["means"][flagged],
                                               rtol=1e-12, atol=1e-12)
                    # Repeated calls are bitwise equal (the mask and the order are deterministic).
                    again = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode,
                                          contact=setup.contact)
                    for key in MOMENT_KEYS + ("contact_rows",):
                        self.assertTrue(torch.equal(got[key], again[key]), key)
            norms = setup.norms(setup.oracle, states, decisions, durations, step=FINE, contact=setup.contact)
            coarse = setup.norms(setup.oracle, states, decisions, durations, step=COARSE)
            fine = setup.norms(setup.oracle, states[expected], decisions[expected], durations[expected], step=FINE)
            self.assertTrue(torch.equal(norms, scatter(coarse[~expected], fine, expected)))

    def test_rollouts_compose_path_by_path(self):
        """No path flagged: the plain coarse rollout with the summed increments; every
        path flagged: the plain fine rollout; mixed: path by path, rewards on the knots
        of the step each path used."""
        with Float64():
            setup = Setup(5)
            plant = setup.learned(True)
            for free, touching in ((6, 0), (0, 6), (5, 4)):
                with self.subTest(free=free, touching=touching):
                    states, decisions, durations, expected = setup.rows(free, touching)
                    noise = sample_brownian_increments(
                        durations, path_count=3, noise_dim=plant.noise_dim, maximum_step=FINE,
                        physics_dt=FINE, controller_refresh_steps=None, seed=7,
                    )
                    got = setup.rollout(plant, states, decisions, durations, noise, step=FINE,
                                        contact=setup.contact)
                    paths = got["contact_paths"].reshape(-1)
                    self.assertTrue(torch.equal(paths, expected.repeat_interleave(3)))
                    self.assertTrue(torch.equal(got["brownian_increments"], noise))
                    summed = rollouts._coarse_increments(
                        noise, durations, fine_step=FINE, coarse_step=COARSE, physics_dt=FINE
                    )
                    coarse = setup.rollout(plant, states, decisions, durations, summed, step=COARSE)
                    if touching == 0:
                        for key in PATH_KEYS:
                            self.assertTrue(torch.equal(got[key], coarse[key]), key)
                        continue
                    if free == 0:
                        plain = setup.rollout(plant, states, decisions, durations, noise, step=FINE)
                        for key in PATH_KEYS:
                            self.assertTrue(torch.equal(got[key], plain[key]), key)
                        continue
                    owners = torch.div(torch.nonzero(paths).squeeze(-1), 3, rounding_mode="floor")
                    fine = setup.rollout(
                        plant, states[owners], decisions[owners], durations[owners],
                        noise.flatten(0, 1)[paths].unsqueeze(1), step=FINE,
                    )
                    for key in PATH_KEYS:
                        composed = scatter(
                            coarse[key].flatten(0, 1)[~paths], fine[key].flatten(0, 1), paths
                        ).unflatten(0, (states.shape[0], 3))
                        self.assertTrue(torch.equal(got[key], composed.to(got[key].dtype)), key)
                    # Rewards integrated on each path's own knots: the fine paths' differ
                    # from the coarse integration of the same paths.
                    self.assertFalse(torch.equal(got["running_rewards"], coarse["running_rewards"]))

    def test_drawn_noise_is_the_fine_grids(self):
        """Without supplied increments the scheme draws them where the plain fine rollout
        does, so an all-flagged rollout from the same global seed is the plain one."""
        with Float64():
            setup = Setup(6)
            plant = setup.learned(True)
            states, decisions, durations, _ = setup.rows(0, 4)
            config = setup.config
            kwargs = dict(maximum_step=FINE, physics_dt=FINE, controller_refresh_steps=None,
                          path_count=2, integrator="rk4")
            torch.manual_seed(11)
            got = stochastic_rollout(plant, setup.controller, states, decisions, durations,
                                     lambda s: tasks.failure_mask_tensor(s, config),
                                     contact=setup.contact, **kwargs)
            torch.manual_seed(11)
            plain = stochastic_rollout(plant, setup.controller, states, decisions, durations,
                                       lambda s: tasks.failure_mask_tensor(s, config), **kwargs)
            for key in PATH_KEYS + ("brownian_increments",):
                self.assertTrue(torch.equal(got[key], plain[key]), key)


# ------------------------------------------------------------------- 2. detector


def rule(contact, before, after, h):
    """The margin rule of contact_flags, written out in numpy."""
    flagged = np.zeros(len(before), bool)
    for slot, (position, rate) in enumerate(zip(contact.positions, contact.velocities)):
        low, high = contact.lows[slot], contact.highs[slot]
        margin = contact.margins[slot] + contact.velocity_factor * h * np.maximum(
            np.abs(before[:, rate]), np.abs(after[:, rate])
        )
        for knot in (before[:, position], after[:, position]):
            with np.errstate(invalid="ignore"):
                clear = (knot > low + margin) & (knot < high - margin)
            flagged |= ~clear
    return flagged


def cart(x0, v0, x1, v1):
    before = torch.tensor([[x0, 0.0, v0, 0.0]], dtype=torch.float64)
    after = torch.tensor([[x1, 0.0, v1, 0.0]], dtype=torch.float64)
    return before, after


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.spec = cartpole()["state_spec"]
        self.contact = contact_substepping(self.spec, COARSE, FINE)
        self.h = torch.tensor([COARSE], dtype=torch.float64)

    def flag(self, *values):
        before, after = cart(*values)
        return bool(contact_flags(self.contact, before, after, self.h)[0])

    def test_the_scheme_reads_the_state_spec(self):
        contact = self.contact
        self.assertEqual(contact.positions, (0,))
        self.assertEqual(contact.velocities, (2,))
        self.assertEqual((contact.lows, contact.highs), ((-RAIL,), (RAIL,)))
        self.assertAlmostEqual(contact.margins[0], CONTACT_MARGIN_FRACTION * 2 * RAIL, places=15)
        self.assertEqual(contact.velocity_factor, CONTACT_VELOCITY_FACTOR)
        self.assertGreaterEqual(CONTACT_VELOCITY_FACTOR, 1.5)
        self.assertEqual((contact.coarse_step, contact.fine_step), (COARSE, FINE))
        self.assertEqual(contact.record(), {"coarse_step": COARSE, "fine_step": FINE,
                                            "margin_fraction": CONTACT_MARGIN_FRACTION,
                                            "velocity_factor": CONTACT_VELOCITY_FACTOR})
        self.assertIsNone(contact_substepping(self.spec, None, FINE))
        for coarse in (FINE, 0.0015):
            with self.assertRaises(ValueError):
                contact_substepping(self.spec, coarse, FINE)
        with self.assertRaises(ValueError):
            contact_substepping(self.spec, COARSE, FINE, velocity_factor=1.0)
        with self.assertRaises(ValueError):
            contact_substepping(windy()["state_spec"], COARSE, FINE)

    def test_a_fast_approach_across_the_margin_is_flagged(self):
        """10 cm from the rail at 20 m/s: both knots are outside the 3.6 cm absolute
        margin, but the cart crosses it within the step; the rate term catches it."""
        absolute = self.contact.margins[0]
        self.assertGreater(0.10, absolute)
        self.assertTrue(self.flag(RAIL - 0.10, 20.0, RAIL - 0.05, 20.0))
        self.assertTrue(self.flag(-RAIL + 0.10, -20.0, -RAIL + 0.05, -20.0))
        # The same knots at rest are clear: the flag comes from the speed.
        self.assertFalse(self.flag(RAIL - 0.10, 0.0, RAIL - 0.10, 0.0))

    def test_a_bounce_inside_one_step_is_flagged(self):
        """In at 15 m/s, out at 15 m/s, both knots 6 cm inside: the path reached the
        rail between them (tunnelling), which the margin at the knots' speed covers."""
        self.assertTrue(self.flag(RAIL - 0.06, 15.0, RAIL - 0.06, -15.0))

    def test_knots_past_the_range_or_not_finite_are_flagged(self):
        self.assertTrue(self.flag(RAIL + 0.001, 0.0, 0.0, 0.0))
        self.assertTrue(self.flag(0.0, 0.0, -RAIL - 0.001, 0.0))
        self.assertTrue(self.flag(float("nan"), 0.0, 0.0, 0.0))
        self.assertTrue(self.flag(0.0, 0.0, 0.0, float("nan")))
        self.assertTrue(self.flag(0.0, float("inf"), 0.0, 0.0))

    def test_resting_inside_the_range_is_flagged_only_within_the_margin(self):
        """At rest the margin is the absolute one, 1% of the range (3.6 cm): 5 cm inside
        is clear on either side, 2 cm inside is flagged."""
        absolute = self.contact.margins[0]
        for side in (1.0, -1.0):
            self.assertFalse(self.flag(side * (RAIL - 0.05), 0.0, side * (RAIL - 0.05), 0.0))
            self.assertFalse(self.flag(side * (RAIL - absolute - 1e-9), 0.0, side * (RAIL - absolute - 1e-9), 0.0))
            self.assertTrue(self.flag(side * (RAIL - 0.02), 0.0, side * (RAIL - 0.02), 0.0))
            self.assertTrue(self.flag(side * (RAIL - absolute + 1e-9), 0.0, 0.0, 0.0))
        self.assertFalse(self.flag(0.0, 2.0, 0.02, 2.0))

    def test_the_margin_rule_exactly(self):
        rng = np.random.default_rng(8)
        count = 4000
        before = np.stack((rng.uniform(-2.0, 2.0, count), rng.uniform(-3, 3, count),
                           rng.normal(0.0, 5.0, count), rng.normal(0.0, 5.0, count)), axis=-1)
        after = before + np.stack((rng.normal(0.0, 0.05, count), rng.normal(0.0, 0.1, count),
                                   rng.normal(0.0, 2.0, count), rng.normal(0.0, 2.0, count)), axis=-1)
        h = rng.choice([0.002, 0.005, 0.01], count)
        got = contact_flags(self.contact, torch.as_tensor(before), torch.as_tensor(after),
                            torch.as_tensor(h)).numpy()
        expected = rule(self.contact, before, after, h)
        np.testing.assert_array_equal(got, expected)
        self.assertGreater(got.sum(), 100)
        self.assertGreater((~got).sum(), 100)

    def test_two_limited_coordinates(self):
        """The point mass: either slide near its limit flags the row."""
        spec = table_testbed("point_mass-easy", "point_mass_easy")["state_spec"]
        contact = contact_substepping(spec, 0.01, FINE)
        self.assertEqual(contact.positions, (0, 1))
        self.assertEqual(contact.velocities, (2, 3))
        h = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        states = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.285, 0.0, 0.0, 0.0], [0.0, -0.285, 0.0, 0.0]],
                              dtype=torch.float64)
        self.assertEqual(contact_flags(contact, states, states, h).tolist(), [False, True, True])

    def test_no_contact_is_missed_on_fast_approaches(self):
        """Fast approaches toward the rail (up to 15 cm away, up to 10 m/s outward) and
        free rows: every row whose 1 ms oracle path is past the range at any 1 ms knot of
        the interval is flagged by the 5 ms detector."""
        with Float64():
            setup = Setup(9)
            count = 400
            rng = np.random.default_rng(10)
            side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
            states = np.stack(
                (side * (RAIL - rng.uniform(0.0, 0.15, count)), rng.uniform(-math.pi, math.pi, count),
                 side * rng.uniform(0.0, 10.0, count), rng.uniform(-5.0, 5.0, count)), axis=-1)
            states = np.concatenate((states, free_states(rng, 100)))
            states = torch.as_tensor(states)
            decisions = torch.as_tensor(box_decisions(rng, len(states)))
            past = torch.zeros(len(states), dtype=torch.bool)
            for steps in range(0, round(INTERVAL / FINE) + 1):
                if steps == 0:
                    knots = states
                else:
                    durations = torch.full((len(states), 1), steps * FINE, dtype=torch.float64)
                    knots = setup.moments(setup.oracle, states, decisions, durations, step=FINE)["means"]
                past |= (knots[:, 0] > RAIL) | (knots[:, 0] < -RAIL)
            durations = torch.full((len(states), 1), INTERVAL, dtype=torch.float64)
            flagged = setup.moments(setup.oracle, states, decisions, durations, step=FINE,
                                    contact=setup.contact)["contact_rows"]
            self.assertGreater(int(past.sum()), 50)
            self.assertEqual(int((past & ~flagged).sum()), 0)
            self.assertFalse(flagged[-100:].any())


# ------------------------------------------------------------------ 3. gradients


class TestGradients(unittest.TestCase):
    def loss(self, setup, plant, mode, states, decisions, durations, observed):
        moments = setup.moments(plant, states, decisions, durations, step=FINE, mode=mode,
                                create_graph=True, contact=setup.contact)
        batch = {"next_states": observed, "durations": durations}
        return plant_likelihood_loss(batch, moments, setup.spec), moments["contact_rows"]

    def test_the_plant_fit_gradient_through_a_mixed_batch(self):
        with Float64():
            setup = Setup(12)
            states, decisions, durations, expected = setup.rows(5, 4, gentle=True)
            observed = setup.moments(setup.oracle, states, decisions, durations, step=FINE)["means"].detach()
            for learn_diffusion, mode in ((False, "leading_order"), (True, "propagated")):
                with self.subTest(learn_diffusion=learn_diffusion, mode=mode):
                    plant = setup.learned(learn_diffusion)
                    loss, flagged = self.loss(setup, plant, mode, states, decisions, durations, observed)
                    self.assertTrue(torch.equal(flagged, expected))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    parameters = dict(plant.named_parameters())
                    # The port acts only past the rail, on the fine rows.
                    self.assertGreater(parameters["limit_stiffness"].grad.abs().sum().item(), 0.0)
                    for name, index in (
                        ("force_matrix", (1, 0)),
                        ("mass_net.layers.2.bias", (1,)),
                        ("damping_net.layers.0.weight", (0, 1)),
                        ("potential_net.layers.0.weight", (2, 0)),
                        ("limit_stiffness", (0,)),
                        ("limit_damping", (0,)),
                    ):
                        parameter = parameters[name]
                        analytic = parameter.grad[index].item()
                        epsilon = 1e-6
                        with torch.no_grad():
                            parameter[index] += epsilon
                            upper, mask_up = self.loss(setup, plant, mode, states, decisions, durations, observed)
                            parameter[index] -= 2.0 * epsilon
                            lower, mask_down = self.loss(setup, plant, mode, states, decisions, durations, observed)
                            parameter[index] += epsilon
                        self.assertTrue(torch.equal(mask_up, flagged) and torch.equal(mask_down, flagged))
                        numeric = (upper.item() - lower.item()) / (2.0 * epsilon)
                        self.assertAlmostEqual(analytic, numeric, delta=1e-5 * max(1.0, abs(numeric)), msg=name)


# ---------------------------------------------------------------------- 4. noise


class TestNoise(unittest.TestCase):
    def test_coarse_increments_sum_the_fine_ones(self):
        with Float64():
            durations = torch.tensor([[0.01], [0.01], [0.02]], dtype=torch.float64)
            fine = sample_brownian_increments(durations, path_count=2, noise_dim=3, maximum_step=FINE,
                                              physics_dt=FINE, seed=3)
            coarse = rollouts._coarse_increments(fine, durations, fine_step=FINE, coarse_step=COARSE,
                                                 physics_dt=FINE)
            self.assertEqual(tuple(coarse.shape), (3, 2, 4, 3))
            for row, count in enumerate((2, 2, 4)):
                expected = fine[row, :, : 5 * count].unflatten(1, (count, 5)).sum(2)
                torch.testing.assert_close(coarse[row, :, :count], expected, rtol=1e-14, atol=1e-15)
                self.assertFalse(coarse[row, :, count:].any())

    def test_float32_grids_nest(self):
        """Training batches carry float32 durations; the point mass's 20 ms interval at 1 ms
        and 10 ms then has knots ~1e-6 physics steps apart (the first cluster job's failure),
        which still nest."""
        durations = torch.full((4, 1), 0.02, dtype=torch.float32)
        fine = sample_brownian_increments(durations, path_count=2, noise_dim=1, maximum_step=FINE,
                                          physics_dt=FINE, seed=4)
        coarse = rollouts._coarse_increments(fine, durations, fine_step=FINE, coarse_step=0.01,
                                             physics_dt=FINE)
        self.assertEqual(tuple(coarse.shape), (4, 2, 2, 1))
        torch.testing.assert_close(coarse, fine.unflatten(2, (2, 10)).sum(3), rtol=1e-6, atol=1e-7)

    def test_grids_that_do_not_nest_are_rejected(self):
        durations = torch.tensor([[0.007]], dtype=torch.float64)
        fine = torch.zeros(1, 1, 7, 1, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "nest"):
            rollouts._coarse_increments(fine, durations, fine_step=FINE, coarse_step=COARSE, physics_dt=FINE)


# -------------------------------------------------------------------- 5. plumbing


class TestPlumbing(unittest.TestCase):
    def test_setting_and_validation(self):
        self.assertIsNone(model_coarse_step_setting({}))
        self.assertIsNone(model_coarse_step_setting({"model_coarse_step": None}))
        self.assertEqual(model_coarse_step_setting({"model_coarse_step": 0.005}), 0.005)
        for bad in (True, "0.005", 0.0, -0.005, float("inf")):
            with self.assertRaisesRegex(ValueError, "model_coarse_step"):
                model_coarse_step_setting({"model_coarse_step": bad})
        testbed = cartpole()
        config = implicit_config(testbed, model_coarse_step=COARSE)
        self.assertEqual(contact_substepping_setting(config), {
            "coarse_step": COARSE, "fine_step": FINE, "margin_fraction": CONTACT_MARGIN_FRACTION,
            "velocity_factor": CONTACT_VELOCITY_FACTOR})
        self.assertIsNone(contact_substepping_setting(implicit_config(testbed)))
        self.assertIsNone(contact_substepping_setting(implicit_config(testbed, model_coarse_step=None)))
        # A whole multiple (at least 2) of model_step that divides the interval and episode.
        for bad in (0.001, 0.0015, 0.003, 0.004):
            with self.assertRaisesRegex(ValueError, "model_coarse_step"):
                implicit_config(testbed, model_coarse_step=bad)
        explicit_config(testbed, model_coarse_step=0.01)
        # No limited coordinates (and a two-tailed clock): nothing to watch.
        with self.assertRaisesRegex(ValueError, "model_coarse_step"):
            implicit_config(windy(), model_coarse_step=COARSE)

    def test_held_controllers_turn_it_off(self):
        testbed = cartpole()
        testbed["controller"] = {**testbed["controller"], "refresh_mode": "held"}
        config = implicit_config(testbed, model_coarse_step=COARSE)
        self.assertIsNone(contact_substepping_setting(config))
        agent = make_implicit(config)
        self.assertIsNone(agent.contact)

    def test_agents_pass_their_scheme_to_every_integration(self):
        testbed = cartpole()
        agent = make_implicit(implicit_config(testbed, model_coarse_step=COARSE))
        self.assertIsInstance(agent.contact, rollouts.ContactSubstepping)
        self.assertEqual((agent.contact.coarse_step, agent.contact.fine_step), (COARSE, FINE))
        self.assertEqual(
            agent.contact.record(),
            model_contact_substepping(agent.config, agent.state_spec, agent.controller).record(),
        )
        collect(agent, 40)
        batch = agent.replay.sample("train", 8)
        with mock.patch.object(
            implicit_module, "integrate_endpoint_moments", wraps=implicit_module.integrate_endpoint_moments
        ) as moments, mock.patch.object(
            implicit_module, "stochastic_rollout", wraps=implicit_module.stochastic_rollout
        ) as rollout, mock.patch.object(
            implicit_module, "paired_gate_samples", wraps=implicit_module.paired_gate_samples
        ) as gate, mock.patch.object(
            base_module, "integrate_endpoint_moments", wraps=base_module.integrate_endpoint_moments
        ) as diagnostics, mock.patch.object(
            base_module, "path_jacobian_norms", wraps=base_module.path_jacobian_norms
        ) as norms:
            agent.update_online_plant(batch)
            agent.fit_rate_head(1)
            agent._gate_samples(agent.replay.sample("train", 4), 2)
            agent.record_plant_diagnostics(batch)
        for spy in (moments, rollout, gate, diagnostics, norms):
            spy.assert_called()
            self.assertIs(spy.call_args.kwargs["contact"], agent.contact)

        agent = make_explicit(explicit_config(testbed, continuation="rollout", model_coarse_step=COARSE))
        self.assertIsInstance(agent.contact, rollouts.ContactSubstepping)
        collect(agent, 20)
        batch = agent.replay.sample("train", 8)
        with mock.patch.object(
            explicit_module, "integrate_endpoint_moments", wraps=explicit_module.integrate_endpoint_moments
        ) as moments, mock.patch.object(
            explicit_module, "stochastic_rollout", wraps=explicit_module.stochastic_rollout
        ) as rollout:
            agent.update_plant(batch)
            agent.build_critic_target(batch)
        for spy in (moments, rollout):
            spy.assert_called_once()
            self.assertIs(spy.call_args.kwargs["contact"], agent.contact)
        # Off: None everywhere.
        self.assertIsNone(make_implicit(implicit_config(testbed)).contact)

    def test_statistics_count_the_rows(self):
        with Float64():
            setup = Setup(13)
            states, decisions, durations, expected = setup.rows(3, 2)
            rollouts.reset_contact_statistics()
            setup.moments(setup.oracle, states, decisions, durations, step=FINE, contact=setup.contact)
            self.assertEqual(rollouts.contact_statistics(), {"endpoint_moments": (5, int(expected.sum()))})
            rollouts.reset_contact_statistics()
            self.assertEqual(rollouts.contact_statistics(), {})

    def test_protocols_records_fingerprints_and_cells(self):
        with open(CONFIGS / "cartpole_swingup.json", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(raw["model_coarse_step"], COARSE)
        self.assertIn("model_coarse_step", raw["_experiment_choices"])
        # The file is under contact exclusion since 2026-11-11, where the scheme is off
        # (every integration at the coarse step); it is checked on the port convention.
        self.assertEqual(raw.pop("contact_handling"), "exclude")
        excluded = protocol_from_config(dict(raw, contact_handling="exclude"), "x")
        self.assertIsNone(contact_substepping_setting(excluded["implicit_config"]))
        raw["network"]["limit_port"] = True
        proto = protocol_from_config(raw, "x")
        for name in ("explicit_config", "implicit_config"):
            self.assertEqual(proto[name]["model_coarse_step"], COARSE)
        off = copy.deepcopy(raw)
        del off["model_coarse_step"]
        proto_off = protocol_from_config(off, "x")
        self.assertNotIn("model_coarse_step", proto_off["implicit_config"])
        null = dict(copy.deepcopy(raw), model_coarse_step=None)
        self.assertIsNone(contact_substepping_setting(protocol_from_config(null, "x")["implicit_config"]))
        with self.assertRaisesRegex(ValueError, "model_coarse_step"):
            protocol_from_config(dict(copy.deepcopy(raw), model_coarse_step=0.003), "x")
        mixed = copy.deepcopy(proto)
        mixed["implicit_config"]["model_coarse_step"] = None
        with self.assertRaisesRegex(ValueError, "model_coarse_step"):
            P.validate_matched_quantities(mixed)
        # Run records: the arm's setting, None for the CT-SAC baseline; old records load as off.
        proto["arms"] = list(P.all_arms())
        runs = P.build_run_matrix(proto)
        for run in runs:
            setting = P._arm_contact_substepping(run)
            if run["arm"]["algorithm"] == "ct_sac":
                self.assertIsNone(setting)
            else:
                self.assertEqual(setting["coarse_step"], COARSE, run["arm"]["name"])
        self.assertIsNone(reporting.record_contact_substepping({}))
        self.assertEqual(reporting.record_contact_substepping({"contact_substepping": {"coarse_step": 0.005}}),
                         {"coarse_step": 0.005})
        # The resume fingerprint carries the setting only when it is on.
        run = next(r for r in runs if r["arm"]["name"] == "implicit_learned_admissions")
        self.assertIn("contact_substepping", R.run_settings(run)["resolved"]["implicit_config"])
        proto_off["arms"] = list(P.all_arms())
        run_off = next(r for r in P.build_run_matrix(proto_off) if r["arm"]["name"] == "implicit_learned_admissions")
        self.assertNotIn("contact_substepping", R.run_settings(run_off)["resolved"]["implicit_config"])
        self.assertNotEqual(R.run_fingerprint(run), R.run_fingerprint(run_off))
        # The table cell round-trips; an empty cell is off.
        entry = table.find_ph_entry("cartpole-swingup", "top")
        body = copy.deepcopy(raw)
        body.pop("_experiment_choices")
        row = table.config_to_row(body, env_id="cartpole-swingup", mode="top", task=entry.task)
        self.assertEqual(row["model_coarse_step"], "0.005")
        self.assertEqual(table.parse_ph_row(row).config["model_coarse_step"], COARSE)
        row["model_coarse_step"] = ""
        self.assertNotIn("model_coarse_step", table.parse_ph_row(row).config)


if __name__ == "__main__":
    unittest.main()
