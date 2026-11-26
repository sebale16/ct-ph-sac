"""Contact exclusion's damped reach bound (start-state rule version 2, 2026-11-10 evening).

models/ph_sac/rollouts.reach_travel bounds the travel over an interval Delta of the damped
double integrator ydd = s(t) a - k yd, |s(t)| <= 1, from speed |yd(0)| = v:
    x = max(v Delta, v Delta phi1(k Delta) + a Delta^2 phi2(k Delta)),
    phi1(z) = (1 - e^-z) / z, phi2(z) = (z - 1 + e^-z) / z^2,
and the start-state rule's margin is mu = m + reach_travel(beta_s |qdot|, Delta, a, k).

1. The bound against a brute-force simulation (RK4 at Delta / 4000) of the damped double
   integrator over random (v0, a, k = d / m, Delta), both signs, several input histories
   (full push, braking, reversal, random levels, random bang-bang), including the
   undamped limit and starts above the terminal speed a / k; it is attained by the full
   push below the terminal speed; torch and numpy agree; the series and the closed form
   agree at the switch; monotone in v, a, Delta and decreasing in k.
2. The undamped limit is rule version 1 (m + beta_s |v| Delta + a Delta^2 / 2).
3. The three tasks' margins pinned (reacher wrist, cart rail, point-mass slides), old and
   new at zero speed; the reference pose qpos0 maximizes the decay rate and the terminal
   speed over the poses, which makes it the conservative choice for the rise.
4. Plumbing: the agents read the decay rates; the rule's version is in the setting, the
   records and the resume fingerprint of exclude runs only; a resumable checkpoint
   written under rule version 1 is refused with a clear message; port-mode training does
   not read the flags.
"""

import copy
import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import environment.ph_sac.dmc  # noqa: F401
from benchmarks import ph_hyperparams as table
from benchmarks import ph_plant_schedule_drift as harness
from common.ph_config import (
    EXCLUSION_RULE_VERSION,
    contact_exclusion_setting,
)
from environment.ph_sac import tasks
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from models.ph_sac import rollouts
from models.ph_sac.rollouts import (
    contact_exclusion,
    reach_travel,
    reach_travel_numpy,
    start_state_flag_numpy,
    start_state_flags,
    start_state_margins_numpy,
)
from tests.test_ph_algorithms import collect, implicit_config, make_implicit
from tests.test_ph_contact_exclusion import exclude_implicit, flagged, forced_build
from tests.test_ph_contact_substep import COARSE, cartpole
from tests.test_ph_resume import run_captured

SIM_STEPS = 4000


def table_env(env_id, testbed):
    entry = table.find_ph_entry(env_id, "top")
    config = testbed_spec(testbed).config_factory(**copy.deepcopy(entry.config["testbed_settings"]))
    return tasks.make_env(config, decision_mode="feedback", seed=0), config


def rules(env):
    """(old, new): rule version 1 (no damping) and 2 on env's limited coordinates."""
    bounds = env.motor_acceleration_bounds()
    old = contact_exclusion(env.state_spec, bounds, active=True, model_step=0.001)
    new = contact_exclusion(env.state_spec, bounds, decay_rates=env.velocity_decay_rates(),
                            active=True, model_step=0.001)
    return old, new


def simulate(v0, a, k, delta, levels):
    """Brute force: max_t |y(t)| and y(Delta) of ydd = s a - k yd, y(0) = 0, yd(0) = v0.

    levels [S, n_pieces]: the input s, piecewise constant on n_pieces equal pieces of the
    interval. RK4 at Delta / SIM_STEPS, s constant within a step; every array is [S].
    """
    pieces = levels.shape[1]
    h = delta / SIM_STEPS
    y = np.zeros_like(v0)
    v = v0.astype(np.float64).copy()
    largest = np.zeros_like(v0)
    for step in range(SIM_STEPS):
        s = levels[:, min(step * pieces // SIM_STEPS, pieces - 1)]
        force = s * a

        def rate(vel):
            return force - k * vel

        k1y, k1v = v, rate(v)
        k2y, k2v = v + 0.5 * h * k1v, rate(v + 0.5 * h * k1v)
        k3y, k3v = v + 0.5 * h * k2v, rate(v + 0.5 * h * k2v)
        k4y, k4v = v + h * k3v, rate(v + h * k3v)
        y = y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        v = v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        largest = np.maximum(largest, np.abs(y))
    return largest, y


def draw_cases(rng, count):
    """Random (v0 >= 0, a, k, Delta): a fifth undamped, a tenth from rest, about half of
    the damped ones above the terminal speed a / k."""
    a = 10.0 ** rng.uniform(-2.0, 4.0, count)
    k = 10.0 ** rng.uniform(-3.0, 3.0, count)
    k[: count // 5] = 0.0
    delta = rng.uniform(1e-3, 0.2, count)
    terminal = np.where(k > 0.0, a / np.where(k > 0.0, k, 1.0), a * delta)
    v0 = terminal * np.where(rng.random(count) < 0.5, rng.uniform(0.0, 1.0, count),
                             rng.uniform(1.0, 4.0, count))
    v0[rng.random(count) < 0.1] = 0.0
    order = rng.permutation(count)
    return v0[order], a[order], k[order], delta[order]


# ------------------------------------------------------------------- 1. the bound


class TestReachBound(unittest.TestCase):
    def test_bound_against_brute_force(self):
        rng = np.random.default_rng(20261110)
        count = 3000
        v0, a, k, delta = draw_cases(rng, count)
        bound = reach_travel_numpy(v0, delta, a, k)
        ones = np.ones((count, 1))
        histories = {
            "push": (v0, ones),
            "brake": (v0, -ones),
            "reverse": (-v0, ones),
            "levels": (v0 * np.where(rng.random(count) < 0.5, 1.0, -1.0),
                       rng.uniform(-1.0, 1.0, (count, 8))),
            "bang": (v0 * np.where(rng.random(count) < 0.5, 1.0, -1.0),
                     np.where(rng.random((count, 16)) < 0.5, -1.0, 1.0)),
        }
        worst = {}
        for name, (start, levels) in histories.items():
            largest, end = simulate(start, a, k, delta, levels)
            ratio = largest / np.maximum(bound, 1e-300)
            worst[name] = float(ratio.max())
            with self.subTest(history=name):
                self.assertTrue(np.all(np.isfinite(largest)))
                excess = largest - bound * (1.0 + 1e-7) - 1e-13
                self.assertLessEqual(float(excess.max()), 0.0,
                                     f"{name}: case {int(excess.argmax())} exceeds the bound")
            if name == "push":
                push_largest, push_end = largest, end
        print("reach bound: largest simulated / bound per history:",
              {k_: round(v_, 9) for k_, v_ in worst.items()})
        # Below the terminal speed the full push attains the bound; above it the travel is
        # W Delta + (v0 - W) Delta phi1(z) (the closed form of the derivation), below v0 Delta.
        z = k * delta
        terminal = np.where(k > 0.0, a / np.where(k > 0.0, k, 1.0), np.inf)
        below = v0 <= terminal
        np.testing.assert_allclose(push_end[below], bound[below], rtol=1e-6, atol=1e-14)
        above = ~below
        self.assertGreater(int(above.sum()), count // 5)
        phi1 = -np.expm1(-z[above]) / z[above]
        exact = terminal[above] * delta[above] + (v0[above] - terminal[above]) * delta[above] * phi1
        np.testing.assert_allclose(push_end[above], exact, rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(push_largest, np.abs(push_end), rtol=1e-12, atol=1e-15)
        self.assertTrue(np.all(exact <= v0[above] * delta[above] * (1.0 + 1e-12)))

    def test_undamped_limit_and_series(self):
        rng = np.random.default_rng(1)
        v, a, delta = rng.uniform(0, 10, 200), rng.uniform(0, 100, 200), rng.uniform(1e-3, 0.1, 200)
        zero = reach_travel_numpy(v, delta, a, np.zeros(200))
        np.testing.assert_allclose(zero, v * delta + 0.5 * a * delta * delta, rtol=1e-15, atol=0)
        tiny = reach_travel_numpy(v, delta, a, np.full(200, 1e-9))
        np.testing.assert_allclose(tiny, zero, rtol=1e-9)
        # Continuity where the series hands over to the closed form (z = 1e-3).
        switch = rollouts._REACH_SERIES_BELOW
        for d in (0.01, 0.02):
            k_below = np.array([switch * (1.0 - 1e-9) / d])
            k_above = np.array([switch * (1.0 + 1e-9) / d])
            for speed, accel in ((0.0, 50.0), (3.0, 0.0), (2.0, 7.0)):
                left = reach_travel_numpy(speed, d, accel, k_below)
                right = reach_travel_numpy(speed, d, accel, k_above)
                np.testing.assert_allclose(left, right, rtol=1e-11)

    def test_torch_and_numpy_agree(self):
        rng = np.random.default_rng(2)
        v0, a, k, delta = draw_cases(rng, 500)
        k[:20] = rng.uniform(0, 2e-3, 20) / delta[:20]  # around the series switch
        expected = reach_travel_numpy(v0, delta, a, k)
        got = reach_travel(*(torch.as_tensor(x, dtype=torch.float64) for x in (v0, delta, a, k)))
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-14, atol=0)
        signed = reach_travel_numpy(-v0, delta, a, k)
        np.testing.assert_array_equal(signed, expected)

    def test_monotone(self):
        rng = np.random.default_rng(3)
        v0, a, k, delta = draw_cases(rng, 1000)
        base = reach_travel_numpy(v0, delta, a, k)
        up = 1.0 + rng.uniform(0.01, 1.0, 1000)
        self.assertTrue(np.all(reach_travel_numpy(v0 * up, delta, a, k) >= base))
        self.assertTrue(np.all(reach_travel_numpy(v0, delta, a * up, k) >= base))
        self.assertTrue(np.all(reach_travel_numpy(v0, delta * up, a, k) >= base))
        self.assertTrue(np.all(reach_travel_numpy(v0, delta, a, k * up) <= base * (1.0 + 1e-15)))
        # Never below the held start speed, never above the undamped bound.
        self.assertTrue(np.all(base >= v0 * delta))
        self.assertTrue(np.all(base <= (v0 * delta + 0.5 * a * delta * delta) * (1.0 + 1e-14)))


# -------------------------------------------------------- 2. undamped = rule version 1


class TestUndampedIsVersionOne(unittest.TestCase):
    def test_margins_without_damping(self):
        spec = cartpole()["state_spec"]
        rule = contact_exclusion(spec, np.array([10.0, 0.0]), active=True, model_step=COARSE)
        self.assertEqual(rule.decay_rates, (0.0,))
        speeds = np.array([[0.0], [2.0], [-3.0]])
        for delta in (0.01, 0.03):
            expected = 0.036 + 1.5 * np.abs(speeds) * delta + 0.5 * 20.0 * delta * delta
            np.testing.assert_allclose(start_state_margins_numpy(rule, speeds, delta), expected,
                                       rtol=1e-15)
        with self.assertRaisesRegex(ValueError, "decay_rates"):
            contact_exclusion(spec, np.array([10.0, 0.0]), decay_rates=np.array([-1.0, 0.0]),
                              active=True, model_step=COARSE)


# ------------------------------------------------------------- 3. the three tasks


TASKS = {
    # env_id: (testbed, interval, limited coordinates)
    "reacher-easy": ("reacher_easy", 0.02, 1),
    "reacher-hard": ("reacher_hard", 0.02, 1),
    "cartpole-swingup": ("cartpole_swingup", 0.01, 1),
    "point_mass-easy": ("point_mass_easy", 0.02, 2),
}
# (a_i, k_i, m_i, old mu at rest, new mu at rest) per limited coordinate: hand-computed from
# the compiled geometry (capsule inertias as MuJoCo computes them) and agreeing to 1e-5 with
# the values Vista test job 1135791 printed, which are pinned here.
PINNED = {
    "reacher": (2703.770564, 207.40548732, 0.01 * 2.0 * math.radians(160.0), 0.5966046, 0.2547129),
    "cartpole": (19.467041, 4.866760272e-4, 0.036, 0.036 + 0.5 * 19.467041e-4, 0.036 + 0.5 * 19.467041e-4),
    "point_mass": (2.0 * 0.1 / 0.3, 1.0 / 0.3, 0.0058, 0.0058 + 0.5 * (0.2 / 0.3) * 4e-4,
                   0.00593041910),
}


class TestTaskMargins(unittest.TestCase):
    def test_pinned_margins_at_rest(self):
        lines = []
        for env_id, (testbed, interval, count) in TASKS.items():
            env, config = table_env(env_id, testbed)
            self.assertAlmostEqual(float(config["reference_period"]), interval)
            old, new = rules(env)
            self.assertEqual(len(new.positions), count)
            a, k, m, mu_old, mu_new = PINNED[env_id.split("-")[0]]
            rest = np.zeros((1, count))
            got_old = start_state_margins_numpy(old, rest, interval)[0]
            got_new = start_state_margins_numpy(new, rest, interval)[0]
            for i in range(count):
                with self.subTest(env_id=env_id, coordinate=i):
                    self.assertAlmostEqual(new.accelerations[i] / a, 1.0, delta=1e-6)
                    self.assertAlmostEqual(new.decay_rates[i] / k, 1.0, delta=1e-6)
                    self.assertAlmostEqual(new.margins[i] / m, 1.0, delta=1e-12)
                    self.assertAlmostEqual(got_old[i] / mu_old, 1.0, delta=2e-6)
                    self.assertAlmostEqual(got_new[i] / mu_new, 1.0, delta=2e-6)
                    self.assertLessEqual(got_new[i], got_old[i])
                    self.assertEqual(old.decay_rates[i], 0.0)
            lines.append(
                f"{env_id}: interval {interval} s, a {list(np.round(new.accelerations, 6))}, "
                f"k {list(np.round(new.decay_rates, 8))} 1/s, m {list(np.round(new.margins, 8))}, "
                f"terminal speed a/k {[round(x / y, 4) if y > 0 else None for x, y in zip(new.accelerations, new.decay_rates)]}, "
                f"mu at rest old {list(np.round(got_old, 7))} new {list(np.round(got_new, 7))}"
            )
        print("\n".join(["task margins:"] + lines))

    def test_reacher_examples(self):
        env, _ = table_env("reacher-easy", "reacher_easy")
        old, new = rules(env)
        edge = math.radians(160.0)
        cases = (  # wrist position, wrist rate, old flag, new flag
            (edge - 0.40, 0.0, True, False),
            (-edge + 0.40, 0.0, True, False),
            (edge - 0.20, 0.0, True, True),
            (edge - 0.40, 12.0, True, True),
            (edge - 0.40, -12.0, True, True),
            (0.0, 20.0, False, False),
            (edge + 0.01, 0.0, True, True),
            (float("nan"), 0.0, True, True),
        )
        for wrist, rate, expect_old, expect_new in cases:
            state = np.array([0.3, wrist, 0.1, 0.05, 2.0, rate])
            with self.subTest(wrist=wrist, rate=rate):
                self.assertEqual(start_state_flag_numpy(old, state, 0.02), expect_old)
                self.assertEqual(start_state_flag_numpy(new, state, 0.02), expect_new)
                tensor = start_state_flags(new, torch.as_tensor(state).unsqueeze(0),
                                           torch.tensor([[0.02]], dtype=torch.float64))
                self.assertEqual(bool(tensor[0]), expect_new)

    def test_reference_pose_is_the_conservative_pose(self):
        """The rise term uses a_i and k_i at qpos0; it bounds every pose's rise when qpos0
        has the largest decay rate k(q) and the largest terminal speed a(q) / k(q)
        (k(q) (W(q) - v) <= k0 (W0 - v) for 0 <= v <= W0). Checked on a pose grid."""
        report = []
        for env_id, testbed, grid in (
            ("reacher-easy", "reacher_easy",
             [np.array([s, w]) for s in (0.0, 1.0) for w in np.linspace(-math.radians(160), math.radians(160), 161)]),
            ("cartpole-swingup", "cartpole_swingup",
             [np.array([x, t]) for x in (-1.5, 0.0) for t in np.linspace(-math.pi, math.pi, 181)]),
            ("point_mass-easy", "point_mass_easy",
             [np.array([x, y]) for x in (-0.29, 0.0, 0.29) for y in (-0.29, 0.0, 0.29)]),
        ):
            env, _ = table_env(env_id, testbed)
            rule = rules(env)[1]
            dofs = [v - int(env.state_spec["position_dim"]) for v in rule.velocities]
            k0 = env.velocity_decay_rates()[dofs]
            a0 = env.motor_acceleration_bounds()[dofs]
            ks, ws = [], []
            for qpos in grid:
                k = env.velocity_decay_rates(qpos)[dofs]
                a = env.motor_acceleration_bounds(qpos)[dofs]
                ks.append(k)
                ws.append(a / k)
                with self.subTest(env_id=env_id, qpos=qpos.tolist()):
                    self.assertTrue(np.all(k <= k0 * (1.0 + 1e-9)))
                    self.assertTrue(np.all(a / k <= (a0 / k0) * (1.0 + 1e-9)))
            report.append(f"{env_id}: k(q0) {k0.tolist()}, k over poses [{np.min(ks):.6g}, {np.max(ks):.6g}], "
                          f"W(q0) {(a0 / k0).tolist()}, W over poses [{np.min(ws):.6g}, {np.max(ws):.6g}]")
        print("\n".join(["pose dependence:"] + report))


# ------------------------------------------------------------------ 4. plumbing


def tiny_exclude_run(output, budget):
    args = harness.parse_args(["cartpole-swingup", "dense", str(output), "--experiment", "--tiny",
                               "--contact-handling", "exclude", "--arm", "implicit_learned_admissions",
                               "--budget", str(budget), "--plant-diagnostics-every", "25"])
    run = harness.experiment_run(args)
    run["protocol"]["evaluation_every_interactions"] = 40
    return run


class TestPlumbing(unittest.TestCase):
    def test_agents_read_the_decay_rates(self):
        agent = exclude_implicit()
        rates = agent.env.velocity_decay_rates()
        self.assertEqual(rates.shape, (2,))
        self.assertEqual(agent.exclusion.decay_rates, (float(rates[0]),))
        self.assertGreater(agent.exclusion.decay_rates[0], 0.0)
        record = agent.exclusion.record()
        self.assertEqual(record["rule_version"], EXCLUSION_RULE_VERSION)
        self.assertEqual(record["decay_rates"], [float(rates[0])])
        port = make_implicit(implicit_config(cartpole(), model_coarse_step=COARSE))
        self.assertEqual(port.exclusion.decay_rates, agent.exclusion.decay_rates)

    def test_rule_version_in_exclude_settings_only(self):
        self.assertEqual(EXCLUSION_RULE_VERSION, 2)
        config = implicit_config(cartpole(), model_coarse_step=COARSE, contact_handling="exclude")
        self.assertEqual(contact_exclusion_setting(config)["rule_version"], 2)
        self.assertIsNone(contact_exclusion_setting(implicit_config(cartpole(), model_coarse_step=COARSE)))
        with tempfile.TemporaryDirectory() as out:
            run = tiny_exclude_run(out, 60)
            port = harness.experiment_run(harness.parse_args(
                ["cartpole-swingup", "dense", out, "--experiment", "--tiny", "--budget", "60"]))
            resolved = R.run_settings(run)["resolved"]
            self.assertEqual(resolved["implicit_config"]["contact_handling"]["rule_version"], 2)
            for name in ("explicit_config", "implicit_config"):
                self.assertNotIn("contact_handling", R.run_settings(port)["resolved"][name])

    def test_resume_written_under_rule_one_is_refused(self):
        def version_one(config):
            setting = contact_exclusion_setting(config)
            if setting is None:
                return None
            return {key: value for key, value in setting.items() if key != "rule_version"}

        with mock.patch.object(P, "make_training_agent", forced_build()), \
                tempfile.TemporaryDirectory() as out:
            run = tiny_exclude_run(out, 200)
            stopped, _ = run_captured(run, R.resume_options(0, stop_after_interactions=150))
            self.assertIsInstance(stopped, R.ResumableStop)
            path = P.run_output_directory(run) / R.RESUME_NAME
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["fingerprint"], R.run_fingerprint(run))
            with mock.patch.object(R, "contact_exclusion_setting", version_one):
                old_fingerprint = R.run_fingerprint(run)
                old_settings = R.run_settings(run)
            self.assertNotEqual(old_fingerprint, payload["fingerprint"])
            payload["fingerprint"], payload["settings"] = old_fingerprint, old_settings
            torch.save(payload, path)
            with self.assertRaises(R.ResumeMismatch) as caught:
                run_captured(run, R.resume_options(0))
            message = str(caught.exception)
            print("refusal message:", message)
            self.assertIn("contact_handling.rule_version", message)
            self.assertIn("start-state rule changed", message)
            # The checkpoint written under this rule continues to the end.
            payload["fingerprint"], payload["settings"] = R.run_fingerprint(run), R.run_settings(run)
            torch.save(payload, path)
            finished, _ = run_captured(run, R.resume_options(0))
            self.assertEqual(finished["counters"]["interactions"], 200)
            self.assertEqual(finished["contact_handling"]["rule_version"], 2)

    def test_port_training_does_not_read_the_flags(self):
        first = make_implicit(implicit_config(cartpole(), model_coarse_step=COARSE))
        second = make_implicit(implicit_config(cartpole(), model_coarse_step=COARSE))
        collect(first, 40)
        collect(second, 40)
        batch = first.replay.sample("train", 8)
        second.replay.sample("train", 8)
        none, every = flagged(batch, ()), flagged(batch, range(8))
        self.assertEqual(first.update_online_plant(none), second.update_online_plant(every))
        for p, q in zip(first.online_plant.parameters(), second.online_plant.parameters()):
            self.assertTrue(torch.equal(p, q))


if __name__ == "__main__":
    unittest.main()
