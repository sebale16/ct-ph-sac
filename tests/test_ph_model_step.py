"""The per-task model step h: common.ph_config.MODEL_STEPS, the table's model_step column.

The model step is the longest substep of every model integration of the plant (plant fit,
explicit rollout target, rate rollouts, gate paths). Under refresh_mode "continuous" a
decision interval splits into equal substeps of at most h (models/ph_sac/rollouts.py
_continuous_grid), so a step above the 1 ms physics step is effective and is a coarser
RK4 of the same closed loop.

1. Every row's model_step is a whole number of physics steps and divides every duration
   its clock can draw, so the grid's substeps are all exactly h.
2. Every row's model_step lies at or below the measured admissibility bound of its task
   (MEASURED below; on tasks with joint limits also LIMIT_CONTACT), the model step is
   coarsened only on tasks without joint limits (a task whose state spec declares
   limited_position_ranges integrates at its physics step), the windy pendulum and
   in-house Acrobot rows (Euler physics and the semi_implicit map; testbeds not in
   MODEL_STEPS) stay at the physics step, and the table, MODEL_STEPS, the factories and
   the JSON configs agree.
3. lqr-lqr_6_2's 10 ms reaches the algorithm: the protocol built from the row carries it,
   and the implicit agent's plant fit integrates a 30 ms interval in 3 substeps of 10 ms.
4. Contact-triggered fine substepping (model_coarse_step, since 2026-11-10): the per-task
   coarse steps (common.ph_config.MODEL_COARSE_STEPS) are pinned, every row with joint
   limits either integrates every interval at its physics step or uses the scheme with a
   coarse step at most its free-state bound (MEASURED) that divides its interval, and
   rows without joint limits never set it.
"""

import copy
import unittest
from pathlib import Path
from unittest import mock

import torch

import algorithms.implicit_ct_ph_sac as implicit_module
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import _build_testbed, load_protocol, protocol_from_config
from common.ph_config import (
    CONTACT_MARGIN_FRACTION,
    CONTACT_VELOCITY_FACTOR,
    MODEL_COARSE_STEPS,
    MODEL_STEPS,
    contact_substepping_setting,
    default_model_coarse_step,
    default_model_step,
    exclusion_model_step,
    testbed_model_coarse_step,
    testbed_model_step,
)
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from models.ph_sac.rollouts import integration_grid

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "benchmarks" / "configs" / "ph_sac"

# Measured admissibility bounds (seconds), from the 2026-11-09 measurement under
# refresh_mode "continuous" (scratch scripts continuous_ctrl/step/admissible_step.py,
# out/<env_id>.json; the "1e-3" column, uniform decisions, of summary_raw.tsv). Error: the
# relative endpoint error ||x_h - x_env|| / ||x_env - x_0|| (wrapped residuals) of the
# oracle plant's model endpoint over one decision interval (integrate_endpoint_moments,
# rk4, the law at every stage, at most h per substep) against the environment's (1 ms
# MuJoCo RK4, the law at every stage), on 2000 rows: states from 3 s rollouts of uniform
# decisions from the reset law, decisions uniform in the box with half of them at stiffness
# and damping in the top 10% of their range; terminated rows left out. Steps tried: 1, 2,
# 5, 10, 15, 20, 30 ms, those dividing the interval. The bound is the largest tried step
# whose 99th percentile stays below 1e-3. The rows then had failure boundaries where
# cart-pole, reacher and point mass now keep soft joint limits (2026-11-10); these bounds
# do not cover intervals that touch those limits (LIMIT_CONTACT does).
#   env_id: (bound, p99 at the bound, next tried step or None, p99 there)
MEASURED = {
    "pendulum-swingup": (0.01, 6.39e-4, 0.02, 3.06e-3),
    "acrobot-swingup": (0.002, 2.57e-4, 0.005, 1.47e-3),
    "cartpole-balance": (0.005, 3.71e-4, 0.01, 1.47e-3),
    "cartpole-swingup": (0.005, 4.99e-4, 0.01, 1.72e-3),
    "cartpole-two_poles": (0.005, 4.23e-4, 0.01, 1.30e-3),
    "cartpole-three_poles": (0.005, 3.15e-4, 0.01, 1.06e-3),
    # The wrist's closed loop at the box's damping maximum leaves RK4's stability region
    # above 1.1 ms (stiffness max |lambda| 2496 /s).
    "reacher-easy": (0.001, 6.1e-16, 0.002, 7.52e-1),
    # One substep per 20 ms decision: the whole interval.
    "point_mass-easy": (0.02, 9.54e-7, None, None),
    "lqr-lqr_2_1": (0.01, 5.27e-4, 0.015, 1.23e-3),
    "lqr-lqr_6_2": (0.01, 6.66e-4, 0.015, 2.15e-3),
    # Since 2026-11-10 the swimmers run dm_control's 2 ms Euler and their model step is
    # that physics step with the semi_implicit map, the simulator's own step: the
    # one-interval relative error of the oracle model against env.step_dt over 2000 rows
    # from whole uniform-decision episodes (benchmarks/ph_contact_substep.py swimmer,
    # Vista job 1135351; MuJoCo's default solver tolerance) has p99 4.2e-14 (swimmer6) and
    # 8.7e-13 (swimmer15), max 9.1e-14 and 2.7e-6; no coarser step is tried. Before, at
    # 1 ms RK4 physics (2026-11-09): p99 1.88e-4 and 3.34e-4 at 2 ms, 1.07e-3 and 3.02e-2
    # at 5 ms, RK4 stability bounds 9.8 and 3.8 ms.
    "swimmer-swimmer6": (0.002, 4.20e-14, None, None),
    "swimmer-swimmer15": (0.002, 8.65e-13, None, None),
}
# Bounds on intervals that touch a joint limit (2026-11-10, the soft limits kept, MuJoCo's
# constraint force in the oracle): the same error and steps on the "contact" pool of the
# limits measurement (scratch scripts limits/admissible_step_limits.py, out limits/step/
# <env_id>.json and .log; 2000 rows of the state pool with one limited coordinate moved
# to its limit, half past the range, half inside within one interval's travel and moving
# toward it, 1450 to 1958 of them engaging it), uniform decisions. Only the
# 1 ms physics step stays below 1e-3 there: the limit's damping switches on with the row
# (impedance 0.9 at zero penetration), a jump that a longer RK4 step straddles. Whole-
# episode starts (limits/step_full/) agree; the point mass's 20 ms, which passed on 3 s
# rollout starts (p99 3.5e-5, 0.2% of rows at a slide), failed there (2% at a slide, p99
# 3.0e-3 at 20 ms, 2.6e-4 at 10 ms). In short training runs 30% (point mass) to 48%
# (cart-pole) of intervals ended pressed on a limit, and the swimmers' joints sat on a
# limit 45 to 92% of the warm-up time, so every task with joint limits integrates at the
# physics step (common.ph_config.MODEL_STEPS: the model step is coarsened only on tasks
# without joint limits).
#   env_id: (bound, p99 at the bound, next tried step, p99 there)
LIMIT_CONTACT = {
    "cartpole-balance": (0.001, 2.38e-15, 0.002, 3.55e-1),
    "cartpole-swingup": (0.001, 2.28e-15, 0.002, 6.53e-1),
    "cartpole-two_poles": (0.001, 3.96e-15, 0.002, 4.60e-1),
    "cartpole-three_poles": (0.001, 6.33e-15, 0.002, 3.79e-1),
    "reacher-easy": (0.001, 4.35e-16, 0.002, 4.70e-2),
    "point_mass-easy": (0.001, 6.14e-16, 0.002, 3.00e-1),
}
# Rows not measured themselves: the same plant, box, clock, reset and failure set as a
# measured row (checked below); only the reward differs.
SHARED = {
    "acrobot-swingup_sparse": "acrobot-swingup",
    "cartpole-balance_sparse": "cartpole-balance",
    "cartpole-swingup_sparse": "cartpole-swingup",
    "reacher-hard": "reacher-easy",
}
# Testbeds integrated at their physics step (not in MODEL_STEPS): Euler physics with the
# semi_implicit map (the windy pendulum) and the in-house Acrobot.
EXEMPT_TESTBEDS = ("windy_pendulum", "acrobot_xk")


def entries():
    return list(table.load_ph_table())


def whole_steps(test, value, physics_dt):
    ratio = value / physics_dt
    test.assertAlmostEqual(ratio, round(ratio), places=9)
    return int(round(ratio))


class TestModelStepTable(unittest.TestCase):
    def test_every_row_divides_its_durations(self):
        """On a fixed clock the model step divides the interval, so every substep is
        exactly h. On an irregular clock (the rows of mode irregular, since 2026-11-11)
        it need not: an interval of n physics steps takes ceil(n delta / h) equal substeps
        of at most h (exactly h where h divides it), which sum to the interval. Under
        contact exclusion h is the coarse step (exclusion_model_step)."""
        for entry in entries():
            with self.subTest(row=(entry.env_id, entry.mode)):
                task = entry.task
                physics_dt = task["physics_dt"]
                step = entry.config["model_step"]
                if entry.config.get("contact_handling") == "exclude":
                    step = entry.config["model_coarse_step"]
                per_step = whole_steps(self, step, physics_dt)
                self.assertGreaterEqual(per_step, 1)
                low = whole_steps(self, task["min_duration"], physics_dt)
                high = whole_steps(self, task["max_duration"], physics_dt)
                lengths = list(range(low, high + 1))
                fixed = task["clock_kind"] == "fixed"
                self.assertEqual(fixed, low == high)
                for length in lengths:
                    if fixed:
                        self.assertEqual(length % per_step, 0, f"{length} physics steps")
                durations = torch.tensor(
                    [[length * physics_dt] for length in lengths], dtype=torch.float64
                )
                counts, sizes = integration_grid(
                    durations, step, physics_dt=physics_dt, controller_refresh_steps=None
                )
                expected = [-(-length // per_step) for length in lengths]
                self.assertEqual(counts.tolist(), expected)
                for row, (length, count) in enumerate(zip(lengths, counts.tolist())):
                    size = step if length % per_step == 0 else length * physics_dt / count
                    self.assertLessEqual(size, step * (1.0 + 1e-12))
                    torch.testing.assert_close(
                        sizes[row, :count],
                        torch.full((count,), size, dtype=torch.float64),
                        rtol=1e-12,
                        atol=0.0,
                    )
                    self.assertAlmostEqual(float(sizes[row].sum()), length * physics_dt, places=12)

    def test_measured_bounds_are_consistent(self):
        for table_name, bounds in (("MEASURED", MEASURED), ("LIMIT_CONTACT", LIMIT_CONTACT)):
            for env_id, (bound, p99, following, p99_following) in bounds.items():
                with self.subTest(table=table_name, env_id=env_id):
                    self.assertLess(p99, 1e-3)
                    if following is None:
                        continue
                    self.assertGreater(following, bound)
                    self.assertGreaterEqual(p99_following, 1e-3)
        self.assertLessEqual(set(LIMIT_CONTACT), set(MEASURED))

    def test_model_step_is_coarsened_only_without_joint_limits(self):
        """A row whose testbed keeps joint limits (limited_position_ranges; network
        limit_port, or contact_handling exclude since 2026-11-11) has model_step =
        physics_dt, as MODEL_STEPS declares its testbed."""
        limited = set()
        for entry in entries():
            with self.subTest(row=(entry.env_id, entry.mode)):
                testbed = _build_testbed(entry.config["testbed"], entry.config["testbed_settings"])
                has_limits = bool(testbed["state_spec"].get("limited_position_ranges"))
                port = bool(entry.config["network"].get("limit_port"))
                excluded = entry.config.get("contact_handling") == "exclude"
                self.assertFalse(port and excluded)
                self.assertEqual(port or excluded, has_limits)
                if not has_limits:
                    continue
                limited.add(testbed["name"])
                self.assertEqual(entry.config["model_step"], testbed["physics_dt"])
                self.assertEqual(MODEL_STEPS[testbed["name"]], testbed["physics_dt"])
        self.assertEqual(
            limited,
            {f"cartpole_{task}" for task in ("balance", "balance_sparse", "swingup",
                                               "swingup_sparse", "two_poles", "three_poles")}
            | {"reacher_easy", "reacher_hard", "point_mass_easy", "swimmer_swimmer6",
               "swimmer_swimmer15"},
        )

    def test_every_row_is_within_its_measured_bound(self):
        covered = set()
        for entry in entries():
            with self.subTest(row=(entry.env_id, entry.mode)):
                testbed = entry.config["testbed"]
                step = entry.config["model_step"]
                if testbed in EXEMPT_TESTBEDS:
                    self.assertNotIn(testbed, MODEL_STEPS)
                    self.assertEqual(step, entry.task["physics_dt"])
                    continue
                measured = SHARED.get(entry.env_id, entry.env_id)
                self.assertIn(measured, MEASURED, "a new row needs its measured bound")
                bound = MEASURED[measured][0]
                if measured in LIMIT_CONTACT:
                    bound = min(bound, LIMIT_CONTACT[measured][0])
                self.assertLessEqual(step, bound * (1.0 + 1e-12))
                # The bound divides the fixed interval it was measured on (the reference
                # period); a row of mode irregular keeps the top row's step, whose substeps
                # stay at most h on every interval (test_every_row_divides_its_durations).
                interval = entry.task["reference_period"]
                if entry.task["clock_kind"] == "fixed":
                    self.assertEqual(entry.task["min_duration"], interval)
                    self.assertEqual(entry.task["max_duration"], interval)
                whole_steps(self, interval / bound, 1.0)
                self.assertEqual(step, MODEL_STEPS[testbed])
                covered.add(testbed)
        self.assertEqual(covered, set(MODEL_STEPS))
        measured = {
            SHARED.get(entry.env_id, entry.env_id)
            for entry in entries()
            if entry.config["testbed"] in MODEL_STEPS
        }
        self.assertEqual(measured, set(MEASURED))

    def test_shared_rows_have_the_measured_dynamics(self):
        rows = {entry.env_id: entry for entry in entries() if entry.mode == "top"}
        for env_id, measured in SHARED.items():
            with self.subTest(env_id=env_id):
                self.assertEqual(rows[env_id].task, rows[measured].task)
                settings = rows[env_id].config["testbed_settings"]
                reference = rows[measured].config["testbed_settings"]
                for key in ("plant_parameters", "reset_parameters", "failure_limits"):
                    self.assertEqual(settings.get(key), reference.get(key), key)
                self.assertEqual(
                    rows[env_id].config["model_step"], rows[measured].config["model_step"]
                )

    def test_rows_take_their_factory_testbeds_step(self):
        """The row's step is default_model_step of the testbed its factory builds."""
        for entry in entries():
            with self.subTest(row=(entry.env_id, entry.mode)):
                testbed = _build_testbed(
                    entry.config["testbed"], entry.config["testbed_settings"]
                )
                self.assertEqual(testbed["name"], entry.config["testbed"])
                self.assertEqual(entry.config["model_step"], default_model_step(testbed))
                self.assertEqual(
                    default_model_step(testbed),
                    testbed_model_step(testbed["name"], testbed["physics_dt"]),
                )

    def test_json_configs_state_their_testbeds_step(self):
        for path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(config=path.name):
                protocol = load_protocol(str(path))
                testbed = protocol["matched"]["testbed_config"]
                for name in ("explicit_config", "implicit_config"):
                    self.assertEqual(protocol[name]["model_step"], default_model_step(testbed))


# The contact scheme's coarse steps (seconds), by testbed (common.ph_config.
# MODEL_COARSE_STEPS, 2026-11-10): each task's free-state bound in MEASURED (the
# cart-poles' 5 ms; the point mass 10 ms, since its 20 ms failed on whole-episode pools,
# p99 3.0e-3 against 2.6e-4 at 10 ms). Off on the reachers (their 1 ms is the RK4
# stability bound of the wrist's closed loop, not the limit) and the swimmers (the native
# 2 ms Euler step is the model step).
COARSE_STEPS = {
    "cartpole_balance": 0.005,
    "cartpole_balance_sparse": 0.005,
    "cartpole_swingup": 0.005,
    "cartpole_swingup_sparse": 0.005,
    "cartpole_two_poles": 0.005,
    "cartpole_three_poles": 0.005,
    "point_mass_easy": 0.01,
}
# The point mass's whole-episode bound (limits/step_full/, 2026-11-10): MEASURED's 20 ms
# came from 3 s rollout starts and failed on whole 20 s episodes.
WHOLE_EPISODE_BOUND = {"point_mass-easy": 0.01}


class TestContactSubsteppingTable(unittest.TestCase):
    def test_coarse_steps_are_pinned(self):
        self.assertEqual(dict(MODEL_COARSE_STEPS), COARSE_STEPS)
        for name in ("reacher_easy", "reacher_hard", "swimmer_swimmer6", "swimmer_swimmer15"):
            self.assertIsNone(testbed_model_coarse_step(name), name)
        self.assertIsNone(testbed_model_coarse_step("windy_pendulum"))

    def test_every_limited_row_uses_the_fine_step_or_the_contact_scheme(self):
        """A row with joint limits integrates an interval that touches a limit at its
        physics step: its model_step is the physics step, and either every interval
        integrates there (no model_coarse_step) or its model_coarse_step (the
        MODEL_COARSE_STEPS entry) is a whole multiple of model_step within the free-state
        bound. With the limit port (contact-triggered fine substepping) it also divides
        the fixed decision interval and the episode. Under contact exclusion (the
        cart-poles and point mass since 2026-11-11) it is the step of every model
        integration, on any clock (the rows of mode irregular), and there is no fine pass."""
        seen = set()
        for entry in entries():
            with self.subTest(row=(entry.env_id, entry.mode)):
                testbed = _build_testbed(entry.config["testbed"], entry.config["testbed_settings"])
                limited = bool(testbed["state_spec"].get("limited_position_ranges"))
                coarse = entry.config.get("model_coarse_step")
                self.assertEqual(coarse, testbed_model_coarse_step(testbed["name"]))
                self.assertEqual(coarse, default_model_coarse_step(testbed))
                if not limited:
                    self.assertIsNone(coarse)
                    continue
                step = entry.config["model_step"]
                self.assertEqual(step, testbed["physics_dt"])
                if coarse is None:
                    continue
                seen.add(testbed["name"])
                self.assertGreaterEqual(whole_steps(self, coarse, step), 2)
                measured = SHARED.get(entry.env_id, entry.env_id)
                bound = min(MEASURED[measured][0], WHOLE_EPISODE_BOUND.get(measured, 1.0))
                self.assertLessEqual(coarse, bound * (1.0 + 1e-12))
                protocol = protocol_from_config(entry.config, entry.env_id)
                excluded = entry.config.get("contact_handling") == "exclude"
                if not excluded:
                    interval = entry.task["min_duration"]
                    self.assertEqual(entry.task["max_duration"], interval)
                    whole_steps(self, interval, coarse)
                    whole_steps(self, testbed["episode_duration"], coarse)
                for name in ("explicit_config", "implicit_config"):
                    self.assertEqual(protocol[name]["model_coarse_step"], coarse)
                    if excluded:
                        self.assertIsNone(contact_substepping_setting(protocol[name]))
                        self.assertEqual(exclusion_model_step(protocol[name]), coarse)
                        continue
                    self.assertEqual(
                        contact_substepping_setting(protocol[name]),
                        {"coarse_step": coarse, "fine_step": step,
                         "margin_fraction": CONTACT_MARGIN_FRACTION,
                         "velocity_factor": CONTACT_VELOCITY_FACTOR},
                    )
        self.assertEqual(seen, set(COARSE_STEPS))

    def test_json_configs_state_their_testbeds_coarse_step(self):
        for path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(config=path.name):
                protocol = load_protocol(str(path))
                testbed = protocol["matched"]["testbed_config"]
                for name in ("explicit_config", "implicit_config"):
                    self.assertEqual(
                        protocol[name].get("model_coarse_step"), default_model_coarse_step(testbed)
                    )


class TestModelStepReachesTheAlgorithm(unittest.TestCase):
    def test_lqr_6_2_fits_its_plant_in_three_substeps(self):
        raw = copy.deepcopy(table.load_ph_hyperparams_from_table("lqr-lqr_6_2", "top"))
        self.assertEqual(raw["model_step"], 0.01)
        raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
        raw["training"].update(replay_capacity=2000, batch_size=8, warmup_episodes=1)
        raw["protocol"].update(training_seeds=[0], evaluation_reset_seeds=[20000])
        protocol = protocol_from_config(raw, "lqr-lqr_6_2")
        for name in ("explicit_config", "implicit_config"):
            self.assertEqual(protocol[name]["model_step"], 0.01)
        protocol["arms"] = [
            arm for arm in P.all_arms() if arm["name"] == "implicit_learned_admissions"
        ]
        (run,) = P.build_run_matrix(protocol)
        env = make_env(protocol["matched"]["testbed_config"], seed=0)
        agent = P.make_training_agent(run, env)
        self.assertEqual(agent.model_step, 0.01)
        self.assertIsNone(agent.refresh_steps)  # continuous: no refresh points
        counts, sizes = integration_grid(
            torch.tensor([[0.03]], dtype=torch.float64),
            agent.model_step,
            physics_dt=agent.physics_dt,
            controller_refresh_steps=agent.refresh_steps,
        )
        self.assertEqual(counts.tolist(), [3])
        torch.testing.assert_close(
            sizes, torch.full((1, 3), 0.01, dtype=torch.float64), rtol=1e-12, atol=0.0
        )
        # The plant fit on collected rows: 30 ms decisions (an episode's last, cut at the
        # 10 s horizon, is 10 ms: one substep).
        agent.warmup(1)
        batch = agent.replay.sample("train", 16)
        expected = [round(float(d) / 0.01) for d in batch["durations"].reshape(-1)]
        self.assertIn(3, expected)
        captured = []
        original = implicit_module.integrate_endpoint_moments

        def capture(*args, **kwargs):
            moments = original(*args, **kwargs)
            captured.append((kwargs, moments))
            return moments

        with mock.patch.object(implicit_module, "integrate_endpoint_moments", capture):
            agent.update_online_plant(batch)
        ((kwargs, moments),) = captured
        self.assertEqual(kwargs["maximum_step"], 0.01)
        self.assertEqual(moments["step_counts"].tolist(), expected)
        for row, count in enumerate(expected):
            steps = moments["step_sizes"][row].double()
            torch.testing.assert_close(
                steps[:count], torch.full((count,), 0.01, dtype=torch.float64),
                rtol=1e-6, atol=0.0,
            )
            self.assertFalse(steps[count:].any())
        env.close()


if __name__ == "__main__":
    unittest.main()
