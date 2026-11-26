import contextlib
import copy
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import (
    _EXPLICIT,
    _IMPLICIT,
    _OPTIONAL_TESTBED_SETTINGS,
    _OPTIONAL_PROTOCOL,
    _OPTIONAL_TOP_LEVEL,
    _PROTOCOL,
    _TESTBED_FACTORIES,
    _TESTBED_SETTINGS,
    list_arms,
    load_protocol,
    load_protocol_from_table,
    load_protocols,
    main,
    parse_args,
    protocol_from_config,
)
from common.ph_config import (
    MODEL_COARSE_STEPS,
    AdmissionConfig,
    NetworkConfig,
    RateConfig,
    SolverConfig,
    TemperatureConfig,
    TrainingConfig,
    decision_box_entropy_target,
    testbed_model_coarse_step,
    testbed_model_step,
)
from environment.ph_sac.registry import testbed_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "benchmarks" / "configs" / "ph_sac"

# (env_id, mode) -> the JSON config the row must reproduce.
ROUND_TRIPS = {
    ("cartpole-swingup", "top"): "cartpole_swingup.json",
    ("acrobot_xk-swingup", "top"): "acrobot_xk.json",
    ("windy_pendulum-balance", "top"): "windy_pendulum.json",
    ("windy_pendulum-swingup_tq5", "top"): "windy_swingup_tq5.json",
    ("windy_pendulum-swingup_tq12", "top"): "windy_swingup_tq12.json",
    ("windy_pendulum-balance", "smoke"): "smoke_windy.json",
}
DMC_TASKS = (
    "pendulum-swingup",
    "acrobot-swingup",
    "acrobot-swingup_sparse",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cartpole-two_poles",
    "cartpole-three_poles",
    "reacher-easy",
    "reacher-hard",
    "point_mass-easy",
    "lqr-lqr_2_1",
    "lqr-lqr_6_2",
    "swimmer-swimmer6",
    "swimmer-swimmer15",
)
IN_HOUSE_TASKS = (
    "windy_pendulum-balance",
    "windy_pendulum-swingup_tq5",
    "windy_pendulum-swingup_tq12",
    "acrobot_xk-swingup",
)
# The optional reward variants: (env_id, mode) rows beside the top rows, none used in the
# final run (2026-11-11); the top rows keep their tasks' rewards.
VARIANT_ROWS = (
    ("cartpole-two_poles", "lyapunov_cart"),
    ("cartpole-two_poles", "lyapunov_cart_straight"),
    ("cartpole-three_poles", "lyapunov_cart"),
    ("cartpole-three_poles", "lyapunov_cart_straight"),
    ("acrobot-swingup", "lyapunov_shoulder"),
)
# The rows under contact exclusion since 2026-11-11 (make_ct_ph_sac_table.CONTACT_HANDLING):
# no limit port; the other rows with joint limits (reachers, swimmers) keep the port.
EXCLUDED = (
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cartpole-two_poles",
    "cartpole-three_poles",
    "point_mass-easy",
)
PORTED = ("reacher-easy", "reacher-hard", "swimmer-swimmer6", "swimmer-swimmer15")
# The rows of mode "irregular" (the final run's clock, since 2026-11-11): the top row on the
# CT-SAC benchmark's irregular law (tail_p 0.99, tail_split 0.9 on [T/5, 3T]; since 11-12).
IRREGULAR_ROWS = (
    "pendulum-swingup", "acrobot-swingup", "acrobot-swingup_sparse", "cartpole-balance",
    "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse",
    "cartpole-two_poles", "cartpole-three_poles", "point_mass-easy",
)
IRREGULAR_SETTINGS = {"clock_kind": "two_tail_uniform", "tail_probability": 0.99, "tail_split": 0.9}
# Exploratory variant rows on the irregular clock (not reported): (env_id, mode) -> the
# variant row they copy.
EXPLORATORY_ROWS = {("cartpole-two_poles", "lyapunov_cart_irregular"): ("cartpole-two_poles", "lyapunov_cart")}
# The rows left out of the final run's launch plan (2026-11-11), complete and runnable.
SET_ASIDE = ("reacher-easy", "reacher-hard", "swimmer-swimmer6", "swimmer-swimmer15")
WRIST = math.radians(160.0)


def raw_config(name):
    with open(CONFIGS / name, encoding="utf-8") as handle:
        raw = json.load(handle)
    raw.pop("_experiment_choices", None)
    return raw


def entries():
    return {(entry.env_id, entry.mode): entry for entry in table.load_ph_table()}


def implemented(entry):
    """Whether the row's testbed is registered as implemented (not a stub)."""
    return testbed_spec(entry.config["testbed"]).implemented


def unported():
    """The dm_control rows whose testbed is still a registered stub."""
    rows = entries()
    return tuple(env_id for env_id in DMC_TASKS if not implemented(rows[(env_id, "top")]))


class StrictEqualMixin:
    def assertStrictEqual(self, left, right, path="config"):
        """Equal values of equal types: arrays by value and dtype, dicts by keys."""
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            self.assertIsInstance(left, np.ndarray, path)
            self.assertIsInstance(right, np.ndarray, path)
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertTrue(np.array_equal(left, right), path)
            return
        self.assertIs(type(left), type(right), path)
        if isinstance(left, dict):
            self.assertEqual(sorted(left), sorted(right), path)
            for key in left:
                self.assertStrictEqual(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assertStrictEqual(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)


class SchemaTests(unittest.TestCase):
    def keys(self, *path):
        return {
            column.path[len(path)]
            for column in table.CONFIG_COLUMNS
            if column.path[: len(path)] == path and len(column.path) == len(path) + 1
        }

    @staticmethod
    def typed(schema):
        return set(schema.__required_keys__) | set(schema.__optional_keys__)

    def test_columns_cover_every_setting_of_load_protocol(self):
        self.assertEqual(len(table.TABLE_COLUMNS), len(set(table.TABLE_COLUMNS)))
        self.assertEqual(self.keys("network"), self.typed(NetworkConfig))
        self.assertEqual(self.keys("training"), self.typed(TrainingConfig))
        self.assertEqual(self.keys("temperature"), self.typed(TemperatureConfig))
        self.assertEqual(self.keys("implicit", "solver"), self.typed(SolverConfig))
        self.assertEqual(self.keys("implicit", "rate"), self.typed(RateConfig))
        self.assertEqual(self.keys("implicit", "admission"), self.typed(AdmissionConfig))
        self.assertEqual(self.keys("explicit"), set(_EXPLICIT))
        self.assertEqual(
            self.keys("implicit") | {"solver", "rate", "admission"}, set(_IMPLICIT)
        )
        # theory_diagnostics, the one nested optional protocol section, has no columns.
        self.assertEqual(
            self.keys("protocol"),
            set(_PROTOCOL) | set(_OPTIONAL_PROTOCOL) - {"theory_diagnostics"},
        )
        settings = set()
        for names in (*_TESTBED_SETTINGS.values(), *_OPTIONAL_TESTBED_SETTINGS.values()):
            settings.update(names)
        self.assertEqual(self.keys("testbed_settings"), settings)
        # refresh_mode, which replaces the factory's controller refresh_mode, has no
        # column: the rows declare the factory's as task_refresh_mode.
        self.assertEqual(
            self.keys(), {"testbed", "model_step"} | set(_OPTIONAL_TOP_LEVEL) - {"refresh_mode"}
        )

    def test_csv_header_is_the_schema(self):
        with table.table_path().open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(tuple(header), table.TABLE_COLUMNS)


class CellTests(unittest.TestCase):
    def test_cell_syntax(self):
        parse = table.parse_cell
        self.assertEqual(parse("number", ""), (False, None))
        self.assertEqual(parse("number", "none"), (True, None))
        self.assertEqual(parse("str", "none"), (True, "none"))
        self.assertEqual(parse("number", "4"), (True, 4))
        self.assertIs(type(parse("number", "4")[1]), int)
        self.assertIs(type(parse("number", "4.0")[1]), float)
        self.assertEqual(parse("number", "1e-06"), (True, 1e-06))
        self.assertEqual(parse("bool", "True"), (True, True))
        self.assertEqual(parse("bool", "false"), (True, False))
        self.assertEqual(parse("int_list", "256,256"), (True, [256, 256]))
        self.assertEqual(parse("int_list", "[]"), (True, []))
        self.assertEqual(parse("int_ranges", "0-3,7"), (True, [0, 1, 2, 3, 7]))
        self.assertEqual(parse("float_list", "-1.0,0.0"), (True, [-1.0, 0.0]))
        self.assertEqual(parse("dict", "{}"), (True, {}))
        self.assertEqual(
            parse("dict", "a=1;b=0.5;c=x;d=none"),
            (True, {"a": 1, "b": 0.5, "c": "x", "d": None}),
        )
        for kind, text in (
            ("number", "fast"),
            ("bool", "yes"),
            ("int_list", "1.5"),
            ("int_ranges", "9-3"),
            ("dict", "a=1;b"),
            ("dict", "a=1;a=2"),
        ):
            with self.assertRaises(ValueError, msg=(kind, text)):
                parse(kind, text)

    def test_writer_inverts_parser(self):
        column = table.Column("seeds", ("seeds",), "int_ranges", compress=True)
        self.assertEqual(table.format_cell(column, [20000, 20001]), "20000,20001")
        self.assertEqual(table.format_cell(column, list(range(20000, 20032))), "20000-20031")
        plain = table.Column("seeds", ("seeds",), "int_ranges")
        self.assertEqual(table.format_cell(plain, list(range(4))), "0,1,2,3")
        for kind, value in (
            ("number", 0.1 + 0.2),
            ("number", 2 * math.pi),
            ("float_list", [-math.pi, 0.0, 1e-300]),
            ("dict", {"a": 1, "b": math.radians(160.0), "c": None, "d": "text"}),
            ("dict", {}),
            ("int_list", []),
        ):
            column = table.Column("x", ("x",), kind)
            self.assertEqual(table.parse_cell(kind, table.format_cell(column, value)), (True, value))
        with self.assertRaises(ValueError):
            table.format_cell(table.Column("x", ("x",), "str"), None)

    def test_dict_cells_read_like_common_utils(self):
        from common.utils import _parse_dict_from_string

        for entry in table.load_ph_table():
            for name in ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters"):
                value = entry.config["testbed_settings"].get(name)
                if value:
                    cell = table.format_cell(
                        table.Column(name, (name,), "dict"), value
                    )
                    self.assertEqual(_parse_dict_from_string(cell), value, (entry.env_id, name))

    def test_rows_reject_unknown_columns(self):
        with self.assertRaises(ValueError):
            table.parse_ph_row({"mode": "top", "env_id": "x", "algo_typo": "1"})
        with self.assertRaises(ValueError):
            table.config_to_row(
                {**raw_config("smoke_windy.json"), "extra": 1}, env_id="x"
            )
        with_theory = raw_config("smoke_windy.json")
        with_theory["protocol"]["theory_diagnostics"] = {"policy_kind": "stochastic"}
        with self.assertRaises(ValueError):
            table.config_to_row(with_theory, env_id="x")


class TableTests(StrictEqualMixin, unittest.TestCase):
    def test_one_top_row_per_task(self):
        rows = entries()
        tops = {env_id for env_id, mode in rows if mode == "top"}
        self.assertEqual(tops, set(DMC_TASKS) | set(IN_HOUSE_TASKS))
        self.assertEqual(len(DMC_TASKS), 16)
        # Besides the top rows: the smoke row and the optional reward variants
        # (make_ct_ph_sac_table.VARIANTS; tests/test_ph_orbit_rewards.py).
        self.assertEqual(
            set(rows),
            {(e, "top") for e in tops} | {("windy_pendulum-balance", "smoke")} | set(VARIANT_ROWS)
            | {(e, "irregular") for e in IRREGULAR_ROWS} | set(EXPLORATORY_ROWS),
        )

    def test_every_row_parses_and_is_complete(self):
        task_names = {column.path[0] for column in table.TASK_COLUMNS}
        for key, entry in entries().items():
            with self.subTest(row=key):
                # model_coarse_step (contact-triggered fine substepping, since
                # 2026-11-10) only on the rows whose testbed MODEL_COARSE_STEPS lists.
                coarse = {"model_coarse_step"} if entry.config["testbed"] in MODEL_COARSE_STEPS else set()
                # contact_handling (contact exclusion, since 2026-11-11) only on EXCLUDED.
                handling = {"contact_handling"} if key[0] in EXCLUDED else set()
                self.assertEqual(
                    set(entry.config),
                    {"testbed", "testbed_settings", "model_step", "model_integrator",
                     "network", "training", "temperature", "explicit", "implicit",
                     "protocol"} | coarse | handling,
                )
                self.assertEqual(set(entry.task), task_names)
                self.assertTrue(entry.comment)
                config = table.load_ph_hyperparams_from_table(*key)
                self.assertStrictEqual(config, entry.config)

    def test_rows_equal_the_json_configs(self):
        for key, name in ROUND_TRIPS.items():
            with self.subTest(row=key):
                rebuilt = table.load_ph_hyperparams_from_table(*key)
                self.assertStrictEqual(rebuilt, raw_config(name))
                # The writer gives back the same cells, so the table is canonical.
                row = table.config_to_row(
                    raw_config(name), env_id=key[0], mode=key[1], task=entries()[key].task
                )
                self.assertStrictEqual(table.parse_ph_row(row).config, raw_config(name))

    def test_round_trip_protocols_equal_the_json_protocols(self):
        for key, name in ROUND_TRIPS.items():
            with self.subTest(row=key):
                from_table = load_protocol_from_table(*key)
                from_json = load_protocol(str(CONFIGS / name))
                self.assertStrictEqual(from_table, from_json, "protocol")
                rebuilt = protocol_from_config(table.load_ph_hyperparams_from_table(*key), "x")
                self.assertStrictEqual(rebuilt, from_json, "protocol")

    def test_entropy_targets_follow_the_decision_box(self):
        for key, entry in entries().items():
            box = {
                "decision_low": np.array(entry.task["decision_low"]),
                "decision_high": np.array(entry.task["decision_high"]),
            }
            self.assertEqual(
                entry.config["temperature"]["target_entropy"],
                decision_box_entropy_target(box),
                key,
            )
        rows = entries()
        for env_id in (
            "windy_pendulum-balance",
            "windy_pendulum-swingup_tq5",
            "windy_pendulum-swingup_tq12",
            "acrobot_xk-swingup",
            "cartpole-swingup",
        ):
            self.assertEqual(rows[(env_id, "top")].config["temperature"]["target_entropy"], -3.0)
        # The dm_control rows' boxes are scaled to their motors, but the target is -1 per
        # coordinate of the normalized box: -3 for one actuator, -6 for two, whatever the
        # units. On the reacher, -3 plus the log half-widths would have been -14.8. The
        # swimmers have 5 and 14 actuators: -15 and -42.
        two_actuators = {"reacher-easy", "reacher-hard", "point_mass-easy", "lqr-lqr_6_2"}
        many_actuators = {"swimmer-swimmer6": -15.0, "swimmer-swimmer15": -42.0}
        for env_id in DMC_TASKS:
            task = rows[(env_id, "top")].task
            controller = {
                "decision_low": np.array(task["decision_low"], dtype=float),
                "decision_high": np.array(task["decision_high"], dtype=float),
            }
            target = rows[(env_id, "top")].config["temperature"]["target_entropy"]
            self.assertEqual(target, decision_box_entropy_target(controller), env_id)
            expected = -6.0 if env_id in two_actuators else many_actuators.get(env_id, -3.0)
            self.assertEqual(target, expected, env_id)
            self.assertEqual(target, -float(len(task["decision_low"])), env_id)

    def test_unported_rows_parse_but_do_not_build(self):
        rows = entries()
        # Every dm_control row is ported now; the loop below guards any future stub.
        self.assertEqual(unported(), ())
        self.assertNotIn("cartpole-swingup", unported())
        for env_id in unported():
            with self.subTest(env_id=env_id):
                testbed = table.load_ph_hyperparams_from_table(env_id)["testbed"]
                self.assertNotIn(testbed, _TESTBED_FACTORIES)
                self.assertNotIn(testbed, {e.config["testbed"] for k, e in rows.items() if k[0] != env_id})
                with self.assertRaises(NotImplementedError) as caught:
                    load_protocol_from_table(env_id)
                self.assertIn(repr(testbed), str(caught.exception))
                self.assertIn(env_id, str(caught.exception))
        with self.assertRaises(KeyError):
            table.load_ph_hyperparams_from_table("cartpole-swingup", "second")

    def test_declared_task_columns_are_checked_against_the_factory(self):
        for key in ROUND_TRIPS:
            testbed = load_protocol_from_table(*key)["matched"]["testbed_config"]
            self.assertEqual(table.declared_testbed_mismatches(entries()[key].task, testbed), [])
        with table.table_path().open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["env_id"] == "cartpole-swingup":
                row["task_physics_dt"] = "0.002"
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / table.TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table.TABLE_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "task_physics_dt"):
                load_protocol_from_table("cartpole-swingup", "top", tmp)


class DmcRowTests(unittest.TestCase):
    # env_id: (nq, nu, reference period = dm_control control step, episode, motor limit,
    #          failure limits), from the companion document and the paper appendix. No
    #          dm_control row has a failure set since 2026-11-10: the cart-pole slider, the
    #          reacher wrist and the point-mass slides keep dm_control's soft limits.
    EXPECTED = {
        "pendulum-swingup": (1, 1, 0.02, 20.0, 1.0, {}),
        "acrobot-swingup": (2, 1, 0.01, 10.0, 2.0, {}),
        "acrobot-swingup_sparse": (2, 1, 0.01, 10.0, 2.0, {}),
        "cartpole-balance": (2, 1, 0.01, 10.0, 10.0, {}),
        "cartpole-balance_sparse": (2, 1, 0.01, 10.0, 10.0, {}),
        "cartpole-swingup": (2, 1, 0.01, 10.0, 10.0, {}),
        "cartpole-swingup_sparse": (2, 1, 0.01, 10.0, 10.0, {}),
        "cartpole-two_poles": (3, 1, 0.01, 10.0, 10.0, {}),
        "cartpole-three_poles": (4, 1, 0.01, 10.0, 10.0, {}),
        "reacher-easy": (2, 2, 0.02, 20.0, 0.05, {}),
        "reacher-hard": (2, 2, 0.02, 20.0, 0.05, {}),
        "point_mass-easy": (2, 2, 0.02, 20.0, 0.1, {}),
        "lqr-lqr_2_1": (2, 1, 0.03, 10.0, 20.0, {}),
        "lqr-lqr_6_2": (6, 2, 0.03, 10.0, 20.0, {}),
        # The swimmers keep dm_control's soft joint limits: no failure set.
        "swimmer-swimmer6": (8, 5, 0.03, 30.0, 5e-4, {}),
        "swimmer-swimmer15": (17, 14, 0.03, 30.0, 5e-4, {}),
    }

    def test_task_columns_match_the_sources(self):
        rows = entries()
        self.assertEqual(set(self.EXPECTED), set(DMC_TASKS))
        for env_id, (nq, nu, period, episode, limit, failure) in self.EXPECTED.items():
            with self.subTest(env_id=env_id):
                entry = rows[(env_id, "top")]
                task = entry.task
                self.assertEqual((task["nq"], task["nv"], task["nu"]), (nq, nq, nu))
                with_target = env_id.startswith(("reacher", "swimmer"))
                self.assertEqual(task["context_dim"], 2 if with_target else 0)
                self.assertEqual(len(task["actuated_position_indices"]), nu)
                # The swimmers run dm_control's own physics, 2 ms Euler (since 2026-11-10).
                physics = (0.002, "Euler") if env_id.startswith("swimmer") else (0.001, "RK4")
                self.assertEqual((task["physics_dt"], task["integrator"]), physics)
                self.assertEqual(task["clock_kind"], "fixed")
                self.assertEqual(
                    (task["reference_period"], task["min_duration"], task["max_duration"]),
                    (period, period, period),
                )
                self.assertEqual(task["episode_duration"], episode)
                self.assertEqual(task["discount_rate"], 0.5)
                self.assertEqual(task["torque_low"], [-limit] * nu)
                self.assertEqual(task["torque_high"], [limit] * nu)
                # The box is scaled to the motor: stiffness up to 10 tau / offset and damping
                # up to 0.2 s times that, the rule that gives cart-pole's 100 and 20 at +-10.
                self.assertEqual(task["decision_low"][nu:], [0.0] * (2 * nu))
                offsets = task["decision_high"][:nu]
                self.assertEqual(task["decision_low"][:nu], [-o for o in offsets])
                for i, offset in enumerate(offsets):
                    k_max = 10.0 * limit / offset
                    self.assertAlmostEqual(task["decision_high"][nu + i], k_max)
                    self.assertAlmostEqual(task["decision_high"][2 * nu + i], 0.2 * k_max)
                self.assertEqual(task["refresh_steps"], 1)
                self.assertEqual(entry.config["testbed_settings"]["failure_limits"], failure)

    def test_method_columns_copy_cartpole_swingup(self):
        rows = entries()
        reference = rows[("cartpole-swingup", "top")].config
        for env_id in DMC_TASKS:
            with self.subTest(env_id=env_id):
                entry = rows[(env_id, "top")]
                config = copy.deepcopy(entry.config)
                self.assertEqual(config["network"].pop("damping_rank"), entry.task["nv"])
                # Only the swimmer rows set the velocity-dependent damping (fluid drag); of
                # the rows with kept dm_control joint limits the reachers and swimmers set
                # the limit port, the cart-poles and the point mass contact exclusion
                # (since 2026-11-11) and no port.
                for key, expected_value in (
                    ("damping_velocity", True if env_id.startswith("swimmer") else None),
                    ("limit_port", True if env_id in PORTED else None),
                ):
                    value = config["network"].pop(key, None)
                    self.assertEqual(value, expected_value, key)
                self.assertEqual(
                    config.pop("contact_handling", None),
                    "exclude" if env_id in EXCLUDED else None,
                )
                self.assertEqual(config["network"]["diffusion"], "none")
                config["temperature"].pop("target_entropy")
                config["protocol"].pop("output_directory")
                # The model step is per task (common.ph_config.MODEL_STEPS); see
                # tests/test_ph_model_step.py for its bounds.
                self.assertEqual(
                    config.pop("model_step"),
                    testbed_model_step(config["testbed"], entry.task["physics_dt"]),
                )
                # The contact scheme's coarse step is per task too
                # (common.ph_config.MODEL_COARSE_STEPS; tests/test_ph_model_step.py).
                self.assertEqual(
                    config.pop("model_coarse_step", None),
                    testbed_model_coarse_step(config["testbed"]),
                )
                # The swimmers integrate their model with their Euler physics' own map.
                if env_id.startswith("swimmer"):
                    self.assertEqual(config.pop("model_integrator"), "semi_implicit")
                    config["model_integrator"] = reference["model_integrator"]
                for section in ("model_integrator", "network", "training",
                                "temperature", "explicit", "implicit", "protocol"):
                    expected = copy.deepcopy(reference[section])
                    if isinstance(expected, dict):
                        expected.pop("damping_rank", None)
                        expected.pop("target_entropy", None)
                        expected.pop("output_directory", None)
                        expected.pop("limit_port", None)
                    self.assertEqual(config[section], expected, section)
                protocol = config["protocol"]
                self.assertEqual(protocol["interaction_budget"], 500000)
                self.assertEqual(protocol["training_seeds"], list(range(10)))
                self.assertEqual(protocol["evaluation_reset_seeds"], list(range(20000, 20032)))
        directories = [rows[(e, "top")].config["protocol"]["output_directory"] for e in DMC_TASKS]
        self.assertEqual(len(set(directories)), len(directories))

    def test_open_items_are_flagged(self):
        rows = entries()
        for env_id in ("reacher-easy", "reacher-hard"):
            comment = rows[(env_id, "top")].comment
            self.assertIn("context", comment)
            self.assertIn("wrist", comment)
        for env_id in ("lqr-lqr_2_1", "lqr-lqr_6_2"):
            comment = rows[(env_id, "top")].comment
            self.assertIn("OPEN: torque limit +-20", comment)
            self.assertIn("capped at 10 s", comment)
        for env_id in unported():
            self.assertIn("Not ported", rows[(env_id, "top")].comment)
            self.assertIn("Budget 500k decisions", rows[(env_id, "top")].comment)


class FinalRunDecisionTests(unittest.TestCase):
    """The settings decided for the final run on 2026-11-11 (make_ct_ph_sac_table)."""

    def test_contact_exclusion_on_the_cart_poles_and_point_mass(self):
        from common.ph_config import contact_exclusion_setting, contact_substepping_setting

        rows = entries()
        for (env_id, mode), entry in rows.items():
            if env_id not in EXCLUDED + PORTED:
                continue
            with self.subTest(row=(env_id, mode)):
                excluded = env_id in EXCLUDED
                self.assertEqual(entry.config.get("contact_handling"), "exclude" if excluded else None)
                self.assertIs(entry.config["network"].get("limit_port", False), not excluded)
                protocol = load_protocol_from_table(env_id, mode)
                for name in ("explicit_config", "implicit_config"):
                    config = protocol[name]
                    setting = contact_exclusion_setting(config)
                    if excluded:
                        # Every model integration at the coarse step, no fine pass.
                        self.assertEqual(setting["mode"], "exclude")
                        self.assertEqual(setting["model_step"], entry.config["model_coarse_step"])
                        self.assertIsNone(contact_substepping_setting(config))
                        self.assertIn("Contact exclusion (model_contact_handling exclude", entry.comment)
                        self.assertNotIn("Limit port initialised", entry.comment)
                    else:
                        self.assertIsNone(setting)
                        self.assertIn("Limit port initialised", entry.comment)
        # --contact-handling port turns an excluded row back into the port convention.
        (ported,) = load_protocols(parse_args(["--env-id", "point_mass-easy", "--contact-handling", "port"]))
        for name in ("explicit_config", "implicit_config"):
            self.assertNotIn("contact_handling", ported[name])
            self.assertIs(ported[name]["network"]["limit_port"], True)
            self.assertEqual(contact_substepping_setting(ported[name])["coarse_step"], 0.01)

    def test_plant_step_schedule_is_confirmed(self):
        for (env_id, mode), entry in entries().items():
            if env_id.split("-")[0] in ("windy_pendulum", "acrobot_xk"):
                continue
            with self.subTest(row=(env_id, mode)):
                training = entry.config["training"]
                self.assertEqual(
                    (training["plant_step_dense_until"], training["plant_step_every"]), (100000, 4)
                )
                self.assertIn("Plant-step schedule (confirmed 2026-11-11)", entry.comment)
                self.assertNotIn("proposed 2026-11-10)", entry.comment)

    def test_multi_pole_rows_keep_dm_controls_reward(self):
        """The two- and three-pole rows, top and irregular, keep dm_control's smooth reward
        (decided on 2026-11-11, after variant A came no closer to balancing);
        the energy-and-cart Lyapunov reward stays the unused variant row lyapunov_cart."""
        from environment.ph_sac.dmc import cartpole as dmc_cartpole

        rows = entries()
        smooth = {"cart_margin": 2.0, "velocity_margin": 5.0, "failure_rate": 0.0}
        for env_id, poles in (("cartpole-two_poles", 2), ("cartpole-three_poles", 3)):
            for mode in ("top", "irregular"):
                with self.subTest(env_id=env_id, mode=mode):
                    entry = rows[(env_id, mode)]
                    self.assertEqual(entry.config["testbed_settings"]["reward_parameters"], smooth)
                    built = load_protocol_from_table(env_id, mode)["matched"]["testbed_config"]
                    self.assertEqual(dmc_cartpole.cartpole_reward_law(built), "smooth")
                    self.assertIn("Reward: the smooth cart-pole reward", entry.comment)
                    self.assertNotIn("Reward NOT dm_control's", entry.comment)
                    self.assertNotIn("UNTESTED", entry.comment)
            with self.subTest(env_id=env_id, mode="lyapunov_cart"):
                variant = rows[(env_id, "lyapunov_cart")]
                reward = variant.config["testbed_settings"]["reward_parameters"]
                self.assertEqual(tuple(reward), dmc_cartpole.LYAPUNOV_CART_KEYS)
                terms = dmc_cartpole.cartpole_lyapunov_reward_terms(
                    variant.config["testbed_settings"]["plant_parameters"], poles, straight=False
                )
                for key, value in terms.items():
                    self.assertAlmostEqual(reward[key] / value, 1.0, places=12, msg=key)
                self.assertEqual(
                    variant.config["protocol"]["output_directory"],
                    "out/ph_sac/" + env_id.replace("-", "_") + "_lyapunov_cart",
                )
        # acrobot-swingup keeps its Lyapunov reward; the unused variants say so.
        acrobot = rows[("acrobot-swingup", "top")].config["testbed_settings"]["reward_parameters"]
        self.assertNotIn("lyapunov_k_shoulder", acrobot)
        for key in VARIANT_ROWS:
            self.assertIn("Not used in the final run (decided 2026-11-11)", rows[key].comment)

    def test_irregular_rows_copy_their_top_rows_but_the_clock(self):
        from common.ph_time import clock_duration_pmf

        rows = entries()
        for env_id in IRREGULAR_ROWS:
            with self.subTest(env_id=env_id):
                irregular, top = rows[(env_id, "irregular")], rows[(env_id, "top")]
                left, right = copy.deepcopy(irregular.config), copy.deepcopy(top.config)
                settings = left["testbed_settings"]
                self.assertEqual({k: settings.pop(k) for k in IRREGULAR_SETTINGS}, IRREGULAR_SETTINGS)
                self.assertEqual(
                    left["protocol"].pop("output_directory"),
                    right["protocol"].pop("output_directory") + "_irregular",
                )
                self.assertEqual(left, right)
                period = top.task["reference_period"]
                self.assertEqual(irregular.task["reference_period"], period)
                self.assertEqual(top.task["clock_kind"], "fixed")
                self.assertEqual(irregular.task["clock_kind"], "two_tail_uniform")
                # CT-SAC's cheetah-run and walker-run bounds: T/5 and 3T.
                self.assertAlmostEqual(irregular.task["min_duration"], 0.2 * period, places=12)
                self.assertAlmostEqual(irregular.task["max_duration"], 3.0 * period, places=12)
                rest = {k: v for k, v in irregular.task.items()
                        if k not in ("clock_kind", "min_duration", "max_duration")}
                self.assertEqual(rest, {k: top.task[k] for k in rest})
                self.assertIn("Row of mode irregular", irregular.comment)
                protocol = load_protocol_from_table(env_id, "irregular")
                clock = protocol["matched"]["testbed_config"]["clock"]
                for name in ("explicit_config", "implicit_config"):
                    self.assertEqual(protocol[name]["testbed"]["clock"], clock)
                self.assertEqual(clock["kind"], "two_tail_uniform")
                self.assertEqual((clock["tail_probability"], clock["tail_split"]), (0.99, 0.9))
                steps, probabilities = clock_duration_pmf(clock)
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
                mean = float((steps * probabilities).sum()) * clock["physics_dt"]
                # 0.891 at T/5, 0.099 at 3T, 0.01 uniform between: mean 0.4912 T.
                self.assertAlmostEqual(mean, 0.4912 * period, places=12)
                self.assertAlmostEqual(float(probabilities[0]), 0.891, places=12)
                self.assertAlmostEqual(float(probabilities[-1]), 0.099, places=12)
                # The excluded rows keep their coarse step, which needs no fixed clock there.
                if env_id in EXCLUDED:
                    self.assertEqual(protocol["implicit_config"]["contact_handling"], "exclude")
                    self.assertIn("model_coarse_step", irregular.config)
        # The exploratory variant rows: the variant row on the same clock, marked so.
        for key, source in EXPLORATORY_ROWS.items():
            with self.subTest(row=key):
                row, variant = rows[key], rows[source]
                left, right = copy.deepcopy(row.config), copy.deepcopy(variant.config)
                settings = left["testbed_settings"]
                self.assertEqual({k: settings.pop(k) for k in IRREGULAR_SETTINGS}, IRREGULAR_SETTINGS)
                self.assertEqual(
                    left["protocol"].pop("output_directory"),
                    right["protocol"].pop("output_directory") + "_irregular",
                )
                self.assertEqual(left, right)
                self.assertEqual(row.task, rows[(key[0], "irregular")].task)
                self.assertTrue(row.comment.startswith("EXPLORATORY row"))
                self.assertIn(f"--mode {key[1]}", row.comment)
                self.assertEqual(load_protocol_from_table(*key)["implicit_config"]["contact_handling"],
                                 "exclude")
        # With the limit port the coarse step (contact-triggered fine substepping) still
        # needs a fixed clock.
        with self.assertRaisesRegex(ValueError, "fixed clock"):
            load_protocols(parse_args(["--env-id", "cartpole-swingup", "--mode", "irregular",
                                       "--contact-handling", "port"]))

    def test_set_aside_rows_stay_complete_and_say_so(self):
        rows = entries()
        for env_id in SET_ASIDE:
            with self.subTest(env_id=env_id):
                entry = rows[(env_id, "top")]
                self.assertIn("Status 2026-11-11", entry.comment)
                self.assertIn("runnable", entry.comment)
                load_protocol_from_table(env_id)
        for env_id in set(DMC_TASKS) - set(SET_ASIDE):
            self.assertNotIn("Status 2026-11-11", rows[(env_id, "top")].comment, env_id)


class RunnerTests(StrictEqualMixin, unittest.TestCase):
    def test_env_id_builds_the_same_protocol_as_config(self):
        from_table = load_protocols(parse_args(["--env-id", "cartpole-swingup", "--mode", "top"]))
        from_json = load_protocols(
            parse_args(["--config", str(CONFIGS / "cartpole_swingup.json")])
        )
        self.assertEqual(len(from_table), 1)
        self.assertStrictEqual(from_table, from_json, "protocols")

    def test_env_id_takes_the_overrides(self):
        args = parse_args(
            ["--env-id", "windy_pendulum-balance", "--mode", "smoke", "--seeds", "3",
             "--interactions", "99", "--config", str(CONFIGS / "smoke_windy.json")]
        )
        protocols = load_protocols(args)
        self.assertEqual(len(protocols), 2)
        for protocol in protocols:
            self.assertEqual(protocol["training_seeds"], [3])
            self.assertEqual(protocol["matched"]["interaction_budget"], 99)

    def test_checkpoint_replay_reaches_the_protocol(self):
        """Every row keeps the replay; a False cell drops it and an empty one defaults."""
        for key, entry in entries().items():
            with self.subTest(row=key):
                self.assertIs(entry.config["protocol"]["checkpoint_replay"], True)
        key = ("windy_pendulum-balance", "smoke")
        raw = raw_config("smoke_windy.json")
        self.assertIs(protocol_from_config(raw, "x")["checkpoint_replay"], True)
        raw["protocol"]["checkpoint_replay"] = False
        row = table.config_to_row(raw, env_id=key[0], mode=key[1], task=entries()[key].task)
        self.assertEqual(row["protocol_checkpoint_replay"], "False")
        dropped = table.parse_ph_row(row).config
        self.assertIs(protocol_from_config(dropped, "x")["checkpoint_replay"], False)
        row["protocol_checkpoint_replay"] = ""
        absent = table.parse_ph_row(row).config
        self.assertNotIn("checkpoint_replay", absent["protocol"])
        self.assertIs(protocol_from_config(absent, "x")["checkpoint_replay"], True)
        raw["protocol"]["checkpoint_replay"] = "no"
        with self.assertRaisesRegex(ValueError, "checkpoint_replay"):
            protocol_from_config(raw, "x")

    def test_a_source_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])
        self.assertTrue(parse_args(["--list"]).list)

    def test_list_shows_every_row(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--list"])
        printed = output.getvalue().splitlines()
        # The table's header and rows, then the arms (list_arms).
        lines = printed[: 1 + len(table.load_ph_table())]
        self.assertEqual(printed[len(lines) :], list_arms())
        builds = [line.split()[0] for line in lines[1:] if line.endswith("builds")]
        self.assertEqual(
            sorted(builds),
            sorted(env for (env, _), entry in entries().items() if implemented(entry)),
        )
        self.assertLessEqual({env for env, _ in ROUND_TRIPS}, set(builds))
        stubs = [line.split()[0] for line in lines[1:] if line.endswith("parses; not implemented")]
        self.assertEqual(sorted(stubs), sorted(unported()))


if __name__ == "__main__":
    unittest.main()
