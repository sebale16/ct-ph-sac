"""CT-PH-SAC hyperparameter table: benchmarks/hyperparams/ct_ph_sac.csv <-> protocol configs.

A row, keyed by (env_id, mode), holds one complete protocol config of
benchmarks/run_ph_sac.py: the JSON of benchmarks/configs/ph_sac/*.json without its
_experiment_choices, flattened into columns in the style of the other tables in
benchmarks/hyperparams/:

    mode, env_id   the row key; env_id is "<domain>-<task>" (cartpole-swingup), mode is
                   top/second/third as in the other tables (smoke: the smoke-test config;
                   lyapunov_cart, lyapunov_cart_straight and lyapunov_shoulder: optional
                   reward variants of a top row, make_ct_ph_sac_table.VARIANTS, unused in
                   the final run; irregular: a top row on the final run's irregular
                   clock, make_ct_ph_sac_table.IRREGULAR_ROWS)
    env_*          env_testbed is the testbed name; env_<key> is testbed_settings.<key>,
                   the keyword arguments of the testbed's factory (registered in
                   environment/ph_sac/registry.py; common/ph_config.py for the in-house
                   testbeds, common/ph_dmc/ for the dm_control ports)
    task_*         the TestbedConfig that the testbed's factory declares: dimensions
                   (task_nq the physical positions, task_context_dim the context),
                   physics step, integrator, clock, episode, discount rate, decision box,
                   motor limits, controller settings (task_refresh_mode: when the
                   feedback law is evaluated, "continuous" in every factory and row,
                   common.ph_config.REFRESH_MODES). They are not part of the protocol
                   config. For a ported testbed they are checked against the factory
                   when the protocol is built; for an unported one they specify what its
                   factory must declare. The optional top-level config key refresh_mode,
                   which replaces the factory's, has no column: a config that sets it
                   stays a JSON file (benchmarks/run_ph_sac.py --refresh-mode sets it on
                   a row's protocol)
    model_*        model_step, the plant's integration step, per task: the generator
                   writes common.ph_config.MODEL_STEPS (default_model_step; the physics
                   step for a testbed not listed), a divisor of the decision interval
                   pinned against measured bounds by tests/test_ph_model_step.py;
                   model_integrator (optional: an empty cell is "rk4", the
                   default; "semi_implicit" on the Euler rows, the windy pendulum and
                   the swimmers); model_coarse_step (optional: an empty cell is off),
                   the coarse step of contact-triggered fine substepping, the generator
                   writes common.ph_config.MODEL_COARSE_STEPS (the cart-poles 0.005, the
                   point mass 0.01); model_contact_handling (optional: an empty cell is
                   "port"; "exclude" is contact exclusion,
                   common.ph_config.CONTACT_HANDLINGS, which needs model_limit_port False:
                   the six cart-pole rows and point_mass-easy since 2026-11-11,
                   make_ct_ph_sac_table.CONTACT_HANDLING);
                   and network.<key> as model_<key> (model_damping_velocity
                   and model_limit_port: an empty cell is False; damping_velocity True on
                   the swimmer rows, limit_port True on the reacher and swimmer rows; the
                   optional network keys actor_log_std_init and input_matrix_start have no
                   column: a config that sets them stays a JSON file)
    algo_*         training.<key> as algo_training_<key> (algo_training_plant_step_dense_until
                   and algo_training_plant_step_every: the learned plant's online-step
                   schedule, common.ph_config.plant_step_schedule; empty cells are 0 and 1,
                   a step on every row, the schedule of every run before 2026-11-10; the
                   dm_control rows set make_ct_ph_sac_table.PLANT_STEP_SCHEDULE, dense
                   until 100k decisions then every 4th row, confirmed 2026-11-11; algo_training_
                   plant_diagnostics_every: an empty cell is the agents' 10,000; the
                   optional training key warmup_decisions has no column: a config that
                   sets it stays a JSON file),
                   temperature.<key> as
                   algo_temperature_<key>, explicit.<key> as algo_explicit_<key>,
                   implicit.<key> as algo_implicit_<key>, and implicit.solver/rate/
                   admission.<key> as algo_implicit_solver_<key>, algo_implicit_rate_<key>,
                   algo_implicit_admission_<key>
    protocol_*     protocol.<key>
    comment        free text: provenance, choices and open items of the row

Cell syntax (every cell is parsed by parse_cell and written by format_cell):

    (empty)        the key is absent: a NotRequired setting falls back to its default, a
                   required one is reported missing when the protocol is built
    none           JSON null, except in a string column, where it is the string "none"
                   (network.diffusion none); no string setting is ever null
    True, False    booleans
    4, 0.0003      numbers: an integer literal is an int, anything else a float; floats are
                   written with repr(), so every value round-trips exactly
    256,256        integer lists (hidden sizes, coordinate indices); [] is the empty list
    0-9,12         seed lists: comma-separated integers or inclusive ranges a-b. Training
                   seeds are written out one by one, evaluation reset seeds as a range
    -1.0,0.0,0.0   float lists (decision box, motor limits)
    k=v;k=v        dicts of scalars (failure_limits and the reward, reset and plant
                   parameters), the form common.utils._parse_dict_from_string reads; {} is
                   the empty dict
    text           strings (testbed name, activation, device, method choices, output path)

protocol.theory_diagnostics, a nested optional section, has no columns; a config that
sets it stays a JSON file.

Decision boxes (task_decision_low/high) of the dm_control rows follow one rule, applied
by benchmarks/make_ct_ph_sac_table.py: per actuated coordinate the setpoint lies within
+-offset (an offset from the current position, or an absolute wrapped angle on the
pendulum), the stiffness in [0, k_max] with k_max = 10 * torque_limit / offset, and the
damping in [0, 0.2 s * k_max]. The spring torque at the full offset is thus ten times the
motor limit and the damping time constant 0.2 s. The rule gives the cart-pole and
in-house Acrobot box (+-1, 0-100, 0-20 at +-10) and scales the weaker motors (reacher
0.05 N m -> 0.5 and 0.1; point mass 0.1 N over +-0.3 m -> 3.33 and 0.67; pendulum 1 N m
over +-pi -> 3.18 and 0.64; lqr at the chosen +-20 N -> 200 and 40). The entropy target
is decision_box_entropy_target of the resulting box: -1 per decision dimension of the
normalized box, -3 for one actuator and -6 for two, whatever the box's units. The windy
pendulum rows keep the reference runs' box, which predates the rule. That box and target
are the impedance-law arms'; the paper arms decide torques, and each derives its box (the
row's task_torque_low/high) and target (-1 per torque) per run
(common.ph_config.torque_decision_config). No column differs between the decision modes.
"""

from __future__ import annotations

import copy
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from common.ph_config import refresh_mode_setting

DEFAULT_HYPERPARAMS_DIR = Path(__file__).resolve().parent / "hyperparams"
TABLE_NAME = "ct_ph_sac.csv"

_STR = "str"
_NUMBER = "number"
_BOOL = "bool"
_INTS = "int_list"
_SEEDS = "int_ranges"
_FLOATS = "float_list"
_DICT = "dict"
_KINDS = (_STR, _NUMBER, _BOOL, _INTS, _SEEDS, _FLOATS, _DICT)


@dataclass(frozen=True)
class Column:
    """One table column: the nested key it holds, how its cell is parsed, and where.

    target "config" columns fill the protocol config; target "task" columns fill the
    declared TestbedConfig fields. compress writes runs of three or more consecutive
    integers as a range a-b.
    """

    name: str
    path: Tuple[str, ...]
    kind: str
    target: str = "config"
    compress: bool = False


def _columns(
    prefix: str,
    path: Tuple[str, ...],
    keys: Sequence[Tuple[str, str]],
    *,
    target: str = "config",
) -> Tuple[Column, ...]:
    return tuple(Column(prefix + key, path + (key,), kind, target) for key, kind in keys)


ENV_COLUMNS: Tuple[Column, ...] = (
    Column("env_testbed", ("testbed",), _STR),
    *_columns(
        "env_",
        ("testbed_settings",),
        (
            ("task", _STR),
            ("clock_kind", _STR),
            ("tail_probability", _NUMBER),
            ("tail_split", _NUMBER),
            ("beta_shape", _NUMBER),
            ("torque_limit", _NUMBER),
            ("failure_limits", _DICT),
            ("reward_parameters", _DICT),
            ("reset_parameters", _DICT),
            ("plant_parameters", _DICT),
        ),
    ),
)

TASK_COLUMNS: Tuple[Column, ...] = _columns(
    "task_",
    (),
    (
        ("nq", _NUMBER),
        ("nv", _NUMBER),
        ("nu", _NUMBER),
        ("context_dim", _NUMBER),
        ("periodic_position_indices", _INTS),
        ("actuated_position_indices", _INTS),
        ("physics_dt", _NUMBER),
        ("integrator", _STR),
        ("reference_period", _NUMBER),
        ("clock_kind", _STR),
        ("min_duration", _NUMBER),
        ("max_duration", _NUMBER),
        ("episode_duration", _NUMBER),
        ("discount_rate", _NUMBER),
        ("decision_low", _FLOATS),
        ("decision_high", _FLOATS),
        ("torque_low", _FLOATS),
        ("torque_high", _FLOATS),
        ("setpoint_mode", _STR),
        ("setpoint_error", _STR),
        ("refresh_steps", _NUMBER),
        ("refresh_mode", _STR),
    ),
    target="task",
)

METHOD_COLUMNS: Tuple[Column, ...] = (
    Column("model_step", ("model_step",), _NUMBER),
    Column("model_integrator", ("model_integrator",), _STR),
    Column("model_coarse_step", ("model_coarse_step",), _NUMBER),
    Column("model_contact_handling", ("contact_handling",), _STR),
    *_columns(
        "model_",
        ("network",),
        (
            ("hidden_sizes", _INTS),
            ("plant_hidden_sizes", _INTS),
            ("activation", _STR),
            ("mass_floor", _NUMBER),
            ("diffusion_floor", _NUMBER),
            ("damping_rank", _NUMBER),
            ("velocity_feature_scale", _NUMBER),
            ("diffusion", _STR),
            ("damping_velocity", _BOOL),
            ("limit_port", _BOOL),
        ),
    ),
    *_columns(
        "algo_training_",
        ("training",),
        (
            ("seed", _NUMBER),
            ("device", _STR),
            ("replay_capacity", _NUMBER),
            ("batch_size", _NUMBER),
            ("learning_rate", _NUMBER),
            ("gradient_clip", _NUMBER),
            ("warmup_episodes", _NUMBER),
            ("validation_fraction", _NUMBER),
            ("plant_step_dense_until", _NUMBER),
            ("plant_step_every", _NUMBER),
            ("plant_diagnostics_every", _NUMBER),
        ),
    ),
    *_columns(
        "algo_temperature_",
        ("temperature",),
        (
            ("initial_alpha_c", _NUMBER),
            ("learn_temperature", _BOOL),
            ("target_entropy", _NUMBER),
        ),
    ),
    *_columns(
        "algo_explicit_",
        ("explicit",),
        (
            ("plant_source", _STR),
            ("continuation", _STR),
            ("polyak_rate", _NUMBER),
            ("rollout_paths", _NUMBER),
        ),
    ),
    *_columns(
        "algo_implicit_",
        ("implicit",),
        (
            ("plant_source", _STR),
            ("reward_source", _STR),
            ("reward_warm_steps", _NUMBER),
            ("plant_warm_steps", _NUMBER),
            ("plant_fit_covariance", _STR),
            ("temperature_every_rows", _NUMBER),
        ),
    ),
    *_columns(
        "algo_implicit_solver_",
        ("implicit", "solver"),
        (
            ("resolvent_step", _NUMBER),
            ("policy_draws", _NUMBER),
            ("inner_steps", _NUMBER),
            ("residual_estimator", _STR),
        ),
    ),
    *_columns(
        "algo_implicit_rate_",
        ("implicit", "rate"),
        (
            ("replay_weight_max", _NUMBER),
            ("noise_variance_floor", _NUMBER),
            ("rollout_weight", _NUMBER),
            ("model_weight", _NUMBER),
            ("rollout_paths", _NUMBER),
            ("model_points", _NUMBER),
            ("rate_steps", _NUMBER),
            ("actor_steps", _NUMBER),
        ),
    ),
    *_columns(
        "algo_implicit_admission_",
        ("implicit", "admission"),
        (
            ("efolds", _NUMBER),
            ("margin_z", _NUMBER),
            ("initial_rows", _NUMBER),
            ("initial_paths", _NUMBER),
            ("pilot_batches", _NUMBER),
            ("real_row_budget", _NUMBER),
            ("simulated_path_budget", _NUMBER),
            ("allocation", _STR),
            ("inference", _STR),
            ("plant_update", _STR),
            ("polyak_rate", _NUMBER),
            ("gate_failure_budget", _NUMBER),
            ("return_bound", _NUMBER),
            ("min_horizon", _NUMBER),
        ),
    ),
    *_columns(
        "protocol_",
        ("protocol",),
        (
            ("interaction_budget", _NUMBER),
            ("temperature_per_second", _NUMBER),
            ("training_seeds", _SEEDS),
        ),
    ),
    Column(
        "protocol_evaluation_reset_seeds",
        ("protocol", "evaluation_reset_seeds"),
        _SEEDS,
        compress=True,
    ),
    *_columns(
        "protocol_",
        ("protocol",),
        (
            ("evaluation_every_interactions", _NUMBER),
            ("deterministic_evaluation", _BOOL),
            ("ablation_polyak_rate", _NUMBER),
            ("baseline_expectation_samples", _NUMBER),
            ("checkpoint_replay", _BOOL),
            ("output_directory", _STR),
        ),
    ),
)

KEY_COLUMNS = ("mode", "env_id")
COMMENT_COLUMN = "comment"
CONFIG_COLUMNS: Tuple[Column, ...] = ENV_COLUMNS + METHOD_COLUMNS
SCHEMA: Tuple[Column, ...] = ENV_COLUMNS + TASK_COLUMNS + METHOD_COLUMNS
TABLE_COLUMNS: Tuple[str, ...] = (
    KEY_COLUMNS + tuple(column.name for column in SCHEMA) + (COMMENT_COLUMN,)
)
_IGNORED_CONFIG_KEYS = ("_experiment_choices",)

_INT_LITERAL = re.compile(r"[+-]?\d+")
_RANGE = re.compile(r"(\d+)-(\d+)")


@dataclass(frozen=True)
class PHTableEntry:
    """One parsed row: its protocol config, its declared TestbedConfig fields, its comment."""

    env_id: str
    mode: str
    config: dict
    task: dict
    comment: str
    source: str


# ----------------------------------------------------------------------------- cells


def _as_list(values: object) -> list:
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    return list(values)


def parse_scalar(text: str) -> object:
    """A dict value: none, an int literal, a float, or else the text itself."""
    value = text.strip()
    if value.lower() == "none":
        return None
    if _INT_LITERAL.fullmatch(value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _parse_number(text: str) -> object:
    value = parse_scalar(text)
    if value is not None and not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {text!r}")
    return value


def _parse_int(text: str) -> int:
    value = text.strip()
    if not _INT_LITERAL.fullmatch(value):
        raise ValueError(f"expected an integer, got {text!r}")
    return int(value)


def _parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        match = _RANGE.fullmatch(part)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if last < first:
                raise ValueError(f"empty seed range {part!r}")
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(_parse_int(part))
    return seeds


def _parse_dict(text: str) -> dict:
    if text == "{}":
        return {}
    result: dict = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"expected key=value, got {part!r}")
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = parse_scalar(value)
    if not result:
        raise ValueError(f"expected key=value pairs or {{}}, got {text!r}")
    return result


def parse_cell(kind: str, text: Optional[str]) -> Tuple[bool, object]:
    """Return (present, value); an empty cell is an absent key."""
    if kind not in _KINDS:
        raise ValueError(f"Unknown cell kind {kind!r}.")
    value = (text or "").strip()
    if value == "":
        return False, None
    if kind == _STR:
        return True, value
    if value.lower() == "none":
        return True, None
    if kind == _NUMBER:
        return True, _parse_number(value)
    if kind == _BOOL:
        if value.lower() not in ("true", "false"):
            raise ValueError(f"expected True or False, got {value!r}")
        return True, value.lower() == "true"
    if kind == _DICT:
        return True, _parse_dict(value)
    if value == "[]":
        return True, []
    if kind == _INTS:
        return True, [_parse_int(part) for part in value.split(",")]
    if kind == _SEEDS:
        return True, _parse_seeds(value)
    return True, [float(part) for part in value.split(",")]


def _format_number(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return str(value) if isinstance(value, int) else repr(float(value))


def _format_text(value: object) -> str:
    if not isinstance(value, str) or value.strip() != value or not value or "\n" in value:
        raise ValueError(f"string {value!r} has no cell form")
    return value


def _format_dict_value(value: object) -> str:
    if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return _format_number(value)
    text = _format_text(value)
    if ";" in text or parse_scalar(text) != text:
        raise ValueError(f"dict value {value!r} has no cell form")
    return text


def _format_seeds(values: Sequence[int], compress: bool) -> str:
    parts: List[str] = []
    index = 0
    while index < len(values):
        end = index
        while (
            compress
            and end + 1 < len(values)
            and values[end + 1] == values[end] + 1
        ):
            end += 1
        if end - index >= 2:
            parts.append(f"{values[index]}-{values[end]}")
            index = end + 1
        else:
            parts.append(str(values[index]))
            index += 1
    return ",".join(parts)


def format_cell(column: Column, value: object) -> str:
    """The cell that parse_cell(column.kind, cell) reads back as value."""
    kind = column.kind
    if kind == _STR:
        return _format_text(value)
    if value is None:
        return "none"
    if kind == _NUMBER:
        return _format_number(value)
    if kind == _BOOL:
        if not isinstance(value, bool):
            raise ValueError(f"{column.name}: expected a bool, got {value!r}")
        return "True" if value else "False"
    if kind == _DICT:
        if not isinstance(value, Mapping):
            raise ValueError(f"{column.name}: expected a dict, got {value!r}")
        if not value:
            return "{}"
        for key in value:
            if not isinstance(key, str) or not key or set(key) & set("=; "):
                raise ValueError(f"{column.name}: key {key!r} has no cell form")
        return ";".join(f"{key}={_format_dict_value(item)}" for key, item in value.items())
    values = _as_list(value)
    if not values:
        return "[]"
    if kind == _FLOATS:
        return ",".join(repr(float(item)) for item in values)
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{column.name}: expected integers, got {item!r}")
    if kind == _SEEDS:
        if any(item < 0 for item in values):
            raise ValueError(f"{column.name}: seeds must be nonnegative")
        return _format_seeds(values, column.compress)
    return ",".join(str(item) for item in values)


# ------------------------------------------------------------------------------ rows


def _set_path(target: dict, path: Tuple[str, ...], value: object) -> None:
    node = target
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _get_path(source: Mapping[str, object], path: Tuple[str, ...]) -> Tuple[bool, object]:
    node: object = source
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return False, None
        node = node[key]
    return True, node


def parse_ph_row(row: Mapping[Optional[str], Optional[str]], source: str = "row") -> PHTableEntry:
    """Rebuild the protocol config and the declared TestbedConfig fields of one row."""
    if None in row:
        raise ValueError(f"{source}: more cells than columns.")
    unknown = sorted(set(row) - set(TABLE_COLUMNS))
    if unknown:
        raise ValueError(f"{source}: unknown columns {unknown}.")
    env_id = (row.get("env_id") or "").strip()
    mode = (row.get("mode") or "").strip()
    if not env_id or not mode:
        raise ValueError(f"{source}: env_id and mode are required.")
    config: dict = {}
    task: dict = {}
    for column in SCHEMA:
        try:
            present, value = parse_cell(column.kind, row.get(column.name))
        except ValueError as error:
            raise ValueError(f"{source} ({env_id}, {mode}), {column.name}: {error}") from None
        if present:
            _set_path(config if column.target == "config" else task, column.path, value)
    return PHTableEntry(
        env_id=env_id,
        mode=mode,
        config=config,
        task=task,
        comment=(row.get(COMMENT_COLUMN) or "").strip(),
        source=source,
    )


_CONFIG_PATHS = frozenset(column.path for column in CONFIG_COLUMNS)


def _flatten_keys(mapping: Mapping[str, object], prefix: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    """Leaf paths of a config: a column path, or a key no column reaches."""
    paths: List[Tuple[str, ...]] = []
    for key, value in mapping.items():
        path = prefix + (key,)
        nested = isinstance(value, Mapping) and any(
            column[: len(path)] == path for column in _CONFIG_PATHS
        )
        if path in _CONFIG_PATHS or not nested:
            paths.append(path)
        else:
            paths.extend(_flatten_keys(value, path))
    return paths


def config_to_row(
    config: Mapping[str, object],
    *,
    env_id: str,
    mode: str = "top",
    task: Optional[Mapping[str, object]] = None,
    comment: str = "",
) -> Dict[str, str]:
    """The table row of a protocol config (the inverse of parse_ph_row).

    A key without a column (protocol.theory_diagnostics) raises ValueError; such a
    config stays a JSON file. _experiment_choices is dropped.
    """
    body = {key: value for key, value in config.items() if key not in _IGNORED_CONFIG_KEYS}
    unmapped = [".".join(path) for path in _flatten_keys(body, ()) if path not in _CONFIG_PATHS]
    if unmapped:
        raise ValueError(f"{env_id} ({mode}): settings without a table column: {unmapped}.")
    unknown_task = sorted(set(task or {}) - {column.path[0] for column in TASK_COLUMNS})
    if unknown_task:
        raise ValueError(f"{env_id} ({mode}): unknown task fields {unknown_task}.")
    row = {name: "" for name in TABLE_COLUMNS}
    row["mode"] = mode
    row["env_id"] = env_id
    row[COMMENT_COLUMN] = comment
    for column in SCHEMA:
        source = body if column.target == "config" else (task or {})
        present, value = _get_path(source, column.path)
        if present:
            row[column.name] = format_cell(column, value)
    return row


# ----------------------------------------------------------------------------- table


def table_path(hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR) -> Path:
    return Path(hyperparams_dir) / TABLE_NAME


def load_ph_table(
    hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR,
) -> List[PHTableEntry]:
    """Parse every row of ct_ph_sac.csv; (env_id, mode) must be unique."""
    path = table_path(hyperparams_dir)
    if not path.exists():
        raise FileNotFoundError(f"CT-PH-SAC hyperparameter table not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        duplicated = sorted({name for name in header if header.count(name) > 1})
        unknown = sorted(set(header) - set(TABLE_COLUMNS))
        if duplicated or unknown:
            raise ValueError(f"{path}: duplicated columns {duplicated}, unknown columns {unknown}.")
        entries = [
            parse_ph_row(row, f"{path}:{reader.line_num}") for row in reader
        ]
    seen: Dict[Tuple[str, str], str] = {}
    for entry in entries:
        key = (entry.env_id, entry.mode)
        if key in seen:
            raise ValueError(f"{entry.source}: duplicate row {key}, first at {seen[key]}.")
        seen[key] = entry.source
    return entries


def find_ph_entry(
    env_id: str,
    mode: str = "top",
    hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR,
) -> PHTableEntry:
    entries = load_ph_table(hyperparams_dir)
    for entry in entries:
        if entry.env_id == env_id and entry.mode == mode:
            return entry
    modes = sorted(entry.mode for entry in entries if entry.env_id == env_id)
    known = sorted({entry.env_id for entry in entries})
    hint = f"modes for {env_id}: {modes}" if modes else f"known env_ids: {known}"
    raise KeyError(
        f"No row in {table_path(hyperparams_dir)} for env_id={env_id!r}, mode={mode!r}; {hint}."
    )


def load_ph_hyperparams_from_table(
    env_id: str,
    mode: str = "top",
    hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR,
) -> dict:
    """The full protocol config dict of one row, in the shape of a ph_sac JSON config.

    It parses for every row. Whether the protocol can be built depends on the testbed
    having a factory; run_ph_sac.load_protocol_from_table builds it.
    """
    return copy.deepcopy(find_ph_entry(env_id, mode, hyperparams_dir).config)


# ---------------------------------------------------------- declared vs. factory


_TASK_FIELDS: Dict[str, Callable[[Mapping], object]] = {
    # task_nq counts physical positions: an episode-constant context (the reacher's
    # target) sits at the end of the position block and is declared by task_context_dim.
    "nq": lambda t: t["state_spec"]["position_dim"] - t["state_spec"].get("context_dim", 0),
    "nv": lambda t: t["state_spec"]["velocity_dim"],
    "nu": lambda t: t["state_spec"]["action_dim"],
    "context_dim": lambda t: t["state_spec"].get("context_dim", 0),
    "periodic_position_indices": lambda t: _as_list(t["state_spec"]["periodic_position_indices"]),
    "actuated_position_indices": lambda t: _as_list(t["state_spec"]["actuated_position_indices"]),
    "physics_dt": lambda t: t["physics_dt"],
    "integrator": lambda t: t["integrator"],
    "reference_period": lambda t: t["reference_period"],
    "clock_kind": lambda t: t["clock"]["kind"],
    "min_duration": lambda t: t["clock"]["min_duration"],
    "max_duration": lambda t: t["clock"]["max_duration"],
    "episode_duration": lambda t: t["episode_duration"],
    "discount_rate": lambda t: t["discount_rate"],
    "decision_low": lambda t: _as_list(t["controller"]["decision_low"]),
    "decision_high": lambda t: _as_list(t["controller"]["decision_high"]),
    "torque_low": lambda t: _as_list(t["controller"]["torque_low"]),
    "torque_high": lambda t: _as_list(t["controller"]["torque_high"]),
    "setpoint_mode": lambda t: t["controller"]["setpoint_mode"],
    "setpoint_error": lambda t: t["controller"]["setpoint_error"],
    "refresh_steps": lambda t: t["controller"]["refresh_steps"],
    "refresh_mode": lambda t: refresh_mode_setting(t["controller"]),
}


def declared_testbed_mismatches(
    task: Mapping[str, object], testbed_config: Mapping[str, object]
) -> List[str]:
    """Declared task_* fields that differ from the factory's TestbedConfig.

    Absent declarations are skipped; values are compared exactly, lists elementwise.
    """
    mismatches: List[str] = []
    for name, read in _TASK_FIELDS.items():
        if name in task and not _same_value(task[name], read(testbed_config)):
            mismatches.append(
                f"task_{name}: table {task[name]!r}, factory {read(testbed_config)!r}"
            )
    return mismatches


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_value(declared: object, actual: object) -> bool:
    if isinstance(declared, list):
        return (
            isinstance(actual, list)
            and len(declared) == len(actual)
            and all(_same_value(a, b) for a, b in zip(declared, actual))
        )
    if _is_number(declared) and _is_number(actual):
        return float(declared) == float(actual)
    return declared == actual
