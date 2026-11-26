#!/usr/bin/env python
"""Read-only accuracy of a finished run's learned port-Hamiltonian plant on its own replay.

    python -u benchmarks/ph_plant_checkpoint_accuracy.py \
        --runs RUN/cartpole_three_poles/implicit_torque/seed_0 [RUN/.../seed_1 ...] \
        --out plant_accuracy.csv [--rows 20000] [--seed 0] [--partition train] \
        [--state-file auto] [--plant-key online_plant] [--check-rows 16] [--threads N]

Run from the repository root, CPU only, under torch.no_grad(). Each --runs entry is a seed
folder <output>/<testbed>/<arm>/seed_<s> (a file inside it is accepted too). The script
writes nothing but the --out CSV (appended, one row per run; the header is written when
the file is absent; a file with another header is left untouched), never a run folder. A
failing run prints its traceback and the loop goes on; the exit code is 1 if any failed.

Sources (the repository's own code; nothing is re-implemented):
  checkpoint.pt  PHAlgorithm.state_dict (algorithms/ph_base.py), written by
                 evaluations/ph_sac/protocol._train_run.save_checkpoint; holds "replay"
                 only under protocol checkpoint_replay true (the default).
  resume.pt      evaluations/ph_sac/resume.resume_payload: its "agent" entry is
                 PHAlgorithm.resume_state, the same state_dict plus the replay; a run that
                 completes deletes it.
  Both are read with torch.load(map_location="cpu", weights_only=False), as the repo's
  loaders do. --state-file auto reads every present file; the plant comes from the one
  with more interactions (checkpoint.pt on a tie), the replay from that file if it holds
  one, else from the other. The config is the file's state["config"], the arm config the
  agent was built with; protocol.json (two folders up) is only cross-checked.
  Environment: environment.ph_sac.tasks.make_env(config["testbed"], decision mode of its
  controller, training seed). Learned plant: models.ph_sac.plant.PortHamiltonianPlant(
  env.state_spec, config["network"], learn_diffusion=diffusion_is_learned(network)),
  loaded strictly from state["networks"][--plant-key]. True plant: make_oracle_plant(env).
  Torque: env.controller.action(s, env.controller.resolve_decision(s, decision)), as
  PHAlgorithm.record_plant_diagnostics. Flagged rows: the replay's stored near_limit |
  limit_touched (PHReplayBuffer.batch contact_flags, rollouts.batch_contact_flags), the
  rows the plant fit and the diagnostics leave out under contact_handling "exclude".
  Arithmetic in the stored plant's dtype (float32 for the runs), as the diagnostics.

Printed per run. b = drift ds/dt [B,D] at a stored state s and its stored torque; the
state is (y, v): y = P q the observed positions, v = qvel the velocities (cart-poles: y0
cart x [m], y1.. hinge angles [rad], theta_1 from upright, later ones relative; v their
rates). The position block of b is v [m/s, rad/s] in both plants (its error must be 0);
the velocity block holds accelerations [m/s^2, rad/s^2]. e = b_learned - b_true.
  G             learned force_matrix and the true env.oracle_input_matrix(): generalized
                force per unit of the drift's torque input, the motor torque in the box
                [torque_low, torque_high] (cart-poles: newtons, gear 1), not the actor's
                normalized [-1,1] action (column j times the half-width h_j gives that).
                (M, U, D, G) -> c (M, U, D, G) leaves the drift unchanged, so G is fixed
                only up to a scalar c: the gauge fit c = argmin ||M_learned - c M_true||_F
                over sampled states (M_true from MuJoCo, env._mass_matrix) prints G/c.
  drift table   per coordinate: RMS over rows of b_true_i, of e_i, and their ratio; then
                ||e||_rms / ||b_true||_rms over all D coordinates (units mixed, as the
                logged drift_error_max) and over the velocity block alone; median, p90 and
                p99 of the per-row ||e|| / ||b_true||; max ||e|| (the logged quantity) with
                that row's ||b_true||, state and torque. Unflagged rows in full, flagged
                rows briefly.
  action effect d_j = b(s, u_j = high_j) - b(s, u_j = low_j), other actuators at the box
                midpoint: M^-1 G column j times the box width, the part of the drift the
                torque moves (gauge-free). Per velocity coordinate RMS of d_true, of the
                error and the ratio, the overall relative RMS, the mean cosine between
                learned and true d over rows, and the row mean of d per unit torque.
  action-free   b(s, 0) learned vs true, the same table.
  age split     overall relative RMS on the oldest and the newest 20% of the replay (ring
                order from the buffer's position and size, checked by episode ids).
  logged        the last row of oracle.csv beside this script's max over all unflagged
                rows and beside emulated diagnostic batches: batch_size rows drawn with
                replacement from the sampled rows, flagged dropped under exclusion (as
                record_plant_diagnostics), max ||e|| per batch, p10/p50/p90 over batches.
  checks        (self-checks raise and fail the run) measure_oracle_errors of the oracle
                against a second oracle is 0; measure_oracle_errors of the learned plant
                matches this script's max; shapes [B,D] and [B,m]; torques in the box; the
                effect's position block is 0. (printed) on --check-rows unflagged rows:
                the oracle integrated over the stored duration with the stored decision
                (integrate_endpoint_moments at the physics step, the physics map) and a
                MuJoCo replay (env.step_dt from the stored state) against the stored next
                state, and the oracle drift against MuJoCo's forward dynamics.
CSV columns: CSV_COLUMNS below (G and per-coordinate values as JSON strings).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import environment.ph_sac.dmc  # noqa: E402,F401  (registers the dm_control ports)
import environment.ph_sac.tasks  # noqa: E402,F401  (registers the in-house testbeds)
from common.ph_config import (  # noqa: E402
    contact_handling_setting,
    decision_mode_setting,
    input_matrix_start_setting,
    model_integrator_setting,
)
from environment.ph_sac.tasks import make_env, make_oracle_plant  # noqa: E402
from evaluations.ph_sac.diagnostics import measure_oracle_errors  # noqa: E402
from models.ph_sac.plant import (  # noqa: E402
    LIMIT_WIDTH_FLOOR,
    PortHamiltonianPlant,
    diffusion_is_learned,
)
from models.ph_sac.rollouts import (  # noqa: E402
    integrate_endpoint_moments,
    model_contact_exclusion,
    start_state_flags,
    wrapped_endpoint_residual,
)

# Rows per drift call (memory bound only; the drift is row-wise).
CHUNK_ROWS = 4096
# Rows of the measure_oracle_errors self-checks.
SELF_CHECK_ROWS = 4096
# Emulated diagnostic batches.
BOOTSTRAP_BATCHES = 200
# States of the mass-matrix gauge fit.
GAUGE_ROWS = 2000
# Oldest / newest share of the replay for the age split.
AGE_FRACTION = 0.2
PARTITIONS = ("train", "gate")
CSV_COLUMNS = (
    # the requested columns
    "run_dir",
    "task",
    "arm",
    "seed",
    "interactions",
    "n_unflagged",
    "n_flagged",
    "overall_rel_rms_unflagged",
    "action_effect_rel_rms",
    "action_free_rel_rms",
    "max_error_norm_unflagged",
    "learned_G",
    "true_G",
    "per_coordinate_rel_rms_unflagged",
    # extras
    "velocity_rel_rms_unflagged",
    "per_row_rel_median_unflagged",
    "per_row_rel_p90_unflagged",
    "action_effect_mean_cosine",
    "flagged_overall_rel_rms",
    "oldest20_overall_rel_rms",
    "newest20_overall_rel_rms",
    "gauge_scale_c",
    "gauge_mass_residual",
    "learned_G_over_c",
    "emulated_logged_max_p50",
    "logged_interactions",
    "logged_drift_error_max",
    "logged_contact_flagged_share",
    "sampled_flagged_share",
    "plant_file",
    "plant_file_interactions",
    "replay_file",
    "replay_file_interactions",
    "partition",
    "rows_sampled",
    "dtype",
)


# ----------------------------------------------------------------------------- helpers


def say(*parts: object) -> None:
    print(*parts, flush=True)


class SelfCheckError(AssertionError):
    """A self-check failed: the script's conventions do not match the run's."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def _np(value: object) -> np.ndarray:
    """A stored array as numpy (checkpoint.pt keeps numpy arrays, resume.pt tensors)."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rms(values: np.ndarray, axis: Optional[int] = None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def _ratio(numerator: object, denominator: object):
    """numerator / denominator, NaN where the denominator is not positive."""
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    shape = np.broadcast(top, bottom).shape
    out = np.full(shape, np.nan)
    np.divide(
        np.broadcast_to(top, shape),
        np.broadcast_to(bottom, shape),
        out=out,
        where=np.broadcast_to(bottom > 0.0, shape),
    )
    return out if out.ndim else float(out)


def _quantiles(values: np.ndarray, levels: Sequence[float] = (50.0, 90.0, 99.0)) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [math.nan for _ in levels]
    return [float(value) for value in np.percentile(finite, list(levels))]


def _array(values: object, precision: int = 5) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float64), precision=precision, max_line_width=160
    )


def _json_float(value: object) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".8g")
    return str(value)


# ----------------------------------------------------------------------------- loading


def _load(path: Path) -> object:
    """As the repo's own loaders (resume.load_resume_payload, PHAlgorithm.load)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _agent_state(loaded: object, path: Path) -> Tuple[Mapping[str, object], str]:
    """(agent state_dict, how it was found): checkpoint.pt's top level or resume.pt's
    "agent" entry."""
    if isinstance(loaded, Mapping):
        if "networks" in loaded and "config" in loaded:
            return loaded, "PHAlgorithm.state_dict at the top level"
        agent = loaded.get("agent")
        if isinstance(agent, Mapping) and "networks" in agent and "config" in agent:
            return agent, "resume payload['agent'] (PHAlgorithm.resume_state)"
    raise ValueError(
        f"{path} holds no agent state (no 'networks' and 'config' at the top level or under 'agent')."
    )


def _replay_sizes(replay: Optional[Mapping[str, object]]) -> str:
    if replay is None:
        return "absent"
    parts = []
    for name, buffer in replay["buffers"].items():
        parts.append(
            f"{name}: size {int(buffer['size'])}, capacity {int(buffer['capacity'])}, "
            f"position {int(buffer['position'])}"
        )
    return "present (" + "; ".join(parts) + ")"


def load_run_states(run_dir: Path, mode: str) -> dict:
    """Read checkpoint.pt and resume.pt (those present); pick the plant and replay files."""
    files = {"checkpoint": run_dir / "checkpoint.pt", "resume": run_dir / "resume.pt"}
    entries = []
    for name in ("checkpoint", "resume"):
        path = files[name]
        if not path.is_file():
            say(f"  {path.name}: absent")
            continue
        say(f"  {path.name}: {path.stat().st_size / 2**20:.1f} MiB, loading ...")
        loaded = _load(path)
        state, kind = _agent_state(loaded, path)
        top = sorted(str(key) for key in loaded.keys()) if isinstance(loaded, Mapping) else []
        interactions = int(state["counters"]["interactions"])
        say(f"    top-level keys {top}")
        say(f"    agent state: {kind}; counters.interactions {interactions}")
        if isinstance(loaded, Mapping) and "agent" in loaded:
            say(
                f"    payload interactions {loaded.get('interactions')}, format "
                f"{loaded.get('format')}, fingerprint {str(loaded.get('fingerprint'))[:12]}"
            )
        say(f"    networks {sorted(state['networks'])}")
        say(f"    replay {_replay_sizes(state.get('replay'))}")
        entries.append(
            {"name": name, "path": path, "state": state, "interactions": interactions}
        )
    if not entries:
        raise FileNotFoundError(f"{run_dir} holds neither checkpoint.pt nor resume.pt.")
    if len(entries) == 2:
        first, second = (entry["state"]["config"] for entry in entries)
        same = (
            first["testbed"]["name"] == second["testbed"]["name"]
            and int(first["training"]["seed"]) == int(second["training"]["seed"])
        )
        if not same:
            say("  WARNING: checkpoint.pt and resume.pt carry different testbeds or seeds.")
    if mode == "auto":
        plant_entry = max(
            entries, key=lambda entry: (entry["interactions"], entry["name"] == "checkpoint")
        )
    else:
        chosen = [entry for entry in entries if entry["name"] == mode]
        if not chosen:
            raise FileNotFoundError(f"--state-file {mode}: {files[mode]} is absent.")
        plant_entry = chosen[0]
    replay_entry = plant_entry if plant_entry["state"].get("replay") is not None else None
    if replay_entry is None:
        replay_entry = next(
            (entry for entry in entries if entry["state"].get("replay") is not None), None
        )
    return {
        "plant_state": plant_entry["state"],
        "plant_file": plant_entry["path"],
        "plant_interactions": plant_entry["interactions"],
        "replay": None if replay_entry is None else replay_entry["state"]["replay"],
        "replay_file": None if replay_entry is None else replay_entry["path"],
        "replay_interactions": None if replay_entry is None else replay_entry["interactions"],
    }


def replay_rows(replay: Mapping[str, object], partition: str) -> Optional[dict]:
    """The stored rows of the partition(s), with flags, ring age and episode ids.

    Rows are written from index 0 and the ring wraps only when full (PHReplayBuffer.add,
    _filled_rows), so the oldest row is index 0 before the wrap and index `position`
    after it; age = (index - oldest) mod capacity / size, 0 the oldest.
    """
    names = PARTITIONS if partition == "all" else (partition,)
    pieces = []
    order_checks: List[Optional[bool]] = []
    for name in names:
        buffer = replay["buffers"][name]
        size = int(buffer["size"])
        capacity = int(buffer["capacity"])
        position = int(buffer["position"])
        arrays = buffer["arrays"]
        if size == 0 or arrays is None:
            say(f"  replay partition {name}: empty")
            continue
        columns: Dict[str, np.ndarray] = {}
        for key in ("states", "decisions", "durations", "next_states"):
            columns[key] = np.asarray(_np(arrays[key])[:size], dtype=np.float64)
        columns["durations"] = columns["durations"].reshape(size)
        for key in ("limit_touched", "near_limit"):  # absent before 2026-11-10: False
            if key in arrays:
                columns[key] = np.asarray(_np(arrays[key])[:size]).reshape(size).astype(bool)
            else:
                columns[key] = np.zeros(size, dtype=bool)
        oldest = position % capacity if size == capacity else 0
        columns["age"] = ((np.arange(size) - oldest) % capacity) / float(size)
        ids = (replay.get("episode_ids") or {}).get(name)
        if ids is None:
            columns["episode_ids"] = np.full(size, -1, dtype=np.int64)
            order_checks.append(None)
        else:
            ids = np.asarray(_np(ids)[:size]).astype(np.int64)
            ring = (oldest + np.arange(size)) % capacity
            order_checks.append(bool(np.all(np.diff(ids[ring]) >= 0)))
            columns["episode_ids"] = ids
        say(
            f"  replay partition {name}: {size} rows (capacity {capacity}, position "
            f"{position}, wrapped {size == capacity}); episode ids nondecreasing in ring "
            f"order: {order_checks[-1]}"
        )
        pieces.append(columns)
    if not pieces:
        return None
    merged = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    if any(flag is False for flag in order_checks):
        merged["order"] = "broken"
    elif all(flag is True for flag in order_checks):
        merged["order"] = "checked"
    else:
        merged["order"] = "unchecked"
    return merged


def last_oracle_row(run_dir: Path) -> Optional[Dict[str, str]]:
    path = run_dir / "oracle.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def cross_check_protocol(run_dir: Path, config: Mapping[str, object]) -> None:
    """Compare the checkpoint's arm config with protocol.json two folders up (print only)."""
    path = run_dir.parent.parent / "protocol.json"
    if not path.is_file():
        say(f"  {path}: absent (not needed: the checkpoint carries the arm config)")
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        protocol = data.get("protocol", data)
        name = "implicit_config" if "solver" in config else "explicit_config"
        arm_config = protocol.get(name) or {}
        network = arm_config.get("network") or {}
        stored_start = input_matrix_start_setting(config["network"])
        protocol_start = network.get("input_matrix_start", "true")
        say(
            f"  protocol.json: testbed {protocol.get('testbed')!r}, checkpoint_replay "
            f"{protocol.get('checkpoint_replay', True)}, {name}.network.input_matrix_start "
            f"{protocol_start!r}, {name}.contact_handling "
            f"{arm_config.get('contact_handling', 'port')!r}"
        )
        if protocol.get("testbed") != config["testbed"]["name"]:
            say("  WARNING: protocol.json's testbed differs from the checkpoint config's.")
        if protocol_start != stored_start:
            say("  WARNING: protocol.json's input_matrix_start differs from the checkpoint config's.")
    except Exception as error:  # a cross-check only
        say(f"  protocol.json unreadable ({type(error).__name__}: {error}); ignored")


# ------------------------------------------------------------------------ computations


def chunked_drift(plant, states: torch.Tensor, torques: torch.Tensor) -> torch.Tensor:
    """plant.drift(states [B,D], torques [B,m]) -> [B,D], in chunks of CHUNK_ROWS rows."""
    check(states.ndim == 2 and torques.ndim == 2, "drift inputs must be [B,D] and [B,m]")
    check(states.shape[0] == torques.shape[0], "states and torques need the same rows")
    check(int(states.shape[1]) == int(plant.state_dim), f"states must be [B,{plant.state_dim}]")
    check(int(torques.shape[1]) == int(plant.action_dim), f"torques must be [B,{plant.action_dim}]")
    pieces = [
        plant.drift(states[start : start + CHUNK_ROWS], torques[start : start + CHUNK_ROWS])
        for start in range(0, int(states.shape[0]), CHUNK_ROWS)
    ]
    drift = torch.cat(pieces) if pieces else states.new_zeros(states.shape)
    check(tuple(drift.shape) == tuple(states.shape), f"drift shape {tuple(drift.shape)} is not [B,D]")
    return drift


def constant_torques(values: np.ndarray, rows: int, dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=dtype)
        .reshape(1, -1)
        .repeat(rows, 1)
    )


def error_statistics(error: np.ndarray, true: np.ndarray, position_dim: int) -> dict:
    """Per-coordinate and overall relative RMS errors, per-row quantiles and the max."""
    rows = int(error.shape[0])
    if rows == 0:
        return {"rows": 0}
    norm_error = np.linalg.norm(error, axis=1)
    norm_true = np.linalg.norm(true, axis=1)
    per_row = _ratio(norm_error, norm_true)
    worst = int(np.argmax(norm_error))
    velocity_error = np.linalg.norm(error[:, position_dim:], axis=1)
    velocity_true = np.linalg.norm(true[:, position_dim:], axis=1)
    rms_error = _rms(error, axis=0)
    rms_true = _rms(true, axis=0)
    return {
        "rows": rows,
        "rms_error": rms_error,
        "rms_true": rms_true,
        "ratio": _ratio(rms_error, rms_true),
        "overall": _ratio(_rms(norm_error), _rms(norm_true)),
        "velocity": _ratio(_rms(velocity_error), _rms(velocity_true)),
        "position_error_max": float(np.abs(error[:, :position_dim]).max()) if position_dim else 0.0,
        "max_error": float(norm_error[worst]),
        "true_at_worst": float(norm_true[worst]),
        "worst": worst,
        "quantiles": _quantiles(per_row),
        "zero_true_rows": int(np.sum(norm_true <= 0.0)),
        "share_above_10pct": float(np.mean(per_row > 0.1)),
        "share_above_50pct": float(np.mean(per_row > 0.5)),
    }


def print_statistics(
    title: str,
    stats: dict,
    labels: Sequence[str],
    *,
    states: Optional[np.ndarray] = None,
    torques: Optional[np.ndarray] = None,
    brief: bool = False,
) -> None:
    say(f"  {title}: {stats['rows']} rows")
    if not stats["rows"]:
        return
    if not brief:
        say(f"    {'coord':>6} {'RMS b_true':>13} {'RMS e':>13} {'RMS e / RMS b':>14}")
        for label, true, error, ratio in zip(
            labels, stats["rms_true"], stats["rms_error"], stats["ratio"]
        ):
            say(f"    {label:>6} {true:13.5g} {error:13.5g} {ratio:14.4g}")
    say(
        f"    overall ||e||_rms / ||b_true||_rms (all {len(labels)} coordinates): "
        f"{stats['overall']:.4g}; velocity block: {stats['velocity']:.4g}"
    )
    median, p90, p99 = stats["quantiles"]
    say(
        f"    per-row ||e|| / ||b_true||: median {median:.4g}, p90 {p90:.4g}, p99 {p99:.4g}; "
        f"share > 0.1: {stats['share_above_10pct']:.4f}, share > 0.5: "
        f"{stats['share_above_50pct']:.4f} (rows with ||b_true|| = 0: {stats['zero_true_rows']})"
    )
    say(
        f"    max ||e|| over rows: {stats['max_error']:.6g}, at a row with ||b_true|| = "
        f"{stats['true_at_worst']:.6g}"
    )
    if states is not None and torques is not None and not brief:
        worst = stats["worst"]
        say(f"      that row: state {_array(states[worst], 4)}, torque {_array(torques[worst], 4)}")
    if not brief:
        say(
            f"    position block max |e| (both plants give P v there; expected 0): "
            f"{stats['position_error_max']:.3g}"
        )


def action_effects(
    plant,
    oracle,
    states: torch.Tensor,
    low: np.ndarray,
    high: np.ndarray,
    dtype: torch.dtype,
) -> List[dict]:
    """Per actuator j: d = b(s, u_j = high_j) - b(s, u_j = low_j), the others at the box
    midpoint, for the learned and the true plant, [N,D] each (float64 numpy)."""
    rows = int(states.shape[0])
    middle = 0.5 * (low + high)
    effects = []
    for actuator in range(low.size):
        upper = middle.copy()
        upper[actuator] = high[actuator]
        lower = middle.copy()
        lower[actuator] = low[actuator]
        upper_t = constant_torques(upper, rows, dtype)
        lower_t = constant_torques(lower, rows, dtype)
        learned = chunked_drift(plant, states, upper_t) - chunked_drift(plant, states, lower_t)
        true = chunked_drift(oracle, states, upper_t) - chunked_drift(oracle, states, lower_t)
        effects.append(
            {
                "actuator": actuator,
                "span": float(high[actuator] - low[actuator]),
                "learned": learned.double().numpy(),
                "true": true.double().numpy(),
            }
        )
    return effects


def effect_statistics(effects: List[dict], position_dim: int, labels: Sequence[str]) -> dict:
    """Print and return the action-effect comparison on the velocity block."""
    error_square = 0.0
    true_square = 0.0
    cosines = []
    zero_learned = 0
    velocity_labels = list(labels[position_dim:])
    for effect in effects:
        learned, true = effect["learned"], effect["true"]
        scale = max(1.0, float(np.abs(true).max()), float(np.abs(learned).max()))
        position_block = max(
            float(np.abs(learned[:, :position_dim]).max()) if position_dim else 0.0,
            float(np.abs(true[:, :position_dim]).max()) if position_dim else 0.0,
        )
        check(
            position_block <= 1e-6 * scale,
            f"the torque moved the drift's position block by {position_block:.3g}: the "
            "drift convention is not (P v, accelerations)",
        )
        velocity_learned = learned[:, position_dim:]
        velocity_true = true[:, position_dim:]
        difference = velocity_learned - velocity_true
        rms_true = _rms(velocity_true, axis=0)
        rms_error = _rms(difference, axis=0)
        ratio = _ratio(rms_error, rms_true)
        overall = _ratio(
            _rms(np.linalg.norm(difference, axis=1)), _rms(np.linalg.norm(velocity_true, axis=1))
        )
        norm_learned = np.linalg.norm(velocity_learned, axis=1)
        norm_true = np.linalg.norm(velocity_true, axis=1)
        valid = (norm_learned > 0.0) & (norm_true > 0.0)
        cosine = np.zeros(norm_learned.size)
        cosine[valid] = np.sum(velocity_learned[valid] * velocity_true[valid], axis=1) / (
            norm_learned[valid] * norm_true[valid]
        )
        cosines.append(cosine)
        zero_learned += int(np.sum(norm_learned <= 0.0))
        error_square += float(np.sum(difference**2))
        true_square += float(np.sum(velocity_true**2))
        span = effect["span"]
        say(
            f"  action effect of actuator {effect['actuator']} (torque {span:.4g} wide, "
            f"low to high): {velocity_true.shape[0]} rows"
        )
        say(f"    {'coord':>6} {'RMS d_true':>13} {'RMS error':>13} {'error / true':>13}")
        for label, t, e, r in zip(velocity_labels, rms_true, rms_error, ratio):
            say(f"    {label:>6} {t:13.5g} {e:13.5g} {r:13.4g}")
        say(
            f"    relative RMS {overall:.4g}; mean cosine(d_learned, d_true) "
            f"{float(np.mean(cosine)):.5f} (min {float(np.min(cosine)):.4f})"
        )
        say(
            f"    row mean of d per unit torque (the mean column of M^-1 G): true "
            f"{_array(velocity_true.mean(axis=0) / span)}, learned "
            f"{_array(velocity_learned.mean(axis=0) / span)}"
        )
    all_cosines = np.concatenate(cosines) if cosines else np.zeros(0)
    result = {
        "overall": _ratio(math.sqrt(error_square), math.sqrt(true_square)),
        "mean_cosine": float(np.mean(all_cosines)) if all_cosines.size else math.nan,
        "zero_learned": zero_learned,
    }
    say(
        f"  action effect over all actuators: relative RMS {result['overall']:.4g}, mean "
        f"cosine {result['mean_cosine']:.5f}, rows where the learned effect is exactly 0: "
        f"{zero_learned}"
    )
    return result


def gauge_fit(env, plant, states: np.ndarray, dtype: torch.dtype) -> Optional[dict]:
    """The scalar c minimizing sum ||M_learned(y) - c M_true(q)||_F^2 over states.

    M_true is MuJoCo's joint-space inertia (env._mass_matrix at qpos = the state's
    physical positions), M_learned the plant's mass_matrix(y). Skipped when MuJoCo's qpos
    is not the state's physical positions.
    """
    physical = int(env.physical_position_dim)
    if int(env.model.nq) != physical:
        say(f"  gauge fit skipped: MuJoCo nq {int(env.model.nq)} != physical positions {physical}")
        return None
    rows = states[:GAUGE_ROWS]
    if rows.shape[0] == 0:
        return None
    n = int(env.velocity_dim)
    position_dim = int(env.position_dim)
    true = np.stack(
        [np.asarray(env._mass_matrix(row[:physical]), dtype=np.float64)[:n, :n] for row in rows]
    )
    learned = (
        plant.mass_matrix(torch.as_tensor(rows[:, :position_dim], dtype=dtype)).double().numpy()
    )
    check(learned.shape == true.shape == (rows.shape[0], n, n), "mass matrices must be [B,n,n]")
    scale = float(np.sum(learned * true) / np.sum(true * true))
    per_row = np.sum(learned * true, axis=(1, 2)) / np.sum(true * true, axis=(1, 2))
    residual = float(
        np.sqrt(np.sum((learned - scale * true) ** 2)) / np.sqrt(np.sum((scale * true) ** 2))
    )
    reference_true = np.asarray(env._mass_matrix(np.zeros(int(env.model.nq))), dtype=np.float64)[:n, :n]
    reference_learned = plant.mass_matrix(torch.zeros(1, position_dim, dtype=dtype))[0].double().numpy()
    return {
        "rows": int(rows.shape[0]),
        "scale": scale,
        "per_row": _quantiles(per_row, (0.0, 50.0, 100.0)),
        "residual": residual,
        "reference_true": reference_true,
        "reference_learned": reference_learned,
    }


def convention_checks(
    env,
    testbed: Mapping[str, object],
    decision_mode: str,
    seed: int,
    states: np.ndarray,
    decisions: np.ndarray,
    durations: np.ndarray,
    next_states: np.ndarray,
) -> None:
    """Printed checks of the stored (s, decision, duration, s') conventions, float64."""
    spec = env.state_spec
    controller = env.controller
    rows = int(states.shape[0])
    s = torch.as_tensor(states, dtype=torch.float64)
    kappa = torch.as_tensor(decisions, dtype=torch.float64)
    delta = torch.as_tensor(durations, dtype=torch.float64).reshape(-1, 1)
    observed = torch.as_tensor(next_states, dtype=torch.float64)
    motion = float(wrapped_endpoint_residual(observed, s, spec).abs().max())
    oracle64 = make_oracle_plant(env).double()
    physics = str(env.config.get("integrator", ""))
    integrator = "rk4" if physics.upper() == "RK4" else "semi_implicit"

    # 1. The oracle integrated at the physics step over the stored interval.
    moments = integrate_endpoint_moments(
        oracle64,
        controller,
        s,
        kappa,
        delta,
        maximum_step=float(env.physics_dt),
        physics_dt=float(env.physics_dt),
        controller_refresh_steps=int(controller.refresh_steps),
        covariance_mode="leading_order",
        create_graph=False,
        integrator=integrator,
    )
    gap = float(wrapped_endpoint_residual(observed, moments["means"], spec).abs().max())
    say(
        f"  convention check on {rows} unflagged rows (float64): the oracle integrated over "
        f"the stored duration ({integrator} at physics_dt {env.physics_dt}, physics "
        f"integrator {physics!r}) lands {gap:.3g} (max abs, wrapped) from the stored next "
        f"state; the rows' own motion |s' - s| is up to {motion:.3g}"
    )
    if integrator == "rk4" and gap > 1e-6 * max(1.0, motion):
        say("  WARNING: CONVENTION CHECK FAILED (oracle path vs stored next state); read the numbers with care.")

    # 2. A MuJoCo replay of the stored interval from the stored state.
    replay_env = make_env(testbed, decision_mode=decision_mode, seed=seed)
    worst_state = 0.0
    worst_duration = 0.0
    try:
        for row in range(rows):
            replay_env.reset(options={"state": states[row]})
            transition = replay_env.step_dt(decisions[row], float(durations[row]))
            replayed = torch.as_tensor(
                np.asarray(transition["next_state"], dtype=np.float64)
            ).reshape(1, -1)
            worst_state = max(
                worst_state,
                float(wrapped_endpoint_residual(observed[row : row + 1], replayed, spec).abs().max()),
            )
            worst_duration = max(
                worst_duration, abs(float(transition["duration"]) - float(durations[row]))
            )
    finally:
        replay_env.close()
    say(
        f"  convention check: MuJoCo replay (env.step_dt from the stored state, stored "
        f"decision and duration) vs the stored next state: max abs {worst_state:.3g}; "
        f"duration mismatch {worst_duration:.3g} s"
    )
    if worst_state > 1e-6 * max(1.0, motion) or worst_duration > 1e-9:
        say("  WARNING: CONVENTION CHECK FAILED (MuJoCo replay vs stored next state).")

    # 3. The oracle drift against MuJoCo's forward dynamics and the numpy oracle.
    torques = controller.action(s, controller.resolve_decision(s, kappa))
    torch_oracle = oracle64.drift(s, torques).numpy()
    mujoco_rows = np.stack(
        [np.asarray(env.mujoco_drift(states[row], torques[row].numpy()), dtype=np.float64) for row in range(rows)]
    )
    numpy_rows = np.stack(
        [np.asarray(env.oracle_drift(states[row], torques[row].numpy()), dtype=np.float64) for row in range(rows)]
    )
    scale = max(1.0, float(np.abs(mujoco_rows).max()))
    to_mujoco = float(np.abs(torch_oracle - mujoco_rows).max())
    to_numpy = float(np.abs(torch_oracle - numpy_rows).max())
    say(
        f"  convention check: oracle drift (torch, float64) vs MuJoCo forward dynamics "
        f"(env.mujoco_drift): max abs {to_mujoco:.3g}; vs env.oracle_drift (numpy): "
        f"{to_numpy:.3g}; max |b| {scale:.4g}"
    )
    if to_mujoco > 1e-6 * scale:
        say("  WARNING: CONVENTION CHECK FAILED (oracle drift vs MuJoCo forward dynamics).")


# ----------------------------------------------------------------------------- one run


def analyze_run(run_arg: str, args: argparse.Namespace) -> Optional[dict]:
    run_dir = Path(run_arg).expanduser()
    if run_dir.is_file():
        run_dir = run_dir.parent
    run_dir = run_dir.absolute()
    say("=" * 100)
    say(f"RUN {run_dir}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a folder.")
    seed_folder = run_dir.name.startswith("seed_")
    arm = run_dir.parent.name if seed_folder else "unknown"
    folder_task = run_dir.parent.parent.name if seed_folder else "unknown"

    bundle = load_run_states(run_dir, args.state_file)
    state = bundle["plant_state"]
    networks = state.get("networks") or {}
    weights = networks.get(args.plant_key)
    if weights is None:
        raise ValueError(
            f"state['networks'] of {bundle['plant_file'].name} has no {args.plant_key!r} "
            f"(networks: {sorted(networks)}): not a learned-plant arm."
        )
    dtype = weights["force_matrix"].dtype
    check(dtype in (torch.float32, torch.float64), f"unexpected plant dtype {dtype}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            row = _analyze(run_dir, arm, folder_task, bundle, weights, dtype, args)
    finally:
        torch.set_default_dtype(previous)
    say(f"  run analyzed in {time.perf_counter() - started:.1f} s")
    return row


def _analyze(
    run_dir: Path,
    arm: str,
    folder_task: str,
    bundle: dict,
    weights: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Optional[dict]:
    state = bundle["plant_state"]
    config = state["config"]
    testbed = config["testbed"]
    task = str(testbed["name"])
    training = config["training"]
    seed = int(training["seed"])
    network = config["network"]
    decision_mode = decision_mode_setting(testbed["controller"])
    handling = contact_handling_setting(config)
    say(
        f"  task {task} (folder {folder_task}), arm {arm}, training seed {seed}, algorithm "
        f"{'implicit' if 'solver' in config else 'explicit'}, plant_source "
        f"{config.get('plant_source')!r}"
    )
    say(
        f"  plant: state['networks'][{args.plant_key!r}] of {bundle['plant_file'].name} at "
        f"{bundle['plant_interactions']} interactions; config: state['config'] of the same file"
    )
    say(
        f"  decision_mode {decision_mode}, contact_handling {handling}, input_matrix_start "
        f"{input_matrix_start_setting(network)!r}, limit_port {bool(network.get('limit_port', False))}, "
        f"diffusion {network.get('diffusion', 'learned')!r}, trainable_plant_parts "
        f"{config.get('trainable_plant_parts')}, model_integrator "
        f"{model_integrator_setting(config)!r}, batch_size {training.get('batch_size')}, "
        f"replay_capacity {training.get('replay_capacity')}; computing in {dtype}"
    )
    cross_check_protocol(run_dir, config)

    env = make_env(testbed, decision_mode=decision_mode, seed=seed)
    try:
        return _analyze_with_env(
            env, run_dir, arm, task, seed, testbed, decision_mode, bundle, weights, dtype, args
        )
    finally:
        env.close()


def _analyze_with_env(
    env,
    run_dir: Path,
    arm: str,
    task: str,
    seed: int,
    testbed: Mapping[str, object],
    decision_mode: str,
    bundle: dict,
    weights: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Optional[dict]:
    state = bundle["plant_state"]
    config = state["config"]
    training = config["training"]
    network = config["network"]
    spec = env.state_spec
    controller = env.controller
    position_dim = int(env.position_dim)
    velocity_dim = int(env.velocity_dim)
    state_dim = position_dim + velocity_dim
    labels = [f"y{i}" for i in range(position_dim)] + [f"v{i}" for i in range(velocity_dim)]

    # --- the plants (algorithms/ph_base.PHAlgorithm._build_learned_plant; tasks.make_oracle_plant)
    plant = PortHamiltonianPlant(spec, network, learn_diffusion=diffusion_is_learned(network))
    plant.load_state_dict(weights)
    plant.eval()
    plant.requires_grad_(False)
    oracle = make_oracle_plant(env)
    check(int(plant.state_dim) == state_dim and int(oracle.state_dim) == state_dim, "state dimensions differ")
    action_dim = int(plant.action_dim)
    check(int(oracle.action_dim) == action_dim, "action dimensions differ")

    # --- G
    torque_low = np.asarray(env.config["controller"]["torque_low"], dtype=np.float64).reshape(-1)
    torque_high = np.asarray(env.config["controller"]["torque_high"], dtype=np.float64).reshape(-1)
    check(torque_low.size == action_dim and torque_high.size == action_dim, "torque box must have m entries")
    half_width = 0.5 * (torque_high - torque_low)
    learned_g = plant.input_matrix().detach().double().numpy()
    true_g = np.asarray(env.oracle_input_matrix(), dtype=np.float64)
    true_g_as_run = torch.as_tensor(env.oracle_input_matrix()).to(dtype)
    check(
        learned_g.shape == true_g.shape == (velocity_dim, action_dim),
        f"G shapes: learned {learned_g.shape}, true {true_g.shape}, expected {(velocity_dim, action_dim)}",
    )
    check(
        torch.equal(oracle.input_matrix().detach(), true_g_as_run),
        "make_oracle_plant(env).input_matrix() differs from env.oracle_input_matrix()",
    )
    say(
        f"  input matrix G [{velocity_dim}x{action_dim}]: generalized force per unit of the drift's "
        f"torque input = controller.action's motor torque in the box {torque_low.tolist()} .. "
        f"{torque_high.tolist()} (cart-poles: newtons on the slider, gear 1), not the actor's "
        f"normalized [-1,1] action (G per unit normalized action = column j times h_j = "
        f"{half_width.tolist()})"
    )
    say(f"    true G (as the run's '[input-start] ... true' line prints it): {true_g_as_run.tolist()}")
    say(f"    learned G ({args.plant_key}.force_matrix): {plant.force_matrix.detach().tolist()}")
    say(
        f"    learned - true: {(learned_g - true_g).tolist()}; max |learned - true| "
        f"{float(np.abs(learned_g - true_g).max()):.4g}; learned G all zero: "
        f"{bool(np.all(learned_g == 0.0))}"
    )
    say(
        f"    per unit normalized action: true {(true_g * half_width).tolist()}, learned "
        f"{(learned_g * half_width).tolist()}"
    )
    say(
        "    note: (M, U, D, G) -> c (M, U, D, G) leaves the drift unchanged, so G is fixed "
        "only up to a scalar c: see the gauge fit and the action effect M^-1 G below"
    )
    if getattr(plant, "limit_port", False):
        stiffness = plant.limit_stiffness.exp().double().numpy()
        damping = plant.limit_damping.exp().double().numpy()
        width = (LIMIT_WIDTH_FLOOR + torch.nn.functional.softplus(plant.limit_width)).double().numpy()
        say(
            f"    limit port (learned): stiffness {_array(stiffness)}, damping {_array(damping)}, "
            f"width {_array(width)}; start (state spec limit_port_init): {spec.get('limit_port_init')}"
        )

    # --- the replay
    replay = bundle["replay"]
    if replay is None:
        say(
            "  NO REPLAY in checkpoint.pt or resume.pt (protocol checkpoint_replay false and "
            "resume.pt deleted at completion, or an older file). Without the stored states no "
            "accuracy is computed: this run ends here and writes no CSV row."
        )
        return None
    say(
        f"  replay: {bundle['replay_file'].name} at {bundle['replay_interactions']} interactions; "
        f"plant: {bundle['plant_file'].name} at {bundle['plant_interactions']} interactions"
    )
    if bundle["replay_interactions"] != bundle["plant_interactions"]:
        say("  WARNING: the plant and the replay come from different interaction counts.")
    rows = replay_rows(replay, args.partition)
    if rows is None:
        say(f"  the {args.partition} partition holds no rows: no CSV row.")
        return None
    total = int(rows["states"].shape[0])
    count = min(int(args.rows), total)
    sampler = np.random.default_rng(int(args.seed))
    chosen = np.sort(sampler.choice(total, size=count, replace=False))
    states_np = rows["states"][chosen]
    decisions_np = rows["decisions"][chosen]
    durations_np = rows["durations"][chosen]
    next_np = rows["next_states"][chosen]
    near = rows["near_limit"][chosen]
    touched = rows["limit_touched"][chosen]
    ages = rows["age"][chosen]
    episodes = rows["episode_ids"][chosen]
    flagged = near | touched
    check(states_np.shape == (count, state_dim), f"stored states are {states_np.shape}, not [B,{state_dim}]")
    check(
        decisions_np.shape == (count, int(controller.decision_dim)),
        f"stored decisions are {decisions_np.shape}, not [B,{controller.decision_dim}]",
    )

    exclusion = model_contact_exclusion(config, env)
    active = exclusion is not None and exclusion.active
    say(
        "  contact rule: "
        + ("none (no limited coordinates)" if exclusion is None else json.dumps(exclusion.record()))
        + f"; exclusion active (flagged rows left out of the fit and diagnostics): {active}"
    )
    share = float(flagged.mean())
    say(
        f"  sampled {count} of {total} {args.partition} rows (uniform without replacement, "
        f"numpy default_rng({args.seed})): flagged {int(flagged.sum())} (share {share:.4f}) = "
        f"near_limit {int(near.sum())} | limit_touched {int(touched.sum())} (both "
        f"{int((near & touched).sum())})"
    )
    if exclusion is not None:
        recomputed = start_state_flags(
            exclusion,
            torch.as_tensor(states_np, dtype=torch.float64),
            torch.as_tensor(durations_np, dtype=torch.float64),
        ).numpy()
        say(
            f"  start-state rule recomputed with this code: {int(recomputed.sum())} near_limit; "
            f"rows disagreeing with the stored column: {int(np.sum(recomputed != near))}"
        )

    # --- torques (PHAlgorithm.record_plant_diagnostics)
    states_t = torch.as_tensor(states_np, dtype=dtype)
    decisions_t = torch.as_tensor(decisions_np, dtype=dtype)
    torques_t = controller.action(states_t, controller.resolve_decision(states_t, decisions_t))
    check(tuple(torques_t.shape) == (count, action_dim), f"torques are {tuple(torques_t.shape)}, not [B,{action_dim}]")
    low_t = torch.as_tensor(torque_low, dtype=dtype)
    high_t = torch.as_tensor(torque_high, dtype=dtype)
    check(bool(((torques_t >= low_t) & (torques_t <= high_t)).all()), "torques outside the motor box")
    torques_np = torques_t.double().numpy()
    say(
        f"  torques (controller.action of the stored decisions): min {_array(torques_np.min(axis=0), 4)}, "
        f"max {_array(torques_np.max(axis=0), 4)}, share at a box end "
        f"{float(np.mean((torques_np <= torque_low) | (torques_np >= torque_high))):.4f}"
    )

    # --- the drifts at the stored torques
    learned = chunked_drift(plant, states_t, torques_t)
    true = chunked_drift(oracle, states_t, torques_t)
    error_np = (learned - true).double().numpy()
    true_np = true.double().numpy()

    # --- self-check: the oracle against a second oracle, by the diagnostics' own function.
    # Identical inputs and code: 0 (the tolerance only absorbs a BLAS that rounds by
    # buffer alignment; any convention error would be O(|b|)).
    probe = slice(0, min(count, SELF_CHECK_ROWS))
    probe_rows = int(states_np[probe].shape[0])
    self_record = measure_oracle_errors(
        oracle,
        make_oracle_plant(env),
        states_np[probe],
        torques_t[probe].numpy(),
        np.zeros((probe_rows, state_dim)),
        interactions=int(bundle["plant_interactions"]),
    )
    probe_true = true_np[probe]
    probe_scale = max(1.0, float(np.nanmax(np.abs(probe_true)))) if probe_true.size else 1.0
    tolerance = 64.0 * float(torch.finfo(dtype).eps) * probe_scale
    check(
        float(self_record["drift_error_max"]) <= tolerance,
        f"measure_oracle_errors(oracle, second oracle) = {self_record['drift_error_max']}, not 0",
    )
    say(
        f"  self-check: measure_oracle_errors(oracle, second oracle) on {probe_rows} rows = "
        f"{float(self_record['drift_error_max']):.3g} (expected 0): passed"
    )

    finite = np.isfinite(error_np).all(axis=1) & np.isfinite(true_np).all(axis=1)
    if not finite.all():
        say(f"  WARNING: {int((~finite).sum())} rows give non-finite drifts; left out of every statistic.")
    unflagged_idx = np.flatnonzero(~flagged & finite)
    flagged_idx = np.flatnonzero(flagged & finite)

    if unflagged_idx.size:
        sub = unflagged_idx[:SELF_CHECK_ROWS]
        record = measure_oracle_errors(
            plant,
            oracle,
            states_np[sub],
            torques_t[torch.as_tensor(sub, dtype=torch.long)].numpy(),
            np.zeros((sub.size, state_dim)),
            interactions=int(bundle["plant_interactions"]),
        )
        mine = float(np.linalg.norm(error_np[sub], axis=1).max())
        # Rounding of the plant's dtype on the drift's own scale; a convention error is O(1).
        scale = max(1.0, mine, float(np.abs(true_np[sub]).max()), float(np.abs(error_np[sub]).max()))
        check(
            abs(float(record["drift_error_max"]) - mine) <= 1e-4 * scale,
            f"measure_oracle_errors gives drift_error_max {record['drift_error_max']} on "
            f"{sub.size} unflagged rows, this script {mine}",
        )
        say(
            f"  self-check: evaluations/ph_sac/diagnostics.measure_oracle_errors on {sub.size} "
            f"unflagged rows: drift_error_max {float(record['drift_error_max']):.6g}, this "
            f"script {mine:.6g}: passed"
        )

    # --- conventions of the stored transitions (printed)
    if args.check_rows > 0 and unflagged_idx.size:
        picked = unflagged_idx[: args.check_rows]
        convention_checks(
            env,
            testbed,
            decision_mode,
            seed,
            states_np[picked],
            decisions_np[picked],
            durations_np[picked],
            next_np[picked],
        )

    # --- drift accuracy at the stored torques
    say("  --- drift b(s, u) at the stored states and torques")
    stats_unflagged = error_statistics(error_np[unflagged_idx], true_np[unflagged_idx], position_dim)
    print_statistics(
        "UNFLAGGED rows",
        stats_unflagged,
        labels,
        states=states_np[unflagged_idx],
        torques=torques_np[unflagged_idx],
    )
    stats_flagged = error_statistics(error_np[flagged_idx], true_np[flagged_idx], position_dim)
    print_statistics("FLAGGED rows (brief)", stats_flagged, labels, brief=True)

    effect = {"overall": None, "mean_cosine": None}
    stats_free: dict = {"rows": 0}
    gauge = None
    if unflagged_idx.size:
        unflagged_t = torch.as_tensor(unflagged_idx, dtype=torch.long)
        states_unflagged = states_t.index_select(0, unflagged_t)

        # --- action effect (gauge-free M^-1 G)
        say("  --- action effect d = b(s, u_hi) - b(s, u_lo) on the unflagged rows (velocity block)")
        effects = action_effects(plant, oracle, states_unflagged, torque_low, torque_high, dtype)
        effect = effect_statistics(effects, position_dim, labels)

        # --- action-free part
        say("  --- action-free drift b(s, 0) on the unflagged rows")
        zero = torch.zeros(int(unflagged_idx.size), action_dim, dtype=dtype)
        free_learned = chunked_drift(plant, states_unflagged, zero)
        free_true = chunked_drift(oracle, states_unflagged, zero)
        stats_free = error_statistics(
            (free_learned - free_true).double().numpy(), free_true.double().numpy(), position_dim
        )
        print_statistics("b(s, 0)", stats_free, labels)

        # --- gauge of G
        say("  --- mass-matrix gauge (scalar c with M_learned ~ c M_true)")
        try:
            gauge = gauge_fit(env, plant, states_np[unflagged_idx], dtype)
        except Exception as error:  # a side measurement: never stop the run for it
            say(f"  gauge fit failed ({type(error).__name__}: {error}); skipped")
            gauge = None
        if gauge is not None:
            low, median, high = gauge["per_row"]
            say(
                f"    c = {gauge['scale']:.6g} over {gauge['rows']} unflagged states (per-state c: "
                f"min {low:.4g}, median {median:.4g}, max {high:.4g}); relative Frobenius "
                f"residual ||M_learned - c M_true|| / ||c M_true|| = {gauge['residual']:.4g}"
            )
            say(f"    learned G / c: {(learned_g / gauge['scale']).tolist()}; true G: {true_g.tolist()}")
            say(f"    at y = 0: M_true\n{_array(gauge['reference_true'], 5)}")
            say(f"    at y = 0: M_learned / c\n{_array(gauge['reference_learned'] / gauge['scale'], 5)}")

    # --- oldest vs newest rows
    age_results: Dict[str, Optional[float]] = {"oldest": None, "newest": None}
    say(f"  --- age split (replay ring order: {rows['order']})")
    if rows["order"] == "broken":
        say("    episode ids are not nondecreasing in ring order: the order is not recoverable; skipped")
    elif unflagged_idx.size:
        groups = (
            ("oldest", unflagged_idx[ages[unflagged_idx] < AGE_FRACTION]),
            ("newest", unflagged_idx[ages[unflagged_idx] >= 1.0 - AGE_FRACTION]),
        )
        for name, members in groups:
            group = error_statistics(error_np[members], true_np[members], position_dim)
            if not group["rows"]:
                say(f"    {name} {int(AGE_FRACTION * 100)}%: no unflagged rows sampled")
                continue
            age_results[name] = group["overall"]
            say(
                f"    {name} {int(AGE_FRACTION * 100)}% of the replay: {group['rows']} unflagged rows, "
                f"episode ids {int(episodes[members].min())}..{int(episodes[members].max())}: "
                f"overall {group['overall']:.4g}, velocity block {group['velocity']:.4g}, "
                f"max ||e|| {group['max_error']:.5g}"
            )

    # --- the logged diagnostic
    say("  --- the logged diagnostic (oracle.csv)")
    logged = last_oracle_row(run_dir)
    if logged is None:
        say("    oracle.csv: absent or empty")
    else:
        say(
            "    oracle.csv last row: "
            + ", ".join(
                f"{key} {logged.get(key, 'absent')}"
                for key in (
                    "interactions",
                    "sampled_states",
                    "drift_error_max",
                    "value_direction_error_mean",
                    "contact_flagged_share",
                )
            )
        )
    batch = int(training.get("batch_size", 256))
    norms = np.linalg.norm(np.where(finite[:, None], error_np, 0.0), axis=1)
    booter = np.random.default_rng([int(args.seed), 20_261_124])
    maxima: List[float] = []
    kept: List[int] = []
    for _ in range(BOOTSTRAP_BATCHES):
        pick = booter.integers(0, count, size=batch)
        if active:
            pick = pick[~flagged[pick]]
        pick = pick[finite[pick]]
        if pick.size:
            maxima.append(float(norms[pick].max()))
            kept.append(int(pick.size))
    emulated = _quantiles(np.asarray(maxima), (10.0, 50.0, 90.0))
    say(
        f"    this script: max ||e|| over all {stats_unflagged['rows']} sampled unflagged rows "
        f"{stats_unflagged.get('max_error', math.nan):.6g}"
    )
    say(
        f"    emulated diagnostic batches ({BOOTSTRAP_BATCHES} draws of batch_size {batch} with "
        f"replacement from the sampled rows{', flagged dropped' if active else ''}; mean rows "
        f"kept {float(np.mean(kept)) if kept else math.nan:.1f}): max ||e|| p10 {emulated[0]:.5g}, "
        f"p50 {emulated[1]:.5g}, p90 {emulated[2]:.5g}; flagged share of the sampled rows {share:.4f}"
    )

    per_coordinate = {}
    if stats_unflagged["rows"]:
        per_coordinate = {
            label: _json_float(value) for label, value in zip(labels, stats_unflagged["ratio"])
        }
    quantiles = stats_unflagged.get("quantiles", [None, None, None])
    row = {
        "run_dir": str(run_dir),
        "task": task,
        "arm": arm,
        "seed": seed,
        "interactions": int(bundle["plant_interactions"]),
        "n_unflagged": int(unflagged_idx.size),
        "n_flagged": int(flagged_idx.size),
        "overall_rel_rms_unflagged": stats_unflagged.get("overall"),
        "action_effect_rel_rms": effect["overall"],
        "action_free_rel_rms": stats_free.get("overall"),
        "max_error_norm_unflagged": stats_unflagged.get("max_error"),
        "learned_G": json.dumps(learned_g.tolist()),
        "true_G": json.dumps(true_g.tolist()),
        "per_coordinate_rel_rms_unflagged": json.dumps(per_coordinate),
        "velocity_rel_rms_unflagged": stats_unflagged.get("velocity"),
        "per_row_rel_median_unflagged": quantiles[0],
        "per_row_rel_p90_unflagged": quantiles[1],
        "action_effect_mean_cosine": effect["mean_cosine"],
        "flagged_overall_rel_rms": stats_flagged.get("overall"),
        "oldest20_overall_rel_rms": age_results["oldest"],
        "newest20_overall_rel_rms": age_results["newest"],
        "gauge_scale_c": None if gauge is None else gauge["scale"],
        "gauge_mass_residual": None if gauge is None else gauge["residual"],
        "learned_G_over_c": "" if gauge is None else json.dumps((learned_g / gauge["scale"]).tolist()),
        "emulated_logged_max_p50": emulated[1],
        "logged_interactions": None if logged is None else logged.get("interactions"),
        "logged_drift_error_max": None if logged is None else _float_or_none(logged.get("drift_error_max")),
        "logged_contact_flagged_share": None
        if logged is None
        else _float_or_none(logged.get("contact_flagged_share")),
        "sampled_flagged_share": share,
        "plant_file": bundle["plant_file"].name,
        "plant_file_interactions": int(bundle["plant_interactions"]),
        "replay_file": bundle["replay_file"].name,
        "replay_file_interactions": bundle["replay_interactions"],
        "partition": args.partition,
        "rows_sampled": count,
        "dtype": str(dtype).replace("torch.", ""),
    }
    say("  --- summary")
    for key in CSV_COLUMNS:
        say(f"    {key}: {_cell(row.get(key))}")
    return row


# ------------------------------------------------------------------------------- output


def append_csv(path: Path, row: Mapping[str, object]) -> None:
    """Append one row; write the header first when the file is absent or empty. A file
    with another header is left untouched (the row is on stdout)."""
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header != list(CSV_COLUMNS):
            say(
                f"  ERROR: {path} has another header; this run's row is NOT written (it is "
                "printed above). Use another --out."
            )
            return
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([_cell(row.get(name)) for name in CSV_COLUMNS])
        handle.flush()
    say(f"  appended one row to {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True, help="seed folders holding checkpoint.pt")
    parser.add_argument("--out", type=Path, required=True, help="CSV to append to (its folder must exist)")
    parser.add_argument("--rows", type=int, default=20_000, help="replay rows to sample (default 20000)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the private row sampler")
    parser.add_argument("--partition", choices=("train", "gate", "all"), default="train",
                        help="replay partition (default train, the one the plant is fitted on)")
    parser.add_argument("--state-file", choices=("auto", "checkpoint", "resume"), default="auto",
                        help="file of the plant (auto: the one with more interactions)")
    parser.add_argument("--plant-key", default="online_plant",
                        help="state['networks'] entry of the plant (default online_plant, the diagnostics')")
    parser.add_argument("--check-rows", type=int, default=16,
                        help="unflagged rows of the convention checks (0: skip)")
    parser.add_argument("--threads", type=int, default=None, help="torch threads (default: torch's)")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive.")
    if args.check_rows < 0:
        parser.error("--check-rows must be nonnegative.")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = args.out.expanduser().absolute()
    if not out.parent.is_dir():
        say(f"--out folder {out.parent} does not exist (this script creates no folders).")
        return 2
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    say(
        f"ph_plant_checkpoint_accuracy: torch {torch.__version__}, numpy {np.__version__}, "
        f"threads {torch.get_num_threads()}, rows {args.rows}, seed {args.seed}, partition "
        f"{args.partition}, state file {args.state_file}, plant key {args.plant_key}, out {out}"
    )
    failures = 0
    for run in args.runs:
        try:
            row = analyze_run(run, args)
            if row is not None:
                append_csv(out, row)
        except Exception:
            failures += 1
            say(f"FAILED {run}:")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    say("=" * 100)
    say(f"done: {len(args.runs) - failures} of {len(args.runs)} runs without error, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
