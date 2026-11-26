#!/usr/bin/env python
"""Does the learned plant's limit port learn MuJoCo's soft joint-limit forces in the real fit?

Plant-only experiment (2026-11-10): no actor, critic, value or reward head. Transitions come
from the real environment under the uniform-box warm-up decisions the agents use
(PHAlgorithm._uniform_decision, which press on the limits often), and a drift-only
PortHamiltonianPlant is fitted to them with the table row's plant settings: the row's
network (plant_hidden_sizes, damping_rank, velocity_feature_scale, diffusion "none",
damping_velocity on the swimmer), Adam at training.learning_rate, gradient clip
training.gradient_clip, training.batch_size, model_step, model_integrator (rk4 on the RK4
rows; semi_implicit at the 2 ms physics step on the swimmers, whose physics is dm_control's
2 ms Euler since 2026-11-10) with the controller's continuous closed loop, and the loss the
agents call:
integrate_endpoint_moments(..., create_graph=True) then plant_likelihood_loss, which scores a
drift-only plant by plant_drift_loss. The plant is built and initialized as
PHAlgorithm._build_learned_plant does (true input matrix). The agents' warm fit, online
steps and plant-step schedule are not used: plain gradient steps on minibatches drawn with
replacement from a fixed training set.

Variants
  A  no port (network limit_port False)
  B  the port from the compiled start (state spec limit_port_init: the default)
  C  the port from a deliberately wrong start: the compiled start with stiffness and damping
     multiplied by WRONG_START_FACTOR (10), width unchanged (state spec limit_port_init
     entries scaled; since 2026-11-10 evening, the generic start before, now G)
  G  the port from the plant's generic start (spec without limit_port_init: stiffness 1e-2,
     damping 1e-3, width 0.1); variant C of job 1134981; not in the default runs
  D  the oracle plant, no training (evaluated in float64, "D64", and in float32, "D32", the
     learned plants' dtype)
A, B, C and G of one seed share the network initialization (the port's parameters are
created last, with no draw of the RNG) and the minibatch sequence, so they differ only by
the port. Each run writes its settings and port start to DIR/runs/<task>_<variant>_s<S>.json
and prints them in its log header.

Data per task and seed: a training set (uniform box), a held-out set from other episode
seeds (uniform box) and a held-out set with moderate gains (stiffness and damping uniform in
the lower third of their range, setpoint uniform in its box), collected through the current
environment. Every interval records whether it touches a limit: some physics-step knot (the
start, or the end of a physics step: every physics_dt, 1 ms on the RK4 rows, 2 ms on the
swimmers' Euler) with a limited coordinate strictly past its range (the environment's
Transition limit_touched, whose agreement the data's metadata records).

Metrics at each evaluation, on held-out data
  endpoint (sets "uniform", "moderate"; splits all, touch, free, and by the start-knot
  torque: hiF |a| > 0.5 limit, loF |a| < 0.1 limit, crossed with touch/free)
    end_*   ||e|| / ||s' - s|| per row (e the wrapped endpoint residual), med p90 p99;
            end_nrmse = sqrt(sum ||e||^2 / sum ||s' - s||^2) over the split
    pos_*, vel_*  the same on the limited coordinates' positions, resp. velocities
    loss    plant_drift_loss (0.5 ||e||^2 / Delta), the training loss, on the split
  drift (set "drift": the uniform held-out start states plus knots past a range from its
  touching intervals, each with the torque the law gives there)
    acc_*   ||b_hat - b|| / ||b|| on the acceleration block against the oracle, med p90 p99,
            acc_nrmse pooled; lim_* the same on the limited coordinates' accelerations;
            splits inside (no limited coordinate past its range), ramp (past, within the
            compiled port width w0), past (beyond w0), and the same by knot torque
  port parameters every --port-every steps (k, c, w per limited coordinate)
  probes (B, C, G, at the end): the final plant with c halved and doubled, k halved and
  doubled, and the port switched off

Subcommands
  collect    --task T --seed S --out DIR   writes DIR/data/<task>_s<S>.npz and .json and the
             oracle's (D64, D32) rows
  train      --task T --variant A|B|C|G --seed S --out DIR [--max-steps N] [--eval-every N]
             [--save-every N] [--deadline EPOCH]  waits for the data file, trains, appends
             rows after every evaluation (a partial run keeps its rows), saves the plant
             every --save-every steps and at the final evaluation, then probes
  probe      --task T --variant B|C|G --seed S --out DIR [--state FILE]   the probes on a
             saved plant (default: the run's latest state), for a run killed before its own
             probes; rows to DIR/rows/<task>/<variant>_s<S>_probe.tsv
  summarize  --out DIR   tables, mean +- std over seeds at the last step all runs of a task
             evaluated; also written to DIR/summary/*.tsv
  local      --out DIR [--tasks ...] [--seeds ...] [--tiny]   collect, train A B C and
             summarize in one process (a local smoke run)

--tiny shrinks the data (600 train, 300 + 300 held-out rows), the steps (40), the
evaluation cadence (20) and the save cadence (20) to test the pipeline.

Files: DIR/rows/<task>/<variant>_s<S>.tsv (columns task variant seed step wall_s probe set
split metric value), DIR/port/<task>/<variant>_s<S>.tsv (step coordinate k c w k0 c0 w0
k_start c_start w_start; k0 c0 w0 the compiled limit, *_start the run's start),
DIR/runs/<task>_<variant>_s<S>.json (the run's header), DIR/state/<task>/<variant>_s<S>/
(plant_step<N>.pt every --save-every steps, plant_final.pt at the final evaluation,
latest.json), DIR/done/<task>_<variant>_s<S>.json and _plant.pt.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import torch

    from benchmarks import ph_hyperparams as table
    from environment.ph_sac import tasks
    from environment.ph_sac.registry import testbed_spec
    from models.ph_sac.losses import plant_likelihood_loss
    from models.ph_sac.plant import LIMIT_WIDTH_FLOOR, PortHamiltonianPlant
    from models.ph_sac.rollouts import grid_refresh_steps, integrate_endpoint_moments
except ImportError:  # summarize needs numpy alone (a login node's system python)
    if len(sys.argv) < 2 or sys.argv[1] != "summarize":
        raise
    torch = None

TASKS = ("cartpole-swingup", "point_mass-easy", "reacher-easy", "swimmer-swimmer6")
VARIANTS = ("A", "B", "C")  # the default runs
LEARNED_VARIANTS = ("A", "B", "C", "G")  # every trainable variant (G: the generic start)
PORT_VARIANTS = ("B", "C", "G")
WRONG_START_FACTOR = 10.0  # variant C: the compiled stiffness and damping times this
VARIANT_NOTES = {
    "A": "no port",
    "B": "port, compiled start",
    "C": f"port, wrong start: compiled k and c x{WRONG_START_FACTOR:g}, w compiled",
    "G": "port, generic start (1e-2, 1e-3, 0.1)",
}
FULL = {"train": 60_000, "heldout": 10_000, "moderate": 10_000, "drift_knots": 10_000}
TINY = {"train": 600, "heldout": 300, "moderate": 300, "drift_knots": 300}
MODERATE_FRACTION = 1.0 / 3.0
HIGH_FORCE, LOW_FORCE = 0.5, 0.1
EVAL_CHUNK = 2048
ROW_COLUMNS = ("task", "variant", "seed", "step", "wall_s", "probe", "set", "split", "metric", "value")
PROGRESS_STEPS = (10, 100, 500, 1000)  # extra progress lines (s/step) before the first evaluation
PROBES = (("c*0.5", "c", 0.5), ("c*2", "c", 2.0), ("k*0.5", "k", 0.5), ("k*2", "k", 2.0), ("off", None, None))


# ----------------------------------------------------------------------------- setup


def row_config(task: str) -> Tuple[dict, dict]:
    """(the table row's protocol config, the testbed config its factory builds)."""
    config = copy.deepcopy(table.find_ph_entry(task, "top").config)
    testbed = testbed_spec(config["testbed"]).config_factory(**copy.deepcopy(config["testbed_settings"]))
    return config, testbed


def limit_layout(spec: Mapping[str, object]) -> Dict[str, np.ndarray]:
    """Limited position indices, ranges, their velocity indices and the compiled port start."""
    ranges = tuple(spec["limited_position_ranges"])
    projection = np.asarray(spec["observation_projection"])
    n_o = int(spec["position_dim"])
    positions = np.array([int(entry[0]) for entry in ranges])
    dofs = np.array([int(np.argmax(np.abs(projection[i]))) for i in positions])
    start = np.asarray(spec["limit_port_init"], dtype=np.float64)
    return {
        "positions": positions,
        "velocities": n_o + dofs,
        "dofs": dofs,
        "low": np.array([float(entry[1]) for entry in ranges]),
        "high": np.array([float(entry[2]) for entry in ranges]),
        "k0": start[:, 0],
        "c0": start[:, 1],
        "w0": start[:, 2],
    }


def depths(states: np.ndarray, layout: Mapping[str, np.ndarray]) -> np.ndarray:
    """Penetration past the range of every limited coordinate, [N, L] (0 inside)."""
    y = np.asarray(states)[..., layout["positions"]]
    return np.maximum(y - layout["high"], 0.0) + np.maximum(layout["low"] - y, 0.0)


def wrapped(difference: np.ndarray, periodic: Sequence[int]) -> np.ndarray:
    out = np.array(difference, dtype=np.float64, copy=True)
    if len(periodic):
        index = list(periodic)
        out[:, index] = np.mod(out[:, index] + math.pi, 2.0 * math.pi) - math.pi
    return out


def physics_settings(env, config: Mapping[str, object], testbed: Mapping[str, object]) -> Dict[str, object]:
    """The physics, the model map and the touch definition this process uses, read from the
    current code (the table row, its testbed factory and the built environment)."""
    try:
        import mujoco

        mujoco_integrator = mujoco.mjtIntegrator(int(env.model.opt.integrator)).name
    except Exception:  # a name is a convenience; the number is the record
        mujoco_integrator = str(int(env.model.opt.integrator))
    controller = env.controller
    period = float(testbed.get("reference_period", math.nan))
    model_step = float(config["model_step"])
    return {
        "testbed": str(testbed.get("name", config.get("testbed"))),
        "physics_dt": float(env.physics_dt),
        "physics_integrator_config": str(testbed.get("integrator")),
        "physics_integrator_mujoco": mujoco_integrator,
        "physics_timestep_mujoco": float(env.model.opt.timestep),
        "reference_period": period,
        "physics_steps_per_reference_period": period / float(env.physics_dt) if math.isfinite(period) else math.nan,
        "env_stage_feedback": bool(getattr(env, "stage_feedback", False)),
        "refresh_mode": str(getattr(controller, "refresh_mode", "held")),
        "refresh_steps": int(testbed["controller"]["refresh_steps"]),
        "model_step": model_step,
        "model_integrator": str(config.get("model_integrator", "rk4")),
        "model_substeps_per_reference_period": period / model_step if math.isfinite(period) else math.nan,
        "model_coarse_step": config.get("model_coarse_step"),
        "damping_velocity": bool(config["network"].get("damping_velocity", False)),
        "row_limit_port": bool(config["network"].get("limit_port", False)),
        "diffusion": str(config["network"].get("diffusion", "learned")),
        "touch_knot_spacing": float(env.physics_dt),
        "touch_definition": "an interval touches when some knot (its start, or the end of one of its physics steps, every physics_dt) has a limited coordinate strictly past its range",
    }


# ------------------------------------------------------------------------- collection


def collect_set(env, layout, *, count: int, env_seed: int, decision_seed: int, gain_fraction: float, keep_knots: bool) -> dict:
    """count transitions of whole episodes (the last cut at count) under uniform decisions.

    gain_fraction 1 is the full box (the agents' warm-up); 1/3 keeps stiffness and damping in
    the lower third. Records per row the touch flag, depths and the start-knot torque, and with
    keep_knots every knot past a range with the torque the law gives there.
    """
    rng = np.random.default_rng(decision_seed)
    controller = env.controller
    m = int(env.state_spec["action_dim"])
    low = env.action_space.low.copy()
    high = env.action_space.high.copy()
    high[m:] = low[m:] + gain_fraction * (high[m:] - low[m:])
    limit = np.maximum(np.abs(np.asarray(controller.config["torque_low"])), np.abs(np.asarray(controller.config["torque_high"])))
    knots: List[np.ndarray] = []
    original = env.observe

    def recording():
        state = original()
        knots.append(state)
        return state

    env.observe = recording
    rows = {key: [] for key in ("states", "decisions", "durations", "next_states", "torque0")}
    flags = {key: [] for key in ("touch", "env_touch", "start_past", "end_past", "knots_past", "knots", "max_depth_ratio")}
    knot_states, knot_torques = [], []
    episodes = 0
    try:
        while len(rows["states"]) < count:
            env.reset(seed=env_seed if episodes == 0 else None)
            episodes += 1
            while len(rows["states"]) < count:
                decision = rng.uniform(low, high)
                knots.clear()
                transition = env.step_dt(decision, env.sample_duration())
                path = np.array(knots)
                depth = depths(path, layout)
                past = (depth > 0.0).any(axis=-1)
                resolved = controller.resolve_decision_numpy(transition["state"], transition["decision"])
                rows["states"].append(transition["state"])
                rows["decisions"].append(transition["decision"])
                rows["durations"].append(transition["duration"])
                rows["next_states"].append(transition["next_state"])
                rows["torque0"].append(controller.action_numpy(transition["state"], resolved))
                flags["touch"].append(bool(past.any()))
                flags["env_touch"].append(bool(transition.get("limit_touched", past.any())))
                flags["start_past"].append(bool(past[0]))
                flags["end_past"].append(bool(past[-1]))
                flags["knots_past"].append(int(past.sum()))
                flags["knots"].append(len(path))
                flags["max_depth_ratio"].append(float((depth / layout["w0"]).max()))
                if keep_knots and past.any():
                    chosen = path[past]
                    knot_states.append(chosen)
                    knot_torques.append(np.stack([controller.action_numpy(state, resolved) for state in chosen]))
                if transition["terminated"] or transition["truncated"]:
                    break
    finally:
        env.observe = original
    out = {key: np.asarray(value, dtype=np.float64) for key, value in rows.items()}
    out["durations"] = out["durations"].reshape(-1, 1)
    for key, value in flags.items():
        out[key] = np.asarray(value)
    out["force_fraction"] = (np.abs(out["torque0"]) / limit).max(axis=-1)
    out["episodes"] = np.array(episodes)
    if keep_knots:
        dimension = env.state_dim
        out["knot_states"] = np.concatenate(knot_states) if knot_states else np.zeros((0, dimension))
        out["knot_torques"] = np.concatenate(knot_torques) if knot_torques else np.zeros((0, m))
    return out


def share_summary(data: Mapping[str, np.ndarray]) -> Dict[str, float]:
    touch = data["touch"].astype(bool)
    high = data["force_fraction"] > HIGH_FORCE
    low = data["force_fraction"] < LOW_FORCE
    return {
        "rows": int(len(touch)),
        "episodes": int(data["episodes"]),
        "touch_share": float(touch.mean()),
        "start_past_share": float(data["start_past"].mean()),
        "end_past_share": float(data["end_past"].mean()),
        "knots_past_share": float(data["knots_past"].sum() / data["knots"].sum()),
        "deep_share": float((data["max_depth_ratio"] > 1.0).mean()),
        "hiF_share": float(high.mean()),
        "loF_share": float(low.mean()),
        "touch_hiF_share": float((touch & high).mean()),
        "touch_loF_share": float((touch & low).mean()),
        "knots_per_interval": sorted({int(k) for k in np.unique(data["knots"])}),
        "env_touch_disagreements": int((touch != data["env_touch"].astype(bool)).sum()) if "env_touch" in data else -1,
    }


def data_path(out: Path, task: str, seed: int) -> Path:
    return out / "data" / f"{task}_s{seed}.npz"


def collect(task: str, seed: int, out: Path, sizes: Mapping[str, int]) -> Path:
    """Collect one task and seed's data, the drift-evaluation set with oracle accelerations,
    and write the oracle's evaluation rows."""
    started = time.time()
    config, testbed = row_config(task)
    env = tasks.make_env(testbed, seed=seed)
    spec = env.state_spec
    layout = limit_layout(spec)
    settings = physics_settings(env, config, testbed)
    print(f"collect {task} seed {seed} header: " + json.dumps(settings), flush=True)
    base = 1_000_003 * (seed + 1)
    sets = {
        "train": collect_set(env, layout, count=sizes["train"], env_seed=base + 1, decision_seed=base + 11, gain_fraction=1.0, keep_knots=False),
        "uniform": collect_set(env, layout, count=sizes["heldout"], env_seed=base + 2, decision_seed=base + 12, gain_fraction=1.0, keep_knots=True),
        "moderate": collect_set(env, layout, count=sizes["moderate"], env_seed=base + 3, decision_seed=base + 13, gain_fraction=MODERATE_FRACTION, keep_knots=False),
    }
    collected = time.time() - started
    uniform = sets["uniform"]
    rng = np.random.default_rng(base + 21)
    take = rng.permutation(len(uniform["knot_states"]))[: sizes["drift_knots"]]
    drift_states = np.concatenate((uniform["states"], uniform["knot_states"][take]))
    drift_torques = np.concatenate((uniform["torque0"], uniform["knot_torques"][take]))
    limit = np.maximum(np.abs(np.asarray(testbed["controller"]["torque_low"])), np.abs(np.asarray(testbed["controller"]["torque_high"])))
    n_o = int(spec["position_dim"])
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        oracle = tasks.make_oracle_plant(env)
        with torch.no_grad():
            accelerations = oracle.drift(torch.as_tensor(drift_states), torch.as_tensor(drift_torques))[:, n_o:].numpy()
    finally:
        torch.set_default_dtype(previous)
    arrays = {}
    for name, data in sets.items():
        for key, value in data.items():
            arrays[f"{name}__{key}"] = value
    arrays["drift__states"] = drift_states
    arrays["drift__torques"] = drift_torques
    arrays["drift__accelerations"] = accelerations
    arrays["drift__depth_ratio"] = (depths(drift_states, layout) / layout["w0"]).max(axis=-1)
    arrays["drift__force_fraction"] = (np.abs(drift_torques) / limit).max(axis=-1)
    arrays["drift__is_knot"] = np.concatenate((np.zeros(len(uniform["states"]), dtype=bool), np.ones(len(take), dtype=bool)))
    path = data_path(out, task, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, **arrays)
    meta = {
        "task": task,
        "seed": seed,
        "sizes": dict(sizes),
        "collect_seconds": collected,
        "limit": {key: value.tolist() for key, value in layout.items()},
        "shares": {name: share_summary(data) for name, data in sets.items()},
        "drift_rows": int(len(drift_states)),
        "drift_knots": int(len(take)),
        "knots_past_available": int(len(uniform["knot_states"])),
        "model_step": float(config["model_step"]),
        "model_integrator": str(config.get("model_integrator", "rk4")),
        "physics": settings,
        "durations": sorted({float(d) for d in np.unique(sets["train"]["durations"])}),
        "network": config["network"],
        "training": {key: config["training"][key] for key in ("learning_rate", "gradient_clip", "batch_size")},
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    os.replace(temporary, path)
    print(f"collected {task} seed {seed} in {collected:.0f} s: " + json.dumps(meta["shares"]), flush=True)
    evaluate_oracle(task, seed, out, env, config, load_data(out, task, seed))
    print(f"oracle rows {task} seed {seed} done after {time.time() - started:.0f} s", flush=True)
    return path


def load_data(out: Path, task: str, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(data_path(out, task, seed)) as handle:
        sets: Dict[str, Dict[str, np.ndarray]] = {}
        for key in handle.files:
            name, field = key.split("__", 1)
            sets.setdefault(name, {})[field] = handle[key]
    return sets


def wait_for_data(out: Path, task: str, seed: int, timeout: float) -> None:
    path = data_path(out, task, seed)
    waited = time.time()
    while not path.exists():
        if time.time() - waited > timeout:
            raise TimeoutError(f"No data file {path} after {timeout:.0f} s.")
        time.sleep(15.0)


# ------------------------------------------------------------------------- evaluation


def endpoint_predictions(plant, env, config, data: Mapping[str, np.ndarray], dtype: torch.dtype) -> np.ndarray:
    """Noise-free model endpoints of every row, as the fit integrates them (no graph)."""
    controller = env.controller
    refresh = grid_refresh_steps(controller, int(env.config["controller"]["refresh_steps"]))
    out = []
    with torch.no_grad():
        for start in range(0, len(data["states"]), EVAL_CHUNK):
            part = slice(start, start + EVAL_CHUNK)
            moments = integrate_endpoint_moments(
                plant,
                controller,
                torch.as_tensor(data["states"][part], dtype=dtype),
                torch.as_tensor(data["decisions"][part], dtype=dtype),
                torch.as_tensor(data["durations"][part], dtype=dtype),
                maximum_step=float(config["model_step"]),
                physics_dt=float(env.physics_dt),
                controller_refresh_steps=refresh,
                covariance_mode="leading_order",
                create_graph=False,
                integrator=str(config.get("model_integrator", "rk4")),
            )
            out.append(moments["means"].double().numpy())
    return np.concatenate(out)


def _quantiles(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(values) == 0:
        return {f"{prefix}_med": math.nan, f"{prefix}_p90": math.nan, f"{prefix}_p99": math.nan}
    q = np.quantile(values, (0.5, 0.9, 0.99))
    return {f"{prefix}_med": float(q[0]), f"{prefix}_p90": float(q[1]), f"{prefix}_p99": float(q[2])}


def _pooled(error: np.ndarray, scale: np.ndarray) -> float:
    denominator = float(np.square(scale).sum())
    return math.sqrt(float(np.square(error).sum()) / denominator) if denominator > 0 else math.nan


def interval_splits(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    touch = data["touch"].astype(bool)
    high = data["force_fraction"] > HIGH_FORCE
    low = data["force_fraction"] < LOW_FORCE
    return {
        "all": np.ones_like(touch),
        "touch": touch,
        "free": ~touch,
        "touch_hiF": touch & high,
        "touch_loF": touch & low,
        "free_hiF": ~touch & high,
        "free_loF": ~touch & low,
    }


def endpoint_metrics(predicted: np.ndarray, data: Mapping[str, np.ndarray], spec, layout) -> Dict[str, Dict[str, float]]:
    periodic = tuple(spec["periodic_position_indices"])
    error = wrapped(data["next_states"] - predicted, periodic)
    change = wrapped(data["next_states"] - data["states"], periodic)
    tiny = 1e-300
    full_e, full_d = np.linalg.norm(error, axis=-1), np.linalg.norm(change, axis=-1)
    pos = layout["positions"]
    vel = layout["velocities"]
    pos_e, pos_d = np.linalg.norm(error[:, pos], axis=-1), np.linalg.norm(change[:, pos], axis=-1)
    vel_e, vel_d = np.linalg.norm(error[:, vel], axis=-1), np.linalg.norm(change[:, vel], axis=-1)
    loss = 0.5 * np.square(error).sum(-1) / data["durations"].reshape(-1)
    metrics = {}
    for split, mask in interval_splits(data).items():
        row = {"n": float(mask.sum())}
        row.update(_quantiles(full_e[mask] / np.maximum(full_d[mask], tiny), "end"))
        row["end_nrmse"] = _pooled(full_e[mask], full_d[mask])
        row.update(_quantiles(pos_e[mask] / np.maximum(pos_d[mask], tiny), "pos"))
        row.update(_quantiles(vel_e[mask] / np.maximum(vel_d[mask], tiny), "vel"))
        row["pos_nrmse"] = _pooled(pos_e[mask], pos_d[mask])
        row["vel_nrmse"] = _pooled(vel_e[mask], vel_d[mask])
        row["loss"] = float(loss[mask].mean()) if mask.any() else math.nan
        metrics[split] = row
    return metrics


def drift_metrics(plant, data: Mapping[str, np.ndarray], spec, layout, dtype: torch.dtype) -> Dict[str, Dict[str, float]]:
    n_o = int(spec["position_dim"])
    predicted = []
    with torch.no_grad():
        for start in range(0, len(data["states"]), EVAL_CHUNK):
            part = slice(start, start + EVAL_CHUNK)
            drift = plant.drift(torch.as_tensor(data["states"][part], dtype=dtype), torch.as_tensor(data["torques"][part], dtype=dtype))
            predicted.append(drift[:, n_o:].double().numpy())
    predicted = np.concatenate(predicted)
    truth = data["accelerations"]
    error = predicted - truth
    dofs = layout["dofs"]
    acc_e, acc_d = np.linalg.norm(error, axis=-1), np.linalg.norm(truth, axis=-1)
    lim_e, lim_d = np.linalg.norm(error[:, dofs], axis=-1), np.linalg.norm(truth[:, dofs], axis=-1)
    ratio = data["depth_ratio"]
    categories = {"inside": ratio <= 0.0, "ramp": (ratio > 0.0) & (ratio <= 1.0), "past": ratio > 1.0}
    high = data["force_fraction"] > HIGH_FORCE
    low = data["force_fraction"] < LOW_FORCE
    splits = dict(categories)
    for name, mask in categories.items():
        splits[f"{name}_hiF"] = mask & high
        splits[f"{name}_loF"] = mask & low
    tiny = 1e-300
    metrics = {}
    for split, mask in splits.items():
        row = {"n": float(mask.sum())}
        row.update(_quantiles(acc_e[mask] / np.maximum(acc_d[mask], tiny), "acc"))
        row["acc_nrmse"] = _pooled(acc_e[mask], acc_d[mask])
        lim = _quantiles(lim_e[mask] / np.maximum(lim_d[mask], tiny), "lim")
        row.update({key: value for key, value in lim.items() if not key.endswith("p99")})
        row["lim_nrmse"] = _pooled(lim_e[mask], lim_d[mask])
        metrics[split] = row
    return metrics


def evaluate(plant, env, config, sets, layout, dtype: torch.dtype) -> Dict[str, Dict[str, Dict[str, float]]]:
    spec = env.state_spec
    result = {}
    for name in ("uniform", "moderate"):
        result[name] = endpoint_metrics(endpoint_predictions(plant, env, config, sets[name], dtype), sets[name], spec, layout)
    result["drift"] = drift_metrics(plant, sets["drift"], spec, layout, dtype)
    return result


class RowWriter:
    """Appends long-format rows to a TSV, flushed after every evaluation."""

    def __init__(self, path: Path, task: str, variant: str, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.handle.write("\t".join(ROW_COLUMNS) + "\n")
        self.handle.flush()
        self.prefix = (task, variant, str(seed))

    def write(self, step: int, wall: float, result: Mapping[str, Mapping[str, Mapping[str, float]]], probe: str = "none") -> None:
        lines = []
        for name, splits in result.items():
            for split, metrics in splits.items():
                for metric, value in metrics.items():
                    lines.append("\t".join((*self.prefix, str(step), f"{wall:.1f}", probe, name, split, metric, repr(float(value)))))
        self.handle.write("\n".join(lines) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def evaluate_oracle(task: str, seed: int, out: Path, env, config, sets) -> None:
    """Variant D: the oracle plant in float64 (D64) and float32 (D32), no training."""
    import traceback

    layout = limit_layout(env.state_spec)
    for label, dtype in (("D64", torch.float64), ("D32", torch.float32)):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        started = time.time()
        try:
            oracle = tasks.make_oracle_plant(env)
            result = evaluate(oracle, env, config, sets, layout, dtype)
        except Exception:  # an oracle evaluation must not stop the collector after its data is out
            print(f"oracle {label} evaluation of {task} seed {seed} failed:", flush=True)
            traceback.print_exc()
            continue
        finally:
            torch.set_default_dtype(previous)
        writer = RowWriter(out / "rows" / task / f"{label}_s{seed}.tsv", task, label, seed)
        writer.write(0, time.time() - started, result)
        writer.close()


# --------------------------------------------------------------------------- training


def compiled_port_start(spec) -> List[Tuple[float, float, float]]:
    """The compiled limit's (k, c, w) per limited coordinate (state spec limit_port_init)."""
    return [tuple(float(value) for value in entry) for entry in spec["limit_port_init"]]


def variant_port_init(variant: str, spec) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    """The limit_port_init the variant's plant gets through its state spec: the compiled
    start (B), the compiled start with k and c times WRONG_START_FACTOR, w unchanged (C);
    None for A (no port) and G (no limit_port_init: the plant's generic constants)."""
    if variant in ("A", "G"):
        return None
    compiled = compiled_port_start(spec)
    if variant == "B":
        return tuple(compiled)
    if variant == "C":
        return tuple((WRONG_START_FACTOR * k, WRONG_START_FACTOR * c, w) for k, c, w in compiled)
    raise ValueError(f"Unknown variant {variant!r}.")


def build_plant(variant: str, spec, network: Mapping[str, object], env, seed: int) -> PortHamiltonianPlant:
    """The plant of PHAlgorithm._build_learned_plant for one variant, seeded alike."""
    network = dict(network)
    plant_spec = dict(spec)
    if variant == "A":
        network["limit_port"] = False
    elif variant in ("B", "C"):
        network["limit_port"] = True
        plant_spec["limit_port_init"] = variant_port_init(variant, spec)
    elif variant == "G":
        network["limit_port"] = True
        plant_spec.pop("limit_port_init", None)
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    if network.get("diffusion", "learned") != "none":
        raise ValueError("The dm_control rows are drift-only; this experiment assumes diffusion 'none'.")
    torch.manual_seed(10_007 + seed)
    plant = PortHamiltonianPlant(plant_spec, network, learn_diffusion=False)
    plant.initialize_from_physics(torch.as_tensor(env.oracle_input_matrix()), torch.ones(int(spec["velocity_dim"])))
    return plant


def port_values(plant: PortHamiltonianPlant) -> Optional[np.ndarray]:
    """(k, c, w) per limited coordinate, [L, 3]; None without the port."""
    if not plant.limit_port:
        return None
    with torch.no_grad():
        return np.stack(
            (
                plant.limit_stiffness.exp().double().numpy(),
                plant.limit_damping.exp().double().numpy(),
                (LIMIT_WIDTH_FLOOR + torch.nn.functional.softplus(plant.limit_width)).double().numpy(),
            ),
            axis=-1,
        )


def run_probes(plant: PortHamiltonianPlant, run_evaluation, step: int) -> None:
    """Evaluate the PROBES on the plant's current port (each restored afterwards)."""
    for name, which, factor in PROBES:
        saved = {key: value.detach().clone() for key, value in plant.state_dict().items()}
        with torch.no_grad():
            if which == "c":
                plant.limit_damping.add_(math.log(factor))
            elif which == "k":
                plant.limit_stiffness.add_(math.log(factor))
            else:
                plant.limit_port = False
        try:
            run_evaluation(step, probe=name)
        finally:
            plant.limit_port = True
            plant.load_state_dict(saved)


def state_dir(out: Path, task: str, variant: str, seed: int) -> Path:
    return out / "state" / task / f"{variant}_s{seed}"


def save_state(plant: PortHamiltonianPlant, out: Path, task: str, variant: str, seed: int, step: int, name: str, **extra) -> Path:
    """torch.save the plant's state dict to state_dir/name (atomically) and point
    state_dir/latest.json at it, with the port parameters at that step."""
    folder = state_dir(out, task, variant, seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    temporary = folder / (name + ".partial")
    torch.save(plant.state_dict(), temporary)
    os.replace(temporary, path)
    values = port_values(plant)
    latest = {"step": int(step), "file": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "port": None if values is None else values.tolist(), **extra}
    pointer = folder / "latest.json.partial"
    pointer.write_text(json.dumps(latest, indent=1))
    os.replace(pointer, folder / "latest.json")
    return path


def _float_list(values) -> list:
    return [[float(cell) for cell in row] for row in values]


def train(task: str, variant: str, seed: int, out: Path, *, max_steps: int, eval_every: int, port_every: int, save_every: int, deadline: float, wait_timeout: float) -> dict:
    wait_for_data(out, task, seed, wait_timeout)
    started = time.time()
    config, testbed = row_config(task)
    env = tasks.make_env(testbed, seed=seed)
    spec = env.state_spec
    layout = limit_layout(spec)
    sets = load_data(out, task, seed)
    training = config["training"]
    dtype = torch.get_default_dtype()
    plant = build_plant(variant, spec, config["network"], env, seed)
    parameters = list(plant.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(training["learning_rate"]))
    clip = float(training["gradient_clip"])
    batch_size = int(training["batch_size"])
    model_step = float(config["model_step"])
    integrator = str(config.get("model_integrator", "rk4"))
    refresh = grid_refresh_steps(env.controller, int(testbed["controller"]["refresh_steps"]))
    train_set = {key: torch.as_tensor(sets["train"][key], dtype=dtype) for key in ("states", "decisions", "durations", "next_states")}
    rows = len(train_set["states"])
    rng = np.random.default_rng([seed, 17])

    # The run's header: the settings read from the current code, the port start, the stop rule.
    start_port = port_values(plant)
    compiled = np.stack((layout["k0"], layout["c0"], layout["w0"]), -1)
    header = {
        "task": task,
        "variant": variant,
        "variant_note": VARIANT_NOTES[variant],
        "seed": seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "started_epoch": started,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "threads": torch.get_num_threads(),
        "dtype": str(dtype),
        **physics_settings(env, config, testbed),
        "plant_limit_port": bool(plant.limit_port),
        "plant_damping_velocity": bool(getattr(plant, "damping_velocity", False)),
        "plant_learns_diffusion": bool(getattr(plant, "learns_diffusion", True)),
        "grid_refresh_steps": refresh,
        "plant_hidden_sizes": config["network"].get("plant_hidden_sizes"),
        "damping_rank": config["network"].get("damping_rank"),
        "velocity_feature_scale": config["network"].get("velocity_feature_scale"),
        "learning_rate": float(training["learning_rate"]),
        "gradient_clip": clip,
        "batch_size": batch_size,
        "train_rows": rows,
        "heldout_rows": {name: int(len(sets[name]["states"])) for name in ("uniform", "moderate", "drift")},
        "data_durations": sorted({float(d) for d in np.unique(sets["train"]["durations"])}),
        "data_knots_per_interval": sorted({int(k) for k in np.unique(sets["train"]["knots"])}),
        "data_train_touch_share": float(sets["train"]["touch"].astype(bool).mean()),
        "port_start": None if start_port is None else _float_list(start_port),
        "port_compiled": _float_list(compiled),
        "port_start_over_compiled": None if start_port is None else _float_list(start_port / compiled),
        "max_steps": max_steps,
        "eval_every": eval_every,
        "port_every": port_every,
        "save_every": save_every,
        "deadline_epoch": deadline,
        "deadline": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(deadline)) if deadline > 0 else None,
    }
    header_path = out / "runs" / f"{task}_{variant}_s{seed}.json"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(json.dumps(header, indent=1, default=str))
    print(f"{task} {variant} s{seed} header: " + json.dumps(header, default=str), flush=True)

    writer = RowWriter(out / "rows" / task / f"{variant}_s{seed}.tsv", task, variant, seed)
    port_path = out / "port" / task / f"{variant}_s{seed}.tsv"
    port_path.parent.mkdir(parents=True, exist_ok=True)
    port_handle = port_path.open("w", encoding="utf-8")
    port_handle.write("step\tcoordinate\tk\tc\tw\tk0\tc0\tw0\tk_start\tc_start\tw_start\n")

    def log_port(step: int) -> None:
        values = port_values(plant)
        if values is None:
            return
        for index, (k, c, w) in enumerate(values):
            cells = (k, c, w, layout["k0"][index], layout["c0"][index], layout["w0"][index], *start_port[index])
            port_handle.write(f"{step}\t{index}\t" + "\t".join(repr(float(cell)) for cell in cells) + "\n")
        port_handle.flush()

    eval_seconds = 0.0
    train_seconds = 0.0
    recent: List[float] = []

    def run_evaluation(step: int, probe: str = "none") -> None:
        nonlocal eval_seconds
        began = time.time()
        result = evaluate(plant, env, config, sets, layout, dtype)
        if probe == "none":
            result["train"] = {"all": {"loss_recent": float(np.mean(recent)) if recent else math.nan, "train_seconds": train_seconds}}
        writer.write(step, time.time() - started, result, probe)
        eval_seconds = time.time() - began

    log_port(0)
    run_evaluation(0)
    print(f"{task} {variant} s{seed}: step 0 evaluated in {eval_seconds:.1f} s", flush=True)
    step = 0
    stopped = "max_steps"
    probe_count = len(PROBES) if plant.limit_port else 0
    while step < max_steps:
        if deadline > 0 and time.time() + (2 + probe_count) * eval_seconds * 1.25 + 120.0 > deadline:
            stopped = "deadline"
            break
        began = time.time()
        index = torch.as_tensor(rng.integers(0, rows, size=batch_size))
        batch = {key: value[index] for key, value in train_set.items()}
        moments = integrate_endpoint_moments(
            plant,
            env.controller,
            batch["states"],
            batch["decisions"],
            batch["durations"],
            maximum_step=model_step,
            physics_dt=float(env.physics_dt),
            controller_refresh_steps=refresh,
            covariance_mode="leading_order",
            create_graph=True,
            integrator=integrator,
        )
        loss = plant_likelihood_loss(batch, moments, spec)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        step += 1
        value = float(loss.detach())
        if not math.isfinite(value):
            stopped = "nonfinite_loss"
            print(f"{task} {variant} s{seed}: nonfinite loss at step {step}", flush=True)
            break
        recent.append(value)
        if len(recent) > eval_every:
            recent.pop(0)
        train_seconds += time.time() - began
        if step % port_every == 0:
            log_port(step)
        if step in PROGRESS_STEPS and step < eval_every:
            print(
                f"{task} {variant} s{seed}: step {step} loss {np.mean(recent):.4g} "
                f"{train_seconds / step:.3f} s/step, wall {time.time() - started:.0f} s",
                flush=True,
            )
        if step % eval_every == 0:
            run_evaluation(step)
            print(
                f"{task} {variant} s{seed}: step {step} loss {np.mean(recent):.4g} "
                f"{train_seconds / step:.3f} s/step, eval {eval_seconds:.1f} s, wall {time.time() - started:.0f} s",
                flush=True,
            )
        if save_every > 0 and step % save_every == 0:
            save_state(plant, out, task, variant, seed, step, f"plant_step{step:06d}.pt", stopped="running")
    if step % eval_every != 0 and stopped != "nonfinite_loss":
        log_port(step)
        run_evaluation(step)
    final_port = port_values(plant)
    # The final state before the probes, so that a kill during them keeps it.
    save_state(plant, out, task, variant, seed, step, "plant_final.pt", stopped=stopped)
    print(f"{task} {variant} s{seed}: stopped ({stopped}) at step {step}, final state saved, wall {time.time() - started:.0f} s", flush=True)
    if plant.limit_port and stopped != "nonfinite_loss":
        run_probes(plant, run_evaluation, step)
    writer.close()
    port_handle.close()
    summary = {
        "task": task,
        "variant": variant,
        "variant_note": VARIANT_NOTES[variant],
        "seed": seed,
        "steps": step,
        "stopped": stopped,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(step, 1),
        "wall_seconds": time.time() - started,
        "final_port": None if final_port is None else final_port.tolist(),
        "start_port": header["port_start"],
        "compiled_port": compiled.tolist(),
    }
    done = out / "done" / f"{task}_{variant}_s{seed}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(summary, indent=1))
    torch.save(plant.state_dict(), out / "done" / f"{task}_{variant}_s{seed}_plant.pt")
    print(f"{task} {variant} s{seed}: done, {json.dumps(summary)}", flush=True)
    return summary


def probe_saved(task: str, variant: str, seed: int, out: Path, state: Optional[Path]) -> None:
    """The probes on a saved plant (state, else the run's latest.json), for a run killed
    before its own probes; rows to rows/<task>/<variant>_s<S>_probe.tsv at the saved step."""
    if variant not in PORT_VARIANTS:
        raise ValueError(f"Variant {variant!r} has no port to probe.")
    folder = state_dir(out, task, variant, seed)
    if state is None:
        latest = json.loads((folder / "latest.json").read_text())
        state, step = folder / latest["file"], int(latest["step"])
    else:
        digits = "".join(ch for ch in state.stem if ch.isdigit())
        step = int(digits) if digits else -1
    started = time.time()
    config, testbed = row_config(task)
    env = tasks.make_env(testbed, seed=seed)
    spec = env.state_spec
    layout = limit_layout(spec)
    sets = load_data(out, task, seed)
    dtype = torch.get_default_dtype()
    plant = build_plant(variant, spec, config["network"], env, seed)
    plant.load_state_dict(torch.load(state, map_location="cpu"))
    writer = RowWriter(out / "rows" / task / f"{variant}_s{seed}_probe.tsv", task, variant, seed)

    def run_evaluation(at: int, probe: str = "none") -> None:
        writer.write(at, time.time() - started, evaluate(plant, env, config, sets, layout, dtype), probe)

    print(f"probing {task} {variant} s{seed} from {state} (step {step}), port {port_values(plant).tolist()}", flush=True)
    run_probes(plant, run_evaluation, step)
    writer.close()
    print(f"probes of {task} {variant} s{seed} done in {time.time() - started:.0f} s", flush=True)


# -------------------------------------------------------------------------- summarize


def read_rows(out: Path):
    import csv

    records = []
    for path in sorted((out / "rows").glob("*/*.tsv")):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["seed"] = int(row["seed"])
                row["step"] = int(row["step"])
                row["value"] = float(row["value"])
                records.append(row)
    return records


def read_port_table(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """A port TSV as columns; cells may be plain floats or numpy 2's repr "np.float64(x)"
    (the writer of the 2026-11-10 job 1134981 wrote the latter)."""
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("\t")]
        rows.append([float(cell[len("np.float64("):-1]) if cell.startswith("np.float64(") else float(cell) for cell in cells])
    table_ = np.array(rows, dtype=np.float64)
    return {name: table_[:, i] for i, name in enumerate(header)}


def read_state_dict(path: Path) -> Dict[str, np.ndarray]:
    """A torch.save'd state dict of plain tensors as numpy arrays, without torch."""
    import collections
    import pickle
    import zipfile

    kinds = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
             "IntStorage": np.int32, "BoolStorage": np.bool_, "HalfStorage": np.float16}
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0]

        def rebuild(storage, offset, size, stride, *rest):
            if not size:
                return np.array(storage[offset])
            item = storage.itemsize
            view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride))
            return np.array(view)

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return rebuild
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                if module == "torch" and name in kinds:
                    return name
                raise pickle.UnpicklingError(f"Unexpected {module}.{name} in {path}.")

            def persistent_load(self, pid):
                _, kind, key, _, _ = pid
                kind = kind if isinstance(kind, str) else getattr(kind, "__name__", str(kind))
                return np.frombuffer(archive.read(f"{prefix}/data/{key}"), dtype=kinds[kind])

        import io

        return dict(Reader(io.BytesIO(archive.read(f"{prefix}/data.pkl"))).load())


def _fmt(values: Sequence[float]) -> str:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return "nan"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.3g}"
    return f"{mean:.3g}±{float(np.std(values, ddof=1)):.2g}"


def summarize(out: Path) -> None:
    records = read_rows(out)
    if not records:
        print("no rows yet")
        return
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[tuple, float] = {}
    steps: Dict[tuple, set] = {}
    probe_steps: Dict[tuple, set] = {}
    for row in records:
        key = (row["task"], row["variant"], row["seed"], row["step"], row["probe"], row["set"], row["split"], row["metric"])
        index[key] = row["value"]
        probe_steps.setdefault((row["task"], row["variant"], row["seed"], row["probe"]), set()).add(row["step"])
        if row["probe"] == "none":
            steps.setdefault((row["task"], row["variant"], row["seed"]), set()).add(row["step"])
    tasks_seen = [task for task in TASKS if any(key[0] == task for key in steps)] + sorted(
        {key[0] for key in steps} - set(TASKS)
    )
    # The trainable variants with rows, in LEARNED_VARIANTS order.
    variants = tuple(v for v in LEARNED_VARIANTS if any(key[1] == v for key in steps))
    port_variants = tuple(v for v in variants if v in PORT_VARIANTS)
    lines: List[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # Data shares.
    emit("# Share of intervals that touch a limit (mean over seeds); knots per interval; touch flags disagreeing with the env's")
    emit("task\tset\trows\ttouch\tstart_past\tend_past\tknots_past\tdeep(>w0)\thiF\tloF\ttouch_hiF\ttouch_loF\tknots\tenv_disagree")
    for task in tasks_seen:
        metas = [json.loads(path.read_text()) for path in sorted((out / "data").glob(f"{task}_s*.json"))]
        for name in ("train", "uniform", "moderate"):
            shares = [meta["shares"][name] for meta in metas if name in meta.get("shares", {})]
            if not shares:
                continue
            cells = [f"{np.mean([s[key] for s in shares]):.3f}" for key in (
                "touch_share", "start_past_share", "end_past_share", "knots_past_share", "deep_share",
                "hiF_share", "loF_share", "touch_hiF_share", "touch_loF_share")]
            cells.append(",".join(str(k) for k in sorted({k for s in shares for k in s.get("knots_per_interval", [])})) or "?")
            cells.append(str(sum(s.get("env_touch_disagreements", 0) for s in shares)))
            emit("\t".join([task, name, str(shares[0]["rows"])] + cells))
        physics = [meta["physics"] for meta in metas if "physics" in meta]
        if physics:
            p = physics[0]
            emit(f"{task}\tphysics {p['physics_integrator_mujoco']} at {p['physics_dt']} s ({p['physics_steps_per_reference_period']:.0f} steps per "
                 f"{p['reference_period']} s), model {p['model_integrator']} at {p['model_step']} s, damping_velocity {p['damping_velocity']}, "
                 f"touch knots every {p['touch_knot_spacing']} s")
    emit()

    # Run headers (variant, port start over the compiled limit, threads, start time).
    headers = [json.loads(path.read_text()) for path in sorted((out / "runs").glob("*.json"))]
    if headers:
        emit("# Run headers: variant, limit_port, port start / compiled (geometric mean over coordinates of k, c, w), started, host, threads")
        for h in headers:
            ratio = h.get("port_start_over_compiled")
            cells = "-" if not ratio else "\t".join(f"{float(np.exp(np.mean(np.log(np.asarray(ratio)[:, i])))):.3g}" for i in range(3))
            emit("\t".join(str(x) for x in (h["task"], h["variant"], f"s{h['seed']}", h.get("variant_note", ""), h.get("plant_limit_port"),
                                            cells, h.get("started"), h.get("host"), h.get("threads"))))
        emit()

    # Runs.
    emit("# Runs: last evaluated step per seed, s/step, stop reason")
    common: Dict[str, int] = {}
    for task in tasks_seen:
        learned = {key: value for key, value in steps.items() if key[0] == task and key[1] in variants}
        if learned:
            common[task] = min(max(found) for found in learned.values())
            shared = set.intersection(*learned.values())
            common[task] = max(step for step in shared if step <= common[task]) if shared else 0
        for (t, variant, seed), found in sorted(learned.items()):
            done_path = out / "done" / f"{t}_{variant}_s{seed}.json"
            done = json.loads(done_path.read_text()) if done_path.exists() else {}
            last = max(found)
            per_step = done.get("seconds_per_step")
            if per_step is None:  # a running (or killed) run: its training seconds at the last evaluation
                spent = index.get((t, variant, seed, last, "none", "train", "all", "train_seconds"))
                per_step = spent / last if spent is not None and last > 0 else float("nan")
            emit(f"{t}\t{variant}\ts{seed}\tlast {last}\t{per_step:.3f} s/step\t{done.get('stopped', 'running')}")
        emit(f"{task}\tcommon step {common.get(task, 0)}")
    emit()

    def value_list(task, variant, step, probe, name, split, metric):
        seeds = sorted({key[2] for key in steps if key[0] == task and key[1] == variant})
        out_values = []
        for seed in seeds:
            use_step = step
            if probe != "none" or variant.startswith("D"):
                candidates = probe_steps.get((task, variant, seed, probe))
                if not candidates:
                    continue
                use_step = max(candidates) if probe != "none" else 0
            value = index.get((task, variant, seed, use_step, probe, name, split, metric))
            if value is not None:
                out_values.append(value)
        return out_values

    variants_all = variants + ("D64", "D32")
    endpoint_columns = ("n", "end_med", "end_p90", "end_p99", "end_nrmse", "pos_med", "vel_med", "vel_p90", "vel_nrmse", "loss")
    for name in ("uniform", "moderate"):
        emit(f"# Endpoint error, held-out {name} decisions, at the common step (mean±sd over seeds)")
        emit("task\tvariant\tsplit\t" + "\t".join(endpoint_columns))
        for task in tasks_seen:
            splits = ("all", "touch", "free") + (("touch_hiF", "touch_loF", "free_hiF", "free_loF") if task.startswith("cartpole") else ())
            for split in splits:
                for variant in variants_all:
                    cells = [_fmt(value_list(task, variant, common.get(task, 0), "none", name, split, metric)) for metric in endpoint_columns]
                    if all(cell == "nan" for cell in cells):
                        continue
                    emit("\t".join([task, variant, split] + cells))
        emit()

    drift_columns = ("n", "acc_med", "acc_p90", "acc_p99", "acc_nrmse", "lim_med", "lim_nrmse")
    emit("# Drift (acceleration) error against the oracle, uniform held-out starts plus knots past a range, at the common step")
    emit("task\tvariant\tsplit\t" + "\t".join(drift_columns))
    for task in tasks_seen:
        splits = ("inside", "ramp", "past") + (("inside_hiF", "inside_loF", "past_hiF", "past_loF") if task.startswith("cartpole") else ())
        for split in splits:
            for variant in variants + ("D32",):
                cells = [_fmt(value_list(task, variant, common.get(task, 0), "none", "drift", split, metric)) for metric in drift_columns]
                if all(cell == "nan" for cell in cells):
                    continue
                emit("\t".join([task, variant, split] + cells))
    emit()

    emit("# Trajectory of the port parameters (geometric mean over coordinates of value / compiled, mean over seeds)")
    emit("task\tvariant\tstep\tk/k0\tc/c0\tw/w0\tk (coordinate 0)\tc (coordinate 0)\tw (coordinate 0)")
    for task in tasks_seen:
        for variant in port_variants:
            tables = []
            for path in sorted((out / "port" / task).glob(f"{variant}_s*.tsv")):
                data = read_port_table(path)
                if data is not None:
                    tables.append(data)
            if not tables:
                continue
            last = min(int(t["step"].max()) for t in tables)
            marks = sorted({0, 200, 1000, 2000, 5000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000,
                            80_000, 100_000, 120_000, 150_000, 200_000, last})
            for mark in marks:
                if mark > last:
                    continue
                ratios, first = [], []
                for t in tables:
                    available = t["step"][t["step"] <= mark]
                    if not len(available):
                        continue
                    at = t["step"] == available.max()
                    ratios.append([float(np.exp(np.mean(np.log(t[f][at] / t[f + "0"][at])))) for f in ("k", "c", "w")])
                    zero = at & (t["coordinate"] == 0)
                    first.append([float(t[f][zero][0]) for f in ("k", "c", "w")])
                if not ratios:
                    continue
                ratios, first = np.array(ratios), np.array(first)
                emit("\t".join([task, variant, str(mark)] + [_fmt(ratios[:, i]) for i in range(3)] + [_fmt(first[:, i]) for i in range(3)]))
    emit()

    emit("# Final port in the learned gauge (k/G and c/G on each limited dof over the compiled k0, c0; G the learned")
    emit("# input-matrix entry of the dof's actuator, 1 in the true gauge, gear 1): per seed, geometric mean over coordinates")
    emit("# (a run without its done file reads its latest saved state, state/<task>/<variant>_s<S>/latest.json; steps marked 'saved')")
    emit("task\tvariant\tseed\tsteps\tG(limited dofs)\tk/k0\tc/c0\t(k/G)/k0\t(c/G)/c0\tw/w0")
    for task in tasks_seen:
        for variant in variants:
            sources = []
            for seed in sorted({key[2] for key in steps if key[0] == task and key[1] == variant}):
                finished = out / "done" / f"{task}_{variant}_s{seed}_plant.pt"
                pointer = state_dir(out, task, variant, seed) / "latest.json"
                if finished.exists():
                    sources.append((seed, finished, None))
                elif pointer.exists():
                    try:
                        latest = json.loads(pointer.read_text())
                        sources.append((seed, pointer.parent / latest["file"], f"{latest['step']} saved"))
                    except Exception as error:
                        emit(f"{task}\t{variant}\ts{seed}\tunreadable {pointer}: {error!r}")
            for seed, path, saved_steps in sources:
                try:
                    state = read_state_dict(path)
                    if saved_steps is None:
                        done = json.loads(path.with_name(path.name.replace("_plant.pt", ".json")).read_text())
                    else:
                        done = {"steps": saved_steps}
                    meta = json.loads(sorted((out / "data").glob(f"{task}_s*.json"))[0].read_text())
                    dofs = np.asarray(meta["limit"]["dofs"], dtype=int)
                    force = state["force_matrix"]
                    actuator = np.argmax(np.abs(force[dofs]), axis=-1)
                    gains = force[dofs, actuator]
                    cells = [f"{np.exp(np.mean(np.log(np.abs(gains)))):.3g}"]
                    if "limit_stiffness" in state:
                        k = np.exp(state["limit_stiffness"].astype(np.float64))
                        c = np.exp(state["limit_damping"].astype(np.float64))
                        w = 1e-3 + np.log1p(np.exp(state["limit_width"].astype(np.float64)))
                        k0, c0, w0 = (np.asarray(meta["limit"][key]) for key in ("k0", "c0", "w0"))
                        geo = lambda values: f"{float(np.exp(np.mean(np.log(np.abs(values))))):.3g}"
                        cells += [geo(k / k0), geo(c / c0), geo(k / np.abs(gains) / k0), geo(c / np.abs(gains) / c0), geo(w / w0)]
                    emit("\t".join([task, variant, str(seed), str(done.get("steps"))] + cells))
                except Exception as error:  # the gauge table is an extra; never block the summary
                    emit(f"{task}\t{variant}\t{path.name}\tunreadable: {error!r}")
    emit()

    emit("# Probes at each run's final step: held-out uniform, touching intervals (end_med, vel_med, end_nrmse) and drift past/ramp nrmse")
    emit("task\tvariant\tprobe\tend_med\tvel_med\tend_nrmse\tfree end_nrmse\tdrift past acc_nrmse\tdrift ramp acc_nrmse")
    for task in tasks_seen:
        for variant in port_variants:
            for probe in ("none",) + tuple(name for name, _, _ in PROBES):
                def final(name, split, metric):
                    values = []
                    for seed in sorted({key[2] for key in steps if key[0] == task and key[1] == variant}):
                        candidates = probe_steps.get((task, variant, seed, probe))
                        if candidates:
                            value = index.get((task, variant, seed, max(candidates), probe, name, split, metric))
                            if value is not None:
                                values.append(value)
                    return _fmt(values)
                cells = [final("uniform", "touch", "end_med"), final("uniform", "touch", "vel_med"), final("uniform", "touch", "end_nrmse"),
                         final("uniform", "free", "end_nrmse"), final("drift", "past", "acc_nrmse"), final("drift", "ramp", "acc_nrmse")]
                if all(cell == "nan" for cell in cells):
                    continue
                emit("\t".join([task, variant, probe] + cells))
    emit()

    emit("# Learning curves: held-out uniform end_nrmse on touch / free intervals by step (mean over seeds)")
    for task in tasks_seen:
        emit(f"{task}\tstep\t" + "\t".join(f"{v} touch\t{v} free" for v in variants))
        all_steps = sorted(set().union(*[found for key, found in steps.items() if key[0] == task and key[1] in variants]) or {0})
        for step in all_steps:
            if step > common.get(task, 0) and step % 10_000:
                continue
            cells = []
            for variant in variants:
                for split in ("touch", "free"):
                    values = value_list(task, variant, step, "none", "uniform", split, "end_nrmse")
                    cells.append(f"{np.mean(values):.4g}" if values else "")
            emit("\t".join([task, str(step)] + cells))
    emit()
    text = "\n".join(lines)
    (summary_dir / "summary.txt").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("collect", "train", "probe", "summarize", "local"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--variant", choices=LEARNED_VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=LEARNED_VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--port-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None, help="save the plant every N steps (a multiple of --eval-every keeps rows and states aligned)")
    parser.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; stop training in time for the final evaluation and probes")
    parser.add_argument("--wait-timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--state", type=Path, default=None, help="probe: the saved plant (default: the run's latest.json)")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    sizes = TINY if args.tiny else FULL
    max_steps = args.max_steps if args.max_steps is not None else (40 if args.tiny else 60_000)
    eval_every = args.eval_every if args.eval_every is not None else (20 if args.tiny else 2000)
    port_every = args.port_every if args.port_every is not None else (10 if args.tiny else 200)
    save_every = args.save_every if args.save_every is not None else (20 if args.tiny else 10_000)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        collect(args.task, args.seed, args.out, sizes)
    elif args.command == "train":
        train(args.task, args.variant, args.seed, args.out, max_steps=max_steps, eval_every=eval_every,
              port_every=port_every, save_every=save_every, deadline=args.deadline, wait_timeout=args.wait_timeout)
    elif args.command == "probe":
        probe_saved(args.task, args.variant, args.seed, args.out, args.state)
    elif args.command == "summarize":
        summarize(args.out)
    else:
        for task in args.tasks:
            for seed in args.seeds:
                collect(task, seed, args.out, sizes)
                for variant in args.variants:
                    train(task, variant, seed, args.out, max_steps=max_steps, eval_every=eval_every,
                          port_every=port_every, save_every=save_every, deadline=args.deadline, wait_timeout=60.0)
        summarize(args.out)


if __name__ == "__main__":
    main()
