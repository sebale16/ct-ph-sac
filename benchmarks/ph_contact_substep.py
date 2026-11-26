"""Accuracy of contact-triggered fine substepping, and the swimmers at dm_control's 2 ms Euler.

Usage (from the repository root; THREADS sets torch's threads, default 2):
    python benchmarks/ph_contact_substep.py contact ENV_ID OUT_JSON [COUNT]
    python benchmarks/ph_contact_substep.py swimmer ENV_ID OUT_JSON [COUNT] [EPISODES]
    python benchmarks/ph_contact_substep.py reach ENV_ID OUT_JSON [EPISODES]

reach (reacher-easy, reacher-hard, cartpole-swingup, point_mass-easy, ...): contact
exclusion's start-state rule against the environment's limit touches (2026-11-10 evening,
rule version 2, the damped reach bound). Two pools of EPISODES (default 20) whole episodes
each from the reset law: "uniform" (decisions uniform in the box) and "biased" (each
limited coordinate's actuator pushes toward an edge, full setpoint offset, stiffness in
the top 30%, damping in the bottom 10%, on 80% of the intervals; the edge flips after
exponential holds of mean HOLD[family]; every other actuator either pushes too, in a
direction drawn per hold, or is uniform; the remaining 20% of intervals uniform), so that
approaches and contact are common. Per interval: the environment's limit_touched (a
limited coordinate past its range at a physics knot, the start included), the rule's flag
in version 1 (no damping, the experiment of 2026-11-10) and version 2, the endpoint travel
of each limited coordinate against the reach term of the rule. Per pool: the share of
intervals that touch, flagged by each rule, touched but not flagged (missed; must be under
1% of the touched ones, ideally none) and flagged without touching (false positives), the
largest endpoint travel over the reach term, speeds, and the worst misses (start position,
distance to the edge, speed toward it, margins).

contact (cartpole-swingup, cartpole-three_poles, point_mass-easy, ...): the admissible-step
measurement of 2026-11-09/10 (scratchpad continuous_ctrl/step/admissible_step.py and
limits/admissible_step_limits.py, adapted) for the scheme of the row's model_coarse_step
(common.ph_config.MODEL_COARSE_STEPS). Reference: the environment, env.step_dt over one
decision interval (1 ms MuJoCo RK4, the law at every RK4 stage, the soft limits solved by
MuJoCo). Model: integrate_endpoint_moments of the oracle plant (rk4, the law at every
stage), three ways: "contact" (the scheme: coarse step, flagged rows at 1 ms), "coarse"
(every row at the coarse step) and "fine" (every row at 1 ms). Error per row:
||x_model - x_env|| / ||x_env - x_0|| (wrapped residuals); p50, p99, p99.9 and max, over
all rows and over the rows that touch a limit ("engaged": a limited coordinate past its
range at the start, the end or any 1 ms knot of the interval, from the oracle's 1 ms path,
which equals the environment's to ~1e-15 on these tasks) and those that do not. Pools of
COUNT rows (default 2000), decisions uniform in the box with half at stiffness and damping
in the top 10% of their range:
  episodes  interval starts of whole uniform-decision episodes from the reset law
  limit     the episode pool with one limited coordinate moved to its limit, moving
            outward at up to V_MAX: half past the range (by up to P_MAX), half inside
            within one interval's travel (the "contact" pool of 2026-11-10's
            limits/admissible_step_limits.py), so that at least half of the intervals
            touch a limit
  fast      fast approaches: one limited coordinate inside its range by d, moving toward
            the edge at V_FAST (uniform), d uniform in [0, 1.5 x the interval's travel]
For each pool: the share of rows the detector flags, the engaged rows it missed (must be
zero), and the flagged rows that do not touch (the margin's cost).

swimmer (swimmer-swimmer6, swimmer-swimmer15): the table's convention since 2026-11-10,
dm_control's 2 ms Euler physics with the semi_implicit model map at the 2 ms model step.
1. Agreement: the oracle model's one-interval endpoint (15 substeps) against env.step_dt
   on COUNT rows (states from whole uniform-decision episodes, uniform box decisions,
   half at the top 10% of stiffness and damping), at MuJoCo's default solver tolerance
   (1e-8) and at 1e-12: relative and absolute error, also on the rows with a joint past
   its range at the start or the end.
2. Discretization change: the 2 ms Euler endpoint against the 1 ms RK4 environment's (the
   convention before) on the same rows, relative to the interval's displacement.
3. Stability: one-interval growth on the same rows: the velocity growth
   ||qdot_end|| / max(||qdot_start||, 0.1) in both environments, and, on a subset
   (JAC_COUNT), the spectral radius of the one-interval closed-loop Jacobian of the
   oracle model (the product of the step Jacobians of augmented_closed_loop_step, physical
   block) for the semi_implicit 2 ms map and the rk4 1 ms map; then EPISODES (default 20)
   whole episodes of uniform decisions at 2 ms Euler: non-finite states, exploding rates
   (|qdot| > 1e3) and the largest rate and growth.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
torch.set_default_dtype(torch.float64)

import environment.ph_sac.dmc  # noqa: E402,F401
import models.ph_sac.rollouts as R  # noqa: E402
from benchmarks.ph_hyperparams import find_ph_entry  # noqa: E402
from common.ph_config import testbed_model_coarse_step  # noqa: E402
from common.ph_dmc._common import fixed_clock  # noqa: E402
from environment.ph_sac.registry import testbed_spec  # noqa: E402
from environment.ph_sac.tasks import make_env, make_oracle_plant  # noqa: E402

T0 = time.time()
P_MAX = {"cartpole": 0.05, "point_mass": 0.005, "reacher": 0.05}
V_MAX = {"cartpole": 3.0, "point_mass": 0.2, "reacher": 5.0}
V_FAST = {"cartpole": (3.0, 8.0), "point_mass": (0.3, 1.0), "reacher": (5.0, 15.0)}


def log(*parts):
    print(f"[{time.time() - T0:8.1f}s]", *parts, flush=True)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    out = {"n": int(values.size), "nonfinite": int(values.size - finite.size)}
    if finite.size:
        out.update(
            median=float(np.median(finite)),
            p99=float(np.quantile(finite, 0.99)),
            p999=float(np.quantile(finite, 0.999)),
            max=float(np.max(finite)),
        )
    return out


def table_config(env_id):
    entry = find_ph_entry(env_id, "top")
    testbed = entry.config["testbed"]
    return entry, testbed_spec(testbed).config_factory(**copy.deepcopy(entry.config["testbed_settings"]))


def decisions(env, rng, count):
    low, high = env.action_space.low, env.action_space.high
    m = env.state_spec["action_dim"]
    chosen = rng.uniform(low, high, size=(count, low.size))
    half = count // 2
    chosen[:half, m:2 * m] = high[m:2 * m] * rng.uniform(0.9, 1.0, size=(half, m))
    chosen[:half, 2 * m:] = high[2 * m:] * rng.uniform(0.9, 1.0, size=(half, m))
    return chosen[rng.permutation(count)]


def episode_pool(env, size, seed, *, seconds=None):
    """Interval starts of uniform-decision episodes from the reset law (whole episodes, or
    their first seconds)."""
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    states, episodes = [], 0
    limit = math.inf if seconds is None else seconds
    while len(states) < size:
        env.reset(seed=seed + 1000 + episodes)
        episodes += 1
        while not env.done and env.time < limit - 1e-9:
            states.append(env.observe())
            env.step_dt(rng.uniform(low, high), env.sample_duration())
    return np.asarray(states), episodes


def env_endpoints(env, starts, chosen, duration):
    ends = np.zeros_like(starts)
    terminated = np.zeros(len(starts), bool)
    for row in range(len(starts)):
        env.reset(options={"state": starts[row]})
        transition = env.step_dt(chosen[row], duration)
        ends[row] = transition["next_state"]
        terminated[row] = bool(transition["terminated"])
    return ends, terminated


def model_endpoints(plant, env, starts, chosen, duration, step, *, integrator="rk4", contact=None,
                    physics_dt=None):
    count = len(starts)
    moments = R.integrate_endpoint_moments(
        plant, env.controller, torch.as_tensor(starts), torch.as_tensor(chosen),
        torch.full((count, 1), duration), maximum_step=step,
        physics_dt=env.physics_dt if physics_dt is None else physics_dt,
        controller_refresh_steps=None, covariance_mode="leading_order", create_graph=False,
        integrator=integrator, contact=contact,
    )
    flagged = moments.get("contact_rows")
    return moments["means"].numpy(), None if flagged is None else flagged.numpy()


def relative_errors(env, starts, ends, model):
    residual = R.wrapped_endpoint_residual(
        torch.as_tensor(ends), torch.as_tensor(model), env.state_spec
    ).numpy()
    error = np.linalg.norm(residual, axis=1)
    displacement = np.linalg.norm(ends - starts, axis=1)
    return error / np.maximum(displacement, 1e-300), error


# ----------------------------------------------------------------------------- contact


def contact_main(env_id, out, count):
    family = env_id.split("-")[0]
    entry, config = table_config(env_id)
    coarse = entry.config.get("model_coarse_step") or testbed_model_coarse_step(config["name"])
    if coarse is None:
        raise SystemExit(f"{env_id} has no model_coarse_step.")
    fine = float(entry.config["model_step"])
    env = make_env(config, decision_mode="feedback", seed=0)
    plant = make_oracle_plant(env).double()
    spec = config["state_spec"]
    contact = R.contact_substepping(spec, coarse, fine)
    ranges = [(int(i), float(lo), float(hi)) for i, lo, hi in spec["limited_position_ranges"]]
    position_dim = int(spec["position_dim"])
    projection = np.asarray(spec["observation_projection"])
    dof = {i: int(np.argmax(np.abs(projection[i]))) for i, _, _ in ranges}
    duration = float(config["reference_period"])
    steps = round(duration / env.physics_dt)

    def past(states):
        mask = np.zeros(len(states), bool)
        for index, low, high in ranges:
            mask |= (states[:, index] > high) | (states[:, index] < low)
        return mask

    def engaged(starts, chosen, ends):
        touched = past(starts) | past(ends)
        for k in range(1, steps):
            prefix, _ = model_endpoints(plant, env, starts, chosen, k * env.physics_dt, fine)
            touched |= past(prefix)
        return touched

    def to_limit(rng, states):
        states = states.copy()
        for row in range(len(states)):
            index, low, high = ranges[rng.integers(len(ranges))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            bound = high if sign > 0 else low
            speed = V_MAX[family] * rng.uniform(0.0, 1.0)
            if rng.random() < 0.5:
                states[row, index] = bound + sign * P_MAX[family] * rng.uniform(0.0, 1.0)
            else:
                states[row, index] = bound - sign * speed * duration * rng.uniform(0.0, 1.0)
            states[row, position_dim + dof[index]] = sign * speed
        return states

    def to_fast(rng, states):
        states = states.copy()
        low_speed, high_speed = V_FAST[family]
        for row in range(len(states)):
            index, low, high = ranges[rng.integers(len(ranges))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            bound = high if sign > 0 else low
            speed = rng.uniform(low_speed, high_speed)
            states[row, index] = bound - sign * 1.5 * speed * duration * rng.uniform(0.0, 1.0)
            states[row, position_dim + dof[index]] = sign * speed
        return states

    pool, episodes = episode_pool(env, 8 * count, 0)
    log(env_id, "pool", pool.shape, "episodes", episodes, "coarse", coarse, "fine", fine,
        "pool past-range share", round(float(past(pool).mean()), 5))
    rng = np.random.default_rng(1)
    picks = rng.choice(len(pool), size=3 * count, replace=len(pool) < 3 * count)
    pools = {
        "episodes": pool[picks[:count]],
        "limit": to_limit(rng, pool[picks[count:2 * count]]),
        "fast": to_fast(rng, pool[picks[2 * count:]]),
    }
    report = {
        "env_id": env_id, "coarse_step": coarse, "fine_step": fine, "interval": duration,
        "count": count, "pool": int(len(pool)), "pool_episodes": episodes,
        "pool_past_share": float(past(pool).mean()), "ranges": ranges,
        "detector": dict(contact.record()), "margins": list(contact.margins),
        "p_max": P_MAX[family], "v_max": V_MAX[family], "v_fast": V_FAST[family],
    }
    for name, starts in pools.items():
        chosen = decisions(env, rng, count)
        ends, terminated = env_endpoints(env, starts, chosen, duration)
        touched = engaged(starts, chosen, ends)
        result = {"engaged_share": float(touched.mean()), "terminated": int(terminated.sum())}
        flagged = None
        for method, step, scheme in (("contact", fine, contact), ("coarse", coarse, None),
                                     ("fine", fine, None)):
            model, rows = model_endpoints(plant, env, starts, chosen, duration, step, contact=scheme)
            relative, absolute = relative_errors(env, starts, ends, model)
            result[method] = {
                "all": stats(relative), "engaged": stats(relative[touched]),
                "free": stats(relative[~touched]), "abs_all": stats(absolute),
            }
            if rows is not None:
                flagged = rows
        result["flagged_share"] = float(flagged.mean())
        result["missed_engaged"] = int((touched & ~flagged).sum())
        result["flagged_not_engaged_share"] = float((flagged & ~touched).mean())
        report[name] = result
        line = " | ".join(
            f"{method} p99 {result[method]['all']['p99']:.2e} p99.9 {result[method]['all']['p999']:.2e} "
            f"max {result[method]['all']['max']:.2e}"
            for method in ("contact", "coarse", "fine")
        )
        log(f"[{name}] engaged {result['engaged_share']:.3f} flagged {result['flagged_share']:.3f} "
            f"missed {result['missed_engaged']} false-flag {result['flagged_not_engaged_share']:.3f} :: {line}")
    out.write_text(json.dumps(report, indent=1))
    log("wrote", out)


# ----------------------------------------------------------------------------- swimmer


def rk4_config(config):
    """The swimmer config with the physics before 2026-11-10: 1 ms RK4."""
    converted = copy.deepcopy(config)
    converted["physics_dt"] = 0.001
    converted["integrator"] = "RK4"
    converted["clock"] = fixed_clock(float(config["reference_period"]), 0.001)
    return converted


def interval_jacobian_radius(plant, env, starts, chosen, duration, step, integrator, chunk=50):
    """Spectral radius and 2-norm of the one-interval closed-loop Jacobian of the model map
    (physical block: positions without the context, and velocities)."""
    spec = env.state_spec
    position_dim = int(spec["position_dim"])
    context = int(spec.get("context_dim", 0))
    keep = list(range(position_dim - context)) + list(range(position_dim, env.state_dim))
    pieces = round(duration / step)
    radius, norm = [], []
    for first in range(0, len(starts), chunk):
        states = torch.as_tensor(starts[first:first + chunk])
        resolved = env.controller.resolve_decision(states, torch.as_tensor(chosen[first:first + chunk]))
        count = states.shape[0]
        held = states.new_zeros(count, plant.action_dim)
        mask = torch.ones(count, dtype=torch.bool)
        h = torch.full((count,), step)
        total = torch.eye(env.state_dim).expand(count, -1, -1).clone()
        with torch.no_grad():
            for _ in range(pieces):
                states, held, jacobian = R.augmented_closed_loop_step(
                    plant, env.controller, states, held, resolved, h, mask, integrator=integrator
                )
                total = jacobian[:, : env.state_dim, : env.state_dim] @ total
        block = total[:, keep][:, :, keep]
        radius.append(torch.linalg.eigvals(block).abs().max(dim=-1).values.numpy())
        norm.append(torch.linalg.matrix_norm(block, ord=2).numpy())
    return np.concatenate(radius), np.concatenate(norm)


def velocity_growth(env, starts, ends):
    position_dim = env.position_dim
    start = np.linalg.norm(starts[:, position_dim:], axis=1)
    end = np.linalg.norm(ends[:, position_dim:], axis=1)
    return end / np.maximum(start, 0.1)


def swimmer_main(env_id, out, count, episodes):
    entry, config = table_config(env_id)
    if (config["physics_dt"], config["integrator"]) != (0.002, "Euler"):
        raise SystemExit(f"{env_id}: expected the 2 ms Euler table config, got {config['physics_dt']}, {config['integrator']}.")
    if entry.config.get("model_integrator") != "semi_implicit" or float(entry.config["model_step"]) != 0.002:
        raise SystemExit(f"{env_id}: expected semi_implicit at 2 ms in the row.")
    env = make_env(config, decision_mode="feedback", seed=0)
    env_rk4 = make_env(rk4_config(config), decision_mode="feedback", seed=0)
    plant = make_oracle_plant(env).double()
    duration = float(config["reference_period"])
    pool, pool_episodes = episode_pool(env, 4 * count, 0)
    rng = np.random.default_rng(1)
    starts = pool[rng.choice(len(pool), size=count, replace=len(pool) < count)]
    chosen = decisions(env, rng, count)
    n = int(config["plant_parameters"]["n_links"])
    joint_range = 2.0 * math.pi / n

    def past(states):
        return (np.abs(states[:, 3 : n + 2]) > joint_range).any(axis=1)

    report = {"env_id": env_id, "count": count, "pool": int(len(pool)), "pool_episodes": pool_episodes,
              "pool_past_share": float(past(pool).mean())}
    log(env_id, "pool", pool.shape, "past share", round(report["pool_past_share"], 4))
    model, _ = model_endpoints(plant, env, starts, chosen, duration, 0.002, integrator="semi_implicit")
    for tolerance in (None, 1e-12):
        label = "agreement_default_tolerance" if tolerance is None else "agreement_tolerance_1e-12"
        env.model.opt.tolerance = 1e-8 if tolerance is None else tolerance
        ends, terminated = env_endpoints(env, starts, chosen, duration)
        relative, absolute = relative_errors(env, starts, ends, model)
        limited = past(starts) | past(ends)
        report[label] = {"rel": stats(relative), "abs": stats(absolute), "rel_limited": stats(relative[limited]),
                         "abs_limited": stats(absolute[limited]), "limited_share": float(limited.mean()),
                         "terminated": int(terminated.sum())}
        log(f"[{label}] rel p99 {report[label]['rel']['p99']:.2e} max {report[label]['rel']['max']:.2e} "
            f"abs max {report[label]['abs']['max']:.2e} limited share {limited.mean():.3f}")
    env.model.opt.tolerance = 1e-8
    euler_ends, _ = env_endpoints(env, starts, chosen, duration)
    rk4_ends, _ = env_endpoints(env_rk4, starts, chosen, duration)
    relative, absolute = relative_errors(env, starts, rk4_ends, euler_ends)
    report["euler_2ms_vs_rk4_1ms"] = {"rel": stats(relative), "abs": stats(absolute)}
    log(f"[euler vs rk4] rel p50 {report['euler_2ms_vs_rk4_1ms']['rel']['median']:.2e} "
        f"p99 {report['euler_2ms_vs_rk4_1ms']['rel']['p99']:.2e} max {report['euler_2ms_vs_rk4_1ms']['rel']['max']:.2e}")
    report["velocity_growth_euler"] = stats(velocity_growth(env, starts, euler_ends))
    report["velocity_growth_rk4"] = stats(velocity_growth(env, starts, rk4_ends))
    jac_count = int(os.environ.get("JAC_COUNT", "400" if n <= 6 else "200"))
    radius_euler, norm_euler = interval_jacobian_radius(
        plant, env, starts[:jac_count], chosen[:jac_count], duration, 0.002, "semi_implicit")
    radius_rk4, norm_rk4 = interval_jacobian_radius(
        plant, env, starts[:jac_count], chosen[:jac_count], duration, 0.001, "rk4")
    report["jacobian_radius_euler_2ms"] = stats(radius_euler)
    report["jacobian_radius_rk4_1ms"] = stats(radius_rk4)
    report["jacobian_norm_euler_2ms"] = stats(norm_euler)
    report["jacobian_norm_rk4_1ms"] = stats(norm_rk4)
    report["jacobian_radius_ratio"] = stats(radius_euler / np.maximum(radius_rk4, 1e-300))
    log(f"[growth] velocity euler p99 {report['velocity_growth_euler']['p99']:.3g} max {report['velocity_growth_euler']['max']:.3g}"
        f" | rk4 p99 {report['velocity_growth_rk4']['p99']:.3g} max {report['velocity_growth_rk4']['max']:.3g}"
        f" | jacobian radius euler p99 {report['jacobian_radius_euler_2ms']['p99']:.4g} max {report['jacobian_radius_euler_2ms']['max']:.4g}"
        f" rk4 p99 {report['jacobian_radius_rk4_1ms']['p99']:.4g} max {report['jacobian_radius_rk4_1ms']['max']:.4g}")
    # Whole episodes of uniform decisions at 2 ms Euler.
    low, high = env.action_space.low, env.action_space.high
    episode_rng = np.random.default_rng(2)
    summaries = []
    for episode in range(episodes):
        env.reset(seed=5000 + episode)
        fastest, growth, nonfinite, intervals = 0.0, 0.0, False, 0
        while not env.done:
            start = env.observe()
            transition = env.step_dt(episode_rng.uniform(low, high), env.sample_duration())
            end = transition["next_state"]
            intervals += 1
            if not np.isfinite(end).all():
                nonfinite = True
                break
            fastest = max(fastest, float(np.abs(end[env.position_dim:]).max()))
            growth = max(growth, float(velocity_growth(env, start[None], end[None])[0]))
        summaries.append({"intervals": intervals, "nonfinite": nonfinite, "max_rate": fastest,
                          "max_velocity_growth": growth, "exploding": bool(nonfinite or fastest > 1e3)})
    report["episodes"] = {
        "count": episodes,
        "nonfinite": int(sum(s["nonfinite"] for s in summaries)),
        "exploding": int(sum(s["exploding"] for s in summaries)),
        "max_rate": float(max(s["max_rate"] for s in summaries)),
        "max_velocity_growth": float(max(s["max_velocity_growth"] for s in summaries)),
        "per_episode": summaries,
    }
    log(f"[episodes] {episodes}: nonfinite {report['episodes']['nonfinite']} exploding {report['episodes']['exploding']} "
        f"max |qdot| {report['episodes']['max_rate']:.3g} max growth {report['episodes']['max_velocity_growth']:.3g}")
    out.write_text(json.dumps(report, indent=1))
    log("wrote", out)


# ------------------------------------------------------------------------------- reach

HOLD = {"cartpole": 1.5, "point_mass": 4.0, "reacher": 1.0}
PUSH_SHARE = 0.8
MISS_LIMIT = 0.01


def biased_source(env, rng, config, family):
    """A decision source for one episode: pushes the limited coordinates toward edges."""
    spec = config["state_spec"]
    low, high = env.action_space.low, env.action_space.high
    m = int(spec["action_dim"])
    actuated = [int(i) for i in spec["actuated_position_indices"]]
    limited = {int(entry[0]) for entry in spec["limited_position_ranges"]}
    hold = HOLD[family]
    signs = np.where(rng.random(m) < 0.5, -1.0, 1.0)
    pushing = np.array([actuated[j] in limited or rng.random() < 0.5 for j in range(m)])
    state = {"next_flip": rng.exponential(hold)}

    def source(time):
        if time >= state["next_flip"]:
            signs[:] = np.where(rng.random(m) < 0.5, -1.0, 1.0)
            pushing[:] = [actuated[j] in limited or rng.random() < 0.5 for j in range(m)]
            state["next_flip"] = time + rng.exponential(hold)
        decision = rng.uniform(low, high)
        if rng.random() < PUSH_SHARE:
            for j in range(m):
                if not pushing[j]:
                    continue
                decision[j] = high[j] if signs[j] > 0 else low[j]
                decision[m + j] = low[m + j] + (high[m + j] - low[m + j]) * rng.uniform(0.7, 1.0)
                decision[2 * m + j] = low[2 * m + j] + (high[2 * m + j] - low[2 * m + j]) * rng.uniform(0.0, 0.1)
        return decision

    return source


def collect_pool(env, config, family, episodes, seed, biased):
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    starts, ends, durations, touched = [], [], [], []
    for episode in range(episodes):
        env.reset(seed=seed + 1000 + episode)
        source = biased_source(env, rng, config, family) if biased else None
        while not env.done:
            decision = source(env.time) if biased else rng.uniform(low, high)
            transition = env.step_dt(decision, env.sample_duration())
            starts.append(transition["state"])
            ends.append(transition["next_state"])
            durations.append(transition["duration"])
            touched.append(bool(transition["limit_touched"]))
    return (np.asarray(starts), np.asarray(ends), np.asarray(durations), np.asarray(touched, bool))


def rule_flags(rule, starts, durations):
    """The start-state rule on every row, vectorized, with its margins [N, n]."""
    margins = R.start_state_margins_numpy(rule, starts[:, list(rule.velocities)], durations[:, None])
    positions = starts[:, list(rule.positions)]
    clear = np.all((positions > np.asarray(rule.lows) + margins)
                   & (positions < np.asarray(rule.highs) - margins), axis=1)
    return ~clear, margins


def reach_main(env_id, out, episodes):
    entry, config = table_config(env_id)
    env = make_env(config, decision_mode="feedback", seed=0)
    spec = config["state_spec"]
    bounds, decay = env.motor_acceleration_bounds(), env.velocity_decay_rates()
    step = float(entry.config["model_step"])
    old = R.contact_exclusion(spec, bounds, active=True, model_step=step)
    new = R.contact_exclusion(spec, bounds, decay_rates=decay, active=True, model_step=step)
    periodic = set(int(i) for i in spec["periodic_position_indices"])
    report = {"env_id": env_id, "episodes_per_pool": episodes, "interval": float(config["reference_period"]),
              "rule_old": dict(old.record()), "rule_new": dict(new.record()),
              "push_share": PUSH_SHARE, "hold": HOLD}
    log(env_id, "rule v2", json.dumps(new.record()))
    verdict = True
    for name, seed, biased in (("uniform", 100, False), ("biased", 500, True)):
        starts, ends, durations, touched = collect_pool(env, config, env_id.split("-")[0], episodes,
                                                        seed, biased)
        flag_old, margin_old = rule_flags(old, starts, durations)
        flag_new, margin_new = rule_flags(new, starts, durations)
        check = np.random.default_rng(0).choice(len(starts), size=min(200, len(starts)), replace=False)
        for row in check:  # the vectorized rule is the stored one
            assert flag_new[row] == R.start_state_flag_numpy(new, starts[row], durations[row])
            assert flag_old[row] == R.start_state_flag_numpy(old, starts[row], durations[row])
        positions = starts[:, list(new.positions)]
        lows, highs = np.asarray(new.lows), np.asarray(new.highs)
        past = np.any((positions < lows) | (positions > highs), axis=1)
        travel = ends[:, list(new.positions)] - positions
        for k, index in enumerate(new.positions):
            if index in periodic:
                travel[:, k] = np.remainder(travel[:, k] + math.pi, 2.0 * math.pi) - math.pi
        travel = np.abs(travel)
        reach_new = margin_new - np.asarray(new.margins)
        reach_old = margin_old - np.asarray(old.margins)
        ratio_new = np.max(travel / np.maximum(reach_new, 1e-300), axis=1)
        speeds = np.abs(starts[:, list(new.velocities)])
        rows = len(starts)
        n_touched = int(touched.sum())
        missed_new = touched & ~flag_new
        missed_old = touched & ~flag_old
        result = {
            "rows": rows,
            "touched_share": n_touched / rows,
            "touched": n_touched,
            "touched_from_inside": int((touched & ~past).sum()),
            "start_past_share": float(past.mean()),
            "flagged_old_share": float(flag_old.mean()),
            "flagged_new_share": float(flag_new.mean()),
            "missed_old": int(missed_old.sum()),
            "missed_new": int(missed_new.sum()),
            "missed_new_share_of_touched": float(missed_new.sum() / max(n_touched, 1)),
            "false_positive_old_share": float((flag_old & ~touched).mean()),
            "false_positive_new_share": float((flag_new & ~touched).mean()),
            "false_positive_new_share_of_flagged": float((flag_new & ~touched).sum() / max(int(flag_new.sum()), 1)),
            "travel_over_reach_new": stats(ratio_new),
            "travel_over_reach_new_unflagged": stats(ratio_new[~flag_new]),
            "travel_over_reach_old": stats(np.max(travel / np.maximum(reach_old, 1e-300), axis=1)),
            "speed": {str(i): stats(speeds[:, k]) for k, i in enumerate(new.velocities)},
            "speed_touched": {str(i): stats(speeds[touched, k]) for k, i in enumerate(new.velocities)},
            "all_rates_touched_max": np.max(np.abs(starts[touched][:, int(spec["position_dim"]):]), axis=0).tolist()
            if n_touched else None,
        }
        # Worst misses: the furthest from the edge relative to the new margin.
        worst = []
        for row in np.nonzero(missed_new)[0]:
            distance_low = positions[row] - lows
            distance_high = highs - positions[row]
            gap = np.minimum(distance_low, distance_high)
            k = int(np.argmin(gap - margin_new[row]))
            toward = 1.0 if distance_high[k] < distance_low[k] else -1.0
            worst.append({
                "row": int(row), "coordinate": int(new.positions[k]), "start": float(positions[row, k]),
                "distance_to_edge": float(gap[k]), "speed_toward_edge": float(toward * starts[row, new.velocities[k]]),
                "margin_new": float(margin_new[row, k]), "margin_old": float(margin_old[row, k]),
                "end": float(ends[row, new.positions[k]]), "duration": float(durations[row]),
                "rates": starts[row, int(spec["position_dim"]):].tolist(),
            })
        worst.sort(key=lambda item: item["margin_new"] - item["distance_to_edge"])
        result["worst_misses"] = worst[:10]
        report[name] = result
        ok = result["missed_new_share_of_touched"] < MISS_LIMIT
        verdict &= ok
        log(f"REACH {env_id} [{name}] rows {rows} touched {result['touched_share']:.4f} "
            f"(from inside {result['touched_from_inside']}) flagged old {result['flagged_old_share']:.4f} "
            f"new {result['flagged_new_share']:.4f} missed old {result['missed_old']} new {result['missed_new']} "
            f"({result['missed_new_share_of_touched']:.4%} of touched) false-pos old "
            f"{result['false_positive_old_share']:.4f} new {result['false_positive_new_share']:.4f} "
            f"travel/reach max {result['travel_over_reach_new']['max']:.3f} "
            f"{'PASS' if ok else 'FAIL'}")
        for item in worst[:5]:
            log("   miss", json.dumps(item))
    report["verdict"] = "PASS" if verdict else "FAIL"
    out.write_text(json.dumps(report, indent=1))
    log("REACH_VERDICT", env_id, report["verdict"], "wrote", out)


def main(argv):
    if len(argv) < 3 or argv[0] not in ("contact", "swimmer", "reach"):
        raise SystemExit(__doc__)
    mode, env_id, out = argv[0], argv[1], Path(argv[2])
    if mode == "reach":
        reach_main(env_id, out, int(argv[3]) if len(argv) > 3 else 20)
        return
    count = int(argv[3]) if len(argv) > 3 else 2000
    if mode == "contact":
        contact_main(env_id, out, count)
    else:
        swimmer_main(env_id, out, count, int(argv[4]) if len(argv) > 4 else 20)


if __name__ == "__main__":
    main(sys.argv[1:])
