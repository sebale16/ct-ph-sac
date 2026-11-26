"""Dense vs scheduled plant fit (and contact exclusion): learned drift error and return.

Two modes.

1. Quick comparison (the 2026-11-10 schedule check; unchanged):
    python benchmarks/ph_plant_schedule_drift.py ENV_ID SCHEDULE OUT_DIR [--budget N]
        [--arm ARM] [--threads T]

SCHEDULE is "dense" (a plant step on every row) or "D,k" (every row through decision D,
then every k-th row: training plant_step_dense_until, plant_step_every). The row's config
(benchmarks/hyperparams/ct_ph_sac.csv, mode top) with its batch size and warm fits
(2000/2000), replay 20000, one warm-up episode and --budget decisions (default 4000); the
plant diagnostics are forced every 250 decisions (the run record's oracle[].drift_error_max)
and, on the same schedule, a relative drift error is taken on a fixed probe: the first 2048
training rows, read without drawing from any stream, velocity block, rms over rows of
|b_phi - b| over rms |b| (b the true drift). A final evaluation on 8 reset seeds. Writes
OUT_DIR/<env>__<schedule>.summary.json and prints one line per diagnostic:
decisions, drift_error_max, relative rms error, plant steps so far.

    python benchmarks/ph_plant_schedule_drift.py lqr-lqr_2_1 dense out/plant_schedule
    python benchmarks/ph_plant_schedule_drift.py cartpole-swingup 1500,4 out/plant_schedule

2. Experiment (--experiment; the contact-exclusion and thinning experiment of 2026-11-10):
    python benchmarks/ph_plant_schedule_drift.py ENV_ID SCHEDULE OUT_DIR --experiment
        [--contact-handling port|exclude] [--seed S] [--arm ARM] [--budget N]
        [--probe-every 5000] [--plant-diagnostics-every 5000] [--threads T]
        [--resume [--resume-every N] [--stop-after-interactions N]]
        [--tiny] [--force-contact P]

The row's full table config (replay, warm-up, warm fits, evaluation cadence and seeds),
--budget decisions (default 200000), one seed, the plant schedule SCHEDULE, the plant
diagnostics (oracle[].drift_error_max) every --plant-diagnostics-every decisions through
the config (training plant_diagnostics_every), and contact_handling (exclude: no limit
port, common.ph_config.CONTACT_HANDLINGS; port: the limit port, the row as it was on
2026-11-10, i.e. a row under exclusion gets limit_port back). The run is
run_ph_sac.py's (evaluations/ph_sac/protocol.run_seed) in OUT_DIR/<tag>/<testbed>/<arm>/
seed_<s>, tag = <env>__<arm>__<handling>__<schedule>; with --resume it is restartable
exactly as run_ph_sac.py --resume (SIGUSR1/SIGTERM: resume.pt written, exit code 75; the
same command continues it; a finished run is skipped).

Every --probe-every decisions the run appends one JSON line to
<run folder>/exclusion_diagnostics.jsonl (draws from no stream of the run):
  interactions, plant_gradient_steps, the last oracle record's drift_error_max;
  probe errors of the online plant against the oracle drift (velocity block, relative rms
    |b_phi - b| / |b| over rows) on three fixed probe sets of 1024 states each, built from
    a separate seeded environment with uniform decisions: "free" (states the start-state
    rule leaves unflagged), "near" (a limited coordinate within 2% of the range inside an
    edge, moving toward it), "past" (up to 0.5% of the range past an edge); tasks without
    limited coordinates have "free" only;
  replay counts over every stored row (train and gate partitions): rows, limit_touched,
    near_limit (start-state rule), flagged (either), touched_and_near (touched rows the
    start-state rule caught); differences between lines give per-window shares;
  timing: perf_counter, cumulative evaluation and diagnostic seconds of this process and
    a process id, so a window inside one process gives ms per training decision.
A restart drops the lines at or past its checkpoint and writes them again. The summary
command (numpy only) is benchmarks/ph_plant_schedule_summary.py.

--tiny shrinks the run for smoke tests (small networks and batches, 1 s episodes, one
warm-up episode, short solves); --force-contact P starts each training episode with
probability P with a limited coordinate within 1% of the range of an edge, moving
toward it (drawn from the environment's reset stream, so a resumed run is exact).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DIAGNOSTICS_EVERY = 250
PROBE_ROWS = 2048
PROBE_STATES = 1024
PROBE_SEED = 7_654_321
DIAGNOSTICS_NAME = "exclusion_diagnostics.jsonl"
TINY_EPISODE_SECONDS = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("env_id")
    parser.add_argument("schedule", help='"dense" or "D,k"')
    parser.add_argument("out", type=Path)
    parser.add_argument("--budget", type=int, default=None)
    # The default stays the arm of the 2026-11-10 experiment, whose runs a command
    # without --arm continues (the tag holds the arm); the paper's implicit arm since
    # 2026-11-13 is implicit_torque (any arm of protocol.all_arms is accepted).
    parser.add_argument(
        "--arm",
        default="implicit_learned_admissions",
        help="Arm (default implicit_learned_admissions; the paper's implicit arm: implicit_torque).",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--contact-handling", choices=("port", "exclude"), default="port")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-every", type=int, default=5000)
    parser.add_argument("--plant-diagnostics-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-every", type=int, default=10_000)
    parser.add_argument("--stop-after-interactions", type=int, default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--force-contact", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.budget is None:
        args.budget = 200_000 if args.experiment else 4000
    if args.stop_after_interactions is not None and not args.resume:
        parser.error("--stop-after-interactions needs --resume.")
    return args


def _schedule_tag(schedule: str) -> str:
    return schedule.replace(",", "_")


def _apply_schedule(training: dict, schedule: str) -> None:
    for key in ("plant_step_dense_until", "plant_step_every"):
        training.pop(key, None)
    if schedule != "dense":
        dense_until, every = (int(x) for x in schedule.split(","))
        training.update(plant_step_dense_until=dense_until, plant_step_every=every)


def relative_drift_error(states, actions, learned, oracle, position_dim):
    """Velocity block: rms over rows of |b_phi - b| over rms |b| (nan without rows)."""
    import torch

    if states.shape[0] == 0:
        return float("nan")
    with torch.no_grad():
        error = (learned.drift(states, actions) - oracle.drift(states, actions))[:, position_dim:]
        scale = oracle.drift(states, actions)[:, position_dim:]
    numerator = error.square().sum(-1).mean().sqrt()
    denominator = scale.square().sum(-1).mean().sqrt()
    return float(numerator / denominator)


# ----------------------------------------------------------------------------- legacy


def legacy_main(args):
    import numpy as np
    import torch

    from benchmarks.ph_hyperparams import load_ph_hyperparams_from_table
    from benchmarks.run_ph_sac import protocol_from_config
    from environment.ph_sac.tasks import make_oracle_plant
    from evaluations.ph_sac import protocol as P
    from unittest import mock

    if args.threads:
        torch.set_num_threads(args.threads)
    tag = f"{args.env_id}__{args.arm}__{_schedule_tag(args.schedule)}"
    raw = copy.deepcopy(load_ph_hyperparams_from_table(args.env_id, "top"))
    training = raw["training"]
    training.update(replay_capacity=20000, warmup_episodes=1)
    _apply_schedule(training, args.schedule)
    raw["protocol"].update(
        interaction_budget=args.budget,
        training_seeds=[0],
        evaluation_reset_seeds=list(range(20000, 20008)),
        evaluation_every_interactions=args.budget,
        output_directory=str(args.out / tag),
    )
    proto = protocol_from_config(raw, tag)
    proto["arms"] = [arm for arm in P.all_arms() if arm["name"] == args.arm]
    (run,) = P.build_run_matrix(proto)
    relative = []
    oracle = {}

    def probe(agent):
        buffer = agent.replay.buffers["train"]
        rows = min(PROBE_ROWS, buffer.size())
        dtype = torch.get_default_dtype()
        states = torch.as_tensor(buffer._arrays["states"][:rows], dtype=dtype)
        decisions = torch.as_tensor(buffer._arrays["decisions"][:rows], dtype=dtype)
        with torch.no_grad():
            resolved = agent.controller.resolve_decision(states, decisions)
            actions = agent.controller.action(states, resolved)
        true = oracle.setdefault("plant", make_oracle_plant(agent.env))
        return relative_drift_error(
            states, actions, agent.online_plant, true, agent.env.position_dim
        )

    build = P.make_training_agent

    def capture(spec, env):
        agent = build(spec, env)
        agent.plant_diagnostics_every = DIAGNOSTICS_EVERY
        original = agent.record_plant_diagnostics

        def record(batch):
            original(batch)
            relative.append(
                {
                    "interactions": agent.interactions,
                    "relative_drift_rms": probe(agent),
                    "plant_steps": int(agent._counters["plant_gradient_steps"]),
                }
            )

        agent.record_plant_diagnostics = record
        return agent

    start = time.time()
    with mock.patch.object(P, "make_training_agent", capture):
        result = P.run_seed(run)
    final = result["evaluations"][-1]
    summary = {
        "env_id": args.env_id,
        "arm": args.arm,
        "schedule": args.schedule,
        "budget": args.budget,
        "batch_size": training["batch_size"],
        "threads": torch.get_num_threads(),
        "wall_seconds": time.time() - start,
        "plant_gradient_steps": result["counters"]["plant_gradient_steps"],
        "final_interactions": final["interactions"],
        "final_discounted_return_mean": float(
            np.mean([e["discounted_return"] for e in final["episodes"]])
        ),
        "final_episode_return_mean": float(
            np.mean([e["episode_return"] for e in final["episodes"]])
        ),
        "oracle": [
            {"interactions": r["interactions"], "drift_error_max": r["drift_error_max"]}
            for r in result["oracle"]
        ],
        "relative": relative,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("oracle", "relative")}))
    for logged, measured in zip(summary["oracle"], relative):
        print(
            logged["interactions"],
            f"{logged['drift_error_max']:.4g}",
            f"{measured['relative_drift_rms']:.4g}",
            measured["plant_steps"],
        )


# ------------------------------------------------------------------------- experiment


def experiment_raw_config(args) -> dict:
    """The row's config with the experiment's settings (see the module docstring)."""
    from benchmarks.ph_hyperparams import load_ph_hyperparams_from_table

    raw = copy.deepcopy(load_ph_hyperparams_from_table(args.env_id, "top"))
    training = raw["training"]
    _apply_schedule(training, args.schedule)
    training["plant_diagnostics_every"] = int(args.plant_diagnostics_every)
    if args.contact_handling == "exclude":
        raw["contact_handling"] = "exclude"
        raw["network"].pop("limit_port", None)
    elif raw.pop("contact_handling", "port") == "exclude":
        # A row under contact exclusion (the cart-poles and point mass since 2026-11-11)
        # back to the limit port, as the rows were when this experiment ran.
        raw["network"]["limit_port"] = True
    tag = (
        f"{args.env_id}__{args.arm}__{args.contact_handling}__{_schedule_tag(args.schedule)}"
    )
    raw["protocol"].update(
        interaction_budget=int(args.budget),
        training_seeds=[int(args.seed)],
        output_directory=str(args.out / tag),
    )
    if args.tiny:
        raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
        training.update(batch_size=16, replay_capacity=4000, warmup_episodes=1)
        raw["explicit"].update(rollout_paths=2)
        implicit = raw["implicit"]
        implicit.update(reward_warm_steps=5, plant_warm_steps=5, temperature_every_rows=4)
        implicit["solver"].update(inner_steps=8, policy_draws=2)
        implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
        raw["protocol"].update(evaluation_reset_seeds=[3], evaluation_every_interactions=200)
    return raw


def experiment_run(args):
    """The RunSpec of the experiment's one arm and seed."""
    from benchmarks.run_ph_sac import protocol_from_config
    from evaluations.ph_sac import protocol as P

    raw = experiment_raw_config(args)
    tag = Path(raw["protocol"]["output_directory"]).name
    proto = protocol_from_config(raw, tag)
    if args.tiny:
        for testbed in (
            proto["matched"]["testbed_config"],
            proto["explicit_config"]["testbed"],
            proto["implicit_config"]["testbed"],
        ):
            testbed["episode_duration"] = TINY_EPISODE_SECONDS
    proto["arms"] = [arm for arm in P.all_arms() if arm["name"] == args.arm]
    if not proto["arms"]:
        raise ValueError(f"Unknown arm {args.arm!r}.")
    P.validate_matched_quantities(proto)
    (run,) = P.build_run_matrix(proto)
    return run


def force_contact(env, probability: float) -> None:
    """Start an episode near a limit with this probability (the env's reset stream)."""
    import numpy as np

    ranges = tuple(env.state_spec.get("limited_position_ranges", ()))
    if not ranges or probability <= 0.0:
        return
    projection = np.asarray(env.state_spec["observation_projection"])
    original = env._reset_state

    def reset_state(options=None):
        original(options)
        if options and "state" in options:
            return
        draw = env._reset_rng.random(5)
        if draw[0] >= probability:
            return
        state = env.observe()
        index, low, high = ranges[int(draw[1] * len(ranges)) % len(ranges)]
        side = 1.0 if draw[2] < 0.5 else -1.0
        edge = high if side > 0 else low
        state[int(index)] = edge - side * 0.01 * (high - low) * draw[3]
        rate = env.position_dim + int(np.argmax(np.abs(projection[int(index)])))
        state[rate] = side * (0.2 + 2.0 * draw[4])
        env._set_state(state)

    env._reset_state = reset_state


def probe_sets(env, exclusion, rng_seed: int = PROBE_SEED):
    """Fixed probe states and decisions: {"free"/"near"/"past": (states, decisions)}.

    From a separate environment (the training env's config, its own seed) run with
    uniform decisions from resets; "free" keeps the visited states the start-state rule
    leaves unflagged; "near" and "past" move one limited coordinate of visited states
    within 2% of the range inside an edge (moving toward it at up to the pool's 90th
    percentile speed of that coordinate) or up to 0.5% past it.
    """
    import numpy as np

    from environment.ph_sac.tasks import make_env
    from models.ph_sac.rollouts import start_state_flag_numpy

    rng = np.random.default_rng(rng_seed)
    probe_env = make_env(env.config, decision_mode=env.decision_mode, seed=rng_seed)
    low = probe_env.action_space.low
    high = probe_env.action_space.high
    pool = []
    state, _ = probe_env.reset(seed=rng_seed)
    while len(pool) < 8 * PROBE_STATES:
        pool.append(state.copy())
        transition = probe_env.step_dt(rng.uniform(low, high), probe_env.sample_duration())
        state = transition["next_state"]
        if transition["terminated"] or transition["truncated"] or len(pool) % 500 == 0:
            state, _ = probe_env.reset()
    pool = np.asarray(pool)
    duration = float(env.config["clock"]["max_duration"])
    sets = {}
    if exclusion is not None:
        free = np.array(
            [not start_state_flag_numpy(exclusion, s, duration) for s in pool], dtype=bool
        )
        candidates = pool[free]
    else:
        candidates = pool
    chosen = candidates[rng.choice(len(candidates), size=min(PROBE_STATES, len(candidates)), replace=False)]
    sets["free"] = (chosen, rng.uniform(low, high, size=(chosen.shape[0], low.size)))
    if exclusion is not None:
        for name, inside, past in (("near", 0.02, 0.0), ("past", 0.0, 0.005)):
            rows = pool[rng.choice(len(pool), size=PROBE_STATES, replace=False)].copy()
            which = rng.integers(len(exclusion.positions), size=PROBE_STATES)
            sides = np.where(rng.random(PROBE_STATES) < 0.5, -1.0, 1.0)
            for row, k, side in zip(rows, which, sides):
                low_k, high_k = exclusion.lows[k], exclusion.highs[k]
                span = high_k - low_k
                edge = high_k if side > 0 else low_k
                if name == "near":
                    row[exclusion.positions[k]] = edge - side * span * inside * rng.random()
                else:
                    row[exclusion.positions[k]] = edge + side * span * past * rng.random()
                speeds = np.abs(pool[:, exclusion.velocities[k]])
                row[exclusion.velocities[k]] = side * rng.uniform(0.0, max(np.quantile(speeds, 0.9), 1e-3))
            sets[name] = (rows, rng.uniform(low, high, size=(rows.shape[0], low.size)))
    probe_env.close()
    return sets


class ExperimentDiagnostics:
    """Writes the experiment's JSON lines (module docstring); reads no training stream."""

    def __init__(self, path: Path, every: int):
        self.path = Path(path)
        self.every = int(every)
        self.process = f"{os.uname().nodename}:{os.getpid()}:{time.time():.0f}"
        self.evaluation_seconds = 0.0
        self.diagnostic_seconds = 0.0
        self.sets = None
        self.oracle = None
        self.truncated = False

    def _truncate(self, interactions: int) -> None:
        """Drop lines at or past interactions (written after the checkpoint we resume)."""
        if not self.path.exists():
            return
        kept = []
        for line in self.path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(entry.get("interactions", 0)) < interactions:
                kept.append(line)
        self.path.write_text("".join(f"{line}\n" for line in kept))

    def attach(self, agent) -> None:
        import torch

        from environment.ph_sac.tasks import make_oracle_plant

        original = agent._after_interaction
        diagnostics = self

        def after_interaction():
            original()
            if not diagnostics.truncated:
                diagnostics.truncated = True
                diagnostics._truncate(agent.interactions)
                # This process's first row: the anchor of its timing windows.
                diagnostics.write(agent, event="resume" if agent.interactions > 1 else "start")
            if agent.interactions % diagnostics.every == 0:
                started = time.perf_counter()
                if diagnostics.sets is None:
                    with torch.random.fork_rng(devices=[]):
                        diagnostics.sets = probe_sets(agent.env, agent.exclusion)
                        diagnostics.oracle = make_oracle_plant(agent.env)
                # The probe sets' construction counts before this line's clock reading.
                diagnostics.diagnostic_seconds += time.perf_counter() - started
                started = time.perf_counter()
                diagnostics.write(agent, event="probe")
                diagnostics.diagnostic_seconds += time.perf_counter() - started

        agent._after_interaction = after_interaction

    def write(self, agent, *, event: str) -> None:
        import numpy as np
        import torch

        entry = {
            "event": event,
            "interactions": int(agent.interactions),
            "plant_gradient_steps": int(agent._counters["plant_gradient_steps"]),
            "process": self.process,
            "perf_counter": time.perf_counter(),
            "evaluation_seconds": self.evaluation_seconds,
            "diagnostic_seconds": self.diagnostic_seconds,
            "time": time.time(),
        }
        if agent.oracle_records and agent.oracle_records[-1]["interactions"] == agent.interactions:
            record = agent.oracle_records[-1]
            entry["drift_error_max"] = float(record["drift_error_max"])
            if "contact_flagged_share" in record:
                entry["oracle_flagged_share"] = float(record["contact_flagged_share"])
        counts = {"rows": 0, "limit_touched": 0, "near_limit": 0, "flagged": 0, "touched_and_near": 0}
        for buffer in agent.replay.buffers.values():
            columns = buffer.contact_columns()
            touched, near = columns["limit_touched"], columns["near_limit"]
            counts["rows"] += int(touched.size)
            counts["limit_touched"] += int(touched.sum())
            counts["near_limit"] += int(near.sum())
            counts["flagged"] += int((touched | near).sum())
            counts["touched_and_near"] += int((touched & near).sum())
        entry["replay"] = counts
        if event == "probe" and self.sets is not None and agent.online_plant is not None:
            dtype = torch.get_default_dtype()
            errors = {}
            for name, (states, decisions) in self.sets.items():
                s = torch.as_tensor(states, dtype=dtype)
                d = torch.as_tensor(decisions, dtype=dtype)
                with torch.no_grad():
                    actions = agent.controller.action(s, agent.controller.resolve_decision(s, d))
                errors[name] = relative_drift_error(
                    s, actions, agent.online_plant, self.oracle, agent.env.position_dim
                )
            entry["probe_relative_rms"] = errors
            entry["probe_rows"] = {name: int(v[0].shape[0]) for name, v in self.sets.items()}
        if np.isfinite(entry["perf_counter"]):
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def experiment_main(args) -> int:
    import torch
    from unittest import mock

    from evaluations.ph_sac import protocol as P
    from evaluations.ph_sac import resume as R

    if args.threads:
        torch.set_num_threads(args.threads)
    run = experiment_run(args)
    folder = P.run_output_directory(run)
    folder.mkdir(parents=True, exist_ok=True)
    diagnostics = ExperimentDiagnostics(folder / DIAGNOSTICS_NAME, args.probe_every)
    build = P.make_training_agent
    evaluate = P.evaluate_fixed_seeds

    def capture(spec, env):
        if args.force_contact > 0.0:
            force_contact(env, args.force_contact)
        agent = build(spec, env)
        diagnostics.attach(agent)
        if agent.exclusion is not None:
            print("exclusion rule:", json.dumps(agent.exclusion.record()), flush=True)
        print("model_step:", agent.model_step, "contact substepping:",
              None if agent.contact is None else agent.contact.record(), flush=True)
        return agent

    def timed_evaluate(*a, **k):
        started = time.perf_counter()
        try:
            return evaluate(*a, **k)
        finally:
            diagnostics.evaluation_seconds += time.perf_counter() - started

    resume = None
    if args.resume:
        resume = R.resume_options(args.resume_every, args.stop_after_interactions)
    started = time.time()
    try:
        with R.stop_on_signals(), mock.patch.object(P, "make_training_agent", capture), \
                mock.patch.object(P, "evaluate_fixed_seeds", timed_evaluate):
            result = P.run_seed(run, resume)
    except R.ResumableStop as stop:
        print(f"Stopped for a restart: {stop}", file=sys.stderr, flush=True)
        return R.RESUMABLE_EXIT_CODE
    final = result["evaluations"][-1]
    episodes = final["episodes"]
    print(json.dumps({
        "run": run["run_id"],
        "folder": str(folder),
        "interactions": int(result["counters"]["interactions"]),
        "plant_gradient_steps": int(result["counters"]["plant_gradient_steps"]),
        "final_episode_return_mean": float(sum(e["episode_return"] for e in episodes) / len(episodes)),
        "wall_seconds_this_process": time.time() - started,
        "contact_handling": result.get("contact_handling"),
    }), flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.experiment:
        sys.exit(experiment_main(args))
    legacy_main(args)


if __name__ == "__main__":
    main()
