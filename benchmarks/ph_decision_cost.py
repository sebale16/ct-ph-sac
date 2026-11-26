"""Per-decision training cost of one arm on one table row, cluster-comparable.

Usage:
    python benchmarks/ph_decision_cost.py ENV_ID ARM OUT_DIR [--plant-step-every K]
        [--plant-step-dense-until D] [--threads N] [--old] [--half-batch]

The row's config (benchmarks/hyperparams/ct_ph_sac.csv, mode top) with the row's batch
size, rate rollout paths and model points (they set the cost), replay 20000, one warm-up
episode and warm fits of 50 steps (timing only), written to OUT_DIR/<tag>.json and loaded
through benchmarks.run_ph_sac.load_protocol (the --config path).

Plant-step schedule (training plant_step_dense_until / plant_step_every,
common.ph_config.plant_step_schedule): the script REPLACES the row's schedule by the
flags, so the default (--plant-step-every 1) times the dense plant fit, a step on every
row, whatever the row says; --plant-step-every 4 times one step every 4th row from
decision 0 (--plant-step-dense-until 0, the default), which is the table's schedule after
its dense phase. Dense vs every-4 on one row (ARM is any arm of protocol.all_arms: the
paper arms with torque decisions, e.g. implicit_torque, or an ablation arm such as the
impedance-law implicit_learned_admissions):
    python benchmarks/ph_decision_cost.py cartpole-swingup implicit_torque out/timing
    python benchmarks/ph_decision_cost.py cartpole-swingup implicit_torque out/timing --plant-step-every 4
The tag (file names) gets __every<K> (and __dense<D>) when the schedule is not dense.

--old forces the convention of the Vista runs before 2026-11-09: model_step 0.001,
model_integrator semi_implicit and top-level refresh_mode held (and no contact
substepping). --half-batch halves the batch (only after a run at full size exceeded a
memory cap). --no-contact drops the row's model_coarse_step (contact-triggered fine
substepping, since 2026-11-10), so every interval integrates at model_step, as before;
the tag gets __nocontact. With the scheme on, the result records, per consumer (plant fit
endpoint moments, rollouts, Jacobian norms), the rows integrated and the share the
detector sent to the fine step over the timed window (contact_rows, contact_fraction;
models/ph_sac/rollouts.contact_statistics).

Timing: torch --threads threads (default 2; also set OMP_NUM_THREADS/MKL_NUM_THREADS to
match). Interaction stamps from the learn callback. warm = decisions of one episode; the
warm fit runs at interaction warm + 1. "inner" = mean wall time per decision over
interactions warm+10 .. warm+310 (300 decisions). Implicit arms run on to warm + 420, so
the first outer iteration's burst (rate-head and actor steps, temperature, at inner step
K = 400) is inside: "burst" = the largest stamp gap after warm + 390 less the inner mean,
"amortized" = inner + burst / K. Peak memory: getrusage ru_maxrss of the process (MB).
The result (OUT_DIR/<tag>.result.json, also printed) records the schedule and the plant
gradient steps taken.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from benchmarks.ph_hyperparams import load_ph_hyperparams_from_table  # noqa: E402
from benchmarks.run_ph_sac import load_protocol  # noqa: E402
from environment.ph_sac.tasks import make_env  # noqa: E402
from evaluations.ph_sac import protocol as P  # noqa: E402
from models.ph_sac.rollouts import contact_statistics  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("env_id")
    parser.add_argument("arm", choices=[arm["name"] for arm in P.all_arms()])
    parser.add_argument("out", type=Path)
    parser.add_argument("--plant-step-every", type=int, default=1)
    parser.add_argument("--plant-step-dense-until", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--half-batch", action="store_true")
    parser.add_argument("--no-contact", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.env_id}__{args.arm}"
    if args.plant_step_every != 1:
        tag += f"__every{args.plant_step_every}"
        if args.plant_step_dense_until:
            tag += f"__dense{args.plant_step_dense_until}"
    tag += ("__old" if args.old else "") + ("__half" if args.half_batch else "")
    tag += "__nocontact" if args.no_contact else ""
    raw = copy.deepcopy(load_ph_hyperparams_from_table(args.env_id, "top"))
    if args.no_contact or args.old:
        raw.pop("model_coarse_step", None)
    raw["training"].update(
        replay_capacity=20000,
        warmup_episodes=1,
        plant_step_dense_until=args.plant_step_dense_until,
        plant_step_every=args.plant_step_every,
    )
    if args.half_batch:
        raw["training"]["batch_size"] //= 2
    raw["implicit"].update(reward_warm_steps=50, plant_warm_steps=50)
    raw["protocol"].update(
        training_seeds=[0], evaluation_reset_seeds=[20000], output_directory=str(out / "run")
    )
    if args.old:
        raw.update(model_step=0.001, model_integrator="semi_implicit", refresh_mode="held")
    config_path = out / f"{tag}.json"
    config_path.write_text(json.dumps(raw, indent=2))
    proto = load_protocol(str(config_path))
    proto["arms"] = list(P.all_arms())
    run = next(r for r in P.build_run_matrix(proto) if r["arm"]["name"] == args.arm)
    testbed = proto["matched"]["testbed_config"]
    env = make_env(testbed, decision_mode=run["arm"]["decision_mode"], seed=0)
    agent = P.make_training_agent(run, env)
    warm = math.ceil(round(testbed["episode_duration"] / testbed["reference_period"], 9))
    implicit = args.arm.startswith("implicit")
    K = int(proto["implicit_config"]["solver"]["inner_steps"])
    total = warm + (420 if implicit else 310)
    stamps = {}
    a, b = warm + 10, warm + 310
    contact_marks = {}

    def callback(_, metrics):
        interactions = int(metrics["interactions"])
        stamps[interactions] = time.perf_counter()
        if interactions in (a, b):
            contact_marks[interactions] = contact_statistics()

    t0 = time.perf_counter()
    agent.learn(total, callback=callback)
    wall = time.perf_counter() - t0
    inner = (stamps[b] - stamps[a]) / (b - a)
    counters = agent.counters()
    result = {
        "env_id": args.env_id, "arm": args.arm, "old_convention": args.old,
        "half_batch": args.half_batch,
        "plant_step_dense_until": args.plant_step_dense_until,
        "plant_step_every": args.plant_step_every,
        "plant_gradient_steps": int(counters["plant_gradient_steps"]),
        "batch_size": raw["training"]["batch_size"],
        "rate_rollout_paths": raw["implicit"]["rate"]["rollout_paths"],
        "model_points": raw["implicit"]["rate"]["model_points"],
        "explicit_rollout_paths": raw["explicit"]["rollout_paths"],
        "model_step": proto["implicit_config"]["model_step"],
        "model_integrator": raw.get("model_integrator", "rk4"),
        "model_coarse_step": raw.get("model_coarse_step"),
        "contact_active": getattr(agent, "contact", None) is not None,
        "refresh_mode": testbed["controller"].get("refresh_mode"),
        "warm_decisions": warm, "warm_seconds": stamps[warm] - t0,
        "warm_fit_seconds": stamps[warm + 1] - stamps[warm],
        "inner_ms_per_decision": 1e3 * inner, "wall_seconds": wall,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "threads": torch.get_num_threads(),
    }
    if implicit:
        gaps = {
            k: stamps[k] - stamps[k - 1]
            for k in range(warm + 391, total + 1)
            if k in stamps and k - 1 in stamps
        }
        at = max(gaps, key=gaps.get)
        burst = gaps[at] - inner
        result.update(
            burst_seconds=burst, burst_at=at - warm, inner_steps_K=K,
            amortized_ms_per_decision=1e3 * (inner + burst / K),
            outer_iterations=int(getattr(agent, "outer_iteration", -1)),
        )
    else:
        result["amortized_ms_per_decision"] = 1e3 * inner
    if a in contact_marks and b in contact_marks:
        rows, fractions = {}, {}
        for consumer, (total_rows, fine_rows) in contact_marks[b].items():
            before = contact_marks[a].get(consumer, (0, 0))
            counted = total_rows - before[0]
            rows[consumer] = [counted, fine_rows - before[1]]
            fractions[consumer] = (fine_rows - before[1]) / counted if counted else None
        result["contact_rows"] = rows
        result["contact_fraction"] = fractions
    (out / f"{tag}.result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
