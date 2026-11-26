"""Summary of the contact-exclusion and thinning experiment (numpy only; seconds).

    python benchmarks/ph_plant_schedule_summary.py ROOT [--points 20000,50000,100000,150000,200000]

ROOT holds the runs of benchmarks/ph_plant_schedule_drift.py --experiment:
ROOT/<env>__<arm>__<handling>__<schedule>/<testbed>/<arm>/seed_<s>/ with
exclusion_diagnostics.jsonl, run_record.json (written at each evaluation and at a stop)
and resume_complete.json once finished. Runs in progress are summarized as far as they
got. Prints, per task x arm x handling x schedule (mean +- sd over seeds, n seeds):
  1. progress: decisions reached per seed, finished seeds;
  2. plant error: relative rms drift error of the online plant on the fixed probes
     (free / near / past limits) and the logged oracle drift_error_max, at --points;
  3. evaluation: mean undiscounted episode return over the evaluation episodes, at every
     evaluation;
  4. contact: share of collected rows flagged (start-state rule or touched), touched, and
     the share of touched rows the start-state rule caught, per 10,000-decision window and
     overall;
  5. cost: ms per training decision (evaluation and diagnostics excluded) per
     10,000-decision window, plant gradient steps;
  6. thinning: change of the probe errors from 100,000 to 200,000 decisions, dense vs
     thinned (D,k) at each handling, with the plant steps taken in between.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW = 10_000


def read_lines(path: Path) -> list:
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def read_record(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["result"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def collect(root: Path) -> dict:
    """{(env, arm, handling, schedule): {seed: {"lines", "record", "complete"}}}."""
    runs = defaultdict(dict)
    for folder in sorted(root.glob("*/*/*/seed_*")):
        tag = folder.parents[2].name
        parts = tag.split("__")
        if len(parts) != 4:
            continue
        seed = int(folder.name.split("_", 1)[1])
        runs[tuple(parts)][seed] = {
            "lines": read_lines(folder / "exclusion_diagnostics.jsonl"),
            "record": read_record(folder / "run_record.json"),
            "complete": (folder / "resume_complete.json").exists(),
        }
    return runs


def fmt(values, digits=3) -> str:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "-"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.{digits}g} (1)"
    return f"{mean:.{digits}g}+-{float(np.std(values, ddof=1)):.2g} ({len(values)})"


def probe_at(lines, interactions):
    for entry in lines:
        if entry.get("event") == "probe" and entry["interactions"] == interactions:
            return entry
    return None


def latest(lines):
    return max((e["interactions"] for e in lines), default=0)


def windows(lines):
    """Per 10k window: flagged, touched, caught shares; ms per decision (same process)."""
    probes = sorted(
        (e for e in lines if e.get("event") in ("probe", "start", "resume")),
        key=lambda e: (e["interactions"], e["perf_counter"]),
    )
    contact = {}
    by_point = {e["interactions"]: e for e in probes if e.get("event") == "probe"}
    for end in sorted(by_point):
        if end % WINDOW:
            continue
        start = end - WINDOW
        if start not in by_point and start != 0:
            continue
        before = by_point[start]["replay"] if start else {k: 0 for k in by_point[end]["replay"]}
        after = by_point[end]["replay"]
        rows = after["rows"] - before["rows"]
        if rows <= 0:
            continue
        touched = after["limit_touched"] - before["limit_touched"]
        caught = after["touched_and_near"] - before["touched_and_near"]
        contact[end] = {
            "flagged": (after["flagged"] - before["flagged"]) / rows,
            "touched": touched / rows,
            "caught": caught / touched if touched else float("nan"),
        }
    cost = defaultdict(lambda: [0.0, 0])
    for left, right in zip(probes, probes[1:]):
        if left["process"] != right["process"]:
            continue
        decisions = right["interactions"] - left["interactions"]
        if decisions <= 0:
            continue
        seconds = (
            (right["perf_counter"] - left["perf_counter"])
            - (right["evaluation_seconds"] - left["evaluation_seconds"])
            - (right["diagnostic_seconds"] - left["diagnostic_seconds"])
        )
        window = (right["interactions"] + WINDOW - 1) // WINDOW * WINDOW
        cost[window][0] += seconds
        cost[window][1] += decisions
    ms = {w: 1000.0 * s / n for w, (s, n) in cost.items() if n > 0}
    return contact, ms


def evaluation_means(record):
    if record is None:
        return {}
    means = {}
    for checkpoint in record.get("evaluations", []):
        episodes = checkpoint.get("episodes", [])
        values = [float(e.get("episode_return", float("nan"))) for e in episodes]
        if values:
            means[int(checkpoint["interactions"])] = float(np.nanmean(values))
    return means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--points", default="20000,50000,100000,150000,200000")
    args = parser.parse_args(argv)
    points = [int(p) for p in args.points.split(",")]
    runs = collect(args.root)
    if not runs:
        print(f"no runs under {args.root}")
        return
    keys = sorted(runs)

    print("== 1. progress (decisions per seed; * finished)")
    for key in keys:
        seeds = runs[key]
        reached = " ".join(
            f"s{s}:{latest(v['lines'])}{'*' if v['complete'] else ''}" for s, v in sorted(seeds.items())
        )
        print(f"{' '.join(key):<75} {reached}")

    print("\n== 2. plant error: probe relative rms (free / near / past) and oracle drift_error_max")
    for key in keys:
        seeds = runs[key]
        print(f"-- {' '.join(key)}")
        for point in points:
            entries = [probe_at(v["lines"], point) for v in seeds.values()]
            entries = [e for e in entries if e is not None]
            if not entries:
                continue
            cells = []
            for name in ("free", "near", "past"):
                values = [e.get("probe_relative_rms", {}).get(name) for e in entries]
                if any(v is not None for v in values):
                    cells.append(f"{name} {fmt(values)}")
            cells.append(f"drift_error_max {fmt([e.get('drift_error_max') for e in entries])}")
            cells.append(f"plant steps {fmt([e['plant_gradient_steps'] for e in entries], 6)}")
            print(f"   {point:>7}: " + " | ".join(cells))

    print("\n== 3. evaluation: mean episode return (undiscounted) per evaluation")
    for key in keys:
        per_seed = [evaluation_means(v["record"]) for v in runs[key].values()]
        marks = sorted({m for means in per_seed for m in means})
        if not marks:
            continue
        cells = [f"{m // 1000}k {fmt([means.get(m) for means in per_seed])}" for m in marks]
        print(f"-- {' '.join(key)}\n   " + " | ".join(cells))

    print("\n== 4. contact: per 10k window flagged share / touched share / start rule caught (of touched)")
    for key in keys:
        per_seed = [windows(v["lines"])[0] for v in runs[key].values()]
        marks = sorted({m for c in per_seed for m in c})
        print(f"-- {' '.join(key)}")
        for mark in marks:
            rows = [c[mark] for c in per_seed if mark in c]
            print(
                f"   {mark // 1000:>4}k: flagged {fmt([r['flagged'] for r in rows])} | "
                f"touched {fmt([r['touched'] for r in rows])} | caught {fmt([r['caught'] for r in rows])}"
            )
        totals = []
        for v in runs[key].values():
            probes = [e for e in v["lines"] if "replay" in e]
            if probes:
                last = max(probes, key=lambda e: e["interactions"])["replay"]
                totals.append(last)
        if totals:
            print(
                "   overall: flagged "
                + fmt([t["flagged"] / t["rows"] for t in totals if t["rows"]])
                + " | touched "
                + fmt([t["limit_touched"] / t["rows"] for t in totals if t["rows"]])
                + " | caught "
                + fmt([t["touched_and_near"] / t["limit_touched"] for t in totals if t["limit_touched"]])
                + f" | touched rows {sum(t['limit_touched'] for t in totals)} of {sum(t['rows'] for t in totals)}"
            )

    print("\n== 5. cost: ms per training decision per 10k window (evaluation, diagnostics excluded)")
    for key in keys:
        per_seed = [windows(v["lines"])[1] for v in runs[key].values()]
        marks = sorted({m for c in per_seed for m in c})
        if not marks:
            continue
        cells = [f"{m // 1000}k {fmt([c.get(m) for c in per_seed])}" for m in marks]
        print(f"-- {' '.join(key)}\n   " + " | ".join(cells))

    print("\n== 6. thinning: probe error change 100k -> 200k (200k minus 100k), plant steps in between")
    groups = defaultdict(dict)
    for key in keys:
        env, arm, handling, schedule = key
        groups[(env, arm, handling)][schedule] = runs[key]
    for (env, arm, handling), schedules in sorted(groups.items()):
        for schedule, seeds in sorted(schedules.items()):
            cells = []
            for name in ("free", "near", "past"):
                changes = []
                for v in seeds.values():
                    a, b = probe_at(v["lines"], 100_000), probe_at(v["lines"], 200_000)
                    if a and b and name in a.get("probe_relative_rms", {}):
                        changes.append(b["probe_relative_rms"][name] - a["probe_relative_rms"][name])
                if changes:
                    cells.append(f"{name} {fmt(changes)}")
            steps = []
            for v in seeds.values():
                a, b = probe_at(v["lines"], 100_000), probe_at(v["lines"], 200_000)
                if a and b:
                    steps.append(b["plant_gradient_steps"] - a["plant_gradient_steps"])
            if steps:
                cells.append(f"plant steps {fmt(steps, 6)}")
            print(f"{env} {arm} {handling} {schedule}: " + (" | ".join(cells) if cells else "not at 200k yet"))


if __name__ == "__main__":
    main()
