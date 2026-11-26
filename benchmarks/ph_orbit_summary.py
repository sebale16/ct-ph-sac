"""Summary of runs with the Lyapunov ("orbit") rewards (numpy only; reads run records).

    python benchmarks/ph_orbit_summary.py ROOT [--all-evaluations]

ROOT holds one folder per group (a reward variant), each with the runs of
benchmarks/run_ph_sac.py --output ROOT/<group>/<arm>_seed<s> (any layout below the group
works: every run_record.json under it is one run, its arm and seed read from the record).
Runs in progress are summarized as far as their records go (a record is rewritten at every
evaluation and at a stop). Prints, per group and arm:
  progress   decisions per seed at the last record (* finished, r resumable checkpoint)
             and the record's write time (HH:MM), training seconds per decision from the
             record's wall_seconds (evaluation time excluded; "-" where the agent reports
             none, as the CT-SAC baseline's records do);
  last       the evaluation every seed has reached last (mean +- sd over seeds of the
             per-seed means over the evaluation episodes): score = undiscounted episode
             return, rate = score / time survived, upright_seconds, max_hold_seconds,
             capture (share of episodes held upright >= 1 s), mean and final V / V_down,
             and the time shares with V / V_down below 1e-2 and 1e-3 (orbit_fraction_1e_2,
             orbit_fraction_1e_3; absent on rewards without a Lyapunov function);
  curve      the score at every evaluation (mean over the seeds that reached it);
with --all-evaluations the "last" line for every evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = (
    ("score", "episode_return"),
    ("rate", "rate_executed"),
    ("upright_s", "upright_seconds"),
    ("hold_s", "max_hold_seconds"),
    ("V/Vd_mean", "lyapunov_ratio_mean"),
    ("V/Vd_final", "lyapunov_ratio_final"),
    ("orbit<1e-2", "orbit_fraction_1e_2"),
    ("orbit<1e-3", "orbit_fraction_1e_3"),
)


def read_record(path: Path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["result"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def fmt(values, digits=3):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "-"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g}+-{float(np.std(values, ddof=1)):.2g}"


def episode_means(checkpoint):
    """Per-checkpoint means over its episodes of FIELDS (None when absent) and capture."""
    episodes = checkpoint.get("episodes", [])
    means = {}
    for label, key in FIELDS:
        values = [float(e[key]) for e in episodes if key in e and e[key] is not None]
        means[label] = float(np.nanmean(values)) if values else None
    holds = [float(e["max_hold_seconds"]) for e in episodes if "max_hold_seconds" in e]
    means["capture"] = float(np.mean([h >= 1.0 for h in holds])) if holds else None
    return means


def collect(root: Path):
    """{group: {arm: {seed: run}}} for every run_record.json under ROOT/<group>/."""
    groups = defaultdict(lambda: defaultdict(dict))
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(group.rglob("run_record.json")):
            record = read_record(path)
            if record is None:
                continue
            run = record.get("run", {})
            arm = run.get("arm", {}).get("name", path.parents[1].name)
            seed = int(run.get("training_seed", path.parent.name.split("_")[-1]))
            counters = record.get("counters", {})
            groups[group.name][arm][seed] = {
                "decisions": int(counters.get("interactions", 0)),
                "wall": float(counters.get("wall_seconds", float("nan"))),
                "written": time.strftime("%H:%M", time.localtime(path.stat().st_mtime)),
                "complete": (path.parent / "resume_complete.json").exists(),
                "resumable": (path.parent / "resume.pt").exists(),
                "evaluations": {
                    int(c["interactions"]): episode_means(c) for c in record.get("evaluations", [])
                },
            }
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--all-evaluations", action="store_true")
    args = parser.parse_args(argv)
    groups = collect(args.root)
    if not groups:
        print(f"no run records under {args.root}")
        return
    for group, arms in groups.items():
        print(f"== {group}")
        for arm, seeds in sorted(arms.items()):
            progress = " ".join(
                f"s{s}:{r['decisions']}{'*' if r['complete'] else ''}{'r' if r['resumable'] else ''}"
                f"@{r['written']}"
                for s, r in sorted(seeds.items())
            )
            per_decision = [
                r["wall"] / r["decisions"] for r in seeds.values() if r["decisions"] > 0 and r["wall"] > 0.0
            ]
            print(f"  {arm}: decisions {progress} | s/decision {fmt(per_decision)}")
            marks = sorted({m for r in seeds.values() for m in r["evaluations"]})
            common = [m for m in marks if all(m in r["evaluations"] for r in seeds.values())]
            shown = common if args.all_evaluations else common[-1:]
            for mark in shown:
                rows = [r["evaluations"][mark] for r in seeds.values()]
                cells = [f"{label} {fmt([row[label] for row in rows])}" for label, _ in FIELDS
                         if any(row[label] is not None for row in rows)]
                cells.append(f"capture {fmt([row['capture'] for row in rows])}")
                print(f"    at {mark}: " + " | ".join(cells))
            curve = [
                f"{m // 1000}k {fmt([r['evaluations'][m]['score'] for r in seeds.values() if m in r['evaluations']])}"
                for m in marks
            ]
            if curve:
                print("    score curve: " + " | ".join(curve))


if __name__ == "__main__":
    main()
