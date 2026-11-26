"""Persistence and comparison reports for the PH-SAC protocol."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Mapping, Optional, Sequence, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.ph_config import (
    DEFAULT_FLAGGED_VALUE_GRADIENT,
    DEFAULT_REWARD_OBSERVATION,
    FLAGGED_VALUE_GRADIENT_KEY,
    LEGACY_REFRESH_MODE,
    PLANT_STEP_DEFAULTS,
    REWARD_OBSERVATION_KEY,
)

if TYPE_CHECKING:
    from .diagnostics import TheoryDiagnosticsRecord
    from .protocol import ArmName, EvaluationProtocol, RunResult

RECORD_NAME = "run_record.json"

UNITS = {
    "durations and times": "seconds of simulated physical time",
    "discounted_return": "reward units, discounted at the matched rate per second",
    "wall_seconds": "seconds of wall-clock training time, excluding evaluation",
    "discount_rate": "per second",
    "temperature_per_second": "reward units per second per nat",
}

# The "gradient steps, all components" axis. Only implicit records (and explicit ones under
# reward_observation "end_state") carry reward_gradient_steps; a key a record lacks counts
# zero.
GRADIENT_STEP_KEYS = (
    "critic_gradient_steps",
    "value_gradient_steps",
    "rate_gradient_steps",
    "actor_gradient_steps",
    "temperature_gradient_steps",
    "plant_gradient_steps",
    "reward_gradient_steps",
)

NOT_A_CERTIFICATE = (
    "No figure below is a certificate unless the evidence column says "
    "`certified_uniform`. Sampled maxima and the transfer score J do not establish "
    "uniform accuracy."
)


class ArmSummary(TypedDict):
    """Reward-return/compute columns, grouped by the evaluated policy kind.

    Return, failure and compute columns describe the last evaluation of each seed:
    interactions, gradient steps and wall time are per-seed means; accepted counts are
    totals over the arm's seeds.
    """

    testbed: str
    arm: ArmName
    policy_kind: Literal["stochastic", "deterministic_mean"]
    training_seed_count: int
    episode_count: int
    mean_discounted_return: float
    return_standard_error_across_training_seeds: float
    failure_rate: float
    interactions: int
    gradient_steps: int
    wall_seconds: float
    accepted_admissions: int


def _encode(value: object) -> object:
    """Make records JSON-compatible; arrays keep their type, tensors become summaries."""
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, torch.Tensor):
        return {
            "__tensor_summary__": {
                "shape": list(value.shape),
                "norm": float(value.detach().float().norm()),
            }
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"__callable__": getattr(value, "__name__", repr(value))}
    return value


def _decode(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "dtype"}:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_encode(payload), handle, indent=2)


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_manifest(protocol: EvaluationProtocol, path: str) -> None:
    """Write a reproducibility record of all matched settings and seed lists."""
    _write_json({"units": UNITS, "protocol": protocol}, Path(path))


def write_run_records(result: RunResult, output_directory: str) -> Sequence[str]:
    """Write per-episode and per-update records with explicit physical-time units."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    record = output / RECORD_NAME
    _write_json({"units": UNITS, "result": result}, record)
    paths = [str(record)]

    episodes = [
        {"interactions": checkpoint["interactions"], **episode}
        for checkpoint in result["evaluations"]
        for episode in checkpoint["episodes"]
    ]
    tables = (
        ("episodes.csv", episodes),
        ("solver.csv", result["solver"]),
        ("admissions.csv", result["admissions"]),
        ("oracle.csv", result["oracle"]),
        ("plant_calibration.csv", result["plant_calibration"]),
    )
    for name, rows in tables:
        if rows:
            _write_csv(rows, output / name)
            paths.append(str(output / name))
    return paths


def record_decision_mode(result: Mapping[str, object]) -> str:
    """The run's decision mode: the record's decision_mode, else its arm's, else "feedback".

    Records written before the field existed carry the arm's (ct_sac_torque "torque",
    the CT-PH-SAC arms "feedback"); one without either was a feedback run.
    """
    if "decision_mode" in result:
        return str(result["decision_mode"])
    arm = result.get("run", {}).get("arm", {})
    return str(arm.get("decision_mode", "feedback"))


def record_refresh_mode(result: Mapping[str, object]) -> str:
    """The run's controller refresh_mode: the record's, else LEGACY_REFRESH_MODE ("held").

    Records written before the field existed (2026-11-09) ran the sample-and-hold
    controller, whatever their testbed config says (it had no refresh_mode key, and
    the key's default is now "continuous").
    """
    return str(result.get("refresh_mode", LEGACY_REFRESH_MODE))


def record_plant_step_schedule(result: Mapping[str, object]) -> dict:
    """The run's learned-plant step schedule {"dense_until": D, "every": k}.

    Records written before the field existed (2026-11-10) took a plant step on every row:
    {"dense_until": 0, "every": 1}, the default of common.ph_config.PLANT_STEP_DEFAULTS.
    """
    stored = result.get("plant_step_schedule")
    if stored is None:
        return {
            "dense_until": PLANT_STEP_DEFAULTS["plant_step_dense_until"],
            "every": PLANT_STEP_DEFAULTS["plant_step_every"],
        }
    return {"dense_until": int(stored["dense_until"]), "every": int(stored["every"])}


def record_contact_substepping(result: Mapping[str, object]) -> Optional[dict]:
    """The run's contact-triggered fine substepping (common.ph_config.
    contact_substepping_setting), None when it was off.

    Records written before the field existed (2026-11-10) integrated every row at their
    model step: None.
    """
    stored = result.get("contact_substepping")
    if stored is None:
        return None
    return {key: float(value) for key, value in dict(stored).items()}


def record_contact_handling(result: Mapping[str, object]) -> Optional[dict]:
    """The run's contact exclusion (common.ph_config.contact_exclusion_setting), None under
    contact_handling "port".

    Records written before the field existed (2026-11-10) fitted the limit port: None.
    """
    stored = result.get("contact_handling")
    if stored is None:
        return None
    return dict(stored)


def record_reward_observation(result: Mapping[str, object]) -> str:
    """The run's reward datum (common.ph_config.REWARD_OBSERVATIONS): the record's
    reward_observation, written only under "end_state" (since 2026-11-22), else
    DEFAULT_REWARD_OBSERVATION ("interval"): every record without it learned from the
    interval rewards."""
    return str(result.get(REWARD_OBSERVATION_KEY, DEFAULT_REWARD_OBSERVATION))


def record_flagged_value_gradient(result: Mapping[str, object]) -> str:
    """The run's value gradient at contact exclusion's flagged rows
    (common.ph_config.FLAGGED_VALUE_GRADIENTS): the record's flagged_value_gradient, written
    only by an implicit arm under "stop_successor" (since 2026-11-23), else
    DEFAULT_FLAGGED_VALUE_GRADIENT ("full"): every record without it (the explicit and
    CT-SAC arms' records never carry it; they read no such residual)."""
    return str(result.get(FLAGGED_VALUE_GRADIENT_KEY, DEFAULT_FLAGGED_VALUE_GRADIENT))


def _mark_legacy_refresh_mode(result: dict) -> None:
    """Write "held" into the controller configs of an old record that has none."""
    protocol = result.get("run", {}).get("protocol", {})
    testbeds = [protocol.get("matched", {}).get("testbed_config")]
    testbeds += [protocol.get(key, {}).get("testbed") for key in ("explicit_config", "implicit_config")]
    for testbed in testbeds:
        if isinstance(testbed, dict) and isinstance(testbed.get("controller"), dict):
            testbed["controller"].setdefault("refresh_mode", LEGACY_REFRESH_MODE)


def load_run_records(paths: Sequence[str]) -> Sequence[RunResult]:
    """Read protocol records for comparison and diagnostic reports; tables are skipped.

    Every loaded result carries decision_mode (record_decision_mode fills it in for
    records written before the field existed), refresh_mode (record_refresh_mode:
    "held" for records written before the field existed, whose embedded testbed
    controller configs then get refresh_mode "held" as well), plant_step_schedule
    (record_plant_step_schedule: a step on every row for records without it),
    contact_substepping (record_contact_substepping: None, off, for records without it),
    contact_handling (record_contact_handling: None, the limit port, without it),
    reward_observation (record_reward_observation: "interval" without it) and
    flagged_value_gradient (record_flagged_value_gradient: "full" without it).
    """
    results = []
    for path in paths:
        if Path(path).suffix != ".json":
            continue
        with open(path, encoding="utf-8") as handle:
            result = _decode(json.load(handle))["result"]
        result["decision_mode"] = record_decision_mode(result)
        if "refresh_mode" not in result:
            _mark_legacy_refresh_mode(result)
        result["refresh_mode"] = record_refresh_mode(result)
        result["plant_step_schedule"] = record_plant_step_schedule(result)
        result["contact_substepping"] = record_contact_substepping(result)
        result["contact_handling"] = record_contact_handling(result)
        result[REWARD_OBSERVATION_KEY] = record_reward_observation(result)
        result[FLAGGED_VALUE_GRADIENT_KEY] = record_flagged_value_gradient(result)
        results.append(result)
    return results


def _testbed(result: RunResult) -> str:
    return result["run"]["protocol"]["testbed"]


def _arm(result: RunResult) -> str:
    return result["run"]["arm"]["name"]


def _gradient_steps(counters: Mapping[str, float]) -> int:
    return int(sum(counters.get(key, 0) for key in GRADIENT_STEP_KEYS))


def summarize_arms(results: Sequence[RunResult]) -> Sequence[ArmSummary]:
    """Keep policy kinds separate and use training seeds as the replication unit."""
    grouped: dict[tuple[str, str, str], list[tuple[RunResult, list[dict]]]] = defaultdict(list)
    for result in results:
        if not result["evaluations"]:
            continue
        final = result["evaluations"][-1]["episodes"]
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for episode in final:
            by_kind[episode["policy_kind"]].append(episode)
        for kind, episodes in by_kind.items():
            grouped[(_testbed(result), _arm(result), kind)].append((result, episodes))

    summaries: list[ArmSummary] = []
    for (testbed, arm, kind), seeds in grouped.items():
        seed_returns = np.array(
            [
                np.mean([episode["discounted_return"] for episode in episodes])
                for _, episodes in seeds
            ]
        )
        episodes = [episode for _, group in seeds for episode in group]
        count = len(seeds)
        counters = [result["counters"] for result, _ in seeds]
        summaries.append(
            {
                "testbed": testbed,
                "arm": arm,
                "policy_kind": kind,
                "training_seed_count": count,
                "episode_count": len(episodes),
                "mean_discounted_return": float(seed_returns.mean()),
                "return_standard_error_across_training_seeds": (
                    float(seed_returns.std(ddof=1) / math.sqrt(count))
                    if count > 1
                    else math.nan
                ),
                "failure_rate": float(np.mean([episode["failed"] for episode in episodes])),
                "interactions": int(round(np.mean([c["interactions"] for c in counters]))),
                "gradient_steps": int(round(np.mean([_gradient_steps(c) for c in counters]))),
                "wall_seconds": float(np.mean([c["wall_seconds"] for c in counters])),
                "accepted_admissions": sum(
                    bool(record["accepted"])
                    for result, _ in seeds
                    for record in result["admissions"]
                ),
            }
        )
    return summaries


def _by_testbed(results: Iterable[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[_testbed(result)].append(result)
    return grouped


def _arm_colors(names: Iterable[str]) -> dict[str, tuple]:
    palette = plt.get_cmap("tab10").colors
    return {name: palette[i % len(palette)] for i, name in enumerate(sorted(set(names)))}


def _checkpoint_x(checkpoint: Mapping[str, object], axis: str) -> float:
    counters = checkpoint["counters"]
    if axis == "interactions":
        return float(checkpoint["interactions"])
    if axis == "gradient_steps":
        return float(_gradient_steps(counters))
    return float(counters["wall_seconds"])


def _seed_curves(
    runs: Sequence[RunResult], axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align seeds by checkpoint index; returns x, return mean/SE across seeds, failure."""
    length = min(len(run["evaluations"]) for run in runs)
    xs, returns, failures = [], [], []
    for run in runs:
        checkpoints = run["evaluations"][:length]
        xs.append([_checkpoint_x(c, axis) for c in checkpoints])
        returns.append(
            [np.mean([e["discounted_return"] for e in c["episodes"]]) for c in checkpoints]
        )
        failures.append([np.mean([e["failed"] for e in c["episodes"]]) for c in checkpoints])
    returns_array = np.asarray(returns)
    errors = (
        returns_array.std(axis=0, ddof=1) / math.sqrt(len(runs))
        if len(runs) > 1
        else np.zeros(length)
    )
    return (
        np.mean(xs, axis=0),
        returns_array.mean(axis=0),
        errors,
        np.mean(failures, axis=0),
    )


AXIS_LABELS = {
    "interactions": "Interactions (decisions)",
    "gradient_steps": "Gradient steps, all components",
    "wall_seconds": "Training wall time (s)",
}


def plot_learning_curves(
    results: Sequence[RunResult],
    output_directory: str,
    *,
    x_axis: Literal["interactions", "gradient_steps", "wall_seconds"],
) -> Sequence[str]:
    """Plot return/failure curves against matched samples and reported compute."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for testbed, runs in _by_testbed(results).items():
        arms: dict[str, list[RunResult]] = defaultdict(list)
        for run in runs:
            if run["evaluations"]:
                arms[_arm(run)].append(run)
        if not arms:
            continue
        colors = _arm_colors(arms)
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for arm, arm_runs in sorted(arms.items()):
            x, mean, error, failure = _seed_curves(arm_runs, x_axis)
            top.plot(x, mean, color=colors[arm], label=arm)
            top.fill_between(x, mean - error, mean + error, color=colors[arm], alpha=0.2)
            bottom.plot(x, failure, color=colors[arm])
        top.set_ylabel("Discounted reward return")
        top.set_title(f"{testbed}: mean over training seeds, band is one standard error")
        bottom.set_ylabel("Failure rate")
        bottom.set_ylim(-0.02, 1.02)
        bottom.set_xlabel(AXIS_LABELS[x_axis])
        top.legend(loc="center left", bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=8)
        figure.tight_layout()
        path = output / f"learning_curves_{testbed}_{x_axis}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def _plot_gate(runs: Sequence[RunResult], path: Path) -> bool:
    with_records = [run for run in runs if run["admissions"]]
    if not with_records:
        return False
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for run in with_records:
        records = run["admissions"]
        index = [record["admission_index"] for record in records]
        upper = [record["interval_upper"] for record in records]
        top.plot(index, upper, color="#555555", linewidth=1, zorder=1)
        accepted = [record["accepted"] for record in records]
        top.scatter(
            index,
            upper,
            c=["#1b7f3b" if flag else "#b3261e" for flag in accepted],
            s=18,
            zorder=2,
        )
        bottom.plot(
            index,
            [record["transfer_diagnostic"] for record in records],
            color="#555555",
            linewidth=1,
        )
    top.axhline(0.0, color="black", linestyle="--", linewidth=1)
    top.set_ylabel("Production interval upper limit\n(green admitted, red not admitted)")
    bottom.set_ylabel("Transfer score J (sampled)")
    bottom.set_xlabel("Admission")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_oracle(runs: Sequence[RunResult], path: Path) -> bool:
    with_records = [run for run in runs if run["oracle"]]
    if not with_records:
        return False
    colors = _arm_colors(_arm(run) for run in with_records)
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fields = (
        ("drift_error_max", "Sampled max drift error"),
        ("diffusion_covariance_error_max", "Sampled max covariance error"),
        ("value_direction_error_max", "Sampled max value-direction error"),
    )
    labelled = set()
    for run in with_records:
        arm = _arm(run)
        label = None if arm in labelled else arm
        labelled.add(arm)
        x = [record["interactions"] for record in run["oracle"]]
        for axis, (field, _) in zip(axes, fields):
            axis.plot(
                x,
                [record[field] for record in run["oracle"]],
                color=colors[arm],
                linewidth=1,
                label=label,
            )
    for axis, (_, title) in zip(axes, fields):
        axis.set_ylabel(title, fontsize=8)
    axes[-1].set_xlabel("Interactions (decisions)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_calibration(runs: Sequence[RunResult], path: Path) -> bool:
    rows = [row for run in runs for row in run["plant_calibration"]]
    if not rows:
        return False
    latest = max(row["interactions"] for row in rows)
    rows = [row for row in rows if row["interactions"] == latest]
    centers = [
        500.0 * (row["duration_lower_seconds"] + row["duration_upper_seconds"])
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.scatter(centers, [row["mean_mahalanobis_square"] for row in rows], s=24)
    axis.axhline(
        rows[0]["state_dimension"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="state dimension (calibrated mean)",
    )
    axis.set_xlabel("Duration bin center (ms)")
    axis.set_ylabel("Mean Mahalanobis square")
    axis.set_title(f"Plant calibration at {latest} interactions")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def plot_diagnostics(results: Sequence[RunResult], output_directory: str) -> Sequence[str]:
    """Plot gate, oracle, and duration-calibration diagnostic figures."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    plotters = (
        ("gate", _plot_gate),
        ("oracle", _plot_oracle),
        ("plant_calibration", _plot_calibration),
    )
    for testbed, runs in _by_testbed(results).items():
        for name, plotter in plotters:
            path = output / f"{name}_{testbed}.png"
            if plotter(runs, path):
                paths.append(str(path))
    return paths


def write_theory_records(
    records: Sequence[TheoryDiagnosticsRecord], output_directory: str
) -> Sequence[str]:
    """Retain old/new-policy provenance, actual-step traces, premises, and bias floors."""
    path = Path(output_directory) / "theory_records.json"
    _write_json(
        {
            "note": (
                "Optimizer tensors are stored as shape and norm. Evidence kinds separate "
                "sampled diagnostics, assumed premises, and certified premises."
            ),
            "records": records,
        },
        path,
    )
    return [str(path)]


def plot_theory_diagnostics(
    records: Sequence[TheoryDiagnosticsRecord], output_directory: str
) -> Sequence[str]:
    """Separate sampled diagnostics, supplied uniform evidence, and objective/policy kinds."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    series: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    bounds: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        context = record["context"]
        key = (context["policy_kind"], context["objective"])
        for sampled in record["sampled"]:
            series[key].append(
                (
                    context["outer_iteration"],
                    sampled["raw_policy_residual_max"],
                    sampled["candidate_change_max"],
                )
            )
        bound = record["performance_bound"]
        if bound is not None:
            bounds[key + (bound["evidence"]["kind"],)].append(
                (context["outer_iteration"], bound["policy_loss_upper_bound"])
            )

    paths = []
    if series:
        figure, axis = plt.subplots(figsize=(9, 4))
        for (kind, objective), points in sorted(series.items()):
            points = sorted(points)
            x = [point[0] for point in points]
            axis.plot(x, [p[1] for p in points], label=f"{kind}/{objective}: raw residual")
            axis.plot(
                x,
                [p[2] for p in points],
                linestyle="--",
                label=f"{kind}/{objective}: candidate change",
            )
        axis.set_xlabel("Outer iteration")
        axis.set_ylabel("Sampled maximum (not a uniform bound)")
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        path = output / "theory_sampled_diagnostics.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    if bounds:
        figure, axis = plt.subplots(figsize=(9, 4))
        for (kind, objective, evidence), points in sorted(bounds.items()):
            points = sorted(points)
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=f"{kind}/{objective}, evidence {evidence}",
            )
        axis.set_xlabel("Outer iteration")
        axis.set_ylabel("Policy-loss upper bound")
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        path = output / "theory_policy_bounds.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _format(value: object) -> str:
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    return lines


def _theory_sections(results: Sequence[RunResult]) -> list[str]:
    records = [
        record for result in results for record in result.get("theory_records", [])
    ]
    sampled = [
        {
            "outer iteration": record["context"]["outer_iteration"],
            "policy kind": record["context"]["policy_kind"],
            "objective": record["context"]["objective"],
            "raw residual max": item["raw_policy_residual_max"],
            "candidate change max": item["candidate_change_max"],
            "evidence": item["evidence"]["kind"],
        }
        for record in records
        for item in record["sampled"]
    ]
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        bound = record["performance_bound"]
        if bound is None:
            continue
        by_claim[bound["claim_kind"]].append(
            {
                "outer iteration": record["context"]["outer_iteration"],
                "policy kind": bound["context"]["policy_kind"],
                "objective": bound["context"]["objective"],
                "policy loss bound": bound["policy_loss_upper_bound"],
                "soft return bound": bound["soft_return_lower_bound"],
                "reward return bound": bound["reward_return_lower_bound"],
                "threshold": bound["task_threshold"],
                "threshold satisfied": bound["threshold_satisfied"],
                "evidence": bound["evidence"]["kind"],
            }
        )

    columns = (
        "outer iteration",
        "policy kind",
        "objective",
        "policy loss bound",
        "soft return bound",
        "reward return bound",
        "threshold",
        "threshold satisfied",
        "evidence",
    )
    lines = ["## Sampled theory diagnostics", ""]
    if sampled:
        lines += _markdown_table(sampled, list(sampled[0]))
    else:
        lines.append("None recorded.")
    for title, claim, note in (
        (
            "Conditional bounds",
            "conditional",
            "These hold only if their stated uniform premises hold.",
        ),
        (
            "Certificates",
            "certified",
            "These rest on the supplied certified uniform evidence. A return threshold "
            "certifies a task criterion only if the reward encodes that criterion.",
        ),
    ):
        lines += ["", f"## {title}", "", note, ""]
        if by_claim[claim]:
            lines += _markdown_table(by_claim[claim], columns)
        else:
            lines.append("None recorded.")
    if by_claim["diagnostic"]:
        lines += ["", "## Bounds evaluated with sampled inputs (diagnostics only)", ""]
        lines += _markdown_table(by_claim["diagnostic"], columns)
    return lines


def write_comparison_report(
    protocol: EvaluationProtocol,
    results: Sequence[RunResult],
    output_directory: str,
) -> str:
    """Keep reward/soft bounds distinct; a return threshold alone is not task success."""
    matched = protocol["matched"]
    lines = [
        f"# CT-PH-SAC comparison: {protocol['testbed']}",
        "",
        NOT_A_CERTIFICATE,
        "",
        "## Matched quantities",
        "",
        f"- Interaction budget: {matched['interaction_budget']} decisions",
        f"- Discount rate: {matched['discount_rate']} per second",
        f"- Per-second temperature: {matched['temperature_per_second']}",
        f"- Clock: {matched['testbed_config']['clock']['kind']}",
        f"- Training seeds: {list(protocol['training_seeds'])}",
        f"- Evaluation reset seeds: {len(matched['evaluation_reset_seeds'])}",
        "",
        "## Reward return and compute, last evaluation",
        "",
        "Returns are reward-only and exclude entropy. The replication unit is the "
        "training seed; n/a marks a standard error from a single seed. The deterministic "
        "mean and the stochastic soft actor are different policies, so their rows must "
        "not be compared as one.",
        "",
    ]
    summaries = [dict(summary) for summary in summarize_arms(results)]
    columns = (
        "arm",
        "policy_kind",
        "training_seed_count",
        "episode_count",
        "mean_discounted_return",
        "return_standard_error_across_training_seeds",
        "failure_rate",
        "interactions",
        "gradient_steps",
        "wall_seconds",
        "accepted_admissions",
    )
    lines += _markdown_table(summaries, columns) if summaries else ["No evaluations recorded."]

    solver_rows = []
    gate_rows = []
    oracle_rows = []
    for result in results:
        arm, seed = _arm(result), result["run"]["training_seed"]
        if result["solver"]:
            solver = result["solver"]
            solver_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "outer iterations": len(solver),
                    "budget exits": sum(r["status"] == "budget_exit" for r in solver),
                    "mean inner iterations": float(
                        np.mean([record["inner_iterations"] for record in solver])
                    ),
                }
            )
        if result["admissions"]:
            gate = result["admissions"]
            gate_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "tests": len(gate),
                    "accepted": sum(bool(record["accepted"]) for record in gate),
                    "evidence against": sum(r["status"] == "evidence_against" for r in gate),
                    "inconclusive": sum(
                        r["status"] == "inconclusive_at_validation_cap" for r in gate
                    ),
                    "real rows": sum(r["real_rows_used"] for r in gate),
                    "acquired rows": sum(r["real_rows_acquired"] for r in gate),
                    "simulated paths": sum(r["simulated_paths_used"] for r in gate),
                    "last corrected mean": gate[-1]["corrected_mean"],
                    "last upper limit": gate[-1]["interval_upper"],
                    "last J": gate[-1]["transfer_diagnostic"],
                }
            )
        if result["oracle"]:
            last = result["oracle"][-1]
            oracle_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "sampled states": last["sampled_states"],
                    "drift error max": last["drift_error_max"],
                    "covariance error max": last["diffusion_covariance_error_max"],
                    "value-direction error max": last["value_direction_error_max"],
                }
            )
    lines += ["", "# Diagnostics", "", "These are sampled measurements, not uniform bounds.", ""]
    for title, rows in (
        ("Solver", solver_rows),
        ("Admission gate", gate_rows),
        ("Sampled oracle errors", oracle_rows),
    ):
        lines += [f"## {title}", ""]
        lines += _markdown_table(rows, list(rows[0])) if rows else ["None recorded."]
        lines.append("")
    lines += _theory_sections(results)

    path = Path(output_directory) / f"comparison_{protocol['testbed']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
