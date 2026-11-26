"""The adaptation experiment: warm starts from finished runs and their records.

A protocol of the experiment trains on a changed robot (testbed_settings physical_change,
common.ph_dmc._common.with_physical_change) and may start every run from a finished run's
checkpoint.pt (protocol key warm_start) and restrict the implicit arms' plant fit to some
of the plant's parts (implicit key trainable_plant_parts). docs/ct_ph_sac_layout.md
("Adaptation experiment") lists what a warm start takes, what it keeps fresh and why.

warm_start: {"checkpoints": {arm name: path template}, "require_interactions": n
(optional)}. A template is a path with the fields {seed} (the run's training seed, which
is the source run's), {arm} (the arm's name) and {testbed}; every arm the protocol runs
needs one, and only the CT-SAC baseline and the implicit arms take a warm start. With
require_interactions the source run must have collected exactly n decisions (a guard
against the intermediate checkpoint of a run still in progress).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, TypedDict

from common.ph_config import trainable_plant_parts_setting

if TYPE_CHECKING:
    from .protocol import EvaluationProtocol, RunSpec

# The arms' algorithms that take a warm start (algorithms.ph_baseline.TorqueCTSACBaseline,
# algorithms.implicit_ct_ph_sac.ImplicitCTPHSAC); the explicit arms have none.
WARM_START_ALGORITHMS = ("ct_sac", "implicit")
_WARM_START_KEYS = {"checkpoints", "require_interactions"}


class _WarmStartRequired(TypedDict):
    checkpoints: Mapping[str, str]


class WarmStartConfig(_WarmStartRequired, total=False):
    """The protocol key warm_start (module docstring)."""

    require_interactions: int


def _expand(template: str, *, seed: int, arm: str, testbed: str) -> str:
    try:
        return template.format(seed=seed, arm=arm, testbed=testbed)
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            f"warm_start checkpoint {template!r} may use only the fields {{seed}}, {{arm}} "
            f"and {{testbed}} ({error})."
        ) from None


def validate_warm_start(protocol: EvaluationProtocol) -> None:
    """Check the protocol's warm_start (when present): its keys, path templates and count."""
    if "warm_start" not in protocol:
        return
    config = protocol["warm_start"]
    if not isinstance(config, Mapping) or "checkpoints" not in config or (
        set(config) - _WARM_START_KEYS
    ):
        raise ValueError(
            f"warm_start needs checkpoints and optionally require_interactions, not {config!r}."
        )
    checkpoints = config["checkpoints"]
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("warm_start checkpoints must map arm names to checkpoint paths.")
    for arm, template in checkpoints.items():
        if not isinstance(template, str) or not template:
            raise ValueError(f"warm_start checkpoint of {arm!r} must be a path, not {template!r}.")
        _expand(template, seed=0, arm=str(arm), testbed=protocol["testbed"])
    required = config.get("require_interactions")
    if required is not None and (
        isinstance(required, bool) or not isinstance(required, int) or required < 1
    ):
        raise ValueError(f"warm_start require_interactions must be a positive integer, not {required!r}.")


def validate_warm_start_arms(protocol: EvaluationProtocol) -> None:
    """Every arm of a warm-start protocol has a checkpoint and an algorithm that takes one.

    run_protocol checks this before the first run (an arm selection, --arms, comes after
    the protocol is built, so build_protocol cannot).
    """
    if "warm_start" not in protocol:
        return
    checkpoints = protocol["warm_start"]["checkpoints"]
    for arm in protocol["arms"]:
        if arm["algorithm"] not in WARM_START_ALGORITHMS:
            raise ValueError(
                f"Arm {arm['name']!r} ({arm['algorithm']}) takes no warm start; a warm-start "
                f"protocol runs only {WARM_START_ALGORITHMS} arms (select them with --arms)."
            )
        if arm["name"] not in checkpoints:
            raise ValueError(f"warm_start has no checkpoint for arm {arm['name']!r}.")


def warm_start_checkpoint(run: RunSpec) -> Path:
    """The source checkpoint of a warm-started run: its arm's template, expanded."""
    protocol = run["protocol"]
    template = protocol["warm_start"]["checkpoints"][run["arm"]["name"]]
    return Path(
        _expand(
            template,
            seed=int(run["training_seed"]),
            arm=run["arm"]["name"],
            testbed=protocol["testbed"],
        )
    )


def apply_warm_start(agent: object, run: RunSpec) -> Dict[str, object]:
    """Warm-start a freshly built agent from its run's source checkpoint; return the record.

    The record: checkpoint (the expanded path), source_interactions, the networks loaded,
    the carried temperature (alpha_c, and for the baseline alpha_decision, its alpha per
    reference period T, alpha_c T, or alpha_c under ct_sac_reward_units "reference_period")
    and, for an implicit arm, trainable_plant_parts (None: every part). The baseline refuses
    a source checkpoint saved under other reward units or another reward source.
    """
    if run["arm"]["algorithm"] not in WARM_START_ALGORITHMS:
        raise ValueError(f"Arm {run['arm']['name']!r} takes no warm start.")
    path = warm_start_checkpoint(run)
    if not path.exists():
        raise FileNotFoundError(f"No warm-start checkpoint at {path} for {run['run_id']}.")
    loaded = agent.warm_start(
        path, require_interactions=run["protocol"]["warm_start"].get("require_interactions")
    )
    return {"checkpoint": str(path), **loaded}


def physical_change_record(testbed: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """The testbed's physical change as run records store it, None without one:
    kind, scale and every changed plant parameter with its nominal and physical value."""
    change = testbed.get("physical_change")
    if change is None:
        return None
    nominal = testbed["plant_parameters"]
    physical = testbed["physical_plant_parameters"]
    changed = {
        key: {"nominal": float(nominal[key]), "physical": float(physical[key])}
        for key in sorted(physical)
        if physical[key] != nominal[key]
    }
    return {"kind": change["kind"], "scale": float(change["scale"]), "changed": changed}


def adaptation_record(
    run: RunSpec, warm_start: Optional[Mapping[str, object]]
) -> Dict[str, object]:
    """The run record's adaptation fields, each only when the run has it: warm_start
    (apply_warm_start's record), physical_change (physical_change_record) and
    trainable_plant_parts (an implicit arm's config key). Empty for every other run."""
    fields: Dict[str, object] = {}
    if warm_start is not None:
        fields["warm_start"] = dict(warm_start)
    change = physical_change_record(run["protocol"]["matched"]["testbed_config"])
    if change is not None:
        fields["physical_change"] = change
    if run["arm"]["algorithm"] == "implicit":
        parts = trainable_plant_parts_setting(run["protocol"]["implicit_config"])
        if parts is not None:
            fields["trainable_plant_parts"] = list(parts)
    return fields
