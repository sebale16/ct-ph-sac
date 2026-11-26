"""Exact restarts of protocol runs: the resumable checkpoint, its fingerprint, signals.

A run started with resume enabled (benchmarks/run_ph_sac.py --resume) writes one
resumable checkpoint per run, RESUME_NAME in the run's output folder, overwritten in
place: every `every_interactions` decisions, after every evaluation, and when SIGUSR1 or
SIGTERM arrives. Each is written at the agent's next safe point (the top of its loop,
before the next row is collected; set_resume_hook of the agents), atomically (temporary
file, fsync, rename, fsync of the folder), so a kill during the write leaves the previous
checkpoint intact. After a signal the run writes it and stops with ResumableStop, which
benchmarks/run_ph_sac.py turns into exit code RESUMABLE_EXIT_CODE.

The same command with --resume continues from it: the agent is built afresh, restored
(networks, target and frozen copies, optimizers, temperatures, the full replay with its
streams and positions, every counter and schedule position, the environment mid-episode,
every random stream, the evaluation records and schedule) and continues to the same final
result as an uninterrupted run, bitwise at a fixed thread count (tests/test_ph_resume.py).
A finished run deletes its resumable checkpoint and leaves COMPLETE_NAME, so a later
--resume skips it. A checkpoint or marker written for different settings is refused:
both carry run_fingerprint, a SHA-256 of the run's resolved settings and RESUME_FORMAT.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, TypedDict

import numpy as np
import torch

from common.ph_config import (
    CT_SAC_REWARD_SOURCE_KEY,
    CT_SAC_REWARD_UNITS_KEY,
    DEFAULT_CT_SAC_REWARD_SOURCE,
    DEFAULT_CT_SAC_REWARD_UNITS,
    DEFAULT_REWARD_OBSERVATION,
    IMPLICIT_DEFAULTS,
    NETWORK_COLUMNLESS_KEYS,
    PLANT_STEP_DEFAULTS,
    REWARD_OBSERVATION_KEY,
    STEP_ABLATION_CHOICES,
    contact_exclusion_setting,
    contact_substepping_setting,
    ct_sac_reward_units_setting,
    effective_ct_sac_reward_source,
    implicit_setting,
    model_integrator_setting,
    network_start_settings,
    plant_step_schedule,
    refresh_mode_setting,
    reward_observation_setting,
    step_ablation_setting,
)

if TYPE_CHECKING:
    from .protocol import RunSpec

RESUME_NAME = "resume.pt"
COMPLETE_NAME = "resume_complete.json"
# Bumped whenever the content of a resumable checkpoint changes meaning.
RESUME_FORMAT = 1
# The exit code of a run stopped by a signal after writing its resumable checkpoint
# (sysexits EX_TEMPFAIL: try again). A wrapper script requeues on it.
RESUMABLE_EXIT_CODE = 75
DEFAULT_RESUME_EVERY = 10_000
STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
# Protocol keys that do not change what a run computes, left out of its fingerprint.
_UNFINGERPRINTED = ("output_directory", "arms", "training_seeds", "checkpoint_replay")
# Protocol keys that change only some arms' runs, left out of the fingerprint's protocol:
# _resolved_settings enters each, resolved, for the arms that read it and only off its
# default, so that every other run's fingerprint is that of the code before the key (and
# an absent key and its default give the same fingerprint). reward_observation enters a
# CT-PH-SAC arm's settings as resolved reward_observation and a CT-SAC arm's only through
# its effective ct_sac_reward_source (common.ph_config.effective_ct_sac_reward_source).
_PER_ARM_PROTOCOL = (CT_SAC_REWARD_UNITS_KEY, CT_SAC_REWARD_SOURCE_KEY, REWARD_OBSERVATION_KEY)


class ResumeOptions(TypedDict):
    """every_interactions: resumable checkpoints every this many decisions (0: only at
    evaluations and signals). stop_after_interactions: testing aid, a stop requested as a
    signal would once a run crosses this many decisions in this process (None: never)."""

    every_interactions: int
    stop_after_interactions: Optional[int]


class ResumableStop(Exception):
    """The run wrote its resumable checkpoint and stopped; rerun with --resume."""


class ResumeMismatch(ValueError):
    """A resumable checkpoint or completion marker belongs to different settings."""


class _StopRequest:
    """Set by the signal handler; read by the protocol runner at its checkpoints."""

    def __init__(self) -> None:
        self.signal: Optional[int] = None

    @property
    def requested(self) -> bool:
        return self.signal is not None

    def request(self, signum: int) -> None:
        if self.signal is None:
            self.signal = int(signum)

    def clear(self) -> None:
        self.signal = None


STOP = _StopRequest()


def resume_options(
    every_interactions: int = DEFAULT_RESUME_EVERY,
    stop_after_interactions: Optional[int] = None,
) -> ResumeOptions:
    if isinstance(every_interactions, bool) or int(every_interactions) < 0:
        raise ValueError("every_interactions must be a nonnegative integer.")
    if stop_after_interactions is not None and int(stop_after_interactions) < 1:
        raise ValueError("stop_after_interactions must be positive.")
    return {
        "every_interactions": int(every_interactions),
        "stop_after_interactions": None
        if stop_after_interactions is None
        else int(stop_after_interactions),
    }


_SIGNAL_DEPTH = [0]


@contextlib.contextmanager
def stop_on_signals() -> Iterator[_StopRequest]:
    """Within the block, SIGUSR1 and SIGTERM request a stop instead of ending the process.

    Blocks nest (main, run_protocol, run_seed): only the outermost installs the handlers
    and clears the request on entry, and puts the previous handlers back and clears the
    request on exit, so a request made inside an inner block reaches the outer ones.
    """
    outermost = _SIGNAL_DEPTH[0] == 0
    previous = {}

    def handler(signum: int, _frame: object) -> None:
        STOP.request(signum)

    _SIGNAL_DEPTH[0] += 1
    try:
        if outermost:
            STOP.clear()
            for number in STOP_SIGNALS:
                previous[number] = signal.signal(number, handler)
        yield STOP
    finally:
        _SIGNAL_DEPTH[0] -= 1
        if outermost:
            for number, old in previous.items():
                signal.signal(number, old)
            STOP.clear()


# ------------------------------------------------------------------ fingerprint


def _canonical(value: object) -> object:
    """JSON-ready form with exact floats; arrays keep their dtype, tuples become lists."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"ndarray": _canonical(value.tolist()), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"callable": getattr(value, "__name__", repr(value))}
    return value


def _resolved_settings(run: RunSpec) -> dict:
    """Settings code defaults fill in, so a later change of a default is detected."""
    protocol = run["protocol"]
    resolved = {}
    for name in ("explicit_config", "implicit_config"):
        config = protocol[name]
        dense_until, every = plant_step_schedule(config["training"])
        resolved[name] = {
            "model_integrator": model_integrator_setting(config),
            "refresh_mode": refresh_mode_setting(config["testbed"]["controller"]),
            "plant_step_dense_until": dense_until,
            "plant_step_every": every,
        }
        # Contact substepping with its detector constants, only when it is on, so that the
        # fingerprints of runs without it are those of the code before it existed.
        contact = contact_substepping_setting(config)
        if contact is not None:
            resolved[name]["contact_substepping"] = dict(contact)
        # Contact exclusion with its rule's constants, only when it is on (the same reason).
        exclusion = contact_exclusion_setting(config)
        if exclusion is not None:
            resolved[name]["contact_handling"] = dict(exclusion)
    resolved["implicit_config"].update(
        {key: implicit_setting(protocol["implicit_config"], key) for key in IMPLICIT_DEFAULTS}
    )
    # The value-step ablation, resolved (Delta_ref included), only for an arm that carries
    # one of its choices (protocol.step_ablation_arms), so that every other run's
    # fingerprint is that of the code before it.
    arm = run["arm"]
    if any(name in arm for name in STEP_ABLATION_CHOICES):
        config = dict(protocol["implicit_config"])
        config.update({name: arm[name] for name in STEP_ABLATION_CHOICES if name in arm})
        resolved["implicit_config"]["step_ablation"] = dict(step_ablation_setting(config))
    # The implicit CT-SAC arms' numbers (tau, K, N, ...), resolved, only for an arm that
    # carries ct_sac_implicit_step (protocol.baseline_implicit_arms), for the same reason.
    if "ct_sac_implicit_step" in arm:
        from common.ph_config import ct_sac_implicit_setting

        resolved["explicit_config"]["ct_sac_implicit"] = ct_sac_implicit_setting(
            protocol["explicit_config"]
        )
    # The CT-SAC arms' reward units (protocol ct_sac_reward_units, _PER_ARM_PROTOCOL), only
    # for a CT-SAC arm and only off "second", for the same reason.
    if arm["algorithm"] == "ct_sac":
        units = ct_sac_reward_units_setting(protocol)
        if units != DEFAULT_CT_SAC_REWARD_UNITS:
            resolved[CT_SAC_REWARD_UNITS_KEY] = units
        # Their reward source (protocol ct_sac_reward_source, and "end_state" under
        # reward_observation "end_state"), the same way: only off "interval". A run under
        # reward_observation "end_state" thus fingerprints as one under
        # ct_sac_reward_source "end_state": both compute the same.
        source = effective_ct_sac_reward_source(protocol)
        if source != DEFAULT_CT_SAC_REWARD_SOURCE:
            resolved[CT_SAC_REWARD_SOURCE_KEY] = source
    # The CT-PH-SAC arms' reward datum (protocol reward_observation), only for those arms
    # and only off "interval", for the same reason.
    if arm["algorithm"] in ("explicit", "implicit"):
        observation = reward_observation_setting(protocol)
        if observation != DEFAULT_REWARD_OBSERVATION:
            resolved[REWARD_OBSERVATION_KEY] = observation
    resolved["plant_step_defaults"] = dict(PLANT_STEP_DEFAULTS)
    return resolved


_CONFIGS = ("explicit_config", "implicit_config")


def _fingerprint_protocol(protocol: Mapping[str, object]) -> Mapping[str, object]:
    """The protocol as the fingerprint reads it: the column-less network keys
    (common.ph_config.NETWORK_COLUMNLESS_KEYS) resolved, and left out at their defaults
    (network_start_settings), so that a config stating a default resumes a run started
    without the key, and back, while any other value must match. The protocol itself
    (unchanged) when no network sets one: every earlier run."""
    networks = {
        name: protocol[name].get("network", {}) for name in _CONFIGS if name in protocol
    }
    if not any(key in network for network in networks.values() for key in NETWORK_COLUMNLESS_KEYS):
        return protocol
    changed = dict(protocol)
    for name, network in networks.items():
        resolved = {
            key: value for key, value in network.items() if key not in NETWORK_COLUMNLESS_KEYS
        }
        resolved.update(network_start_settings(network))
        changed[name] = {**protocol[name], "network": resolved}
    return changed


def run_settings(run: RunSpec) -> dict:
    """What a run computes from: its protocol (less _UNFINGERPRINTED and
    _PER_ARM_PROTOCOL; _fingerprint_protocol's), arm, seed and the resolved defaults, with
    RESUME_FORMAT."""
    left_out = _UNFINGERPRINTED + _PER_ARM_PROTOCOL
    protocol = {
        key: value
        for key, value in _fingerprint_protocol(run["protocol"]).items()
        if key not in left_out
    }
    return _canonical(
        {
            "format": RESUME_FORMAT,
            "protocol": protocol,
            "arm": run["arm"],
            "training_seed": run["training_seed"],
            "resolved": _resolved_settings(run),
        }
    )


def run_fingerprint(run: RunSpec) -> str:
    """SHA-256 of run_settings: equal exactly when a restart would compute the same run."""
    text = json.dumps(run_settings(run), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _differences(stored: object, current: object, path: str = "") -> list:
    if isinstance(stored, dict) and isinstance(current, dict):
        found = []
        for key in sorted(set(stored) | set(current)):
            where = f"{path}.{key}" if path else key
            if key not in stored or key not in current:
                found.append(where)
            else:
                found.extend(_differences(stored[key], current[key], where))
        return found
    return [] if stored == current else [path or "(root)"]


def check_fingerprint(stored: Mapping[str, object], run: RunSpec, where: Path) -> None:
    """Raise ResumeMismatch unless the stored fingerprint is this run's."""
    current = run_fingerprint(run)
    if stored.get("fingerprint") == current:
        return
    settings = stored.get("settings")
    changed = (
        _differences(settings, run_settings(run))[:12]
        if isinstance(settings, dict)
        else ["(no stored settings)"]
    )
    rule = ""
    if any(path.endswith("contact_handling.rule_version") for path in changed):
        rule = (
            " Contact exclusion's start-state rule changed (rule_version, "
            "common.ph_config.EXCLUSION_RULE_VERSION): the stored replay was flagged under "
            "the other rule, so the run cannot continue under this one."
        )
    raise ResumeMismatch(
        f"{where} was written for different settings (stored fingerprint "
        f"{str(stored.get('fingerprint'))[:12]}, this run's {current[:12]}; differing: "
        f"{', '.join(changed) or 'format'}).{rule} A resume must use the same config, arm, "
        "seed, budget and code convention; use another output directory, or delete the "
        "file to start this run afresh."
    )


def library_versions() -> dict:
    import mujoco

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def warn_if_versions_differ(stored: Mapping[str, object], where: Path) -> None:
    """Different library versions may round differently: warn, do not refuse."""
    versions = stored.get("versions") or {}
    current = library_versions()
    changed = {k: (versions.get(k), v) for k, v in current.items() if versions.get(k) != v}
    if changed:
        print(
            f"warning: {where} was written with other library versions {changed}; the "
            "continuation may not be bitwise that of an uninterrupted run.",
            file=sys.stderr,
        )


# ------------------------------------------------------------------- RNG and I/O


def capture_global_rng() -> dict:
    """The process-global python, numpy and torch CPU streams."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_global_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def _fsync_folder(folder: Path) -> None:
    try:
        descriptor = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    with open(path, "rb+") as handle:
        os.fsync(handle.fileno())


def atomic_torch_save(payload: object, path: Path) -> None:
    """torch.save to a temporary file in path's folder, fsync, rename over path, fsync.

    A failure or kill before the rename leaves any earlier file at path untouched; the
    temporary file is removed on failure.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    _fsync_folder(path.parent)


def atomic_write_text(text: str, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    _fsync_folder(path.parent)


def resume_payload(
    run: RunSpec, agent: object, runner: Mapping[str, object]
) -> dict:
    """The resumable checkpoint of agent at a safe point, with the runner's state."""
    return {
        "format": RESUME_FORMAT,
        "fingerprint": run_fingerprint(run),
        "settings": run_settings(run),
        "versions": library_versions(),
        "interactions": int(agent.counters()["interactions"]),
        "agent": agent.resume_state(),
        "runner": dict(runner),
        "global_rng": capture_global_rng(),
    }


def load_resume_payload(path: Path, run: RunSpec) -> dict:
    """Read a resumable checkpoint and check that it is this run's."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    check_fingerprint(payload, run, path)
    warn_if_versions_differ(payload, path)
    return payload


def write_completion(path: Path, run: RunSpec, interactions: int) -> None:
    atomic_write_text(
        json.dumps(
            {
                "fingerprint": run_fingerprint(run),
                "settings": run_settings(run),
                "interactions": int(interactions),
            },
            indent=1,
        ),
        path,
    )


def read_completion(path: Path, run: RunSpec) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    check_fingerprint(stored, run, path)
    return stored
