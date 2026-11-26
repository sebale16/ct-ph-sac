"""Replay and held-out partitions for interval transitions."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import numpy as np
import torch

from common.ph_config import DEFAULT_REWARD_OBSERVATION, check_reward_observation
from common.ph_types import Transition, TransitionBatch

_ARRAY_FIELDS = (
    "states",
    "decisions",
    "durations",
    "horizons",
    "rewards",
    "next_states",
    "terminated",
    "truncated",
    "times",
    "next_times",
)
# Contact-exclusion columns (since 2026-11-10): the environment's limit_touched and the
# agent's start-state rule near_limit, bool [capacity, 1]. A state or file written before
# them (or a transition without them) reads as False.
_CONTACT_FIELDS = ("limit_touched", "near_limit")
# The end-state reward column (since 2026-11-22), float [capacity, 1]: each row's
# Transition end_reward_rate, held only by a buffer under reward_observation "end_state"
# (common.ph_config.REWARD_OBSERVATION_KEY), whose batches give it as end_rewards and leave
# the interval rewards out. A buffer of either mode refuses a state of the other.
_END_REWARD_FIELDS = ("end_rewards",)
_PARTITIONS = ("train", "gate")
# ReplayPartitions.state_dict's argument `copy` shadows the module there.
_copy_state = copy.deepcopy


def _padded_rows(
    array: np.ndarray, rows: int, filled: int, fill: object, what: str
) -> np.ndarray:
    """Copy stored rows into `rows` rows, filling the unwritten ones with `fill`.

    A state holds either the `filled` written rows or, as earlier checkpoints did, all
    `rows`; any other length cannot be reconstructed and raises.
    """
    array = np.asarray(array)
    stored = array.shape[0]
    if stored == rows:
        return array.copy()
    if stored != filled:
        raise ValueError(
            f"Replay state {what} has {stored} rows; expected its {filled} filled rows "
            f"or its capacity of {rows}."
        )
    padded = np.full((rows, *array.shape[1:]), fill, dtype=array.dtype)
    padded[:stored] = array
    return padded


class PHReplayBuffer:
    """Retain decisions, elapsed durations, scheduled horizons and termination flags.

    reward_observation (common.ph_config.REWARD_OBSERVATIONS; default "interval", every
    earlier buffer, bitwise) "end_state" adds the column end_rewards (each transition must
    carry end_reward_rate), and its batches give end_rewards in place of rewards: the
    interval rewards are stored (for offline checks) but never handed out, so a learner
    that still read them would fail with a KeyError.
    """

    def __init__(
        self,
        capacity: int,
        *,
        device: str = "cpu",
        seed: int,
        reward_observation: str = DEFAULT_REWARD_OBSERVATION,
    ) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be at least one.")
        self.capacity = int(capacity)
        self.device = device
        self.position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)
        self._arrays: Optional[Dict[str, np.ndarray]] = None
        self.reward_observation = check_reward_observation(reward_observation)

    @property
    def end_state(self) -> bool:
        """Whether the buffer holds end-state rates (reward_observation "end_state")."""
        return self.reward_observation == "end_state"

    def _allocate(self, transition: Transition) -> None:
        state_dim = np.size(transition["state"])
        decision_dim = np.size(transition["decision"])
        self._arrays = {
            "states": np.zeros((self.capacity, state_dim)),
            "decisions": np.zeros((self.capacity, decision_dim)),
            "durations": np.zeros((self.capacity, 1)),
            "horizons": np.zeros((self.capacity, 1)),
            "rewards": np.zeros((self.capacity, 1)),
            "next_states": np.zeros((self.capacity, state_dim)),
            "terminated": np.zeros((self.capacity, 1), dtype=bool),
            "truncated": np.zeros((self.capacity, 1), dtype=bool),
            "times": np.zeros((self.capacity, 1)),
            "next_times": np.zeros((self.capacity, 1)),
            "limit_touched": np.zeros((self.capacity, 1), dtype=bool),
            "near_limit": np.zeros((self.capacity, 1), dtype=bool),
        }
        if self.end_state:
            self._arrays["end_rewards"] = np.zeros((self.capacity, 1))

    def add(self, transition: Transition) -> None:
        if self.end_state and "end_reward_rate" not in transition:
            raise ValueError(
                "A replay under reward_observation 'end_state' needs each transition's "
                "end_reward_rate."
            )
        if self._arrays is None:
            self._allocate(transition)
        arrays = self._arrays
        index = self.position
        arrays["states"][index] = transition["state"]
        arrays["decisions"][index] = transition["decision"]
        arrays["durations"][index] = transition["duration"]
        arrays["horizons"][index] = transition["scheduled_duration"]
        arrays["rewards"][index] = transition["reward"]
        arrays["next_states"][index] = transition["next_state"]
        arrays["terminated"][index] = transition["terminated"]
        arrays["truncated"][index] = transition["truncated"]
        arrays["times"][index] = transition["time"]
        arrays["next_times"][index] = transition["next_time"]
        arrays["limit_touched"][index] = bool(transition.get("limit_touched", False))
        arrays["near_limit"][index] = bool(transition.get("near_limit", False))
        if self.end_state:
            arrays["end_rewards"][index] = transition["end_reward_rate"]
        self.position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        return self.batch(self._rng.integers(0, self._size, size=int(batch_size)))

    def batch(self, indices: np.ndarray) -> TransitionBatch:
        """Convert stored rows to tensors of the default dtype.

        Under reward_observation "end_state" the batch holds end_rewards (the rows'
        end-state rates) where it holds rewards otherwise, and no rewards.
        """
        dtype = torch.get_default_dtype()
        arrays = self._arrays

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(
                arrays[name][indices].astype(np.float64), dtype=dtype, device=self.device
            )

        reward_column = "end_rewards" if self.end_state else "rewards"
        return {
            "states": tensor("states"),
            "decisions": tensor("decisions"),
            "durations": tensor("durations"),
            "horizons": tensor("horizons"),
            reward_column: tensor(reward_column),
            "next_states": tensor("next_states"),
            "terminated": tensor("terminated"),
            "truncated": tensor("truncated"),
            "contact_flags": torch.as_tensor(
                (arrays["near_limit"][indices] | arrays["limit_touched"][indices]).astype(
                    np.float64
                ),
                dtype=dtype,
                device=self.device,
            ),
        }

    def contact_columns(self) -> Dict[str, np.ndarray]:
        """Copies of the stored rows' limit_touched and near_limit, bool [size] each."""
        if self._arrays is None:
            empty = np.zeros(0, dtype=bool)
            return {name: empty.copy() for name in _CONTACT_FIELDS}
        return {
            name: self._arrays[name][: self._size, 0].astype(bool).copy()
            for name in _CONTACT_FIELDS
        }

    def size(self) -> int:
        return self._size

    def _filled_rows(self) -> Dict[str, np.ndarray]:
        """Copies of the written rows, which are always the leading `size` rows.

        Rows are written from index 0 and the ring wraps only once every row is
        written, so before the wrap the rows past `size` are unwritten zeros and after
        it `size` is the capacity.
        """
        return {name: array[: self._size].copy() for name, array in self._arrays.items()}

    def _check_reward_column(self, names: Optional[object]) -> None:
        """Raise ValueError unless stored arrays (their names; None: an empty buffer, which
        either mode loads) hold the end-state column exactly when this buffer does."""
        if names is None:
            return
        stored = "end_rewards" in names
        if stored and not self.end_state:
            raise ValueError(
                "The replay state was written under reward_observation 'end_state' (it has "
                "the end_rewards column); this buffer runs under 'interval'."
            )
        if self.end_state and not stored:
            raise ValueError(
                "The replay state has no end_rewards column (it was written under "
                "reward_observation 'interval'); this buffer runs under 'end_state'."
            )

    def _restore_arrays(
        self, arrays: Optional[Mapping[str, np.ndarray]]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Pad stored rows back to the capacity with the zeros of unwritten rows."""
        if arrays is None:
            return None
        restored = {
            name: _padded_rows(np.asarray(array), self.capacity, self._size, 0, f"array {name!r}")
            for name, array in arrays.items()
        }
        for name in _CONTACT_FIELDS:
            # States written before the contact columns existed: no row flagged.
            restored.setdefault(name, np.zeros((self.capacity, 1), dtype=bool))
        return restored

    def state_dict(self, copy_rows: bool = True) -> Mapping[str, object]:
        """Copy the state, keeping only the filled rows of each array.

        copy_rows False gives the filled rows as tensor views of the live arrays (no
        copy; torch.save writes only those rows), valid until the next add.
        """
        if self._arrays is None:
            arrays = None
        elif copy_rows:
            arrays = self._filled_rows()
        else:
            arrays = {
                name: torch.from_numpy(array[: self._size])
                for name, array in self._arrays.items()
            }
        return {
            "capacity": self.capacity,
            "position": self.position,
            "size": self._size,
            "rng": copy.deepcopy(self._rng.bit_generator.state),
            "arrays": arrays,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore the stored buffer, capacity included; full-capacity arrays load too.

        A state written under the other reward_observation raises before anything changes.
        """
        self._check_reward_column(state["arrays"])
        self.capacity = int(state["capacity"])
        self.position = int(state["position"])
        self._size = int(state["size"])
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])
        self._arrays = self._restore_arrays(state["arrays"])

    def save(self, path: Path) -> None:
        arrays = {} if self._arrays is None else self._filled_rows()
        meta = {
            "capacity": self.capacity,
            "position": self.position,
            "size": self._size,
            "rng": self._rng.bit_generator.state,
        }
        with open(path, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as stored:
            self._check_reward_column(stored.files if "states" in stored else None)
            meta = json.loads(str(stored["meta"]))
            self.capacity = int(meta["capacity"])
            self.position = int(meta["position"])
            self._size = int(meta["size"])
            self._rng.bit_generator.state = meta["rng"]
            self._arrays = self._restore_arrays(
                {
                    name: stored[name]
                    for name in _ARRAY_FIELDS + _CONTACT_FIELDS + _END_REWARD_FIELDS
                    if name in stored
                }
                if "states" in stored
                else None
            )


class ReplayPartitions:
    """Keep train and admission data disjoint; record row provenance.

    reward_observation (default "interval") is given to both buffers (PHReplayBuffer).
    """

    def __init__(
        self,
        *,
        capacity: int,
        validation_fraction: float,
        device: str,
        seed: int,
        reward_observation: str = DEFAULT_REWARD_OBSERVATION,
    ) -> None:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("The held-out fraction must be nonnegative and leave training rows.")
        self.validation_fraction = float(validation_fraction)
        self.reward_observation = check_reward_observation(reward_observation)
        seeds = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint32)
        self._rng = np.random.default_rng(int(seeds[0]))
        capacities = {
            "train": int(capacity),
            "gate": max(1, math.ceil(validation_fraction * capacity)),
        }
        # seeds[2] seeded a removed partition; skipping it keeps the train and gate streams unchanged.
        streams = {"train": seeds[1], "gate": seeds[3]}
        self.buffers = {
            name: PHReplayBuffer(
                capacities[name],
                device=device,
                seed=int(streams[name]),
                reward_observation=self.reward_observation,
            )
            for name in _PARTITIONS
        }
        self.episode_ids = {
            name: np.full(capacities[name], -1, dtype=np.int64) for name in _PARTITIONS
        }
        self.reuse_counts = {
            name: np.zeros(capacities[name], dtype=np.int64) for name in _PARTITIONS
        }

    def add(
        self,
        transition: Transition,
        *,
        episode_id: int,
        partition: Optional[Literal["train", "gate"]] = None,
    ) -> str:
        """Route without training on admission rows; retain dependence/reuse metadata.

        A named partition overrides the random routing, for rows acquired to validate.
        """
        if partition is None:
            draw = self._rng.random()
            partition = "gate" if draw < self.validation_fraction else "train"
        buffer = self.buffers[partition]
        index = buffer.position
        buffer.add(transition)
        self.episode_ids[partition][index] = int(episode_id)
        self.reuse_counts[partition][index] = 0
        return partition

    def partition_size(self, partition: Literal["train", "gate"]) -> int:
        return self.buffers[partition].size()

    def sample(
        self, partition: Literal["train", "gate"], batch_size: int
    ) -> TransitionBatch:
        """Training rows are drawn with replacement; held-out rows without, counting reuse."""
        buffer = self.buffers[partition]
        size = buffer.size()
        if partition == "train":
            if size == 0:
                raise ValueError("The train partition has no rows.")
            return buffer.sample(batch_size)
        if size < batch_size:
            raise ValueError(
                f"The {partition} partition has {size} rows, fewer than the "
                f"{batch_size} requested."
            )
        indices = buffer._rng.choice(size, size=batch_size, replace=False)
        self.reuse_counts[partition][indices] += 1
        return buffer.batch(indices)

    def unused_rows(self, partition: Literal["train", "gate"]) -> int:
        """Count stored rows that no earlier read has inspected."""
        size = self.buffers[partition].size()
        return int(np.count_nonzero(self.reuse_counts[partition][:size] == 0))

    def draw_unused(
        self, partition: Literal["train", "gate"], count: int
    ) -> TransitionBatch:
        """Draw rows no earlier read inspected, for fresh evidence; reading marks them used."""
        buffer = self.buffers[partition]
        unused = np.flatnonzero(self.reuse_counts[partition][: buffer.size()] == 0)
        if unused.size < count:
            raise ValueError(
                f"The {partition} partition has {unused.size} unused rows, fewer than "
                f"the {count} requested."
            )
        indices = buffer._rng.choice(unused, size=count, replace=False)
        self.reuse_counts[partition][indices] += 1
        return buffer.batch(indices)

    def draw_for_allocation(
        self, partition: Literal["train", "gate"], count: int
    ) -> TransitionBatch:
        """Prefer rows already inspected, which may guide allocation but not score a batch."""
        buffer = self.buffers[partition]
        size = buffer.size()
        if size < count:
            raise ValueError(
                f"The {partition} partition has {size} rows, fewer than the "
                f"{count} requested."
            )
        reuse = self.reuse_counts[partition][:size]
        order = np.lexsort((buffer._rng.random(size), reuse == 0))
        indices = order[:count]
        reuse[indices] += 1
        return buffer.batch(indices)

    def provenance(
        self, partition: Literal["train", "gate"]
    ) -> Mapping[str, np.ndarray]:
        """Report episode identifiers and reuse counts of the stored rows."""
        size = self.buffers[partition].size()
        return {
            "episode_ids": self.episode_ids[partition][:size].copy(),
            "reuse_counts": self.reuse_counts[partition][:size].copy(),
        }

    def state_dict(self, copy: bool = True) -> Mapping[str, object]:
        """Copy the state; row metadata keeps only the rows its buffer has filled.

        copy False gives every filled row as a tensor view of the live arrays instead of a
        copy (the resumable checkpoint, saved at once); loading reads either form.
        """
        sizes = {name: buffer.size() for name, buffer in self.buffers.items()}

        def rows(values: np.ndarray, size: int) -> object:
            return values[:size].copy() if copy else torch.from_numpy(values[:size])

        return {
            "validation_fraction": self.validation_fraction,
            "rng": _copy_state(self._rng.bit_generator.state),
            "buffers": {
                name: buffer.state_dict(copy_rows=copy)
                for name, buffer in self.buffers.items()
            },
            "episode_ids": {
                name: rows(ids, sizes[name]) for name, ids in self.episode_ids.items()
            },
            "reuse_counts": {
                name: rows(uses, sizes[name]) for name, uses in self.reuse_counts.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore; unwritten rows get their initial episode id -1 and reuse count 0.

        A state written under the other reward_observation raises before anything changes.
        """
        for name, buffer in self.buffers.items():
            buffer._check_reward_column(state["buffers"][name]["arrays"])
        self.validation_fraction = float(state["validation_fraction"])
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])
        for name, buffer in self.buffers.items():
            buffer.load_state_dict(state["buffers"][name])

        def restore(rows: Mapping[str, np.ndarray], fill: int, what: str) -> dict:
            return {
                name: _padded_rows(
                    values,
                    self.buffers[name].capacity,
                    self.buffers[name].size(),
                    fill,
                    f"{what} of {name!r}",
                )
                for name, values in rows.items()
            }

        self.episode_ids = restore(state["episode_ids"], -1, "episode_ids")
        self.reuse_counts = restore(state["reuse_counts"], 0, "reuse_counts")
