import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common.ph_replay import PHReplayBuffer, ReplayPartitions


def make_transition(i, terminated=False, truncated=False):
    return {
        "state": np.array([i, i + 0.5]),
        "decision": np.array([i * 0.1, 1.0, 2.0]),
        "duration": 0.01 + 0.001 * i,
        "scheduled_duration": 0.02 + 0.001 * i,
        "reward": -float(i),
        "next_state": np.array([i + 1.0, i + 1.5]),
        "terminated": terminated,
        "truncated": truncated,
        "time": float(i),
        "next_time": i + 0.01,
    }


class TestPHReplayBuffer(unittest.TestCase):
    def test_sample_shapes_and_values(self):
        buffer = PHReplayBuffer(16, seed=0)
        for i in range(5):
            buffer.add(make_transition(i, terminated=i == 2, truncated=i == 3))
        batch = buffer.sample(64)
        self.assertEqual(batch["states"].shape, (64, 2))
        self.assertEqual(batch["decisions"].shape, (64, 3))
        for name in ("durations", "rewards", "terminated", "truncated"):
            self.assertEqual(batch[name].shape, (64, 1))
            self.assertEqual(batch[name].dtype, torch.get_default_dtype())
        row = batch["states"][:, 0].round().long()
        self.assertTrue(torch.equal(batch["rewards"][:, 0], -row.float()))
        self.assertTrue(torch.equal(batch["terminated"][:, 0], (row == 2).float()))
        self.assertTrue(torch.equal(batch["truncated"][:, 0], (row == 3).float()))
        self.assertTrue(
            torch.allclose(batch["durations"][:, 0], 0.01 + 0.001 * row.float())
        )

    def test_ring_overwrites_oldest(self):
        buffer = PHReplayBuffer(4, seed=0)
        for i in range(6):
            buffer.add(make_transition(i))
        self.assertEqual(buffer.size(), 4)
        rows = set(buffer.sample(200)["rewards"][:, 0].tolist())
        self.assertEqual(rows, {-2.0, -3.0, -4.0, -5.0})

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            PHReplayBuffer(4, seed=0).sample(1)

    def test_save_load_continues_identically(self):
        buffer = PHReplayBuffer(8, seed=3)
        for i in range(5):
            buffer.add(make_transition(i))
        buffer.sample(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            buffer.save(path)
            restored = PHReplayBuffer(1, seed=99)
            restored.load(path)
        self.assertEqual(restored.size(), 5)
        self.assertEqual(restored.position, buffer.position)
        expected = buffer.sample(32)
        actual = restored.sample(32)
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(tensor, actual[name]), name)


class TestReplayPartitions(unittest.TestCase):
    def fill(self, rows=2000, **kwargs):
        options = dict(
            capacity=4000,
            validation_fraction=0.1,
            device="cpu",
            seed=0,
        )
        options.update(kwargs)
        partitions = ReplayPartitions(**options)
        routes = [
            partitions.add(make_transition(i), episode_id=i // 100) for i in range(rows)
        ]
        return partitions, routes

    def test_routing_fractions(self):
        partitions, routes = self.fill(rows=20000, capacity=40000)
        self.assertAlmostEqual(routes.count("gate") / 20000, 0.1, delta=0.01)
        self.assertEqual(routes.count("gate") + routes.count("train"), 20000)
        self.assertEqual(partitions.partition_size("gate"), routes.count("gate"))
        self.assertEqual(partitions.partition_size("train"), routes.count("train"))

    def test_streams_keep_their_seeds(self):
        """Routing, train and gate draws use the seeds they had beside a third partition."""
        partitions, _ = self.fill(rows=0)
        seeds = np.random.SeedSequence(0).generate_state(4, dtype=np.uint32)
        routing = np.random.default_rng(int(seeds[0]))
        self.assertEqual(partitions._rng.random(), routing.random())
        for name, index in (("train", 1), ("gate", 3)):
            expected = PHReplayBuffer(10, device="cpu", seed=int(seeds[index]))
            self.assertEqual(
                partitions.buffers[name]._rng.bit_generator.state,
                expected._rng.bit_generator.state,
            )

    def test_partitions_are_disjoint(self):
        partitions, _ = self.fill()
        seen = {}
        for name in ("train", "gate"):
            size = partitions.partition_size(name)
            rows = partitions.buffers[name].batch(np.arange(size))["rewards"]
            seen[name] = set(rows[:, 0].tolist())
            self.assertEqual(len(seen[name]), size)
        self.assertFalse(seen["train"] & seen["gate"])
        self.assertEqual(len(seen["train"] | seen["gate"]), 2000)

    def test_provenance_records_episode(self):
        partitions, routes = self.fill()
        provenance = partitions.provenance("gate")
        rows = partitions.buffers["gate"].batch(
            np.arange(partitions.partition_size("gate"))
        )["rewards"][:, 0]
        expected = (-rows).long().numpy() // 100
        np.testing.assert_array_equal(provenance["episode_ids"], expected)
        self.assertTrue((provenance["reuse_counts"] == 0).all())

    def test_draw_unused_never_repeats_a_row(self):
        partitions, _ = self.fill()
        size = partitions.partition_size("gate")
        first = partitions.draw_unused("gate", size // 2)
        counts = partitions.provenance("gate")["reuse_counts"]
        self.assertEqual(counts.sum(), size // 2)
        self.assertEqual(counts.max(), 1)
        self.assertEqual(len(set(first["rewards"][:, 0].tolist())), size // 2)
        self.assertEqual(partitions.unused_rows("gate"), size - size // 2)

        second = partitions.draw_unused("gate", size - size // 2)
        self.assertEqual(partitions.unused_rows("gate"), 0)
        self.assertFalse(
            set(first["rewards"][:, 0].tolist()) & set(second["rewards"][:, 0].tolist())
        )
        with self.assertRaisesRegex(ValueError, "unused rows"):
            partitions.draw_unused("gate", 1)

    def test_allocation_prefers_inspected_rows(self):
        partitions, _ = self.fill()
        scored = partitions.draw_unused("gate", 6)
        pilot = partitions.draw_for_allocation("gate", 4)
        self.assertLessEqual(
            set(pilot["rewards"][:, 0].tolist()), set(scored["rewards"][:, 0].tolist())
        )
        unused = partitions.unused_rows("gate")
        partitions.draw_for_allocation("gate", 8)
        self.assertEqual(partitions.unused_rows("gate"), unused - 2)

    def test_a_named_partition_overrides_routing(self):
        partitions, _ = self.fill(rows=0)
        for i in range(20):
            self.assertEqual(
                partitions.add(make_transition(i), episode_id=i, partition="gate"), "gate"
            )
        self.assertEqual(partitions.partition_size("gate"), 20)
        self.assertEqual(partitions.partition_size("train"), 0)

    def test_too_few_rows_raises(self):
        partitions, _ = self.fill(rows=20)
        size = partitions.partition_size("gate")
        with self.assertRaisesRegex(ValueError, "gate partition"):
            partitions.sample("gate", size + 1)

    def test_overwrite_resets_reuse(self):
        partitions, _ = self.fill(rows=0, capacity=10, validation_fraction=0.5)
        for i in range(100):
            partitions.add(make_transition(i), episode_id=i)
            if partitions.partition_size("gate") == 5:
                partitions.sample("gate", 5)
        gate = partitions.provenance("gate")
        rows = partitions.buffers["gate"].batch(np.arange(5))["rewards"][:, 0]
        np.testing.assert_array_equal(gate["episode_ids"], (-rows).long().numpy())
        self.assertLess(gate["reuse_counts"].sum(), 50)

    def test_state_dict_round_trip(self):
        partitions, _ = self.fill()
        partitions.sample("gate", 7)
        state = partitions.state_dict()
        restored = ReplayPartitions(
            capacity=4000,
            validation_fraction=0.1,
            device="cpu",
            seed=123,
        )
        restored.load_state_dict(state)

        for name in ("train", "gate"):
            self.assertEqual(restored.partition_size(name), partitions.partition_size(name))
        expected_route = partitions.add(make_transition(5000), episode_id=50)
        actual_route = restored.add(make_transition(5000), episode_id=50)
        self.assertEqual(expected_route, actual_route)
        for name in ("train", "gate"):
            expected = partitions.sample(name, 5)
            actual = restored.sample(name, 5)
            for key, tensor in expected.items():
                self.assertTrue(torch.equal(tensor, actual[key]), key)
        np.testing.assert_array_equal(
            partitions.provenance("gate")["reuse_counts"],
            restored.provenance("gate")["reuse_counts"],
        )


def assert_same_buffer(case, left, right):
    """Capacity, position, size, RNG and every array row (padding included) agree."""
    case.assertEqual(left.capacity, right.capacity)
    case.assertEqual(left.position, right.position)
    case.assertEqual(left.size(), right.size())
    case.assertEqual(left._rng.bit_generator.state, right._rng.bit_generator.state)
    case.assertEqual(set(left._arrays), set(right._arrays))
    for name, array in left._arrays.items():
        case.assertEqual(array.shape, right._arrays[name].shape, name)
        case.assertEqual(array.dtype, right._arrays[name].dtype, name)
        np.testing.assert_array_equal(array, right._arrays[name], name)


def assert_same_partitions(case, left, right):
    case.assertEqual(left._rng.bit_generator.state, right._rng.bit_generator.state)
    for name in ("train", "gate"):
        assert_same_buffer(case, left.buffers[name], right.buffers[name])
        for field in ("episode_ids", "reuse_counts"):
            expected, actual = getattr(left, field)[name], getattr(right, field)[name]
            case.assertEqual(expected.dtype, actual.dtype, field)
            np.testing.assert_array_equal(expected, actual, field)


class TestCompactState(unittest.TestCase):
    """State dicts keep only filled rows and restore the identical buffer."""

    def filled(self, capacity, rows, seed=0):
        buffer = PHReplayBuffer(capacity, seed=seed)
        for i in range(rows):
            buffer.add(make_transition(i, terminated=i % 7 == 0, truncated=i % 5 == 0))
        buffer.sample(3)
        return buffer

    def test_partly_filled_buffer_stores_its_rows(self):
        buffer = self.filled(100, 10)
        state = buffer.state_dict()
        for name, array in state["arrays"].items():
            self.assertEqual(array.shape[0], 10, name)
        restored = PHReplayBuffer(5, seed=99)
        restored.load_state_dict(state)
        assert_same_buffer(self, buffer, restored)
        for i in range(10, 20):
            buffer.add(make_transition(i))
            restored.add(make_transition(i))
        expected, actual = buffer.sample(32), restored.sample(32)
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(tensor, actual[name]), name)

    def test_wrapped_ring_round_trips(self):
        buffer = self.filled(8, 13)
        self.assertEqual((buffer.position, buffer.size()), (5, 8))
        state = buffer.state_dict()
        for name, array in state["arrays"].items():
            self.assertEqual(array.shape[0], 8, name)
        restored = PHReplayBuffer(8, seed=99)
        restored.load_state_dict(state)
        assert_same_buffer(self, buffer, restored)
        for i in range(13, 16):
            buffer.add(make_transition(i))
            restored.add(make_transition(i))
        assert_same_buffer(self, buffer, restored)

    def test_full_capacity_state_still_loads(self):
        buffer = self.filled(100, 10)
        state = dict(buffer.state_dict())
        state["arrays"] = {name: array.copy() for name, array in buffer._arrays.items()}
        restored = PHReplayBuffer(5, seed=99)
        restored.load_state_dict(state)
        assert_same_buffer(self, buffer, restored)

    def test_other_lengths_are_rejected(self):
        buffer = self.filled(100, 10)
        state = dict(buffer.state_dict())
        state["arrays"] = {name: array[:20].copy() for name, array in buffer._arrays.items()}
        with self.assertRaisesRegex(ValueError, "filled rows"):
            PHReplayBuffer(100, seed=0).load_state_dict(state)

    def test_empty_buffer_round_trips(self):
        state = PHReplayBuffer(10, seed=0).state_dict()
        self.assertIsNone(state["arrays"])
        restored = PHReplayBuffer(10, seed=4)
        restored.load_state_dict(state)
        self.assertEqual(restored.size(), 0)
        self.assertIsNone(restored._arrays)

    def test_npz_save_stores_filled_rows(self):
        buffer = self.filled(100, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            buffer.save(path)
            with np.load(path) as stored:
                self.assertEqual(stored["states"].shape, (10, 2))
            restored = PHReplayBuffer(1, seed=99)
            restored.load(path)
        assert_same_buffer(self, buffer, restored)

    def partitions(self, rows, capacity=4000):
        partitions = ReplayPartitions(
            capacity=capacity, validation_fraction=0.1, device="cpu", seed=0
        )
        for i in range(rows):
            partitions.add(make_transition(i), episode_id=i // 10)
        partitions.sample("gate", min(3, partitions.partition_size("gate")))
        return partitions

    def fresh_partitions(self, capacity=4000):
        return ReplayPartitions(
            capacity=capacity, validation_fraction=0.1, device="cpu", seed=7
        )

    def test_partitions_store_filled_metadata(self):
        partitions = self.partitions(400)
        state = partitions.state_dict()
        for name in ("train", "gate"):
            size = partitions.partition_size(name)
            self.assertLess(size, partitions.buffers[name].capacity)
            self.assertEqual(state["episode_ids"][name].shape, (size,))
            self.assertEqual(state["reuse_counts"][name].shape, (size,))
            for array in state["buffers"][name]["arrays"].values():
                self.assertEqual(array.shape[0], size)
        restored = self.fresh_partitions()
        restored.load_state_dict(state)
        assert_same_partitions(self, partitions, restored)
        # Unwritten rows keep the constructor's fill values.
        self.assertTrue((restored.episode_ids["train"][400:] == -1).all())
        self.assertTrue((restored.reuse_counts["gate"][100:] == 0).all())

    def test_wrapped_partitions_round_trip(self):
        partitions = self.partitions(300, capacity=20)
        self.assertEqual(partitions.partition_size("train"), 20)
        self.assertEqual(partitions.partition_size("gate"), 2)
        restored = self.fresh_partitions(capacity=20)
        restored.load_state_dict(partitions.state_dict())
        assert_same_partitions(self, partitions, restored)

    def test_a_ring_smaller_than_its_adds_wraps_with_consistent_metadata(self):
        """replay_capacity below the run's decisions (300,000 rows for 500,000 decisions),
        no held-out partition (validation_fraction 0.0): 7 rows for 23 adds."""
        capacity, adds = 7, 23
        partitions = ReplayPartitions(
            capacity=capacity, validation_fraction=0.0, device="cpu", seed=0
        )
        for i in range(adds):
            transition = dict(make_transition(i), near_limit=i % 3 == 0, limit_touched=i % 5 == 0)
            self.assertEqual(partitions.add(transition, episode_id=i // 4), "train")
        train = partitions.buffers["train"]
        self.assertEqual(partitions.partition_size("train"), capacity)
        self.assertEqual(partitions.partition_size("gate"), 0)
        self.assertEqual(partitions.buffers["gate"].capacity, 1)
        self.assertEqual(train.position, adds % capacity)
        # Slot k holds the latest add i with i % capacity == k, and that add's metadata.
        kept = [max(i for i in range(adds) if i % capacity == k) for k in range(capacity)]
        rows = train.batch(np.arange(capacity))
        self.assertEqual((-rows["rewards"][:, 0]).long().tolist(), kept)
        self.assertEqual(rows["states"][:, 0].long().tolist(), kept)
        np.testing.assert_array_equal(partitions.episode_ids["train"], np.array(kept) // 4)
        self.assertTrue((partitions.reuse_counts["train"] == 0).all())
        flags = [i % 3 == 0 or i % 5 == 0 for i in kept]
        self.assertEqual(rows["contact_flags"][:, 0].bool().tolist(), flags)
        columns = train.contact_columns()
        np.testing.assert_array_equal(columns["near_limit"], [i % 3 == 0 for i in kept])
        np.testing.assert_array_equal(columns["limit_touched"], [i % 5 == 0 for i in kept])
        # The latest rows by position (PHAlgorithm.policy_diagnostics' indices), newest first.
        latest = (train.position - 1 - np.arange(capacity)) % train.capacity
        self.assertEqual(
            (-train.batch(latest)["rewards"][:, 0]).long().tolist(),
            list(range(adds - 1, adds - 1 - capacity, -1)),
        )
        # Uniform draws read only the rows still stored.
        drawn = set((-partitions.sample("train", 200)["rewards"][:, 0]).long().tolist())
        self.assertLessEqual(drawn, set(kept))
        # Both state forms (the checkpoint's copy, the resumable checkpoint's views) restore
        # the wrapped ring, and the restored ring keeps wrapping as the original does.
        for copy_rows in (True, False):
            with self.subTest(copy_rows=copy_rows):
                restored = ReplayPartitions(
                    capacity=capacity, validation_fraction=0.0, device="cpu", seed=9
                )
                restored.load_state_dict(partitions.state_dict(copy=copy_rows))
                self.assertEqual(
                    restored._rng.bit_generator.state, partitions._rng.bit_generator.state
                )
                assert_same_buffer(self, train, restored.buffers["train"])
                self.assertEqual(restored.partition_size("gate"), 0)
                for field in ("episode_ids", "reuse_counts"):
                    np.testing.assert_array_equal(
                        getattr(restored, field)["train"], getattr(partitions, field)["train"]
                    )
        twin = ReplayPartitions(capacity=capacity, validation_fraction=0.0, device="cpu", seed=9)
        twin.load_state_dict(partitions.state_dict())
        for i in range(adds, adds + 4):
            for each in (partitions, twin):
                each.add(make_transition(i), episode_id=i // 4)
        assert_same_buffer(self, train, twin.buffers["train"])
        np.testing.assert_array_equal(twin.episode_ids["train"], partitions.episode_ids["train"])
        self.assertEqual(train.position, (adds + 4) % capacity)

    def test_full_capacity_partitions_state_still_loads(self):
        partitions = self.partitions(400)
        state = copy.deepcopy(partitions.state_dict())
        for name, buffer in partitions.buffers.items():
            state["buffers"][name]["arrays"] = {
                field: array.copy() for field, array in buffer._arrays.items()
            }
        state["episode_ids"] = {k: v.copy() for k, v in partitions.episode_ids.items()}
        state["reuse_counts"] = {k: v.copy() for k, v in partitions.reuse_counts.items()}
        restored = self.fresh_partitions()
        restored.load_state_dict(state)
        assert_same_partitions(self, partitions, restored)


# The stored columns and the batch keys of a buffer under reward_observation "interval"
# (the default), in order, as before the key existed.
INTERVAL_ARRAYS = (
    "states", "decisions", "durations", "horizons", "rewards", "next_states", "terminated",
    "truncated", "times", "next_times", "limit_touched", "near_limit",
)
INTERVAL_BATCH = (
    "states", "decisions", "durations", "horizons", "rewards", "next_states", "terminated",
    "truncated", "contact_flags",
)


def end_transition(i, terminated=False, truncated=False):
    """make_transition(i) with its end-state rate 0.5 + i (the state's first entry + 0.5)."""
    return dict(make_transition(i, terminated, truncated), end_reward_rate=0.5 + i)


class TestEndStateColumn(unittest.TestCase):
    """reward_observation "end_state" (common.ph_config.REWARD_OBSERVATION_KEY): the column
    end_rewards, batches with end_rewards and without rewards, and states of one mode
    refused by the other; "interval" is the buffer of every earlier run."""

    def filled(self, capacity, rows, *, end_state=True, seed=0):
        observation = "end_state" if end_state else "interval"
        buffer = PHReplayBuffer(capacity, seed=seed, reward_observation=observation)
        make = end_transition if end_state else make_transition
        for i in range(rows):
            buffer.add(make(i, terminated=i % 7 == 0, truncated=i % 5 == 0))
        buffer.sample(3)
        return buffer

    def test_default_buffer_is_unchanged(self):
        buffer = PHReplayBuffer(16, seed=0)
        self.assertEqual(buffer.reward_observation, "interval")
        stated = PHReplayBuffer(16, seed=0, reward_observation="interval")
        for i in range(5):
            # A transition with an end-state rate is stored as one without it.
            buffer.add(end_transition(i))
            stated.add(make_transition(i))
        assert_same_buffer(self, buffer, stated)
        self.assertEqual(list(buffer._arrays), list(INTERVAL_ARRAYS))
        batch = buffer.batch(np.arange(5))
        self.assertEqual(list(batch), list(INTERVAL_BATCH))
        self.assertEqual(list(buffer.sample(4)), list(INTERVAL_BATCH))
        self.assertEqual(list(buffer.state_dict()["arrays"]), list(INTERVAL_ARRAYS))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            buffer.save(path)
            with np.load(path) as stored:
                self.assertEqual(set(stored.files), {"meta", *INTERVAL_ARRAYS})
        partitions = ReplayPartitions(capacity=40, validation_fraction=0.25, device="cpu", seed=0)
        self.assertEqual(partitions.reward_observation, "interval")
        for buffer in partitions.buffers.values():
            self.assertEqual(buffer.reward_observation, "interval")

    def test_end_state_rows_store_the_end_rate_and_hide_the_interval_reward(self):
        buffer = PHReplayBuffer(16, seed=0, reward_observation="end_state")
        for i in range(5):
            buffer.add(end_transition(i, terminated=i == 2))
        self.assertEqual(list(buffer._arrays), [*INTERVAL_ARRAYS, "end_rewards"])
        self.assertEqual(buffer._arrays["end_rewards"].shape, (16, 1))
        batch = buffer.batch(np.arange(5))
        expected_keys = [
            "end_rewards" if key == "rewards" else key for key in INTERVAL_BATCH
        ]
        self.assertEqual(list(batch), expected_keys)
        self.assertNotIn("rewards", batch)
        with self.assertRaises(KeyError):
            batch["rewards"]
        self.assertEqual(batch["end_rewards"].shape, (5, 1))
        self.assertEqual(batch["end_rewards"].dtype, torch.get_default_dtype())
        self.assertTrue(
            torch.equal(
                batch["end_rewards"][:, 0],
                torch.tensor([0.5 + i for i in range(5)], dtype=torch.get_default_dtype()),
            )
        )
        # The interval rewards stay stored (for offline checks) but are never handed out.
        np.testing.assert_array_equal(buffer._arrays["rewards"][:5, 0], [-float(i) for i in range(5)])
        sampled = buffer.sample(64)
        self.assertNotIn("rewards", sampled)
        self.assertTrue(torch.equal(sampled["end_rewards"][:, 0], sampled["states"][:, 0] + 0.5))
        # Both partitions of a ReplayPartitions take the mode, and every draw hides rewards.
        # 60 rows at a held-out fraction 0.25: about 15 held-out rows (fewer than 2 has
        # probability below 1e-6).
        partitions = ReplayPartitions(
            capacity=80, validation_fraction=0.25, device="cpu", seed=0,
            reward_observation="end_state",
        )
        for buffer in partitions.buffers.values():
            self.assertEqual(buffer.reward_observation, "end_state")
        for i in range(60):
            partitions.add(end_transition(i), episode_id=i)
        size = partitions.partition_size("gate")
        self.assertGreaterEqual(size, 2)
        for drawn in (
            partitions.sample("train", 8),
            partitions.sample("gate", 2),
            partitions.draw_for_allocation("gate", 2),
        ):
            self.assertNotIn("rewards", drawn)
            self.assertTrue(torch.equal(drawn["end_rewards"][:, 0], drawn["states"][:, 0] + 0.5))

    def test_end_state_needs_the_field(self):
        buffer = PHReplayBuffer(4, seed=0, reward_observation="end_state")
        with self.assertRaisesRegex(ValueError, "end_reward_rate"):
            buffer.add(make_transition(0))
        self.assertEqual(buffer.size(), 0)
        self.assertIsNone(buffer._arrays)
        buffer.add(end_transition(0))
        with self.assertRaisesRegex(ValueError, "end_reward_rate"):
            buffer.add(make_transition(1))
        self.assertEqual((buffer.size(), buffer.position), (1, 1))
        for bad in ("end", "End_state", None, 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "reward_observation"):
                PHReplayBuffer(4, seed=0, reward_observation=bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "reward_observation"):
                ReplayPartitions(
                    capacity=4, validation_fraction=0.0, device="cpu", seed=0,
                    reward_observation=bad,
                )

    def test_state_dict_and_npz_round_trip_with_the_column(self):
        # Partly filled, and a wrapped ring.
        for capacity, rows in ((100, 10), (8, 13)):
            with self.subTest(capacity=capacity, rows=rows):
                buffer = self.filled(capacity, rows)
                state = buffer.state_dict()
                self.assertIn("end_rewards", state["arrays"])
                self.assertEqual(state["arrays"]["end_rewards"].shape, (min(rows, capacity), 1))
                restored = PHReplayBuffer(5, seed=99, reward_observation="end_state")
                restored.load_state_dict(state)
                assert_same_buffer(self, buffer, restored)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "replay.npz"
                    buffer.save(path)
                    with np.load(path) as stored:
                        self.assertIn("end_rewards", stored.files)
                    loaded = PHReplayBuffer(1, seed=99, reward_observation="end_state")
                    loaded.load(path)
                assert_same_buffer(self, buffer, loaded)
                for i in range(rows, rows + 3):
                    for each in (buffer, restored, loaded):
                        each.add(end_transition(i))
                expected = buffer.sample(16)
                for other in (restored, loaded):
                    actual = other.sample(16)
                    self.assertEqual(list(actual), list(expected))
                    for name, tensor in expected.items():
                        self.assertTrue(torch.equal(tensor, actual[name]), name)
        partitions = ReplayPartitions(
            capacity=80, validation_fraction=0.25, device="cpu", seed=0,
            reward_observation="end_state",
        )
        for i in range(60):
            partitions.add(end_transition(i), episode_id=i // 4)
        partitions.sample("gate", 2)
        for copy_rows in (True, False):
            with self.subTest(copy_rows=copy_rows):
                restored = ReplayPartitions(
                    capacity=80, validation_fraction=0.25, device="cpu", seed=9,
                    reward_observation="end_state",
                )
                restored.load_state_dict(partitions.state_dict(copy=copy_rows))
                assert_same_partitions(self, partitions, restored)

    def test_a_state_of_the_other_mode_is_refused(self):
        interval, end = self.filled(16, 6, end_state=False), self.filled(16, 6)
        target = PHReplayBuffer(16, seed=1, reward_observation="end_state")
        with self.assertRaisesRegex(ValueError, "reward_observation"):
            target.load_state_dict(interval.state_dict())
        plain = PHReplayBuffer(16, seed=1)
        with self.assertRaisesRegex(ValueError, "reward_observation"):
            plain.load_state_dict(end.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            interval_path, end_path = Path(directory) / "a.npz", Path(directory) / "b.npz"
            interval.save(interval_path)
            end.save(end_path)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                target.load(interval_path)
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                plain.load(end_path)
        # A refused load changes nothing.
        for buffer in (target, plain):
            self.assertEqual((buffer.capacity, buffer.position, buffer.size()), (16, 0, 0))
            self.assertIsNone(buffer._arrays)
        # An empty state carries no column and loads in either mode.
        target.load_state_dict(PHReplayBuffer(16, seed=2).state_dict())
        self.assertEqual(target.size(), 0)
        # Partitions refuse the other mode before anything changes.
        keyed = ReplayPartitions(
            capacity=40, validation_fraction=0.25, device="cpu", seed=3,
            reward_observation="end_state",
        )
        default = ReplayPartitions(capacity=40, validation_fraction=0.25, device="cpu", seed=4)
        for i in range(30):
            keyed.add(end_transition(i), episode_id=i)
            default.add(make_transition(i), episode_id=i)
        for source, sink in ((default, keyed), (keyed, default)):
            routing = copy.deepcopy(sink._rng.bit_generator.state)
            sizes = {name: sink.partition_size(name) for name in ("train", "gate")}
            with self.assertRaisesRegex(ValueError, "reward_observation"):
                sink.load_state_dict(source.state_dict())
            self.assertEqual(sink._rng.bit_generator.state, routing)
            self.assertEqual({name: sink.partition_size(name) for name in ("train", "gate")}, sizes)


if __name__ == "__main__":
    unittest.main()
