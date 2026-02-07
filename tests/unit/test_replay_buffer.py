"""Unit tests for the experience replay buffer.

Covers ReplayBuffer creation, transition management, FIFO eviction,
sampling semantics, batch tensor collation, persistence to Parquet,
and the clear() method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.replay_buffer.buffer import Episode, ReplayBuffer, Transition

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

# Use small array shapes to keep tests fast.
VOXEL_SHAPE = (2, 4, 4, 4)
PROPRIO_DIM = 6
GOAL_DIM = 7
ACTION_DIM = 6


def _make_transition(
    reward: float = 0.0,
    value_target: float = 0.0,
    done: bool = False,
    seed: int | None = None,
) -> Transition:
    """Create a Transition with small random arrays."""
    rng = np.random.RandomState(seed)
    return Transition(
        voxels=rng.randn(*VOXEL_SHAPE).astype(np.float32),
        proprio=rng.randn(PROPRIO_DIM).astype(np.float32),
        goal=rng.randn(GOAL_DIM).astype(np.float32),
        action=rng.randn(ACTION_DIM).astype(np.float32),
        reward=reward,
        value_target=value_target,
        done=done,
    )


def _fill_buffer(buf: ReplayBuffer, n: int) -> list[Transition]:
    """Add *n* transitions to *buf* and return them."""
    transitions = [_make_transition(reward=float(i), seed=i) for i in range(n)]
    for t in transitions:
        buf.add_transition(t)
    return transitions


# ================================================================== #
# ReplayBuffer creation
# ================================================================== #


class TestReplayBufferCreation:
    """Verify constructor behaviour and initial state."""

    def test_buffer_starts_empty(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_capacity_is_stored(self):
        buf = ReplayBuffer(capacity=42)
        assert buf.capacity == 42

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=-5)

    def test_persist_dir_created(self, tmp_path: Path):
        subdir = tmp_path / "replay_data"
        ReplayBuffer(capacity=10, persist_dir=subdir)
        assert subdir.is_dir()


# ================================================================== #
# add_transition and basic length
# ================================================================== #


class TestAddTransition:
    """Verify that transitions are added correctly."""

    def test_add_single_transition(self):
        buf = ReplayBuffer(capacity=10)
        buf.add_transition(_make_transition())
        assert len(buf) == 1

    def test_add_multiple_transitions(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer(buf, 5)
        assert len(buf) == 5

    def test_fill_to_capacity(self):
        cap = 10
        buf = ReplayBuffer(capacity=cap)
        _fill_buffer(buf, cap)
        assert len(buf) == cap


# ================================================================== #
# FIFO eviction
# ================================================================== #


class TestFIFOEviction:
    """Oldest transitions should be evicted when buffer is full."""

    def test_eviction_keeps_size_at_capacity(self):
        cap = 5
        buf = ReplayBuffer(capacity=cap)
        _fill_buffer(buf, cap + 3)
        assert len(buf) == cap

    def test_oldest_transition_is_evicted(self):
        cap = 3
        buf = ReplayBuffer(capacity=cap)
        transitions = _fill_buffer(buf, cap)

        # Add one more -- reward=99 to distinguish it
        newest = _make_transition(reward=99.0)
        buf.add_transition(newest)

        assert len(buf) == cap
        # The oldest (transitions[0], reward=0.0) should be gone.
        remaining_rewards = [t.reward for t in buf._buffer]
        assert transitions[0].reward not in remaining_rewards
        assert 99.0 in remaining_rewards

    def test_eviction_order_preserves_newest(self):
        cap = 4
        buf = ReplayBuffer(capacity=cap)
        _fill_buffer(buf, cap + 6)
        # After adding cap+6 items to a buffer of size cap, the last cap items survive
        expected_rewards = [float(i) for i in range(6, cap + 6)]
        actual_rewards = [t.reward for t in buf._buffer]
        assert actual_rewards == expected_rewards


# ================================================================== #
# sample
# ================================================================== #


class TestSample:
    """Verify sampling semantics and error handling."""

    def test_sample_returns_correct_count(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer(buf, 20)
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_returns_transition_objects(self):
        buf = ReplayBuffer(capacity=50)
        _fill_buffer(buf, 10)
        batch = buf.sample(3)
        for item in batch:
            assert isinstance(item, Transition)

    def test_sample_full_buffer(self):
        cap = 10
        buf = ReplayBuffer(capacity=cap)
        _fill_buffer(buf, cap)
        batch = buf.sample(cap)
        assert len(batch) == cap

    def test_sample_raises_when_batch_exceeds_buffer(self):
        buf = ReplayBuffer(capacity=100)
        _fill_buffer(buf, 3)
        with pytest.raises(ValueError, match="Requested batch_size=5"):
            buf.sample(5)

    def test_sample_raises_on_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError, match="Requested batch_size=1"):
            buf.sample(1)

    def test_sample_single_element(self):
        buf = ReplayBuffer(capacity=10)
        t = _make_transition(reward=42.0)
        buf.add_transition(t)
        batch = buf.sample(1)
        assert len(batch) == 1
        assert batch[0].reward == 42.0


# ================================================================== #
# add_episode
# ================================================================== #


class TestAddEpisode:
    """Verify that add_episode ingests all transitions from an Episode."""

    def test_add_episode_all_transitions_present(self):
        buf = ReplayBuffer(capacity=100)
        transitions = [_make_transition(reward=float(i)) for i in range(5)]
        episode = Episode(
            transitions=transitions,
            total_reward=sum(t.reward for t in transitions),
            length=len(transitions),
            success=True,
        )
        buf.add_episode(episode)
        assert len(buf) == 5

    def test_add_episode_rewards_match(self):
        buf = ReplayBuffer(capacity=100)
        rewards = [1.0, 2.0, 3.0]
        transitions = [_make_transition(reward=r) for r in rewards]
        episode = Episode(transitions=transitions, total_reward=6.0, length=3)
        buf.add_episode(episode)

        actual_rewards = sorted(t.reward for t in buf._buffer)
        assert actual_rewards == sorted(rewards)

    def test_add_empty_episode(self):
        buf = ReplayBuffer(capacity=100)
        episode = Episode(transitions=[], total_reward=0.0, length=0)
        buf.add_episode(episode)
        assert len(buf) == 0

    def test_add_episode_respects_capacity(self):
        cap = 3
        buf = ReplayBuffer(capacity=cap)
        transitions = [_make_transition(reward=float(i)) for i in range(5)]
        episode = Episode(transitions=transitions, total_reward=10.0, length=5)
        buf.add_episode(episode)
        assert len(buf) == cap
        # Only the last 3 should survive
        expected_rewards = [2.0, 3.0, 4.0]
        actual_rewards = [t.reward for t in buf._buffer]
        assert actual_rewards == expected_rewards


# ================================================================== #
# sample_batch_tensors
# ================================================================== #


class TestSampleBatchTensors:
    """Verify that sample_batch_tensors returns correct shapes and types."""

    @pytest.fixture()
    def filled_buffer(self) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=50)
        _fill_buffer(buf, 20)
        return buf

    def test_returns_dict_with_expected_keys(self, filled_buffer: ReplayBuffer):
        batch = filled_buffer.sample_batch_tensors(4)
        expected_keys = {"voxels", "proprio", "goal", "action", "reward", "value_target", "done"}
        assert set(batch.keys()) == expected_keys

    def test_voxels_shape(self, filled_buffer: ReplayBuffer):
        bs = 4
        batch = filled_buffer.sample_batch_tensors(bs)
        assert batch["voxels"].shape == (bs, *VOXEL_SHAPE)

    def test_proprio_shape(self, filled_buffer: ReplayBuffer):
        bs = 4
        batch = filled_buffer.sample_batch_tensors(bs)
        assert batch["proprio"].shape == (bs, PROPRIO_DIM)

    def test_goal_shape(self, filled_buffer: ReplayBuffer):
        bs = 4
        batch = filled_buffer.sample_batch_tensors(bs)
        assert batch["goal"].shape == (bs, GOAL_DIM)

    def test_action_shape(self, filled_buffer: ReplayBuffer):
        bs = 4
        batch = filled_buffer.sample_batch_tensors(bs)
        assert batch["action"].shape == (bs, ACTION_DIM)

    def test_scalar_arrays_shape(self, filled_buffer: ReplayBuffer):
        bs = 4
        batch = filled_buffer.sample_batch_tensors(bs)
        assert batch["reward"].shape == (bs,)
        assert batch["value_target"].shape == (bs,)
        assert batch["done"].shape == (bs,)

    def test_scalar_arrays_dtype(self, filled_buffer: ReplayBuffer):
        batch = filled_buffer.sample_batch_tensors(4)
        assert batch["reward"].dtype == np.float32
        assert batch["value_target"].dtype == np.float32
        assert batch["done"].dtype == np.float32

    def test_batch_size_one(self, filled_buffer: ReplayBuffer):
        batch = filled_buffer.sample_batch_tensors(1)
        assert batch["voxels"].shape[0] == 1
        assert batch["reward"].shape == (1,)


# ================================================================== #
# clear()
# ================================================================== #


class TestClear:
    """Verify that clear() empties the buffer."""

    def test_clear_empties_buffer(self):
        buf = ReplayBuffer(capacity=50)
        _fill_buffer(buf, 20)
        assert len(buf) > 0
        buf.clear()
        assert len(buf) == 0

    def test_clear_then_add(self):
        buf = ReplayBuffer(capacity=50)
        _fill_buffer(buf, 10)
        buf.clear()
        buf.add_transition(_make_transition(reward=77.0))
        assert len(buf) == 1
        assert buf._buffer[0].reward == 77.0

    def test_clear_already_empty(self):
        buf = ReplayBuffer(capacity=10)
        buf.clear()  # should not raise
        assert len(buf) == 0

    def test_sample_after_clear_raises(self):
        buf = ReplayBuffer(capacity=50)
        _fill_buffer(buf, 10)
        buf.clear()
        with pytest.raises(ValueError):
            buf.sample(1)


# ================================================================== #
# save_to_parquet
# ================================================================== #


class TestSaveToParquet:
    """Verify Parquet persistence (uses tmp_path fixture)."""

    def test_save_creates_file(self, tmp_path: Path):
        buf = ReplayBuffer(capacity=50, persist_dir=tmp_path / "parquet")
        _fill_buffer(buf, 5)
        result = buf.save_to_parquet(tag="test")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".parquet"

    def test_save_filename_contains_tag(self, tmp_path: Path):
        buf = ReplayBuffer(capacity=50, persist_dir=tmp_path / "parquet")
        _fill_buffer(buf, 3)
        result = buf.save_to_parquet(tag="mysnap")
        assert result is not None
        assert "mysnap" in result.name

    def test_save_returns_none_without_persist_dir(self):
        buf = ReplayBuffer(capacity=50, persist_dir=None)
        _fill_buffer(buf, 5)
        result = buf.save_to_parquet()
        assert result is None

    def test_save_empty_buffer_returns_none(self, tmp_path: Path):
        buf = ReplayBuffer(capacity=50, persist_dir=tmp_path / "parquet")
        result = buf.save_to_parquet()
        assert result is None

    def test_saved_file_is_readable(self, tmp_path: Path):
        """Verify the saved Parquet file can be read back by pyarrow."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        buf = ReplayBuffer(capacity=50, persist_dir=tmp_path / "parquet")
        _fill_buffer(buf, 8)
        path = buf.save_to_parquet(tag="read_test")
        assert path is not None

        table = pq.read_table(path)
        assert table.num_rows == 8
        assert "reward" in table.column_names
        assert "done" in table.column_names
        assert "index" in table.column_names

    def test_save_multiple_snapshots(self, tmp_path: Path):
        buf = ReplayBuffer(capacity=50, persist_dir=tmp_path / "parquet")
        _fill_buffer(buf, 5)
        p1 = buf.save_to_parquet(tag="snap1")
        _fill_buffer(buf, 3)
        p2 = buf.save_to_parquet(tag="snap2")
        assert p1 is not None and p2 is not None
        assert p1 != p2
        assert p1.exists() and p2.exists()
