"""Experience replay buffer with optional Parquet persistence.

Stores transition tuples from self-play episodes.  Supports in-memory
FIFO eviction and periodic flushing to Parquet files for long-term
storage and offline analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Transition:
    """A single RL transition from self-play.

    Attributes:
        voxels: Voxel observation ``(C, D, H, W)``.
        proprio: Proprioception vector.
        goal: Goal pose.
        action: Selected action.
        reward: Scalar reward.
        mcts_policy: MCTS-improved action probabilities (target for policy).
        value_target: Bootstrapped value target for value head.
        done: Whether the episode ended after this transition.
    """

    voxels: np.ndarray
    proprio: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    reward: float
    mcts_policy: np.ndarray | None = None
    value_target: float = 0.0
    done: bool = False


@dataclass
class Episode:
    """A complete self-play episode."""

    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    success: bool = False


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with optional Parquet snapshots.

    Args:
        capacity: Maximum number of transitions.  Oldest transitions are
            evicted first when the buffer is full.
        persist_dir: Optional directory for Parquet snapshots.
    """

    def __init__(self, capacity: int, persist_dir: Path | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer: list[Transition] = []
        self._persist_dir = persist_dir
        self._total_added: int = 0

        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info("replay_buffer_created", capacity=capacity, persist_dir=str(persist_dir))

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def add_transition(self, transition: Transition) -> None:
        """Add a single transition, evicting oldest if at capacity."""
        if len(self._buffer) >= self._capacity:
            self._buffer.pop(0)
        self._buffer.append(transition)
        self._total_added += 1

    def add_episode(self, episode: Episode) -> None:
        """Add all transitions from an episode."""
        for t in episode.transitions:
            self.add_transition(t)
        logger.debug(
            "episode_added_to_buffer",
            episode_length=episode.length,
            buffer_size=len(self._buffer),
            total_added=self._total_added,
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """Uniformly sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of ``Transition`` instances.

        Raises:
            ValueError: If *batch_size* exceeds buffer size.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has "
                f"{len(self._buffer)} transitions"
            )
        return random.sample(self._buffer, batch_size)

    def sample_batch_tensors(
        self, batch_size: int
    ) -> dict[str, np.ndarray | list[float]]:
        """Sample and collate a batch into stacked numpy arrays.

        Returns:
            Dict with keys: ``voxels``, ``proprio``, ``goal``, ``action``,
            ``reward``, ``value_target``, ``done``.
        """
        batch = self.sample(batch_size)
        return {
            "voxels": np.stack([t.voxels for t in batch]),
            "proprio": np.stack([t.proprio for t in batch]),
            "goal": np.stack([t.goal for t in batch]),
            "action": np.stack([t.action for t in batch]),
            "reward": np.array([t.reward for t in batch], dtype=np.float32),
            "value_target": np.array(
                [t.value_target for t in batch], dtype=np.float32
            ),
            "done": np.array([t.done for t in batch], dtype=np.float32),
        }

    def save_to_parquet(self, tag: str = "snapshot") -> Path | None:
        """Persist current buffer contents to a Parquet file.

        Args:
            tag: Filename tag for the snapshot.

        Returns:
            Path to the written file, or ``None`` if no persist_dir.
        """
        if self._persist_dir is None:
            logger.warning("persist_dir not set, skipping parquet save")
            return None

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed, skipping parquet save")
            return None

        if not self._buffer:
            return None

        records: list[dict[str, Any]] = []
        for i, t in enumerate(self._buffer):
            records.append(
                {
                    "index": i,
                    "reward": t.reward,
                    "value_target": t.value_target,
                    "done": t.done,
                    "action": t.action.tolist(),
                    "proprio": t.proprio.tolist(),
                }
            )

        table = pa.Table.from_pylist(records)
        path = self._persist_dir / f"replay_{tag}_{self._total_added}.parquet"
        pq.write_table(table, path, compression="zstd")
        logger.info("replay_buffer_saved", path=str(path), size=len(self._buffer))
        return path

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self._buffer.clear()
        logger.info("replay_buffer_cleared")
