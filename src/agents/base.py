"""Agent protocol for the ZeroG-RL system."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class AgentProtocol(Protocol):
    """Structural typing interface for RL agents."""

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        *,
        use_mcts: bool = True,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Select an action given an observation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.
            use_mcts: Whether to use MCTS search.
            deterministic: If True, use greedy action selection.

        Returns:
            ``(action, info)`` tuple.
        """
        ...

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient update on a batch of experience.

        Args:
            batch: Dict with keys ``voxels``, ``proprio``, ``action``,
                ``value_target`` (numpy arrays).

        Returns:
            Dict of loss metrics.
        """
        ...

    def train_mode(self) -> None:
        """Set agent to training mode."""
        ...

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpointing."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore agent state from a checkpoint."""
        ...
