"""Continuous-action MCTS engine with progressive widening.

Extends AlphaZero-style MCTS to continuous action spaces by sampling
candidate actions from a learned policy distribution and applying
*progressive widening* to bound the branching factor.

Algorithm (A0C — AlphaZero for Continuous control):
  1. SELECT — Walk from root using PUCT until reaching a leaf.
  2. EXPAND  — If visit-count triggers widening, sample new action from policy.
  3. EVALUATE — Use value network to estimate leaf value.
  4. BACKUP  — Propagate discounted value estimates up the tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import structlog

from src.config import MCTSConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols for dependency injection
# ---------------------------------------------------------------------------


class PolicyValuePredictor(Protocol):
    """Interface expected by MCTS for neural-network queries."""

    def predict(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """Return ``(sampled_actions, value)`` for a single observation.

        Args:
            observation: Dict with keys ``"voxels"`` and ``"proprio"``.

        Returns:
            A tuple of (sampled actions array of shape ``(K, action_dim)``,
            scalar value estimate).
        """
        ...


class EnvironmentModel(Protocol):
    """Interface for a *model* of the environment used inside MCTS rollouts.

    For the real system this is a lightweight copy of the simulation state,
    not the full physics engine.
    """

    def clone_state(self) -> object:
        """Return a serialisable snapshot of the current env state."""
        ...

    def set_state(self, state: object) -> None:
        """Restore environment to a previously cloned state."""
        ...

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool]:
        """Step environment, return ``(obs, reward, done)``."""
        ...


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A single node in the MCTS search tree.

    Attributes:
        visit_count: Number of times this node has been visited (N).
        value_sum: Accumulated backed-up value (W).
        prior: Prior probability from the policy network (P).
        action: The action that led to this node from its parent.
        children: Mapping from child index to ``MCTSNode``.
        reward: Immediate reward received when transitioning here.
        is_terminal: Whether this node represents a terminal state.
        env_state: Saved environment state (for state-restoring MCTS).
    """

    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    action: np.ndarray | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    reward: float = 0.0
    is_terminal: bool = False
    env_state: object | None = None

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, a) = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ---------------------------------------------------------------------------
# MCTS engine
# ---------------------------------------------------------------------------


class MCTSEngine:
    """Monte Carlo Tree Search for continuous action spaces.

    Uses progressive widening and PUCT selection, compatible with
    AlphaZero-style self-play.

    Args:
        config: MCTS hyper-parameters.
        predictor: Neural-network policy/value query interface.
    """

    def __init__(self, config: MCTSConfig, predictor: PolicyValuePredictor) -> None:
        self._config = config
        self._predictor = predictor

    def search(
        self,
        observation: dict[str, np.ndarray],
        env_model: EnvironmentModel | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run MCTS from the given observation and return an improved action.

        Args:
            observation: Current observation dict with ``"voxels"`` and
                ``"proprio"`` keys.
            env_model: Optional environment model for state-restoring MCTS.
                When ``None``, MCTS uses value-only backup (no real rollouts).

        Returns:
            ``(action, info)`` where *action* is the MCTS-improved action
            of shape ``(action_dim,)`` and *info* contains search statistics.
        """
        root = MCTSNode()
        candidate_actions, root_value = self._predictor.predict(observation)
        root.visit_count = 1
        root.value_sum = root_value

        if env_model is not None:
            root.env_state = env_model.clone_state()

        # Populate initial children from policy samples
        self._expand_node(root, candidate_actions)

        # Add Dirichlet noise at root for exploration
        self._add_dirichlet_noise(root)

        # Run simulations
        for sim_idx in range(self._config.num_simulations):
            node = root
            search_path: list[MCTSNode] = [node]

            # SELECT
            while node.children and not node.is_terminal:
                child_idx = self._select_child(node)
                node = node.children[child_idx]
                search_path.append(node)

            # EVALUATE leaf and BACKUP
            leaf_value = self._evaluate_leaf(
                node, search_path, observation, root, env_model
            )
            self._backup(search_path, leaf_value)

        # Select action from root children based on visit counts
        action = self._select_action(root)
        info = self._gather_stats(root)

        logger.debug(
            "mcts_search_complete",
            num_simulations=self._config.num_simulations,
            root_visits=root.visit_count,
            num_children=len(root.children),
            root_q=root.q_value,
        )

        return action, info

    def _evaluate_leaf(
        self,
        node: MCTSNode,
        search_path: list[MCTSNode],
        observation: dict[str, np.ndarray],
        root: MCTSNode,
        env_model: EnvironmentModel | None,
    ) -> float:
        """Evaluate a leaf node and optionally expand it.

        Returns:
            The estimated value of the leaf state.
        """
        if node.is_terminal:
            return 0.0

        if env_model is not None and node.action is not None:
            return self._evaluate_with_env(
                node, search_path, observation, root, env_model
            )

        # Value-only: re-predict at this state
        actions, leaf_value = self._predictor.predict(observation)
        if len(node.children) < self._config.max_children and node.visit_count > 0:
            self._expand_node(node, actions)
        return leaf_value

    def _evaluate_with_env(
        self,
        node: MCTSNode,
        search_path: list[MCTSNode],
        observation: dict[str, np.ndarray],
        root: MCTSNode,
        env_model: EnvironmentModel,
    ) -> float:
        """Evaluate a leaf by stepping the environment model along the search path."""
        env_model.set_state(root.env_state)
        obs = observation
        for path_node in search_path[1:]:
            if path_node.action is not None:
                obs, reward, done = env_model.step(path_node.action)
                path_node.reward = reward
                if done:
                    path_node.is_terminal = True
        node.env_state = env_model.clone_state()
        actions, leaf_value = self._predictor.predict(obs)
        if len(node.children) < self._config.max_children:
            self._expand_node(node, actions)
        return leaf_value

    def _expand_node(self, node: MCTSNode, candidate_actions: np.ndarray) -> None:
        """Add children to *node* from candidate action samples."""
        existing = len(node.children)
        budget = self._config.max_children - existing
        if budget <= 0:
            return

        n_new = min(budget, len(candidate_actions))
        for i in range(n_new):
            child_idx = existing + i
            child = MCTSNode(
                action=candidate_actions[i].copy(),
                prior=1.0 / self._config.max_children,  # uniform prior over samples
            )
            node.children[child_idx] = child

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into root child priors for exploration."""
        if not root.children:
            return
        eps = self._config.dirichlet_epsilon
        alpha = self._config.dirichlet_alpha
        n_children = len(root.children)
        noise = np.random.dirichlet([alpha] * n_children)
        for i, (idx, child) in enumerate(root.children.items()):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _select_child(self, node: MCTSNode) -> int:
        """Select child using PUCT (Polynomial UCT) criterion.

        UCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        total_visits = node.visit_count
        sqrt_total = math.sqrt(total_visits)

        best_score = -float("inf")
        best_idx = -1

        for idx, child in node.children.items():
            exploitation = child.q_value
            exploration = (
                self._config.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _backup(self, search_path: list[MCTSNode], leaf_value: float) -> None:
        """Propagate value estimate up the search path."""
        discount = self._config.discount
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            value = node.reward + discount * value
            node.value_sum += value

    def _select_action(self, root: MCTSNode) -> np.ndarray:
        """Select final action from root based on visit counts and temperature."""
        if not root.children:
            raise RuntimeError("MCTS root has no children — cannot select action")

        indices = sorted(root.children.keys())
        visits = np.array([root.children[i].visit_count for i in indices], dtype=np.float64)

        temp = self._config.temperature
        if temp < 1e-8:
            # Greedy: pick most-visited
            best = indices[int(np.argmax(visits))]
        else:
            # Softmax over visit counts
            logits = np.log(visits + 1e-8) / temp
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            best = indices[int(np.random.choice(len(indices), p=probs))]

        action = root.children[best].action
        if action is None:
            raise RuntimeError("Selected child has no action")
        return action

    def _gather_stats(self, root: MCTSNode) -> dict[str, float]:
        """Collect diagnostic statistics from the search tree."""
        if not root.children:
            return {"root_visits": float(root.visit_count)}

        child_visits = [c.visit_count for c in root.children.values()]
        child_qs = [c.q_value for c in root.children.values()]

        return {
            "root_visits": float(root.visit_count),
            "num_children": float(len(root.children)),
            "max_child_visits": float(max(child_visits)),
            "mean_child_q": float(np.mean(child_qs)),
            "max_child_q": float(max(child_qs)),
            "min_child_q": float(min(child_qs)),
        }

    def get_action_probs(self, root: MCTSNode) -> tuple[list[np.ndarray], np.ndarray]:
        """Return action-probability pairs from root visit counts.

        Used to construct MCTS-improved policy targets for training.

        Args:
            root: The root node after search.

        Returns:
            ``(actions, probs)`` — list of action arrays and normalised
            visit-count probabilities.
        """
        if not root.children:
            return [], np.array([])

        indices = sorted(root.children.keys())
        actions = [root.children[i].action for i in indices]
        visits = np.array(
            [root.children[i].visit_count for i in indices], dtype=np.float64
        )
        probs = visits / visits.sum() if visits.sum() > 0 else visits
        return actions, probs
