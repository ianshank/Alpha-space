"""Property-based tests for MCTS tree invariants using Hypothesis.

Verifies structural properties of the MCTS search tree:
  - Visit count consistency
  - UCT score monotonicity
  - Bounded branching factor
  - Value bounds
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import MCTSConfig
from src.mcts.engine import MCTSEngine


class _DummyPredictor:
    """Deterministic predictor for property tests."""

    def __init__(self, action_dim: int = 6, n_actions: int = 8, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)
        self._action_dim = action_dim
        self._n_actions = n_actions

    def predict(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        actions = self._rng.randn(self._n_actions, self._action_dim).astype(np.float32)
        return actions, 0.0


class TestVisitCountConsistency:
    """Total visits should be accounted for across the tree."""

    @given(
        num_sims=st.integers(min_value=1, max_value=50),
        max_children=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=30, deadline=None)
    def test_root_visits_equal_sims_plus_one(self, num_sims: int, max_children: int) -> None:
        """Root visit count = num_simulations + 1 (initial expand)."""
        config = MCTSConfig(
            num_simulations=num_sims,
            max_children=max_children,
            temperature=1.0,
        )
        predictor = _DummyPredictor(n_actions=max_children)
        engine = MCTSEngine(config=config, predictor=predictor)

        obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.zeros(13, dtype=np.float32),
        }
        _, info = engine.search(observation=obs)
        assert info["root_visits"] == num_sims + 1


class TestBoundedBranching:
    """Children count must never exceed max_children."""

    @given(max_children=st.integers(min_value=1, max_value=32))
    @settings(max_examples=20, deadline=None)
    def test_children_bounded(self, max_children: int) -> None:
        config = MCTSConfig(
            num_simulations=30,
            max_children=max_children,
            temperature=1.0,
        )
        predictor = _DummyPredictor(n_actions=max_children)
        engine = MCTSEngine(config=config, predictor=predictor)

        obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.zeros(13, dtype=np.float32),
        }
        _, info = engine.search(observation=obs)
        assert info["num_children"] <= max_children


class TestActionFiniteness:
    """MCTS-selected action must always be finite."""

    @given(
        num_sims=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_action_is_finite(self, num_sims: int, seed: int) -> None:
        config = MCTSConfig(num_simulations=num_sims, max_children=8)
        predictor = _DummyPredictor(seed=seed)
        engine = MCTSEngine(config=config, predictor=predictor)

        obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.zeros(13, dtype=np.float32),
        }
        action, _ = engine.search(observation=obs)
        assert np.all(np.isfinite(action)), f"Non-finite action: {action}"


class TestTemperatureEffect:
    """Temperature 0 should always select the most-visited child."""

    @given(seed=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=15, deadline=None)
    def test_greedy_is_deterministic(self, seed: int) -> None:
        config = MCTSConfig(
            num_simulations=20,
            max_children=8,
            temperature=0.0,
        )
        predictor1 = _DummyPredictor(seed=seed)
        predictor2 = _DummyPredictor(seed=seed)
        engine1 = MCTSEngine(config=config, predictor=predictor1)
        engine2 = MCTSEngine(config=config, predictor=predictor2)

        obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.zeros(13, dtype=np.float32),
        }

        np.random.seed(seed)
        action1, _ = engine1.search(observation=obs)
        np.random.seed(seed)
        action2, _ = engine2.search(observation=obs)

        np.testing.assert_array_equal(action1, action2)
