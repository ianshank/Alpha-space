"""Unit tests for src.mcts.engine.

Covers MCTSNode properties, MCTSEngine.search, progressive widening,
greedy selection, and search statistics collection.  A mock predictor is
used so no real neural network is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.config import MCTSConfig
from src.mcts.engine import MCTSEngine, MCTSNode

# ---------------------------------------------------------------------------
# Mock predictor
# ---------------------------------------------------------------------------


class MockPredictor:
    """Deterministic mock that satisfies the PolicyValuePredictor protocol.

    Returns ``num_candidates`` random actions of the configured dimension and
    a fixed scalar value estimate on every call.
    """

    def __init__(
        self,
        action_dim: int = 6,
        num_candidates: int = 8,
        value: float = 0.5,
        *,
        seed: int = 0,
    ) -> None:
        self.action_dim = action_dim
        self.num_candidates = num_candidates
        self.value = value
        self._rng = np.random.RandomState(seed)
        self.call_count = 0

    def predict(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        self.call_count += 1
        actions = self._rng.randn(self.num_candidates, self.action_dim).astype(np.float32)
        return actions, self.value


# ---------------------------------------------------------------------------
# Mock environment model
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    step_count: int = 0


class MockEnvironmentModel:
    """Trivial environment model for state-restoring MCTS tests."""

    def __init__(self, action_dim: int = 6, max_steps: int = 100) -> None:
        self._state = _FakeState()
        self._action_dim = action_dim
        self._max_steps = max_steps

    def clone_state(self) -> _FakeState:
        return _FakeState(step_count=self._state.step_count)

    def set_state(self, state: object) -> None:
        assert isinstance(state, _FakeState)
        self._state = _FakeState(step_count=state.step_count)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool]:
        self._state.step_count += 1
        obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.zeros(13, dtype=np.float32),
        }
        reward = 0.0
        done = self._state.step_count >= self._max_steps
        return obs, reward, done


# ---------------------------------------------------------------------------
# Helper to build a dummy observation dict
# ---------------------------------------------------------------------------


def _dummy_obs() -> dict[str, np.ndarray]:
    return {
        "voxels": np.random.randn(4, 16, 16, 16).astype(np.float32),
        "proprio": np.random.randn(13).astype(np.float32),
    }


# ===================================================================
# MCTSNode
# ===================================================================


class TestMCTSNode:
    """Basic property tests for MCTSNode."""

    def test_default_values(self) -> None:
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert node.action is None
        assert node.children == {}
        assert node.reward == 0.0
        assert not node.is_terminal

    def test_q_value_zero_visits(self) -> None:
        """Q-value should be 0 when visit_count is 0."""
        node = MCTSNode()
        assert node.q_value == 0.0

    def test_q_value_computation(self) -> None:
        """Q = W / N."""
        node = MCTSNode(visit_count=4, value_sum=10.0)
        assert node.q_value == pytest.approx(2.5)

    def test_q_value_negative(self) -> None:
        node = MCTSNode(visit_count=2, value_sum=-6.0)
        assert node.q_value == pytest.approx(-3.0)

    def test_q_value_single_visit(self) -> None:
        node = MCTSNode(visit_count=1, value_sum=7.0)
        assert node.q_value == pytest.approx(7.0)

    def test_children_are_independent(self) -> None:
        """Each node must have its own children dict (no shared default)."""
        a = MCTSNode()
        b = MCTSNode()
        a.children[0] = MCTSNode(visit_count=99)
        assert 0 not in b.children

    def test_action_stored(self) -> None:
        action = np.array([1.0, 2.0, 3.0])
        node = MCTSNode(action=action)
        np.testing.assert_array_equal(node.action, action)

    def test_is_terminal_flag(self) -> None:
        node = MCTSNode(is_terminal=True)
        assert node.is_terminal


# ===================================================================
# MCTSEngine -- search basics
# ===================================================================


class TestMCTSEngineSearch:
    """Integration-level tests for MCTSEngine.search."""

    def test_search_returns_action_and_info(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        action, info = engine.search(_dummy_obs())
        assert isinstance(action, np.ndarray)
        assert isinstance(info, dict)

    def test_search_action_shape(self, mcts_config: MCTSConfig) -> None:
        action_dim = 6
        predictor = MockPredictor(action_dim=action_dim)
        engine = MCTSEngine(mcts_config, predictor)
        action, _ = engine.search(_dummy_obs())
        assert action.shape == (action_dim,)

    def test_search_action_is_finite(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        action, _ = engine.search(_dummy_obs())
        assert np.isfinite(action).all()

    def test_search_uses_predictor(self, mcts_config: MCTSConfig) -> None:
        """The predictor should be called at least once during search."""
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        engine.search(_dummy_obs())
        assert predictor.call_count >= 1

    def test_search_with_different_action_dims(self, mcts_config: MCTSConfig) -> None:
        for action_dim in [3, 6, 12]:
            predictor = MockPredictor(action_dim=action_dim)
            engine = MCTSEngine(mcts_config, predictor)
            action, _ = engine.search(_dummy_obs())
            assert action.shape == (action_dim,)


# ===================================================================
# MCTSEngine -- progressive widening
# ===================================================================


class TestProgressiveWidening:
    """Verify that the branching factor is bounded by max_children."""

    def test_children_bounded_by_max_children(self) -> None:
        """Root children must not exceed max_children no matter how many
        simulations are run."""
        config = MCTSConfig(
            num_simulations=50,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=20)
        engine = MCTSEngine(config, predictor)

        # Build the root manually to inspect it
        root = MCTSNode()
        obs = _dummy_obs()
        candidate_actions, root_value = predictor.predict(obs)
        root.visit_count = 1
        root.value_sum = root_value
        engine._expand_node(root, candidate_actions)

        assert len(root.children) <= config.max_children

    def test_expand_node_respects_budget(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=3,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=10)
        engine = MCTSEngine(config, predictor)

        node = MCTSNode()
        actions = np.random.randn(10, 6).astype(np.float32)
        engine._expand_node(node, actions)
        assert len(node.children) == 3  # max_children

    def test_expand_node_no_op_when_full(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=2,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=10)
        engine = MCTSEngine(config, predictor)

        node = MCTSNode()
        actions = np.random.randn(10, 6).astype(np.float32)
        engine._expand_node(node, actions)
        assert len(node.children) == 2

        # Second expansion should be a no-op
        engine._expand_node(node, actions)
        assert len(node.children) == 2

    def test_search_respects_max_children(self) -> None:
        """End-to-end: root never exceeds max_children after full search."""
        config = MCTSConfig(
            num_simulations=30,
            max_children=5,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=20)
        engine = MCTSEngine(config, predictor)
        _action, info = engine.search(_dummy_obs())
        assert info["num_children"] <= config.max_children


# ===================================================================
# MCTSEngine -- greedy selection (temperature=0)
# ===================================================================


class TestGreedySelection:
    """Test that temperature=0 yields greedy (most-visited) action selection."""

    def test_greedy_selects_most_visited(self) -> None:
        """With temperature 0, _select_action must pick the child with the
        highest visit count."""
        config = MCTSConfig(
            num_simulations=1,
            max_children=8,
            c_puct=1.0,
            temperature=0.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        # Build a synthetic root with known visit counts
        root = MCTSNode()
        best_action = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float32)
        root.children[0] = MCTSNode(visit_count=1, action=np.zeros(6, dtype=np.float32))
        root.children[1] = MCTSNode(visit_count=100, action=best_action)
        root.children[2] = MCTSNode(visit_count=5, action=np.ones(6, dtype=np.float32))

        selected = engine._select_action(root)
        np.testing.assert_array_equal(selected, best_action)

    def test_search_with_zero_temperature(self) -> None:
        """search() should work without errors when temperature is 0."""
        config = MCTSConfig(
            num_simulations=10,
            max_children=4,
            c_puct=1.0,
            temperature=0.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)
        action, _info = engine.search(_dummy_obs())
        assert action.shape == (6,)
        assert np.isfinite(action).all()

    def test_greedy_is_deterministic(self) -> None:
        """Greedy selection with the same tree must always pick the same child."""
        config = MCTSConfig(
            num_simulations=1,
            max_children=8,
            c_puct=1.0,
            temperature=0.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode()
        root.children[0] = MCTSNode(visit_count=3, action=np.zeros(6, dtype=np.float32))
        root.children[1] = MCTSNode(visit_count=10, action=np.ones(6, dtype=np.float32))

        a1 = engine._select_action(root)
        a2 = engine._select_action(root)
        np.testing.assert_array_equal(a1, a2)


# ===================================================================
# MCTSEngine -- search statistics
# ===================================================================


class TestSearchStatistics:
    """Verify that _gather_stats and search return meaningful diagnostics."""

    def test_info_contains_expected_keys(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        _, info = engine.search(_dummy_obs())
        expected_keys = {
            "root_visits",
            "num_children",
            "max_child_visits",
            "mean_child_q",
            "max_child_q",
            "min_child_q",
        }
        assert expected_keys.issubset(info.keys())

    def test_root_visits_equals_simulations_plus_one(
        self,
        mcts_config: MCTSConfig,
    ) -> None:
        """Root visit count = 1 (initial) + num_simulations (backup)."""
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        _, info = engine.search(_dummy_obs())
        assert info["root_visits"] == mcts_config.num_simulations + 1

    def test_num_children_positive(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        _, info = engine.search(_dummy_obs())
        assert info["num_children"] > 0

    def test_max_child_visits_positive(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        _, info = engine.search(_dummy_obs())
        assert info["max_child_visits"] >= 1

    def test_gather_stats_empty_root(self) -> None:
        """When root has no children, stats should contain only root_visits."""
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode(visit_count=5)
        stats = engine._gather_stats(root)
        assert stats == {"root_visits": 5.0}

    def test_q_value_stats_are_finite(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6, value=1.0)
        engine = MCTSEngine(mcts_config, predictor)
        _, info = engine.search(_dummy_obs())
        assert np.isfinite(info["mean_child_q"])
        assert np.isfinite(info["max_child_q"])
        assert np.isfinite(info["min_child_q"])


# ===================================================================
# MCTSEngine -- backup
# ===================================================================


class TestBackup:
    """Tests for the _backup value propagation method."""

    def test_backup_increments_visit_counts(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        nodes = [MCTSNode(), MCTSNode(), MCTSNode()]
        engine._backup(nodes, leaf_value=1.0)
        for node in nodes:
            assert node.visit_count == 1

    def test_backup_accumulates_value(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=1.0,  # no discounting
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode(reward=0.0)
        child = MCTSNode(reward=0.0)
        engine._backup([root, child], leaf_value=5.0)
        # child gets 0 + 1.0*5.0 = 5.0, root gets 0 + 1.0*5.0 = 5.0
        assert child.value_sum == pytest.approx(5.0)
        assert root.value_sum == pytest.approx(5.0)

    def test_backup_with_discount(self) -> None:
        discount = 0.5
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=discount,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode(reward=0.0)
        child = MCTSNode(reward=0.0)
        engine._backup([root, child], leaf_value=10.0)
        # child: value = reward(0) + discount(0.5) * 10.0 = 5.0
        # root:  value = reward(0) + discount(0.5) * 5.0  = 2.5
        assert child.value_sum == pytest.approx(5.0)
        assert root.value_sum == pytest.approx(2.5)

    def test_backup_with_rewards(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=1.0,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode(reward=2.0)
        child = MCTSNode(reward=3.0)
        engine._backup([root, child], leaf_value=1.0)
        # child: value = 3.0 + 1.0*1.0 = 4.0
        # root:  value = 2.0 + 1.0*4.0 = 6.0
        assert child.value_sum == pytest.approx(4.0)
        assert root.value_sum == pytest.approx(6.0)


# ===================================================================
# MCTSEngine -- with environment model
# ===================================================================


class TestMCTSWithEnvModel:
    """Tests for state-restoring MCTS using a mock environment model."""

    def test_search_with_env_model(self) -> None:
        config = MCTSConfig(
            num_simulations=8,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=6)
        env_model = MockEnvironmentModel(action_dim=6)
        engine = MCTSEngine(config, predictor)
        action, info = engine.search(_dummy_obs(), env_model=env_model)
        assert action.shape == (6,)
        assert np.isfinite(action).all()
        assert info["root_visits"] == config.num_simulations + 1


# ===================================================================
# MCTSEngine -- get_action_probs
# ===================================================================


class TestGetActionProbs:
    """Tests for get_action_probs (MCTS policy target extraction)."""

    def test_probs_sum_to_one(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        # Run a search to populate a tree, then build root manually for
        # controlled testing.
        root = MCTSNode()
        root.children[0] = MCTSNode(
            visit_count=3,
            action=np.zeros(6, dtype=np.float32),
        )
        root.children[1] = MCTSNode(
            visit_count=7,
            action=np.ones(6, dtype=np.float32),
        )
        actions, probs = engine.get_action_probs(root)
        assert len(actions) == 2
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)

    def test_probs_proportional_to_visits(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        root = MCTSNode()
        root.children[0] = MCTSNode(
            visit_count=10,
            action=np.zeros(6, dtype=np.float32),
        )
        root.children[1] = MCTSNode(
            visit_count=30,
            action=np.ones(6, dtype=np.float32),
        )
        _, probs = engine.get_action_probs(root)
        assert probs[0] == pytest.approx(0.25)
        assert probs[1] == pytest.approx(0.75)

    def test_empty_root_returns_empty(self, mcts_config: MCTSConfig) -> None:
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(mcts_config, predictor)
        root = MCTSNode()
        actions, probs = engine.get_action_probs(root)
        assert actions == []
        assert probs.shape == (0,)


# ===================================================================
# MCTSEngine -- edge cases
# ===================================================================


class TestEdgeCases:
    """Edge-case and error-handling tests."""

    def test_select_action_raises_on_empty_root(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)
        root = MCTSNode()
        with pytest.raises(RuntimeError, match="no children"):
            engine._select_action(root)

    def test_single_simulation(self) -> None:
        config = MCTSConfig(
            num_simulations=1,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6)
        engine = MCTSEngine(config, predictor)
        action, info = engine.search(_dummy_obs())
        assert action.shape == (6,)
        assert info["root_visits"] == 2.0  # 1 initial + 1 simulation

    def test_high_simulation_count(self) -> None:
        config = MCTSConfig(
            num_simulations=100,
            max_children=8,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=10)
        engine = MCTSEngine(config, predictor)
        action, info = engine.search(_dummy_obs())
        assert action.shape == (6,)
        assert info["root_visits"] == 101.0


# ===================================================================
# MCTSEngine -- Dirichlet noise
# ===================================================================


class TestDirichletNoise:
    """Tests for Dirichlet noise at root."""

    def test_noise_modifies_priors(self, mcts_config: MCTSConfig) -> None:
        """Dirichlet noise should modify child priors."""
        predictor = MockPredictor(action_dim=6, num_candidates=8)
        engine = MCTSEngine(mcts_config, predictor)

        # Build root with children that have uniform priors
        root = MCTSNode()
        candidate_actions, _ = predictor.predict(_dummy_obs())
        engine._expand_node(root, candidate_actions)

        # Record initial priors (should all be uniform)
        initial_priors = [child.prior for child in root.children.values()]
        initial_prior = 1.0 / mcts_config.max_children
        for prior in initial_priors:
            assert prior == pytest.approx(initial_prior)

        # Add Dirichlet noise
        engine._add_dirichlet_noise(root)

        # After noise: priors should differ from initial uniform
        modified_priors = [child.prior for child in root.children.values()]
        # At least some priors should have changed
        assert any(abs(p - initial_prior) > 1e-6 for p in modified_priors), (
            "Dirichlet noise should modify at least some priors"
        )

    def test_noise_sums_to_approximately_one(self, mcts_config: MCTSConfig) -> None:
        """After Dirichlet noise, priors should still approximately sum to 1."""
        predictor = MockPredictor(action_dim=6, num_candidates=8)
        engine = MCTSEngine(mcts_config, predictor)

        root = MCTSNode()
        candidate_actions, _ = predictor.predict(_dummy_obs())
        engine._expand_node(root, candidate_actions)

        # Add Dirichlet noise
        engine._add_dirichlet_noise(root)

        # Priors should sum to approximately 1
        prior_sum = sum(child.prior for child in root.children.values())
        assert prior_sum == pytest.approx(1.0, abs=1e-6)

    def test_epsilon_zero_no_noise(self) -> None:
        """With dirichlet_epsilon=0, priors should remain at initial values."""
        config = MCTSConfig(
            num_simulations=10,
            max_children=4,
            c_puct=1.0,
            temperature=1.0,
            discount=0.99,
            dirichlet_epsilon=0.0,  # No noise
            dirichlet_alpha=0.3,
        )
        predictor = MockPredictor(action_dim=6, num_candidates=8)
        engine = MCTSEngine(config, predictor)

        root = MCTSNode()
        candidate_actions, _ = predictor.predict(_dummy_obs())
        engine._expand_node(root, candidate_actions)

        # Record initial priors
        initial_priors = [child.prior for child in root.children.values()]

        # Add "noise" (but epsilon=0 so should be no-op)
        engine._add_dirichlet_noise(root)

        # Priors should be unchanged
        final_priors = [child.prior for child in root.children.values()]
        for init, final in zip(initial_priors, final_priors, strict=True):
            assert init == pytest.approx(final)
