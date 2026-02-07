"""Unit tests for ZeroGAgent.

Tests agent construction, action selection, training updates, mode switching,
and state persistence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.agents.zero_g_agent import ZeroGAgent
from src.config import SystemConfig

# ---------------------------------------------------------------------------
# Agent construction tests
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    """Test agent initialization and configuration."""

    def test_creates_with_default_config(self, system_config: SystemConfig) -> None:
        """Test that agent constructs successfully with default SystemConfig."""
        agent = ZeroGAgent(system_config)

        # Verify network exists
        assert agent.network is not None
        assert hasattr(agent.network, "parameters")

        # Verify optimizer exists
        assert agent.optimizer is not None
        assert hasattr(agent.optimizer, "step")

    def test_device_is_cpu_when_no_gpu(self, system_config: SystemConfig) -> None:
        """Test that device is CPU when use_gpu=False in config."""
        assert system_config.use_gpu is False
        agent = ZeroGAgent(system_config)

        assert agent.device == torch.device("cpu")
        assert str(agent.device) == "cpu"

    def test_custom_skill_blend_alpha(self, system_config: SystemConfig) -> None:
        """Test that custom skill_blend_alpha is correctly stored."""
        alpha = 0.3
        agent = ZeroGAgent(system_config, skill_blend_alpha=alpha)

        # Access private attribute to verify it was set correctly
        assert agent._skill_blend_alpha == alpha


# ---------------------------------------------------------------------------
# Action selection tests
# ---------------------------------------------------------------------------


class TestSelectAction:
    """Test action selection in various modes."""

    def test_mcts_returns_valid_action(
        self,
        system_config: SystemConfig,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test that MCTS mode returns a valid action with correct shape and range."""
        agent = ZeroGAgent(system_config)
        agent.eval_mode()

        action, info = agent.select_action(sample_observation, use_mcts=True)

        # Verify shape
        assert action.shape == (6,)

        # Verify range [-1, 1]
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

        # Verify it's float32
        assert action.dtype == np.float32

        # Verify info dict is returned
        assert isinstance(info, dict)

    def test_direct_network_returns_valid_action(
        self,
        system_config: SystemConfig,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test that direct network inference (no MCTS) returns valid action."""
        agent = ZeroGAgent(system_config)
        agent.eval_mode()

        action, info = agent.select_action(sample_observation, use_mcts=False)

        # Verify shape
        assert action.shape == (6,)

        # Verify range [-1, 1]
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

        # Verify info contains network_action flag
        assert "network_action" in info
        assert info["network_action"] == 1.0

    def test_deterministic_vs_stochastic(
        self,
        system_config: SystemConfig,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test that deterministic mode gives same action twice, stochastic may differ."""
        agent = ZeroGAgent(system_config)
        agent.eval_mode()

        # Deterministic: should give same action twice
        action1, _ = agent.select_action(sample_observation, use_mcts=False, deterministic=True)
        action2, _ = agent.select_action(sample_observation, use_mcts=False, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)

        # Stochastic: may differ (though not guaranteed to differ with fixed seed)
        # We just verify it runs without error
        action3, _ = agent.select_action(sample_observation, use_mcts=False, deterministic=False)
        assert action3.shape == (6,)

    def test_action_shape_matches_config(
        self,
        system_config: SystemConfig,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test that action dimension matches network config."""
        expected_action_dim = system_config.network.action_dim
        assert expected_action_dim == 6

        agent = ZeroGAgent(system_config)
        agent.eval_mode()

        action, _ = agent.select_action(sample_observation, use_mcts=False)

        assert action.shape == (expected_action_dim,)


# ---------------------------------------------------------------------------
# Update / training tests
# ---------------------------------------------------------------------------


class TestUpdate:
    """Test gradient update step."""

    @pytest.fixture()
    def sample_batch(self, system_config: SystemConfig) -> dict[str, Any]:
        """Create a sample training batch."""
        batch_size = 4
        res = system_config.network.voxel_resolution
        ch = system_config.network.voxel_channels

        return {
            "voxels": np.random.randn(batch_size, ch, res, res, res).astype(np.float32),
            "proprio": np.random.randn(batch_size, 13).astype(np.float32),
            "action": np.random.randn(batch_size, 6).astype(np.float32),
            "value_target": np.random.randn(batch_size).astype(np.float32),
        }

    def test_update_returns_loss_dict(
        self,
        system_config: SystemConfig,
        sample_batch: dict[str, Any],
    ) -> None:
        """Test that update returns a dict with loss metrics."""
        agent = ZeroGAgent(system_config)

        losses = agent.update(sample_batch)

        # Verify expected keys
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses

        # Verify all values are floats
        assert isinstance(losses["policy_loss"], float)
        assert isinstance(losses["value_loss"], float)
        assert isinstance(losses["entropy"], float)

    def test_update_modifies_parameters(
        self,
        system_config: SystemConfig,
        sample_batch: dict[str, Any],
    ) -> None:
        """Test that update actually modifies network parameters."""
        agent = ZeroGAgent(system_config)

        # Snapshot parameters before update
        param_before = [p.clone() for p in agent.network.parameters()]

        # Run update
        agent.update(sample_batch)

        # Verify at least some parameters changed
        param_after = list(agent.network.parameters())
        assert len(param_before) == len(param_after)

        # Check that at least one parameter changed
        params_changed = False
        for p_before, p_after in zip(param_before, param_after):
            if not torch.allclose(p_before, p_after):
                params_changed = True
                break

        assert params_changed, "No parameters were updated"

    def test_update_batch_format(
        self,
        system_config: SystemConfig,
    ) -> None:
        """Test that update works with the expected batch format from ReplayBuffer."""
        agent = ZeroGAgent(system_config)

        batch_size = 4
        res = system_config.network.voxel_resolution
        ch = system_config.network.voxel_channels

        # Create batch in expected format
        batch = {
            "voxels": np.random.randn(batch_size, ch, res, res, res).astype(np.float32),
            "proprio": np.random.randn(batch_size, 13).astype(np.float32),
            "action": np.random.randn(batch_size, 6).astype(np.float32),
            "value_target": np.random.randn(batch_size).astype(np.float32),
        }

        # Should not raise any errors
        losses = agent.update(batch)

        assert isinstance(losses, dict)
        assert len(losses) > 0


# ---------------------------------------------------------------------------
# Mode switching tests
# ---------------------------------------------------------------------------


class TestModeSwitching:
    """Test train/eval mode switching."""

    def test_train_mode(self, system_config: SystemConfig) -> None:
        """Test that train_mode sets network to training mode."""
        agent = ZeroGAgent(system_config)

        # Set to eval first
        agent.eval_mode()

        # Switch to train
        agent.train_mode()

        # Verify network is in training mode
        assert agent.network.training is True

    def test_eval_mode(self, system_config: SystemConfig) -> None:
        """Test that eval_mode sets network to evaluation mode."""
        agent = ZeroGAgent(system_config)

        # Set to train first
        agent.train_mode()

        # Switch to eval
        agent.eval_mode()

        # Verify network is in eval mode
        assert agent.network.training is False


# ---------------------------------------------------------------------------
# State dict tests
# ---------------------------------------------------------------------------


class TestStateDict:
    """Test save/load functionality."""

    def test_save_and_load(self, system_config: SystemConfig) -> None:
        """Test that state_dict can be saved and loaded to a new agent."""
        # Create first agent and get its state
        agent1 = ZeroGAgent(system_config)
        state = agent1.state_dict()

        # Verify state has expected keys
        assert "network" in state
        assert "optimizer" in state

        # Create second agent and load state
        agent2 = ZeroGAgent(system_config)
        agent2.load_state_dict(state)

        # Verify parameters match
        for p1, p2 in zip(agent1.network.parameters(), agent2.network.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_determinism_after_load(
        self,
        system_config: SystemConfig,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test that after loading state, same input gives same output."""
        # Create agent, get action, save state
        agent1 = ZeroGAgent(system_config)
        agent1.eval_mode()

        action1, _ = agent1.select_action(sample_observation, use_mcts=False, deterministic=True)
        state = agent1.state_dict()

        # Create new agent, load state, get action
        agent2 = ZeroGAgent(system_config)
        agent2.load_state_dict(state)
        agent2.eval_mode()

        action2, _ = agent2.select_action(sample_observation, use_mcts=False, deterministic=True)

        # Actions should match
        np.testing.assert_array_almost_equal(action1, action2)
