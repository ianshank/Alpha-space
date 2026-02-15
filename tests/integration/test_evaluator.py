"""Integration tests for the standalone evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.checkpointing.checkpoint_manager import CheckpointManager
from src.config import SystemConfig
from src.evaluation.evaluator import Evaluator
from src.logging_config import setup_logging
from src.networks.policy_value_net import SpatialPolicyValueNetwork
from src.utils.common import get_device


@pytest.fixture(autouse=True)
def _setup_logging() -> None:
    setup_logging(log_level="WARNING")


@pytest.fixture()
def saved_checkpoint(system_config: SystemConfig) -> Path:
    """Train a tiny network and save a checkpoint."""
    device = get_device(prefer_gpu=False)
    net = SpatialPolicyValueNetwork(system_config.network).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    mgr = CheckpointManager(checkpoint_dir=system_config.checkpoint_dir)
    path = mgr.save(
        policy_net=net,
        optimizer=optimizer,
        episode=0,
        config=system_config,
        metrics={"test": 1.0},
    )
    return path


class TestEvaluator:
    def test_evaluator_runs(self, system_config: SystemConfig, saved_checkpoint: Path) -> None:
        """Evaluator completes N episodes and returns metrics."""
        evaluator = Evaluator(config=system_config, checkpoint_path=saved_checkpoint)
        result = evaluator.run(num_episodes=3, deterministic=True)

        assert "mean_reward" in result
        assert "success_rate" in result
        assert "mean_length" in result
        assert result["num_episodes"] == 3
        assert len(result["episode_rewards"]) == 3

    def test_evaluator_stochastic(
        self, system_config: SystemConfig, saved_checkpoint: Path
    ) -> None:
        """Evaluator runs with stochastic (non-deterministic) actions."""
        evaluator = Evaluator(config=system_config, checkpoint_path=saved_checkpoint)
        result = evaluator.run(num_episodes=2, deterministic=False)

        assert result["num_episodes"] == 2

    def test_evaluator_nonexistent_checkpoint_raises(self, system_config: SystemConfig) -> None:
        """Evaluator raises when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Evaluator(
                config=system_config,
                checkpoint_path=Path("/tmp/nonexistent.pt"),
            )
