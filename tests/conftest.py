"""Shared pytest fixtures for the ZeroG-RL test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import (
    EnvironmentConfig,
    MCTSConfig,
    NetworkConfig,
    RewardConfig,
    SystemConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Device / seed
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    """CPU device for all tests (GPU not required in CI)."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def fixed_seed() -> int:
    """Seed all RNGs before every test for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Configs (small/fast values for testing)
# ---------------------------------------------------------------------------


@pytest.fixture()
def network_config() -> NetworkConfig:
    return NetworkConfig(
        voxel_resolution=16,
        voxel_channels=4,
        proprioception_dim=13,
        hidden_dim=64,
        num_res_blocks=1,
        action_dim=6,
    )


@pytest.fixture()
def mcts_config() -> MCTSConfig:
    return MCTSConfig(
        num_simulations=4,
        c_puct=1.0,
        max_children=4,
        temperature=1.0,
        discount=0.99,
    )


@pytest.fixture()
def training_config() -> TrainingConfig:
    return TrainingConfig(
        num_episodes=2,
        batch_size=4,
        learning_rate=1e-3,
        gradient_clip_norm=1.0,
        checkpoint_interval=1,
        eval_interval=1,
        replay_buffer_size=50,
        epochs_per_update=1,
    )


@pytest.fixture()
def env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        simulator="mock",
        parallel_envs=1,
        max_episode_steps=10,
        workspace_size=10.0,
        time_step=0.05,
        max_thrust=10.0,
        max_torque=2.0,
        spacecraft_mass=100.0,
        spacecraft_inertia=(10.0, 10.0, 10.0),
    )


@pytest.fixture()
def reward_config() -> RewardConfig:
    return RewardConfig()


@pytest.fixture()
def system_config(
    network_config: NetworkConfig,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    env_config: EnvironmentConfig,
    reward_config: RewardConfig,
    tmp_path: Path,
) -> SystemConfig:
    return SystemConfig(
        network=network_config,
        mcts=mcts_config,
        training=training_config,
        environment=env_config,
        reward=reward_config,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        seed=42,
        use_gpu=False,
        debug=True,
        no_logging=True,
    )


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_voxels(network_config: NetworkConfig) -> torch.Tensor:
    """Batch of random voxel observations."""
    res = network_config.voxel_resolution
    ch = network_config.voxel_channels
    return torch.randn(4, ch, res, res, res)


@pytest.fixture()
def sample_proprio() -> torch.Tensor:
    """Batch of random proprioception vectors."""
    return torch.randn(4, 13)


@pytest.fixture()
def sample_observation(network_config: NetworkConfig) -> dict[str, np.ndarray]:
    """Single observation dict (numpy)."""
    res = network_config.voxel_resolution
    ch = network_config.voxel_channels
    return {
        "voxels": np.random.randn(ch, res, res, res).astype(np.float32),
        "proprio": np.random.randn(13).astype(np.float32),
        "goal": np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
