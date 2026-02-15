"""Integration tests for the full training loop.

These tests verify that the complete pipeline — environment creation,
self-play with MCTS, replay buffer storage, and policy gradient updates —
works end-to-end without errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SystemConfig, load_config
from src.environments.factory import make_env, make_envs
from src.logging_config import setup_logging
from src.training.trainer import Trainer


@pytest.fixture(autouse=True)
def _setup_logging() -> None:
    setup_logging(log_level="WARNING")


class TestEndToEndTraining:
    """Full training loop integration tests."""

    def test_single_episode_completes(self, system_config: SystemConfig) -> None:
        """A single self-play episode runs to completion."""
        system_config.training.num_episodes = 1
        system_config.training.checkpoint_interval = 1
        system_config.training.eval_interval = 1

        trainer = Trainer(config=system_config)
        summary = trainer.train()

        assert isinstance(summary, dict)
        assert "mean_episode_reward" in summary

    def test_multi_episode_training(self, system_config: SystemConfig) -> None:
        """Multiple episodes train without errors and produce checkpoints."""
        system_config.training.num_episodes = 3
        system_config.training.checkpoint_interval = 2
        system_config.training.eval_interval = 3

        trainer = Trainer(config=system_config)
        summary = trainer.train()

        assert "mean_episode_reward" in summary
        assert "mean_episode_length" in summary

        # Final checkpoint should exist
        ckpt_files = list(system_config.checkpoint_dir.glob("*.pt"))
        assert len(ckpt_files) >= 1

    def test_checkpoint_resume(self, system_config: SystemConfig) -> None:
        """Training can be paused and resumed from a checkpoint."""
        system_config.training.num_episodes = 2
        system_config.training.checkpoint_interval = 1

        # Phase 1: train 2 episodes
        trainer1 = Trainer(config=system_config)
        trainer1.train()

        # Find a checkpoint
        ckpt_files = sorted(system_config.checkpoint_dir.glob("*.pt"))
        assert len(ckpt_files) >= 1
        resume_path = ckpt_files[-1]

        # Phase 2: resume for 1 more episode
        system_config.training.num_episodes = 3
        trainer2 = Trainer(config=system_config, resume_from=resume_path)
        summary = trainer2.train()

        assert isinstance(summary, dict)

    def test_training_creates_replay_data(self, system_config: SystemConfig) -> None:
        """Training populates the replay buffer with transitions."""
        system_config.training.num_episodes = 2
        system_config.training.replay_buffer_size = 200

        trainer = Trainer(config=system_config)
        trainer.train()

        # The buffer should have received transitions
        assert len(trainer._replay) > 0


class TestEnvironmentIntegration:
    """Environment creation and stepping integration tests."""

    def test_make_env_from_config(self, system_config: SystemConfig) -> None:
        """Factory creates a usable environment."""
        env = make_env(system_config)
        obs, _info = env.reset(seed=42)

        assert "voxels" in obs
        assert "proprio" in obs
        assert "goal" in obs
        assert obs["proprio"].shape == (13,)

    def test_env_step_produces_valid_output(self, system_config: SystemConfig) -> None:
        """Stepping the environment returns well-formed outputs."""
        env = make_env(system_config)
        _obs, _ = env.reset(seed=42)

        action = np.zeros(6, dtype=np.float32)
        _next_obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "position_error" in info

    def test_episode_runs_to_truncation(self, system_config: SystemConfig) -> None:
        """An episode truncates at max_episode_steps."""
        env = make_env(system_config)
        _obs, _ = env.reset(seed=42)
        max_steps = system_config.environment.max_episode_steps

        step_count = 0
        done = False
        for _ in range(max_steps + 5):
            action = np.random.uniform(-1, 1, size=6).astype(np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            step_count += 1
            if terminated or truncated:
                done = True
                break

        assert done
        assert step_count <= max_steps

    def test_make_envs_creates_multiple(self, system_config: SystemConfig) -> None:
        """Factory creates the requested number of environments."""
        envs = make_envs(system_config, n=3)
        assert len(envs) == 3
        for env in envs:
            obs, _ = env.reset()
            assert "voxels" in obs

    def test_env_reset_with_different_seeds(self, system_config: SystemConfig) -> None:
        """Different seeds produce different initial states."""
        env = make_env(system_config)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)

        # Proprioception should differ (random initial pose)
        assert not np.allclose(obs1["proprio"], obs2["proprio"])


class TestConfigLoadIntegration:
    """Config loading from actual YAML files."""

    def test_smoke_test_config_loads(self) -> None:
        """The smoke_test.yaml config file loads and validates."""
        config = load_config("configs/smoke_test.yaml")
        assert config.environment.simulator == "mock"
        assert config.training.num_episodes == 1
        assert config.network.voxel_resolution == 16

    def test_quick_test_config_loads(self) -> None:
        """The quick_test.yaml config file loads and validates."""
        config = load_config("configs/quick_test.yaml")
        assert config.training.num_episodes == 10

    def test_docking_task_config_loads(self) -> None:
        """The docking_task.yaml config file loads and validates."""
        config = load_config("configs/docking_task.yaml")
        assert config.network.voxel_resolution == 64
        assert config.training.num_episodes == 1000

    def test_fixture_config_loads(self) -> None:
        """The test fixture config loads and validates."""
        config = load_config("tests/fixtures/configs/smoke_test.yaml")
        assert config.environment.simulator == "mock"
