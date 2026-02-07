"""Stand-alone policy evaluator and report generator.

Runs a trained policy on a configurable number of episodes without MCTS
and with deterministic (greedy) actions, then computes aggregate metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from src.checkpointing.checkpoint_manager import CheckpointManager
from src.config import SystemConfig
from src.environments.factory import make_env
from src.networks.policy_value_net import SpatialPolicyValueNetwork
from src.utils.common import get_device, seed_everything

logger = structlog.get_logger(__name__)


class Evaluator:
    """Evaluate a trained policy checkpoint.

    Args:
        config: System configuration.
        checkpoint_path: Path to the checkpoint file.
    """

    def __init__(self, config: SystemConfig, checkpoint_path: Path) -> None:
        self._config = config
        seed_everything(config.seed)
        self._device = get_device(prefer_gpu=config.use_gpu)

        self._network = SpatialPolicyValueNetwork(config.network).to(self._device)

        ckpt_mgr = CheckpointManager(checkpoint_dir=config.checkpoint_dir)
        ckpt_mgr.load(
            path=checkpoint_path,
            policy_net=self._network,
            device=str(self._device),
            restore_rng=False,
        )
        self._network.eval()
        logger.info("evaluator_initialized", checkpoint=str(checkpoint_path))

    def run(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Run evaluation episodes and return aggregate metrics.

        Args:
            num_episodes: How many episodes to run.
            deterministic: If ``True`` use the policy mean (greedy).

        Returns:
            Dict with ``mean_reward``, ``success_rate``, ``mean_length``,
            ``std_reward``, and per-episode ``episode_rewards``.
        """
        rewards: list[float] = []
        lengths: list[int] = []
        successes: list[bool] = []

        for ep_idx in range(num_episodes):
            reward, length, success = self._run_one(
                seed_offset=ep_idx,
                deterministic=deterministic,
            )
            rewards.append(reward)
            lengths.append(length)
            successes.append(success)

            logger.debug(
                "eval_episode",
                episode=ep_idx,
                reward=round(reward, 3),
                length=length,
                success=success,
            )

        result = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "num_episodes": num_episodes,
            "episode_rewards": rewards,
        }
        summary = {k: v for k, v in result.items() if k != "episode_rewards"}
        logger.info("evaluation_complete", **summary)
        return result

    def _run_one(
        self,
        seed_offset: int,
        deterministic: bool,
    ) -> tuple[float, int, bool]:
        """Execute a single evaluation episode."""
        env = make_env(self._config)
        obs, _ = env.reset(seed=self._config.seed + 200_000 + seed_offset)

        total_reward = 0.0
        terminated = False

        for step in range(self._config.environment.max_episode_steps):
            with torch.no_grad():
                voxels_t = (
                    torch.from_numpy(obs["voxels"]).unsqueeze(0).float().to(self._device)
                )
                proprio_t = (
                    torch.from_numpy(obs["proprio"]).unsqueeze(0).float().to(self._device)
                )
                actions, _, _ = self._network.act(
                    voxels_t, proprio_t, deterministic=deterministic
                )
            action = actions.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward, step + 1, terminated
