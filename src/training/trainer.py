"""Self-play training orchestrator.

Coordinates the full training loop:
  1. Run self-play episodes with MCTS-guided exploration.
  2. Store trajectories in replay buffer.
  3. Update policy/value network on sampled batches.
  4. Periodically evaluate and checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn

from src.checkpointing.checkpoint_manager import CheckpointManager
from src.config import SystemConfig
from src.debugging.instrumentation import log_execution_time, log_tensor_stats
from src.environments.base import ZeroGEnv
from src.environments.factory import make_env
from src.mcts.engine import MCTSEngine, PolicyValuePredictor
from src.networks.policy_value_net import SpatialPolicyValueNetwork
from src.replay_buffer.buffer import Episode, ReplayBuffer, Transition
from src.utils.common import Timer, get_device, seed_everything

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Neural-network ↔ MCTS adapter
# ---------------------------------------------------------------------------


class NetworkPredictor:
    """Wraps a ``SpatialPolicyValueNetwork`` for use by the MCTS engine.

    Converts numpy observations to tensors, runs inference, and returns
    sampled candidate actions as numpy arrays.

    Args:
        network: The policy / value network.
        device: Torch device for inference.
        num_candidates: Number of actions to sample from the policy for
            MCTS expansion.
    """

    def __init__(
        self,
        network: SpatialPolicyValueNetwork,
        device: torch.device,
        num_candidates: int = 32,
    ) -> None:
        self._net = network
        self._device = device
        self._num_candidates = num_candidates

    def predict(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """Predict action candidates and state value.

        Args:
            observation: Dict with ``voxels`` and ``proprio``.

        Returns:
            ``(candidate_actions, value)`` — actions shape
            ``(num_candidates, action_dim)``, value is a scalar.
        """
        self._net.eval()
        with torch.no_grad():
            voxels = torch.from_numpy(observation["voxels"]).unsqueeze(0).float().to(self._device)
            proprio = torch.from_numpy(observation["proprio"]).unsqueeze(0).float().to(self._device)

            action_dist, value = self._net(voxels, proprio)
            candidates = action_dist.sample((self._num_candidates,))  # (K, 1, action_dim)
            candidates = candidates.squeeze(1)  # (K, action_dim)

        return (
            candidates.cpu().numpy(),
            value.item(),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrates self-play training.

    Args:
        config: Full system configuration.
        resume_from: Optional checkpoint path to resume from.
    """

    def __init__(
        self,
        config: SystemConfig,
        resume_from: Path | None = None,
    ) -> None:
        self._config = config

        # Seed
        seed_everything(config.seed)

        # Device
        self._device = get_device(prefer_gpu=config.use_gpu)

        # Network
        self._network = SpatialPolicyValueNetwork(config.network).to(self._device)

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Checkpoint manager
        self._ckpt_mgr = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_to_keep=10,
        )

        # Replay buffer
        self._replay = ReplayBuffer(
            capacity=config.training.replay_buffer_size,
            persist_dir=config.log_dir / "replay",
        )

        # Starting episode
        self._start_episode = 0

        # Resume
        if resume_from is not None:
            ep, _, _ = self._ckpt_mgr.load(
                path=resume_from,
                policy_net=self._network,
                optimizer=self._optimizer,
                device=str(self._device),
            )
            self._start_episode = ep + 1
            logger.info("training_resumed", from_episode=self._start_episode)

        # MCTS predictor & engine
        self._predictor = NetworkPredictor(
            network=self._network,
            device=self._device,
            num_candidates=config.mcts.max_children,
        )
        self._mcts = MCTSEngine(config=config.mcts, predictor=self._predictor)

        # Metrics accumulator
        self._metrics: dict[str, list[float]] = {
            "episode_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Summary dict with final metrics.
        """
        total_episodes = self._config.training.num_episodes

        logger.info(
            "training_started",
            total_episodes=total_episodes,
            start_episode=self._start_episode,
            device=str(self._device),
        )

        for episode_idx in range(self._start_episode, total_episodes):
            # --- Self-play episode ---
            with Timer("self_play_episode"):
                episode = self._run_episode(episode_idx)
            self._replay.add_episode(episode)

            self._metrics["episode_reward"].append(episode.total_reward)
            self._metrics["episode_length"].append(float(episode.length))

            logger.info(
                "episode_completed",
                episode=episode_idx,
                reward=round(episode.total_reward, 3),
                length=episode.length,
                success=episode.success,
                buffer_size=len(self._replay),
            )

            # --- Policy update ---
            if len(self._replay) >= self._config.training.batch_size:
                with Timer("policy_update"):
                    losses = self._update_policy()
                self._metrics["policy_loss"].append(losses["policy_loss"])
                self._metrics["value_loss"].append(losses["value_loss"])
                self._metrics["entropy"].append(losses["entropy"])

            # --- Checkpoint ---
            if (episode_idx + 1) % self._config.training.checkpoint_interval == 0:
                self._ckpt_mgr.save(
                    policy_net=self._network,
                    optimizer=self._optimizer,
                    episode=episode_idx,
                    config=self._config,
                    metrics=self._latest_metrics(),
                )

            # --- Evaluation ---
            if (episode_idx + 1) % self._config.training.eval_interval == 0:
                eval_result = self._evaluate(episode_idx)
                logger.info("evaluation_result", episode=episode_idx, **eval_result)

        # Final checkpoint
        self._ckpt_mgr.save(
            policy_net=self._network,
            optimizer=self._optimizer,
            episode=total_episodes - 1,
            config=self._config,
            metrics=self._latest_metrics(),
            tag="final",
        )

        summary = self._latest_metrics()
        logger.info("training_completed", **summary)
        return summary

    # ------------------------------------------------------------------ #
    # Self-play
    # ------------------------------------------------------------------ #

    def _run_episode(self, episode_idx: int) -> Episode:
        """Generate one self-play episode with MCTS."""
        env = make_env(self._config)
        obs, _ = env.reset(seed=self._config.seed + episode_idx)

        episode = Episode()
        total_reward = 0.0

        for step in range(self._config.environment.max_episode_steps):
            # MCTS search
            action, mcts_info = self._mcts.search(observation=obs)

            # Clip to valid range
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition
            transition = Transition(
                voxels=obs["voxels"],
                proprio=obs["proprio"],
                goal=obs["goal"],
                action=action,
                reward=reward,
                value_target=0.0,  # will be computed below
                done=terminated or truncated,
            )
            episode.transitions.append(transition)
            total_reward += reward

            if terminated or truncated:
                break
            obs = next_obs

        # Compute discounted value targets (backward pass)
        self._compute_value_targets(episode)

        episode.total_reward = total_reward
        episode.length = len(episode.transitions)
        episode.success = terminated if "terminated" in dir() else False

        return episode

    def _compute_value_targets(self, episode: Episode) -> None:
        """Compute discounted-return value targets for each transition."""
        gamma = self._config.mcts.discount
        running_return = 0.0
        for t in reversed(episode.transitions):
            running_return = t.reward + gamma * running_return * (1.0 - float(t.done))
            t.value_target = running_return

    # ------------------------------------------------------------------ #
    # Policy update
    # ------------------------------------------------------------------ #

    def _update_policy(self) -> dict[str, float]:
        """Run one gradient update on a sampled batch."""
        self._network.train()
        cfg = self._config.training

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(cfg.epochs_per_update):
            batch = self._replay.sample_batch_tensors(cfg.batch_size)

            voxels = torch.from_numpy(batch["voxels"]).float().to(self._device)
            proprio = torch.from_numpy(batch["proprio"]).float().to(self._device)
            actions = torch.from_numpy(batch["action"]).float().to(self._device)
            value_targets = torch.from_numpy(batch["value_target"]).float().to(self._device)

            log_probs, entropy, values = self._network.evaluate_actions(
                voxels, proprio, actions
            )

            # Policy loss: negative log-likelihood (encourage MCTS actions)
            policy_loss = -log_probs.mean()

            # Value loss: MSE against discounted returns
            value_loss = nn.functional.mse_loss(values.squeeze(-1), value_targets)

            # Entropy bonus
            entropy_mean = entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + cfg.value_loss_weight * value_loss
                - cfg.entropy_weight * entropy_mean
            )

            self._optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self._network.parameters(), cfg.gradient_clip_norm
            )
            log_tensor_stats("gradient_norm", torch.tensor([grad_norm.item()]))

            self._optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()

        n = cfg.epochs_per_update
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _evaluate(self, episode_idx: int, num_episodes: int = 5) -> dict[str, float]:
        """Run greedy evaluation episodes (no MCTS, deterministic actions)."""
        self._network.eval()
        rewards: list[float] = []
        successes: list[float] = []

        for i in range(num_episodes):
            env = make_env(self._config)
            obs, _ = env.reset(seed=self._config.seed + 100_000 + i)
            total_reward = 0.0

            for _ in range(self._config.environment.max_episode_steps):
                with torch.no_grad():
                    voxels_t = (
                        torch.from_numpy(obs["voxels"]).unsqueeze(0).float().to(self._device)
                    )
                    proprio_t = (
                        torch.from_numpy(obs["proprio"]).unsqueeze(0).float().to(self._device)
                    )
                    actions, _, _ = self._network.act(
                        voxels_t, proprio_t, deterministic=True
                    )
                action = actions.squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            successes.append(1.0 if terminated else 0.0)

        return {
            "eval_mean_reward": float(np.mean(rewards)),
            "eval_success_rate": float(np.mean(successes)),
        }

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _latest_metrics(self) -> dict[str, float]:
        """Return the last 100 episodes' mean metrics."""
        window = 100
        result: dict[str, float] = {}
        for key, values in self._metrics.items():
            if values:
                recent = values[-window:]
                result[f"mean_{key}"] = float(np.mean(recent))
        return result
