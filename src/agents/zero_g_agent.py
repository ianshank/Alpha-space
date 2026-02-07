"""Concrete agent implementation for the ZeroG-RL system.

This agent composes:
  - SpatialPolicyValueNetwork (policy/value network)
  - MCTSEngine (tree search for action selection)
  - NetworkPredictor (MCTS adapter)
  - Optimizer (Adam)

The agent owns the policy gradient update step and supports skill blending.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn

from src.config import SystemConfig
from src.mcts.engine import MCTSEngine
from src.networks.policy_value_net import SpatialPolicyValueNetwork
from src.training.trainer import NetworkPredictor
from src.utils.common import get_device

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Skill protocol (structural typing)
# ---------------------------------------------------------------------------


class SkillProtocol:
    """Protocol for skill-based policies that can be blended with the NN policy."""

    def compute_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Compute action from observation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ZeroG agent
# ---------------------------------------------------------------------------


class ZeroGAgent:
    """Concrete RL agent for the ZeroG spacecraft docking task.

    Composes a policy/value network, MCTS engine, and optimizer. Supports
    both MCTS-guided action selection and direct neural network inference,
    with optional skill blending.

    Args:
        config: Full system configuration.
        device: Torch device for network and optimizer.
        skills: Optional list of skill objects for action blending.
        skill_blend_alpha: Blend weight for skill actions (0.0 = pure NN,
            1.0 = pure skill).
    """

    def __init__(
        self,
        config: SystemConfig,
        device: torch.device | None = None,
        skills: list[SkillProtocol] | None = None,
        skill_blend_alpha: float = 0.0,
    ) -> None:
        self._config = config
        self._device = device if device is not None else get_device(config.use_gpu)
        self._skills = skills or []
        self._skill_blend_alpha = skill_blend_alpha

        # Network
        self._network = SpatialPolicyValueNetwork(config.network).to(self._device)

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # MCTS predictor & engine
        self._predictor = NetworkPredictor(
            network=self._network,
            device=self._device,
            num_candidates=config.mcts.max_children,
        )
        self._mcts = MCTSEngine(config=config.mcts, predictor=self._predictor)

        logger.info(
            "agent_created",
            device=str(self._device),
            num_skills=len(self._skills),
            skill_blend_alpha=self._skill_blend_alpha,
        )

    # ------------------------------------------------------------------ #
    # Public API (AgentProtocol)
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        *,
        use_mcts: bool = True,
        deterministic: bool = False,
        use_skills: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Select an action given an observation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.
            use_mcts: Whether to use MCTS search.
            deterministic: If True, use greedy action selection.
            use_skills: If True, blend skill actions with NN actions.

        Returns:
            ``(action, info)`` tuple where *action* is ``(action_dim,)``
            and *info* contains diagnostic information.
        """
        info: dict[str, float] = {}

        if use_mcts:
            # MCTS search
            action, mcts_info = self._mcts.search(observation=observation)
            info.update({k: v for k, v in mcts_info.items() if isinstance(v, (int, float))})
        else:
            # Direct NN inference
            action = self._network_action(observation, deterministic=deterministic)
            info["network_action"] = 1.0

        # Skill blending
        if use_skills and self._skills:
            action = self._blend_skill_action(observation, action, info)

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        return action, info

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient update on a batch of experience.

        Args:
            batch: Dict with keys ``voxels``, ``proprio``, ``action``,
                ``value_target`` (numpy arrays).

        Returns:
            Dict of loss metrics: ``policy_loss``, ``value_loss``, ``entropy``.
        """
        self._network.train()
        cfg = self._config.training

        # Convert to tensors
        voxels = torch.from_numpy(batch["voxels"]).float().to(self._device)
        proprio = torch.from_numpy(batch["proprio"]).float().to(self._device)
        actions = torch.from_numpy(batch["action"]).float().to(self._device)
        value_targets = torch.from_numpy(batch["value_target"]).float().to(self._device)

        # Forward pass
        log_probs, entropy, values = self._network.evaluate_actions(voxels, proprio, actions)

        # Policy loss: negative log-likelihood
        policy_loss = -log_probs.mean()

        # Value loss: MSE against discounted returns
        value_loss = nn.functional.mse_loss(values.squeeze(-1), value_targets)

        # Entropy bonus
        entropy_mean = entropy.mean()

        # Total loss
        loss = policy_loss + cfg.value_loss_weight * value_loss - cfg.entropy_weight * entropy_mean

        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self._network.parameters(), cfg.gradient_clip_norm)

        # Optimizer step
        self._optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
        }

    def train_mode(self) -> None:
        """Set agent to training mode."""
        self._network.train()

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self._network.eval()

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpointing.

        Returns:
            Dict with keys ``network`` and ``optimizer``.
        """
        return {
            "network": self._network.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore agent state from a checkpoint.

        Args:
            state: Dict with keys ``network`` and ``optimizer``.
        """
        self._network.load_state_dict(state["network"])
        self._optimizer.load_state_dict(state["optimizer"])
        logger.info("agent_state_loaded")

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _network_action(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action directly from the network (no MCTS).

        Args:
            observation: Dict with keys ``voxels`` and ``proprio``.
            deterministic: If True, return the distribution mean.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        self._network.eval()
        with torch.no_grad():
            voxels = torch.from_numpy(observation["voxels"]).unsqueeze(0).float().to(self._device)
            proprio = torch.from_numpy(observation["proprio"]).unsqueeze(0).float().to(self._device)
            actions, _, _ = self._network.act(voxels, proprio, deterministic=deterministic)
            action = actions.squeeze(0).cpu().numpy()
        return action

    def _blend_skill_action(
        self,
        observation: dict[str, np.ndarray],
        nn_action: np.ndarray,
        info: dict[str, float],
    ) -> np.ndarray:
        """Blend neural network action with skill-based actions.

        Uses a weighted average with blend weight ``self._skill_blend_alpha``.
        If multiple skills are available, their actions are averaged first.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.
            nn_action: Action from the neural network.
            info: Info dict to update with blend diagnostics.

        Returns:
            Blended action array of shape ``(action_dim,)``.
        """
        if not self._skills or self._skill_blend_alpha <= 0.0:
            return nn_action

        # Compute skill actions
        skill_actions = []
        for skill in self._skills:
            try:
                skill_action = skill.compute_action(observation)
                skill_actions.append(skill_action)
            except Exception as e:
                logger.warning("skill_action_failed", skill=type(skill).__name__, error=str(e))

        if not skill_actions:
            return nn_action

        # Average skill actions
        mean_skill_action = np.mean(skill_actions, axis=0)

        # Blend with NN action
        alpha = self._skill_blend_alpha
        blended_action = (1.0 - alpha) * nn_action + alpha * mean_skill_action

        info["skill_blend_alpha"] = alpha
        info["num_skills_used"] = float(len(skill_actions))

        return blended_action

    # ------------------------------------------------------------------ #
    # Additional properties
    # ------------------------------------------------------------------ #

    @property
    def network(self) -> SpatialPolicyValueNetwork:
        """Access to the underlying policy/value network."""
        return self._network

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Access to the optimizer."""
        return self._optimizer

    @property
    def device(self) -> torch.device:
        """Access to the device."""
        return self._device
