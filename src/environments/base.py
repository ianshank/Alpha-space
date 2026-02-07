"""Abstract base and Gymnasium-compatible wrapper for ZeroG environments.

Defines the observation and action spaces for the 6-DOF docking task,
plus a factory function that returns the configured simulator backend.
"""

from __future__ import annotations

import abc
from typing import Any

import gymnasium as gym
import numpy as np
import structlog

from src.config import EnvironmentConfig, RewardConfig

logger = structlog.get_logger(__name__)


class ZeroGEnv(gym.Env[dict[str, np.ndarray], np.ndarray], abc.ABC):
    """Abstract Gymnasium environment for zero-gravity docking tasks.

    Subclasses must implement :meth:`_sim_reset` and :meth:`_sim_step`.

    Observation dict keys:
        ``voxels``:  ``(C, D, H, W)`` voxel grid.
        ``proprio``: ``(proprio_dim,)`` proprioception vector
            ``[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]``.
        ``goal``:    ``(7,)`` target pose ``[x, y, z, qw, qx, qy, qz]``.

    Action: ``(6,)`` normalised actions in ``[-1, 1]``, scaled to
        ``[max_thrust, max_torque]`` internally.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_config: EnvironmentConfig,
        reward_config: RewardConfig,
        voxel_resolution: int = 64,
        voxel_channels: int = 4,
    ) -> None:
        super().__init__()
        self._env_cfg = env_config
        self._reward_cfg = reward_config
        self._voxel_res = voxel_resolution
        self._voxel_ch = voxel_channels
        self._step_count = 0

        # Action space: normalised [-1, 1] for 6-DOF
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation space (dict)
        self.observation_space = gym.spaces.Dict(
            {
                "voxels": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(voxel_channels, voxel_resolution, voxel_resolution, voxel_resolution),
                    dtype=np.float32,
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
                ),
                "goal": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        obs = self._sim_reset(seed=seed, options=options)
        logger.debug("env_reset", env_id=id(self))
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale normalised action to physical units
        scaled = self._scale_action(action)

        obs, raw_reward_info = self._sim_step(scaled)
        self._step_count += 1

        reward = self._compute_reward(obs, raw_reward_info)
        terminated = self._check_success(obs)
        truncated = self._step_count >= self._env_cfg.max_episode_steps

        info: dict[str, Any] = {
            "step": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            **raw_reward_info,
        }

        if terminated:
            reward += self._reward_cfg.success_bonus
            logger.info("episode_success", step=self._step_count)

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalised ``[-1,1]`` action to physical force/torque."""
        forces = action[:3] * self._env_cfg.max_thrust
        torques = action[3:] * self._env_cfg.max_torque
        return np.concatenate([forces, torques])

    def _compute_reward(
        self,
        obs: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> float:
        """Shaped reward: penalise position/orientation error, fuel use."""
        pos_err = float(info.get("position_error", 0.0))
        ori_err = float(info.get("orientation_error", 0.0))
        vel_mag = float(info.get("velocity_magnitude", 0.0))
        fuel = float(info.get("fuel_used", 0.0))

        reward = (
            -self._reward_cfg.position_weight * pos_err
            - self._reward_cfg.orientation_weight * ori_err
            - self._reward_cfg.velocity_weight * vel_mag
            - self._reward_cfg.fuel_weight * fuel
            + self._reward_cfg.time_penalty
        )
        return reward

    def _check_success(self, obs: dict[str, np.ndarray]) -> bool:
        """Check if agent achieved the docking goal."""
        proprio = obs["proprio"]
        goal = obs["goal"]
        pos_error = float(np.linalg.norm(proprio[:3] - goal[:3]))
        # Quaternion distance (angle between orientations)
        q_agent = proprio[3:7]
        q_goal = goal[3:7]
        dot = float(np.abs(np.dot(q_agent, q_goal)))
        dot = min(dot, 1.0)
        angle_error_deg = float(2.0 * np.degrees(np.arccos(dot)))

        return (
            pos_error < self._env_cfg.position_tolerance
            and angle_error_deg < self._env_cfg.orientation_tolerance_deg
        )

    # ------------------------------------------------------------------ #
    # Abstract simulation interface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def _sim_reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset the underlying simulator and return initial observation."""

    @abc.abstractmethod
    def _sim_step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Step the simulator with a *physical* action (not normalised).

        Returns:
            ``(observation, info_dict)`` where *info_dict* contains at
            least: ``position_error``, ``orientation_error``,
            ``velocity_magnitude``, ``fuel_used``.
        """
