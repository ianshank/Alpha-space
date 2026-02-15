"""Mock zero-gravity environment backed by the internal physics engine.

Used for fast development, testing, and CI — does not require Unity or
Isaac Sim.  Integrates the ``ZeroGDynamics`` module for physically-
consistent simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from src.config import EnvironmentConfig, RewardConfig
from src.environments.base import ZeroGEnv
from src.physics.zero_g_dynamics import RigidBodyState, ZeroGDynamics
from src.utils.common import normalize_quaternion, quaternion_angular_distance

logger = structlog.get_logger(__name__)


class MockZeroGEnv(ZeroGEnv):
    """Lightweight mock environment with internal rigid-body physics.

    The workspace is an empty cube — no obstacles.  The spacecraft starts
    at a random pose and must dock at a randomly sampled goal pose.
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        reward_config: RewardConfig,
        voxel_resolution: int = 64,
        voxel_channels: int = 4,
    ) -> None:
        super().__init__(
            env_config=env_config,
            reward_config=reward_config,
            voxel_resolution=voxel_resolution,
            voxel_channels=voxel_channels,
        )
        self._dynamics = ZeroGDynamics()
        self._state: RigidBodyState | None = None
        self._goal_pose: np.ndarray = np.zeros(7, dtype=np.float64)
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # State management (for MCTS environment model)
    # ------------------------------------------------------------------ #

    def clone_state(self) -> dict[str, Any]:
        """Snapshot the full environment state for MCTS."""
        return {
            "rb_state": self._state.clone() if self._state else None,
            "goal_pose": self._goal_pose.copy(),
            "step_count": self._step_count,
        }

    def set_state(self, state: object) -> None:
        """Restore environment from a snapshot."""
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict, got {type(state).__name__}")
        self._state = state["rb_state"].clone() if state["rb_state"] else None
        self._goal_pose = state["goal_pose"].copy()
        self._step_count = state["step_count"]

    # ------------------------------------------------------------------ #
    # Simulation implementation
    # ------------------------------------------------------------------ #

    def _sim_reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ws = self._env_cfg.workspace_size

        # Random start pose
        start_pos = self._rng.uniform(-ws / 2, ws / 2, size=3)
        start_quat = self._random_quaternion()
        start_vel = self._rng.uniform(-0.5, 0.5, size=3)
        start_ang_vel = self._rng.uniform(-0.1, 0.1, size=3)

        self._state = RigidBodyState(
            position=start_pos,
            velocity=start_vel,
            orientation=start_quat,
            angular_velocity=start_ang_vel,
            mass=self._env_cfg.spacecraft_mass,
            inertia=np.array(self._env_cfg.spacecraft_inertia, dtype=np.float64),
        )

        # Random goal pose (zero velocity required at dock)
        goal_pos = self._rng.uniform(-ws / 4, ws / 4, size=3)
        goal_quat = self._random_quaternion()
        self._goal_pose = np.concatenate([goal_pos, goal_quat])

        return self._build_observation()

    def _sim_step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized; call reset() first")

        force = action[:3].astype(np.float64)
        torque = action[3:].astype(np.float64)

        self._state = self._dynamics.step(
            state=self._state,
            force=force,
            torque=torque,
            dt=self._env_cfg.time_step,
            validate=True,
        )

        obs = self._build_observation()

        info = {
            "position_error": float(np.linalg.norm(self._state.position - self._goal_pose[:3])),
            "orientation_error": self._orientation_error(),
            "velocity_magnitude": float(np.linalg.norm(self._state.velocity)),
            "fuel_used": float(np.linalg.norm(action)),
        }

        return obs, info

    # ------------------------------------------------------------------ #
    # Observation building
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Construct the observation dict from current state."""
        if self._state is None:
            raise RuntimeError("Environment not initialized; call reset() first")

        # Voxels: empty workspace (all zeros) — mock env has no obstacles
        voxels = np.zeros(
            (self._voxel_ch, self._voxel_res, self._voxel_res, self._voxel_res),
            dtype=np.float32,
        )

        # Proprioception: [pos(3), quat(4), vel(3), ang_vel(3)]
        proprio = np.concatenate(
            [
                self._state.position,
                self._state.orientation,
                self._state.velocity,
                self._state.angular_velocity,
            ]
        ).astype(np.float32)

        goal = self._goal_pose.astype(np.float32)

        return {"voxels": voxels, "proprio": proprio, "goal": goal}

    def _orientation_error(self) -> float:
        """Compute angular distance between agent and goal orientations (degrees)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized; call reset() first")
        return quaternion_angular_distance(
            self._state.orientation, self._goal_pose[3:7]
        )

    def _random_quaternion(self) -> np.ndarray:
        """Sample a uniformly random unit quaternion.

        Uses the Shoemake uniform sampling method, reordered to the
        Hamilton convention ``[w, x, y, z]`` used throughout this project.
        """
        u = self._rng.random(3)
        # Shoemake components (original order: x, y, z, w)
        x = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
        y = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
        z = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        w = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        q = np.array([w, x, y, z], dtype=np.float64)
        return normalize_quaternion(q)
