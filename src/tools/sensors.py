"""Sensor tools for extracting derived quantities from observations."""

from __future__ import annotations

import numpy as np
import structlog

from src.tools.base import Tool
from src.tools.registry import register_tool
from src.utils.common import normalize_quaternion, quaternion_multiply

logger = structlog.get_logger(__name__)


@register_tool
class DistanceToGoalTool(Tool):
    """Calculate Euclidean distance from current position to goal position."""

    @property
    def name(self) -> str:
        return "DistanceToGoal"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> float:
        """Compute distance from position to goal.

        Args:
            observation: Dict containing 'proprio' and 'goal' arrays.

        Returns:
            Euclidean distance in meters.
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        position = proprio[:3]
        goal_position = goal[:3]

        distance = float(np.linalg.norm(goal_position - position))

        logger.debug(
            "distance_to_goal_computed",
            position=position.tolist(),
            goal_position=goal_position.tolist(),
            distance=distance,
        )

        return distance


@register_tool
class OrientationErrorTool(Tool):
    """Calculate angular distance between current and goal orientation."""

    @property
    def name(self) -> str:
        return "OrientationError"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> float:
        """Compute orientation error in degrees.

        Args:
            observation: Dict containing 'proprio' and 'goal' arrays.

        Returns:
            Angular distance in degrees.
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        current_quat = normalize_quaternion(proprio[3:7])
        goal_quat = normalize_quaternion(goal[3:7])

        # Compute error quaternion: q_error = q_goal * q_current^-1
        # Quaternion conjugate (inverse for unit quaternions): [w, -x, -y, -z]
        current_quat_inv = current_quat * np.array([1, -1, -1, -1])
        error_quat = quaternion_multiply(goal_quat, current_quat_inv)

        # Extract angle from error quaternion: angle = 2 * arccos(w)
        # Clamp w to [-1, 1] for numerical stability
        w = np.clip(error_quat[0], -1.0, 1.0)
        angle_rad = 2.0 * np.arccos(np.abs(w))
        angle_deg = float(np.degrees(angle_rad))

        logger.debug(
            "orientation_error_computed",
            current_quat=current_quat.tolist(),
            goal_quat=goal_quat.tolist(),
            error_deg=angle_deg,
        )

        return angle_deg


@register_tool
class VelocityMagnitudeTool(Tool):
    """Extract linear and angular velocity magnitudes."""

    @property
    def name(self) -> str:
        return "VelocityMagnitude"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> dict[str, float]:
        """Compute velocity magnitudes.

        Args:
            observation: Dict containing 'proprio' array.

        Returns:
            Dict with 'linear' and 'angular' velocity magnitudes.
        """
        proprio = observation["proprio"]

        linear_vel = proprio[7:10]
        angular_vel = proprio[10:13]

        linear_mag = float(np.linalg.norm(linear_vel))
        angular_mag = float(np.linalg.norm(angular_vel))

        logger.debug(
            "velocity_magnitude_computed",
            linear_vel=linear_vel.tolist(),
            angular_vel=angular_vel.tolist(),
            linear_mag=linear_mag,
            angular_mag=angular_mag,
        )

        return {"linear": linear_mag, "angular": angular_mag}


@register_tool
class DockingProgressTool(Tool):
    """Composite metric for docking progress using position and orientation errors."""

    @property
    def name(self) -> str:
        return "DockingProgress"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> float:
        """Compute docking progress score (0 to 1).

        Args:
            observation: Dict containing 'proprio' and 'goal' arrays.
            **kwargs: Optional position_tolerance (default 0.1) and
                     orientation_tolerance_deg (default 5.0).

        Returns:
            Progress score: 1.0 when perfectly docked, 0.0 when far away.
        """
        position_tolerance: float = kwargs.get("position_tolerance", 0.1)  # type: ignore
        orientation_tolerance_deg: float = kwargs.get("orientation_tolerance_deg", 5.0)  # type: ignore

        # Get position and orientation errors
        distance_tool = DistanceToGoalTool()
        orientation_tool = OrientationErrorTool()

        distance = distance_tool(observation)
        angle_error = orientation_tool(observation)

        # Exponential decay for each component
        # When distance = position_tolerance, position_score ≈ 0.37 (e^-1)
        # When angle_error = orientation_tolerance_deg, orientation_score ≈ 0.37
        position_score = np.exp(-distance / position_tolerance)
        orientation_score = np.exp(-angle_error / orientation_tolerance_deg)

        # Combined score (geometric mean for balanced contribution)
        progress = float(np.sqrt(position_score * orientation_score))

        logger.debug(
            "docking_progress_computed",
            distance=distance,
            angle_error=angle_error,
            position_score=position_score,
            orientation_score=orientation_score,
            progress=progress,
        )

        return progress
