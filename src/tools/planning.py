"""Planning tools for trajectory and fuel estimation."""

from __future__ import annotations

import numpy as np
import structlog

from src.tools.base import Tool
from src.tools.registry import register_tool

logger = structlog.get_logger(__name__)


@register_tool
class TrajectoryPlannerTool(Tool):
    """Estimate straight-line trajectory metrics to goal."""

    def __init__(self, max_thrust: float = 10.0, spacecraft_mass: float = 100.0) -> None:
        """Initialize trajectory planner.

        Args:
            max_thrust: Maximum thrust force in Newtons.
            spacecraft_mass: Spacecraft mass in kilograms.
        """
        self._max_thrust = max_thrust
        self._spacecraft_mass = spacecraft_mass
        self._max_acceleration = max_thrust / spacecraft_mass

    @property
    def name(self) -> str:
        return "TrajectoryPlanner"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> dict[str, float]:
        """Estimate trajectory metrics to goal.

        Args:
            observation: Dict containing 'proprio' and 'goal' arrays.

        Returns:
            Dict with 'distance' and 'estimated_time' keys.
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        position = proprio[:3]
        goal_position = goal[:3]
        # Straight-line distance
        distance = float(np.linalg.norm(goal_position - position))

        # Simple time estimate using constant acceleration
        # Assumes accelerate halfway, decelerate halfway
        # d = a*t^2 => t = sqrt(d/a)
        if distance > 1e-6:
            estimated_time = float(np.sqrt(2 * distance / self._max_acceleration))
        else:
            estimated_time = 0.0

        logger.debug(
            "trajectory_planned",
            position=position.tolist(),
            goal_position=goal_position.tolist(),
            distance=distance,
            estimated_time=estimated_time,
            max_acceleration=self._max_acceleration,
        )

        return {"distance": distance, "estimated_time": estimated_time}


@register_tool
class FuelEstimatorTool(Tool):
    """Estimate delta-V required for braking maneuver."""

    def __init__(self, angular_scale: float = 0.5) -> None:
        """Initialize fuel estimator.

        Args:
            angular_scale: Scaling factor for angular velocity contribution.
        """
        self._angular_scale = angular_scale

    @property
    def name(self) -> str:
        return "FuelEstimator"

    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> float:
        """Estimate delta-V for braking from current velocity to zero.

        Args:
            observation: Dict containing 'proprio' array.

        Returns:
            Estimated delta-V in m/s.
        """
        proprio = observation["proprio"]

        linear_vel = proprio[7:10]
        angular_vel = proprio[10:13]

        linear_dv = float(np.linalg.norm(linear_vel))
        angular_dv = float(np.linalg.norm(angular_vel) * self._angular_scale)

        total_dv = linear_dv + angular_dv

        logger.debug(
            "fuel_estimated",
            linear_vel=linear_vel.tolist(),
            angular_vel=angular_vel.tolist(),
            linear_dv=linear_dv,
            angular_dv=angular_dv,
            total_dv=total_dv,
        )

        return total_dv
