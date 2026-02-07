"""Concrete skill implementations for spacecraft control primitives."""

from __future__ import annotations

import numpy as np
import structlog

from src.skills.base import Skill
from src.skills.registry import register_skill
from src.utils.common import clamp_actions, normalize_quaternion, quaternion_multiply

logger = structlog.get_logger(__name__)


@register_skill
class TranslateToGoalSkill(Skill):
    """Proportional thrust toward goal position.

    Computes a normalized direction vector from current position to goal
    position and scales it by the gain parameter to produce a translational
    force command.
    """

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize the translation skill.

        Args:
            gain: Proportional gain for thrust magnitude (default 1.0).
        """
        self._gain = gain

    @property
    def name(self) -> str:
        """Human-readable skill name."""
        return "translate_to_goal"

    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute proportional thrust toward goal position.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(6,)`` with thrust in [:3] and zero torque in [3:].
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        current_pos = proprio[:3]
        goal_pos = goal[:3]

        # Compute direction vector to goal
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            # Already at goal, no thrust needed
            logger.debug("translate_at_goal", distance=distance)
            thrust = np.zeros(3, dtype=np.float64)
        else:
            # Normalize and scale by gain
            thrust = (direction / distance) * self._gain

        # Zero torque
        torque = np.zeros(3, dtype=np.float64)

        # Combine and clamp
        action = np.concatenate([thrust, torque])
        action = clamp_actions(action, -1.0, 1.0)

        logger.debug("translate_action", distance=distance, thrust_norm=np.linalg.norm(thrust))
        return action


@register_skill
class AlignToGoalSkill(Skill):
    """Proportional torque toward goal orientation.

    Computes the quaternion error between current and goal orientation,
    extracts the rotation axis and angle, and applies proportional torque
    to minimize the error.
    """

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize the alignment skill.

        Args:
            gain: Proportional gain for torque magnitude (default 1.0).
        """
        self._gain = gain

    @property
    def name(self) -> str:
        """Human-readable skill name."""
        return "align_to_goal"

    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute proportional torque toward goal orientation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(6,)`` with zero thrust in [:3] and torque in [3:].
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        current_quat = proprio[3:7]
        goal_quat = goal[3:7]

        # Normalize quaternions
        current_quat = normalize_quaternion(current_quat)
        goal_quat = normalize_quaternion(goal_quat)

        # Compute quaternion error: q_error = q_goal * conj(q_current)
        # Conjugate is [w, -x, -y, -z]
        current_quat_conj = current_quat * np.array([1, -1, -1, -1])
        q_error = quaternion_multiply(goal_quat, current_quat_conj)

        # Extract axis-angle representation
        # For unit quaternion [w, x, y, z], angle = 2*arccos(w), axis = [x,y,z]/sin(angle/2)
        w = q_error[0]
        xyz = q_error[1:4]

        # Clamp w to valid range for arccos
        w = np.clip(w, -1.0, 1.0)
        angle = 2.0 * np.arccos(w)

        # Compute torque
        if abs(angle) < 1e-6:
            # Already aligned, no torque needed
            logger.debug("align_at_goal", angle=angle)
            torque = np.zeros(3, dtype=np.float64)
        else:
            # Axis-angle torque: proportional to rotation axis scaled by angle
            # For small angles, sin(angle/2) ≈ angle/2, so xyz/sin(angle/2) ≈ 2*xyz/angle
            # But for robustness, we use the full formula
            sin_half_angle = np.sin(angle / 2.0)
            if abs(sin_half_angle) > 1e-6:
                axis = xyz / sin_half_angle
            else:
                # Fallback: use xyz directly (close to aligned)
                axis = xyz

            torque = axis * angle * self._gain

        # Zero thrust
        thrust = np.zeros(3, dtype=np.float64)

        # Combine and clamp
        action = np.concatenate([thrust, torque])
        action = clamp_actions(action, -1.0, 1.0)

        logger.debug("align_action", angle=angle, torque_norm=np.linalg.norm(torque))
        return action


@register_skill
class BrakeSkill(Skill):
    """Opposing force and torque to current velocity.

    Applies thrust and torque in the opposite direction of current linear
    and angular velocities to reduce motion.
    """

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize the brake skill.

        Args:
            gain: Proportional gain for brake magnitude (default 1.0).
        """
        self._gain = gain

    @property
    def name(self) -> str:
        """Human-readable skill name."""
        return "brake"

    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute opposing force and torque to current velocities.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(6,)`` with brake thrust in [:3] and brake torque in [3:].
        """
        proprio = observation["proprio"]

        linear_vel = proprio[7:10]
        angular_vel = proprio[10:13]

        # Opposing thrust and torque
        thrust = -self._gain * linear_vel
        torque = -self._gain * angular_vel

        # Combine and clamp
        action = np.concatenate([thrust, torque])
        action = clamp_actions(action, -1.0, 1.0)

        logger.debug(
            "brake_action",
            linear_vel_norm=np.linalg.norm(linear_vel),
            angular_vel_norm=np.linalg.norm(angular_vel),
        )
        return action


@register_skill
class StationKeepSkill(Skill):
    """Combination of brake and small positional correction toward goal.

    Blends braking behavior with a gentle translational correction to
    maintain position near the goal.
    """

    def __init__(self, gain: float = 0.5, brake_gain: float = 1.0) -> None:
        """Initialize the station-keeping skill.

        Args:
            gain: Proportional gain for positional correction (default 0.5).
            brake_gain: Proportional gain for velocity damping (default 1.0).
        """
        self._gain = gain
        self._brake_gain = brake_gain

    @property
    def name(self) -> str:
        """Human-readable skill name."""
        return "station_keep"

    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute station-keeping action combining brake and position correction.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(6,)`` with combined thrust and torque.
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        current_pos = proprio[:3]
        goal_pos = goal[:3]
        linear_vel = proprio[7:10]
        angular_vel = proprio[10:13]

        # Small positional correction
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            position_thrust = np.zeros(3, dtype=np.float64)
        else:
            position_thrust = (direction / distance) * self._gain

        # Velocity damping (brake)
        brake_thrust = -self._brake_gain * linear_vel

        # Combine thrust components
        thrust = position_thrust + brake_thrust

        # Brake angular velocity
        torque = -self._brake_gain * angular_vel

        # Combine and clamp
        action = np.concatenate([thrust, torque])
        action = clamp_actions(action, -1.0, 1.0)

        logger.debug(
            "station_keep_action",
            distance=distance,
            linear_vel_norm=np.linalg.norm(linear_vel),
        )
        return action


@register_skill
class ApproachSkill(Skill):
    """Composite skill: translate + align + distance-dependent brake blend.

    When far from the goal, translation dominates. As the spacecraft
    approaches the goal, braking behavior increases to ensure a smooth
    final approach.
    """

    def __init__(
        self,
        translate_gain: float = 1.0,
        align_gain: float = 0.8,
        brake_gain: float = 1.0,
        brake_distance_threshold: float = 5.0,
    ) -> None:
        """Initialize the approach skill.

        Args:
            translate_gain: Gain for translational thrust (default 1.0).
            align_gain: Gain for alignment torque (default 0.8).
            brake_gain: Gain for braking when close (default 1.0).
            brake_distance_threshold: Distance below which braking increases (default 5.0).
        """
        self._translate_gain = translate_gain
        self._align_gain = align_gain
        self._brake_gain = brake_gain
        self._brake_distance_threshold = brake_distance_threshold

    @property
    def name(self) -> str:
        """Human-readable skill name."""
        return "approach"

    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute composite approach action with distance-dependent blending.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.

        Returns:
            Action array of shape ``(6,)`` with blended thrust and torque.
        """
        proprio = observation["proprio"]
        goal = observation["goal"]

        current_pos = proprio[:3]
        goal_pos = goal[:3]
        current_quat = proprio[3:7]
        goal_quat = goal[3:7]
        linear_vel = proprio[7:10]
        angular_vel = proprio[10:13]

        # Compute distance for blending
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)

        # Translation component
        if distance < 1e-6:
            translate_thrust = np.zeros(3, dtype=np.float64)
        else:
            translate_thrust = (direction / distance) * self._translate_gain

        # Brake component (increases as distance decreases)
        if distance < self._brake_distance_threshold:
            # Blend factor: 0 at threshold, 1 at distance=0
            brake_blend = float(1.0 - (distance / self._brake_distance_threshold))
        else:
            brake_blend = 0.0

        brake_thrust = -self._brake_gain * brake_blend * linear_vel

        # Combine thrust: far = translate only, close = translate + brake
        thrust = translate_thrust + brake_thrust

        # Alignment torque (always active)
        current_quat = normalize_quaternion(current_quat)
        goal_quat = normalize_quaternion(goal_quat)

        current_quat_conj = current_quat * np.array([1, -1, -1, -1])
        q_error = quaternion_multiply(goal_quat, current_quat_conj)

        w = np.clip(q_error[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)

        if abs(angle) < 1e-6:
            align_torque = np.zeros(3, dtype=np.float64)
        else:
            xyz = q_error[1:4]
            sin_half_angle = np.sin(angle / 2.0)
            if abs(sin_half_angle) > 1e-6:
                axis = xyz / sin_half_angle
            else:
                axis = xyz
            align_torque = axis * angle * self._align_gain

        # Angular brake component (also increases when close)
        brake_torque = -self._brake_gain * brake_blend * angular_vel

        # Combine torque
        torque = align_torque + brake_torque

        # Combine and clamp
        action = np.concatenate([thrust, torque])
        action = clamp_actions(action, -1.0, 1.0)

        logger.debug(
            "approach_action",
            distance=distance,
            brake_blend=brake_blend,
            angle=angle,
        )
        return action
