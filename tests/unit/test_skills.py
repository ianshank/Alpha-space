"""Unit tests for spacecraft control skills.

Tests primitive skills including TranslateToGoalSkill, AlignToGoalSkill,
BrakeSkill, StationKeepSkill, ApproachSkill, and the skill registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.skills.primitives import (
    AlignToGoalSkill,
    ApproachSkill,
    BrakeSkill,
    StationKeepSkill,
    TranslateToGoalSkill,
)
from src.skills.registry import get_skill, list_skills

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def skill_obs() -> dict[str, np.ndarray]:
    """Sample observation for skill testing.

    Returns:
        Observation dict with voxels, proprio, and goal.
    """
    return {
        "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
        "proprio": np.array(
            [
                1.0,
                2.0,
                3.0,  # position
                1.0,
                0.0,
                0.0,
                0.0,  # orientation quaternion
                0.5,
                0.3,
                0.1,  # linear velocity
                0.1,
                0.05,
                0.02,  # angular velocity
            ],
            dtype=np.float32,
        ),
        "goal": np.array(
            [
                5.0,
                5.0,
                5.0,  # goal position
                1.0,
                0.0,
                0.0,
                0.0,  # goal orientation quaternion
            ],
            dtype=np.float32,
        ),
    }


@pytest.fixture()
def at_goal_obs() -> dict[str, np.ndarray]:
    """Observation where spacecraft is already at goal."""
    return {
        "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
        "proprio": np.array(
            [
                5.0,
                5.0,
                5.0,  # position (matches goal)
                1.0,
                0.0,
                0.0,
                0.0,  # orientation
                0.0,
                0.0,
                0.0,  # zero linear velocity
                0.0,
                0.0,
                0.0,  # zero angular velocity
            ],
            dtype=np.float32,
        ),
        "goal": np.array(
            [
                5.0,
                5.0,
                5.0,  # goal position
                1.0,
                0.0,
                0.0,
                0.0,  # goal orientation
            ],
            dtype=np.float32,
        ),
    }


@pytest.fixture()
def aligned_obs() -> dict[str, np.ndarray]:
    """Observation where spacecraft is already aligned with goal orientation."""
    return {
        "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
        "proprio": np.array(
            [
                1.0,
                2.0,
                3.0,  # position
                1.0,
                0.0,
                0.0,
                0.0,  # orientation (matches goal)
                0.5,
                0.3,
                0.1,  # linear velocity
                0.1,
                0.05,
                0.02,  # angular velocity
            ],
            dtype=np.float32,
        ),
        "goal": np.array(
            [
                5.0,
                5.0,
                5.0,  # goal position
                1.0,
                0.0,
                0.0,
                0.0,  # goal orientation (matches proprio)
            ],
            dtype=np.float32,
        ),
    }


@pytest.fixture()
def zero_velocity_obs() -> dict[str, np.ndarray]:
    """Observation with zero velocities."""
    return {
        "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
        "proprio": np.array(
            [
                1.0,
                2.0,
                3.0,  # position
                1.0,
                0.0,
                0.0,
                0.0,  # orientation
                0.0,
                0.0,
                0.0,  # zero linear velocity
                0.0,
                0.0,
                0.0,  # zero angular velocity
            ],
            dtype=np.float32,
        ),
        "goal": np.array(
            [
                5.0,
                5.0,
                5.0,  # goal position
                1.0,
                0.0,
                0.0,
                0.0,  # goal orientation
            ],
            dtype=np.float32,
        ),
    }


# ---------------------------------------------------------------------------
# TranslateToGoalSkill tests
# ---------------------------------------------------------------------------


class TestTranslateToGoalSkill:
    """Test translation skill for moving toward goal position."""

    def test_action_shape_and_range(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that action has correct shape and values in [-1, 1]."""
        skill = TranslateToGoalSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Verify shape
        assert action.shape == (6,)

        # Verify range
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_thrust_direction_toward_goal(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that thrust vector points toward goal position."""
        skill = TranslateToGoalSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Extract position and goal from observation
        current_pos = skill_obs["proprio"][:3]
        goal_pos = skill_obs["goal"][:3]

        # Compute expected direction
        direction = goal_pos - current_pos
        direction_norm = direction / np.linalg.norm(direction)

        # Extract thrust from action
        thrust = action[:3]
        thrust_norm = thrust / np.linalg.norm(thrust)

        # Verify thrust points in the same direction as goal
        dot_product = np.dot(thrust_norm, direction_norm)
        assert dot_product > 0.99  # Allow small numerical error

    def test_zero_torque(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that torque component is zero (translation only)."""
        skill = TranslateToGoalSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Torque should be zero
        torque = action[3:]
        np.testing.assert_array_almost_equal(torque, np.zeros(3))

    def test_at_goal_produces_zero_action(self, at_goal_obs: dict[str, np.ndarray]) -> None:
        """Test that when position equals goal, thrust is zero."""
        skill = TranslateToGoalSkill(gain=1.0)
        action = skill.compute_action(at_goal_obs)

        # Thrust should be zero when at goal
        thrust = action[:3]
        assert np.linalg.norm(thrust) < 1e-5


# ---------------------------------------------------------------------------
# AlignToGoalSkill tests
# ---------------------------------------------------------------------------


class TestAlignToGoalSkill:
    """Test alignment skill for orienting toward goal."""

    def test_action_shape_and_range(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that action has correct shape and values in [-1, 1]."""
        skill = AlignToGoalSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Verify shape
        assert action.shape == (6,)

        # Verify range
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_already_aligned_zero_torque(self, aligned_obs: dict[str, np.ndarray]) -> None:
        """Test that when orientations match, torque is zero."""
        skill = AlignToGoalSkill(gain=1.0)
        action = skill.compute_action(aligned_obs)

        # Torque should be very small when aligned
        torque = action[3:]
        assert np.linalg.norm(torque) < 1e-5

    def test_zero_thrust(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that thrust component is zero (alignment only)."""
        skill = AlignToGoalSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Thrust should be zero
        thrust = action[:3]
        np.testing.assert_array_almost_equal(thrust, np.zeros(3))


# ---------------------------------------------------------------------------
# BrakeSkill tests
# ---------------------------------------------------------------------------


class TestBrakeSkill:
    """Test braking skill for velocity damping."""

    def test_action_shape_and_range(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that action has correct shape and values in [-1, 1]."""
        skill = BrakeSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Verify shape
        assert action.shape == (6,)

        # Verify range
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_opposing_velocity_direction(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that brake action opposes current velocity."""
        skill = BrakeSkill(gain=1.0)
        action = skill.compute_action(skill_obs)

        # Extract velocities from observation
        linear_vel = skill_obs["proprio"][7:10]
        angular_vel = skill_obs["proprio"][10:13]

        # Extract thrust and torque from action
        thrust = action[:3]
        torque = action[3:]

        # Verify thrust opposes linear velocity (negative dot product)
        if np.linalg.norm(linear_vel) > 1e-6:
            thrust_norm = thrust / np.linalg.norm(thrust)
            vel_norm = linear_vel / np.linalg.norm(linear_vel)
            dot_product = np.dot(thrust_norm, vel_norm)
            assert dot_product < -0.99  # Should be opposite direction

        # Verify torque opposes angular velocity
        if np.linalg.norm(angular_vel) > 1e-6:
            torque_norm = torque / np.linalg.norm(torque)
            ang_vel_norm = angular_vel / np.linalg.norm(angular_vel)
            dot_product = np.dot(torque_norm, ang_vel_norm)
            assert dot_product < -0.99

    def test_zero_velocity_zero_action(self, zero_velocity_obs: dict[str, np.ndarray]) -> None:
        """Test that zero velocity produces zero brake action."""
        skill = BrakeSkill(gain=1.0)
        action = skill.compute_action(zero_velocity_obs)

        # Action should be zero when velocities are zero
        np.testing.assert_array_almost_equal(action, np.zeros(6))


# ---------------------------------------------------------------------------
# StationKeepSkill tests
# ---------------------------------------------------------------------------


class TestStationKeepSkill:
    """Test station-keeping skill."""

    def test_action_shape_and_range(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that action has correct shape and values in [-1, 1]."""
        skill = StationKeepSkill(gain=0.5, brake_gain=1.0)
        action = skill.compute_action(skill_obs)

        # Verify shape
        assert action.shape == (6,)

        # Verify range
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_combines_brake_and_position(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that station-keeping combines braking and position correction."""
        skill = StationKeepSkill(gain=0.5, brake_gain=1.0)
        action = skill.compute_action(skill_obs)

        # Should produce non-zero action when not at goal and have velocity
        thrust = action[:3]
        torque = action[3:]

        # Should have some thrust component (position + brake)
        assert np.linalg.norm(thrust) > 0.0

        # Should have some torque component (angular brake)
        assert np.linalg.norm(torque) > 0.0


# ---------------------------------------------------------------------------
# ApproachSkill tests
# ---------------------------------------------------------------------------


class TestApproachSkill:
    """Test composite approach skill."""

    def test_action_shape_and_range(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that action has correct shape and values in [-1, 1]."""
        skill = ApproachSkill(
            translate_gain=1.0,
            align_gain=0.8,
            brake_gain=1.0,
            brake_distance_threshold=5.0,
        )
        action = skill.compute_action(skill_obs)

        # Verify shape
        assert action.shape == (6,)

        # Verify range
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_far_from_goal_translation_dominates(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that when far from goal, translation dominates over braking."""
        # Modify observation to be far from goal
        far_obs = skill_obs.copy()
        far_obs["proprio"] = far_obs["proprio"].copy()
        far_obs["proprio"][:3] = np.array([0.0, 0.0, 0.0])  # Far from goal at [5, 5, 5]
        far_obs["proprio"][7:10] = np.array([0.5, 0.3, 0.1])  # Some velocity

        skill = ApproachSkill(
            translate_gain=1.0,
            align_gain=0.8,
            brake_gain=1.0,
            brake_distance_threshold=5.0,
        )
        action = skill.compute_action(far_obs)

        # Should have significant thrust
        thrust = action[:3]
        assert np.linalg.norm(thrust) > 0.1

    def test_close_to_goal_braking_increases(self) -> None:
        """Test that braking increases as spacecraft gets close to goal."""
        # Create observation close to goal
        close_obs = {
            "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
            "proprio": np.array(
                [
                    4.0,
                    4.5,
                    4.5,  # Close to goal [5, 5, 5]
                    1.0,
                    0.0,
                    0.0,
                    0.0,  # orientation
                    1.0,
                    1.0,
                    1.0,  # High velocity
                    0.0,
                    0.0,
                    0.0,  # zero angular velocity
                ],
                dtype=np.float32,
            ),
            "goal": np.array(
                [
                    5.0,
                    5.0,
                    5.0,  # goal position
                    1.0,
                    0.0,
                    0.0,
                    0.0,  # goal orientation
                ],
                dtype=np.float32,
            ),
        }

        skill = ApproachSkill(
            translate_gain=1.0,
            align_gain=0.8,
            brake_gain=1.0,
            brake_distance_threshold=5.0,
        )
        action = skill.compute_action(close_obs)

        # Should produce action (combination of translate and brake)
        thrust = action[:3]
        assert np.linalg.norm(thrust) > 0.0


# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    """Test skill registration and factory."""

    def test_list_skills_returns_five(self) -> None:
        """Test that five skills are registered."""
        skills = list_skills()

        # Should have 5 skills
        assert len(skills) == 5

        # Verify expected skill names
        expected_skills = {
            "TranslateToGoalSkill",
            "AlignToGoalSkill",
            "BrakeSkill",
            "StationKeepSkill",
            "ApproachSkill",
        }
        assert set(skills) == expected_skills

    def test_get_skill_by_name(self, skill_obs: dict[str, np.ndarray]) -> None:
        """Test that each skill can be retrieved by class name."""
        skill_names = [
            "TranslateToGoalSkill",
            "AlignToGoalSkill",
            "BrakeSkill",
            "StationKeepSkill",
            "ApproachSkill",
        ]

        for name in skill_names:
            skill = get_skill(name)
            assert skill is not None
            assert hasattr(skill, "compute_action")

            # Verify it can compute actions
            action = skill.compute_action(skill_obs)
            assert action.shape == (6,)

    def test_unknown_skill_raises_key_error(self) -> None:
        """Test that requesting unknown skill raises ValueError."""
        with pytest.raises(ValueError, match="Unknown skill"):
            get_skill("NonexistentSkill")
