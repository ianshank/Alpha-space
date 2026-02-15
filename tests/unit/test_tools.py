"""Comprehensive unit tests for src/tools/sensors.py and src/tools/planning.py.

Tests sensor tools (DistanceToGoalTool, OrientationErrorTool, VelocityMagnitudeTool,
DockingProgressTool) and planning tools (TrajectoryPlannerTool, FuelEstimatorTool),
as well as the tool registry functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.planning import FuelEstimatorTool, TrajectoryPlannerTool
from src.tools.registry import get_tool, list_tools
from src.tools.sensors import (
    DistanceToGoalTool,
    DockingProgressTool,
    OrientationErrorTool,
    VelocityMagnitudeTool,
)


@pytest.fixture()
def tool_obs() -> dict[str, np.ndarray]:
    """Standard observation dictionary for tool testing."""
    return {
        "voxels": np.zeros((4, 16, 16, 16), dtype=np.float32),
        "proprio": np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.1, 0.05, 0.02],
            dtype=np.float32,
        ),
        "goal": np.array([5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }


# =========================================================================
# DistanceToGoalTool
# =========================================================================


class TestDistanceToGoalTool:
    """Tests for the DistanceToGoalTool sensor."""

    def test_distance_positive(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Distance should be positive when not at goal."""
        tool = DistanceToGoalTool()
        distance = tool(tool_obs)
        assert distance > 0.0

    def test_distance_at_goal_is_zero(self, tool_obs: dict[str, np.ndarray]) -> None:
        """When position matches goal, distance should be zero."""
        tool = DistanceToGoalTool()
        # Set position to match goal position
        tool_obs["proprio"][:3] = tool_obs["goal"][:3]
        distance = tool(tool_obs)
        assert distance == 0.0

    def test_known_distance(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Verify computed distance matches expected value."""
        tool = DistanceToGoalTool()
        # Position: [1, 2, 3], Goal: [5, 5, 5]
        # Distance = sqrt((5-1)^2 + (5-2)^2 + (5-3)^2) = sqrt(16 + 9 + 4) = sqrt(29)
        expected_distance = np.sqrt(29.0)
        distance = tool(tool_obs)
        assert np.isclose(distance, expected_distance, rtol=1e-6)


# =========================================================================
# OrientationErrorTool
# =========================================================================


class TestOrientationErrorTool:
    """Tests for the OrientationErrorTool sensor."""

    def test_zero_error_when_aligned(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Same quaternion should result in zero degrees error."""
        tool = OrientationErrorTool()
        # Set current orientation to match goal orientation
        tool_obs["proprio"][3:7] = tool_obs["goal"][3:7]
        error = tool(tool_obs)
        assert np.isclose(error, 0.0, atol=1e-6)

    def test_positive_error_when_misaligned(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Different quaternions should result in positive angle error."""
        tool = OrientationErrorTool()
        # Set different orientation (90 degree rotation around z-axis)
        tool_obs["proprio"][3:7] = np.array([0.707, 0.0, 0.0, 0.707])
        tool_obs["goal"][3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        error = tool(tool_obs)
        assert error > 0.0

    def test_returns_degrees(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Output should be in degrees (not radians)."""
        tool = OrientationErrorTool()
        # 90 degree rotation around z-axis
        tool_obs["proprio"][3:7] = np.array([0.707, 0.0, 0.0, 0.707])
        tool_obs["goal"][3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        error = tool(tool_obs)
        # Should be approximately 90 degrees
        assert 80.0 < error < 100.0


# =========================================================================
# VelocityMagnitudeTool
# =========================================================================


class TestVelocityMagnitudeTool:
    """Tests for the VelocityMagnitudeTool sensor."""

    def test_returns_dict_with_linear_and_angular(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Should return dict with 'linear' and 'angular' keys."""
        tool = VelocityMagnitudeTool()
        result = tool(tool_obs)
        assert isinstance(result, dict)
        assert "linear" in result
        assert "angular" in result

    def test_zero_velocity(self, tool_obs: dict[str, np.ndarray]) -> None:
        """When velocities are zero, both magnitudes should be zero."""
        tool = VelocityMagnitudeTool()
        tool_obs["proprio"][7:13] = 0.0
        result = tool(tool_obs)
        assert result["linear"] == 0.0
        assert result["angular"] == 0.0

    def test_correct_magnitudes(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Verify computed magnitudes match expected values."""
        tool = VelocityMagnitudeTool()
        # Linear velocity: [0.5, 0.3, 0.1]
        # Angular velocity: [0.1, 0.05, 0.02]
        expected_linear = np.linalg.norm([0.5, 0.3, 0.1])
        expected_angular = np.linalg.norm([0.1, 0.05, 0.02])
        result = tool(tool_obs)
        assert np.isclose(result["linear"], expected_linear, rtol=1e-6)
        assert np.isclose(result["angular"], expected_angular, rtol=1e-6)


# =========================================================================
# DockingProgressTool
# =========================================================================


class TestDockingProgressTool:
    """Tests for the DockingProgressTool sensor."""

    def test_progress_at_goal_is_one(self, tool_obs: dict[str, np.ndarray]) -> None:
        """At goal with matching orientation should give progress ~1.0."""
        tool = DockingProgressTool()
        # Set position and orientation to match goal
        tool_obs["proprio"][:3] = tool_obs["goal"][:3]
        tool_obs["proprio"][3:7] = tool_obs["goal"][3:7]
        progress = tool(tool_obs)
        assert np.isclose(progress, 1.0, rtol=1e-2)

    def test_progress_far_away_is_near_zero(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Far from goal should give progress close to zero."""
        tool = DockingProgressTool()
        # Move position very far from goal
        tool_obs["proprio"][:3] = np.array([100.0, 100.0, 100.0])
        progress = tool(tool_obs)
        assert progress < 0.1

    def test_progress_in_range_zero_one(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Progress should always be in range [0, 1]."""
        tool = DockingProgressTool()
        progress = tool(tool_obs)
        assert 0.0 <= progress <= 1.0


# =========================================================================
# TrajectoryPlannerTool
# =========================================================================


class TestTrajectoryPlannerTool:
    """Tests for the TrajectoryPlannerTool planning tool."""

    def test_returns_dict(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Should return dict with 'distance' and 'estimated_time' keys."""
        tool = TrajectoryPlannerTool()
        result = tool(tool_obs)
        assert isinstance(result, dict)
        assert "distance" in result
        assert "estimated_time" in result

    def test_at_goal_zero_distance(self, tool_obs: dict[str, np.ndarray]) -> None:
        """At goal, distance and time should be zero."""
        tool = TrajectoryPlannerTool()
        tool_obs["proprio"][:3] = tool_obs["goal"][:3]
        result = tool(tool_obs)
        assert result["distance"] == 0.0
        assert result["estimated_time"] == 0.0

    def test_positive_time_estimate(self, tool_obs: dict[str, np.ndarray]) -> None:
        """When not at goal, time estimate should be positive."""
        tool = TrajectoryPlannerTool()
        result = tool(tool_obs)
        assert result["estimated_time"] > 0.0


# =========================================================================
# FuelEstimatorTool
# =========================================================================


class TestFuelEstimatorTool:
    """Tests for the FuelEstimatorTool planning tool."""

    def test_returns_float(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Should return a numeric value."""
        tool = FuelEstimatorTool()
        result = tool(tool_obs)
        assert isinstance(result, (float, np.floating))

    def test_zero_velocity_zero_fuel(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Zero velocity should result in zero fuel estimate."""
        tool = FuelEstimatorTool()
        tool_obs["proprio"][7:13] = 0.0
        result = tool(tool_obs)
        assert result == 0.0

    def test_positive_fuel_when_moving(self, tool_obs: dict[str, np.ndarray]) -> None:
        """Non-zero velocity should result in positive fuel estimate."""
        tool = FuelEstimatorTool()
        result = tool(tool_obs)
        assert result > 0.0


# =========================================================================
# ToolRegistry
# =========================================================================


class TestToolRegistry:
    """Tests for the tool registry system."""

    def test_list_tools_returns_six(self) -> None:
        """Verify that 6 tools are registered."""
        tools = list_tools()
        assert len(tools) == 6

    def test_get_tool_by_name(self) -> None:
        """Should be able to retrieve tools by name."""
        tool = get_tool("DistanceToGoalTool")
        assert isinstance(tool, DistanceToGoalTool)

        tool = get_tool("OrientationErrorTool")
        assert isinstance(tool, OrientationErrorTool)

        tool = get_tool("VelocityMagnitudeTool")
        assert isinstance(tool, VelocityMagnitudeTool)

        tool = get_tool("DockingProgressTool")
        assert isinstance(tool, DockingProgressTool)

        tool = get_tool("TrajectoryPlannerTool")
        assert isinstance(tool, TrajectoryPlannerTool)

        tool = get_tool("FuelEstimatorTool")
        assert isinstance(tool, FuelEstimatorTool)

    def test_unknown_tool_raises(self) -> None:
        """Getting an unknown tool should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            get_tool("NonExistentTool")
