"""Tools module for observation analysis and decision support."""

from __future__ import annotations

from src.tools.base import Tool
from src.tools.planning import FuelEstimatorTool, TrajectoryPlannerTool
from src.tools.registry import get_tool, list_tools, register_tool
from src.tools.sensors import (
    DistanceToGoalTool,
    DockingProgressTool,
    OrientationErrorTool,
    VelocityMagnitudeTool,
)

__all__ = [
    "DistanceToGoalTool",
    "DockingProgressTool",
    "FuelEstimatorTool",
    "OrientationErrorTool",
    "Tool",
    "TrajectoryPlannerTool",
    "VelocityMagnitudeTool",
    "get_tool",
    "list_tools",
    "register_tool",
]
