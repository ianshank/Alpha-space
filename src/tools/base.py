"""Tool base class for the ZeroG-RL system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Tool(ABC):
    """Abstract base class for observation analysis tools.

    Tools extract derived quantities from observations to support
    decision-making by the agent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tool name."""
        ...

    @abstractmethod
    def __call__(self, observation: dict[str, np.ndarray], **kwargs: object) -> Any:
        """Execute the tool on an observation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.
                - proprio[:3] = position, proprio[3:7] = orientation quaternion [w,x,y,z],
                  proprio[7:10] = linear velocity, proprio[10:13] = angular velocity
                - goal[:3] = target position, goal[3:7] = target orientation quaternion

        Returns:
            Tool-specific result.
        """
        ...
