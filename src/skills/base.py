"""Skill base class for the ZeroG-RL system."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Skill(ABC):
    """Abstract base class for spacecraft control skills.

    Skills compute deterministic control actions from observations,
    providing domain-knowledge-based primitives that can be blended
    with learned policies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable skill name."""
        ...

    @property
    def weight(self) -> float:
        """Blending weight priority (default 1.0)."""
        return 1.0

    @abstractmethod
    def compute_action(self, observation: dict[str, np.ndarray], **kwargs: object) -> np.ndarray:
        """Compute a 6-DOF action from the observation.

        Args:
            observation: Dict with keys ``voxels``, ``proprio``, ``goal``.
                - proprio[:3] = position, proprio[3:7] = orientation quaternion [w,x,y,z],
                  proprio[7:10] = linear velocity, proprio[10:13] = angular velocity
                - goal[:3] = target position, goal[3:7] = target orientation quaternion

        Returns:
            Action array of shape ``(6,)`` in ``[-1, 1]``.
        """
        ...
