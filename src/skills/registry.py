"""Skill registry and factory pattern for the ZeroG-RL system."""

from __future__ import annotations

from typing import Any

import structlog

from src.skills.base import Skill

logger = structlog.get_logger(__name__)

_SKILL_REGISTRY: dict[str, type[Skill]] = {}


def register_skill(cls: type[Skill]) -> type[Skill]:
    """Decorator to register a skill class in the global registry.

    Args:
        cls: A concrete ``Skill`` subclass.

    Returns:
        The same class (allows chaining).

    Example::

        @register_skill
        class MySkill(Skill):
            ...
    """
    # Use the class name as the registry key
    name = cls.__name__
    if name in _SKILL_REGISTRY:
        logger.warning("skill_already_registered", name=name, cls=cls.__qualname__)
    _SKILL_REGISTRY[name] = cls
    logger.info("skill_registered", name=name, cls=cls.__qualname__)
    return cls


def get_skill(name: str, **kwargs: Any) -> Skill:
    """Factory function to create a skill instance by name.

    Args:
        name: The skill class name (e.g., ``"TranslateToGoalSkill"``).
        **kwargs: Keyword arguments passed to the skill's ``__init__``.

    Returns:
        An instance of the requested skill.

    Raises:
        ValueError: If the skill name is not registered.

    Example::

        skill = get_skill("TranslateToGoalSkill", gain=0.5)
    """
    cls = _SKILL_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_SKILL_REGISTRY))
        raise ValueError(f"Unknown skill '{name}'. Available: {available}")

    skill = cls(**kwargs)
    logger.debug("skill_created", name=name, skill_instance=skill.name)
    return skill


def list_skills() -> list[str]:
    """List all registered skill names.

    Returns:
        Sorted list of registered skill class names.
    """
    return sorted(_SKILL_REGISTRY.keys())
