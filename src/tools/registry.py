"""Tool registry with factory pattern."""

from __future__ import annotations

from typing import Any

import structlog

from src.tools.base import Tool

logger = structlog.get_logger(__name__)

_TOOL_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(cls: type[Tool]) -> type[Tool]:
    """Class decorator that registers a Tool in the global registry."""
    name = cls.__name__
    if name in _TOOL_REGISTRY:
        logger.warning("tool_already_registered", name=name, cls=cls.__qualname__)
    _TOOL_REGISTRY[name] = cls
    logger.info("tool_registered", name=name, cls=cls.__qualname__)
    return cls


def get_tool(name: str, **kwargs: Any) -> Tool:
    """Instantiate a registered tool by name.

    Args:
        name: The tool class name (e.g., ``"DistanceToGoalTool"``).
        **kwargs: Keyword arguments passed to the tool's ``__init__``.

    Returns:
        An instance of the requested tool.

    Raises:
        ValueError: If the tool name is not registered.
    """
    cls = _TOOL_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_TOOL_REGISTRY))
        raise ValueError(f"Unknown tool '{name}'. Available: {available}")
    tool = cls(**kwargs)
    logger.debug("tool_created", name=name, tool_instance=tool.name)
    return tool


def list_tools() -> list[str]:
    """Return sorted list of registered tool names."""
    return sorted(_TOOL_REGISTRY)
