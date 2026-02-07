"""Tool registry with factory pattern."""

from __future__ import annotations

from typing import Any

import structlog

from src.tools.base import Tool

logger = structlog.get_logger(__name__)

_TOOL_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(cls: type[Tool]) -> type[Tool]:
    """Class decorator that registers a Tool in the global registry."""
    _TOOL_REGISTRY[cls.__name__] = cls
    return cls


def get_tool(name: str, **kwargs: Any) -> Tool:
    """Instantiate a registered tool by name."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list_tools()}")
    return _TOOL_REGISTRY[name](**kwargs)


def list_tools() -> list[str]:
    """Return sorted list of registered tool names."""
    return sorted(_TOOL_REGISTRY)
