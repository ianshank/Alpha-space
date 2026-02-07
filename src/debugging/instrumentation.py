"""Debugging and profiling decorators / utilities.

Provides lightweight wrappers for timing, memory tracking, and GPU memory
reporting that can be toggled on/off via log-level configuration.
"""

from __future__ import annotations

import functools
import time
import tracemalloc
from typing import Any, Callable, ParamSpec, TypeVar

import structlog
import torch

P = ParamSpec("P")
R = TypeVar("R")

logger = structlog.get_logger(__name__)


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs the wall-clock execution time at DEBUG level.

    Args:
        func: The function to instrument.

    Returns:
        Wrapped function that logs elapsed seconds on each call.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(
            "function_executed",
            function=func.__qualname__,
            elapsed_seconds=round(elapsed, 6),
        )
        return result

    return wrapper


def log_memory_usage(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs peak memory allocation during a function call.

    Uses ``tracemalloc`` to measure the allocation delta. Only useful when
    called from an already-traced context or when tracing is explicitly
    started inside the decorator.

    Args:
        func: The function to instrument.

    Returns:
        Wrapped function that logs current and peak memory (MB).
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            logger.debug(
                "function_memory_usage",
                function=func.__qualname__,
                current_mb=round(current / (1024 * 1024), 3),
                peak_mb=round(peak / (1024 * 1024), 3),
            )
            return result
        finally:
            if not was_tracing:
                tracemalloc.stop()

    return wrapper


def log_gpu_memory(device: torch.device | str = "cuda") -> None:
    """Log current GPU memory usage at DEBUG level.

    Silently no-ops when CUDA is unavailable or *device* is CPU.

    Args:
        device: The torch device to query.
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
    logger.debug(
        "gpu_memory_usage",
        device=str(device),
        allocated_gb=round(allocated_gb, 4),
        reserved_gb=round(reserved_gb, 4),
    )


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log summary statistics of a tensor at DEBUG level.

    Useful for monitoring gradient norms, activation magnitudes, etc.

    Args:
        name: Descriptive label for the tensor.
        tensor: The tensor to summarise.
    """
    if tensor.numel() == 0:
        logger.debug("tensor_stats", name=name, empty=True)
        return

    data = tensor.detach().float()
    std_val = round(data.std().item(), 6) if data.numel() > 1 else 0.0
    logger.debug(
        "tensor_stats",
        name=name,
        shape=list(tensor.shape),
        mean=round(data.mean().item(), 6),
        std=std_val,
        min=round(data.min().item(), 6),
        max=round(data.max().item(), 6),
        norm=round(data.norm().item(), 6),
    )
