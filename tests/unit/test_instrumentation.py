"""Unit tests for debugging instrumentation decorators."""

from __future__ import annotations

import time

import torch

from src.debugging.instrumentation import (
    log_execution_time,
    log_gpu_memory,
    log_memory_usage,
    log_tensor_stats,
)


class TestLogExecutionTime:
    def test_wrapped_function_runs(self) -> None:
        @log_execution_time
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    def test_preserves_return_value(self) -> None:
        @log_execution_time
        def identity(x: str) -> str:
            return x

        assert identity("hello") == "hello"

    def test_wraps_name_preserved(self) -> None:
        @log_execution_time
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"


class TestLogMemoryUsage:
    def test_wrapped_function_runs(self) -> None:
        @log_memory_usage
        def allocate() -> list[int]:
            return list(range(100))

        result = allocate()
        assert len(result) == 100

    def test_preserves_return_value(self) -> None:
        @log_memory_usage
        def compute() -> int:
            return 42

        assert compute() == 42


class TestLogGpuMemory:
    def test_cpu_device_noop(self) -> None:
        """Calling with CPU device should not raise."""
        log_gpu_memory("cpu")

    def test_cuda_string(self) -> None:
        """Calling with 'cuda' string works (no-ops if no GPU)."""
        log_gpu_memory("cuda")

    def test_device_object(self) -> None:
        """Calling with torch.device works."""
        log_gpu_memory(torch.device("cpu"))


class TestLogTensorStats:
    def test_normal_tensor(self) -> None:
        """Logging a normal tensor should not raise."""
        t = torch.randn(10, 10)
        log_tensor_stats("test_tensor", t)

    def test_empty_tensor(self) -> None:
        """Logging an empty tensor should not raise."""
        t = torch.empty(0)
        log_tensor_stats("empty", t)

    def test_scalar_tensor(self) -> None:
        """Logging a scalar tensor should not raise."""
        t = torch.tensor(3.14)
        log_tensor_stats("scalar", t)

    def test_integer_tensor(self) -> None:
        """Integer tensor converted to float for stats."""
        t = torch.arange(10)
        log_tensor_stats("int_tensor", t)
