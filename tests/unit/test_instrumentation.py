"""Unit tests for debugging instrumentation decorators."""

from __future__ import annotations

from unittest.mock import patch

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
        """Calling with CPU device should not raise and returns None."""
        result = log_gpu_memory("cpu")
        assert result is None

    def test_cuda_string(self) -> None:
        """Calling with 'cuda' string works (no-ops if no GPU)."""
        result = log_gpu_memory("cuda")
        assert result is None

    def test_device_object(self) -> None:
        """Calling with torch.device works."""
        result = log_gpu_memory(torch.device("cpu"))
        assert result is None


class TestLogTensorStats:
    def test_normal_tensor(self) -> None:
        """Logging a normal tensor should log stats via the logger."""
        t = torch.randn(10, 10)
        with patch("src.debugging.instrumentation.logger") as mock_logger:
            log_tensor_stats("test_tensor", t)
            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args
            assert call_kwargs[0][0] == "tensor_stats"
            assert call_kwargs[1]["name"] == "test_tensor"
            assert "mean" in call_kwargs[1]
            assert "std" in call_kwargs[1]

    def test_empty_tensor(self) -> None:
        """Logging an empty tensor should log with empty=True."""
        t = torch.empty(0)
        with patch("src.debugging.instrumentation.logger") as mock_logger:
            log_tensor_stats("empty", t)
            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args
            assert call_kwargs[1]["empty"] is True

    def test_scalar_tensor(self) -> None:
        """Logging a scalar tensor should report std=0."""
        t = torch.tensor(3.14)
        with patch("src.debugging.instrumentation.logger") as mock_logger:
            log_tensor_stats("scalar", t)
            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args
            assert call_kwargs[1]["std"] == 0.0

    def test_integer_tensor(self) -> None:
        """Integer tensor converted to float for stats."""
        t = torch.arange(10)
        with patch("src.debugging.instrumentation.logger") as mock_logger:
            log_tensor_stats("int_tensor", t)
            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args
            assert call_kwargs[1]["name"] == "int_tensor"
            assert call_kwargs[1]["shape"] == [10]


class TestStructlogCapture:
    """Tests for verifying structured log output."""

    def test_log_tensor_stats_emits_event(self) -> None:
        """log_tensor_stats should emit a structured log event via mock."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("src.debugging.instrumentation.logger") as mock_logger:
            log_tensor_stats("test_tensor", t)

            # Verify debug was called
            mock_logger.debug.assert_called_once()

            # Extract the call kwargs
            call_args = mock_logger.debug.call_args
            assert call_args[0][0] == "tensor_stats"
            kwargs = call_args[1]
            assert kwargs["name"] == "test_tensor"
            assert "shape" in kwargs
            assert "mean" in kwargs
            assert kwargs["mean"] == 3.0
            assert kwargs["min"] == 1.0
            assert kwargs["max"] == 5.0
