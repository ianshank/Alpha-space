"""Common utility functions for the ZeroG-RL training system.

Provides deterministic seeding, device selection, quaternion math,
action clamping, timing utilities, and path validation.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Seeds Python's ``random`` module, NumPy, PyTorch (CPU and CUDA), and
    configures CuDNN for deterministic behaviour.

    Args:
        seed: Non-negative integer used as the global seed.

    Raises:
        ValueError: If *seed* is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 – legacy API required for broad compat
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Python hash seed (only effective if set before interpreter start, but
    # we record it for completeness).
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("seeded_all_rngs", seed=seed, cuda_available=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available ``torch.device``.

    Args:
        prefer_gpu: When ``True`` (default), return a CUDA device if one is
            available.  Otherwise, always return CPU.

    Returns:
        A ``torch.device`` instance.
    """
    use_cuda = prefer_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("device_selected", device=str(device), prefer_gpu=prefer_gpu)
    return device


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

_QUATERNION_NORM_EPS: float = 1e-10


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion (or batch of quaternions) to unit length.

    Args:
        q: Quaternion array of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Unit-length quaternion(s) with the same shape as *q*.

    Raises:
        ValueError: If *q* does not have the expected shape or if any
            quaternion has a norm smaller than ``1e-10``.
    """
    q = np.asarray(q, dtype=np.float64)

    if q.ndim == 1:
        if q.shape[0] != 4:
            raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}")
        norm = np.linalg.norm(q)
        if norm < _QUATERNION_NORM_EPS:
            raise ValueError(
                f"Quaternion norm {norm} is below threshold {_QUATERNION_NORM_EPS}"
            )
        return q / norm

    if q.ndim == 2:
        if q.shape[1] != 4:
            raise ValueError(f"Expected quaternions of shape (N, 4), got {q.shape}")
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(norms < _QUATERNION_NORM_EPS):
            raise ValueError(
                "One or more quaternion norms are below threshold "
                f"{_QUATERNION_NORM_EPS}"
            )
        return q / norms

    raise ValueError(f"Expected 1-D or 2-D array, got ndim={q.ndim}")


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion(s) ``[w, x, y, z]`` to 3x3 rotation matrices.

    Uses the standard Hamilton convention where *w* is the scalar part.

    Args:
        q: Quaternion array of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Rotation matrix of shape ``(3, 3)`` or ``(N, 3, 3)``.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[np.newaxis, :]

    q = normalize_quaternion(q)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Pre-compute repeated products.
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    n = q.shape[0]
    rot = np.empty((n, 3, 3), dtype=np.float64)

    rot[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[:, 0, 1] = 2.0 * (xy - wz)
    rot[:, 0, 2] = 2.0 * (xz + wy)

    rot[:, 1, 0] = 2.0 * (xy + wz)
    rot[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[:, 1, 2] = 2.0 * (yz - wx)

    rot[:, 2, 0] = 2.0 * (xz - wy)
    rot[:, 2, 1] = 2.0 * (yz + wx)
    rot[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    if single:
        return rot[0]
    return rot


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute the Hamilton product of two quaternions ``[w, x, y, z]``.

    Supports broadcasting: both inputs may be single ``(4,)`` or batched
    ``(N, 4)`` arrays.

    Args:
        q1: First quaternion(s), shape ``(4,)`` or ``(N, 4)``.
        q2: Second quaternion(s), shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Hamilton product ``q1 * q2`` with shape ``(4,)`` or ``(N, 4)``.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    single = q1.ndim == 1 and q2.ndim == 1
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    result = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )

    if single:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Action clamping
# ---------------------------------------------------------------------------


def clamp_actions(
    actions: np.ndarray,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Clamp action values to a safe range.

    Args:
        actions: Array of action values (arbitrary shape).
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        A new array with values clamped to ``[min_val, max_val]``.

    Raises:
        ValueError: If ``min_val > max_val``.
    """
    if min_val > max_val:
        raise ValueError(
            f"min_val ({min_val}) must be <= max_val ({max_val})"
        )
    return np.clip(actions, min_val, max_val)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that measures wall-clock elapsed time.

    The elapsed time is recorded using ``time.perf_counter`` and logged at
    ``DEBUG`` level when the context exits.

    Args:
        name: Human-readable label included in the log message.
        log: A structlog-compatible logger.  Defaults to the module logger.

    Attributes:
        elapsed: Elapsed time in seconds, available after the context exits.

    Example::

        with Timer("forward_pass") as t:
            output = model(x)
        print(t.elapsed)
    """

    def __init__(
        self,
        name: str,
        log: structlog.typing.FilteringBoundLogger | None = None,
    ) -> None:
        self._name = name
        self._log = log or logger
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
        self._log.debug("timer_elapsed", name=self._name, seconds=self.elapsed)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_path(
    path: Path,
    base_dir: Path | None = None,
) -> Path:
    """Validate and resolve a filesystem path.

    When *base_dir* is provided the function ensures that the resolved path
    is located within *base_dir*, guarding against path-traversal attacks
    (e.g. ``../../etc/passwd``).

    Args:
        path: The path to validate.
        base_dir: If given, *path* must resolve to a location inside this
            directory.

    Returns:
        The resolved (absolute) path.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the resolved path escapes *base_dir*.
    """
    resolved = path.resolve()

    if base_dir is not None:
        resolved_base = base_dir.resolve()
        # Use os.path.commonpath to avoid string-prefix false positives
        # (e.g. /data2 being treated as inside /data).
        try:
            common = Path(os.path.commonpath([resolved, resolved_base]))
        except ValueError:
            # On Windows, paths on different drives have no common path.
            raise ValueError(
                f"Path {resolved} is not relative to base directory {resolved_base}"
            ) from None

        if common != resolved_base:
            raise ValueError(
                f"Path {resolved} is not relative to base directory {resolved_base}"
            )

    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved
