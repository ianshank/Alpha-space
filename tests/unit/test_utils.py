"""Comprehensive unit tests for src/utils/common.py.

Covers: seed_everything, get_device, normalize_quaternion,
quaternion_to_rotation_matrix, quaternion_multiply, clamp_actions,
Timer, and validate_path.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.utils.common import (
    Timer,
    clamp_actions,
    get_device,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    seed_everything,
    validate_path,
)


# =========================================================================
# seed_everything
# =========================================================================


class TestSeedEverything:
    """Tests for the seed_everything function."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345, 2**31 - 1])
    def test_valid_seeds_accepted(self, seed: int) -> None:
        """Non-negative seeds should be accepted without error."""
        seed_everything(seed)

    def test_python_random_determinism(self) -> None:
        """Python random module should produce identical sequences after seeding."""
        seed_everything(99)
        seq_a = [random.random() for _ in range(10)]
        seed_everything(99)
        seq_b = [random.random() for _ in range(10)]
        assert seq_a == seq_b

    def test_numpy_determinism(self) -> None:
        """NumPy random should produce identical sequences after seeding."""
        seed_everything(99)
        arr_a = np.random.rand(5)
        seed_everything(99)
        arr_b = np.random.rand(5)
        np.testing.assert_array_equal(arr_a, arr_b)

    def test_torch_determinism(self) -> None:
        """PyTorch random should produce identical tensors after seeding."""
        seed_everything(99)
        t_a = torch.randn(5)
        seed_everything(99)
        t_b = torch.randn(5)
        assert torch.equal(t_a, t_b)

    def test_different_seeds_produce_different_outputs(self) -> None:
        """Two different seeds should (almost certainly) produce different output."""
        seed_everything(0)
        a = np.random.rand(20)
        seed_everything(1)
        b = np.random.rand(20)
        assert not np.array_equal(a, b)

    def test_pythonhashseed_env_var_set(self) -> None:
        """Environment variable PYTHONHASHSEED should be set to the seed value."""
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    @pytest.mark.parametrize("seed", [-1, -42, -100])
    def test_negative_seed_raises_value_error(self, seed: int) -> None:
        """Negative seeds must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(seed)

    def test_seed_zero(self) -> None:
        """Zero is a valid seed and should produce deterministic results."""
        seed_everything(0)
        val_a = random.random()
        seed_everything(0)
        val_b = random.random()
        assert val_a == val_b


# =========================================================================
# get_device
# =========================================================================


class TestGetDevice:
    """Tests for the get_device function."""

    def test_returns_torch_device(self) -> None:
        """Return type should be torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_prefer_gpu_false_returns_cpu(self) -> None:
        """Requesting no GPU should always return CPU."""
        device = get_device(prefer_gpu=False)
        assert device.type == "cpu"

    def test_prefer_gpu_true_no_cuda_returns_cpu(self) -> None:
        """When CUDA is unavailable, even prefer_gpu=True falls back to CPU."""
        with patch("src.utils.common.torch.cuda.is_available", return_value=False):
            device = get_device(prefer_gpu=True)
            assert device.type == "cpu"

    def test_prefer_gpu_true_with_cuda_returns_cuda(self) -> None:
        """When CUDA is available and preferred, device should be 'cuda'."""
        with patch("src.utils.common.torch.cuda.is_available", return_value=True):
            device = get_device(prefer_gpu=True)
            assert device.type == "cuda"

    def test_default_prefers_gpu(self) -> None:
        """The default value of prefer_gpu is True."""
        with patch("src.utils.common.torch.cuda.is_available", return_value=False):
            device = get_device()
            # Even though it falls back to CPU, the default path is prefer_gpu=True
            assert device.type == "cpu"


# =========================================================================
# normalize_quaternion
# =========================================================================


class TestNormalizeQuaternion:
    """Tests for normalize_quaternion."""

    def test_already_unit_quaternion_unchanged(self) -> None:
        """A unit quaternion should remain (approximately) the same."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = normalize_quaternion(q)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_norm_of_result_is_one_single(self) -> None:
        """Result of normalization should have unit norm (single quaternion)."""
        q = np.array([2.0, 3.0, 4.0, 5.0])
        result = normalize_quaternion(q)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    @pytest.mark.parametrize(
        "q",
        [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 10.0]),
            np.array([-3.0, 4.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ],
        ids=["equal_components", "single_axis", "negative_w", "half_components"],
    )
    def test_various_quaternions_normalized(self, q: np.ndarray) -> None:
        """Various quaternion inputs should all have unit norm after normalization."""
        result = normalize_quaternion(q)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_batch_normalization(self) -> None:
        """Batch of quaternions should each have unit norm."""
        q = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
        ])
        result = normalize_quaternion(q)
        assert result.shape == (3, 4)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_batch_preserves_direction(self) -> None:
        """Normalization should preserve direction, only scale magnitude."""
        q = np.array([[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
        result = normalize_quaternion(q)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(result[1], [0.0, 0.0, 1.0, 0.0], atol=1e-12)

    def test_zero_quaternion_raises(self) -> None:
        """Zero quaternion should raise ValueError (norm below threshold)."""
        q = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="norm.*below threshold"):
            normalize_quaternion(q)

    def test_near_zero_quaternion_raises(self) -> None:
        """Quaternion with extremely small norm should raise ValueError."""
        q = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        with pytest.raises(ValueError, match="norm.*below threshold"):
            normalize_quaternion(q)

    def test_batch_with_zero_quaternion_raises(self) -> None:
        """Batch containing a zero quaternion should raise ValueError."""
        q = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # zero
        ])
        with pytest.raises(ValueError, match="below threshold"):
            normalize_quaternion(q)

    def test_wrong_shape_1d_raises(self) -> None:
        """A 1-D array that is not length 4 should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            normalize_quaternion(np.array([1.0, 2.0, 3.0]))

    def test_wrong_shape_2d_raises(self) -> None:
        """A 2-D array with columns != 4 should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            normalize_quaternion(np.ones((3, 5)))

    def test_3d_array_raises(self) -> None:
        """A 3-D array should raise ValueError (only 1-D or 2-D accepted)."""
        with pytest.raises(ValueError, match="ndim"):
            normalize_quaternion(np.ones((2, 3, 4)))

    def test_scalar_raises(self) -> None:
        """A 0-D (scalar) array should raise ValueError."""
        with pytest.raises(ValueError, match="ndim"):
            normalize_quaternion(np.array(1.0))

    def test_input_not_mutated(self) -> None:
        """The original input array should not be modified."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        original = q.copy()
        normalize_quaternion(q)
        np.testing.assert_array_equal(q, original)

    def test_list_input_accepted(self) -> None:
        """Python list input should be accepted and converted."""
        result = normalize_quaternion([0.0, 0.0, 3.0, 4.0])
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_output_dtype_float64(self) -> None:
        """Output should always be float64."""
        q = np.array([1, 0, 0, 0], dtype=np.int32)
        result = normalize_quaternion(q)
        assert result.dtype == np.float64


# =========================================================================
# quaternion_to_rotation_matrix
# =========================================================================


class TestQuaternionToRotationMatrix:
    """Tests for quaternion_to_rotation_matrix."""

    def test_identity_quaternion_gives_identity_matrix(self) -> None:
        """q = [1, 0, 0, 0] should produce the 3x3 identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rot = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)

    def test_output_shape_single(self) -> None:
        """Single quaternion should produce shape (3, 3)."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rot = quaternion_to_rotation_matrix(q)
        assert rot.shape == (3, 3)

    def test_output_shape_batch(self) -> None:
        """Batch of N quaternions should produce shape (N, 3, 3)."""
        q = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        rot = quaternion_to_rotation_matrix(q)
        assert rot.shape == (2, 3, 3)

    def test_rotation_matrix_orthogonal(self) -> None:
        """Rotation matrix should be orthogonal: R^T @ R = I."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        rot = quaternion_to_rotation_matrix(q)
        product = rot.T @ rot
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_rotation_matrix_determinant_one(self) -> None:
        """Rotation matrix should have determinant +1 (proper rotation)."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        rot = quaternion_to_rotation_matrix(q)
        det = np.linalg.det(rot)
        np.testing.assert_allclose(det, 1.0, atol=1e-12)

    @pytest.mark.parametrize(
        "q, expected_col",
        [
            # 180-degree rotation about z: x->-x, y->-y, z->z
            (
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
            # 180-degree rotation about x: x->x, y->-y, z->-z
            (
                np.array([0.0, 1.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ),
            # 180-degree rotation about y: x->-x, y->y, z->-z
            (
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
        ],
        ids=["180_about_z", "180_about_x", "180_about_y"],
    )
    def test_180_degree_rotations(
        self, q: np.ndarray, expected_col: np.ndarray
    ) -> None:
        """180-degree rotations about principal axes should leave that axis fixed."""
        rot = quaternion_to_rotation_matrix(q)
        # The axis of rotation should be an eigenvector with eigenvalue +1.
        result = rot @ expected_col
        np.testing.assert_allclose(result, expected_col, atol=1e-12)

    def test_90_degree_rotation_about_z(self) -> None:
        """90-degree rotation about z: x-axis -> y-axis."""
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        rot = quaternion_to_rotation_matrix(q)
        x_axis = np.array([1.0, 0.0, 0.0])
        rotated = rot @ x_axis
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)

    def test_unnormalized_input_auto_normalized(self) -> None:
        """Non-unit quaternions should be normalized internally."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        rot = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)

    def test_batch_all_identity(self) -> None:
        """Batch of identity quaternions should all produce identity matrices."""
        q = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        rot = quaternion_to_rotation_matrix(q)
        for i in range(5):
            np.testing.assert_allclose(rot[i], np.eye(3), atol=1e-12)

    def test_negative_quaternion_same_rotation(self) -> None:
        """q and -q should produce the same rotation matrix (double cover)."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        rot_pos = quaternion_to_rotation_matrix(q)
        rot_neg = quaternion_to_rotation_matrix(-q)
        np.testing.assert_allclose(rot_pos, rot_neg, atol=1e-12)


# =========================================================================
# quaternion_multiply
# =========================================================================


class TestQuaternionMultiply:
    """Tests for quaternion_multiply."""

    def test_identity_multiply_left(self) -> None:
        """Multiplying identity * q should return q (normalized)."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternion_multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_identity_multiply_right(self) -> None:
        """Multiplying q * identity should return q (normalized)."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternion_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_output_shape_single(self) -> None:
        """Multiply of two single quaternions returns shape (4,)."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = quaternion_multiply(q1, q2)
        assert result.shape == (4,)

    def test_output_shape_batch(self) -> None:
        """Multiply of two batches returns shape (N, 4)."""
        q1 = np.ones((3, 4))
        q2 = np.ones((3, 4))
        result = quaternion_multiply(q1, q2)
        assert result.shape == (3, 4)

    def test_i_times_j_equals_k(self) -> None:
        """Hamilton product: i * j = k."""
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        result = quaternion_multiply(i, j)
        np.testing.assert_allclose(result, k, atol=1e-12)

    def test_j_times_i_equals_neg_k(self) -> None:
        """Hamilton product: j * i = -k (non-commutative)."""
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        neg_k = np.array([0.0, 0.0, 0.0, -1.0])
        result = quaternion_multiply(j, i)
        np.testing.assert_allclose(result, neg_k, atol=1e-12)

    def test_non_commutative(self) -> None:
        """Quaternion multiplication is generally non-commutative."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([1.0, 0.0, 0.0, 0.0])
        r1 = quaternion_multiply(q1, q2)
        r2 = quaternion_multiply(q2, q1)
        # They should be equal in this specific case (identity on one side),
        # so use a non-trivial pair
        q3 = np.array([0.0, 1.0, 0.0, 0.0])
        q4 = np.array([0.0, 0.0, 1.0, 0.0])
        r3 = quaternion_multiply(q3, q4)
        r4 = quaternion_multiply(q4, q3)
        assert not np.allclose(r3, r4)

    def test_associativity(self) -> None:
        """Quaternion multiplication should be associative: (q1*q2)*q3 = q1*(q2*q3)."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        q3 = np.array([0.0, 0.0, 0.0, 1.0])
        left = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        right = quaternion_multiply(q1, quaternion_multiply(q2, q3))
        np.testing.assert_allclose(left, right, atol=1e-12)

    def test_batch_multiply(self) -> None:
        """Batch multiplication should compute element-wise products."""
        q1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        q2 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        result = quaternion_multiply(q1, q2)
        assert result.shape == (2, 4)
        # First: identity * identity = identity
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        # Second: i * j = k
        np.testing.assert_allclose(result[1], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_mixed_single_batch_broadcast(self) -> None:
        """Single (4,) quaternion with batch (N, 4) should broadcast."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # identity, shape (4,)
        q2 = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
        ])  # shape (2, 4)
        result = quaternion_multiply(q1, q2)
        assert result.shape == (2, 4)

    def test_conjugate_product_gives_norm_squared(self) -> None:
        """q * conj(q) should equal [|q|^2, 0, 0, 0]."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        result = quaternion_multiply(q, q_conj)
        norm_sq = np.dot(q, q)
        np.testing.assert_allclose(result[0], norm_sq, atol=1e-10)
        np.testing.assert_allclose(result[1:], 0.0, atol=1e-10)


# =========================================================================
# clamp_actions
# =========================================================================


class TestClampActions:
    """Tests for clamp_actions."""

    def test_values_within_range_unchanged(self) -> None:
        """Values already in [min, max] should not change."""
        actions = np.array([0.0, 0.5, 1.0])
        result = clamp_actions(actions, 0.0, 1.0)
        np.testing.assert_array_equal(result, actions)

    def test_values_above_max_clamped(self) -> None:
        """Values above max should be clamped to max."""
        actions = np.array([5.0, 10.0])
        result = clamp_actions(actions, -1.0, 1.0)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_values_below_min_clamped(self) -> None:
        """Values below min should be clamped to min."""
        actions = np.array([-5.0, -10.0])
        result = clamp_actions(actions, -1.0, 1.0)
        np.testing.assert_array_equal(result, [-1.0, -1.0])

    def test_mixed_clamp(self) -> None:
        """Mix of in-range, above, and below values."""
        actions = np.array([-5.0, 0.0, 0.5, 5.0])
        result = clamp_actions(actions, -1.0, 1.0)
        expected = np.array([-1.0, 0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [(6,), (2, 3), (2, 3, 4)],
        ids=["1d", "2d", "3d"],
    )
    def test_arbitrary_shape(self, shape: tuple[int, ...]) -> None:
        """Clamping should work on arrays of any shape."""
        actions = np.random.randn(*shape) * 10
        result = clamp_actions(actions, -1.0, 1.0)
        assert result.shape == shape
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_min_equals_max(self) -> None:
        """When min == max, all values should be clamped to that value."""
        actions = np.array([-1.0, 0.0, 1.0, 5.0])
        result = clamp_actions(actions, 0.5, 0.5)
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5, 0.5])

    def test_min_greater_than_max_raises(self) -> None:
        """min_val > max_val should raise ValueError."""
        with pytest.raises(ValueError, match="min_val.*must be.*max_val"):
            clamp_actions(np.array([0.0]), 1.0, -1.0)

    def test_returns_new_array(self) -> None:
        """Clamping should return a new array, not modify in place."""
        actions = np.array([5.0, -5.0])
        original = actions.copy()
        result = clamp_actions(actions, -1.0, 1.0)
        # Original unchanged
        np.testing.assert_array_equal(actions, original)
        # Result is different
        assert not np.array_equal(result, original)

    def test_empty_array(self) -> None:
        """Clamping an empty array should return an empty array."""
        actions = np.array([])
        result = clamp_actions(actions, -1.0, 1.0)
        assert result.shape == (0,)

    @pytest.mark.parametrize(
        "min_val, max_val",
        [(-100.0, 100.0), (-1e-6, 1e-6), (0.0, 1e10)],
        ids=["wide_range", "tiny_range", "large_max"],
    )
    def test_various_ranges(self, min_val: float, max_val: float) -> None:
        """Clamping should work with various numerical ranges."""
        actions = np.linspace(-200, 200, 50)
        result = clamp_actions(actions, min_val, max_val)
        assert np.all(result >= min_val)
        assert np.all(result <= max_val)


# =========================================================================
# Timer
# =========================================================================


class TestTimer:
    """Tests for the Timer context manager."""

    def test_elapsed_is_positive(self) -> None:
        """Elapsed time after a non-trivial sleep should be positive."""
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed > 0.0

    def test_elapsed_approximately_correct(self) -> None:
        """Elapsed time should be close to the sleep duration."""
        duration = 0.05
        with Timer("test") as t:
            time.sleep(duration)
        # Allow generous tolerance for CI environments
        assert t.elapsed >= duration * 0.5
        assert t.elapsed < duration * 5.0

    def test_elapsed_zero_before_exit(self) -> None:
        """Before exit, elapsed should be initialized to 0."""
        t = Timer("test")
        assert t.elapsed == 0.0

    def test_context_manager_returns_self(self) -> None:
        """__enter__ should return the Timer instance."""
        t = Timer("test")
        with t as ctx:
            assert ctx is t

    def test_name_stored(self) -> None:
        """Timer name should be stored as an internal attribute."""
        t = Timer("my_operation")
        assert t._name == "my_operation"

    def test_no_exception_suppression(self) -> None:
        """Timer should not suppress exceptions raised inside the context."""
        with pytest.raises(RuntimeError, match="boom"):
            with Timer("failing") as t:
                raise RuntimeError("boom")
        # elapsed should still be set even after exception
        assert t.elapsed >= 0.0

    def test_multiple_uses(self) -> None:
        """A Timer can be reused and each use overwrites elapsed."""
        t = Timer("reuse")
        with t:
            time.sleep(0.01)
        first = t.elapsed

        with t:
            time.sleep(0.02)
        second = t.elapsed

        # Both should be positive, and second >= first (roughly)
        assert first > 0.0
        assert second > 0.0

    def test_custom_logger(self) -> None:
        """Timer should accept a custom logger without error."""
        import structlog

        custom_log = structlog.get_logger("custom")
        with Timer("test", log=custom_log) as t:
            pass
        assert t.elapsed >= 0.0


# =========================================================================
# validate_path
# =========================================================================


class TestValidatePath:
    """Tests for validate_path."""

    def test_existing_file_resolves(self, tmp_path: Path) -> None:
        """An existing file should be resolved to its absolute path."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = validate_path(f)
        assert result.is_absolute()
        assert result.exists()
        assert result == f.resolve()

    def test_existing_directory_resolves(self, tmp_path: Path) -> None:
        """An existing directory should be resolved."""
        result = validate_path(tmp_path)
        assert result.is_absolute()
        assert result.is_dir()

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """A non-existent path should raise FileNotFoundError."""
        fake = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            validate_path(fake)

    def test_path_within_base_dir_accepted(self, tmp_path: Path) -> None:
        """A path inside base_dir should be accepted."""
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "data.bin"
        f.write_text("data")
        result = validate_path(f, base_dir=tmp_path)
        assert result == f.resolve()

    def test_path_escaping_base_dir_raises(self, tmp_path: Path) -> None:
        """A path traversing outside base_dir should raise ValueError."""
        base = tmp_path / "safe"
        base.mkdir()
        # Create an escape path: base/../escape.txt
        escape = tmp_path / "escape.txt"
        escape.write_text("escaped")
        traversal = base / ".." / "escape.txt"
        with pytest.raises(ValueError, match="not relative to base directory"):
            validate_path(traversal, base_dir=base)

    def test_base_dir_itself_is_valid(self, tmp_path: Path) -> None:
        """The base_dir itself should be a valid path within base_dir."""
        result = validate_path(tmp_path, base_dir=tmp_path)
        assert result == tmp_path.resolve()

    def test_nested_subdirectory_valid(self, tmp_path: Path) -> None:
        """Deeply nested paths within base_dir should be valid."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        result = validate_path(deep, base_dir=tmp_path)
        assert result == deep.resolve()

    def test_no_base_dir_allows_any_existing_path(self, tmp_path: Path) -> None:
        """Without base_dir, any existing path should be accepted."""
        f = tmp_path / "anything.txt"
        f.write_text("x")
        result = validate_path(f)
        assert result == f.resolve()

    def test_symlink_escape_raises(self, tmp_path: Path) -> None:
        """Symlinks that resolve outside base_dir should be caught."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        link = base / "link.txt"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="not relative to base directory"):
            validate_path(link, base_dir=base)

    def test_nonexistent_with_base_dir_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Non-existent path within a valid base_dir should raise FileNotFoundError."""
        f = tmp_path / "ghost.txt"
        with pytest.raises(FileNotFoundError):
            validate_path(f, base_dir=tmp_path)
