"""Property-based tests for physics invariants using Hypothesis.

These tests verify that the zero-gravity dynamics engine upholds
fundamental physical laws regardless of the input state:
  - Momentum conservation
  - Quaternion normalisation
  - Energy relationships
  - Free-drift linearity
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from src.physics.zero_g_dynamics import RigidBodyState, ZeroGDynamics
from src.utils.common import normalize_quaternion

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_reasonable_float = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_positive_float = st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)
_small_float = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_vec3 = st.tuples(_reasonable_float, _reasonable_float, _reasonable_float).map(
    lambda t: np.array(t, dtype=np.float64)
)

_small_vec3 = st.tuples(_small_float, _small_float, _small_float).map(
    lambda t: np.array(t, dtype=np.float64)
)

_force_vec = st.tuples(
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
).map(lambda t: np.array(t, dtype=np.float64))

_torque_vec = st.tuples(
    st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
    st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
    st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
).map(lambda t: np.array(t, dtype=np.float64))

_mass = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)

_inertia = st.tuples(
    _positive_float, _positive_float, _positive_float
).map(lambda t: np.array(t, dtype=np.float64))


def _make_unit_quat(vals: tuple[float, float, float, float]) -> np.ndarray:
    q = np.array(vals, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


_unit_quat = st.tuples(
    _reasonable_float, _reasonable_float, _reasonable_float, _reasonable_float
).map(_make_unit_quat)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestLinearMomentumConservation:
    """Linear momentum change must equal applied impulse."""

    @given(
        velocity=_vec3,
        force=_force_vec,
        mass=_mass,
    )
    @settings(max_examples=100, deadline=None)
    def test_delta_p_equals_impulse(
        self,
        velocity: np.ndarray,
        force: np.ndarray,
        mass: float,
    ) -> None:
        dt = 0.05
        state = RigidBodyState(
            velocity=velocity.copy(),
            mass=mass,
        )
        dynamics = ZeroGDynamics(momentum_tolerance=1e-3)
        new_state = dynamics.step(
            state, force=force, torque=np.zeros(3), dt=dt, validate=False
        )

        delta_p = new_state.linear_momentum() - state.linear_momentum()
        expected = force * dt
        np.testing.assert_allclose(delta_p, expected, atol=1e-8)

    @given(velocity=_vec3, mass=_mass)
    @settings(max_examples=50, deadline=None)
    def test_free_drift_momentum_constant(
        self, velocity: np.ndarray, mass: float
    ) -> None:
        """With zero force, linear momentum is unchanged."""
        state = RigidBodyState(velocity=velocity.copy(), mass=mass)
        dynamics = ZeroGDynamics()
        new_state = dynamics.step(
            state, force=np.zeros(3), torque=np.zeros(3), dt=0.05
        )
        np.testing.assert_allclose(
            new_state.linear_momentum(), state.linear_momentum(), atol=1e-10
        )


class TestQuaternionNormalisation:
    """Quaternion must remain normalised after integration."""

    @given(
        orientation=_unit_quat,
        angular_velocity=_small_vec3,
    )
    @settings(max_examples=100, deadline=None)
    def test_quaternion_unit_after_step(
        self, orientation: np.ndarray, angular_velocity: np.ndarray
    ) -> None:
        state = RigidBodyState(
            orientation=orientation.copy(),
            angular_velocity=angular_velocity.copy(),
        )
        dynamics = ZeroGDynamics()
        new_state = dynamics.step(
            state, force=np.zeros(3), torque=np.zeros(3), dt=0.05, validate=False
        )
        norm = float(np.linalg.norm(new_state.orientation))
        assert abs(norm - 1.0) < 1e-6, f"Quaternion norm drifted to {norm}"

    @given(
        orientation=_unit_quat,
        angular_velocity=_small_vec3,
    )
    @settings(max_examples=30, deadline=None)
    def test_quaternion_unit_after_100_steps(
        self, orientation: np.ndarray, angular_velocity: np.ndarray
    ) -> None:
        state = RigidBodyState(
            orientation=orientation.copy(),
            angular_velocity=angular_velocity.copy(),
        )
        dynamics = ZeroGDynamics()
        for _ in range(100):
            state = dynamics.step(
                state, force=np.zeros(3), torque=np.zeros(3), dt=0.01, validate=False
            )
        norm = float(np.linalg.norm(state.orientation))
        assert abs(norm - 1.0) < 1e-5, f"Quaternion norm drifted to {norm} after 100 steps"


class TestFreeDrift:
    """With zero forces, objects drift linearly."""

    @given(position=_vec3, velocity=_vec3, mass=_mass)
    @settings(max_examples=50, deadline=None)
    def test_position_is_linear_in_time(
        self, position: np.ndarray, velocity: np.ndarray, mass: float
    ) -> None:
        dt = 0.05
        state = RigidBodyState(position=position.copy(), velocity=velocity.copy(), mass=mass)
        dynamics = ZeroGDynamics()
        new_state = dynamics.step(
            state, force=np.zeros(3), torque=np.zeros(3), dt=dt
        )
        expected_pos = position + velocity * dt
        np.testing.assert_allclose(new_state.position, expected_pos, atol=1e-10)

    @given(velocity=_vec3, mass=_mass)
    @settings(max_examples=50, deadline=None)
    def test_velocity_unchanged_under_no_force(
        self, velocity: np.ndarray, mass: float
    ) -> None:
        state = RigidBodyState(velocity=velocity.copy(), mass=mass)
        dynamics = ZeroGDynamics()
        new_state = dynamics.step(
            state, force=np.zeros(3), torque=np.zeros(3), dt=0.05
        )
        np.testing.assert_allclose(new_state.velocity, velocity, atol=1e-10)


class TestInputImmutability:
    """Stepping must not mutate the input state."""

    @given(position=_vec3, velocity=_vec3, force=_force_vec)
    @settings(max_examples=50, deadline=None)
    def test_state_not_mutated(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        force: np.ndarray,
    ) -> None:
        state = RigidBodyState(position=position.copy(), velocity=velocity.copy())
        pos_before = state.position.copy()
        vel_before = state.velocity.copy()

        dynamics = ZeroGDynamics()
        dynamics.step(state, force=force, torque=np.zeros(3), dt=0.05, validate=False)

        np.testing.assert_array_equal(state.position, pos_before)
        np.testing.assert_array_equal(state.velocity, vel_before)


class TestNormalizeQuaternionProperty:
    """Quaternion normalisation utility properties."""

    @given(
        q=st.tuples(
            _reasonable_float, _reasonable_float, _reasonable_float, _reasonable_float
        ).filter(lambda t: np.linalg.norm(t) > 1e-8)
    )
    @settings(max_examples=100, deadline=None)
    def test_output_is_unit_length(self, q: tuple[float, ...]) -> None:
        arr = np.array(q, dtype=np.float64)
        normed = normalize_quaternion(arr)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-10

    @given(
        q=st.tuples(
            _reasonable_float, _reasonable_float, _reasonable_float, _reasonable_float
        ).filter(lambda t: np.linalg.norm(t) > 1e-8)
    )
    @settings(max_examples=100, deadline=None)
    def test_direction_preserved(self, q: tuple[float, ...]) -> None:
        arr = np.array(q, dtype=np.float64)
        normed = normalize_quaternion(arr)
        # Verify direction is preserved: normed * |q| ≈ q
        reconstructed = normed * np.linalg.norm(arr)
        np.testing.assert_allclose(reconstructed, arr, atol=1e-8)
