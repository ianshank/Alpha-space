"""Unit tests for zero-gravity rigid-body dynamics.

Covers RigidBodyState dataclass methods, ZeroGDynamics integration,
quaternion normalisation invariants, conservation-law validation,
and PhysicsViolationError handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.zero_g_dynamics import (
    PhysicsViolationError,
    RigidBodyState,
    ZeroGDynamics,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

ZERO3 = np.zeros(3, dtype=np.float64)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _make_state(
    position=None,
    velocity=None,
    orientation=None,
    angular_velocity=None,
    mass: float = 100.0,
    inertia=None,
) -> RigidBodyState:
    """Convenience factory for RigidBodyState with sensible defaults."""
    _f64 = np.float64

    def _vec3(val: object, default: np.ndarray) -> np.ndarray:
        return np.array(val, dtype=_f64) if val is not None else default

    return RigidBodyState(
        position=_vec3(position, np.zeros(3, dtype=_f64)),
        velocity=_vec3(velocity, np.zeros(3, dtype=_f64)),
        orientation=_vec3(orientation, IDENTITY_QUAT.copy()),
        angular_velocity=_vec3(angular_velocity, np.zeros(3, dtype=_f64)),
        mass=mass,
        inertia=_vec3(inertia, np.array([10.0, 10.0, 10.0], dtype=_f64)),
    )


# ================================================================== #
# RigidBodyState creation and clone
# ================================================================== #


class TestRigidBodyStateCreation:
    """Verify default construction and clone semantics."""

    def test_default_state_is_at_rest_at_origin(self):
        state = RigidBodyState()
        np.testing.assert_array_equal(state.position, ZERO3)
        np.testing.assert_array_equal(state.velocity, ZERO3)
        np.testing.assert_array_equal(state.orientation, IDENTITY_QUAT)
        np.testing.assert_array_equal(state.angular_velocity, ZERO3)
        assert state.mass == 100.0
        np.testing.assert_array_equal(state.inertia, [10.0, 10.0, 10.0])

    def test_custom_state_fields(self):
        state = _make_state(
            position=[1, 2, 3],
            velocity=[0.5, -0.5, 0],
            mass=50.0,
            inertia=[5, 10, 15],
        )
        np.testing.assert_array_equal(state.position, [1, 2, 3])
        np.testing.assert_array_equal(state.velocity, [0.5, -0.5, 0])
        assert state.mass == 50.0
        np.testing.assert_array_equal(state.inertia, [5, 10, 15])

    def test_clone_produces_equal_but_independent_copy(self):
        original = _make_state(position=[1, 2, 3], velocity=[4, 5, 6])
        cloned = original.clone()

        # Values should be equal
        np.testing.assert_array_equal(cloned.position, original.position)
        np.testing.assert_array_equal(cloned.velocity, original.velocity)
        np.testing.assert_array_equal(cloned.orientation, original.orientation)
        np.testing.assert_array_equal(cloned.angular_velocity, original.angular_velocity)
        assert cloned.mass == original.mass
        np.testing.assert_array_equal(cloned.inertia, original.inertia)

        # Mutating clone must not affect original
        cloned.position[0] = 999.0
        assert original.position[0] == 1.0

    def test_clone_preserves_mass(self):
        state = _make_state(mass=73.5)
        assert state.clone().mass == 73.5


# ================================================================== #
# linear_momentum, angular_momentum, kinetic_energy
# ================================================================== #


class TestRigidBodyStatePhysics:
    """Verify the computed physical quantities."""

    def test_linear_momentum_zero_velocity(self):
        state = _make_state(mass=50.0)
        np.testing.assert_array_equal(state.linear_momentum(), ZERO3)

    def test_linear_momentum_nonzero(self):
        state = _make_state(velocity=[1, 2, 3], mass=10.0)
        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(state.linear_momentum(), expected)

    def test_angular_momentum_zero_omega(self):
        state = _make_state(inertia=[5, 10, 15])
        np.testing.assert_array_equal(state.angular_momentum(), ZERO3)

    def test_angular_momentum_nonzero(self):
        state = _make_state(angular_velocity=[1, 2, 3], inertia=[5, 10, 15])
        expected = np.array([5.0, 20.0, 45.0])
        np.testing.assert_allclose(state.angular_momentum(), expected)

    def test_kinetic_energy_at_rest_is_zero(self):
        state = RigidBodyState()
        assert state.kinetic_energy() == 0.0

    def test_kinetic_energy_translational_only(self):
        state = _make_state(velocity=[3, 4, 0], mass=2.0)
        # KE = 0.5 * 2.0 * (9 + 16) = 25.0
        assert state.kinetic_energy() == pytest.approx(25.0)

    def test_kinetic_energy_rotational_only(self):
        state = _make_state(angular_velocity=[1, 0, 0], inertia=[6, 10, 10])
        # KE = 0.5 * 6 * 1 = 3.0
        assert state.kinetic_energy() == pytest.approx(3.0)

    def test_kinetic_energy_combined(self):
        state = _make_state(
            velocity=[1, 0, 0],
            angular_velocity=[0, 2, 0],
            mass=4.0,
            inertia=[10, 5, 10],
        )
        # KE_trans = 0.5 * 4 * 1 = 2
        # KE_rot = 0.5 * 5 * 4 = 10
        assert state.kinetic_energy() == pytest.approx(12.0)

    @pytest.mark.parametrize("mass", [0.1, 1.0, 50.0, 500.0])
    def test_linear_momentum_scales_with_mass(self, mass: float):
        v = np.array([1.0, 0.0, 0.0])
        state = _make_state(velocity=v, mass=mass)
        np.testing.assert_allclose(state.linear_momentum(), mass * v)

    @pytest.mark.parametrize(
        "inertia",
        [[1, 1, 1], [5, 10, 15], [100, 200, 300]],
        ids=["uniform-1", "asymmetric", "heavy"],
    )
    def test_angular_momentum_scales_with_inertia(self, inertia):
        omega = np.array([1.0, 1.0, 1.0])
        state = _make_state(angular_velocity=omega, inertia=inertia)
        expected = np.array(inertia, dtype=np.float64) * omega
        np.testing.assert_allclose(state.angular_momentum(), expected)


# ================================================================== #
# ZeroGDynamics.step -- zero force (free drift)
# ================================================================== #


class TestZeroGDynamicsFreeDrift:
    """With zero force and zero torque the body should drift at constant velocity."""

    @pytest.fixture()
    def dynamics(self) -> ZeroGDynamics:
        return ZeroGDynamics()

    def test_position_updates_with_constant_velocity(self, dynamics: ZeroGDynamics):
        state = _make_state(position=[0, 0, 0], velocity=[1, 0, 0])
        dt = 0.1
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=dt)
        # Symplectic Euler: new_pos = old_pos + new_vel * dt
        # new_vel = old_vel (no force) = [1,0,0]
        np.testing.assert_allclose(new.position, [0.1, 0, 0], atol=1e-12)

    def test_velocity_unchanged_under_zero_force(self, dynamics: ZeroGDynamics):
        state = _make_state(velocity=[3.0, -2.0, 1.0])
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.05)
        np.testing.assert_allclose(new.velocity, state.velocity, atol=1e-12)

    def test_multi_step_drift_is_linear(self, dynamics: ZeroGDynamics):
        state = _make_state(velocity=[2, 0, 0])
        dt = 0.01
        for _ in range(100):
            state = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=dt)
        # After 100 steps of dt=0.01 at v=2 m/s: total displacement ~2.0 m
        np.testing.assert_allclose(state.position[0], 2.0, atol=1e-10)

    @pytest.mark.parametrize("mass", [1.0, 10.0, 100.0, 1000.0])
    def test_free_drift_independent_of_mass(self, dynamics: ZeroGDynamics, mass: float):
        state = _make_state(velocity=[1, 0, 0], mass=mass)
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.1)
        np.testing.assert_allclose(new.velocity, [1, 0, 0], atol=1e-12)


# ================================================================== #
# ZeroGDynamics.step -- nonzero force (correct acceleration)
# ================================================================== #


class TestZeroGDynamicsWithForce:
    """Verify F=ma integration with applied forces."""

    @pytest.fixture()
    def dynamics(self) -> ZeroGDynamics:
        return ZeroGDynamics()

    def test_constant_force_produces_correct_acceleration(self, dynamics: ZeroGDynamics):
        mass = 10.0
        state = _make_state(mass=mass)
        force = np.array([10.0, 0.0, 0.0])
        dt = 0.1
        new = dynamics.step(state, force=force, torque=ZERO3, dt=dt)
        # a = F/m = 1 m/s^2, dv = a*dt = 0.1
        np.testing.assert_allclose(new.velocity, [0.1, 0, 0], atol=1e-12)
        # pos = old_pos + new_vel * dt = 0 + 0.1 * 0.1 = 0.01 (symplectic)
        np.testing.assert_allclose(new.position, [0.01, 0, 0], atol=1e-12)

    @pytest.mark.parametrize("mass", [1.0, 25.0, 200.0])
    def test_acceleration_inversely_proportional_to_mass(
        self, dynamics: ZeroGDynamics, mass: float
    ):
        force = np.array([0, 0, 50.0])
        state = _make_state(mass=mass)
        dt = 0.1
        new = dynamics.step(state, force=force, torque=ZERO3, dt=dt)
        expected_dv = 50.0 / mass * dt
        np.testing.assert_allclose(new.velocity[2], expected_dv, atol=1e-12)

    def test_torque_changes_angular_velocity(self, dynamics: ZeroGDynamics):
        state = _make_state(inertia=[10, 10, 10])
        torque = np.array([0, 0, 5.0])
        dt = 0.1
        new = dynamics.step(state, force=ZERO3, torque=torque, dt=dt)
        # alpha = tau / I = 0.5 rad/s^2, d_omega = 0.5 * 0.1 = 0.05
        np.testing.assert_allclose(new.angular_velocity[2], 0.05, atol=1e-12)

    def test_three_axis_force_and_torque(self, dynamics: ZeroGDynamics):
        mass = 20.0
        inertia = [5.0, 10.0, 20.0]
        state = _make_state(mass=mass, inertia=inertia)
        force = np.array([20.0, -10.0, 40.0])
        torque = np.array([5.0, 10.0, -20.0])
        dt = 0.05

        new = dynamics.step(state, force=force, torque=torque, dt=dt)

        # Linear: a = F/m
        expected_vel = force / mass * dt
        np.testing.assert_allclose(new.velocity, expected_vel, atol=1e-12)

        # Angular: alpha = tau / I (omega starts at 0, so gyroscopic term is 0)
        expected_omega = torque / np.array(inertia) * dt
        np.testing.assert_allclose(new.angular_velocity, expected_omega, atol=1e-12)

    @pytest.mark.parametrize(
        "inertia",
        [[1, 1, 1], [10, 20, 30], [100, 100, 100]],
        ids=["uniform-small", "asymmetric", "uniform-large"],
    )
    def test_angular_acceleration_scales_with_inertia(self, dynamics: ZeroGDynamics, inertia):
        torque = np.array([10.0, 0, 0])
        state = _make_state(inertia=inertia)
        dt = 0.1
        new = dynamics.step(state, force=ZERO3, torque=torque, dt=dt)
        expected_domega = 10.0 / inertia[0] * dt
        np.testing.assert_allclose(new.angular_velocity[0], expected_domega, atol=1e-12)


# ================================================================== #
# Quaternion normalisation after step
# ================================================================== #


class TestQuaternionNormalisation:
    """The orientation quaternion must stay unit-length after every step."""

    @pytest.fixture()
    def dynamics(self) -> ZeroGDynamics:
        return ZeroGDynamics()

    def test_quaternion_unit_after_single_step(self, dynamics: ZeroGDynamics):
        state = _make_state(angular_velocity=[0.5, -0.3, 0.1])
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.1)
        quat_norm = np.linalg.norm(new.orientation)
        assert quat_norm == pytest.approx(1.0, abs=1e-12)

    def test_quaternion_unit_after_many_steps(self, dynamics: ZeroGDynamics):
        state = _make_state(angular_velocity=[1.0, 0.5, -0.2])
        for _ in range(500):
            state = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.01)
        quat_norm = np.linalg.norm(state.orientation)
        assert quat_norm == pytest.approx(1.0, abs=1e-10)

    def test_quaternion_unit_with_large_torque(self, dynamics: ZeroGDynamics):
        state = _make_state()
        torque = np.array([50.0, -30.0, 20.0])
        for _ in range(100):
            state = dynamics.step(state, force=ZERO3, torque=torque, dt=0.01, validate=False)
        quat_norm = np.linalg.norm(state.orientation)
        assert quat_norm == pytest.approx(1.0, abs=1e-10)

    def test_identity_quaternion_unchanged_without_rotation(self, dynamics: ZeroGDynamics):
        state = _make_state()
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.1)
        np.testing.assert_allclose(new.orientation, IDENTITY_QUAT, atol=1e-12)


# ================================================================== #
# PhysicsViolationError
# ================================================================== #


class TestPhysicsViolationError:
    """Verify that invariant checks raise PhysicsViolationError appropriately."""

    def test_physics_violation_error_is_runtime_error(self):
        assert issubclass(PhysicsViolationError, RuntimeError)

    def test_physics_violation_error_has_message(self):
        err = PhysicsViolationError("bad physics")
        assert "bad physics" in str(err)

    def test_no_violation_on_valid_step(self):
        dynamics = ZeroGDynamics()
        state = _make_state()
        # This should NOT raise
        dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.1, validate=True)

    def test_validate_false_skips_check(self):
        """When validate=False, the momentum check should not run,
        so even a manipulated scenario won't raise."""
        dynamics = ZeroGDynamics(momentum_tolerance=1e-15)
        state = _make_state(velocity=[1, 0, 0])
        # Valid step; just confirming validate=False doesn't raise
        dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.1, validate=False)

    def test_manual_linear_momentum_violation(self):
        """Directly invoke the internal check to verify it can raise."""
        dynamics = ZeroGDynamics(momentum_tolerance=1e-10)
        prev_p = np.array([10.0, 0.0, 0.0])
        # new_p deviates from prev_p + impulse
        new_p = np.array([10.0, 0.0, 0.0])
        impulse = np.array([5.0, 0.0, 0.0])  # expected delta = 5, but actual = 0
        with pytest.raises(PhysicsViolationError, match="Linear momentum conservation"):
            dynamics._check_linear_momentum(prev_p, new_p, impulse)

    def test_manual_angular_momentum_violation(self):
        """Directly invoke the internal angular check to verify it can raise."""
        dynamics = ZeroGDynamics(momentum_tolerance=1e-10)
        prev_angular = np.array([0.0, 0.0, 0.0])
        new_angular = np.array([0.0, 0.0, 0.0])
        impulse = np.array([10.0, 0.0, 0.0])
        gyro = np.zeros(3)
        with pytest.raises(PhysicsViolationError, match="Angular momentum conservation"):
            dynamics._check_angular_momentum(prev_angular, new_angular, impulse, gyro)


# ================================================================== #
# Parametrised mass / inertia combinations
# ================================================================== #


class TestParametrisedMassInertia:
    """Run dynamics with various mass and inertia combinations."""

    @pytest.fixture()
    def dynamics(self) -> ZeroGDynamics:
        return ZeroGDynamics()

    @pytest.mark.parametrize(
        "mass,inertia",
        [
            (0.5, [0.1, 0.1, 0.1]),
            (1.0, [1.0, 1.0, 1.0]),
            (10.0, [5.0, 10.0, 15.0]),
            (100.0, [50.0, 50.0, 50.0]),
            (500.0, [100.0, 200.0, 300.0]),
        ],
        ids=["tiny", "unit", "asymmetric", "default-like", "heavy"],
    )
    def test_step_succeeds_and_conserves_energy_no_force(
        self, dynamics: ZeroGDynamics, mass: float, inertia: list[float]
    ):
        """Under zero external force, kinetic energy should be conserved."""
        state = _make_state(
            velocity=[1.0, 0.5, -0.3],
            angular_velocity=[0.1, -0.05, 0.02],
            mass=mass,
            inertia=inertia,
        )
        ke_before = state.kinetic_energy()
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=0.01)
        ke_after = new.kinetic_energy()
        # Symplectic Euler preserves energy well for small dt
        np.testing.assert_allclose(ke_after, ke_before, rtol=1e-4)

    @pytest.mark.parametrize(
        "mass,inertia",
        [
            (1.0, [1.0, 1.0, 1.0]),
            (50.0, [10.0, 20.0, 30.0]),
            (200.0, [100.0, 100.0, 100.0]),
        ],
        ids=["light", "medium", "heavy"],
    )
    def test_momentum_conservation_under_force(
        self, dynamics: ZeroGDynamics, mass: float, inertia: list[float]
    ):
        """Linear momentum change should equal the applied impulse."""
        state = _make_state(velocity=[0.5, 0, 0], mass=mass, inertia=inertia)
        force = np.array([5.0, -3.0, 1.0])
        dt = 0.05
        p_before = state.linear_momentum()
        new = dynamics.step(state, force=force, torque=ZERO3, dt=dt)
        p_after = new.linear_momentum()
        np.testing.assert_allclose(p_after - p_before, force * dt, atol=1e-10)

    @pytest.mark.parametrize("mass", [0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_state_is_not_mutated(self, dynamics: ZeroGDynamics, mass: float):
        """The step method should not mutate the input state."""
        state = _make_state(velocity=[1, 2, 3], mass=mass)
        original_pos = state.position.copy()
        original_vel = state.velocity.copy()
        dynamics.step(state, force=np.array([1, 0, 0]), torque=ZERO3, dt=0.1)
        np.testing.assert_array_equal(state.position, original_pos)
        np.testing.assert_array_equal(state.velocity, original_vel)


# ================================================================== #
# Gyroscopic effects
# ================================================================== #


class TestGyroscopicEffects:
    """Tests for gyroscopic coupling in angular dynamics."""

    def test_asymmetric_inertia_causes_precession(self) -> None:
        """Asymmetric inertia with angular velocity should show gyroscopic coupling."""
        dynamics = ZeroGDynamics()

        # Setup state with asymmetric inertia and angular velocity on MULTIPLE axes.
        # A single-axis spin (e.g. [0,0,2]) produces omega x (I*omega) = 0
        # because both vectors are parallel. We need off-axis components.
        inertia = [10.0, 20.0, 30.0]
        angular_velocity = [1.0, 0.0, 2.0]  # Multi-axis spin

        state = _make_state(
            inertia=inertia,
            angular_velocity=angular_velocity,
        )

        # Step with zero torque - gyroscopic effects should still occur
        # cross([1,0,2], [10,0,60]) = [0, -40, 0] -> non-zero coupling on y
        dt = 0.01
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=dt, validate=False)

        # Euler's equations: omega_dot = I^-1 * (tau - omega x (I * omega))
        # With zero torque, the gyroscopic term drives angular velocity changes.
        # Specifically the y-component should develop from the cross product.
        assert abs(new.angular_velocity[1]) > 1e-6, (
            "Asymmetric inertia with multi-axis rotation should show gyroscopic coupling"
        )

    def test_symmetric_inertia_no_coupling(self) -> None:
        """Symmetric inertia should show no gyroscopic coupling."""
        dynamics = ZeroGDynamics()

        # Setup state with symmetric inertia (sphere)
        inertia = [10.0, 10.0, 10.0]
        angular_velocity = [0.0, 0.0, 1.0]  # Spin around z-axis

        state = _make_state(
            inertia=inertia,
            angular_velocity=angular_velocity,
        )

        # Step with zero torque
        dt = 0.1
        new = dynamics.step(state, force=ZERO3, torque=ZERO3, dt=dt, validate=False)

        # With symmetric inertia, there should be no gyroscopic coupling
        # Angular velocity should remain on the same axis
        # omega x (I * omega) = 0 when I is isotropic

        # Check that x and y components remain essentially zero
        np.testing.assert_allclose(new.angular_velocity[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(new.angular_velocity[1], 0.0, atol=1e-12)

        # z component should be unchanged (no torque)
        np.testing.assert_allclose(new.angular_velocity[2], state.angular_velocity[2], atol=1e-12)
