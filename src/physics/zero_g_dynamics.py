"""Zero-gravity rigid-body dynamics for 6-DOF spacecraft control.

Implements Newton-Euler equations in a zero-gravity environment:
  - Linear: F = m * a  (no gravity term)
  - Angular: tau = I * alpha + omega x (I * omega)

Quaternions are used for orientation and are re-normalised after every
integration step to prevent numerical drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

from src.utils.common import normalize_quaternion, quaternion_multiply

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Named constants for physics thresholds
# ---------------------------------------------------------------------------

_LINEAR_MOMENTUM_ABS_FLOOR: float = 1e-10
_ANGULAR_MOMENTUM_ABS_FLOOR: float = 1e-8
_ANGULAR_IMPULSE_NORM_EPS: float = 1e-10
_QUATERNION_INTEGRATION_FACTOR: float = 0.5


@dataclass
class RigidBodyState:
    """Full state of a 6-DOF rigid body in zero-gravity.

    Convention:
        - Quaternion is ``[w, x, y, z]`` (Hamilton scalar-first).
        - Linear quantities are in world frame.
        - Angular velocity is in body frame.

    Attributes:
        position: World-frame position [x, y, z] in metres.
        velocity: World-frame linear velocity [vx, vy, vz] in m/s.
        orientation: Unit quaternion [w, x, y, z].
        angular_velocity: Body-frame angular velocity [ωx, ωy, ωz] in rad/s.
        mass: Scalar mass in kg.
        inertia: Principal moments of inertia [Ixx, Iyy, Izz] in kg·m².
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    mass: float = 100.0
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 10.0], dtype=np.float64)
    )

    def linear_momentum(self) -> np.ndarray:
        """Compute linear momentum ``p = m * v``."""
        return self.mass * self.velocity

    def angular_momentum(self) -> np.ndarray:
        """Compute angular momentum ``L = I * ω`` (body frame)."""
        return self.inertia * self.angular_velocity  # type: ignore[no-any-return]

    def kinetic_energy(self) -> float:
        """Compute total kinetic energy (translational + rotational)."""
        ke_trans = 0.5 * self.mass * float(np.dot(self.velocity, self.velocity))
        ke_rot = 0.5 * float(np.dot(self.inertia * self.angular_velocity, self.angular_velocity))
        return ke_trans + ke_rot

    def clone(self) -> RigidBodyState:
        """Return a deep copy of the state."""
        return RigidBodyState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass,
            inertia=self.inertia.copy(),
        )


class ZeroGDynamics:
    """Symplectic integrator for zero-gravity rigid-body dynamics.

    Uses semi-implicit Euler integration (symplectic Euler) which preserves
    energy better than explicit Euler over long horizons.

    Args:
        momentum_tolerance: Maximum allowed momentum conservation error
            as a fraction of the applied impulse magnitude.  Used for
            runtime validation when *validate* is ``True``.
    """

    def __init__(self, momentum_tolerance: float = 1e-3) -> None:
        self._momentum_tolerance = momentum_tolerance

    def step(
        self,
        state: RigidBodyState,
        force: np.ndarray,
        torque: np.ndarray,
        dt: float,
        validate: bool = True,
    ) -> RigidBodyState:
        """Advance the rigid-body state by one timestep.

        Args:
            state: Current rigid-body state (not mutated).
            force: World-frame force vector [Fx, Fy, Fz] in N.
            torque: Body-frame torque vector [τx, τy, τz] in N·m.
            dt: Timestep in seconds.
            validate: When ``True``, check momentum conservation after
                integration.

        Returns:
            New ``RigidBodyState`` after integration.

        Raises:
            PhysicsViolationError: If momentum conservation check fails.
        """
        force = np.asarray(force, dtype=np.float64)
        torque = np.asarray(torque, dtype=np.float64)

        prev_linear_momentum = state.linear_momentum()
        prev_angular_momentum = state.angular_momentum()

        # --- Linear dynamics (world frame) ---
        acceleration = force / state.mass
        new_velocity = state.velocity + acceleration * dt
        new_position = state.position + new_velocity * dt  # symplectic: use new vel

        # --- Angular dynamics (body frame) ---
        # Euler's rotation equation: I * alpha = tau - omega x (I * omega)
        omega = state.angular_velocity
        inertia_omega = state.inertia * omega
        gyroscopic = np.cross(omega, inertia_omega)
        angular_acceleration = (torque - gyroscopic) / state.inertia
        new_angular_velocity = omega + angular_acceleration * dt

        # --- Orientation update via quaternion integration ---
        # dq/dt = 0.5 * q ⊗ [0, ω]
        omega_quat = np.array([0.0, *new_angular_velocity], dtype=np.float64)
        q_dot = _QUATERNION_INTEGRATION_FACTOR * quaternion_multiply(state.orientation, omega_quat)
        new_orientation = state.orientation + q_dot * dt
        new_orientation = normalize_quaternion(new_orientation)

        new_state = RigidBodyState(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            angular_velocity=new_angular_velocity,
            mass=state.mass,
            inertia=state.inertia.copy(),
        )

        # --- Conservation validation ---
        if validate:
            self._check_linear_momentum(
                prev_momentum=prev_linear_momentum,
                new_momentum=new_state.linear_momentum(),
                applied_impulse=force * dt,
            )
            self._check_angular_momentum(
                prev_momentum=prev_angular_momentum,
                new_momentum=new_state.angular_momentum(),
                applied_impulse=torque * dt,
                gyroscopic_impulse=gyroscopic * dt,
            )

        return new_state

    def _check_linear_momentum(
        self,
        prev_momentum: np.ndarray,
        new_momentum: np.ndarray,
        applied_impulse: np.ndarray,
    ) -> None:
        """Verify that Δp = impulse within tolerance."""
        delta_p = new_momentum - prev_momentum
        error = np.linalg.norm(delta_p - applied_impulse)
        impulse_mag = np.linalg.norm(applied_impulse)

        threshold = max(self._momentum_tolerance * impulse_mag, _LINEAR_MOMENTUM_ABS_FLOOR)
        if error > threshold:  # type: ignore[operator]
            logger.error(
                "linear_momentum_violation",
                error=float(error),
                threshold=float(threshold),  # type: ignore[arg-type]
                delta_p=delta_p.tolist(),
                impulse=applied_impulse.tolist(),
            )
            raise PhysicsViolationError(
                f"Linear momentum conservation violated: error={error:.6e}, "
                f"threshold={threshold:.6e}"
            )

    def _check_angular_momentum(
        self,
        prev_momentum: np.ndarray,
        new_momentum: np.ndarray,
        applied_impulse: np.ndarray,
        gyroscopic_impulse: np.ndarray,
    ) -> None:
        """Verify angular momentum conservation accounting for gyroscopic torque."""
        delta_momentum = new_momentum - prev_momentum
        expected = applied_impulse - gyroscopic_impulse
        error = np.linalg.norm(delta_momentum - expected)
        impulse_mag = np.linalg.norm(expected) + _ANGULAR_IMPULSE_NORM_EPS

        threshold = max(self._momentum_tolerance * impulse_mag, _ANGULAR_MOMENTUM_ABS_FLOOR)
        if error > threshold:  # type: ignore[operator]
            logger.error(
                "angular_momentum_violation",
                error=float(error),
                threshold=float(threshold),  # type: ignore[arg-type]
            )
            raise PhysicsViolationError(
                f"Angular momentum conservation violated: error={error:.6e}, "
                f"threshold={threshold:.6e}"
            )


class PhysicsViolationError(RuntimeError):
    """Raised when a physics invariant check fails."""
