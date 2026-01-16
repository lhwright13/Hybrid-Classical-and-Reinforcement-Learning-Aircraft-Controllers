"""Simplified 6-DOF (6 Degrees of Freedom) aircraft dynamics model.

This module provides a simplified physics simulation for fixed-wing aircraft.
It's designed for rapid prototyping and RL training, trading physical accuracy
for computational speed and training efficiency.

Features:
- 6-DOF rigid body dynamics (position, velocity, attitude, angular rates)
- Simplified aerodynamic model (lift, drag, side force)
- Thrust model
- Gravity
- Numerical integration (RK4)

Not modeled (for simplicity):
- Detailed aerodynamic coefficients
- Propeller effects
- Ground effect
- Wind/turbulence (can be added via sensor noise)
"""

import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass
from controllers.types import AircraftState, ControlSurfaces

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class AircraftParams:
    """Physical parameters for aircraft dynamics.

    These are simplified parameters for a small fixed-wing aircraft
    configured as a "trainer" - inherently stable and easy to fly.

    Stability Design:
    - Positive static margin (Cm_alpha < 0) - pitch stable
    - Weathercock stability (Cn_beta > 0) - yaw stable
    - Dihedral effect (Cl_beta < 0) - roll stable
    - Strong damping on all axes
    """
    # Mass properties - LIGHT for responsive handling
    mass: float = 8.0  # kg (lighter aircraft)
    inertia_xx: float = 0.4  # kg⋅m² (roll inertia - reduced for fast roll)
    inertia_yy: float = 0.6  # kg⋅m² (pitch inertia - reduced for fast pitch)
    inertia_zz: float = 0.7  # kg⋅m² (yaw inertia - reduced for fast yaw)

    # Aerodynamic reference
    wing_area: float = 0.5  # m²
    wing_span: float = 2.0  # m
    chord: float = 0.25  # m (mean aerodynamic chord)

    # Aerodynamic coefficients
    cl_0: float = 0.4  # Zero-alpha lift coefficient
    cl_alpha: float = 5.0  # Lift curve slope (1/rad)
    cd_0: float = 0.025  # Parasitic drag coefficient
    cd_alpha2: float = 0.04  # Induced drag coefficient

    # Control effectiveness - HIGH for crisp response
    cl_elevator: float = 0.6  # Lift change per elevator deflection (1/rad)
    cm_elevator: float = -2.0  # Pitch moment per elevator (1/rad)
    cy_rudder: float = 0.5  # Side force per rudder (1/rad)
    cn_rudder: float = -0.40  # Yaw moment per rudder (1/rad) - HIGH for fast yaw
    cl_aileron: float = 0.50  # Roll moment per aileron (1/rad) - HIGH for fast roll

    # === CRITICAL STABILITY DERIVATIVES ===
    # Longitudinal static stability (pitch)
    cm_alpha: float = -0.15  # Static margin (1/rad) - mild stability

    # Lateral-directional stability
    cn_beta: float = 0.04  # Weathercock stability (1/rad) - mild
    cl_beta: float = -0.02  # Dihedral effect (1/rad) - minimal

    # Damping coefficients - LOW for crisp response
    damping_roll: float = -0.8  # Roll damping
    damping_pitch: float = -2.0  # Pitch damping
    damping_yaw: float = -0.6  # Yaw damping

    # Thrust
    max_thrust: float = 50.0  # N (max thrust)

    # Environment
    air_density: float = 1.225  # kg/m³ (sea level)
    gravity: float = 9.81  # m/s²

    # =============================================================================
    # Numerical Stability and Safety Limits
    # =============================================================================

    # Minimum values for numerical stability (prevent divide-by-zero)
    min_airspeed_aero: float = 10.0  # m/s - minimum airspeed for aero calculations
    min_u_velocity: float = 0.1  # m/s - minimum forward velocity to prevent singularity

    # Control surface deflection limits (degrees)
    max_elevator_deflection: float = 30.0  # degrees
    max_aileron_deflection: float = 30.0  # degrees
    max_rudder_deflection: float = 30.0  # degrees

    # Propeller thrust model (linear decay with airspeed)
    # At V=0: thrust = max_thrust, at V=thrust_zero_velocity: thrust = 0
    thrust_zero_velocity: float = 50.0  # m/s - airspeed where thrust drops to zero

    # State variable safety limits
    max_velocity: float = 100.0  # m/s - maximum body frame velocity
    max_angular_rate: float = 360.0  # deg/s - maximum angular rate
    max_pitch_angle: float = 85.0  # deg - maximum pitch to prevent gimbal lock
    max_alpha: float = 30.0  # deg - maximum angle of attack (prevents unrealistic aero)

    # Derivative limits (prevent numerical explosion)
    max_acceleration: float = 50.0  # m/s² - maximum acceleration
    max_angular_acceleration: float = 1000.0  # rad/s² - maximum angular acceleration

    # Timestep limits
    max_timestep: float = 1.0  # seconds - maximum allowed timestep
    min_timestep: float = 1e-6  # seconds - minimum allowed timestep

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Mass properties validation
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        if self.inertia_xx <= 0 or self.inertia_yy <= 0 or self.inertia_zz <= 0:
            raise ValueError("All inertia values must be positive")

        # Aerodynamic properties validation
        if self.wing_area <= 0:
            raise ValueError(f"Wing area must be positive, got {self.wing_area}")
        if self.wing_span <= 0:
            raise ValueError(f"Wing span must be positive, got {self.wing_span}")
        if self.chord <= 0:
            raise ValueError(f"Chord must be positive, got {self.chord}")

        # Environment validation
        if not (0 < self.air_density < 10):
            raise ValueError(f"Invalid air density: {self.air_density} kg/m³")
        if not (0 < self.gravity < 20):
            raise ValueError(f"Invalid gravity: {self.gravity} m/s²")

        # Thrust validation
        if self.max_thrust < 0:
            raise ValueError(f"Max thrust cannot be negative, got {self.max_thrust}")

        logger.info(f"AircraftParams validated: mass={self.mass}kg, wing_area={self.wing_area}m²")


class Simplified6DOF:
    """Simplified 6-DOF aircraft dynamics simulator.

    State vector (13D):
    - position: [x, y, z] in NED frame (m)
    - velocity: [u, v, w] in body frame (m/s)
    - attitude: [roll, pitch, yaw] Euler angles (rad)
    - angular_rate: [p, q, r] in body frame (rad/s)

    Control inputs (4D):
    - elevator: [-1, 1] (normalized)
    - aileron: [-1, 1] (normalized)
    - rudder: [-1, 1] (normalized)
    - throttle: [0, 1] (normalized)
    """

    def __init__(self, params: Optional[AircraftParams] = None):
        """Initialize 6-DOF simulator.

        Args:
            params: Aircraft parameters. If None, uses default small aircraft.
        """
        self.params = params or AircraftParams()

        # State: [position(3), velocity(3), attitude(3), angular_rate(3)]
        self.state = np.zeros(12)

        # Control inputs
        self.controls = ControlSurfaces()

        # Time
        self.time = 0.0

        # Pre-computed radian conversions (params don't change during simulation)
        self._max_pitch_rad = np.radians(self.params.max_pitch_angle)
        self._max_rate_rad = np.radians(self.params.max_angular_rate)
        self._max_alpha_rad = np.radians(self.params.max_alpha)
        self._max_elevator_rad = np.radians(self.params.max_elevator_deflection)
        self._max_aileron_rad = np.radians(self.params.max_aileron_deflection)
        self._max_rudder_rad = np.radians(self.params.max_rudder_deflection)

    def reset(self, initial_state: Optional[AircraftState] = None) -> None:
        """Reset simulator to initial state.

        Args:
            initial_state: Initial state. If None, uses default (level flight at 100m).
        """
        self.time = 0.0

        if initial_state is None:
            # Default: level flight at 100m altitude, 20 m/s airspeed
            self.state = np.array([
                0.0, 0.0, -100.0,  # position (NED)
                20.0, 0.0, 0.0,    # velocity (body frame)
                0.0, 0.0, 0.0,     # attitude (roll, pitch, yaw)
                0.0, 0.0, 0.0      # angular rates
            ])
        else:
            self.state = np.concatenate([
                initial_state.position,
                initial_state.velocity,
                initial_state.attitude,
                initial_state.angular_rate
            ])
            self.time = initial_state.time

    def set_controls(self, controls: ControlSurfaces) -> None:
        """Set control surface deflections.

        Args:
            controls: Control surface commands (normalized -1 to 1, throttle 0 to 1)
        """
        # Clamp control inputs to valid ranges to prevent unrealistic deflections
        self.controls = ControlSurfaces(
            elevator=np.clip(controls.elevator, -1.0, 1.0),
            aileron=np.clip(controls.aileron, -1.0, 1.0),
            rudder=np.clip(controls.rudder, -1.0, 1.0),
            throttle=np.clip(controls.throttle, 0.0, 1.0)
        )

    def step(self, dt: float) -> AircraftState:
        """Advance simulation by dt seconds using RK4 integration.

        Args:
            dt: Time step (seconds), must be in range (min_timestep, max_timestep]

        Returns:
            Updated aircraft state

        Raises:
            ValueError: If dt is outside valid range
        """
        # Validate timestep
        if dt <= self.params.min_timestep or dt > self.params.max_timestep:
            raise ValueError(
                f"Invalid timestep dt={dt:.6f}s, must be in "
                f"({self.params.min_timestep}, {self.params.max_timestep}]"
            )

        # RK4 integration for better accuracy
        k1 = self._dynamics(self.state, self.controls)
        k2 = self._dynamics(self.state + 0.5 * dt * k1, self.controls)
        k3 = self._dynamics(self.state + 0.5 * dt * k2, self.controls)
        k4 = self._dynamics(self.state + dt * k3, self.controls)

        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.time += dt

        # Clamp state variables to safe ranges
        # Position (NED) - no strict limits but check for NaN
        self.state[0:3] = np.nan_to_num(self.state[0:3], nan=0.0, posinf=10000.0, neginf=-10000.0)

        # Velocity (body frame) - use configured limits
        max_vel = self.params.max_velocity
        self.state[3:6] = np.clip(self.state[3:6], -max_vel, max_vel)

        # Attitude (Euler angles) - wrap and clamp
        # Roll: wrap to [-π, π]
        self.state[6] = np.arctan2(np.sin(self.state[6]), np.cos(self.state[6]))
        # Pitch: clamp to safe range (avoid gimbal lock)
        self.state[7] = np.clip(self.state[7], -self._max_pitch_rad, self._max_pitch_rad)
        # Yaw: wrap to [-π, π]
        self.state[8] = np.arctan2(np.sin(self.state[8]), np.cos(self.state[8]))

        # Angular rates - use configured limits
        self.state[9:12] = np.clip(self.state[9:12], -self._max_rate_rad, self._max_rate_rad)

        # Ground collision detection
        altitude = -self.state[2]  # NED: down is positive, altitude is negative of down
        if altitude < 0:
            logger.error(
                f"Ground collision detected! Altitude: {altitude:.2f}m, "
                f"Time: {self.time:.2f}s - Resetting to ground level"
            )
            self.state[2] = 0.0  # Reset to ground level (down = 0)
            self.state[5] = max(0.0, self.state[5])  # Zero or reverse vertical velocity (w)

        # Final NaN check
        if not np.all(np.isfinite(self.state)):
            logger.error(
                f"Non-finite state detected at t={self.time:.2f}s, "
                f"resetting to safe values"
            )
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0)

        return self.get_state()

    def get_state(self) -> AircraftState:
        """Get current state as AircraftState object.

        Returns:
            Current aircraft state
        """
        position = self.state[0:3]
        velocity = self.state[3:6]
        attitude = self.state[6:9]
        angular_rate = self.state[9:12]

        # Compute airspeed (magnitude of velocity in body frame)
        airspeed = np.linalg.norm(velocity)

        # Altitude (positive up, NED has down positive)
        altitude = -position[2]

        # Compute heading and ground speed from ground velocity in NED frame
        # Transform body velocity to NED frame
        velocity_ned = self._body_to_ned(velocity, attitude)
        # Heading: arctan2(east, north) in NED convention
        # 0° = North, 90° = East, 180° = South, 270° = West
        heading = np.arctan2(velocity_ned[1], velocity_ned[0])
        # Ground speed: horizontal velocity magnitude (ignores vertical component)
        ground_speed = np.linalg.norm(velocity_ned[:2])

        return AircraftState(
            time=self.time,
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_rate=angular_rate,
            airspeed=airspeed,
            altitude=altitude,
            ground_speed=ground_speed,
            heading=heading
        )

    def _dynamics(self, state: np.ndarray, controls: ControlSurfaces) -> np.ndarray:
        """Compute state derivatives (6-DOF equations of motion).

        Args:
            state: Current state [pos(3), vel(3), att(3), rates(3)]
            controls: Control inputs

        Returns:
            State derivative d(state)/dt
        """
        # Unpack state
        position = state[0:3]
        velocity = state[3:6]  # Body frame: [u, v, w]
        attitude = state[6:9]  # Euler angles: [roll, pitch, yaw]
        angular_rate = state[9:12]  # Body rates: [p, q, r]

        u, v, w = velocity
        phi, theta, psi = attitude
        p, q, r = angular_rate

        # === Forces and Moments ===

        # Airspeed and angles
        airspeed = np.linalg.norm(velocity)
        # Use consistent minimum airspeed for all calculations
        safe_airspeed = max(airspeed, self.params.min_airspeed_aero)

        # Angle of attack (use safe minimum for u to handle near-vertical flight)
        # This prevents discontinuity when u ≈ 0
        min_u = self.params.min_u_velocity
        u_safe = max(abs(u), min_u) * np.sign(u) if abs(u) > 1e-6 else min_u
        alpha = np.arctan2(w, u_safe)
        alpha = np.clip(alpha, -self._max_alpha_rad, self._max_alpha_rad)

        # Sideslip angle (not actively used in current model, but calculated for completeness)
        beta = np.arcsin(np.clip(v / safe_airspeed, -1, 1))

        # Dynamic pressure
        q_dyn = 0.5 * self.params.air_density * airspeed**2

        # === Aerodynamic forces (body frame) ===

        # Lift coefficient (affected by alpha and elevator)
        elevator_rad = controls.elevator * self._max_elevator_rad
        cl = self.params.cl_0 + self.params.cl_alpha * alpha + \
             self.params.cl_elevator * elevator_rad

        # Drag coefficient
        cd = self.params.cd_0 + self.params.cd_alpha2 * alpha**2

        # Side force coefficient (affected by rudder)
        rudder_rad = controls.rudder * self._max_rudder_rad
        cy = self.params.cy_rudder * rudder_rad

        # Aerodynamic forces
        lift = q_dyn * self.params.wing_area * cl
        drag = q_dyn * self.params.wing_area * cd
        side_force = q_dyn * self.params.wing_area * cy

        # Transform to body frame (simplified - assuming small angles)
        fx_aero = -drag * np.cos(alpha) + lift * np.sin(alpha)
        fy_aero = side_force
        fz_aero = -drag * np.sin(alpha) - lift * np.cos(alpha)

        # === Thrust ===
        # Propeller thrust model: linear decay from static thrust to zero
        # T = T_max * (1 - V/V_max) for V < V_max, else 0
        # This matches real propeller behavior where thrust decreases with airspeed
        V_max = self.params.thrust_zero_velocity
        thrust_factor = max(0.0, 1.0 - airspeed / V_max)
        thrust = self.params.max_thrust * controls.throttle * thrust_factor
        fx_thrust = thrust
        fy_thrust = 0.0
        fz_thrust = 0.0

        # === Gravity (transform to body frame) ===
        # Gravity acceleration components in body frame (m/s²)
        g = self.params.gravity
        ax_grav = -g * np.sin(theta)
        ay_grav = g * np.cos(theta) * np.sin(phi)
        az_grav = g * np.cos(theta) * np.cos(phi)

        # Total force (body frame) - F = ma, so F_grav = m * a_grav
        mass = self.params.mass
        fx = fx_aero + fx_thrust + ax_grav * mass
        fy = fy_aero + fy_thrust + ay_grav * mass
        fz = fz_aero + fz_thrust + az_grav * mass

        # === Moments (body frame) ===

        aileron_rad = controls.aileron * self._max_aileron_rad

        # Roll moment (aileron + damping + dihedral effect)
        l_moment = q_dyn * self.params.wing_area * self.params.wing_span * \
                   (self.params.cl_aileron * aileron_rad + \
                    self.params.damping_roll * p * self.params.wing_span / (2 * safe_airspeed) + \
                    self.params.cl_beta * beta)

        # Pitch moment (elevator + damping + static stability)
        # cm_alpha provides static pitch stability - nose-down moment when alpha increases
        m_moment = q_dyn * self.params.wing_area * self.params.chord * \
                   (self.params.cm_elevator * elevator_rad + \
                    self.params.cm_alpha * alpha + \
                    self.params.damping_pitch * q * self.params.chord / (2 * safe_airspeed))

        # Yaw moment (rudder + damping + weathercock stability)
        n_moment = q_dyn * self.params.wing_area * self.params.wing_span * \
                   (self.params.cn_rudder * rudder_rad + \
                    self.params.damping_yaw * r * self.params.wing_span / (2 * safe_airspeed) + \
                    self.params.cn_beta * beta)

        # === State derivatives ===

        # Position derivative (NED frame) - need to transform velocity from body to NED
        pos_dot = self._body_to_ned(velocity, attitude)

        # Velocity derivative (body frame)
        vel_dot = np.array([
            fx / self.params.mass - q * w + r * v,
            fy / self.params.mass - r * u + p * w,
            fz / self.params.mass - p * v + q * u
        ])

        # Attitude derivative (Euler angle rates)
        # Clamp theta to avoid singularity at ±90° (gimbal lock)
        theta_safe = np.clip(theta, -self._max_pitch_rad, self._max_pitch_rad)
        cos_theta = np.cos(theta_safe)
        tan_theta = np.tan(theta_safe)

        att_dot = np.array([
            p + np.sin(phi) * tan_theta * q + np.cos(phi) * tan_theta * r,
            np.cos(phi) * q - np.sin(phi) * r,
            (np.sin(phi) * q + np.cos(phi) * r) / cos_theta
        ])

        # Angular acceleration (Euler's equations with gyroscopic coupling)
        # Full equations: ṗ = (L - (Izz-Iyy)qr) / Ixx, etc.
        Ixx = self.params.inertia_xx
        Iyy = self.params.inertia_yy
        Izz = self.params.inertia_zz

        rate_dot = np.array([
            (l_moment - (Izz - Iyy) * q * r) / Ixx,
            (m_moment - (Ixx - Izz) * p * r) / Iyy,
            (n_moment - (Iyy - Ixx) * p * q) / Izz
        ])

        # Clamp angular accelerations to prevent numerical explosion
        max_ang_accel = self.params.max_angular_acceleration
        rate_dot = np.clip(rate_dot, -max_ang_accel, max_ang_accel)

        # Clamp velocity derivatives
        max_accel = self.params.max_acceleration
        vel_dot = np.clip(vel_dot, -max_accel, max_accel)

        # Combine all derivatives
        state_dot = np.concatenate([pos_dot, vel_dot, att_dot, rate_dot])

        # Safety check: Replace NaN/Inf with zeros
        if not np.all(np.isfinite(state_dot)):
            logger.warning(
                f"Non-finite values in state derivative at t={self.time:.2f}s, "
                f"clamping to zero"
            )
            state_dot = np.nan_to_num(state_dot, nan=0.0, posinf=0.0, neginf=0.0)

        return state_dot

    def _body_to_ned(self, vel_body: np.ndarray, attitude: np.ndarray) -> np.ndarray:
        """Transform velocity from body frame to NED frame.

        Args:
            vel_body: Velocity in body frame [u, v, w]
            attitude: Euler angles [roll, pitch, yaw]

        Returns:
            Velocity in NED frame [vn, ve, vd]
        """
        phi, theta, psi = attitude

        # Rotation matrix from body to NED
        R = np.array([
            [np.cos(theta)*np.cos(psi),
             np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi),
             np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),
             np.sin(phi)*np.cos(theta),
             np.cos(phi)*np.cos(theta)]
        ])

        return R @ vel_body
