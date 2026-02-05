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
        self._state = np.zeros(12)

        # Control inputs
        self._controls = ControlSurfaces()

        # Time
        self._time = 0.0

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
        self._time = 0.0

        if initial_state is None:
            # Default: level flight at 100m altitude, 20 m/s airspeed
            self._state = np.array([
                0.0, 0.0, -100.0,  # position (NED)
                20.0, 0.0, 0.0,    # velocity (body frame)
                0.0, 0.0, 0.0,     # attitude (roll, pitch, yaw)
                0.0, 0.0, 0.0      # angular rates
            ])
        else:
            self._state = np.concatenate([
                initial_state.position,
                initial_state.velocity,
                initial_state.attitude,
                initial_state.angular_rate
            ])
            self._time = initial_state.time

    def set_controls(self, controls: ControlSurfaces) -> None:
        """Set control surface deflections.

        Args:
            controls: Control surface commands (normalized -1 to 1, throttle 0 to 1)
        """
        # Clamp control inputs to valid ranges to prevent unrealistic deflections
        self._controls = ControlSurfaces(
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
        k1 = self._dynamics(self._state, self._controls)
        k2 = self._dynamics(self._state + 0.5 * dt * k1, self._controls)
        k3 = self._dynamics(self._state + 0.5 * dt * k2, self._controls)
        k4 = self._dynamics(self._state + dt * k3, self._controls)

        self._state = self._state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._time += dt

        # Clamp state variables to safe ranges
        # Position (NED) - no strict limits but check for NaN
        self._state[0:3] = np.nan_to_num(self._state[0:3], nan=0.0, posinf=10000.0, neginf=-10000.0)

        # Velocity (body frame) - use configured limits
        max_vel = self.params.max_velocity
        self._state[3:6] = np.clip(self._state[3:6], -max_vel, max_vel)

        # Attitude (Euler angles) - wrap and clamp
        # Roll: wrap to [-π, π]
        self._state[6] = np.arctan2(np.sin(self._state[6]), np.cos(self._state[6]))
        # Pitch: clamp to safe range (avoid gimbal lock)
        self._state[7] = np.clip(self._state[7], -self._max_pitch_rad, self._max_pitch_rad)
        # Yaw: wrap to [-π, π]
        self._state[8] = np.arctan2(np.sin(self._state[8]), np.cos(self._state[8]))

        # Angular rates - use configured limits
        self._state[9:12] = np.clip(self._state[9:12], -self._max_rate_rad, self._max_rate_rad)

        # Ground collision detection
        altitude = -self._state[2]  # NED: down is positive, altitude is negative of down
        if altitude < 0:
            logger.error(
                f"Ground collision detected! Altitude: {altitude:.2f}m, "
                f"Time: {self._time:.2f}s - Resetting to ground level"
            )
            self._state[2] = 0.0  # Reset to ground level (down = 0)
            self._state[5] = max(0.0, self._state[5])  # Zero or reverse vertical velocity (w)

        # Final NaN check
        if not np.all(np.isfinite(self._state)):
            logger.error(
                f"Non-finite state detected at t={self._time:.2f}s, "
                f"resetting to safe values"
            )
            self._state = np.nan_to_num(self._state, nan=0.0, posinf=0.0, neginf=0.0)

        return self.get_state()

    def get_state(self) -> AircraftState:
        """Get current state as AircraftState object.

        Returns:
            Current aircraft state
        """
        position = self._state[0:3]
        velocity = self._state[3:6]
        attitude = self._state[6:9]
        angular_rate = self._state[9:12]

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
            time=self._time,
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
        velocity = state[3:6]  # Body frame: [u, v, w]
        attitude = state[6:9]  # Euler angles: [roll, pitch, yaw]
        angular_rate = state[9:12]  # Body rates: [p, q, r]

        u, v, w = velocity
        phi, theta, psi = attitude
        p, q, r = angular_rate

        # Pre-compute trig values used throughout (each computed once)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        # === Forces and Moments ===

        # Airspeed and angles
        airspeed = np.linalg.norm(velocity)
        safe_airspeed = max(airspeed, self.params.min_airspeed_aero)

        # Angle of attack
        min_u = self.params.min_u_velocity
        u_safe = max(abs(u), min_u) * np.sign(u) if abs(u) > 1e-6 else min_u
        alpha = np.arctan2(w, u_safe)
        alpha = np.clip(alpha, -self._max_alpha_rad, self._max_alpha_rad)

        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)

        # Sideslip angle
        beta = np.arcsin(np.clip(v / safe_airspeed, -1, 1))

        # Dynamic pressure
        q_dyn = 0.5 * self.params.air_density * airspeed**2

        # === Aerodynamic forces (body frame) ===

        elevator_rad = controls.elevator * self._max_elevator_rad
        cl = (self.params.cl_0 + self.params.cl_alpha * alpha
              + self.params.cl_elevator * elevator_rad)

        cd = self.params.cd_0 + self.params.cd_alpha2 * alpha**2

        rudder_rad = controls.rudder * self._max_rudder_rad
        cy = self.params.cy_rudder * rudder_rad

        q_S = q_dyn * self.params.wing_area
        lift = q_S * cl
        drag = q_S * cd
        side_force = q_S * cy

        fx_aero = -drag * cos_alpha + lift * sin_alpha
        fy_aero = side_force
        fz_aero = -drag * sin_alpha - lift * cos_alpha

        # === Thrust ===
        V_max = self.params.thrust_zero_velocity
        thrust_factor = max(0.0, 1.0 - airspeed / V_max)
        thrust = self.params.max_thrust * controls.throttle * thrust_factor

        # === Gravity (transform to body frame) ===
        g = self.params.gravity
        mass = self.params.mass
        fx = fx_aero + thrust + (-g * sin_theta) * mass
        fy = fy_aero + (g * cos_theta * sin_phi) * mass
        fz = fz_aero + (g * cos_theta * cos_phi) * mass

        # === Moments (body frame) ===

        aileron_rad = controls.aileron * self._max_aileron_rad
        half_span_over_V = self.params.wing_span / (2 * safe_airspeed)
        half_chord_over_V = self.params.chord / (2 * safe_airspeed)

        # Roll moment (aileron + damping + dihedral effect)
        l_moment = q_S * self.params.wing_span * (
            self.params.cl_aileron * aileron_rad
            + self.params.damping_roll * p * half_span_over_V
            + self.params.cl_beta * beta)

        # Pitch moment (elevator + damping + static stability)
        m_moment = q_S * self.params.chord * (
            self.params.cm_elevator * elevator_rad
            + self.params.cm_alpha * alpha
            + self.params.damping_pitch * q * half_chord_over_V)

        # Yaw moment (rudder + damping + weathercock stability)
        n_moment = q_S * self.params.wing_span * (
            self.params.cn_rudder * rudder_rad
            + self.params.damping_yaw * r * half_span_over_V
            + self.params.cn_beta * beta)

        # === State derivatives ===

        # Position derivative (body to NED) - inline rotation to reuse trig values
        sin_phi_sin_theta = sin_phi * sin_theta
        cos_phi_sin_theta = cos_phi * sin_theta
        pos_dot = np.array([
            cos_theta * cos_psi * u
            + (sin_phi_sin_theta * cos_psi - cos_phi * sin_psi) * v
            + (cos_phi_sin_theta * cos_psi + sin_phi * sin_psi) * w,
            cos_theta * sin_psi * u
            + (sin_phi_sin_theta * sin_psi + cos_phi * cos_psi) * v
            + (cos_phi_sin_theta * sin_psi - sin_phi * cos_psi) * w,
            -sin_theta * u
            + sin_phi * cos_theta * v
            + cos_phi * cos_theta * w
        ])

        # Velocity derivative (body frame)
        inv_mass = 1.0 / mass
        vel_dot = np.array([
            fx * inv_mass - q * w + r * v,
            fy * inv_mass - r * u + p * w,
            fz * inv_mass - p * v + q * u
        ])

        # Attitude derivative (Euler angle rates)
        theta_safe = np.clip(theta, -self._max_pitch_rad, self._max_pitch_rad)
        cos_theta_safe = np.cos(theta_safe)
        tan_theta_safe = np.tan(theta_safe)

        att_dot = np.array([
            p + sin_phi * tan_theta_safe * q + cos_phi * tan_theta_safe * r,
            cos_phi * q - sin_phi * r,
            (sin_phi * q + cos_phi * r) / cos_theta_safe
        ])

        # Angular acceleration (Euler's equations with gyroscopic coupling)
        Ixx = self.params.inertia_xx
        Iyy = self.params.inertia_yy
        Izz = self.params.inertia_zz

        rate_dot = np.array([
            (l_moment - (Izz - Iyy) * q * r) / Ixx,
            (m_moment - (Ixx - Izz) * p * r) / Iyy,
            (n_moment - (Iyy - Ixx) * p * q) / Izz
        ])

        # Clamp derivatives to prevent numerical explosion
        rate_dot = np.clip(rate_dot,
                           -self.params.max_angular_acceleration,
                           self.params.max_angular_acceleration)
        vel_dot = np.clip(vel_dot,
                          -self.params.max_acceleration,
                          self.params.max_acceleration)

        # Combine all derivatives
        state_dot = np.concatenate([pos_dot, vel_dot, att_dot, rate_dot])

        # Safety check: Replace NaN/Inf with zeros
        if not np.all(np.isfinite(state_dot)):
            logger.warning(
                f"Non-finite values in state derivative at t={self._time:.2f}s, "
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
