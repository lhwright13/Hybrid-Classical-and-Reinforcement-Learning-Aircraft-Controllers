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
from typing import Optional
from dataclasses import dataclass
from controllers.types import AircraftState, ControlSurfaces


@dataclass
class AircraftParams:
    """Physical parameters for aircraft dynamics.

    These are simplified parameters for a small fixed-wing aircraft
    (similar to a Cessna 172 or large RC plane).
    """
    # Mass properties
    mass: float = 10.0  # kg (typical for large RC aircraft)
    inertia_xx: float = 0.5  # kg⋅m² (roll inertia)
    inertia_yy: float = 1.0  # kg⋅m² (pitch inertia)
    inertia_zz: float = 1.2  # kg⋅m² (yaw inertia)

    # Aerodynamic reference
    wing_area: float = 0.5  # m²
    wing_span: float = 2.0  # m
    chord: float = 0.25  # m (mean aerodynamic chord)

    # Aerodynamic coefficients (tuned to match JSBSim)
    cl_0: float = 0.35  # Zero-alpha lift coefficient (increased for better lift)
    cl_alpha: float = 5.0  # Lift curve slope (1/rad) - matches theory: 2π/(1+2/AR) for AR=8
    cd_0: float = 0.025  # Parasitic drag coefficient (reduced - cleaner aircraft)
    cd_alpha2: float = 0.05  # Induced drag coefficient - matches CL²/(π×AR×e) with AR=8, e=0.8

    # Control effectiveness - CONSERVATIVE for stability
    cl_elevator: float = 0.5  # Lift change per elevator deflection (1/rad)
    cm_elevator: float = -1.0  # Pitch moment per elevator (1/rad)
    cy_rudder: float = 0.3  # Side force per rudder (1/rad)
    cn_rudder: float = -0.1  # Yaw moment per rudder (1/rad)
    cl_aileron: float = 0.15  # Roll moment per aileron (1/rad)

    # Damping coefficients (stabilizing) - HIGH for maximum stability
    damping_roll: float = -2.0  # Roll damping
    damping_pitch: float = -8.0  # Pitch damping
    damping_yaw: float = -1.5  # Yaw damping

    # Thrust
    max_thrust: float = 50.0  # N (max thrust)

    # Environment
    air_density: float = 1.225  # kg/m³ (sea level)
    gravity: float = 9.81  # m/s²


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
            dt: Time step (seconds)

        Returns:
            Updated aircraft state
        """
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

        # Velocity (body frame) - realistic limits
        self.state[3:6] = np.clip(self.state[3:6], -100.0, 100.0)  # ±100 m/s

        # Attitude (Euler angles) - wrap and clamp
        # Roll: wrap to [-π, π]
        self.state[6] = np.arctan2(np.sin(self.state[6]), np.cos(self.state[6]))
        # Pitch: clamp to safe range (avoid gimbal lock)
        self.state[7] = np.clip(self.state[7], np.radians(-85), np.radians(85))
        # Yaw: wrap to [-π, π]
        self.state[8] = np.arctan2(np.sin(self.state[8]), np.cos(self.state[8]))

        # Angular rates - realistic limits
        self.state[9:12] = np.clip(self.state[9:12], np.radians(-360), np.radians(360))  # ±360°/s

        # Final NaN check
        if not np.all(np.isfinite(self.state)):
            print("ERROR: Non-finite state detected, resetting to safe values")
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

        # Compute heading from ground velocity in NED frame
        # Transform body velocity to NED frame
        velocity_ned = self._body_to_ned(velocity, attitude)
        # Heading: arctan2(east, north) in NED convention
        # 0° = North, 90° = East, 180° = South, 270° = West
        heading = np.arctan2(velocity_ned[1], velocity_ned[0])

        return AircraftState(
            time=self.time,
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_rate=angular_rate,
            airspeed=airspeed,
            altitude=altitude,
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
        safe_airspeed = max(airspeed, 10.0)  # Use at least 10 m/s for all aero calculations

        # Angle of attack (use safe minimum for u to handle near-vertical flight)
        # This prevents discontinuity when u ≈ 0
        u_safe = max(abs(u), 0.1) * np.sign(u) if abs(u) > 1e-6 else 0.1
        alpha = np.arctan2(w, u_safe)
        alpha = np.clip(alpha, np.radians(-30), np.radians(30))  # Limit to ±30°

        # Sideslip angle (not actively used in current model, but calculated for completeness)
        beta = np.arcsin(np.clip(v / safe_airspeed, -1, 1))

        # Dynamic pressure
        q_dyn = 0.5 * self.params.air_density * airspeed**2

        # === Aerodynamic forces (body frame) ===

        # Lift coefficient (affected by alpha and elevator)
        elevator_rad = controls.elevator * np.radians(30)  # Max 30 deg deflection
        cl = self.params.cl_0 + self.params.cl_alpha * alpha + \
             self.params.cl_elevator * elevator_rad

        # Drag coefficient
        cd = self.params.cd_0 + self.params.cd_alpha2 * alpha**2

        # Side force coefficient (affected by rudder)
        rudder_rad = controls.rudder * np.radians(30)
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
        # Propeller thrust model: thrust decreases linearly with airspeed
        # This matches propeller physics where efficiency drops at high speed
        V_ref = 15.0  # m/s - reference velocity (tuned to match JSBSim)
        # Linear decay is more aggressive than sqrt, better matches JSBSim behavior
        thrust_factor = V_ref / max(airspeed, V_ref)
        thrust = self.params.max_thrust * controls.throttle * thrust_factor
        fx_thrust = thrust
        fy_thrust = 0.0
        fz_thrust = 0.0

        # === Gravity (transform to body frame) ===
        g = self.params.gravity
        fx_grav = -g * np.sin(theta)
        fy_grav = g * np.cos(theta) * np.sin(phi)
        fz_grav = g * np.cos(theta) * np.cos(phi)

        # Total force (body frame)
        fx = fx_aero + fx_thrust + fx_grav * self.params.mass
        fy = fy_aero + fy_thrust + fy_grav * self.params.mass
        fz = fz_aero + fz_thrust + fz_grav * self.params.mass

        # === Moments (body frame) ===

        aileron_rad = controls.aileron * np.radians(30)

        # Roll moment (aileron + damping)
        l_moment = q_dyn * self.params.wing_area * self.params.wing_span * \
                   (self.params.cl_aileron * aileron_rad + \
                    self.params.damping_roll * p * self.params.wing_span / (2 * safe_airspeed))

        # Pitch moment (elevator + damping)
        m_moment = q_dyn * self.params.wing_area * self.params.chord * \
                   (self.params.cm_elevator * elevator_rad + \
                    self.params.damping_pitch * q * self.params.chord / (2 * safe_airspeed))

        # Yaw moment (rudder + damping)
        n_moment = q_dyn * self.params.wing_area * self.params.wing_span * \
                   (self.params.cn_rudder * rudder_rad + \
                    self.params.damping_yaw * r * self.params.wing_span / (2 * safe_airspeed))

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
        theta_safe = np.clip(theta, np.radians(-85), np.radians(85))
        cos_theta = np.cos(theta_safe)
        tan_theta = np.tan(theta_safe)

        att_dot = np.array([
            p + np.sin(phi) * tan_theta * q + np.cos(phi) * tan_theta * r,
            np.cos(phi) * q - np.sin(phi) * r,
            (np.sin(phi) * q + np.cos(phi) * r) / cos_theta
        ])

        # Angular acceleration (simplified - assuming diagonal inertia)
        rate_dot = np.array([
            l_moment / self.params.inertia_xx,
            m_moment / self.params.inertia_yy,
            n_moment / self.params.inertia_zz
        ])

        # Clamp angular accelerations to prevent numerical explosion
        rate_dot = np.clip(rate_dot, -1000.0, 1000.0)  # ±1000 rad/s²

        # Clamp velocity derivatives
        vel_dot = np.clip(vel_dot, -50.0, 50.0)  # ±50 m/s²

        # Combine all derivatives
        state_dot = np.concatenate([pos_dot, vel_dot, att_dot, rate_dot])

        # Safety check: Replace NaN/Inf with zeros
        if not np.all(np.isfinite(state_dot)):
            print("Warning: Non-finite values in state derivative, clamping to zero")
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
