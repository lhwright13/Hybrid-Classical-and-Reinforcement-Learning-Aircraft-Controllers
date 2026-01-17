"""Level 1: Waypoint Navigation Agent - 3D waypoint navigation."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.hsa_agent import HSAAgent
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig, Waypoint
)
from controllers.config_loader import FlightControlConfig, GuidanceConfig


class WaypointAgent(BaseAgent):
    """Level 1: Waypoint navigation agent.

    Commands 3D waypoints, outputs HSA commands to Level 2.

    Guidance algorithms:
    - Line-of-sight (LOS): Direct path to waypoint
    - Proportional navigation (PN): More aggressive pursuit

    Architecture:
        Waypoint → guidance → HSA command → Level 2 → surfaces

    Use cases:
    - Autonomous navigation
    - Mission planning (survey, patrol)
    - Multi-waypoint routes
    - Return-to-home
    - Obstacle avoidance (high-level)
    """

    def __init__(
        self,
        config: ControllerConfig,
        guidance_type: str = 'LOS',
        flight_config: FlightControlConfig = None
    ):
        """Initialize waypoint agent.

        Args:
            config: Controller configuration (legacy)
            guidance_type: Guidance algorithm ('LOS', 'PP', or 'PN')
            flight_config: Flight control configuration with guidance params
        """
        self.config = config
        self.guidance_type = guidance_type

        # Use flight_config if provided, otherwise use defaults
        if flight_config is not None:
            self.guidance = flight_config.guidance
            self.max_bank_angle = flight_config.hsa.max_bank_angle
        else:
            self.guidance = GuidanceConfig()
            self.max_bank_angle = 25.0

        # Inner loop: HSA controller (Level 2)
        self.hsa_agent = HSAAgent(config, flight_config)

        # Waypoint acceptance radius (meters) - from guidance config
        self.acceptance_radius = self.guidance.acceptance_radius

        # Proportional navigation constant (for PN guidance) - from guidance config
        self.N = self.guidance.pn_gain

        # Heading command smoothing (disabled for waypoint navigation)
        self.last_heading_cmd = None
        self.max_heading_rate = None  # No rate limiting - allow sharp turns at waypoints

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.WAYPOINT
        """
        return ControlMode.WAYPOINT

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = 0.01
    ) -> ControlSurfaces:
        """Compute surfaces from waypoint commands.

        Guidance flow:
        1. Compute vector from current position to waypoint
        2. Compute desired heading (LOS or PN)
        3. Compute desired altitude
        4. Use waypoint speed or maintain current speed
        5. Send HSA command to Level 2

        Args:
            command: Waypoint control command
            state: Current aircraft state
            dt: Time step in seconds (default 0.01 for 100 Hz)

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not WAYPOINT
        """
        assert command.mode == ControlMode.WAYPOINT, \
            f"Waypoint agent expects WAYPOINT mode, got {command.mode}"

        waypoint = command.waypoint

        # Position error vector (NED frame)
        error = np.array([
            waypoint.north - state.north,
            waypoint.east - state.east,
            waypoint.down - state.down
        ])

        # Distance to waypoint
        horizontal_distance = np.linalg.norm(error[:2])  # Horizontal only
        distance_3d = np.linalg.norm(error)

        # === Guidance Algorithm ===
        if self.guidance_type == 'LOS':
            # Line-of-sight guidance: point directly at waypoint
            heading_cmd = np.arctan2(error[1], error[0])  # arctan2(east, north)

            # Turn anticipation: Start banking early to reduce overshoot
            V = max(state.airspeed, 10.0)  # Use at least 10 m/s for calculation
            max_bank = np.radians(self.guidance.los_max_bank)
            turn_radius = V**2 / (9.81 * np.tan(max_bank))  # R = V^2/(g*tan(phi))

            # Compute heading change required
            heading_error = heading_cmd - state.heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Wrap

            # If we're close to waypoint and need to turn, start turning early
            # Turn anticipation distance = turn_radius (conservative)
            anticipation_dist = turn_radius * abs(heading_error) / np.radians(90)  # Scale with turn angle
            lead_angle_rad = np.radians(self.guidance.los_lead_angle)

            if horizontal_distance < anticipation_dist and abs(heading_error) > np.radians(20):
                # We're in turn anticipation zone - start banking NOW
                # Add a "lead point" ahead of the waypoint in the turn direction
                # Blend between direct LOS and lead angle based on distance
                blend = 1.0 - (horizontal_distance / anticipation_dist)
                heading_cmd = heading_cmd + blend * lead_angle_rad * np.sign(heading_error)
                heading_cmd = np.arctan2(np.sin(heading_cmd), np.cos(heading_cmd))  # Wrap

        elif self.guidance_type == 'PP' or self.guidance_type == 'PURE_PURSUIT':
            # Pure Pursuit guidance
            V = max(state.airspeed, 10.0)

            # Calculate achievable turn radius at max bank
            max_bank = np.radians(self.max_bank_angle)
            turn_radius = V**2 / (9.81 * np.tan(max_bank))

            # Lookahead distance from config
            lookahead_dist = self.guidance.lookahead_time * V

            # Reduce lookahead near waypoint for tighter corner entry
            proximity_dist = self.guidance.proximity_scale_distance * turn_radius
            if horizontal_distance < proximity_dist:
                scale = 0.6 + 0.4 * (horizontal_distance / proximity_dist)
                lookahead_dist *= scale

            # Clamp lookahead to configured bounds
            lookahead_dist = np.clip(
                lookahead_dist,
                self.guidance.lookahead_min,
                self.guidance.lookahead_max
            )

            if horizontal_distance > 1.0:  # Avoid division by zero
                # Always use carrot-based pursuit for smooth tracking
                direction = error[:2] / horizontal_distance
                effective_lookahead = min(lookahead_dist, horizontal_distance)
                carrot = direction * effective_lookahead
                heading_cmd = np.arctan2(carrot[1], carrot[0])
            else:
                # Very close - point directly at waypoint
                heading_cmd = np.arctan2(error[1], error[0])

        else:
            # Default to LOS
            heading_cmd = np.arctan2(error[1], error[0])

        # Wrap heading to [-π, π]
        heading_cmd = np.arctan2(np.sin(heading_cmd), np.cos(heading_cmd))

        # === Heading Rate Limiting ===
        # Optional: Prevent instantaneous heading changes at waypoint corners
        if self.max_heading_rate is not None and self.last_heading_cmd is not None:
            # Compute heading change (wrap to [-π, π])
            heading_change = heading_cmd - self.last_heading_cmd
            heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))

            # Limit rate of change
            max_change = self.max_heading_rate * dt
            if abs(heading_change) > max_change:
                # Rate limit the heading command
                heading_cmd = self.last_heading_cmd + np.sign(heading_change) * max_change
                # Wrap result
                heading_cmd = np.arctan2(np.sin(heading_cmd), np.cos(heading_cmd))

        # Store for next iteration (for rate limiting if enabled)
        self.last_heading_cmd = heading_cmd

        # === Altitude Command ===
        # Desired altitude from waypoint
        altitude_cmd = waypoint.altitude

        # === Speed Command ===
        # Use waypoint speed if specified, otherwise maintain current
        if waypoint.speed is not None:
            speed_cmd = waypoint.speed
        else:
            speed_cmd = state.airspeed

        # Speed reduction near waypoints for sharper turns
        heading_to_wp = np.arctan2(error[1], error[0])
        heading_error = abs(heading_to_wp - state.heading)
        heading_error = abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

        threshold_dist = self.guidance.turn_threshold_distance
        threshold_angle = np.radians(self.guidance.turn_threshold_angle)

        if horizontal_distance < threshold_dist and heading_error > threshold_angle:
            # Reduce speed for sharp turns
            reduction = self.guidance.max_speed_reduction * (1.0 - horizontal_distance / threshold_dist)
            speed_cmd = speed_cmd * (1.0 - reduction)
            speed_cmd = max(speed_cmd, self.guidance.min_speed)

        # === Create HSA Command (Level 2) ===
        hsa_cmd = ControlCommand(
            mode=ControlMode.HSA,
            heading=heading_cmd,
            speed=speed_cmd,
            altitude=altitude_cmd
        )

        # Inner loop: HSA → surfaces (Level 2)
        # Pass dt for consistent timing across all control levels
        surfaces = self.hsa_agent.compute_action(hsa_cmd, state, dt)

        return surfaces

    def reached_waypoint(self, state: AircraftState, waypoint: Waypoint) -> bool:
        """Check if waypoint has been reached.

        Args:
            state: Current aircraft state
            waypoint: Target waypoint

        Returns:
            True if within acceptance radius
        """
        error = np.array([
            waypoint.north - state.north,
            waypoint.east - state.east,
            waypoint.down - state.down
        ])
        distance = np.linalg.norm(error)
        return distance < self.acceptance_radius

    def reset(self):
        """Reset agent state."""
        self.last_heading_cmd = None
        self.hsa_agent.reset()

    def __repr__(self) -> str:
        """String representation."""
        return (f"WaypointAgent(guidance={self.guidance_type}, "
                f"acceptance_radius={self.acceptance_radius:.1f}m)")
