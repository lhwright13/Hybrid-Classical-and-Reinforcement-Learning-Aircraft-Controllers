"""Level 1: Waypoint Navigation Agent - 3D waypoint navigation."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.hsa_agent import HSAAgent
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig, Waypoint
)


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

    def __init__(self, config: ControllerConfig, guidance_type: str = 'LOS'):
        """Initialize waypoint agent.

        Args:
            config: Controller configuration
            guidance_type: Guidance algorithm ('LOS' or 'PN')
        """
        self.config = config
        self.guidance_type = guidance_type

        # Inner loop: HSA controller (Level 2)
        self.hsa_agent = HSAAgent(config)

        # Waypoint acceptance radius (meters)
        self.acceptance_radius = 10.0

        # Proportional navigation constant (for PN guidance)
        self.N = 3.0  # Typical range 3-5

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.WAYPOINT
        """
        return ControlMode.WAYPOINT

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState
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

        elif self.guidance_type == 'PN':
            # Proportional navigation: more aggressive
            # TODO: Implement PN guidance
            # For now, fall back to LOS
            heading_cmd = np.arctan2(error[1], error[0])

        else:
            # Default to LOS
            heading_cmd = np.arctan2(error[1], error[0])

        # Wrap heading to [-π, π]
        heading_cmd = np.arctan2(np.sin(heading_cmd), np.cos(heading_cmd))

        # === Altitude Command ===
        # Desired altitude from waypoint
        altitude_cmd = waypoint.altitude

        # === Speed Command ===
        # Use waypoint speed if specified, otherwise maintain current
        if waypoint.speed is not None:
            speed_cmd = waypoint.speed
        else:
            speed_cmd = state.airspeed

        # === Create HSA Command (Level 2) ===
        hsa_cmd = ControlCommand(
            mode=ControlMode.HSA,
            heading=heading_cmd,
            speed=speed_cmd,
            altitude=altitude_cmd
        )

        # Inner loop: HSA → surfaces (Level 2)
        surfaces = self.hsa_agent.compute_action(hsa_cmd, state)

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
        self.hsa_agent.reset()

    def __repr__(self) -> str:
        """String representation."""
        return (f"WaypointAgent(guidance={self.guidance_type}, "
                f"acceptance_radius={self.acceptance_radius:.1f}m)")
