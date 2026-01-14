"""Level 5: Surface Control Agent - Direct actuation layer."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.types import ControlMode, ControlCommand, AircraftState, ControlSurfaces


class SurfaceAgent(BaseAgent):
    """Level 5: Direct surface control agent.

    This is the lowest level of control. It receives surface commands and
    applies them directly to the aircraft. No PID control at this level -
    pure actuation with saturation limits.

    Use cases:
    - Novel aircraft configurations
    - Optimal control research
    - RL training at lowest level
    - Direct surface commands from expert pilots
    """

    def __init__(self, config: dict = None):
        """Initialize surface agent.

        Args:
            config: Configuration dictionary (optional)
                - surface_limits: Dict with min/max for each surface
        """
        self.config = config or {}

        # Surface limits (normalized -1 to 1, except throttle 0 to 1)
        limits = self.config.get('surface_limits', {})
        self.elevator_min = limits.get('elevator_min', -1.0)
        self.elevator_max = limits.get('elevator_max', 1.0)
        self.aileron_min = limits.get('aileron_min', -1.0)
        self.aileron_max = limits.get('aileron_max', 1.0)
        self.rudder_min = limits.get('rudder_min', -1.0)
        self.rudder_max = limits.get('rudder_max', 1.0)
        self.throttle_min = limits.get('throttle_min', 0.0)
        self.throttle_max = limits.get('throttle_max', 1.0)

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.SURFACE
        """
        return ControlMode.SURFACE

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = None
    ) -> ControlSurfaces:
        """Apply surface commands directly with saturation.

        Args:
            command: Surface control command
            state: Current aircraft state (not used at this level)
            dt: Time step (not used at this level, included for API compatibility)

        Returns:
            ControlSurfaces: Saturated surface deflections

        Raises:
            AssertionError: If command mode is not SURFACE
        """
        assert command.mode == ControlMode.SURFACE, \
            f"Surface agent expects SURFACE mode, got {command.mode}"

        # Direct pass-through with saturation
        surfaces = ControlSurfaces(
            elevator=np.clip(
                command.elevator,
                self.elevator_min,
                self.elevator_max
            ),
            aileron=np.clip(
                command.aileron,
                self.aileron_min,
                self.aileron_max
            ),
            rudder=np.clip(
                command.rudder,
                self.rudder_min,
                self.rudder_max
            ),
            throttle=np.clip(
                command.throttle,
                self.throttle_min,
                self.throttle_max
            )
        )

        return surfaces

    def __repr__(self) -> str:
        """String representation."""
        return (f"SurfaceAgent(elevator=[{self.elevator_min}, {self.elevator_max}], "
                f"aileron=[{self.aileron_min}, {self.aileron_max}], "
                f"rudder=[{self.rudder_min}, {self.rudder_max}], "
                f"throttle=[{self.throttle_min}, {self.throttle_max}])")
