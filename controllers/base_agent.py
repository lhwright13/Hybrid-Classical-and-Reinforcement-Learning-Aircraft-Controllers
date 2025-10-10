"""Base agent interface for multi-level control hierarchy."""

from abc import ABC, abstractmethod
from controllers.types import ControlMode, ControlCommand, AircraftState, ControlSurfaces


class BaseAgent(ABC):
    """Abstract base class for control agents at all levels.

    All agents must implement:
    - get_control_level(): Return the control level they operate at
    - compute_action(): Compute control outputs from commands and state
    """

    @abstractmethod
    def get_control_level(self) -> ControlMode:
        """Get the control level this agent operates at.

        Returns:
            ControlMode: The control level (WAYPOINT, HSA, ATTITUDE, RATE, or SURFACE)
        """
        pass

    @abstractmethod
    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState
    ) -> ControlSurfaces:
        """Compute control surface deflections from command and state.

        Args:
            command: Control command at this agent's level
            state: Current aircraft state

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command.mode doesn't match this agent's level
        """
        pass

    def reset(self):
        """Reset agent state (PIDs, integrators, etc.).

        Override in subclasses that have internal state.
        """
        pass
