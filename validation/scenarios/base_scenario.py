"""Base class for validation scenarios."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

import pandas as pd

from interfaces.aircraft import AircraftInterface
from controllers.types import AircraftState, ControlSurfaces


class ValidationScenario(ABC):
    """Base class for physics validation scenarios.

    A scenario defines:
    - Initial conditions
    - Control sequence over time
    - Duration and timestep
    - Expected behavior
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize scenario.

        Args:
            config: Scenario configuration dict
        """
        self.config = config or {}
        self.duration = self.config.get('duration', 10.0)  # seconds
        self.dt = self.config.get('dt', 0.01)  # 100 Hz default

    @abstractmethod
    def get_name(self) -> str:
        """Get scenario name.

        Returns:
            Scenario name string
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get scenario description.

        Returns:
            Human-readable description
        """
        pass

    @abstractmethod
    def get_initial_conditions(self) -> AircraftState:
        """Get initial aircraft state.

        Returns:
            Initial state for both simulators
        """
        pass

    @abstractmethod
    def get_control_function(self) -> Callable[[float], ControlSurfaces]:
        """Get control sequence function.

        Returns:
            Function that takes time (seconds) and returns ControlSurfaces

        Example:
            >>> control_fn = scenario.get_control_function()
            >>> surfaces = control_fn(t=1.5)  # Get controls at t=1.5s
        """
        pass

    def run_simulation(self, backend: AircraftInterface) -> pd.DataFrame:
        """Run scenario on a given backend.

        Args:
            backend: Aircraft simulation backend (Simplified6DOF or JSBSim)

        Returns:
            DataFrame with time series of all state variables
        """
        # Reset to initial conditions
        initial_state = self.get_initial_conditions()
        backend.reset(initial_state)

        # Get control function
        control_fn = self.get_control_function()

        # Storage for trajectory
        data = []

        # Simulation loop
        num_steps = int(self.duration / self.dt)
        for i in range(num_steps):
            t = i * self.dt

            # Get controls for this timestep
            controls = control_fn(t)

            # Apply controls
            backend.set_controls(controls)

            # Step simulation
            state = backend.step(self.dt)

            # Record state
            record = {
                'time': state.time,
                'north': state.position[0],
                'east': state.position[1],
                'down': state.position[2],
                'altitude': state.altitude,
                'u': state.velocity[0],
                'v': state.velocity[1],
                'w': state.velocity[2],
                'airspeed': state.airspeed,
                'roll': state.attitude[0],
                'pitch': state.attitude[1],
                'yaw': state.attitude[2],
                'p': state.angular_rate[0],
                'q': state.angular_rate[1],
                'r': state.angular_rate[2],
                'elevator': controls.elevator,
                'aileron': controls.aileron,
                'rudder': controls.rudder,
                'throttle': controls.throttle,
            }
            data.append(record)

        return pd.DataFrame(data)

    def get_expected_metrics(self) -> Dict[str, Any]:
        """Get expected success criteria for this scenario.

        Returns:
            Dict with expected RMSE thresholds, correlation, etc.

        Example:
            {
                'position_rmse_threshold': 10.0,  # meters
                'attitude_rmse_threshold': 5.0,   # degrees
                'min_correlation': 0.90,
            }
        """
        # Default conservative thresholds
        return {
            'position_rmse_threshold': 20.0,  # meters
            'attitude_rmse_threshold': 10.0,  # degrees
            'min_correlation': 0.80,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.get_name()} (duration={self.duration}s, dt={self.dt}s)"
