"""Aircraft backend interface for simulation and hardware."""

from abc import ABC, abstractmethod
from typing import Optional
from controllers.types import AircraftState, ControlSurfaces


class AircraftInterface(ABC):
    """Abstract interface for aircraft backends.

    This interface enables complete swappability between:
    - Simulation backends (JSBSim, SimplifiedBackend)
    - Real hardware (TeensyBackend)
    - Hardware-in-the-Loop (HILBackend)

    Zero code changes required to switch backends - only config changes.
    """

    @abstractmethod
    def step(self, dt: float) -> AircraftState:
        """Advance simulation/hardware by dt seconds.

        For simulation: Steps the physics forward by dt
        For hardware: Returns latest sensor data (dt is nominal, not enforced)

        Args:
            dt: Time step in seconds (simulation) or nominal update period (hardware)

        Returns:
            Updated aircraft state with latest sensor data

        Example:
            >>> backend = JSBSimBackend(config)
            >>> state = backend.step(dt=0.01)  # Step 10ms forward
            >>> print(state.altitude)
            100.5
        """
        pass

    @abstractmethod
    def set_controls(self, surfaces: ControlSurfaces) -> None:
        """Set control surface commands.

        For simulation: Immediately updates actuator commands
        For hardware: Sends commands to flight controller (may have latency)

        Args:
            surfaces: Control surface deflections and throttle
                - All surfaces: -1 to 1 (normalized)
                - Throttle: 0 to 1

        Example:
            >>> surfaces = ControlSurfaces(
            ...     elevator=0.1,   # Slight nose up
            ...     aileron=-0.2,   # Roll left
            ...     rudder=0.0,
            ...     throttle=0.7
            ... )
            >>> backend.set_controls(surfaces)
        """
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[AircraftState] = None) -> AircraftState:
        """Reset aircraft to initial state.

        For simulation: Resets physics to initial conditions
        For hardware: Not truly resettable, returns current state

        Args:
            initial_state: Desired initial state (None = use default from config)

        Returns:
            Actual initial state after reset

        Example:
            >>> initial = AircraftState(altitude=100.0, airspeed=20.0, ...)
            >>> state = backend.reset(initial)
            >>> print(state.altitude)
            100.0
        """
        pass

    @abstractmethod
    def get_state(self) -> AircraftState:
        """Get current aircraft state.

        Returns:
            Current full state

        Example:
            >>> state = backend.get_state()
            >>> print(f"Altitude: {state.altitude:.2f}m")
            Altitude: 105.32m
        """
        pass

    def close(self) -> None:
        """Clean up resources (close connections, files, etc.).

        Called when backend is no longer needed. Optional to implement.

        Example:
            >>> backend.close()  # Closes serial port for hardware
        """
        pass  # Optional

    @abstractmethod
    def get_backend_type(self) -> str:
        """Return backend type identifier.

        Returns:
            "simulation", "hardware", or "hil"

        Example:
            >>> backend.get_backend_type()
            'simulation'
        """
        pass

    def get_dt_nominal(self) -> float:
        """Get nominal time step for this backend.

        Returns:
            Nominal dt in seconds (default 0.01 = 100 Hz)

        Example:
            >>> backend.get_dt_nominal()
            0.01
        """
        return 0.01  # Default 100 Hz

    def is_real_hardware(self) -> bool:
        """Check if this is real hardware (vs simulation).

        Returns:
            True if hardware, False if simulation

        Example:
            >>> if backend.is_real_hardware():
            ...     print("Running on real aircraft - be careful!")
        """
        return self.get_backend_type() in ["hardware", "hil"]

    def supports_reset(self) -> bool:
        """Check if backend supports reset to arbitrary state.

        Returns:
            True if reset is fully supported (simulation), False otherwise (hardware)

        Example:
            >>> if not backend.supports_reset():
            ...     print("Warning: Cannot reset hardware to initial state")
        """
        return self.get_backend_type() == "simulation"

    def get_info(self) -> dict:
        """Get additional backend information (optional).

        Returns:
            Dictionary with backend-specific information

        Example:
            >>> info = backend.get_info()
            >>> print(info)
            {'aircraft_model': 'f16', 'dt_sim': 0.005, 'sensor_noise': True}
        """
        return {
            'backend_type': self.get_backend_type(),
            'dt_nominal': self.get_dt_nominal(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(type={self.get_backend_type()})"
