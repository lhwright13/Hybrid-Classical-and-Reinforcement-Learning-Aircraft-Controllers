"""Simulation backend implementing AircraftInterface.

This module provides a complete simulation backend that can be swapped
with hardware backends with zero code changes.
"""

from typing import Optional
from interfaces.aircraft import AircraftInterface
from controllers.types import AircraftState, ControlSurfaces
from simulation.simplified_6dof import Simplified6DOF, AircraftParams


class SimulationAircraftBackend(AircraftInterface):
    """Simulation backend using simplified 6-DOF physics.

    This backend implements the AircraftInterface for simulation purposes.
    It can be easily swapped with JSBSim or hardware backends.

    Example:
        >>> config = {'aircraft_type': 'cessna', 'dt_physics': 0.001}
        >>> backend = SimulationAircraftBackend(config)
        >>> initial = AircraftState(altitude=100.0, airspeed=20.0)
        >>> backend.reset(initial)
        >>> backend.set_controls(ControlSurfaces(elevator=0.1, throttle=0.7))
        >>> state = backend.step(dt=0.01)
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize simulation backend.

        Args:
            config: Configuration dictionary with optional keys:
                - aircraft_type: 'cessna', 'rc_plane', or 'custom'
                - dt_physics: Physics timestep for sub-stepping (default: 0.001s)
                - params: Custom AircraftParams (overrides aircraft_type)
        """
        config = config or {}

        # Select aircraft parameters
        aircraft_type = config.get('aircraft_type', 'rc_plane')
        if 'params' in config:
            params = config['params']
        else:
            params = self._get_aircraft_params(aircraft_type)

        # Create physics simulator
        self.physics = Simplified6DOF(params)

        # Physics sub-stepping for numerical stability
        self.dt_physics = config.get('dt_physics', 0.001)  # 1ms physics step

        # Current state
        self._state = None

    def _get_aircraft_params(self, aircraft_type: str) -> AircraftParams:
        """Get aircraft parameters by type.

        Args:
            aircraft_type: 'cessna', 'rc_plane', or 'custom'

        Returns:
            Aircraft parameters
        """
        if aircraft_type == 'cessna':
            # Cessna 172 (scaled down for simplicity)
            return AircraftParams(
                mass=15.0,
                inertia_xx=1.0,
                inertia_yy=2.0,
                inertia_zz=2.5,
                wing_area=1.0,
                wing_span=3.0,
                max_thrust=80.0
            )
        elif aircraft_type == 'rc_plane':
            # Large RC plane (default)
            return AircraftParams()  # Uses defaults
        else:
            # Default to RC plane
            return AircraftParams()

    def step(self, dt: float) -> AircraftState:
        """Advance simulation by dt seconds.

        Uses sub-stepping for numerical stability: if dt > dt_physics,
        splits into multiple physics steps.

        Args:
            dt: Time step in seconds (can be larger than dt_physics)

        Returns:
            Updated aircraft state
        """
        # Sub-step physics for stability
        num_substeps = max(1, int(dt / self.dt_physics))
        dt_substep = dt / num_substeps

        for _ in range(num_substeps):
            self._state = self.physics.step(dt_substep)

        return self._state

    def set_controls(self, surfaces: ControlSurfaces) -> None:
        """Set control surface commands.

        Args:
            surfaces: Control surface deflections and throttle
                - All surfaces: -1 to 1 (normalized)
                - Throttle: 0 to 1
        """
        self.physics.set_controls(surfaces)

    def reset(self, initial_state: Optional[AircraftState] = None) -> AircraftState:
        """Reset simulation to initial state.

        Args:
            initial_state: Desired initial state (None = use default)

        Returns:
            Actual initial state after reset
        """
        self.physics.reset(initial_state)
        self._state = self.physics.get_state()
        return self._state

    def get_state(self) -> AircraftState:
        """Get current aircraft state.

        Returns:
            Current full state
        """
        if self._state is None:
            # If never initialized, return default state
            self._state = self.physics.get_state()
        return self._state

    def get_backend_type(self) -> str:
        """Return backend type identifier.

        Returns:
            "simulation"
        """
        return "simulation"

    def get_dt_nominal(self) -> float:
        """Get nominal time step for this backend.

        Returns:
            Nominal dt in seconds (100 Hz = 0.01s)
        """
        return 0.01  # 100 Hz

    def get_info(self) -> dict:
        """Get backend information.

        Returns:
            Dictionary with backend-specific information
        """
        info = super().get_info()
        info.update({
            'physics_engine': 'simplified_6dof',
            'dt_physics': self.dt_physics,
            'aircraft_mass': self.physics.params.mass,
            'max_thrust': self.physics.params.max_thrust,
        })
        return info

    def __repr__(self) -> str:
        """String representation."""
        return f"SimulationAircraftBackend(physics=Simplified6DOF, dt={self.dt_physics})"
