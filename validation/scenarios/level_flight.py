"""Level flight validation scenario."""

import numpy as np
from typing import Callable

from controllers.types import AircraftState, ControlSurfaces
from .base_scenario import ValidationScenario


class LevelFlightScenario(ValidationScenario):
    """Trimmed level flight scenario.

    Tests basic aerodynamic equilibrium and steady-state behavior.
    This should show the best agreement between simplified and JSBSim models.

    Initial Conditions:
    - Altitude: 100m AGL
    - Airspeed: 20 m/s
    - Level attitude (0° roll, 0° pitch)

    Controls:
    - Fixed trim (elevator, throttle to maintain level flight)
    - Zero aileron, rudder

    Duration: 30 seconds

    Expected Outcome:
    - Both models maintain altitude ±2m
    - Airspeed stable ±1 m/s
    - Position RMSE < 5m
    - Correlation > 0.98
    """

    def __init__(self, config=None):
        """Initialize level flight scenario."""
        if config is None:
            config = {}

        # Override defaults for this scenario
        config.setdefault('duration', 30.0)
        config.setdefault('dt', 0.01)

        super().__init__(config)

        # Trim controls (tuned for ~20 m/s level flight)
        # These values are approximate - real trim depends on aircraft
        self.trim_elevator = self.config.get('trim_elevator', 0.0)
        self.trim_throttle = self.config.get('trim_throttle', 0.5)

    def get_name(self) -> str:
        """Get scenario name."""
        return "Level Flight"

    def get_description(self) -> str:
        """Get scenario description."""
        return ("Trimmed level flight at 100m altitude, 20 m/s airspeed. "
                "Tests basic aerodynamic equilibrium.")

    def get_initial_conditions(self) -> AircraftState:
        """Get initial state for level flight."""
        return AircraftState(
            time=0.0,
            position=np.array([0.0, 0.0, -100.0]),  # NED: 100m altitude
            velocity=np.array([20.0, 0.0, 0.0]),     # 20 m/s forward
            attitude=np.zeros(3),                     # Level flight
            angular_rate=np.zeros(3),                 # No rotation
            airspeed=20.0,
            altitude=100.0,
            ground_speed=20.0,
            heading=0.0
        )

    def get_control_function(self) -> Callable[[float], ControlSurfaces]:
        """Get control function for level flight.

        Returns constant trim controls throughout the flight.
        """
        def controls(t: float) -> ControlSurfaces:
            return ControlSurfaces(
                elevator=self.trim_elevator,
                aileron=0.0,
                rudder=0.0,
                throttle=self.trim_throttle
            )

        return controls

    def get_expected_metrics(self):
        """Expected metrics for level flight (should agree well)."""
        return {
            'position_rmse_threshold': 5.0,   # meters
            'attitude_rmse_threshold': 2.0,   # degrees
            'min_correlation': 0.98,
        }
