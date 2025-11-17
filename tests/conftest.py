"""Pytest fixtures for test suite.

This module provides common fixtures to eliminate code duplication across tests.
"""

import numpy as np
import pytest

from controllers.types import ControllerConfig, AircraftState
from simulation import SimulationAircraftBackend


@pytest.fixture
def default_config() -> ControllerConfig:
    """Create default controller configuration for testing.

    These gains are tuned for the simplified 6-DOF RC plane model.

    Returns:
        ControllerConfig with default gains and limits
    """
    return ControllerConfig(
        # Rate PID gains (inner loop)
        roll_rate_gains=ControllerConfig().roll_rate_gains,
        pitch_rate_gains=ControllerConfig().pitch_rate_gains,

        # Angle PID gains (outer loop)
        roll_angle_gains=ControllerConfig().roll_angle_gains,
        pitch_angle_gains=ControllerConfig().pitch_angle_gains,

        # Yaw gains
        yaw_gains=ControllerConfig().yaw_gains,

        # Limits
        max_roll=30.0,
        max_pitch=30.0,
        max_roll_rate=180.0,
        max_pitch_rate=180.0,
        max_yaw_rate=160.0,

        dt=0.01  # 100 Hz
    )


@pytest.fixture
def aircraft_backend():
    """Create default aircraft simulation backend.

    Returns:
        SimulationAircraftBackend configured for RC plane
    """
    return SimulationAircraftBackend({'aircraft_type': 'rc_plane'})


@pytest.fixture
def level_flight_state() -> AircraftState:
    """Create standard level flight state for testing.

    Standard conditions:
    - Altitude: 100m
    - Airspeed: 20 m/s
    - Heading: 0° (North)
    - Level attitude (zero roll, pitch, yaw)
    - Zero angular rates

    Returns:
        AircraftState in stable level flight
    """
    return AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),  # NED frame (z=-100 = 100m altitude)
        velocity=np.array([20.0, 0.0, 0.0]),    # 20 m/s forward in body frame
        attitude=np.zeros(3),                    # Level attitude (roll=0, pitch=0, yaw=0)
        angular_rate=np.zeros(3),                # No rotation
        airspeed=20.0,
        altitude=100.0,
        heading=0.0                              # Heading North (0° in NED convention)
    )
