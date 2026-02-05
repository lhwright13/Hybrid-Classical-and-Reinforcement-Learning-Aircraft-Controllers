"""Interfaces for agent, aircraft, and sensor implementations.

This module provides abstract interfaces for:
- RLAgentInterface: RL/learning agents (observations -> commands)
- AircraftInterface: Aircraft simulation backends
- SensorInterface: Sensor models (perfect, noisy, etc.)
- AircraftRegistry: Multi-aircraft management
"""

from interfaces.agent import RLAgentInterface
from interfaces.aircraft import AircraftInterface
from interfaces.sensor import SensorInterface, PerfectSensorInterface, NoisySensorInterface
from interfaces.aircraft_registry import AircraftRegistry, AircraftStatus, AircraftInfo

__all__ = [
    "RLAgentInterface",
    "AircraftInterface",
    "SensorInterface",
    "PerfectSensorInterface",
    "NoisySensorInterface",
    "AircraftRegistry",
    "AircraftStatus",
    "AircraftInfo",
]
