"""Flight controller modules - Multi-level control hierarchy."""

# Types
from controllers.types import (
    ControlMode,
    AircraftState,
    Waypoint,
    ControlCommand,
    ControlSurfaces,
    PIDState,
    Telemetry,
    PIDGains,
    ControllerConfig,
)

# Agents (5-level hierarchy)
from controllers.base_agent import BaseAgent
from controllers.surface_agent import SurfaceAgent
from controllers.rate_agent import RateAgent
from controllers.attitude_agent import AttitudeAgent
from controllers.hsa_agent import HSAAgent
from controllers.waypoint_agent import WaypointAgent

# Mission planning
from controllers.mission_planner import MissionPlanner, MissionState

__all__ = [
    # Types
    "ControlMode",
    "AircraftState",
    "Waypoint",
    "ControlCommand",
    "ControlSurfaces",
    "PIDState",
    "Telemetry",
    "PIDGains",
    "ControllerConfig",
    # Agents
    "BaseAgent",
    "SurfaceAgent",      # Level 5: Direct surface control
    "RateAgent",         # Level 4: Rate control (inner loop)
    "AttitudeAgent",     # Level 3: Attitude control (outer loop)
    "HSAAgent",          # Level 2: HSA control
    "WaypointAgent",     # Level 1: Waypoint navigation
    # Mission planning
    "MissionPlanner",
    "MissionState",
]
