# Abstraction Layers - Interface Contracts for Swappability

## Overview

This document defines **all interface contracts** in the system. These abstractions enable:
- Swapping simulation ↔ hardware backends
- Swapping classical ↔ RL ↔ hybrid agents
- Swapping sensor implementations
- Zero code changes when switching implementations

**Critical Principle**: Code to interfaces, not implementations.

## Core Interfaces

### 1. BaseAgent Interface

The fundamental contract that ALL agents must implement.

```python
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from controllers.types import (
    AircraftState,
    ControlMode,
    ControlCommand,
    Transition,
)


class BaseAgent(ABC):
    """Abstract base class for all agent types.

    All agents (classical, RL, hybrid, human-in-loop) must implement this interface.
    """

    @abstractmethod
    def get_control_level(self) -> ControlMode:
        """Return which control level this agent commands at.

        Returns:
            ControlMode: One of WAYPOINT, HSA, STICK_THROTTLE, SURFACE
        """
        pass

    @abstractmethod
    def reset(self, initial_state: AircraftState) -> None:
        """Reset agent to initial state.

        Called at the beginning of each episode.

        Args:
            initial_state: Initial aircraft state
        """
        pass

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> ControlCommand:
        """Compute action given observation.

        Args:
            observation: State observation (format depends on control level)

        Returns:
            ControlCommand with fields populated for the agent's control level
        """
        pass

    def update(self, transition: Transition) -> None:
        """Update agent with transition (optional, for learning agents).

        Args:
            transition: (state, action, reward, next_state, done)
        """
        pass  # Optional: only learning agents need this

    def save(self, path: str) -> None:
        """Save agent state/policy to disk.

        Args:
            path: File path to save to
        """
        pass  # Optional

    def load(self, path: str) -> None:
        """Load agent state/policy from disk.

        Args:
            path: File path to load from
        """
        pass  # Optional

    # Optional multi-level support
    def switch_control_level(self, level: ControlMode) -> None:
        """Switch control level (for adaptive agents).

        Args:
            level: New control level to operate at
        """
        raise NotImplementedError("Agent does not support level switching")

    def get_observation_space(self, level: ControlMode) -> dict:
        """Get observation space definition for a control level.

        Returns:
            dict with 'shape', 'low', 'high' keys
        """
        raise NotImplementedError("Agent must define observation space")

    def get_action_space(self, level: ControlMode) -> dict:
        """Get action space definition for a control level.

        Returns:
            dict with 'shape', 'low', 'high' keys
        """
        raise NotImplementedError("Agent must define action space")
```

### 2. AircraftInterface

Unified interface for all aircraft backends (sim or real).

```python
from abc import ABC, abstractmethod
from typing import Optional
from controllers.types import AircraftState, ControlSurfaces


class AircraftInterface(ABC):
    """Abstract interface for aircraft backends.

    Implementations:
    - SimulationAircraftBackend (JSBSim, etc.)
    - HardwareAircraftBackend (Teensy/dRehmFlight)
    """

    @abstractmethod
    def step(self, dt: float) -> AircraftState:
        """Advance simulation/hardware by dt seconds.

        Args:
            dt: Time step in seconds

        Returns:
            Updated aircraft state
        """
        pass

    @abstractmethod
    def set_controls(self, surfaces: ControlSurfaces) -> None:
        """Set control surface commands.

        Args:
            surfaces: Control surface deflections and throttle
        """
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[AircraftState] = None) -> AircraftState:
        """Reset aircraft to initial state.

        Args:
            initial_state: Desired initial state (None = use default)

        Returns:
            Actual initial state after reset
        """
        pass

    @abstractmethod
    def get_state(self) -> AircraftState:
        """Get current aircraft state.

        Returns:
            Current full state
        """
        pass

    def close(self) -> None:
        """Clean up resources (close connections, files, etc.)."""
        pass  # Optional

    # Metadata
    @abstractmethod
    def get_backend_type(self) -> str:
        """Return backend type identifier.

        Returns:
            "simulation" or "hardware"
        """
        pass

    def get_dt_nominal(self) -> float:
        """Get nominal time step for this backend.

        Returns:
            Nominal dt in seconds
        """
        return 0.01  # Default 100 Hz
```

### 3. SensorInterface

Unified interface for sensor data (perfect sim, noisy sim, or real sensors).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class IMUData:
    """IMU sensor readings."""
    accel: np.ndarray  # [ax, ay, az] in body frame (m/s²)
    gyro: np.ndarray   # [p, q, r] in body frame (rad/s)
    mag: np.ndarray    # [mx, my, mz] magnetometer (normalized)
    timestamp: float   # seconds


@dataclass
class GPSData:
    """GPS sensor readings."""
    latitude: float    # degrees
    longitude: float   # degrees
    altitude: float    # meters
    velocity_ned: np.ndarray  # [vN, vE, vD] m/s
    fix_quality: int   # 0=no fix, 1=GPS, 2=DGPS, etc.
    timestamp: float   # seconds


@dataclass
class SensorReadings:
    """Combined sensor readings."""
    imu: IMUData
    gps: GPSData
    airspeed: float    # m/s
    altitude_agl: float  # meters above ground
    timestamp: float   # seconds


class SensorInterface(ABC):
    """Abstract interface for sensor systems."""

    @abstractmethod
    def get_imu_data(self) -> IMUData:
        """Get latest IMU data."""
        pass

    @abstractmethod
    def get_gps_data(self) -> GPSData:
        """Get latest GPS data."""
        pass

    @abstractmethod
    def get_sensor_readings(self) -> SensorReadings:
        """Get all sensor readings."""
        pass

    def calibrate(self) -> None:
        """Calibrate sensors (optional)."""
        pass
```

### 4. StateEstimatorInterface

Interface for state estimation (sensor fusion).

```python
from abc import ABC, abstractmethod


class StateEstimatorInterface(ABC):
    """Abstract interface for state estimators."""

    @abstractmethod
    def update(self, sensor_data: SensorReadings, dt: float) -> AircraftState:
        """Update state estimate with new sensor data.

        Args:
            sensor_data: Latest sensor readings
            dt: Time since last update

        Returns:
            Estimated aircraft state
        """
        pass

    @abstractmethod
    def reset(self, initial_state: AircraftState) -> None:
        """Reset estimator to initial state."""
        pass

    def predict(self, controls: ControlSurfaces, dt: float) -> AircraftState:
        """Predict state forward (optional, for EKF/UKF).

        Args:
            controls: Control inputs
            dt: Time step

        Returns:
            Predicted state
        """
        raise NotImplementedError("Estimator does not support prediction")
```

### 5. ControlLevelInterface

Interface for each control level implementation.

```python
from abc import ABC, abstractmethod


class ControlLevelInterface(ABC):
    """Abstract interface for control level implementations."""

    @abstractmethod
    def compute_command(
        self,
        state: AircraftState,
        input_command: ControlCommand,
        dt: float
    ) -> ControlCommand:
        """Process input command and produce output for next level.

        Args:
            state: Current aircraft state
            input_command: Command from agent or previous level
            dt: Time step

        Returns:
            Command for next control level
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state (integral terms, etc.)."""
        pass

    @abstractmethod
    def get_level(self) -> ControlMode:
        """Return which level this controller implements."""
        pass
```

## Observation & Action Space Definitions

### Observation Space by Control Level

```python
OBSERVATION_SPACES = {
    ControlMode.WAYPOINT: {
        "shape": (12,),
        "low": np.array([-np.inf] * 12),
        "high": np.array([np.inf] * 12),
        "names": [
            "position_error_north",
            "position_error_east",
            "position_error_down",
            "distance_to_waypoint",
            "heading_to_waypoint",
            "current_heading",
            "current_airspeed",
            "current_altitude",
            "velocity_north",
            "velocity_east",
            "velocity_down",
            "heading_error"
        ]
    },
    ControlMode.HSA: {
        "shape": (12,),
        "low": np.array([-np.inf] * 12),
        "high": np.array([np.inf] * 12),
        "names": [
            "current_heading",
            "current_airspeed",
            "current_altitude",
            "heading_error",
            "speed_error",
            "altitude_error",
            "roll",
            "pitch",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
            "vertical_speed"
        ]
    },
    ControlMode.STICK_THROTTLE: {
        "shape": (10,),
        "low": np.array([-np.pi, -np.pi, -np.pi, -10, -10, -10, 0, -10, -np.pi, -np.pi]),
        "high": np.array([np.pi, np.pi, np.pi, 10, 10, 10, 50, 10, np.pi, np.pi]),
        "names": [
            "roll",
            "pitch",
            "yaw",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
            "airspeed",
            "vertical_speed",
            "roll_error",
            "pitch_error"
        ]
    },
    ControlMode.SURFACE: {
        "shape": (14,),
        "low": np.array([-np.pi]*3 + [-10]*3 + [-50]*3 + [0] + [-1]*4),
        "high": np.array([np.pi]*3 + [10]*3 + [50]*3 + [50] + [1]*4),
        "names": [
            "roll", "pitch", "yaw",
            "p", "q", "r",
            "u", "v", "w",
            "airspeed",
            "elevator_pos",
            "aileron_pos",
            "rudder_pos",
            "throttle_pos"
        ]
    }
}
```

### Action Space by Control Level

```python
ACTION_SPACES = {
    ControlMode.WAYPOINT: {
        "shape": (4,),
        "low": np.array([-1000, -1000, -500, 10]),
        "high": np.array([1000, 1000, 0, 40]),
        "names": ["waypoint_north", "waypoint_east", "waypoint_down", "speed"]
    },
    ControlMode.HSA: {
        "shape": (3,),
        "low": np.array([0, 10, 0]),
        "high": np.array([2*np.pi, 40, 500]),
        "names": ["heading", "speed", "altitude"]
    },
    ControlMode.STICK_THROTTLE: {
        "shape": (4,),
        "low": np.array([-1, -1, -1, 0]),
        "high": np.array([1, 1, 1, 1]),
        "names": ["roll_stick", "pitch_stick", "yaw_stick", "throttle"]
    },
    ControlMode.SURFACE: {
        "shape": (4,),
        "low": np.array([-1, -1, -1, 0]),
        "high": np.array([1, 1, 1, 1]),
        "names": ["elevator", "aileron", "rudder", "throttle"]
    }
}
```

## Configuration Schema

### Agent Configuration

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for any agent."""
    agent_type: str  # "classical", "rl", "hybrid", "hierarchical"
    control_level: ControlMode
    model_path: Optional[str] = None  # For RL agents
    hyperparameters: Optional[dict] = None
    observation_normalization: bool = True
    action_clipping: bool = True


@dataclass
class BackendConfig:
    """Configuration for aircraft backend."""
    backend_type: str  # "simulation" or "hardware"
    # Simulation-specific
    simulator: Optional[str] = "jsbsim"  # "jsbsim", "simplified"
    aircraft_model: Optional[str] = "f16"
    # Hardware-specific
    serial_port: Optional[str] = "/dev/ttyUSB0"
    baudrate: Optional[int] = 115200
    # Common
    dt: float = 0.01  # Time step
    sensor_noise: bool = False


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    agent: AgentConfig
    backend: BackendConfig
    control_frequency: float = 100.0  # Hz
    logging_enabled: bool = True
    visualization_enabled: bool = False
    seed: int = 42  # For reproducibility
```

## Interface Versioning

All interfaces include version numbers for backward compatibility.

```python
class BaseAgent(ABC):
    INTERFACE_VERSION = "1.0.0"

    def get_interface_version(self) -> str:
        return self.INTERFACE_VERSION


class AircraftInterface(ABC):
    INTERFACE_VERSION = "1.0.0"

    def get_interface_version(self) -> str:
        return self.INTERFACE_VERSION
```

## Error Handling Contracts

### Error Types

```python
class InterfaceError(Exception):
    """Base exception for interface errors."""
    pass


class AgentError(InterfaceError):
    """Agent-related errors."""
    pass


class BackendError(InterfaceError):
    """Backend-related errors."""
    pass


class SensorError(InterfaceError):
    """Sensor-related errors."""
    pass


class ConfigurationError(InterfaceError):
    """Configuration validation errors."""
    pass
```

### Error Handling Protocol

```python
from enum import Enum


class ErrorSeverity(Enum):
    WARNING = 1   # Log warning, continue
    ERROR = 2     # Attempt recovery, fallback
    CRITICAL = 3  # Emergency shutdown


@dataclass
class ErrorReport:
    """Standardized error report."""
    severity: ErrorSeverity
    component: str  # "agent", "backend", "sensor", etc.
    message: str
    timestamp: float
    traceback: Optional[str] = None
```

## Timing and Synchronization Contracts

### Timing Interface

```python
class TimingInterface(ABC):
    """Interface for time management."""

    @abstractmethod
    def get_time(self) -> float:
        """Get current time in seconds."""
        pass

    @abstractmethod
    def sleep_until(self, target_time: float) -> None:
        """Sleep until target time."""
        pass

    @abstractmethod
    def is_realtime(self) -> bool:
        """Return True if running in real-time, False if as-fast-as-possible."""
        pass
```

### Synchronization Protocol

All components must adhere to timing contracts:

```python
class ComponentTiming:
    """Timing requirements for a component."""
    update_rate: float  # Hz
    max_latency: float  # seconds
    priority: int       # 0=highest, lower=less critical
```

## Data Type Contracts

All data types are defined in `controllers/types.py` (already created).

### Type Validation

```python
def validate_aircraft_state(state: AircraftState) -> bool:
    """Validate aircraft state is physically reasonable."""
    checks = [
        np.all(np.isfinite(state.position)),
        np.all(np.isfinite(state.velocity)),
        np.all(np.isfinite(state.attitude)),
        -np.pi <= state.roll <= np.pi,
        -np.pi/2 <= state.pitch <= np.pi/2,
        state.airspeed >= 0,
    ]
    return all(checks)


def validate_control_surfaces(surfaces: ControlSurfaces) -> bool:
    """Validate control surfaces are in valid range."""
    checks = [
        -1.0 <= surfaces.elevator <= 1.0,
        -1.0 <= surfaces.aileron <= 1.0,
        -1.0 <= surfaces.rudder <= 1.0,
        0.0 <= surfaces.throttle <= 1.0,
    ]
    return all(checks)
```

## Swappability Examples

### Example 1: Swap Simulation → Hardware

```python
# Configuration change ONLY
config_sim = BackendConfig(backend_type="simulation", simulator="jsbsim")
config_hw = BackendConfig(backend_type="hardware", serial_port="/dev/ttyUSB0")

# Agent code is IDENTICAL
from interfaces.aircraft_interface import create_aircraft_backend

backend = create_aircraft_backend(config_hw)  # Was: config_sim
agent = load_agent(agent_config)

# Same control loop
while True:
    state = backend.get_state()
    action = agent.get_action(state)
    backend.set_controls(action)
    backend.step(dt)
```

### Example 2: Swap Classical → RL Agent

```python
# Configuration change ONLY
config_classical = AgentConfig(
    agent_type="classical",
    control_level=ControlMode.STICK_THROTTLE
)
config_rl = AgentConfig(
    agent_type="rl",
    control_level=ControlMode.STICK_THROTTLE,
    model_path="trained_models/ppo_stick.zip"
)

# Backend code is IDENTICAL
from interfaces.agent_interface import create_agent

agent = create_agent(config_rl)  # Was: config_classical
backend = create_aircraft_backend(backend_config)

# Same control loop
while True:
    state = backend.get_state()
    action = agent.get_action(state)
    backend.set_controls(action)
    backend.step(dt)
```

## Interface Factories

Factory functions ensure correct instantiation.

```python
def create_agent(config: AgentConfig) -> BaseAgent:
    """Factory for creating agents."""
    if config.agent_type == "classical":
        from agents.classical_agent import ClassicalAgent
        return ClassicalAgent(config)
    elif config.agent_type == "rl":
        from agents.rl_agent import RLAgent
        return RLAgent(config)
    elif config.agent_type == "hybrid":
        from agents.hybrid_agent import HybridAgent
        return HybridAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def create_aircraft_backend(config: BackendConfig) -> AircraftInterface:
    """Factory for creating aircraft backends."""
    if config.backend_type == "simulation":
        if config.simulator == "jsbsim":
            from simulation.jsbsim_backend import JSBSimBackend
            return JSBSimBackend(config)
        elif config.simulator == "simplified":
            from simulation.simplified_backend import SimplifiedBackend
            return SimplifiedBackend(config)
    elif config.backend_type == "hardware":
        from hardware.teensy_backend import TeensyBackend
        return TeensyBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {config.backend_type}")
```

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-10-09
**Related Documents**:
- 01_ARCHITECTURE.md (system architecture)
- 05_AGENT_INTERFACE_CONTROL.md (agent integration guide)
- controllers/types.py (data type definitions)
