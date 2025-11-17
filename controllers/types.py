"""Common data types for the flight controller system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ControlMode(Enum):
    """Flight controller command modes.

    5-level control hierarchy:
    - Level 1: WAYPOINT - Navigate to waypoint coordinates
    - Level 2: HSA - Heading, Speed, Altitude control
    - Level 3: ATTITUDE - Angle mode (roll/pitch/yaw angles)
    - Level 4: RATE - Rate mode (p/q/r angular velocities)
    - Level 5: SURFACE - Direct control surface deflection
    """

    WAYPOINT = 1  # Level 1: Navigate to waypoint coordinates
    HSA = 2  # Level 2: Heading, Speed, Altitude control
    ATTITUDE = 3  # Level 3: Angle mode (roll/pitch/yaw angles)
    RATE = 4  # Level 4: Rate mode (p/q/r angular velocities)
    SURFACE = 5  # Level 5: Direct control surface deflection


@dataclass
class AircraftState:
    """Complete aircraft state vector.

    Attributes:
        time: Simulation time (seconds)
        position: Position in NED frame [N, E, D] (meters)
        velocity: Velocity in body frame [u, v, w] (m/s)
        attitude: Euler angles [roll, pitch, yaw] (radians)
        angular_rate: Angular rates in body frame [p, q, r] (rad/s)
        airspeed: True airspeed (m/s)
        altitude: Altitude above ground (meters, positive up)
        ground_speed: Ground speed (m/s)
        heading: Heading angle (radians, 0=North, clockwise positive)
    """

    time: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_rate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    airspeed: float = 0.0
    altitude: float = 0.0
    ground_speed: float = 0.0
    heading: float = 0.0

    @property
    def roll(self) -> float:
        """Roll angle in radians."""
        return self.attitude[0]

    @property
    def pitch(self) -> float:
        """Pitch angle in radians."""
        return self.attitude[1]

    @property
    def yaw(self) -> float:
        """Yaw angle in radians."""
        return self.attitude[2]

    @property
    def p(self) -> float:
        """Roll rate in rad/s."""
        return self.angular_rate[0]

    @property
    def q(self) -> float:
        """Pitch rate in rad/s."""
        return self.angular_rate[1]

    @property
    def r(self) -> float:
        """Yaw rate in rad/s."""
        return self.angular_rate[2]

    @property
    def north(self) -> float:
        """North position in meters."""
        return self.position[0]

    @property
    def east(self) -> float:
        """East position in meters."""
        return self.position[1]

    @property
    def down(self) -> float:
        """Down position in meters (negative of altitude)."""
        return self.position[2]


@dataclass
class Waypoint:
    """3D waypoint for navigation.

    Attributes:
        north: North coordinate (meters)
        east: East coordinate (meters)
        down: Down coordinate (meters, negative of altitude)
        speed: Desired airspeed at waypoint (m/s), None for current speed
        heading: Desired heading at waypoint (radians), None for auto
    """

    north: float
    east: float
    down: float
    speed: Optional[float] = None
    heading: Optional[float] = None

    @property
    def altitude(self) -> float:
        """Altitude (positive up) in meters."""
        return -self.down

    @classmethod
    def from_ned(cls, north: float, east: float, down: float, **kwargs):
        """Create waypoint from NED coordinates."""
        return cls(north=north, east=east, down=down, **kwargs)

    @classmethod
    def from_altitude(
        cls, north: float, east: float, altitude: float, **kwargs
    ):
        """Create waypoint from altitude (positive up)."""
        return cls(north=north, east=east, down=-altitude, **kwargs)


@dataclass
class ControlCommand:
    """Control command in any of the 5 modes.

    Only fields relevant to the current mode should be populated.
    """

    mode: ControlMode
    timestamp: float = 0.0

    # Level 1: Waypoint
    waypoint: Optional[Waypoint] = None

    # Level 2: HSA (Heading, Speed, Altitude)
    heading: Optional[float] = None  # radians
    speed: Optional[float] = None  # m/s
    altitude: Optional[float] = None  # meters

    # Level 3: Attitude (Angle mode - roll/pitch/yaw angles)
    roll_angle: Optional[float] = None  # radians
    pitch_angle: Optional[float] = None  # radians
    yaw_angle: Optional[float] = None  # radians

    # Level 4: Rate (p/q/r angular velocities)
    roll_rate: Optional[float] = None  # rad/s
    pitch_rate: Optional[float] = None  # rad/s
    yaw_rate: Optional[float] = None  # rad/s

    # Throttle (shared across levels 3-5)
    throttle: Optional[float] = None  # 0 to 1

    # Level 5: Surface Deflection (normalized -1 to 1, except throttle 0 to 1)
    elevator: Optional[float] = None
    aileron: Optional[float] = None
    rudder: Optional[float] = None
    # throttle is shared with other levels


@dataclass
class ControlSurfaces:
    """Control surface deflections.

    All surfaces normalized to -1 to 1 range (except throttle 0 to 1).
    Positive conventions:
    - elevator: nose up
    - aileron: right wing down
    - rudder: nose right
    - throttle: 0=idle, 1=full
    """

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [elevator, aileron, rudder, throttle]."""
        return np.array(
            [self.elevator, self.aileron, self.rudder, self.throttle]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create from numpy array [elevator, aileron, rudder, throttle]."""
        return cls(
            elevator=arr[0], aileron=arr[1], rudder=arr[2], throttle=arr[3]
        )


@dataclass
class PIDState:
    """PID controller state for one axis.

    Attributes:
        error: Current error
        integral: Accumulated integral term
        derivative: Derivative term
        output: PID output
        setpoint: Desired setpoint
        measurement: Current measurement
    """

    error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    output: float = 0.0
    setpoint: float = 0.0
    measurement: float = 0.0


@dataclass
class Telemetry:
    """Telemetry data for logging and visualization.

    Contains aircraft state plus control outputs and performance metrics.
    """

    # State
    state: AircraftState = field(default_factory=AircraftState)

    # Control outputs
    surfaces: ControlSurfaces = field(default_factory=ControlSurfaces)

    # PID states (for debugging/tuning)
    roll_pid: Optional[PIDState] = None
    pitch_pid: Optional[PIDState] = None
    yaw_pid: Optional[PIDState] = None

    # Performance metrics
    tracking_error: Optional[np.ndarray] = None  # Error from desired state
    control_effort: float = 0.0  # Total control effort

    # Mode and command
    mode: Optional[ControlMode] = None
    command: Optional[ControlCommand] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "time": self.state.time,
            "north": self.state.north,
            "east": self.state.east,
            "down": self.state.down,
            "altitude": self.state.altitude,
            "roll": self.state.roll,
            "pitch": self.state.pitch,
            "yaw": self.state.yaw,
            "p": self.state.p,
            "q": self.state.q,
            "r": self.state.r,
            "airspeed": self.state.airspeed,
            "ground_speed": self.state.ground_speed,
            "heading": self.state.heading,
            "elevator": self.surfaces.elevator,
            "aileron": self.surfaces.aileron,
            "rudder": self.surfaces.rudder,
            "throttle": self.surfaces.throttle,
            "mode": self.mode.name if self.mode else None,
            "control_effort": self.control_effort,
        }


@dataclass
class PIDGains:
    """PID controller gains.

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        i_limit: Integral windup limit
    """

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_limit: float = 25.0


@dataclass
class ControllerConfig:
    """Flight controller configuration.

    All PID gains and limits for the complete controller.
    """

    # Angle mode PID gains
    roll_angle_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.2, ki=0.3, kd=0.05)
    )
    pitch_angle_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.2, ki=0.3, kd=0.05)
    )

    # Rate mode PID gains
    roll_rate_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.15, ki=0.2, kd=0.0002)
    )
    pitch_rate_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.15, ki=0.2, kd=0.0002)
    )

    # Yaw PID gains
    yaw_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.3, ki=0.05, kd=0.00015)
    )

    # HSA (Heading/Speed/Altitude) PID gains
    heading_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=1.2, ki=0.08, kd=0.3)
    )
    speed_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.1, ki=0.01, kd=0.0)
    )
    altitude_gains: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.25, ki=0.02, kd=0.15)
    )

    # Limits
    max_roll: float = 30.0  # degrees for angle mode
    max_pitch: float = 30.0  # degrees for angle mode
    max_roll_rate: float = 180.0  # deg/s for rate mode
    max_pitch_rate: float = 180.0  # deg/s for rate mode
    max_yaw_rate: float = 160.0  # deg/s

    # Control loop rate
    dt: float = 0.01  # 100 Hz default
