"""Level 2: HSA Control Agent - Heading, Speed, Altitude control."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.attitude_agent import AttitudeAgent, wrap_angle
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)

# Import C++ PID controller
import aircraft_controls_bindings as acb


class HSAAgent(BaseAgent):
    """Level 2: HSA (Heading/Speed/Altitude) controller.

    Commands HSA state variables, outputs attitude commands to Level 3.

    HSA Control:
    - Heading: Yaw angle to point aircraft in desired direction
    - Speed: Airspeed control via throttle
    - Altitude: Height above ground

    Architecture:
        HSA error → PID → attitude command → Level 3 → surfaces

    Use cases:
    - Formation flight (maintain relative HSA)
    - Loitering (circle at constant altitude)
    - Area coverage patterns
    - Terrain following
    - Traffic avoidance (altitude separation)
    """

    def __init__(self, config: ControllerConfig):
        """Initialize HSA agent with PID controllers.

        Args:
            config: Controller configuration
        """
        self.config = config

        # HSA controllers (simple PIDs in Python - not performance critical)
        # Heading controller
        heading_config = acb.PIDConfig()
        heading_config.gains = acb.PIDGains(kp=2.0, ki=0.1, kd=0.5)
        heading_config.integral_min = -10.0
        heading_config.integral_max = 10.0
        heading_config.output_min = -np.radians(30)  # Max roll command
        heading_config.output_max = np.radians(30)
        self.heading_pid = acb.PIDController(heading_config)

        # Speed controller
        speed_config = acb.PIDConfig()
        speed_config.gains = acb.PIDGains(kp=0.1, ki=0.01, kd=0.0)
        speed_config.integral_min = -10.0
        speed_config.integral_max = 10.0
        speed_config.output_min = -0.5  # Throttle adjustment
        speed_config.output_max = 0.5
        self.speed_pid = acb.PIDController(speed_config)

        # Altitude controller
        altitude_config = acb.PIDConfig()
        altitude_config.gains = acb.PIDGains(kp=0.05, ki=0.01, kd=0.2)
        altitude_config.integral_min = -10.0
        altitude_config.integral_max = 10.0
        altitude_config.output_min = -np.radians(15)  # Max pitch command
        altitude_config.output_max = np.radians(15)
        self.altitude_pid = acb.PIDController(altitude_config)

        # Inner loop: Attitude controller (Level 3)
        self.attitude_agent = AttitudeAgent(config)

        # Baseline throttle
        self.baseline_throttle = 0.5

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.HSA
        """
        return ControlMode.HSA

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = 0.01
    ) -> ControlSurfaces:
        """Compute surfaces from HSA commands.

        Control strategy:
        - Heading → roll angle (coordinated turn)
        - Altitude → pitch angle (climb/descend)
        - Speed → throttle

        Args:
            command: HSA control command
            state: Current aircraft state
            dt: Time step in seconds (default 0.01 for 100 Hz)

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not HSA
        """
        assert command.mode == ControlMode.HSA, \
            f"HSA agent expects HSA mode, got {command.mode}"

        # === Heading Control ===
        # Compute heading error (wrap to [-π, π])
        heading_error = wrap_angle(command.heading - state.heading)

        # Heading PID → roll angle (coordinated turn)
        # For small bank angles: roll ≈ (V²/g) * yaw_rate
        # Simplified: use heading error to command roll
        roll_angle = self.heading_pid.compute(command.heading, state.heading, dt)

        # Limit roll angle
        roll_angle = np.clip(roll_angle, -np.radians(30), np.radians(30))

        # === Altitude Control ===
        # Altitude error → pitch angle
        # PID on altitude → desired climb rate → pitch angle
        altitude_error = command.altitude - state.altitude
        pitch_angle = self.altitude_pid.compute(command.altitude, state.altitude, dt)

        # Limit pitch angle
        pitch_angle = np.clip(pitch_angle, -np.radians(15), np.radians(15))

        # === Speed Control ===
        # Speed error → throttle
        speed_error = command.speed - state.airspeed
        throttle_adjustment = self.speed_pid.compute(command.speed, state.airspeed, dt)

        throttle = self.baseline_throttle + throttle_adjustment
        throttle = np.clip(throttle, 0.0, 1.0)

        # === Yaw Coordination ===
        # For coordinated turn, yaw angle should match heading
        # Wrap to [-π, π] to avoid large angle errors in Level 3
        yaw_angle = wrap_angle(command.heading)

        # Create attitude command (Level 3)
        attitude_cmd = ControlCommand(
            mode=ControlMode.ATTITUDE,
            roll_angle=roll_angle,
            pitch_angle=pitch_angle,
            yaw_angle=yaw_angle,
            throttle=throttle
        )

        # Inner loop: attitude → surfaces (Level 3)
        # Pass dt to inner loop for consistent timing
        surfaces = self.attitude_agent.compute_action(attitude_cmd, state, dt)

        return surfaces

    def reset(self):
        """Reset PID controllers."""
        self.heading_pid.reset()
        self.speed_pid.reset()
        self.altitude_pid.reset()
        self.attitude_agent.reset()

    def __repr__(self) -> str:
        """String representation."""
        return (f"HSAAgent(heading_kp={self.heading_pid.get_gains().kp:.3f}, "
                f"speed_kp={self.speed_pid.get_gains().kp:.3f}, "
                f"altitude_kp={self.altitude_pid.get_gains().kp:.3f})")
