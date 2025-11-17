"""Level 3: Attitude Control Agent - Angle mode (outer loop)."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.rate_agent import RateAgent
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)

# Import C++ PID controller
import aircraft_controls_bindings as acb
from controllers.utils.pid_utils import create_pid_config


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π].

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-π, π]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AttitudeAgent(BaseAgent):
    """Level 3: Attitude controller (angle mode, outer loop).

    Commands attitude angles (roll, pitch, yaw), outputs rate commands.
    Forms the outer loop of cascaded controller with Level 4 (rate) as inner loop.

    Architecture:
        Level 3 (Attitude): angle error → PID → rate command
        Level 4 (Rate): rate error → PID → surface deflection

    Use cases:
    - Stabilized flight (hold attitude)
    - Smooth maneuvers
    - Formation flight (maintain relative attitude)
    - Beginner-friendly control
    """

    def __init__(self, config: ControllerConfig):
        """Initialize attitude agent with cascaded PIDs.

        Args:
            config: Controller configuration with angle and rate PID gains
        """
        self.config = config

        # Create C++ PID configurations for angle control (outer loop)
        # Output is rate command (rad/s), use config rate limits
        roll_angle_config = create_pid_config(
            config.roll_angle_gains,
            output_min=-np.radians(config.max_roll_rate),
            output_max=np.radians(config.max_roll_rate)
        )
        pitch_angle_config = create_pid_config(
            config.pitch_angle_gains,
            output_min=-np.radians(config.max_pitch_rate),
            output_max=np.radians(config.max_pitch_rate)
        )
        yaw_angle_config = create_pid_config(
            config.yaw_gains,
            output_min=-np.radians(config.max_yaw_rate),
            output_max=np.radians(config.max_yaw_rate)
        )

        # Create multi-axis PID controller for angles (C++)
        self.angle_controller = acb.MultiAxisPIDController(
            roll_angle_config,
            pitch_angle_config,
            yaw_angle_config
        )

        # Inner loop: Rate controller (Level 4)
        self.rate_agent = RateAgent(config)

        # Angle limits (rad)
        self.max_roll = np.radians(config.max_roll)
        self.max_pitch = np.radians(config.max_pitch)

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.ATTITUDE
        """
        return ControlMode.ATTITUDE

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = 0.01
    ) -> ControlSurfaces:
        """Compute surfaces from angle commands via cascaded control.

        Cascaded control flow:
        1. Angle error → PID → rate command (outer loop)
        2. Rate command → Level 4 → surface deflection (inner loop)

        Args:
            command: Attitude angle command (roll, pitch, yaw angles)
            state: Current aircraft state
            dt: Time step in seconds (default 0.01 for 100 Hz outer loop)

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not ATTITUDE
        """
        assert command.mode == ControlMode.ATTITUDE, \
            f"Attitude agent expects ATTITUDE mode, got {command.mode}"

        # Clip angle commands to limits
        roll_cmd = np.clip(command.roll_angle, -self.max_roll, self.max_roll)
        pitch_cmd = np.clip(command.pitch_angle, -self.max_pitch, self.max_pitch)
        yaw_cmd = wrap_angle(command.yaw_angle) if command.yaw_angle is not None else 0.0  # Wrap yaw to [-π, π]

        # Angle setpoint
        angle_setpoint = acb.Vector3(roll_cmd, pitch_cmd, yaw_cmd)

        # Current angle measurement (wrap yaw to [-π, π])
        current_roll = state.roll
        current_pitch = state.pitch
        current_yaw = wrap_angle(state.yaw)

        angle_measurement = acb.Vector3(current_roll, current_pitch, current_yaw)

        # Outer loop: angles → rate commands (PID)
        # Run at slower rate than inner loop (100 Hz vs 1000 Hz)
        # dt is now passed as parameter (default 0.01 for 100 Hz)
        rate_output = self.angle_controller.compute(angle_setpoint, angle_measurement, dt)

        # Limit rate commands using config values (shouldn't exceed these with proper gains)
        p_cmd = np.clip(rate_output.roll, -np.radians(self.config.max_roll_rate), np.radians(self.config.max_roll_rate))
        q_cmd = np.clip(rate_output.pitch, -np.radians(self.config.max_pitch_rate), np.radians(self.config.max_pitch_rate))
        r_cmd = np.clip(rate_output.yaw, -np.radians(self.config.max_yaw_rate), np.radians(self.config.max_yaw_rate))

        # Create rate command for Level 4 (inner loop)
        rate_cmd = ControlCommand(
            mode=ControlMode.RATE,
            roll_rate=p_cmd,
            pitch_rate=q_cmd,
            yaw_rate=r_cmd,
            throttle=command.throttle
        )

        # Inner loop: rate commands → surfaces (Level 4)
        surfaces = self.rate_agent.compute_action(rate_cmd, state)

        return surfaces

    def reset(self):
        """Reset PID controllers in both loops."""
        self.angle_controller.reset()
        self.rate_agent.reset()

    def __repr__(self) -> str:
        """String representation."""
        roll_gains = self.angle_controller.get_gains(0)
        pitch_gains = self.angle_controller.get_gains(1)
        yaw_gains = self.angle_controller.get_gains(2)
        return (f"AttitudeAgent(outer: roll_kp={roll_gains.kp:.3f}, "
                f"pitch_kp={pitch_gains.kp:.3f}, yaw_kp={yaw_gains.kp:.3f}, "
                f"inner: {self.rate_agent})")
