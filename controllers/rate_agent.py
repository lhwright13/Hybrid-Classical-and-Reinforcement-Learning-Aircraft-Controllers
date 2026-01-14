"""Level 4: Rate Control Agent - Inner loop rate control."""

import numpy as np
import logging
from controllers.base_agent import BaseAgent
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)

# Import C++ PID controller
import aircraft_controls_bindings as acb
from controllers.utils import create_pid_config, validate_command

logger = logging.getLogger(__name__)


class RateAgent(BaseAgent):
    """Level 4: Rate mode controller (inner loop).

    Commands angular rates (p, q, r), outputs surface deflections.
    Uses high-frequency C++ PID controllers for tight control.

    This is the inner loop of the cascaded controller architecture:
    - Level 3 (Attitude) commands desired rates
    - Level 4 (Rate) tracks those rates → outputs surfaces

    Use cases:
    - Acrobatic flight (direct rate control)
    - Aggressive maneuvers (flips, rolls)
    - Expert piloting (RC "acro mode")
    - Inner loop for Level 3 angle mode
    """

    def __init__(self, config: ControllerConfig):
        """Initialize rate agent with PID controllers.

        Args:
            config: Controller configuration with rate PID gains
        """
        self.config = config

        # Create C++ PID configurations for rate control
        roll_rate_config = create_pid_config(config.roll_rate_gains)
        pitch_rate_config = create_pid_config(config.pitch_rate_gains)
        yaw_rate_config = create_pid_config(config.yaw_gains)

        # Create multi-axis PID controller (C++)
        self.rate_controller = acb.MultiAxisPIDController(
            roll_rate_config,
            pitch_rate_config,
            yaw_rate_config
        )

        # Rate limits (rad/s)
        self.max_roll_rate = np.radians(config.max_roll_rate)  # Convert deg/s to rad/s
        self.max_pitch_rate = np.radians(config.max_pitch_rate)
        self.max_yaw_rate = np.radians(config.max_yaw_rate)

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.RATE
        """
        return ControlMode.RATE

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = None
    ) -> ControlSurfaces:
        """Compute surface deflections from rate commands.

        Args:
            command: Rate control command (p, q, r desired)
            state: Current aircraft state
            dt: Time step in seconds. If None, uses config.rate_loop_dt.
                IMPORTANT: Pass the actual simulation dt for correct PID behavior.

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not RATE
        """
        assert command.mode == ControlMode.RATE, \
            f"Rate agent expects RATE mode, got {command.mode}"

        # Validate required command fields
        validate_command(command, "RATE", ["roll_rate", "pitch_rate", "yaw_rate"])

        # Rate setpoint (clip to rate limits)
        p_cmd = np.clip(command.roll_rate, -self.max_roll_rate, self.max_roll_rate)
        q_cmd = np.clip(command.pitch_rate, -self.max_pitch_rate, self.max_pitch_rate)
        r_cmd = np.clip(command.yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        rate_setpoint = acb.Vector3(p_cmd, q_cmd, r_cmd)

        # Current rate measurement
        rate_measurement = acb.Vector3(state.p, state.q, state.r)

        # PID control (C++ - high frequency for tight control)
        # Use provided dt or fall back to configured rate_loop_dt
        actual_dt = dt if dt is not None else self.config.rate_loop_dt
        output = self.rate_controller.compute(rate_setpoint, rate_measurement, actual_dt)

        # Map PID output to surfaces
        # Output: roll → aileron, pitch → elevator, yaw → rudder
        # Note: Elevator sign is inverted because positive elevator = nose down in aero convention
        surfaces = ControlSurfaces(
            aileron=output.roll,     # p (roll rate) → aileron
            elevator=-output.pitch,  # q (pitch rate) → elevator (INVERTED)
            rudder=output.yaw,       # r (yaw rate) → rudder
            throttle=command.throttle  # Pass through
        )

        # Already saturated by PID output limits, but double-check
        surfaces.aileron = np.clip(surfaces.aileron, -1.0, 1.0)
        surfaces.elevator = np.clip(surfaces.elevator, -1.0, 1.0)
        surfaces.rudder = np.clip(surfaces.rudder, -1.0, 1.0)
        surfaces.throttle = np.clip(surfaces.throttle, 0.0, 1.0)

        return surfaces

    def reset(self):
        """Reset PID controllers."""
        self.rate_controller.reset()

    def __repr__(self) -> str:
        """String representation."""
        roll_gains = self.rate_controller.get_gains(0)
        pitch_gains = self.rate_controller.get_gains(1)
        yaw_gains = self.rate_controller.get_gains(2)
        return (f"RateAgent(roll_kp={roll_gains.kp:.3f}, "
                f"pitch_kp={pitch_gains.kp:.3f}, "
                f"yaw_kp={yaw_gains.kp:.3f})")
