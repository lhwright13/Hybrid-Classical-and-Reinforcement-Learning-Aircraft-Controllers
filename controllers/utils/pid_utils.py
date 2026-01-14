"""Utility functions for PID controller configuration and common math operations.

This module provides helper functions to eliminate duplicate PID configuration
code across controller implementations, as well as common mathematical utilities
and input validation.
"""

import numpy as np
import logging
import aircraft_controls_bindings as acb
from controllers.types import PIDGains, ControlCommand

logger = logging.getLogger(__name__)


def create_pid_config(
    gains: PIDGains,
    output_min: float = -1.0,
    output_max: float = 1.0
) -> acb.PIDConfig:
    """Create C++ PID configuration from Python gains.

    This helper function consolidates the repeated pattern of creating
    PID configurations with integral limits and output saturation.

    Args:
        gains: PIDGains object with kp, ki, kd, i_limit
        output_min: Minimum output value (default: -1.0)
        output_max: Maximum output value (default: 1.0)

    Returns:
        Configured acb.PIDConfig object ready for use

    Example:
        >>> from controllers.types import PIDGains
        >>> gains = PIDGains(kp=0.15, ki=0.2, kd=0.0002, i_limit=0.5)
        >>> config = create_pid_config(gains)
        >>> controller = acb.PIDController(config)
    """
    config = acb.PIDConfig()
    config.gains = acb.PIDGains(gains.kp, gains.ki, gains.kd)
    config.integral_min = -gains.i_limit
    config.integral_max = gains.i_limit
    config.output_min = output_min
    config.output_max = output_max
    return config


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π].

    This function is used to normalize angular differences for PID controllers,
    ensuring the shortest path is taken when commanding attitude changes.

    Args:
        angle: Angle in radians (can be any value)

    Returns:
        Wrapped angle in range [-π, π]

    Example:
        >>> wrap_angle(3.5 * np.pi)  # Returns ~-0.5π
        >>> wrap_angle(0.5 * np.pi)   # Returns 0.5π (unchanged)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def validate_command(
    command: ControlCommand,
    mode: str,
    required_fields: list[str]
) -> None:
    """Validate that required command fields are not None.

    This helper prevents controllers from processing invalid commands
    that would cause AttributeError or incorrect behavior.

    Args:
        command: The control command to validate
        mode: Control mode name for error messages (e.g., "RATE", "ATTITUDE")
        required_fields: List of field names that must not be None

    Raises:
        ValueError: If any required field is None

    Example:
        >>> validate_command(cmd, "RATE", ["roll_rate", "pitch_rate", "yaw_rate"])
    """
    for field in required_fields:
        value = getattr(command, field, None)
        if value is None:
            raise ValueError(
                f"Invalid {mode} command: '{field}' is None. "
                f"All of {required_fields} must be specified."
            )
