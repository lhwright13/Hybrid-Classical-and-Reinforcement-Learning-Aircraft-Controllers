"""Utility functions for PID controller configuration.

This module provides helper functions to eliminate duplicate PID configuration
code across controller implementations.
"""

import aircraft_controls_bindings as acb
from controllers.types import PIDGains


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
