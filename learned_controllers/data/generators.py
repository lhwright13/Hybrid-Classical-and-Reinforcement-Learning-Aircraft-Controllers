"""Data generators for training rate controllers."""

import numpy as np
from typing import Tuple, Optional, Literal


class RateCommandGenerator:
    """Generate diverse rate command profiles for training.

    Creates varied angular rate commands to ensure the policy learns
    across different flight regimes and maneuver types.
    """

    def __init__(
        self,
        max_roll_rate: float = np.radians(180),  # rad/s
        max_pitch_rate: float = np.radians(180),  # rad/s
        max_yaw_rate: float = np.radians(160),   # rad/s
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        rng_seed: Optional[int] = None,
    ):
        """Initialize command generator.

        Args:
            max_roll_rate: Maximum roll rate command (rad/s)
            max_pitch_rate: Maximum pitch rate command (rad/s)
            max_yaw_rate: Maximum yaw rate command (rad/s)
            difficulty: Difficulty level affecting command magnitudes
            rng_seed: Random seed for reproducibility
        """
        self.max_roll_rate = max_roll_rate
        self.max_pitch_rate = max_pitch_rate
        self.max_yaw_rate = max_yaw_rate
        self.difficulty = difficulty
        self.rng = np.random.RandomState(rng_seed)

        # Pre-allocate max rates array (used in every command generation method)
        self._max_rates = np.array([max_roll_rate, max_pitch_rate, max_yaw_rate])

        # Difficulty scaling
        self.difficulty_scale = {
            "easy": 0.3,      # 30% of max rates (~54 deg/s)
            "medium": 0.5,    # 50% of max rates (~90 deg/s)
            "hard": 0.7,      # 70% of max rates (~126 deg/s) - achievable
        }[difficulty]

    def generate_step_command(
        self,
        num_axes: int = 1,
        hold_time: float = 2.0,
    ) -> Tuple[np.ndarray, str]:
        """Generate step input command.

        Args:
            num_axes: Number of axes to command (1-3)
            hold_time: Time to hold command (seconds)

        Returns:
            Tuple of (command [p, q, r], description)
        """
        command = np.zeros(3)
        active_axes = self.rng.choice(3, size=min(num_axes, 3), replace=False)

        for axis in active_axes:
            # Random magnitude scaled by difficulty
            magnitude = self.rng.uniform(0.3, 1.0) * self.difficulty_scale
            command[axis] = self.rng.choice([-1, 1]) * magnitude * self._max_rates[axis]

        axis_names = ["roll", "pitch", "yaw"]
        desc = f"Step: {', '.join([axis_names[i] for i in active_axes])}"

        return command, desc

    def generate_ramp_command(
        self,
        duration: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate ramp (gradual) rate command.

        Args:
            duration: Ramp duration (seconds)

        Returns:
            Tuple of (start_command [p,q,r], end_command [p,q,r], description)
        """
        start = np.zeros(3)

        end = np.zeros(3)
        num_axes = self.rng.choice([1, 2, 3])
        active_axes = self.rng.choice(3, size=num_axes, replace=False)

        for axis in active_axes:
            magnitude = self.rng.uniform(0.3, 1.0) * self.difficulty_scale
            end[axis] = self.rng.choice([-1, 1]) * magnitude * self._max_rates[axis]

        desc = f"Ramp: {duration:.1f}s"
        return start, end, desc

    def generate_sine_command(
        self,
        frequency: Optional[float] = None,
        amplitude_scale: float = 0.5,
    ) -> Tuple[float, np.ndarray, str]:
        """Generate sinusoidal rate command.

        Args:
            frequency: Oscillation frequency (Hz), random if None
            amplitude_scale: Amplitude scaling (0-1)

        Returns:
            Tuple of (frequency, amplitudes [p,q,r], description)
        """
        if frequency is None:
            # Random frequency between 0.1 and 2 Hz
            frequency = self.rng.uniform(0.1, 2.0)

        # Random amplitudes
        amplitudes = np.zeros(3)
        num_axes = self.rng.choice([1, 2])  # 1-2 axes for sine

        active_axes = self.rng.choice(3, size=num_axes, replace=False)
        for axis in active_axes:
            amp = self.rng.uniform(0.3, 1.0) * amplitude_scale * self.difficulty_scale
            amplitudes[axis] = amp * self._max_rates[axis]

        desc = f"Sine: {frequency:.2f} Hz"
        return frequency, amplitudes, desc

    def generate_multi_axis_command(
        self,
    ) -> Tuple[np.ndarray, str]:
        """Generate coupled multi-axis command.

        Returns:
            Tuple of (command [p,q,r], description)
        """
        # All axes active
        command = np.zeros(3)
        for axis in range(3):
            magnitude = self.rng.uniform(0.4, 1.0) * self.difficulty_scale
            command[axis] = self.rng.choice([-1, 1]) * magnitude * self._max_rates[axis]

        return command, "Multi-axis coupled"

    def generate_random_walk(
        self,
        dt: float = 0.1,
        diffusion: float = 0.1,
    ) -> Tuple[np.ndarray, str]:
        """Generate random walk command (slowly varying).

        Args:
            dt: Time step (seconds)
            diffusion: Random walk diffusion rate

        Returns:
            Tuple of (delta_command [p,q,r], description)
        """
        # Random walk deltas
        delta = self.rng.randn(3) * diffusion * np.sqrt(dt)

        delta *= self.difficulty_scale * self._max_rates

        return delta, "Random walk"


class FlightEnvelopeSampler:
    """Sample initial conditions across flight envelope."""

    def __init__(
        self,
        airspeed_range: Tuple[float, float] = (15.0, 30.0),  # m/s
        altitude_range: Tuple[float, float] = (50.0, 200.0),  # m
        attitude_range: Tuple[float, float] = (np.radians(-15), np.radians(15)),
        rng_seed: Optional[int] = None,
    ):
        """Initialize envelope sampler.

        Args:
            airspeed_range: (min, max) airspeed (m/s)
            altitude_range: (min, max) altitude (m)
            attitude_range: (min, max) roll/pitch (rad)
            rng_seed: Random seed for reproducibility
        """
        self.airspeed_range = airspeed_range
        self.altitude_range = altitude_range
        self.attitude_range = attitude_range
        self.rng = np.random.RandomState(rng_seed)

    def sample(self) -> dict:
        """Sample random initial conditions.

        Returns:
            Dictionary with initial state parameters
        """
        airspeed = self.rng.uniform(*self.airspeed_range)
        altitude = self.rng.uniform(*self.altitude_range)

        # Small initial attitudes
        roll = self.rng.uniform(*self.attitude_range)
        pitch = self.rng.uniform(*self.attitude_range)
        yaw = self.rng.uniform(0, 2 * np.pi)

        # Random initial rates (small)
        p = self.rng.uniform(-0.1, 0.1)
        q = self.rng.uniform(-0.1, 0.1)
        r = self.rng.uniform(-0.1, 0.1)

        return {
            "airspeed": airspeed,
            "altitude": altitude,
            "attitude": np.array([roll, pitch, yaw]),
            "angular_rate": np.array([p, q, r]),
        }
