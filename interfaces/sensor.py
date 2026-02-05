"""Sensor interface for different sensor implementations."""

from abc import ABC, abstractmethod
from controllers.types import AircraftState


class SensorInterface(ABC):
    """Abstract interface for sensor implementations.

    This interface enables swappability between:
    - Perfect sensors (simulation ground truth)
    - Noisy sensors (simulation with realistic noise)
    - Real sensors (hardware IMU, GPS, etc.)

    Sensor interface sits between aircraft backend and agent/controller,
    allowing sensor noise and filtering to be modeled independently.
    """

    @abstractmethod
    def get_state(self) -> AircraftState:
        """Get current sensor reading as aircraft state.

        Returns:
            Aircraft state as measured by sensors (may include noise)

        Example:
            >>> sensor = NoisySensorInterface(noise_config)
            >>> state = sensor.get_state()
            >>> print(state.altitude)  # May differ from true altitude
            99.87
        """
        pass

    @abstractmethod
    def update(self, true_state: AircraftState) -> None:
        """Update sensor with new true state from backend.

        Sensor applies noise model and/or filtering to convert true state
        to sensor reading.

        Args:
            true_state: Ground truth state from backend

        Example:
            >>> true_state = backend.step(dt=0.01)
            >>> sensor.update(true_state)
            >>> measured_state = sensor.get_state()  # With noise applied
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset sensor to initial state.

        Clears any accumulated state (filters, biases, etc.)

        Example:
            >>> sensor.reset()
        """
        pass

    @abstractmethod
    def get_sensor_type(self) -> str:
        """Return sensor type identifier.

        Returns:
            "perfect", "noisy", or "hardware"

        Example:
            >>> sensor.get_sensor_type()
            'noisy'
        """
        pass

    def is_perfect(self) -> bool:
        """Check if sensor provides perfect measurements.

        Returns:
            True if perfect (ground truth), False if noisy/filtered

        Example:
            >>> if not sensor.is_perfect():
            ...     print("Sensor noise enabled")
        """
        return self.get_sensor_type() == "perfect"

    def get_noise_parameters(self) -> dict:
        """Get sensor noise parameters (optional).

        Returns:
            Dictionary with noise configuration

        Example:
            >>> params = sensor.get_noise_parameters()
            >>> print(params)
            {'imu_gyro_stddev': 0.01, 'gps_position_stddev': 2.0}
        """
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(type={self.get_sensor_type()})"


class PerfectSensorInterface(SensorInterface):
    """Perfect sensor implementation - returns ground truth.

    Use for:
    - Initial debugging
    - Testing control algorithms without sensor noise
    - Establishing performance baselines
    """

    def __init__(self):
        """Initialize perfect sensor."""
        self._state = None

    def get_state(self) -> AircraftState:
        """Return ground truth state."""
        if self._state is None:
            raise RuntimeError("Sensor not yet updated with state")
        return self._state

    def update(self, true_state: AircraftState) -> None:
        """Store ground truth state (no modification)."""
        self._state = true_state

    def reset(self) -> None:
        """Reset sensor."""
        self._state = None

    def get_sensor_type(self) -> str:
        """Return sensor type."""
        return "perfect"


class NoisySensorInterface(SensorInterface):
    """Noisy sensor implementation - adds realistic sensor noise.

    Use for:
    - Realistic simulation
    - Testing robustness to sensor noise
    - Training RL agents with domain randomization
    """

    def __init__(self, noise_config: dict):
        """Initialize noisy sensor.

        Args:
            noise_config: Noise parameters:
                - enabled: bool (default True)
                - imu_gyro_stddev: float (rad/s)
                - imu_accel_stddev: float (m/s^2)
                - gps_position_stddev: float (m)
                - gps_velocity_stddev: float (m/s)
                - airspeed_stddev: float (m/s)
                - altitude_stddev: float (m)
                - attitude_stddev: float (rad)
                - seed: int (optional, for reproducibility)

        Example:
            >>> noise_config = {
            ...     'enabled': True,
            ...     'imu_gyro_stddev': 0.01,
            ...     'gps_position_stddev': 2.0
            ... }
            >>> sensor = NoisySensorInterface(noise_config)
        """
        import numpy as np

        self._config = noise_config
        self._enabled = noise_config.get("enabled", True)

        # Noise standard deviations
        self._gyro_noise = noise_config.get("imu_gyro_stddev", 0.01)
        self._accel_noise = noise_config.get("imu_accel_stddev", 0.1)
        self._gps_pos_noise = noise_config.get("gps_position_stddev", 1.0)
        self._gps_vel_noise = noise_config.get("gps_velocity_stddev", 0.1)
        self._airspeed_noise = noise_config.get("airspeed_stddev", 0.5)
        self._altitude_noise = noise_config.get("altitude_stddev", 0.5)
        self._attitude_noise = noise_config.get("attitude_stddev", 0.01)

        # Bias (slowly varying)
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)

        # Random number generator
        seed = noise_config.get("seed", None)
        self._rng = np.random.default_rng(seed)

        # Current state
        self._state = None

    def get_state(self) -> AircraftState:
        """Return noisy sensor reading."""
        if self._state is None:
            raise RuntimeError("Sensor not yet updated with state")
        return self._state

    def update(self, true_state: AircraftState) -> None:
        """Apply sensor noise to true state."""
        from dataclasses import replace

        if not self._enabled:
            self._state = true_state
            return

        # Add noise to each measurement
        noisy_position = true_state.position + self._rng.normal(
            0, self._gps_pos_noise, 3
        )
        noisy_velocity = true_state.velocity + self._rng.normal(
            0, self._gps_vel_noise, 3
        )
        noisy_attitude = true_state.attitude + self._rng.normal(
            0, self._attitude_noise, 3
        )
        noisy_angular_rate = (
            true_state.angular_rate
            + self._rng.normal(0, self._gyro_noise, 3)
            + self._gyro_bias
        )
        noisy_airspeed = true_state.airspeed + self._rng.normal(
            0, self._airspeed_noise
        )
        noisy_altitude = true_state.altitude + self._rng.normal(
            0, self._altitude_noise
        )

        # Update bias (random walk)
        self._gyro_bias += self._rng.normal(0, 0.0001, 3)
        self._accel_bias += self._rng.normal(0, 0.001, 3)

        # Create noisy state
        self._state = replace(
            true_state,
            position=noisy_position,
            velocity=noisy_velocity,
            attitude=noisy_attitude,
            angular_rate=noisy_angular_rate,
            airspeed=noisy_airspeed,
            altitude=noisy_altitude,
        )

    def reset(self) -> None:
        """Reset biases."""
        import numpy as np

        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._state = None

    def get_sensor_type(self) -> str:
        """Return sensor type."""
        return "noisy"

    def get_noise_parameters(self) -> dict:
        """Return noise configuration."""
        return {
            "enabled": self._enabled,
            "imu_gyro_stddev": self._gyro_noise,
            "imu_accel_stddev": self._accel_noise,
            "gps_position_stddev": self._gps_pos_noise,
            "gps_velocity_stddev": self._gps_vel_noise,
            "airspeed_stddev": self._airspeed_noise,
            "altitude_stddev": self._altitude_noise,
            "attitude_stddev": self._attitude_noise,
        }
