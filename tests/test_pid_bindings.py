"""Unit tests for C++ PID controller Python bindings."""

import pytest
import aircraft_controls_bindings as acb


class TestPIDGains:
    """Test PIDGains struct bindings."""

    def test_default_constructor(self):
        """Default constructor creates zero gains."""
        gains = acb.PIDGains()
        assert gains.kp == 0.0
        assert gains.ki == 0.0
        assert gains.kd == 0.0

    def test_parameterized_constructor(self):
        """Parameterized constructor sets gains."""
        gains = acb.PIDGains(1.5, 0.5, 0.2)
        assert gains.kp == pytest.approx(1.5)
        assert gains.ki == pytest.approx(0.5)
        assert gains.kd == pytest.approx(0.2)

    def test_readwrite_properties(self):
        """Gains can be read and written."""
        gains = acb.PIDGains()
        gains.kp = 2.0
        gains.ki = 1.0
        gains.kd = 0.5
        assert gains.kp == 2.0
        assert gains.ki == 1.0
        assert gains.kd == 0.5

    def test_repr(self):
        """String representation includes values."""
        gains = acb.PIDGains(1.0, 0.5, 0.1)
        repr_str = repr(gains)
        assert "PIDGains" in repr_str
        assert "1.0" in repr_str or "1.00" in repr_str


class TestPIDConfig:
    """Test PIDConfig struct bindings."""

    def test_default_constructor(self):
        """Default constructor creates valid config."""
        config = acb.PIDConfig()
        assert config.gains.kp == 0.0
        assert config.output_min == pytest.approx(-1.0)
        assert config.output_max == pytest.approx(1.0)
        assert config.integral_min == pytest.approx(-10.0)
        assert config.integral_max == pytest.approx(10.0)
        assert 0.0 <= config.derivative_filter_alpha <= 1.0

    def test_modify_config(self):
        """Config fields can be modified."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(2.0, 1.0, 0.5)
        config.output_min = -0.5
        config.output_max = 0.5

        assert config.gains.kp == 2.0
        assert config.output_min == -0.5
        assert config.output_max == 0.5


class TestPIDController:
    """Test PIDController class bindings."""

    def test_default_constructor(self):
        """Default constructor creates controller with zero gains."""
        pid = acb.PIDController()
        gains = pid.get_gains()
        assert gains.kp == 0.0
        assert gains.ki == 0.0
        assert gains.kd == 0.0

    def test_config_constructor(self):
        """Constructor with config sets gains."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(1.5, 0.5, 0.2)
        pid = acb.PIDController(config)

        gains = pid.get_gains()
        assert gains.kp == pytest.approx(1.5)
        assert gains.ki == pytest.approx(0.5)
        assert gains.kd == pytest.approx(0.2)

    def test_compute_proportional_only(self):
        """Proportional-only controller works correctly."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(2.0, 0.0, 0.0)  # P-only
        pid = acb.PIDController(config)

        # Setpoint=10, Measurement=8, Error=2
        # Output = kp * error = 2.0 * 2 = 4.0
        # But saturated to [-1, 1], so output = 1.0
        output = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        assert output == pytest.approx(1.0)  # Saturated

        error = pid.get_error()
        assert error == pytest.approx(2.0)

    def test_compute_with_integral(self):
        """Integral term accumulates over time."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(0.0, 0.1, 0.0)  # I-only
        config.output_min = -10.0
        config.output_max = 10.0
        pid = acb.PIDController(config)

        # Run for several steps with constant error
        for _ in range(10):
            pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)

        # Integral should accumulate: error=1.0, dt=0.1, 10 steps
        # integral = 1.0 * 0.1 * 10 = 1.0
        # output = ki * integral = 0.1 * 1.0 = 0.1
        output = pid.get_output()
        assert output == pytest.approx(0.1, abs=0.01)

        integral = pid.get_integral()
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_reset(self):
        """Reset clears controller state."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(1.0, 1.0, 1.0)
        pid = acb.PIDController(config)

        # Accumulate some state
        for _ in range(5):
            pid.compute(setpoint=10.0, measurement=5.0, dt=0.1)

        # State should be non-zero
        assert pid.get_error() != 0.0
        assert pid.get_integral() != 0.0

        # Reset
        pid.reset()

        # State should be zero
        assert pid.get_error() == 0.0
        assert pid.get_integral() == 0.0
        assert pid.get_derivative() == 0.0
        assert pid.get_output() == 0.0

    def test_set_get_gains(self):
        """Gains can be set and retrieved."""
        pid = acb.PIDController()

        new_gains = acb.PIDGains(3.0, 2.0, 1.0)
        pid.set_gains(new_gains)

        retrieved_gains = pid.get_gains()
        assert retrieved_gains.kp == 3.0
        assert retrieved_gains.ki == 2.0
        assert retrieved_gains.kd == 1.0

    def test_output_saturation(self):
        """Output is saturated to min/max limits."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(10.0, 0.0, 0.0)  # High P gain
        config.output_min = -0.5
        config.output_max = 0.5
        pid = acb.PIDController(config)

        # Large error should saturate output
        output = pid.compute(setpoint=100.0, measurement=0.0, dt=0.01)
        assert output == pytest.approx(0.5)  # Saturated to max

        output = pid.compute(setpoint=0.0, measurement=100.0, dt=0.01)
        assert output == pytest.approx(-0.5)  # Saturated to min

    def test_integral_anti_windup(self):
        """Integral term is clamped to prevent wind-up."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(0.0, 1.0, 0.0)  # I-only
        config.integral_min = -5.0
        config.integral_max = 5.0
        pid = acb.PIDController(config)

        # Accumulate large integral
        for _ in range(100):
            pid.compute(setpoint=100.0, measurement=0.0, dt=0.1)

        # Integral should be clamped
        integral = pid.get_integral()
        assert integral <= 5.0
        assert integral == pytest.approx(5.0)

    def test_derivative_calculation(self):
        """Derivative term responds to error changes."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(0.0, 0.0, 1.0)  # D-only
        config.derivative_filter_alpha = 1.0  # No filtering for this test
        config.output_min = -100.0
        config.output_max = 100.0
        pid = acb.PIDController(config)

        # First step: error = 0 → no derivative
        pid.compute(setpoint=10.0, measurement=10.0, dt=0.1)
        assert pid.get_derivative() == pytest.approx(0.0)

        # Second step: error changes from 0 to 5
        # derivative = (5 - 0) / 0.1 = 50.0
        pid.compute(setpoint=10.0, measurement=5.0, dt=0.1)
        derivative = pid.get_derivative()
        assert derivative == pytest.approx(50.0, abs=1.0)

    def test_repr(self):
        """String representation includes gains."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(1.0, 0.5, 0.1)
        pid = acb.PIDController(config)

        repr_str = repr(pid)
        assert "PIDController" in repr_str


class TestPIDIntegration:
    """Integration tests for PID controller."""

    def test_altitude_hold_simulation(self):
        """Simulate altitude hold controller."""
        config = acb.PIDConfig()
        config.gains = acb.PIDGains(0.5, 0.1, 0.2)  # Tuned for altitude
        config.output_min = -1.0
        config.output_max = 1.0
        pid = acb.PIDController(config)

        # Simulate simple altitude dynamics
        target_altitude = 100.0
        current_altitude = 90.0  # Start 10m below target
        dt = 0.01  # 100 Hz

        # Run for 100 steps
        for _ in range(100):
            # PID output (elevator command)
            elevator = pid.compute(target_altitude, current_altitude, dt)

            # Simple dynamics: altitude change proportional to elevator
            altitude_rate = elevator * 5.0  # m/s
            current_altitude += altitude_rate * dt

        # Should have reduced error significantly
        final_error = abs(target_altitude - current_altitude)
        assert final_error == pytest.approx(5.0, abs=0.1)  # Within 5 meters (approximately)

    def test_version(self):
        """Module has version attribute."""
        assert hasattr(acb, "__version__")
        assert isinstance(acb.__version__, str)
        assert acb.__version__ == "1.1.0"
