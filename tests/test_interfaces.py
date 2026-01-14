"""Unit tests for core interfaces (agent, aircraft, sensor)."""

import pytest
import numpy as np
from abc import ABC
from dataclasses import replace

from interfaces.agent import RLAgentInterface
from interfaces.aircraft import AircraftInterface
from interfaces.sensor import (
    SensorInterface,
    PerfectSensorInterface,
    NoisySensorInterface,
)
from controllers.types import (
    AircraftState,
    ControlMode,
    ControlCommand,
    ControlSurfaces,
)


# =============================================================================
# Test Agent Interface
# =============================================================================


class ConcreteAgent(RLAgentInterface):
    """Minimal concrete agent for testing."""

    def __init__(self, control_level: ControlMode):
        self.control_level = control_level

    def get_control_level(self) -> ControlMode:
        return self.control_level

    def reset(self, initial_state: AircraftState) -> None:
        self.last_state = initial_state

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        # Return a dummy command
        return ControlCommand(
            mode=self.control_level,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
            throttle=0.5,
        )


class TestRLAgentInterface:
    """Test RLAgentInterface interface."""

    def test_cannot_instantiate_abstract_class(self):
        """RLAgentInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            RLAgentInterface()

    def test_concrete_agent_implements_required_methods(self):
        """Concrete agent must implement all abstract methods."""
        agent = ConcreteAgent(ControlMode.RATE)

        # Test get_control_level
        assert agent.get_control_level() == ControlMode.RATE

        # Test reset
        state = AircraftState(altitude=100.0)
        agent.reset(state)
        assert agent.last_state.altitude == 100.0

        # Test get_action
        obs = np.zeros(10)
        command = agent.get_action(obs)
        assert isinstance(command, ControlCommand)
        assert command.mode == ControlMode.RATE

    def test_observation_space_per_level(self):
        """Each control level has correct observation space dimensions."""
        test_cases = [
            (ControlMode.WAYPOINT, 12),
            (ControlMode.HSA, 12),
            (ControlMode.RATE, 10),
            (ControlMode.SURFACE, 14),
        ]

        for level, expected_dim in test_cases:
            agent = ConcreteAgent(level)
            obs_space = agent.get_observation_space()
            assert obs_space["shape"] == (expected_dim,)

    def test_action_space_per_level(self):
        """Each control level has correct action space dimensions."""
        test_cases = [
            (ControlMode.WAYPOINT, 4),
            (ControlMode.HSA, 3),
            (ControlMode.RATE, 4),
            (ControlMode.SURFACE, 4),
        ]

        for level, expected_dim in test_cases:
            agent = ConcreteAgent(level)
            action_space = agent.get_action_space()
            assert action_space["shape"] == (expected_dim,)

    def test_preprocess_observation_stick_level(self):
        """preprocess_observation extracts correct features for rate level."""
        agent = ConcreteAgent(ControlMode.RATE)

        state = AircraftState(
            velocity=np.array([10.0, 5.0, -2.0]),
            attitude=np.array([0.1, 0.2, 0.3]),
            angular_rate=np.array([0.05, -0.01, 0.02]),
            airspeed=15.0,
        )

        obs = agent.preprocess_observation(state)

        # Should be 10D: vel(3), att(3), rates(3), airspeed(1)
        assert obs.shape == (10,)
        assert np.allclose(obs[0:3], state.velocity)
        assert np.allclose(obs[3:6], state.attitude)
        assert np.allclose(obs[6:9], state.angular_rate)
        assert obs[9] == state.airspeed

    def test_switch_control_level_not_implemented_by_default(self):
        """switch_control_level raises NotImplementedError by default."""
        agent = ConcreteAgent(ControlMode.RATE)
        with pytest.raises(NotImplementedError):
            agent.switch_control_level(ControlMode.HSA)

    def test_optional_methods_have_default_implementation(self):
        """Optional methods (update, save, load) should not raise errors."""
        agent = ConcreteAgent(ControlMode.RATE)

        # update() is optional
        transition = {
            "state": np.zeros(10),
            "action": ControlCommand(mode=ControlMode.RATE),
            "reward": 1.0,
            "next_state": np.zeros(10),
            "done": False,
            "info": {},
        }
        agent.update(transition)  # Should not raise

        # save/load are optional
        agent.save("dummy_path.pth")  # Should not raise
        agent.load("dummy_path.pth")  # Should not raise

    def test_repr(self):
        """String representation includes control level."""
        agent = ConcreteAgent(ControlMode.HSA)
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str
        assert "HSA" in repr_str


# =============================================================================
# Test Aircraft Interface
# =============================================================================


class ConcreteAircraft(AircraftInterface):
    """Minimal concrete aircraft for testing."""

    def __init__(self, backend_type: str = "simulation"):
        self.backend_type = backend_type
        self.state = AircraftState()
        self.controls = ControlSurfaces()

    def step(self, dt: float) -> AircraftState:
        # Simple integration: update altitude
        self.state.time += dt
        self.state.altitude += dt * 10.0  # Climb at 10 m/s
        return self.state

    def set_controls(self, surfaces: ControlSurfaces) -> None:
        self.controls = surfaces

    def reset(self, initial_state=None) -> AircraftState:
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = AircraftState(altitude=100.0, airspeed=20.0)
        return self.state

    def get_state(self) -> AircraftState:
        return self.state

    def get_backend_type(self) -> str:
        return self.backend_type


class TestAircraftInterface:
    """Test AircraftInterface."""

    def test_cannot_instantiate_abstract_class(self):
        """AircraftInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AircraftInterface()

    def test_concrete_aircraft_implements_required_methods(self):
        """Concrete aircraft must implement all abstract methods."""
        aircraft = ConcreteAircraft()

        # Test get_backend_type
        assert aircraft.get_backend_type() == "simulation"

        # Test reset
        state = aircraft.reset()
        assert state.altitude == 100.0

        # Test set_controls
        surfaces = ControlSurfaces(elevator=0.1, throttle=0.7)
        aircraft.set_controls(surfaces)
        assert aircraft.controls.elevator == 0.1

        # Test step
        state = aircraft.step(dt=0.1)
        assert state.time == 0.1
        assert state.altitude == 101.0  # 100 + 0.1 * 10

        # Test get_state
        state2 = aircraft.get_state()
        assert state2.altitude == state.altitude

    def test_is_real_hardware(self):
        """is_real_hardware returns True for hardware backends."""
        sim_aircraft = ConcreteAircraft(backend_type="simulation")
        hw_aircraft = ConcreteAircraft(backend_type="hardware")
        hil_aircraft = ConcreteAircraft(backend_type="hil")

        assert not sim_aircraft.is_real_hardware()
        assert hw_aircraft.is_real_hardware()
        assert hil_aircraft.is_real_hardware()

    def test_supports_reset(self):
        """supports_reset returns True only for simulation."""
        sim_aircraft = ConcreteAircraft(backend_type="simulation")
        hw_aircraft = ConcreteAircraft(backend_type="hardware")

        assert sim_aircraft.supports_reset()
        assert not hw_aircraft.supports_reset()

    def test_get_dt_nominal(self):
        """get_dt_nominal returns default 0.01 (100 Hz)."""
        aircraft = ConcreteAircraft()
        assert aircraft.get_dt_nominal() == 0.01

    def test_get_info(self):
        """get_info returns backend information."""
        aircraft = ConcreteAircraft(backend_type="simulation")
        info = aircraft.get_info()

        assert info["backend_type"] == "simulation"
        assert info["dt_nominal"] == 0.01

    def test_close_is_optional(self):
        """close() method is optional and should not raise."""
        aircraft = ConcreteAircraft()
        aircraft.close()  # Should not raise

    def test_repr(self):
        """String representation includes backend type."""
        aircraft = ConcreteAircraft(backend_type="simulation")
        repr_str = repr(aircraft)
        assert "ConcreteAircraft" in repr_str
        assert "simulation" in repr_str


# =============================================================================
# Test Sensor Interface
# =============================================================================


class TestSensorInterface:
    """Test SensorInterface abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """SensorInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SensorInterface()


class TestPerfectSensorInterface:
    """Test PerfectSensorInterface (ground truth)."""

    def test_initialization(self):
        """Perfect sensor initializes correctly."""
        sensor = PerfectSensorInterface()
        assert sensor.get_sensor_type() == "perfect"
        assert sensor.is_perfect()

    def test_update_and_get_state(self):
        """Perfect sensor returns exact ground truth."""
        sensor = PerfectSensorInterface()

        true_state = AircraftState(
            altitude=150.0,
            airspeed=25.0,
            position=np.array([100.0, 200.0, -150.0]),
        )

        sensor.update(true_state)
        measured_state = sensor.get_state()

        # Should be exact match
        assert measured_state.altitude == 150.0
        assert measured_state.airspeed == 25.0
        assert np.allclose(measured_state.position, true_state.position)

    def test_get_state_before_update_raises_error(self):
        """Getting state before update raises RuntimeError."""
        sensor = PerfectSensorInterface()
        with pytest.raises(RuntimeError, match="not yet updated"):
            sensor.get_state()

    def test_reset(self):
        """Reset clears sensor state."""
        sensor = PerfectSensorInterface()

        true_state = AircraftState(altitude=100.0)
        sensor.update(true_state)
        sensor.reset()

        with pytest.raises(RuntimeError):
            sensor.get_state()

    def test_get_noise_parameters_returns_empty(self):
        """Perfect sensor has no noise parameters."""
        sensor = PerfectSensorInterface()
        params = sensor.get_noise_parameters()
        assert params == {}

    def test_repr(self):
        """String representation includes sensor type."""
        sensor = PerfectSensorInterface()
        repr_str = repr(sensor)
        assert "PerfectSensorInterface" in repr_str
        assert "perfect" in repr_str


class TestNoisySensorInterface:
    """Test NoisySensorInterface (realistic sensor noise)."""

    def test_initialization(self):
        """Noisy sensor initializes with noise config."""
        noise_config = {
            "enabled": True,
            "imu_gyro_stddev": 0.02,
            "gps_position_stddev": 2.5,
            "seed": 42,
        }
        sensor = NoisySensorInterface(noise_config)

        assert sensor.get_sensor_type() == "noisy"
        assert not sensor.is_perfect()
        assert sensor.gyro_noise == 0.02
        assert sensor.gps_pos_noise == 2.5

    def test_adds_noise_to_measurements(self):
        """Noisy sensor adds noise to all measurements."""
        noise_config = {
            "enabled": True,
            "imu_gyro_stddev": 0.1,
            "gps_position_stddev": 5.0,
            "attitude_stddev": 0.05,
            "airspeed_stddev": 1.0,
            "seed": 123,
        }
        sensor = NoisySensorInterface(noise_config)

        true_state = AircraftState(
            position=np.array([100.0, 200.0, -150.0]),
            velocity=np.array([20.0, 5.0, -2.0]),
            attitude=np.array([0.1, 0.2, 0.0]),
            angular_rate=np.array([0.05, 0.01, 0.02]),
            airspeed=25.0,
            altitude=150.0,
        )

        sensor.update(true_state)
        measured_state = sensor.get_state()

        # Measurements should be different from true state (with high probability)
        # We use a loose check since noise is random
        assert not np.allclose(measured_state.position, true_state.position, atol=0.1)
        assert not np.allclose(measured_state.attitude, true_state.attitude, atol=0.01)

    def test_disabled_noise_returns_ground_truth(self):
        """When disabled, noisy sensor returns ground truth."""
        noise_config = {
            "enabled": False,
            "imu_gyro_stddev": 100.0,  # Large noise, but disabled
        }
        sensor = NoisySensorInterface(noise_config)

        true_state = AircraftState(altitude=200.0, airspeed=30.0)
        sensor.update(true_state)
        measured_state = sensor.get_state()

        # Should match exactly since noise is disabled
        assert measured_state.altitude == 200.0
        assert measured_state.airspeed == 30.0

    def test_reproducible_with_seed(self):
        """Using same seed produces reproducible noise."""
        noise_config = {
            "enabled": True,
            "gps_position_stddev": 1.0,
            "seed": 999,
        }

        sensor1 = NoisySensorInterface(noise_config)
        sensor2 = NoisySensorInterface(noise_config)

        true_state = AircraftState(position=np.array([0.0, 0.0, 0.0]))

        sensor1.update(true_state)
        sensor2.update(true_state)

        state1 = sensor1.get_state()
        state2 = sensor2.get_state()

        # Should produce identical noise with same seed
        assert np.allclose(state1.position, state2.position)

    def test_bias_drift(self):
        """Gyro/accel biases drift over time."""
        noise_config = {"enabled": True, "seed": 42}
        sensor = NoisySensorInterface(noise_config)

        initial_gyro_bias = sensor.gyro_bias.copy()
        initial_accel_bias = sensor.accel_bias.copy()

        true_state = AircraftState()

        # Update sensor many times
        for _ in range(100):
            sensor.update(true_state)

        # Bias should have drifted (at least one axis should change significantly)
        gyro_changed = not np.allclose(sensor.gyro_bias, initial_gyro_bias, atol=0.001)
        accel_changed = not np.allclose(sensor.accel_bias, initial_accel_bias, atol=0.005)

        # At least gyro bias should have changed after 100 updates
        assert gyro_changed or accel_changed

    def test_reset_clears_bias(self):
        """Reset clears accumulated bias."""
        noise_config = {"enabled": True, "seed": 42}
        sensor = NoisySensorInterface(noise_config)

        # Accumulate some bias
        true_state = AircraftState()
        for _ in range(50):
            sensor.update(true_state)

        # Reset
        sensor.reset()

        # Bias should be zero
        assert np.allclose(sensor.gyro_bias, np.zeros(3))
        assert np.allclose(sensor.accel_bias, np.zeros(3))

    def test_get_noise_parameters(self):
        """get_noise_parameters returns configuration."""
        noise_config = {
            "enabled": True,
            "imu_gyro_stddev": 0.015,
            "gps_position_stddev": 3.0,
            "airspeed_stddev": 0.8,
        }
        sensor = NoisySensorInterface(noise_config)

        params = sensor.get_noise_parameters()

        assert params["enabled"] is True
        assert params["imu_gyro_stddev"] == 0.015
        assert params["gps_position_stddev"] == 3.0
        assert params["airspeed_stddev"] == 0.8

    def test_default_noise_values(self):
        """Sensor uses default noise values if not specified."""
        noise_config = {}  # Empty config
        sensor = NoisySensorInterface(noise_config)

        # Check defaults are applied
        assert sensor.enabled is True  # Default enabled
        assert sensor.gyro_noise == 0.01  # Default from code
        assert sensor.gps_pos_noise == 1.0
        assert sensor.airspeed_noise == 0.5

    def test_repr(self):
        """String representation includes sensor type."""
        noise_config = {"seed": 42}
        sensor = NoisySensorInterface(noise_config)
        repr_str = repr(sensor)
        assert "NoisySensorInterface" in repr_str
        assert "noisy" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestInterfaceIntegration:
    """Test interfaces working together."""

    def test_agent_aircraft_sensor_pipeline(self):
        """Complete pipeline: agent commands → aircraft → sensor → agent."""
        # Create components
        agent = ConcreteAgent(ControlMode.RATE)
        aircraft = ConcreteAircraft()
        sensor = PerfectSensorInterface()

        # Initialize
        initial_state = AircraftState(altitude=100.0, airspeed=20.0)
        aircraft.reset(initial_state)
        agent.reset(initial_state)
        sensor.update(initial_state)

        # Simulation loop
        for _ in range(5):
            # Agent gets observation from sensor
            state = sensor.get_state()
            obs = agent.preprocess_observation(state)

            # Agent produces command
            command = agent.get_action(obs)
            assert command.mode == ControlMode.RATE

            # Convert command to control surfaces (simplified)
            surfaces = ControlSurfaces(
                elevator=command.pitch_rate if command.pitch_rate else 0.0,
                aileron=command.roll_rate if command.roll_rate else 0.0,
                rudder=command.yaw_rate if command.yaw_rate else 0.0,
                throttle=command.throttle if command.throttle else 0.5,
            )

            # Apply to aircraft
            aircraft.set_controls(surfaces)
            new_state = aircraft.step(dt=0.1)

            # Update sensor
            sensor.update(new_state)

        # Check simulation advanced
        final_state = sensor.get_state()
        assert final_state.time > 0.0
        assert final_state.altitude > 100.0  # Should have climbed

    def test_noisy_sensor_with_aircraft(self):
        """Noisy sensor adds realistic noise to aircraft state."""
        aircraft = ConcreteAircraft()
        sensor = NoisySensorInterface({"enabled": True, "seed": 42})

        # Reset aircraft
        true_state = aircraft.reset(AircraftState(altitude=100.0))

        # Update sensor
        sensor.update(true_state)
        measured_state = sensor.get_state()

        # Measured state should differ from true state
        assert measured_state.altitude != true_state.altitude
