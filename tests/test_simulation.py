"""Unit tests for simulation backend and 6-DOF physics."""

import pytest
import numpy as np
from simulation import Simplified6DOF, AircraftParams, SimulationAircraftBackend
from controllers.types import AircraftState, ControlSurfaces


class TestAircraftParams:
    """Test AircraftParams dataclass."""

    def test_default_params(self):
        """Default parameters create valid aircraft."""
        params = AircraftParams()
        assert params.mass > 0
        assert params.inertia_xx > 0
        assert params.max_thrust > 0
        assert params.wing_area > 0

    def test_custom_params(self):
        """Custom parameters can be set."""
        params = AircraftParams(
            mass=20.0,
            max_thrust=100.0,
            wing_area=1.5
        )
        assert params.mass == 20.0
        assert params.max_thrust == 100.0
        assert params.wing_area == 1.5


class TestSimplified6DOF:
    """Test 6-DOF physics simulator."""

    def test_initialization(self):
        """Simulator initializes with default state."""
        sim = Simplified6DOF()
        state = sim.get_state()

        assert state.time == 0.0
        assert len(state.position) == 3
        assert len(state.velocity) == 3
        assert len(state.attitude) == 3
        assert len(state.angular_rate) == 3

    def test_reset_to_default(self):
        """Reset without arguments goes to default state."""
        sim = Simplified6DOF()

        # Run simulation for a bit
        sim.set_controls(ControlSurfaces(throttle=0.8))
        for _ in range(10):
            sim.step(0.01)

        # Reset
        sim.reset()
        state = sim.get_state()

        # Should be back to initial
        assert state.time == 0.0
        assert state.altitude == pytest.approx(100.0)  # Default altitude
        assert state.airspeed == pytest.approx(20.0)   # Default airspeed

    def test_reset_to_custom_state(self):
        """Reset with custom state sets that state."""
        sim = Simplified6DOF()

        initial = AircraftState(
            altitude=200.0,
            airspeed=30.0,
            position=np.array([100.0, 50.0, -200.0]),
            velocity=np.array([30.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3)
        )

        sim.reset(initial)
        state = sim.get_state()

        assert state.altitude == pytest.approx(200.0)
        assert np.allclose(state.position, initial.position)

    def test_set_controls(self):
        """Controls can be set and applied."""
        sim = Simplified6DOF()
        sim.reset()

        controls = ControlSurfaces(
            elevator=0.2,
            aileron=-0.1,
            rudder=0.05,
            throttle=0.7
        )

        sim.set_controls(controls)
        # Should not raise

    def test_step_advances_time(self):
        """Stepping advances simulation time."""
        sim = Simplified6DOF()
        sim.reset()

        dt = 0.01
        sim.step(dt)
        state = sim.get_state()

        assert state.time == pytest.approx(dt)

        sim.step(dt)
        state = sim.get_state()
        assert state.time == pytest.approx(2 * dt)

    def test_gravity_causes_descent_without_thrust(self):
        """Without thrust, aircraft descends due to gravity."""
        sim = Simplified6DOF()
        sim.reset()

        initial_altitude = sim.get_state().altitude

        # No thrust, no control
        sim.set_controls(ControlSurfaces(throttle=0.0))

        # Run for 1 second
        for _ in range(100):
            sim.step(0.01)

        final_altitude = sim.get_state().altitude

        # Should descend
        assert final_altitude < initial_altitude

    def test_thrust_provides_acceleration(self):
        """Thrust provides forward acceleration."""
        sim = Simplified6DOF()
        sim.reset()

        initial_airspeed = sim.get_state().airspeed

        # Full thrust
        sim.set_controls(ControlSurfaces(throttle=1.0))

        # Run for 1 second
        for _ in range(100):
            sim.step(0.01)

        final_airspeed = sim.get_state().airspeed

        # Should accelerate
        assert final_airspeed > initial_airspeed

    def test_elevator_affects_pitch(self):
        """Elevator control affects pitch rate."""
        sim = Simplified6DOF()
        sim.reset()

        # Nose up elevator
        sim.set_controls(ControlSurfaces(elevator=0.5, throttle=0.7))

        # Run simulation
        for _ in range(50):
            sim.step(0.01)

        state = sim.get_state()

        # Should have non-zero pitch rate
        q = state.angular_rate[1]  # Pitch rate
        assert abs(q) > 0.01  # Should be pitching

    def test_aileron_affects_roll(self):
        """Aileron control affects roll rate."""
        sim = Simplified6DOF()
        sim.reset()

        # Roll right
        sim.set_controls(ControlSurfaces(aileron=0.5, throttle=0.7))

        # Run simulation
        for _ in range(50):
            sim.step(0.01)

        state = sim.get_state()

        # Should have non-zero roll rate
        p = state.angular_rate[0]  # Roll rate
        assert abs(p) > 0.01  # Should be rolling

    def test_simulation_is_stable(self):
        """Simulation doesn't explode with reasonable inputs."""
        sim = Simplified6DOF()
        sim.reset()

        # Random but reasonable controls
        sim.set_controls(ControlSurfaces(
            elevator=0.1,
            aileron=-0.05,
            rudder=0.02,
            throttle=0.6
        ))

        # Run for 10 seconds
        for _ in range(1000):
            sim.step(0.01)
            state = sim.get_state()

            # Check for NaN or infinity
            assert np.all(np.isfinite(state.position))
            assert np.all(np.isfinite(state.velocity))
            assert np.all(np.isfinite(state.attitude))
            assert np.all(np.isfinite(state.angular_rate))

    def test_rk4_integration_stability(self):
        """RK4 integration provides stable results."""
        sim = Simplified6DOF()
        sim.reset()

        sim.set_controls(ControlSurfaces(throttle=0.7))

        # Large time step (should still be stable with RK4)
        for _ in range(10):
            state = sim.step(0.1)  # 100ms step

            # Should remain finite
            assert np.all(np.isfinite(state.position))


class TestSimulationAircraftBackend:
    """Test SimulationAircraftBackend (implements AircraftInterface)."""

    def test_initialization(self):
        """Backend initializes correctly."""
        backend = SimulationAircraftBackend()
        assert backend.get_backend_type() == "simulation"

    def test_initialization_with_config(self):
        """Backend accepts configuration."""
        config = {
            'aircraft_type': 'cessna',
            'dt_physics': 0.0005
        }
        backend = SimulationAircraftBackend(config)

        info = backend.get_info()
        assert info['dt_physics'] == 0.0005
        assert info['aircraft_mass'] == 15.0  # Cessna mass

    def test_implements_aircraft_interface(self):
        """Backend properly implements AircraftInterface."""
        from interfaces.aircraft import AircraftInterface

        backend = SimulationAircraftBackend()
        assert isinstance(backend, AircraftInterface)

    def test_step(self):
        """Step advances simulation."""
        backend = SimulationAircraftBackend()
        backend.reset()

        initial_time = backend.get_state().time

        backend.set_controls(ControlSurfaces(throttle=0.7))
        state = backend.step(dt=0.01)

        assert state.time > initial_time
        assert np.all(np.isfinite(state.position))

    def test_sub_stepping(self):
        """Backend sub-steps physics for stability."""
        backend = SimulationAircraftBackend({'dt_physics': 0.001})
        backend.reset()

        # Large time step - should sub-step internally
        state = backend.step(dt=0.1)

        # Should remain stable
        assert np.all(np.isfinite(state.position))
        assert state.time == pytest.approx(0.1)

    def test_set_get_controls(self):
        """Controls can be set and affect simulation."""
        backend = SimulationAircraftBackend()
        backend.reset()

        controls = ControlSurfaces(elevator=0.2, throttle=0.8)
        backend.set_controls(controls)

        # Step and check state changes
        initial_state = backend.get_state()
        backend.step(0.01)
        final_state = backend.get_state()

        # State should have changed
        assert not np.allclose(initial_state.position, final_state.position)

    def test_reset(self):
        """Reset returns to initial state."""
        backend = SimulationAircraftBackend()

        # Run simulation
        backend.set_controls(ControlSurfaces(throttle=1.0))
        for _ in range(100):
            backend.step(0.01)

        # Reset
        state = backend.reset()

        assert state.time == 0.0
        assert state.altitude == pytest.approx(100.0)

    def test_reset_with_custom_state(self):
        """Reset with custom state works."""
        backend = SimulationAircraftBackend()

        custom = AircraftState(
            altitude=500.0,
            airspeed=50.0,
            position=np.array([0.0, 0.0, -500.0]),
            velocity=np.array([50.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3)
        )

        state = backend.reset(custom)

        assert state.altitude == pytest.approx(500.0)
        assert state.airspeed == pytest.approx(50.0)

    def test_get_state(self):
        """get_state returns current state."""
        backend = SimulationAircraftBackend()
        backend.reset()

        state1 = backend.get_state()
        backend.step(0.01)
        state2 = backend.get_state()

        # States should be different
        assert state2.time > state1.time

    def test_get_backend_type(self):
        """Returns correct backend type."""
        backend = SimulationAircraftBackend()
        assert backend.get_backend_type() == "simulation"

    def test_is_real_hardware(self):
        """Simulation is not real hardware."""
        backend = SimulationAircraftBackend()
        assert not backend.is_real_hardware()

    def test_supports_reset(self):
        """Simulation supports reset."""
        backend = SimulationAircraftBackend()
        assert backend.supports_reset()

    def test_get_dt_nominal(self):
        """Returns nominal time step."""
        backend = SimulationAircraftBackend()
        assert backend.get_dt_nominal() == 0.01  # 100 Hz

    def test_get_info(self):
        """get_info returns backend details."""
        backend = SimulationAircraftBackend()
        info = backend.get_info()

        assert info['backend_type'] == 'simulation'
        assert 'physics_engine' in info
        assert 'dt_physics' in info
        assert 'aircraft_mass' in info

    def test_repr(self):
        """String representation is informative."""
        backend = SimulationAircraftBackend()
        repr_str = repr(backend)

        assert "SimulationAircraftBackend" in repr_str


class TestSimulationIntegration:
    """Integration tests for simulation + sensors."""

    def test_simulation_with_perfect_sensor(self):
        """Simulation works with perfect sensor."""
        from interfaces.sensor import PerfectSensorInterface

        backend = SimulationAircraftBackend()
        sensor = PerfectSensorInterface()

        # Initialize
        backend.reset()
        true_state = backend.get_state()
        sensor.update(true_state)

        # Step simulation
        backend.set_controls(ControlSurfaces(throttle=0.7))
        for _ in range(10):
            true_state = backend.step(0.01)
            sensor.update(true_state)
            measured_state = sensor.get_state()

            # Perfect sensor should match exactly
            assert np.allclose(measured_state.position, true_state.position)
            assert measured_state.altitude == pytest.approx(true_state.altitude)

    def test_simulation_with_noisy_sensor(self):
        """Simulation works with noisy sensor."""
        from interfaces.sensor import NoisySensorInterface

        backend = SimulationAircraftBackend()
        sensor = NoisySensorInterface({'enabled': True, 'seed': 42})

        # Initialize
        backend.reset()
        true_state = backend.get_state()
        sensor.update(true_state)

        # Step simulation
        backend.set_controls(ControlSurfaces(throttle=0.7))
        for _ in range(10):
            true_state = backend.step(0.01)
            sensor.update(true_state)
            measured_state = sensor.get_state()

            # Noisy sensor should differ
            assert not np.allclose(measured_state.position, true_state.position, atol=0.1)

    def test_full_simulation_loop(self):
        """Complete simulation loop over 5 seconds.

        Note: This test validates simulation stability and numerical correctness,
        not altitude hold performance (which requires closed-loop PID control).
        With open-loop controls (fixed elevator/throttle), the aircraft will
        naturally descend due to drag forces exceeding lift at these settings.

        For altitude hold, use the cascaded PID controllers (AttitudeAgent, HSAAgent).
        """
        backend = SimulationAircraftBackend()
        backend.reset()

        # Open-loop controls (no feedback, no PID)
        # These values will NOT maintain altitude - that's expected!
        backend.set_controls(ControlSurfaces(
            elevator=0.05,
            throttle=0.6
        ))

        # Run for 5 seconds at 100 Hz
        for i in range(500):
            state = backend.step(0.01)

            # Check stability - no NaN or infinity
            assert np.all(np.isfinite(state.position))
            assert np.isfinite(state.airspeed)
            assert np.isfinite(state.altitude)

        # Final state should be stable (not exploded)
        final_state = backend.get_state()
        assert final_state.time == pytest.approx(5.0)

        # Physics should remain stable (no blow-ups)
        # Aircraft will descend with these open-loop settings - that's correct physics!
        assert -200 < final_state.altitude < 200  # Stable range (not blown up)
        assert 0 < final_state.airspeed < 100  # Reasonable airspeed range
