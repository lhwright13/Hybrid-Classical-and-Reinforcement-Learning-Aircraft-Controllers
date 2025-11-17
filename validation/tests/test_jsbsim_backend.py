"""Unit tests for JSBSimBackend."""

import pytest
import numpy as np

from validation.jsbsim_backend import JSBSimBackend
from controllers.types import AircraftState, ControlSurfaces


class TestJSBSimBackend:
    """Test JSBSimBackend class."""

    def test_initialization(self):
        """Backend initializes successfully."""
        backend = JSBSimBackend()
        assert backend.get_backend_type() == "jsbsim"
        assert backend.fdm is not None

    def test_initialization_with_config(self):
        """Backend accepts configuration."""
        config = {
            'aircraft': 'c172x',
            'dt_physics': 1/120.0
        }
        backend = JSBSimBackend(config)
        info = backend.get_info()

        assert info['aircraft'] == 'c172x'
        assert info['dt_physics'] == pytest.approx(1/120.0)

    def test_implements_aircraft_interface(self):
        """Backend implements AircraftInterface."""
        from interfaces.aircraft import AircraftInterface

        backend = JSBSimBackend()
        assert isinstance(backend, AircraftInterface)

    def test_reset(self):
        """Reset returns to initial state."""
        backend = JSBSimBackend()
        state = backend.reset()

        assert isinstance(state, AircraftState)
        assert state.time == pytest.approx(0.0)
        assert np.all(np.isfinite(state.position))
        assert np.all(np.isfinite(state.velocity))

    def test_step(self):
        """Step advances simulation."""
        backend = JSBSimBackend()
        backend.reset()

        initial_time = backend.get_state().time

        backend.set_controls(ControlSurfaces(throttle=0.7))
        state = backend.step(dt=0.01)

        assert state.time > initial_time
        assert np.all(np.isfinite(state.position))

    def test_set_get_controls(self):
        """Controls can be set."""
        backend = JSBSimBackend()
        backend.reset()

        controls = ControlSurfaces(elevator=0.2, throttle=0.8)
        backend.set_controls(controls)

        # Step and verify state changes
        backend.step(0.01)
        state = backend.get_state()

        # Should have non-zero velocity after thrust applied
        assert state.airspeed > 0

    def test_get_state(self):
        """get_state returns valid AircraftState."""
        backend = JSBSimBackend()
        backend.reset()

        state = backend.get_state()

        assert isinstance(state, AircraftState)
        assert len(state.position) == 3
        assert len(state.velocity) == 3
        assert len(state.attitude) == 3
        assert len(state.angular_rate) == 3

    def test_simulation_stability(self):
        """Simulation remains stable over time."""
        backend = JSBSimBackend()
        backend.reset()

        backend.set_controls(ControlSurfaces(throttle=0.6, elevator=0.05))

        # Run for 5 seconds
        for _ in range(500):
            state = backend.step(0.01)

            # Check for stability (no NaN or inf)
            assert np.all(np.isfinite(state.position))
            assert np.all(np.isfinite(state.velocity))
            assert np.all(np.isfinite(state.attitude))
            assert np.all(np.isfinite(state.angular_rate))

    def test_get_backend_type(self):
        """Returns correct backend type."""
        backend = JSBSimBackend()
        assert backend.get_backend_type() == "jsbsim"

    def test_get_dt_nominal(self):
        """Returns nominal timestep."""
        backend = JSBSimBackend({'dt_physics': 1/120.0})
        assert backend.get_dt_nominal() == pytest.approx(1/120.0)

    def test_get_info(self):
        """get_info returns backend details."""
        backend = JSBSimBackend()
        info = backend.get_info()

        assert info['backend_type'] == 'jsbsim'
        assert 'aircraft' in info
        assert 'dt_physics' in info
        assert 'jsbsim_version' in info

    def test_repr(self):
        """String representation is informative."""
        backend = JSBSimBackend({'aircraft': 'c172x'})
        repr_str = repr(backend)

        assert "JSBSimBackend" in repr_str
        assert "c172x" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
