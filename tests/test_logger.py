"""Tests for multi-aircraft telemetry logger."""

import pytest
import numpy as np
import h5py
from pathlib import Path
import tempfile

from visualization.logger import TelemetryLogger
from controllers.types import AircraftState, ControlCommand, ControlSurfaces, ControlMode


@pytest.fixture
def temp_log_file():
    """Create temporary log file."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_state():
    """Create sample aircraft state."""
    return AircraftState(
        time=1.0,
        position=np.array([100.0, 200.0, -50.0]),
        velocity=np.array([20.0, 0.0, 0.0]),
        attitude=np.array([0.1, 0.05, 1.57]),
        angular_rate=np.array([0.0, 0.0, 0.0]),
        altitude=50.0,
        airspeed=20.0
    )


@pytest.fixture
def sample_command():
    """Create sample control command."""
    return ControlCommand(
        mode=ControlMode.HSA,
        heading=np.radians(90),
        speed=20.0,
        altitude=100.0,
        throttle=0.7
    )


@pytest.fixture
def sample_surfaces():
    """Create sample control surfaces."""
    return ControlSurfaces(
        elevator=-0.1,
        aileron=0.05,
        rudder=0.0,
        throttle=0.7
    )


def test_logger_creation(temp_log_file):
    """Test logger creation and file creation."""
    logger = TelemetryLogger(str(temp_log_file))
    logger.close()

    assert temp_log_file.exists()

    # Check HDF5 structure
    with h5py.File(temp_log_file, 'r') as f:
        assert 'fleet' in f
        assert 'created' in f['fleet'].attrs


def test_register_aircraft(temp_log_file):
    """Test aircraft registration."""
    logger = TelemetryLogger(str(temp_log_file))

    metadata = {'type': 'rc_plane', 'config': 'default'}
    logger.register_aircraft('001', metadata=metadata)

    assert '001' in logger.get_aircraft_list()

    logger.close()

    # Check HDF5
    with h5py.File(temp_log_file, 'r') as f:
        assert 'aircraft_001' in f
        group = f['aircraft_001']
        assert group.attrs['type'] == 'rc_plane'


def test_duplicate_registration(temp_log_file):
    """Test that duplicate registration raises error."""
    logger = TelemetryLogger(str(temp_log_file))
    logger.register_aircraft('001')

    with pytest.raises(ValueError, match="already registered"):
        logger.register_aircraft('001')

    logger.close()


def test_log_state(temp_log_file, sample_state):
    """Test state logging."""
    logger = TelemetryLogger(str(temp_log_file), buffer_size=5)
    logger.register_aircraft('001')

    # Log multiple states
    for i in range(10):
        state = AircraftState(
            time=i * 0.01,
            position=np.array([i, i * 2, -50.0]),
            velocity=np.array([20.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3),
            altitude=50.0,
            airspeed=20.0
        )
        logger.log_state('001', state)

    logger.close()

    # Verify data
    with h5py.File(temp_log_file, 'r') as f:
        group = f['aircraft_001']
        assert 'time' in group
        assert len(group['time']) == 10
        assert 'pos_n' in group
        assert 'altitude' in group
        assert np.allclose(group['time'][:], np.arange(10) * 0.01)


def test_log_command(temp_log_file, sample_command):
    """Test command logging."""
    logger = TelemetryLogger(str(temp_log_file), buffer_size=5)
    logger.register_aircraft('001')

    for i in range(10):
        logger.log_command('001', sample_command, i * 0.01)

    logger.close()

    with h5py.File(temp_log_file, 'r') as f:
        group = f['aircraft_001/commands']
        assert 'time' in group
        assert 'mode' in group
        assert len(group['time']) == 10


def test_log_surfaces(temp_log_file, sample_surfaces):
    """Test surface logging."""
    logger = TelemetryLogger(str(temp_log_file), buffer_size=5)
    logger.register_aircraft('001')

    for i in range(10):
        logger.log_surfaces('001', sample_surfaces, i * 0.01)

    logger.close()

    with h5py.File(temp_log_file, 'r') as f:
        group = f['aircraft_001/surfaces']
        assert 'elevator' in group
        assert len(group['elevator']) == 10


def test_multi_aircraft_logging(temp_log_file, sample_state):
    """Test logging multiple aircraft."""
    logger = TelemetryLogger(str(temp_log_file))

    aircraft_ids = ['001', '002', '003']
    for aircraft_id in aircraft_ids:
        logger.register_aircraft(aircraft_id)

    # Log data for all aircraft
    for i in range(10):
        for aircraft_id in aircraft_ids:
            logger.log_state(aircraft_id, sample_state, i * 0.01)

    logger.close()

    # Verify all aircraft have data
    with h5py.File(temp_log_file, 'r') as f:
        for aircraft_id in aircraft_ids:
            group_name = f'aircraft_{aircraft_id}'
            assert group_name in f
            assert len(f[group_name]['time']) == 10


def test_buffering(temp_log_file, sample_state):
    """Test that buffering works correctly."""
    buffer_size = 5
    logger = TelemetryLogger(str(temp_log_file), buffer_size=buffer_size)
    logger.register_aircraft('001')

    # Log less than buffer size - should not flush yet
    for i in range(buffer_size - 1):
        logger.log_state('001', sample_state, i * 0.01)

    # Check file - should have no data yet (only after flush)
    with h5py.File(temp_log_file, 'r') as f:
        if 'time' in f['aircraft_001']:
            assert len(f['aircraft_001']['time']) == 0

    # Log one more to trigger flush
    logger.log_state('001', sample_state, (buffer_size) * 0.01)

    # Now data should be there
    with h5py.File(temp_log_file, 'r') as f:
        assert len(f['aircraft_001']['time']) == buffer_size

    logger.close()


def test_context_manager(temp_log_file, sample_state):
    """Test logger as context manager."""
    with TelemetryLogger(str(temp_log_file)) as logger:
        logger.register_aircraft('001')
        logger.log_state('001', sample_state)

    # File should be closed and data flushed
    with h5py.File(temp_log_file, 'r') as f:
        assert 'aircraft_001' in f
        assert 'closed' in f['fleet'].attrs


def test_compression(temp_log_file, sample_state):
    """Test that compression works."""
    # Log with compression
    logger = TelemetryLogger(str(temp_log_file), compression='gzip', compression_opts=4)
    logger.register_aircraft('001')

    for i in range(100):
        logger.log_state('001', sample_state, i * 0.01)

    logger.close()

    # File should be relatively small due to compression
    file_size = temp_log_file.stat().st_size
    assert file_size < 60000  # Should be much smaller than uncompressed


def test_unregistered_aircraft_error(temp_log_file, sample_state):
    """Test that logging unregistered aircraft raises error."""
    logger = TelemetryLogger(str(temp_log_file))

    with pytest.raises(ValueError, match="not registered"):
        logger.log_state('999', sample_state)

    logger.close()


def test_fleet_metadata(temp_log_file):
    """Test fleet metadata is stored correctly."""
    with TelemetryLogger(str(temp_log_file)) as logger:
        logger.register_aircraft('001')
        logger.register_aircraft('002')

    with h5py.File(temp_log_file, 'r') as f:
        fleet = f['fleet']
        assert fleet.attrs['num_aircraft'] == 2
        assert '001' in fleet.attrs['aircraft_ids']
        assert '002' in fleet.attrs['aircraft_ids']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
