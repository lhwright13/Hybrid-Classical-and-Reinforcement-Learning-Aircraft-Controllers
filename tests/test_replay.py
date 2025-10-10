"""Tests for multi-aircraft replay system."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from visualization.logger import TelemetryLogger
from visualization.replay import MultiAircraftReplay
from controllers.types import AircraftState, ControlCommand, ControlSurfaces, ControlMode


@pytest.fixture
def sample_log_file():
    """Create a sample log file with test data."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create log with 2 aircraft
    logger = TelemetryLogger(str(temp_path))
    logger.register_aircraft('001', metadata={'type': 'rc_plane'})
    logger.register_aircraft('002', metadata={'type': 'quadrotor'})

    # Log 10 seconds of data at 10 Hz
    dt = 0.1
    for i in range(100):
        t = i * dt

        for aircraft_id in ['001', '002']:
            state = AircraftState(
                time=t,
                position=np.array([t * 10, float(aircraft_id) * 50, -100.0]),
                velocity=np.array([10.0, 0.0, 0.0]),
                attitude=np.array([0.0, 0.0, t * 0.1]),
                angular_rate=np.zeros(3),
                altitude=100.0,
                airspeed=10.0
            )

            command = ControlCommand(
                mode=ControlMode.HSA,
                heading=t * 0.1,
                speed=10.0,
                altitude=100.0,
                throttle=0.5
            )

            surfaces = ControlSurfaces(
                elevator=0.0,
                aileron=0.0,
                rudder=0.0,
                throttle=0.5
            )

            logger.log_state(aircraft_id, state, t)
            logger.log_command(aircraft_id, command, t)
            logger.log_surfaces(aircraft_id, surfaces, t)

    logger.close()

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


def test_replay_creation(sample_log_file):
    """Test replay system creation."""
    replay = MultiAircraftReplay(str(sample_log_file))
    assert replay.filename == sample_log_file


def test_get_available_aircraft(sample_log_file):
    """Test getting available aircraft from log."""
    replay = MultiAircraftReplay(str(sample_log_file))

    aircraft = replay.get_available_aircraft()

    assert len(aircraft) == 2
    assert '001' in aircraft
    assert '002' in aircraft


def test_load_aircraft(sample_log_file):
    """Test loading aircraft data."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    assert '001' in replay.get_loaded_aircraft()
    assert '001' in replay.data
    assert len(replay.data['001'].states) == 100


def test_load_all_aircraft(sample_log_file):
    """Test loading all aircraft."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()  # None = load all

    assert len(replay.get_loaded_aircraft()) == 2


def test_load_with_commands_and_surfaces(sample_log_file):
    """Test loading with commands and surfaces."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'], load_commands=True, load_surfaces=True)

    data = replay.data['001']
    assert data.commands is not None
    assert data.surfaces is not None
    assert len(data.commands) == 100
    assert len(data.surfaces) == 100


def test_get_states_at_time(sample_log_file):
    """Test getting states at specific time."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    states = replay.get_states_at_time(5.0)

    assert len(states) == 2
    assert '001' in states
    assert '002' in states

    # Check state is approximately correct
    assert np.isclose(states['001'].position[0], 50.0, atol=5.0)


def test_get_states_interpolation(sample_log_file):
    """Test state interpolation between samples."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    # Get state at time between samples
    states = replay.get_states_at_time(0.55)  # Between 0.5 and 0.6

    state = states['001']
    # Position should be interpolated
    assert 5.0 < state.position[0] < 6.0


def test_get_states_boundary(sample_log_file):
    """Test getting states at time boundaries."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    # Before start - should get first state
    states = replay.get_states_at_time(-1.0)
    assert np.isclose(states['001'].position[0], 0.0, atol=0.1)

    # After end - should get last state
    states = replay.get_states_at_time(100.0)
    assert np.isclose(states['001'].position[0], 99.0, atol=5.0)


def test_play_generator(sample_log_file):
    """Test playback generator."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    times = list(replay.play(speed=1.0, dt=1.0, end_time=5.0))

    assert len(times) == 6  # 0, 1, 2, 3, 4, 5
    assert times[0] == 0.0
    assert times[-1] == 5.0


def test_play_with_speed(sample_log_file):
    """Test playback with different speeds."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    times = list(replay.play(speed=2.0, dt=1.0, end_time=5.0))

    # With 2x speed, time advances by 2 per step
    assert times[1] == 2.0


def test_seek(sample_log_file):
    """Test seeking to specific time."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    replay.seek(5.0)

    assert replay.get_current_time() == 5.0


def test_seek_clamp(sample_log_file):
    """Test that seek clamps to valid range."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    # Seek before start
    replay.seek(-10.0)
    assert replay.get_current_time() >= 0.0

    # Seek after end
    replay.seek(1000.0)
    assert replay.get_current_time() <= replay.get_duration()


def test_get_duration(sample_log_file):
    """Test getting duration."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    duration = replay.get_duration()

    assert np.isclose(duration, 9.9, atol=0.1)  # ~10 seconds


def test_export_csv(sample_log_file):
    """Test CSV export."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        csv_path = Path(f.name)

    try:
        replay.export_csv('001', str(csv_path))

        assert csv_path.exists()

        # Check file has data
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 100  # Header + 100 data rows
            assert 'time' in lines[0]  # Header

    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_export_csv_not_loaded(sample_log_file):
    """Test that exporting unloaded aircraft raises error."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    with pytest.raises(ValueError, match="not loaded"):
        replay.export_csv('999', 'output.csv')


def test_get_summary(sample_log_file):
    """Test getting replay summary."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    summary = replay.get_summary()

    assert summary['aircraft_count'] == 2
    assert '001' in summary['aircraft_ids']
    assert '002' in summary['aircraft_ids']
    assert summary['duration'] > 0
    assert summary['sample_counts']['001'] == 100


def test_repr(sample_log_file):
    """Test string representation."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft()

    repr_str = repr(replay)

    assert 'aircraft=2' in repr_str
    assert 'duration=' in repr_str


def test_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        MultiAircraftReplay('/nonexistent/file.h5')


def test_metadata_preserved(sample_log_file):
    """Test that metadata is preserved in replay."""
    replay = MultiAircraftReplay(str(sample_log_file))
    replay.load_aircraft(['001'])

    metadata = replay.data['001'].metadata

    assert metadata['type'] == 'rc_plane'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
