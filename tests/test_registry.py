"""Tests for aircraft registry."""

import pytest

from interfaces.aircraft_registry import AircraftRegistry, AircraftStatus


def test_registry_creation():
    """Test registry creation."""
    registry = AircraftRegistry()
    assert len(registry) == 0
    assert registry.count_total() == 0
    assert registry.count_active() == 0


def test_register_aircraft():
    """Test aircraft registration."""
    registry = AircraftRegistry()

    info = registry.register('001', aircraft_type='rc_plane')

    assert '001' in registry
    assert registry.is_registered('001')
    assert info.aircraft_id == '001'
    assert info.aircraft_type == 'rc_plane'
    assert info.status == AircraftStatus.INACTIVE


def test_register_multiple_aircraft():
    """Test registering multiple aircraft."""
    registry = AircraftRegistry()

    for i in range(5):
        registry.register(f'{i:03d}', aircraft_type='rc_plane')

    assert len(registry) == 5
    assert registry.count_total() == 5


def test_duplicate_registration_error():
    """Test that duplicate registration raises error."""
    registry = AircraftRegistry()
    registry.register('001')

    with pytest.raises(ValueError, match="already registered"):
        registry.register('001')


def test_unregister():
    """Test aircraft unregistration."""
    registry = AircraftRegistry()
    registry.register('001')

    assert '001' in registry

    registry.unregister('001')

    assert '001' not in registry


def test_color_assignment():
    """Test automatic color assignment."""
    registry = AircraftRegistry()

    colors = []
    for i in range(12):  # More than color palette size
        registry.register(f'{i:03d}')
        colors.append(registry.get_color(f'{i:03d}'))

    # First 10 should be unique
    assert len(set(colors[:10])) == 10

    # After 10, colors repeat
    assert colors[0] == colors[10]


def test_marker_assignment():
    """Test automatic marker assignment."""
    registry = AircraftRegistry()

    markers = []
    for i in range(12):
        registry.register(f'{i:03d}')
        markers.append(registry.get_marker(f'{i:03d}'))

    # First 10 should be unique
    assert len(set(markers[:10])) == 10


def test_status_update():
    """Test aircraft status updates."""
    registry = AircraftRegistry()
    registry.register('001')

    assert registry.get_status('001') == AircraftStatus.INACTIVE

    registry.update_status('001', AircraftStatus.ACTIVE)

    assert registry.get_status('001') == AircraftStatus.ACTIVE


def test_get_active_aircraft():
    """Test getting active aircraft."""
    registry = AircraftRegistry()

    registry.register('001')
    registry.register('002')
    registry.register('003')

    registry.update_status('001', AircraftStatus.ACTIVE)
    registry.update_status('002', AircraftStatus.ACTIVE)
    registry.update_status('003', AircraftStatus.CRASHED)

    active = registry.get_active_aircraft()

    assert len(active) == 2
    assert '001' in active
    assert '002' in active
    assert '003' not in active


def test_get_aircraft_by_status():
    """Test filtering aircraft by status."""
    registry = AircraftRegistry()

    registry.register('001')
    registry.register('002')
    registry.register('003')

    registry.update_status('001', AircraftStatus.ACTIVE)
    registry.update_status('002', AircraftStatus.CRASHED)
    registry.update_status('003', AircraftStatus.LANDED)

    crashed = registry.get_aircraft_by_status(AircraftStatus.CRASHED)
    assert crashed == ['002']


def test_get_aircraft_by_type():
    """Test filtering aircraft by type."""
    registry = AircraftRegistry()

    registry.register('001', aircraft_type='rc_plane')
    registry.register('002', aircraft_type='quadrotor')
    registry.register('003', aircraft_type='rc_plane')

    planes = registry.get_aircraft_by_type('rc_plane')

    assert len(planes) == 2
    assert '001' in planes
    assert '003' in planes


def test_metadata():
    """Test aircraft metadata storage."""
    registry = AircraftRegistry()
    registry.register('001')

    registry.set_metadata('001', 'config', 'aggressive')
    registry.set_metadata('001', 'battery', 100)

    assert registry.get_metadata('001', 'config') == 'aggressive'
    assert registry.get_metadata('001', 'battery') == 100
    assert registry.get_metadata('001', 'missing', 'default') == 'default'


def test_get_info():
    """Test getting aircraft info."""
    registry = AircraftRegistry()
    info = registry.register('001', aircraft_type='quadrotor', metadata={'test': 'value'})

    retrieved_info = registry.get_info('001')

    assert retrieved_info.aircraft_id == '001'
    assert retrieved_info.aircraft_type == 'quadrotor'
    assert retrieved_info.metadata['test'] == 'value'


def test_get_info_not_registered():
    """Test getting info for unregistered aircraft raises error."""
    registry = AircraftRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get_info('999')


def test_clear():
    """Test clearing registry."""
    registry = AircraftRegistry()

    for i in range(5):
        registry.register(f'{i:03d}')

    assert len(registry) == 5

    registry.clear()

    assert len(registry) == 0


def test_get_summary():
    """Test getting registry summary."""
    registry = AircraftRegistry()

    registry.register('001', aircraft_type='rc_plane')
    registry.register('002', aircraft_type='quadrotor')
    registry.register('003', aircraft_type='rc_plane')

    registry.update_status('001', AircraftStatus.ACTIVE)
    registry.update_status('002', AircraftStatus.ACTIVE)

    summary = registry.get_summary()

    assert summary['total'] == 3
    assert summary['active'] == 2
    assert summary['type_counts']['rc_plane'] == 2
    assert summary['type_counts']['quadrotor'] == 1
    assert '001' in summary['aircraft_ids']


def test_count_methods():
    """Test count methods."""
    registry = AircraftRegistry()

    for i in range(5):
        registry.register(f'{i:03d}')

    registry.update_status('000', AircraftStatus.ACTIVE)
    registry.update_status('001', AircraftStatus.ACTIVE)

    assert registry.count_total() == 5
    assert registry.count_active() == 2


def test_repr():
    """Test string representation."""
    registry = AircraftRegistry()

    registry.register('001')
    registry.register('002')
    registry.update_status('001', AircraftStatus.ACTIVE)

    repr_str = repr(registry)

    assert 'total=2' in repr_str
    assert 'active=1' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
