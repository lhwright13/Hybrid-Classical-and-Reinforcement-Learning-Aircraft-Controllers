"""Central registry for managing multiple aircraft in simulation.

This module provides a registry to track all aircraft, their status,
and automatically assign visualization properties (colors, markers).
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field


class AircraftStatus(Enum):
    """Aircraft operational status."""
    INACTIVE = 0      # Not yet started
    ACTIVE = 1        # Flying normally
    CRASHED = 2       # Crashed
    DISCONNECTED = 3  # Lost connection
    LANDED = 4        # Landed safely
    PAUSED = 5        # Paused


@dataclass
class AircraftInfo:
    """Information about a registered aircraft."""
    aircraft_id: str
    aircraft_type: str = "rc_plane"
    status: AircraftStatus = AircraftStatus.INACTIVE
    color: str = "#1f77b4"  # Hex color for visualization
    marker: str = "o"        # Marker style for plots
    metadata: Dict[str, Any] = field(default_factory=dict)


class AircraftRegistry:
    """Central registry for managing multiple aircraft.

    Features:
    - Register/unregister aircraft
    - Track aircraft status (active, crashed, disconnected, etc.)
    - Auto-assign colors and markers
    - Query active aircraft
    - Store aircraft metadata (config, gains, etc.)

    Example:
        >>> registry = AircraftRegistry()
        >>> registry.register("001", aircraft_type="rc_plane")
        >>> registry.register("002", aircraft_type="quadrotor")
        >>> registry.update_status("001", AircraftStatus.ACTIVE)
        >>> active = registry.get_active_aircraft()  # ["001"]
        >>> color = registry.get_color("001")
    """

    # Color palette (matches plotter and 3D viz)
    COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]

    # Marker styles
    MARKERS = ['o', 's', '^', 'v', 'D', '*', 'p', 'h', '+', 'x']

    def __init__(self):
        """Initialize aircraft registry."""
        self._aircraft: Dict[str, AircraftInfo] = {}
        self._next_color_idx = 0
        self._next_marker_idx = 0

    def register(
        self,
        aircraft_id: str,
        aircraft_type: str = "rc_plane",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AircraftInfo:
        """Register a new aircraft.

        Args:
            aircraft_id: Unique aircraft identifier
            aircraft_type: Type of aircraft (rc_plane, quadrotor, etc.)
            metadata: Optional metadata dictionary

        Returns:
            AircraftInfo object

        Raises:
            ValueError: If aircraft_id already registered
        """
        if aircraft_id in self._aircraft:
            raise ValueError(f"Aircraft {aircraft_id} already registered")

        # Auto-assign color and marker
        color = self.COLORS[self._next_color_idx % len(self.COLORS)]
        marker = self.MARKERS[self._next_marker_idx % len(self.MARKERS)]

        self._next_color_idx += 1
        self._next_marker_idx += 1

        # Create aircraft info
        info = AircraftInfo(
            aircraft_id=aircraft_id,
            aircraft_type=aircraft_type,
            status=AircraftStatus.INACTIVE,
            color=color,
            marker=marker,
            metadata=metadata or {}
        )

        self._aircraft[aircraft_id] = info
        return info

    def unregister(self, aircraft_id: str) -> None:
        """Remove aircraft from registry.

        Args:
            aircraft_id: Aircraft to remove
        """
        if aircraft_id in self._aircraft:
            del self._aircraft[aircraft_id]

    def update_status(self, aircraft_id: str, status: AircraftStatus) -> None:
        """Update aircraft status.

        Args:
            aircraft_id: Aircraft identifier
            status: New status

        Raises:
            KeyError: If aircraft not registered
        """
        if aircraft_id not in self._aircraft:
            raise KeyError(f"Aircraft {aircraft_id} not registered")

        self._aircraft[aircraft_id].status = status

    def get_status(self, aircraft_id: str) -> AircraftStatus:
        """Get aircraft status.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            Aircraft status
        """
        if aircraft_id not in self._aircraft:
            raise KeyError(f"Aircraft {aircraft_id} not registered")

        return self._aircraft[aircraft_id].status

    def get_info(self, aircraft_id: str) -> AircraftInfo:
        """Get aircraft info.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            AircraftInfo object
        """
        if aircraft_id not in self._aircraft:
            raise KeyError(f"Aircraft {aircraft_id} not registered")

        return self._aircraft[aircraft_id]

    def get_color(self, aircraft_id: str) -> str:
        """Get color assigned to aircraft.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            Hex color string
        """
        return self.get_info(aircraft_id).color

    def get_marker(self, aircraft_id: str) -> str:
        """Get marker style assigned to aircraft.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            Marker style string
        """
        return self.get_info(aircraft_id).marker

    def set_metadata(self, aircraft_id: str, key: str, value: Any) -> None:
        """Set metadata for aircraft.

        Args:
            aircraft_id: Aircraft identifier
            key: Metadata key
            value: Metadata value
        """
        info = self.get_info(aircraft_id)
        info.metadata[key] = value

    def get_metadata(self, aircraft_id: str, key: str, default: Any = None) -> Any:
        """Get metadata for aircraft.

        Args:
            aircraft_id: Aircraft identifier
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value
        """
        info = self.get_info(aircraft_id)
        return info.metadata.get(key, default)

    def get_all_aircraft(self) -> List[str]:
        """Get list of all registered aircraft IDs.

        Returns:
            List of aircraft IDs
        """
        return list(self._aircraft.keys())

    def get_active_aircraft(self) -> List[str]:
        """Get list of active aircraft IDs.

        Returns:
            List of active aircraft IDs
        """
        return [
            aid for aid, info in self._aircraft.items()
            if info.status == AircraftStatus.ACTIVE
        ]

    def get_aircraft_by_status(self, status: AircraftStatus) -> List[str]:
        """Get aircraft with specific status.

        Args:
            status: Status to filter by

        Returns:
            List of aircraft IDs
        """
        return [
            aid for aid, info in self._aircraft.items()
            if info.status == status
        ]

    def get_aircraft_by_type(self, aircraft_type: str) -> List[str]:
        """Get aircraft of specific type.

        Args:
            aircraft_type: Aircraft type to filter by

        Returns:
            List of aircraft IDs
        """
        return [
            aid for aid, info in self._aircraft.items()
            if info.aircraft_type == aircraft_type
        ]

    def count_active(self) -> int:
        """Count active aircraft.

        Returns:
            Number of active aircraft
        """
        return len(self.get_active_aircraft())

    def count_total(self) -> int:
        """Count total registered aircraft.

        Returns:
            Total number of aircraft
        """
        return len(self._aircraft)

    def is_registered(self, aircraft_id: str) -> bool:
        """Check if aircraft is registered.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            True if registered
        """
        return aircraft_id in self._aircraft

    def clear(self) -> None:
        """Remove all aircraft from registry."""
        self._aircraft.clear()
        self._next_color_idx = 0
        self._next_marker_idx = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary info
        """
        status_counts: Dict[AircraftStatus, int] = {status: 0 for status in AircraftStatus}
        type_counts: Dict[str, int] = {}

        for info in self._aircraft.values():
            status_counts[info.status] += 1
            type_counts[info.aircraft_type] = type_counts.get(info.aircraft_type, 0) + 1

        return {
            'total': len(self._aircraft),
            'active': status_counts[AircraftStatus.ACTIVE],
            'status_counts': status_counts,
            'type_counts': type_counts,
            'aircraft_ids': list(self._aircraft.keys())
        }

    def __len__(self) -> int:
        """Get number of registered aircraft."""
        return len(self._aircraft)

    def __contains__(self, aircraft_id: str) -> bool:
        """Check if aircraft is registered."""
        return aircraft_id in self._aircraft

    def __repr__(self) -> str:
        """String representation."""
        active = self.count_active()
        total = self.count_total()
        return f"AircraftRegistry(total={total}, active={active})"
