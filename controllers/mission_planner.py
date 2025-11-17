"""Mission Planner - Manages waypoint sequences for autonomous navigation."""

from enum import Enum
from typing import List, Optional
import numpy as np

from controllers.types import Waypoint, ControlCommand, ControlMode, AircraftState


class MissionState(Enum):
    """Mission state enumeration."""
    IDLE = "idle"           # Mission not started
    ACTIVE = "active"       # Mission in progress
    COMPLETE = "complete"   # All waypoints reached
    ABORTED = "aborted"     # Mission aborted


class MissionPlanner:
    """Mission planner for waypoint navigation.

    Manages a sequence of waypoints and provides guidance commands
    to navigate through them. Automatically advances to the next
    waypoint when the current one is reached.

    Features:
    - Waypoint sequencing with auto-advance
    - Mission state tracking
    - Progress reporting (current waypoint, completion %)
    - Configurable acceptance radius
    - Mission statistics (total distance, waypoints reached)

    Example:
        >>> waypoints = [
        ...     Waypoint.from_ned(100, 0, -100),
        ...     Waypoint.from_ned(100, 100, -100),
        ...     Waypoint.from_ned(0, 100, -100),
        ... ]
        >>> planner = MissionPlanner(waypoints, acceptance_radius=15.0)
        >>> planner.start()
        >>>
        >>> # In control loop:
        >>> if not planner.is_complete():
        ...     command = planner.get_waypoint_command()
        ...     planner.update(state)
    """

    def __init__(
        self,
        waypoints: List[Waypoint],
        acceptance_radius: float = 10.0,
        default_speed: Optional[float] = None
    ):
        """Initialize mission planner.

        Args:
            waypoints: List of waypoints to navigate (in order)
            acceptance_radius: Distance threshold for waypoint acceptance (meters)
            default_speed: Default speed if waypoint doesn't specify (m/s)
        """
        if not waypoints:
            raise ValueError("Mission must have at least one waypoint")

        self.waypoints = waypoints
        self.acceptance_radius = acceptance_radius
        self.default_speed = default_speed

        # Mission state
        self.state = MissionState.IDLE
        self.current_waypoint_index = 0
        self.waypoints_reached = 0

        # Statistics
        self.waypoint_arrival_times = []
        self.waypoint_distances = []
        self.mission_start_time: Optional[float] = None
        self.mission_end_time: Optional[float] = None

    def start(self):
        """Start the mission."""
        if self.state == MissionState.IDLE:
            self.state = MissionState.ACTIVE
            self.current_waypoint_index = 0
            self.waypoints_reached = 0

    def abort(self):
        """Abort the mission."""
        self.state = MissionState.ABORTED

    def reset(self):
        """Reset mission to initial state."""
        self.state = MissionState.IDLE
        self.current_waypoint_index = 0
        self.waypoints_reached = 0
        self.waypoint_arrival_times = []
        self.waypoint_distances = []
        self.mission_start_time = None
        self.mission_end_time = None

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint.

        Returns:
            Current waypoint, or None if mission not active
        """
        if self.state != MissionState.ACTIVE:
            return None

        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]

        return None

    def get_waypoint_command(self) -> Optional[ControlCommand]:
        """Get control command for current waypoint.

        Returns:
            ControlCommand with WAYPOINT mode, or None if no active waypoint
        """
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return None

        return ControlCommand(
            mode=ControlMode.WAYPOINT,
            waypoint=waypoint
        )

    def update(self, state: AircraftState) -> bool:
        """Update mission state based on aircraft position.

        Checks if current waypoint has been reached and advances
        to next waypoint if so.

        Args:
            state: Current aircraft state

        Returns:
            True if waypoint was reached and advanced
        """
        if self.state != MissionState.ACTIVE:
            return False

        # Initialize mission start time
        if self.mission_start_time is None:
            self.mission_start_time = state.time

        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False

        # Check if waypoint reached
        if self._is_waypoint_reached(state, waypoint):
            # Record arrival
            self.waypoint_arrival_times.append(state.time)
            self.waypoints_reached += 1

            # Calculate distance error
            distance = self._calculate_distance_to_waypoint(state, waypoint)
            self.waypoint_distances.append(distance)

            # Advance to next waypoint
            self.current_waypoint_index += 1

            # Check if mission complete
            if self.current_waypoint_index >= len(self.waypoints):
                self.state = MissionState.COMPLETE
                self.mission_end_time = state.time

            return True

        return False

    def _is_waypoint_reached(self, state: AircraftState, waypoint: Waypoint) -> bool:
        """Check if waypoint has been reached.

        Args:
            state: Current aircraft state
            waypoint: Target waypoint

        Returns:
            True if within acceptance radius
        """
        distance = self._calculate_distance_to_waypoint(state, waypoint)
        return distance < self.acceptance_radius

    def _calculate_distance_to_waypoint(
        self,
        state: AircraftState,
        waypoint: Waypoint
    ) -> float:
        """Calculate 3D distance from aircraft to waypoint.

        Args:
            state: Current aircraft state
            waypoint: Target waypoint

        Returns:
            Distance in meters
        """
        error = np.array([
            waypoint.north - state.north,
            waypoint.east - state.east,
            waypoint.down - state.down
        ])
        return np.linalg.norm(error)

    def get_distance_to_current_waypoint(self, state: AircraftState) -> Optional[float]:
        """Get distance to current waypoint.

        Args:
            state: Current aircraft state

        Returns:
            Distance in meters, or None if no active waypoint
        """
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return None

        return self._calculate_distance_to_waypoint(state, waypoint)

    def get_progress_percentage(self) -> float:
        """Get mission progress as percentage.

        Returns:
            Progress from 0.0 to 100.0
        """
        if len(self.waypoints) == 0:
            return 100.0

        return (self.waypoints_reached / len(self.waypoints)) * 100.0

    def get_total_mission_distance(self) -> float:
        """Calculate total mission distance (sum of leg distances).

        Returns:
            Total distance in meters
        """
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]

            leg_distance = np.linalg.norm([
                wp2.north - wp1.north,
                wp2.east - wp1.east,
                wp2.down - wp1.down
            ])
            total += leg_distance

        return total

    def get_mission_duration(self) -> Optional[float]:
        """Get total mission duration.

        Returns:
            Duration in seconds, or None if mission not complete
        """
        if self.mission_start_time is None or self.mission_end_time is None:
            return None

        return self.mission_end_time - self.mission_start_time

    def is_active(self) -> bool:
        """Check if mission is active."""
        return self.state == MissionState.ACTIVE

    def is_complete(self) -> bool:
        """Check if mission is complete."""
        return self.state == MissionState.COMPLETE

    def is_aborted(self) -> bool:
        """Check if mission was aborted."""
        return self.state == MissionState.ABORTED

    def get_summary(self) -> dict:
        """Get mission summary statistics.

        Returns:
            Dictionary with mission statistics
        """
        return {
            'state': self.state.value,
            'total_waypoints': len(self.waypoints),
            'waypoints_reached': self.waypoints_reached,
            'progress_percent': self.get_progress_percentage(),
            'current_waypoint_index': self.current_waypoint_index,
            'total_distance_m': self.get_total_mission_distance(),
            'mission_duration_s': self.get_mission_duration(),
            'acceptance_radius_m': self.acceptance_radius,
            'waypoint_arrival_times': self.waypoint_arrival_times,
            'waypoint_distances': self.waypoint_distances
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"MissionPlanner(state={self.state.value}, "
                f"waypoints={self.waypoints_reached}/{len(self.waypoints)}, "
                f"progress={self.get_progress_percentage():.1f}%)")
