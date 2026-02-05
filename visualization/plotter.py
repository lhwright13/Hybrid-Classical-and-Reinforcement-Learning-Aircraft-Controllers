"""Multi-aircraft trajectory plotter."""

import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Optional, Any
from pathlib import Path

_TRAJ_KEYS = ('north', 'east', 'alt', 'time')


class MultiAircraftPlotter:
    """Real-time plotter for multi-aircraft trajectories.

    Accumulates position data and can generate 2D trajectory plots.
    """

    def __init__(self, aircraft_ids: List[str], window_size: int = 1000):
        """Initialize plotter.

        Args:
            aircraft_ids: List of aircraft identifiers to track
            window_size: Maximum number of points to store per aircraft
        """
        self.aircraft_ids = aircraft_ids
        self._window_size = window_size

        self._trajectories: Dict[str, Dict[str, deque]] = {}
        for aid in aircraft_ids:
            self._trajectories[aid] = self._make_trajectory()

        self._waypoints: List[Any] = []
        self._colors = plt.cm.tab10.colors

    def _make_trajectory(self) -> Dict[str, deque]:
        return {key: deque(maxlen=self._window_size) for key in _TRAJ_KEYS}

    def update(self, aircraft_id: str, state: Any):
        """Update trajectory with new state.

        Args:
            aircraft_id: Aircraft identifier
            state: AircraftState object
        """
        if aircraft_id not in self._trajectories:
            self._trajectories[aircraft_id] = self._make_trajectory()

        traj = self._trajectories[aircraft_id]

        north = state.position[0] if hasattr(state, 'position') else state.north
        east = state.position[1] if hasattr(state, 'position') else state.east
        alt = state.altitude if hasattr(state, 'altitude') else -state.position[2]

        traj['north'].append(north)
        traj['east'].append(east)
        traj['alt'].append(alt)
        traj['time'].append(state.time)

    def set_waypoints(self, waypoints: List[Any]):
        """Set waypoints to display on plot.

        Args:
            waypoints: List of Waypoint objects
        """
        self._waypoints = waypoints

    def plot(self, show: bool = True) -> plt.Figure:
        """Generate trajectory plot.

        Args:
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        for i, (aid, traj) in enumerate(self._trajectories.items()):
            if traj['north']:
                color = self._colors[i % len(self._colors)]
                ax1.plot(list(traj['east']), list(traj['north']), '-', color=color,
                        linewidth=2, label=f'Aircraft {aid}', alpha=0.8)
                ax1.plot(traj['east'][0], traj['north'][0], 'o',
                        color=color, markersize=10)
                ax1.plot(traj['east'][-1], traj['north'][-1], 's',
                        color=color, markersize=10)

        if self._waypoints:
            wp_north = [wp.north for wp in self._waypoints]
            wp_east = [wp.east for wp in self._waypoints]
            ax1.plot(wp_east, wp_north, 'r--', linewidth=1, alpha=0.5, label='Planned')
            ax1.scatter(wp_east, wp_north, c='red', s=100, marker='x',
                       linewidths=2, zorder=5)
            for i, wp in enumerate(self._waypoints):
                ax1.annotate(f'WP{i+1}', (wp.east, wp.north),
                           textcoords='offset points', xytext=(5, 5),
                           fontsize=8, color='red')

        ax1.set_xlabel('East (m)', fontsize=12)
        ax1.set_ylabel('North (m)', fontsize=12)
        ax1.set_title('Trajectory - Top View', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        ax2 = axes[1]
        for i, (aid, traj) in enumerate(self._trajectories.items()):
            if traj['time']:
                color = self._colors[i % len(self._colors)]
                ax2.plot(list(traj['time']), list(traj['alt']), '-', color=color,
                        linewidth=2, label=f'Aircraft {aid}')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Altitude (m)', fontsize=12)
        ax2.set_title('Altitude Profile', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def save(self, filepath: str, dpi: int = 150):
        """Save trajectory plot to file.

        Args:
            filepath: Output file path
            dpi: Image resolution
        """
        fig = self.plot(show=False)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved trajectory plot: {filepath}")

    def clear(self, aircraft_id: Optional[str] = None):
        """Clear trajectory data.

        Args:
            aircraft_id: Specific aircraft to clear, or None for all
        """
        if aircraft_id:
            if aircraft_id in self._trajectories:
                self._trajectories[aircraft_id] = self._make_trajectory()
        else:
            for aid in self._trajectories:
                self._trajectories[aid] = self._make_trajectory()
