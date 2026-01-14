"""Multi-aircraft trajectory plotter."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path


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
        self.window_size = window_size

        # Position history: {aircraft_id: {'north': [], 'east': [], 'alt': []}}
        self.trajectories: Dict[str, Dict[str, List[float]]] = {}
        for aid in aircraft_ids:
            self.trajectories[aid] = {
                'north': [],
                'east': [],
                'alt': [],
                'time': []
            }

        self.waypoints: List[Any] = []

        # Colors for different aircraft
        self.colors = plt.cm.tab10.colors

    def update(self, aircraft_id: str, state: Any):
        """Update trajectory with new state.

        Args:
            aircraft_id: Aircraft identifier
            state: AircraftState object
        """
        if aircraft_id not in self.trajectories:
            self.trajectories[aircraft_id] = {
                'north': [], 'east': [], 'alt': [], 'time': []
            }

        traj = self.trajectories[aircraft_id]

        # Extract position (NED frame)
        north = state.position[0] if hasattr(state, 'position') else state.north
        east = state.position[1] if hasattr(state, 'position') else state.east
        alt = state.altitude if hasattr(state, 'altitude') else -state.position[2]

        traj['north'].append(north)
        traj['east'].append(east)
        traj['alt'].append(alt)
        traj['time'].append(state.time)

        # Trim to window size
        if len(traj['north']) > self.window_size:
            for key in traj:
                traj[key] = traj[key][-self.window_size:]

    def set_waypoints(self, waypoints: List[Any]):
        """Set waypoints to display on plot.

        Args:
            waypoints: List of Waypoint objects
        """
        self.waypoints = waypoints

    def plot(self, show: bool = True) -> plt.Figure:
        """Generate trajectory plot.

        Args:
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Top-down view (North vs East)
        ax1 = axes[0]
        for i, (aid, traj) in enumerate(self.trajectories.items()):
            if traj['north']:
                color = self.colors[i % len(self.colors)]
                ax1.plot(traj['east'], traj['north'], '-', color=color,
                        linewidth=2, label=f'Aircraft {aid}', alpha=0.8)
                # Mark start and end
                ax1.plot(traj['east'][0], traj['north'][0], 'o',
                        color=color, markersize=10)
                ax1.plot(traj['east'][-1], traj['north'][-1], 's',
                        color=color, markersize=10)

        # Plot waypoints
        if self.waypoints:
            wp_north = [wp.north for wp in self.waypoints]
            wp_east = [wp.east for wp in self.waypoints]
            ax1.plot(wp_east, wp_north, 'r--', linewidth=1, alpha=0.5, label='Planned')
            ax1.scatter(wp_east, wp_north, c='red', s=100, marker='x',
                       linewidths=2, zorder=5)
            # Label waypoints
            for i, wp in enumerate(self.waypoints):
                ax1.annotate(f'WP{i+1}', (wp.east, wp.north),
                           textcoords='offset points', xytext=(5, 5),
                           fontsize=8, color='red')

        ax1.set_xlabel('East (m)', fontsize=12)
        ax1.set_ylabel('North (m)', fontsize=12)
        ax1.set_title('Trajectory - Top View', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: Altitude over time
        ax2 = axes[1]
        for i, (aid, traj) in enumerate(self.trajectories.items()):
            if traj['time']:
                color = self.colors[i % len(self.colors)]
                ax2.plot(traj['time'], traj['alt'], '-', color=color,
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
            if aircraft_id in self.trajectories:
                for key in self.trajectories[aircraft_id]:
                    self.trajectories[aircraft_id][key] = []
        else:
            for aid in self.trajectories:
                for key in self.trajectories[aid]:
                    self.trajectories[aid][key] = []
