"""Basic matplotlib visualization for simulation telemetry.

This module provides simple real-time and post-processing plotting
for aircraft state variables, including multi-aircraft support.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Dict
from collections import defaultdict
from controllers.types import AircraftState


class TelemetryPlotter:
    """Real-time telemetry plotter using matplotlib.

    Displays key aircraft state variables in real-time during simulation.

    Example:
        >>> plotter = TelemetryPlotter()
        >>> for i in range(100):
        ...     state = backend.step(0.01)
        ...     plotter.update(state)
        >>> plotter.show()
    """

    def __init__(self, window_size: int = 500):
        """Initialize plotter.

        Args:
            window_size: Number of time steps to display (rolling window)
        """
        self.window_size = window_size

        # Data buffers
        self.time = []
        self.altitude = []
        self.airspeed = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.position_n = []
        self.position_e = []

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8))
        self.fig.suptitle('Aircraft Telemetry', fontsize=16)

        # Subplot titles
        titles = [
            'Altitude (m)',
            'Airspeed (m/s)',
            'Position (NED)',
            'Attitude (deg)',
            'Roll (deg)',
            'Pitch (deg)'
        ]

        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def update(self, state: AircraftState) -> None:
        """Update plots with new state.

        Args:
            state: Current aircraft state
        """
        # Append data
        self.time.append(state.time)
        self.altitude.append(state.altitude)
        self.airspeed.append(state.airspeed)
        self.roll.append(np.degrees(state.attitude[0]))
        self.pitch.append(np.degrees(state.attitude[1]))
        self.yaw.append(np.degrees(state.attitude[2]))
        self.position_n.append(state.position[0])
        self.position_e.append(state.position[1])

        # Keep only recent data (rolling window)
        if len(self.time) > self.window_size:
            self.time = self.time[-self.window_size:]
            self.altitude = self.altitude[-self.window_size:]
            self.airspeed = self.airspeed[-self.window_size:]
            self.roll = self.roll[-self.window_size:]
            self.pitch = self.pitch[-self.window_size:]
            self.yaw = self.yaw[-self.window_size:]
            self.position_n = self.position_n[-self.window_size:]
            self.position_e = self.position_e[-self.window_size:]

    def plot(self) -> None:
        """Redraw all plots with current data."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Altitude
        self.axes[0, 0].plot(self.time, self.altitude, 'b-', linewidth=2)
        self.axes[0, 0].set_ylabel('Altitude (m)')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].set_title('Altitude')

        # Airspeed
        self.axes[0, 1].plot(self.time, self.airspeed, 'r-', linewidth=2)
        self.axes[0, 1].set_ylabel('Airspeed (m/s)')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_title('Airspeed')

        # Position (top-down view)
        self.axes[1, 0].plot(self.position_e, self.position_n, 'g-', linewidth=2)
        self.axes[1, 0].plot(self.position_e[-1], self.position_n[-1], 'ro', markersize=8)
        self.axes[1, 0].set_ylabel('North (m)')
        self.axes[1, 0].set_xlabel('East (m)')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].set_title('Position (Top View)')
        self.axes[1, 0].axis('equal')

        # Attitude
        self.axes[1, 1].plot(self.time, self.roll, 'r-', label='Roll', linewidth=2)
        self.axes[1, 1].plot(self.time, self.pitch, 'g-', label='Pitch', linewidth=2)
        self.axes[1, 1].plot(self.time, self.yaw, 'b-', label='Yaw', linewidth=2)
        self.axes[1, 1].set_ylabel('Angle (deg)')
        self.axes[1, 1].set_xlabel('Time (s)')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].set_title('Attitude')

        # Roll detail
        self.axes[2, 0].plot(self.time, self.roll, 'r-', linewidth=2)
        self.axes[2, 0].set_ylabel('Roll (deg)')
        self.axes[2, 0].set_xlabel('Time (s)')
        self.axes[2, 0].grid(True, alpha=0.3)
        self.axes[2, 0].set_title('Roll')

        # Pitch detail
        self.axes[2, 1].plot(self.time, self.pitch, 'g-', linewidth=2)
        self.axes[2, 1].set_ylabel('Pitch (deg)')
        self.axes[2, 1].set_xlabel('Time (s)')
        self.axes[2, 1].grid(True, alpha=0.3)
        self.axes[2, 1].set_title('Pitch')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def show(self) -> None:
        """Display plots (blocking)."""
        self.plot()
        plt.show()

    def save(self, filename: str) -> None:
        """Save current plot to file.

        Args:
            filename: Output filename (e.g., 'telemetry.png')
        """
        self.plot()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")


def plot_telemetry_history(states: List[AircraftState], filename: Optional[str] = None) -> None:
    """Plot telemetry history from list of states.

    This is a convenience function for post-processing simulation data.

    Args:
        states: List of aircraft states
        filename: If provided, save to file instead of showing

    Example:
        >>> states = []
        >>> for i in range(500):
        ...     state = backend.step(0.01)
        ...     states.append(state)
        >>> plot_telemetry_history(states, 'simulation_results.png')
    """
    # Extract data
    time = [s.time for s in states]
    altitude = [s.altitude for s in states]
    airspeed = [s.airspeed for s in states]
    roll = [np.degrees(s.attitude[0]) for s in states]
    pitch = [np.degrees(s.attitude[1]) for s in states]
    yaw = [np.degrees(s.attitude[2]) for s in states]
    pos_n = [s.position[0] for s in states]
    pos_e = [s.position[1] for s in states]

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Aircraft Telemetry History', fontsize=16)

    # Altitude
    axes[0, 0].plot(time, altitude, 'b-', linewidth=2)
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Altitude')

    # Airspeed
    axes[0, 1].plot(time, airspeed, 'r-', linewidth=2)
    axes[0, 1].set_ylabel('Airspeed (m/s)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Airspeed')

    # Position (top-down view)
    axes[1, 0].plot(pos_e, pos_n, 'g-', linewidth=2)
    axes[1, 0].plot(pos_e[0], pos_n[0], 'go', markersize=10, label='Start')
    axes[1, 0].plot(pos_e[-1], pos_n[-1], 'ro', markersize=10, label='End')
    axes[1, 0].set_ylabel('North (m)')
    axes[1, 0].set_xlabel('East (m)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Position (Top View)')
    axes[1, 0].legend()
    axes[1, 0].axis('equal')

    # Attitude
    axes[1, 1].plot(time, roll, 'r-', label='Roll', linewidth=2)
    axes[1, 1].plot(time, pitch, 'g-', label='Pitch', linewidth=2)
    axes[1, 1].plot(time, yaw, 'b-', label='Yaw', linewidth=2)
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Attitude')

    # Roll detail
    axes[2, 0].plot(time, roll, 'r-', linewidth=2)
    axes[2, 0].set_ylabel('Roll (deg)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title('Roll')

    # Pitch detail
    axes[2, 1].plot(time, pitch, 'g-', linewidth=2)
    axes[2, 1].set_ylabel('Pitch (deg)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title('Pitch')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved telemetry plot to {filename}")
    else:
        plt.show()


class MultiAircraftPlotter:
    """Multi-aircraft telemetry plotter using matplotlib.

    Displays telemetry for multiple aircraft simultaneously with color-coding.
    Includes fleet overview and per-aircraft detail views.

    Example:
        >>> plotter = MultiAircraftPlotter(aircraft_ids=["001", "002", "003"])
        >>> for i in range(100):
        ...     for aircraft_id in ["001", "002", "003"]:
        ...         state = aircraft[aircraft_id].step(0.01)
        ...         plotter.update(aircraft_id, state)
        ...     plotter.plot()
    """

    # Color palette for aircraft (up to 10 distinct colors)
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

    def __init__(self, aircraft_ids: List[str], window_size: int = 500):
        """Initialize multi-aircraft plotter.

        Args:
            aircraft_ids: List of aircraft IDs to plot
            window_size: Number of time steps to display (rolling window)
        """
        self.aircraft_ids = aircraft_ids
        self.window_size = window_size

        # Assign colors
        self.colors = {
            aid: self.COLORS[i % len(self.COLORS)]
            for i, aid in enumerate(aircraft_ids)
        }

        # Per-aircraft data buffers
        self.data: Dict[str, Dict[str, List]] = defaultdict(lambda: {
            'time': [],
            'altitude': [],
            'airspeed': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'position_n': [],
            'position_e': []
        })

        # Create figure with subplots (4x3 layout)
        self.fig, self.axes = plt.subplots(4, 3, figsize=(16, 12))
        self.fig.suptitle('Multi-Aircraft Telemetry', fontsize=18)

        plt.tight_layout()

    def update(self, aircraft_id: str, state: AircraftState) -> None:
        """Update data for one aircraft.

        Args:
            aircraft_id: Aircraft identifier
            state: Aircraft state
        """
        if aircraft_id not in self.aircraft_ids:
            # Auto-register new aircraft
            self.aircraft_ids.append(aircraft_id)
            self.colors[aircraft_id] = self.COLORS[len(self.aircraft_ids) % len(self.COLORS)]

        # Append data
        data = self.data[aircraft_id]
        data['time'].append(state.time)
        data['altitude'].append(state.altitude)
        data['airspeed'].append(state.airspeed)
        data['roll'].append(np.degrees(state.attitude[0]))
        data['pitch'].append(np.degrees(state.attitude[1]))
        data['yaw'].append(np.degrees(state.attitude[2]))
        data['position_n'].append(state.position[0])
        data['position_e'].append(state.position[1])

        # Keep only recent data (rolling window)
        if len(data['time']) > self.window_size:
            for key in data.keys():
                data[key] = data[key][-self.window_size:]

    def plot(self) -> None:
        """Redraw all plots with current data."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Row 0: Altitude and Airspeed
        for aircraft_id in self.aircraft_ids:
            data = self.data[aircraft_id]
            if not data['time']:
                continue

            color = self.colors[aircraft_id]

            # Altitude
            self.axes[0, 0].plot(data['time'], data['altitude'],
                               color=color, linewidth=2, label=aircraft_id)

            # Airspeed
            self.axes[0, 1].plot(data['time'], data['airspeed'],
                               color=color, linewidth=2, label=aircraft_id)

        self.axes[0, 0].set_ylabel('Altitude (m)')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].set_title('Altitude (All Aircraft)')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend(loc='best')

        self.axes[0, 1].set_ylabel('Airspeed (m/s)')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].set_title('Airspeed (All Aircraft)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].legend(loc='best')

        # Row 0, Col 2: Position (top-down view)
        for aircraft_id in self.aircraft_ids:
            data = self.data[aircraft_id]
            if not data['time']:
                continue

            color = self.colors[aircraft_id]
            self.axes[0, 2].plot(data['position_e'], data['position_n'],
                               color=color, linewidth=2, label=aircraft_id)
            # Current position marker
            if data['position_e']:
                self.axes[0, 2].plot(data['position_e'][-1], data['position_n'][-1],
                                   'o', color=color, markersize=8)

        self.axes[0, 2].set_ylabel('North (m)')
        self.axes[0, 2].set_xlabel('East (m)')
        self.axes[0, 2].set_title('Position (Top View)')
        self.axes[0, 2].grid(True, alpha=0.3)
        self.axes[0, 2].legend(loc='best')
        self.axes[0, 2].axis('equal')

        # Row 1: Attitude (Roll, Pitch, Yaw)
        for aircraft_id in self.aircraft_ids:
            data = self.data[aircraft_id]
            if not data['time']:
                continue

            color = self.colors[aircraft_id]

            # Roll
            self.axes[1, 0].plot(data['time'], data['roll'],
                               color=color, linewidth=2, label=aircraft_id)

            # Pitch
            self.axes[1, 1].plot(data['time'], data['pitch'],
                               color=color, linewidth=2, label=aircraft_id)

            # Yaw
            self.axes[1, 2].plot(data['time'], data['yaw'],
                               color=color, linewidth=2, label=aircraft_id)

        self.axes[1, 0].set_ylabel('Roll (deg)')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_title('Roll')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend(loc='best')

        self.axes[1, 1].set_ylabel('Pitch (deg)')
        self.axes[1, 1].set_xlabel('Time (s)')
        self.axes[1, 1].set_title('Pitch')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].legend(loc='best')

        self.axes[1, 2].set_ylabel('Yaw (deg)')
        self.axes[1, 2].set_xlabel('Time (s)')
        self.axes[1, 2].set_title('Yaw')
        self.axes[1, 2].grid(True, alpha=0.3)
        self.axes[1, 2].legend(loc='best')

        # Row 2: Fleet metrics
        # Average altitude
        avg_alt = []
        times = []
        if self.data and any(self.data[aid]['time'] for aid in self.aircraft_ids):
            # Get common time base (use first aircraft)
            first_id = next(aid for aid in self.aircraft_ids if self.data[aid]['time'])
            times = self.data[first_id]['time']

            for i, t in enumerate(times):
                alts = []
                for aircraft_id in self.aircraft_ids:
                    data = self.data[aircraft_id]
                    if i < len(data['altitude']):
                        alts.append(data['altitude'][i])
                if alts:
                    avg_alt.append(np.mean(alts))

            if avg_alt:
                self.axes[2, 0].plot(times[:len(avg_alt)], avg_alt, 'k-', linewidth=2)

        self.axes[2, 0].set_ylabel('Avg Altitude (m)')
        self.axes[2, 0].set_xlabel('Time (s)')
        self.axes[2, 0].set_title('Fleet Average Altitude')
        self.axes[2, 0].grid(True, alpha=0.3)

        # Formation spread (std dev of positions)
        spread = []
        if self.data and any(self.data[aid]['time'] for aid in self.aircraft_ids):
            for i in range(len(times)):
                positions = []
                for aircraft_id in self.aircraft_ids:
                    data = self.data[aircraft_id]
                    if i < len(data['position_n']):
                        positions.append([data['position_n'][i], data['position_e'][i]])
                if len(positions) > 1:
                    positions = np.array(positions)
                    spread.append(np.std(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))

            if spread:
                self.axes[2, 1].plot(times[:len(spread)], spread, 'k-', linewidth=2)

        self.axes[2, 1].set_ylabel('Formation Spread (m)')
        self.axes[2, 1].set_xlabel('Time (s)')
        self.axes[2, 1].set_title('Fleet Formation Spread')
        self.axes[2, 1].grid(True, alpha=0.3)

        # Aircraft count over time
        self.axes[2, 2].text(0.5, 0.5, f'Active Aircraft: {len(self.aircraft_ids)}',
                            ha='center', va='center', fontsize=20, transform=self.axes[2, 2].transAxes)
        self.axes[2, 2].set_title('Fleet Status')
        self.axes[2, 2].axis('off')

        # Row 3: Individual aircraft detail (show first 3 aircraft)
        for i, aircraft_id in enumerate(self.aircraft_ids[:3]):
            data = self.data[aircraft_id]
            if not data['time']:
                continue

            color = self.colors[aircraft_id]
            ax = self.axes[3, i]

            # Plot altitude with colored background
            ax.plot(data['time'], data['altitude'], color=color, linewidth=2)
            ax.set_ylabel('Altitude (m)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Aircraft {aircraft_id}')
            ax.grid(True, alpha=0.3)
            ax.patch.set_facecolor(color)
            ax.patch.set_alpha(0.1)

        # Hide unused subplots in row 3
        for i in range(len(self.aircraft_ids[:3]), 3):
            self.axes[3, i].axis('off')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def show(self) -> None:
        """Display plots (blocking)."""
        self.plot()
        plt.show()

    def save(self, filename: str) -> None:
        """Save current plot to file.

        Args:
            filename: Output filename (e.g., 'multi_aircraft_telemetry.png')
        """
        self.plot()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved multi-aircraft plot to {filename}")
