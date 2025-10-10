"""Multi-aircraft replay system for HDF5 flight logs.

This module provides playback capabilities for logged flight data,
with synchronization across multiple aircraft and export to video.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from controllers.types import AircraftState, ControlCommand, ControlSurfaces, ControlMode


@dataclass
class ReplayData:
    """Data for one aircraft's replay."""
    aircraft_id: str
    times: np.ndarray
    states: List[AircraftState]
    commands: Optional[List[ControlCommand]] = None
    surfaces: Optional[List[ControlSurfaces]] = None
    metadata: Dict[str, Any] = None


class MultiAircraftReplay:
    """Multi-aircraft replay system.

    Loads HDF5 flight logs and provides synchronized playback
    of multiple aircraft with controls for speed, seek, etc.

    Features:
    - Load all or subset of aircraft from log
    - Synchronized playback across all aircraft
    - Variable playback speed
    - Seek to specific time
    - Export to video (requires visualization)
    - Export individual aircraft to CSV

    Example:
        >>> replay = MultiAircraftReplay("flight.h5")
        >>> replay.load_aircraft(["001", "002", "003"])
        >>> for t in replay.play(speed=1.0):
        ...     states = replay.get_states_at_time(t)
        ...     # Visualize states
    """

    def __init__(self, filename: str):
        """Initialize replay system.

        Args:
            filename: HDF5 log file path
        """
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"Log file not found: {filename}")

        self.data: Dict[str, ReplayData] = {}
        self._loaded_aircraft: List[str] = []

        # Playback state
        self._current_time = 0.0
        self._min_time = 0.0
        self._max_time = 0.0

    def get_available_aircraft(self) -> List[str]:
        """Get list of aircraft in the log file.

        Returns:
            List of aircraft IDs
        """
        with h5py.File(self.filename, 'r') as f:
            # Look for aircraft groups
            aircraft_ids = []
            for key in f.keys():
                if key.startswith('aircraft_'):
                    aircraft_id = key.replace('aircraft_', '')
                    aircraft_ids.append(aircraft_id)
            return aircraft_ids

    def load_aircraft(
        self,
        aircraft_ids: Optional[List[str]] = None,
        load_commands: bool = True,
        load_surfaces: bool = True
    ) -> None:
        """Load aircraft data from HDF5 file.

        Args:
            aircraft_ids: List of aircraft to load (None = load all)
            load_commands: Load command data
            load_surfaces: Load surface data
        """
        if aircraft_ids is None:
            aircraft_ids = self.get_available_aircraft()

        with h5py.File(self.filename, 'r') as f:
            for aircraft_id in aircraft_ids:
                group_name = f'aircraft_{aircraft_id}'
                if group_name not in f:
                    print(f"Warning: Aircraft {aircraft_id} not found in log")
                    continue

                group = f[group_name]

                # Load states
                times = np.array(group['time'])
                states = []

                for i in range(len(times)):
                    state = AircraftState(
                        time=times[i],
                        position=np.array([
                            group['pos_n'][i],
                            group['pos_e'][i],
                            group['pos_d'][i]
                        ]),
                        velocity=np.array([
                            group['vel_n'][i],
                            group['vel_e'][i],
                            group['vel_d'][i]
                        ]),
                        attitude=np.array([
                            group['roll'][i],
                            group['pitch'][i],
                            group['yaw'][i]
                        ]),
                        angular_rate=np.array([
                            group['p'][i],
                            group['q'][i],
                            group['r'][i]
                        ]),
                        altitude=group['altitude'][i],
                        airspeed=group['airspeed'][i]
                    )
                    states.append(state)

                # Load commands (optional)
                commands = None
                if load_commands and 'commands' in group:
                    cmd_group = group['commands']
                    cmd_times = np.array(cmd_group['time'])
                    commands = []

                    for i in range(len(cmd_times)):
                        cmd = ControlCommand(
                            mode=ControlMode(int(cmd_group['mode'][i])),
                            roll_angle=cmd_group['roll_angle'][i],
                            pitch_angle=cmd_group['pitch_angle'][i],
                            yaw_angle=cmd_group['yaw_angle'][i],
                            roll_rate=cmd_group['roll_rate'][i],
                            pitch_rate=cmd_group['pitch_rate'][i],
                            yaw_rate=cmd_group['yaw_rate'][i],
                            throttle=cmd_group['throttle'][i]
                        )
                        commands.append(cmd)

                # Load surfaces (optional)
                surfaces = None
                if load_surfaces and 'surfaces' in group:
                    surf_group = group['surfaces']
                    surf_times = np.array(surf_group['time'])
                    surfaces = []

                    for i in range(len(surf_times)):
                        surf = ControlSurfaces(
                            elevator=surf_group['elevator'][i],
                            aileron=surf_group['aileron'][i],
                            rudder=surf_group['rudder'][i],
                            throttle=surf_group['throttle'][i]
                        )
                        surfaces.append(surf)

                # Load metadata
                metadata = dict(group.attrs)

                # Store replay data
                self.data[aircraft_id] = ReplayData(
                    aircraft_id=aircraft_id,
                    times=times,
                    states=states,
                    commands=commands,
                    surfaces=surfaces,
                    metadata=metadata
                )

                self._loaded_aircraft.append(aircraft_id)

        # Compute time bounds
        if self.data:
            all_times = [data.times for data in self.data.values()]
            self._min_time = min(times[0] for times in all_times)
            self._max_time = max(times[-1] for times in all_times)
            self._current_time = self._min_time

        print(f"Loaded {len(self.data)} aircraft from {self.filename}")
        print(f"Time range: {self._min_time:.2f}s to {self._max_time:.2f}s ({self._max_time - self._min_time:.2f}s duration)")

    def get_states_at_time(self, t: float) -> Dict[str, AircraftState]:
        """Get states for all aircraft at given time.

        Uses linear interpolation between samples.

        Args:
            t: Time (seconds)

        Returns:
            Dictionary mapping aircraft_id to state
        """
        states = {}

        for aircraft_id, data in self.data.items():
            # Find nearest time index
            idx = np.searchsorted(data.times, t)

            # Clamp to valid range
            if idx == 0:
                states[aircraft_id] = data.states[0]
            elif idx >= len(data.states):
                states[aircraft_id] = data.states[-1]
            else:
                # Linear interpolation
                t0 = data.times[idx - 1]
                t1 = data.times[idx]
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

                state0 = data.states[idx - 1]
                state1 = data.states[idx]

                # Interpolate state
                state = AircraftState(
                    time=t,
                    position=(1 - alpha) * state0.position + alpha * state1.position,
                    velocity=(1 - alpha) * state0.velocity + alpha * state1.velocity,
                    attitude=(1 - alpha) * state0.attitude + alpha * state1.attitude,
                    angular_rate=(1 - alpha) * state0.angular_rate + alpha * state1.angular_rate,
                    altitude=(1 - alpha) * state0.altitude + alpha * state1.altitude,
                    airspeed=(1 - alpha) * state0.airspeed + alpha * state1.airspeed
                )

                states[aircraft_id] = state

        return states

    def play(
        self,
        speed: float = 1.0,
        dt: float = 0.01,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Generator for playback.

        Args:
            speed: Playback speed multiplier (1.0 = real-time)
            dt: Time step for playback (seconds)
            start_time: Start time (None = beginning)
            end_time: End time (None = end)

        Yields:
            Current time
        """
        t_start = start_time if start_time is not None else self._min_time
        t_end = end_time if end_time is not None else self._max_time

        t = t_start
        while t <= t_end:
            self._current_time = t
            yield t
            t += dt * speed

    def seek(self, t: float) -> None:
        """Seek to specific time.

        Args:
            t: Time to seek to (seconds)
        """
        self._current_time = np.clip(t, self._min_time, self._max_time)

    def get_current_time(self) -> float:
        """Get current playback time."""
        return self._current_time

    def get_duration(self) -> float:
        """Get total duration of logged data."""
        return self._max_time - self._min_time

    def get_loaded_aircraft(self) -> List[str]:
        """Get list of loaded aircraft IDs."""
        return self._loaded_aircraft.copy()

    def export_csv(self, aircraft_id: str, output_file: str) -> None:
        """Export aircraft data to CSV.

        Args:
            aircraft_id: Aircraft to export
            output_file: Output CSV file path
        """
        if aircraft_id not in self.data:
            raise ValueError(f"Aircraft {aircraft_id} not loaded")

        import csv

        data = self.data[aircraft_id]
        output_path = Path(output_file)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                'time', 'pos_n', 'pos_e', 'pos_d',
                'vel_n', 'vel_e', 'vel_d',
                'roll', 'pitch', 'yaw',
                'p', 'q', 'r',
                'altitude', 'airspeed'
            ]
            writer.writerow(header)

            # Data
            for state in data.states:
                row = [
                    state.time,
                    state.position[0], state.position[1], state.position[2],
                    state.velocity[0], state.velocity[1], state.velocity[2],
                    state.attitude[0], state.attitude[1], state.attitude[2],
                    state.angular_rate[0], state.angular_rate[1], state.angular_rate[2],
                    state.altitude,
                    state.airspeed
                ]
                writer.writerow(row)

        print(f"Exported {aircraft_id} to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data.

        Returns:
            Summary dictionary
        """
        summary = {
            'filename': str(self.filename),
            'aircraft_count': len(self.data),
            'aircraft_ids': self._loaded_aircraft,
            'duration': self.get_duration(),
            'time_range': (self._min_time, self._max_time),
            'sample_counts': {}
        }

        for aircraft_id, data in self.data.items():
            summary['sample_counts'][aircraft_id] = len(data.states)

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiAircraftReplay(file='{self.filename.name}', "
                f"aircraft={len(self.data)}, duration={self.get_duration():.1f}s)")
