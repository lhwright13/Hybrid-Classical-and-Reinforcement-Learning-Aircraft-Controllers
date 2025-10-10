"""Multi-aircraft telemetry logging with HDF5 backend.

This module provides efficient logging for multiple aircraft simultaneously,
with each aircraft's data stored in separate HDF5 groups for easy access.
"""

import h5py
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime

from controllers.types import AircraftState, ControlCommand, ControlSurfaces


class TelemetryLogger:
    """Multi-aircraft telemetry logger with HDF5 backend.

    Efficiently logs time-series data for multiple aircraft to a single HDF5 file.
    Each aircraft gets its own group: /aircraft_{id}/

    Features:
    - Thread-safe concurrent logging
    - Buffered writes for performance
    - Automatic flushing
    - Metadata storage per aircraft
    - Compression for large datasets

    Example:
        >>> logger = TelemetryLogger("flight.h5")
        >>> logger.register_aircraft("001", metadata={"type": "rc_plane"})
        >>> logger.log_state("001", state, timestamp=0.01)
        >>> logger.log_command("001", command)
        >>> logger.close()
    """

    def __init__(
        self,
        filename: str,
        buffer_size: int = 100,
        compression: str = 'gzip',
        compression_opts: int = 4
    ):
        """Initialize multi-aircraft logger.

        Args:
            filename: HDF5 file path
            buffer_size: Number of samples to buffer before flushing
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_opts: Compression level (1-9 for gzip)
        """
        self.filename = Path(filename)
        self.buffer_size = buffer_size
        self.compression = compression
        self.compression_opts = compression_opts

        # Thread safety
        self._lock = threading.Lock()

        # Per-aircraft buffers
        self._state_buffers: Dict[str, List[Dict]] = defaultdict(list)
        self._command_buffers: Dict[str, List[Dict]] = defaultdict(list)
        self._surface_buffers: Dict[str, List[Dict]] = defaultdict(list)

        # Registered aircraft
        self._aircraft: Dict[str, Dict[str, Any]] = {}

        # Create HDF5 file
        self._create_file()

    def _create_file(self) -> None:
        """Create HDF5 file with fleet metadata."""
        with h5py.File(self.filename, 'w') as f:
            # Fleet-level metadata
            fleet_group = f.create_group('fleet')
            fleet_group.attrs['created'] = datetime.now().isoformat()
            fleet_group.attrs['version'] = '1.0.0'

    def register_aircraft(
        self,
        aircraft_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new aircraft for logging.

        Args:
            aircraft_id: Unique aircraft identifier
            metadata: Optional metadata (aircraft type, config, etc.)
        """
        with self._lock:
            if aircraft_id in self._aircraft:
                raise ValueError(f"Aircraft {aircraft_id} already registered")

            self._aircraft[aircraft_id] = metadata or {}

            # Create HDF5 group for this aircraft
            with h5py.File(self.filename, 'a') as f:
                group = f.create_group(f'aircraft_{aircraft_id}')

                # Store metadata as attributes
                for key, value in self._aircraft[aircraft_id].items():
                    if isinstance(value, (str, int, float, bool)):
                        group.attrs[key] = value

    def log_state(
        self,
        aircraft_id: str,
        state: AircraftState,
        timestamp: Optional[float] = None
    ) -> None:
        """Log aircraft state.

        Args:
            aircraft_id: Aircraft identifier
            state: Aircraft state
            timestamp: Optional timestamp (uses state.time if not provided)
        """
        if aircraft_id not in self._aircraft:
            raise ValueError(f"Aircraft {aircraft_id} not registered. Call register_aircraft() first.")

        t = timestamp if timestamp is not None else state.time

        # Create record
        record = {
            'time': t,
            'pos_n': state.position[0],
            'pos_e': state.position[1],
            'pos_d': state.position[2],
            'vel_n': state.velocity[0],
            'vel_e': state.velocity[1],
            'vel_d': state.velocity[2],
            'roll': state.attitude[0],
            'pitch': state.attitude[1],
            'yaw': state.attitude[2],
            'p': state.angular_rate[0],
            'q': state.angular_rate[1],
            'r': state.angular_rate[2],
            'altitude': state.altitude,
            'airspeed': state.airspeed
        }

        with self._lock:
            self._state_buffers[aircraft_id].append(record)

            # Flush if buffer full
            if len(self._state_buffers[aircraft_id]) >= self.buffer_size:
                self._flush_states(aircraft_id)

    def log_command(
        self,
        aircraft_id: str,
        command: ControlCommand,
        timestamp: float
    ) -> None:
        """Log control command.

        Args:
            aircraft_id: Aircraft identifier
            command: Control command
            timestamp: Timestamp
        """
        if aircraft_id not in self._aircraft:
            raise ValueError(f"Aircraft {aircraft_id} not registered")

        # Create record
        record = {
            'time': timestamp,
            'mode': command.mode.value,
            'roll_angle': command.roll_angle or 0.0,
            'pitch_angle': command.pitch_angle or 0.0,
            'yaw_angle': command.yaw_angle or 0.0,
            'roll_rate': command.roll_rate or 0.0,
            'pitch_rate': command.pitch_rate or 0.0,
            'yaw_rate': command.yaw_rate or 0.0,
            'throttle': command.throttle or 0.0
        }

        with self._lock:
            self._command_buffers[aircraft_id].append(record)

            if len(self._command_buffers[aircraft_id]) >= self.buffer_size:
                self._flush_commands(aircraft_id)

    def log_surfaces(
        self,
        aircraft_id: str,
        surfaces: ControlSurfaces,
        timestamp: float
    ) -> None:
        """Log control surfaces.

        Args:
            aircraft_id: Aircraft identifier
            surfaces: Control surfaces
            timestamp: Timestamp
        """
        if aircraft_id not in self._aircraft:
            raise ValueError(f"Aircraft {aircraft_id} not registered")

        record = {
            'time': timestamp,
            'elevator': surfaces.elevator,
            'aileron': surfaces.aileron,
            'rudder': surfaces.rudder,
            'throttle': surfaces.throttle
        }

        with self._lock:
            self._surface_buffers[aircraft_id].append(record)

            if len(self._surface_buffers[aircraft_id]) >= self.buffer_size:
                self._flush_surfaces(aircraft_id)

    def _flush_states(self, aircraft_id: str) -> None:
        """Flush state buffer to HDF5 (assumes lock held)."""
        if not self._state_buffers[aircraft_id]:
            return

        with h5py.File(self.filename, 'a') as f:
            group = f[f'aircraft_{aircraft_id}']

            # Convert buffer to numpy arrays
            buffer = self._state_buffers[aircraft_id]
            data = {key: np.array([rec[key] for rec in buffer]) for key in buffer[0].keys()}

            # Create or append to datasets
            for key, values in data.items():
                if key in group:
                    # Append to existing dataset
                    dataset = group[key]
                    old_size = dataset.shape[0]
                    dataset.resize((old_size + len(values),))
                    dataset[old_size:] = values
                else:
                    # Create new dataset
                    group.create_dataset(
                        key,
                        data=values,
                        maxshape=(None,),
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )

        self._state_buffers[aircraft_id].clear()

    def _flush_commands(self, aircraft_id: str) -> None:
        """Flush command buffer to HDF5 (assumes lock held)."""
        if not self._command_buffers[aircraft_id]:
            return

        with h5py.File(self.filename, 'a') as f:
            group = f[f'aircraft_{aircraft_id}']

            # Create commands subgroup if needed
            if 'commands' not in group:
                cmd_group = group.create_group('commands')
            else:
                cmd_group = group['commands']

            buffer = self._command_buffers[aircraft_id]
            data = {key: np.array([rec[key] for rec in buffer]) for key in buffer[0].keys()}

            for key, values in data.items():
                if key in cmd_group:
                    dataset = cmd_group[key]
                    old_size = dataset.shape[0]
                    dataset.resize((old_size + len(values),))
                    dataset[old_size:] = values
                else:
                    cmd_group.create_dataset(
                        key,
                        data=values,
                        maxshape=(None,),
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )

        self._command_buffers[aircraft_id].clear()

    def _flush_surfaces(self, aircraft_id: str) -> None:
        """Flush surface buffer to HDF5 (assumes lock held)."""
        if not self._surface_buffers[aircraft_id]:
            return

        with h5py.File(self.filename, 'a') as f:
            group = f[f'aircraft_{aircraft_id}']

            if 'surfaces' not in group:
                surf_group = group.create_group('surfaces')
            else:
                surf_group = group['surfaces']

            buffer = self._surface_buffers[aircraft_id]
            data = {key: np.array([rec[key] for rec in buffer]) for key in buffer[0].keys()}

            for key, values in data.items():
                if key in surf_group:
                    dataset = surf_group[key]
                    old_size = dataset.shape[0]
                    dataset.resize((old_size + len(values),))
                    dataset[old_size:] = values
                else:
                    surf_group.create_dataset(
                        key,
                        data=values,
                        maxshape=(None,),
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )

        self._surface_buffers[aircraft_id].clear()

    def flush(self) -> None:
        """Flush all buffers to disk."""
        with self._lock:
            for aircraft_id in self._aircraft.keys():
                self._flush_states(aircraft_id)
                self._flush_commands(aircraft_id)
                self._flush_surfaces(aircraft_id)

    def close(self) -> None:
        """Flush and close logger."""
        self.flush()

        # Update fleet metadata
        with h5py.File(self.filename, 'a') as f:
            fleet_group = f['fleet']
            fleet_group.attrs['closed'] = datetime.now().isoformat()
            fleet_group.attrs['num_aircraft'] = len(self._aircraft)
            fleet_group.attrs['aircraft_ids'] = ','.join(self._aircraft.keys())

    def get_aircraft_list(self) -> List[str]:
        """Get list of registered aircraft IDs."""
        return list(self._aircraft.keys())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
