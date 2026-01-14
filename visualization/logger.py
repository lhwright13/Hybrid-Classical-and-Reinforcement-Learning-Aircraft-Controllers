"""Telemetry logger for multi-aircraft flight data."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict
import json


class TelemetryLogger:
    """Logger for flight telemetry data.

    Stores aircraft states, commands, and control surfaces to HDF5 or JSON files.
    Supports multiple aircraft simultaneously.
    """

    def __init__(self, filepath: str):
        """Initialize logger.

        Args:
            filepath: Path to output file (.hdf5 or .json)
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.aircraft_data: Dict[str, Dict] = {}
        self.aircraft_metadata: Dict[str, Dict] = {}
        self._use_hdf5 = filepath.endswith('.hdf5') or filepath.endswith('.h5')

        if self._use_hdf5:
            try:
                import h5py
                self._h5file = h5py.File(filepath, 'w')
            except ImportError:
                print("Warning: h5py not available, falling back to JSON logging")
                self._use_hdf5 = False
                self.filepath = self.filepath.with_suffix('.json')
                self._h5file = None
        else:
            self._h5file = None

    def register_aircraft(self, aircraft_id: str, metadata: Optional[Dict] = None):
        """Register an aircraft for logging.

        Args:
            aircraft_id: Unique identifier for the aircraft
            metadata: Optional metadata dict (mission info, etc.)
        """
        self.aircraft_data[aircraft_id] = {
            'states': [],
            'commands': [],
            'surfaces': [],
            'times': []
        }
        self.aircraft_metadata[aircraft_id] = metadata or {}

        if self._use_hdf5 and self._h5file:
            grp = self._h5file.create_group(aircraft_id)
            if metadata:
                for key, val in metadata.items():
                    grp.attrs[key] = str(val)

    def log_state(self, aircraft_id: str, state: Any):
        """Log aircraft state.

        Args:
            aircraft_id: Aircraft identifier
            state: AircraftState object
        """
        if aircraft_id not in self.aircraft_data:
            self.register_aircraft(aircraft_id)

        state_dict = {
            'time': state.time,
            'position': state.position.tolist() if hasattr(state.position, 'tolist') else list(state.position),
            'velocity': state.velocity.tolist() if hasattr(state.velocity, 'tolist') else list(state.velocity),
            'attitude': state.attitude.tolist() if hasattr(state.attitude, 'tolist') else list(state.attitude),
            'angular_rate': state.angular_rate.tolist() if hasattr(state.angular_rate, 'tolist') else list(state.angular_rate),
            'airspeed': state.airspeed,
            'altitude': state.altitude
        }
        self.aircraft_data[aircraft_id]['states'].append(state_dict)
        self.aircraft_data[aircraft_id]['times'].append(state.time)

    def log_command(self, aircraft_id: str, command: Any, time: float):
        """Log control command.

        Args:
            aircraft_id: Aircraft identifier
            command: ControlCommand object
            time: Timestamp
        """
        if aircraft_id not in self.aircraft_data:
            self.register_aircraft(aircraft_id)

        cmd_dict = {
            'time': time,
            'mode': command.mode.name if hasattr(command.mode, 'name') else str(command.mode)
        }
        self.aircraft_data[aircraft_id]['commands'].append(cmd_dict)

    def log_surfaces(self, aircraft_id: str, surfaces: Any, time: float):
        """Log control surface deflections.

        Args:
            aircraft_id: Aircraft identifier
            surfaces: ControlSurfaces object
            time: Timestamp
        """
        if aircraft_id not in self.aircraft_data:
            self.register_aircraft(aircraft_id)

        surf_dict = {
            'time': time,
            'aileron': surfaces.aileron,
            'elevator': surfaces.elevator,
            'rudder': surfaces.rudder,
            'throttle': surfaces.throttle
        }
        self.aircraft_data[aircraft_id]['surfaces'].append(surf_dict)

    def close(self):
        """Close the logger and write data to file."""
        if self._use_hdf5 and self._h5file:
            # Write data to HDF5
            for aircraft_id, data in self.aircraft_data.items():
                grp = self._h5file[aircraft_id]
                if data['times']:
                    grp.create_dataset('times', data=np.array(data['times']))
                if data['states']:
                    # Extract position/attitude arrays
                    positions = np.array([s['position'] for s in data['states']])
                    attitudes = np.array([s['attitude'] for s in data['states']])
                    grp.create_dataset('positions', data=positions)
                    grp.create_dataset('attitudes', data=attitudes)
            self._h5file.close()
        else:
            # Write to JSON
            output = {
                'metadata': self.aircraft_metadata,
                'data': self.aircraft_data
            }
            with open(self.filepath, 'w') as f:
                json.dump(output, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
