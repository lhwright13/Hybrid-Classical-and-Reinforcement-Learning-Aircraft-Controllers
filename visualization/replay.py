"""Flight data replay functionality."""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class MultiAircraftReplay:
    """Replay recorded flight telemetry data.

    Loads data from TelemetryLogger output files and provides
    playback functionality.
    """

    def __init__(self, filepath: str):
        """Initialize replay from log file.

        Args:
            filepath: Path to log file (.hdf5 or .json)
        """
        self.filepath = Path(filepath)
        self.data: Dict[str, Dict] = {}
        self.metadata: Dict[str, Dict] = {}
        self.current_index: Dict[str, int] = {}

        self._load_data()

    def _load_data(self):
        """Load data from file."""
        if self.filepath.suffix in ['.hdf5', '.h5']:
            try:
                import h5py
                with h5py.File(self.filepath, 'r') as f:
                    for aircraft_id in f.keys():
                        grp = f[aircraft_id]
                        self.data[aircraft_id] = {
                            'times': np.array(grp['times']) if 'times' in grp else [],
                            'positions': np.array(grp['positions']) if 'positions' in grp else [],
                            'attitudes': np.array(grp['attitudes']) if 'attitudes' in grp else []
                        }
                        self.metadata[aircraft_id] = dict(grp.attrs)
                        self.current_index[aircraft_id] = 0
            except ImportError:
                print("Warning: h5py not available for replay")
            except Exception as e:
                print(f"Error loading HDF5 file: {e}")
        else:
            # JSON format
            import json
            try:
                with open(self.filepath, 'r') as f:
                    content = json.load(f)
                self.data = content.get('data', {})
                self.metadata = content.get('metadata', {})
                for aircraft_id in self.data:
                    self.current_index[aircraft_id] = 0
            except Exception as e:
                print(f"Error loading JSON file: {e}")

    def get_aircraft_ids(self) -> List[str]:
        """Get list of aircraft IDs in the replay.

        Returns:
            List of aircraft identifiers
        """
        return list(self.data.keys())

    def get_duration(self, aircraft_id: str) -> float:
        """Get total duration for an aircraft.

        Args:
            aircraft_id: Aircraft identifier

        Returns:
            Duration in seconds
        """
        if aircraft_id in self.data:
            times = self.data[aircraft_id].get('times', [])
            if len(times) > 0:
                return float(times[-1] - times[0])
        return 0.0

    def get_state_at_time(self, aircraft_id: str, time: float) -> Optional[Dict]:
        """Get interpolated state at a specific time.

        Args:
            aircraft_id: Aircraft identifier
            time: Time in seconds

        Returns:
            State dictionary or None if not available
        """
        if aircraft_id not in self.data:
            return None

        data = self.data[aircraft_id]
        times = data.get('times', [])

        if len(times) == 0:
            return None

        # Find nearest index
        idx = np.searchsorted(times, time)
        idx = min(idx, len(times) - 1)

        positions = data.get('positions', [])
        attitudes = data.get('attitudes', [])

        state = {'time': times[idx]}
        if len(positions) > idx:
            state['position'] = positions[idx]
        if len(attitudes) > idx:
            state['attitude'] = attitudes[idx]

        return state

    def reset(self, aircraft_id: Optional[str] = None):
        """Reset replay to beginning.

        Args:
            aircraft_id: Specific aircraft to reset, or None for all
        """
        if aircraft_id:
            self.current_index[aircraft_id] = 0
        else:
            for aid in self.current_index:
                self.current_index[aid] = 0

    def __len__(self):
        """Return number of aircraft in replay."""
        return len(self.data)
