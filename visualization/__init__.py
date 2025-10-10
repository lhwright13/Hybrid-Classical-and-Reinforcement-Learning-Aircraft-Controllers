"""Visualization and telemetry logging modules.

This package provides multi-aircraft visualization capabilities including:
- HDF5 telemetry logging
- Real-time matplotlib plotting
- 3D fleet visualization with PyVista
- Replay system for logged data
"""

from .logger import TelemetryLogger
from .plotter import TelemetryPlotter, MultiAircraftPlotter, plot_telemetry_history
from .replay import MultiAircraftReplay, ReplayData

# 3D visualization (PyVista or matplotlib fallback)
try:
    from .aircraft_3d import FleetVisualizer3D
    HAS_PYVISTA = True
except ImportError:
    from .aircraft_3d import SimpleFleetVisualizer3D as FleetVisualizer3D
    HAS_PYVISTA = False

__all__ = [
    # Logging
    'TelemetryLogger',
    # Plotting
    'TelemetryPlotter',
    'MultiAircraftPlotter',
    'plot_telemetry_history',
    # 3D Visualization
    'FleetVisualizer3D',
    'HAS_PYVISTA',
    # Replay
    'MultiAircraftReplay',
    'ReplayData',
]
