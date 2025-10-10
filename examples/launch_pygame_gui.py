#!/usr/bin/env python3
"""Launch the enhanced Pygame-based flight control GUI.

This script starts the Pygame GUI with:
- Drag-and-drop joystick control
- Artificial horizon with roll rotation
- 3D aircraft visualization
- HSA/Waypoint text inputs
- Lat/Lon display

Usage:
    python examples/launch_pygame_gui.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run GUI (v2 - enhanced)
from gui.flight_gui_pygame_v2 import FlightControlGUI

if __name__ == '__main__':
    # Optional: Load custom aircraft model
    # gui = FlightControlGUI(aircraft_model_path="path/to/model.obj")
    gui = FlightControlGUI()
    gui.run()
