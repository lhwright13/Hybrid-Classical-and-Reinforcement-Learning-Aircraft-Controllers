#!/usr/bin/env python3
"""Launch the Pygame GUI with telemetry logging and live plotting.

This script starts the Pygame GUI and automatically launches the live telemetry
plotter in a separate process.

Usage:
    python examples/launch_pygame_gui_with_logging.py
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import time
import threading
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GUI and visualization
from gui.flight_gui_pygame_v2 import FlightControlGUI
from controllers.types import AircraftState


class GUIWithLogging:
    """Wrapper that runs GUI with telemetry logging."""

    def __init__(self, aircraft_model_path=None, log_file="live_telemetry.h5"):
        """Initialize GUI with logging.

        Args:
            aircraft_model_path: Optional path to .obj aircraft model
            log_file: HDF5 file for telemetry logging
        """
        self.gui = FlightControlGUI(aircraft_model_path=aircraft_model_path)
        self.log_file = log_file
        self.running = True

        # Initialize HDF5 file
        with h5py.File(self.log_file, 'w') as f:
            # Create datasets (extendable)
            f.create_dataset('time', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('altitude', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('airspeed', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('roll', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('pitch', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('yaw', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('north', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('east', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('roll_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('pitch_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('yaw_rate', (0,), maxshape=(None,), dtype='f')
            # Commanded values
            f.create_dataset('cmd_roll_angle', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_pitch_angle', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_yaw_angle', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_roll_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_pitch_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_yaw_rate', (0,), maxshape=(None,), dtype='f')

    def logging_loop(self):
        """Background thread that logs telemetry."""
        while self.running:
            try:
                # Get latest state from GUI
                if hasattr(self.gui, 'latest_state') and self.gui.latest_state is not None:
                    state = self.gui.latest_state

                    # Append to HDF5 file
                    with h5py.File(self.log_file, 'a') as f:
                        # Get state dict from GUI worker
                        state_dict = self.gui.current_state if hasattr(self.gui, 'current_state') else {}

                        for name in ['time', 'altitude', 'airspeed', 'roll', 'pitch',
                                     'yaw', 'north', 'east', 'roll_rate', 'pitch_rate', 'yaw_rate',
                                     'cmd_roll_angle', 'cmd_pitch_angle', 'cmd_yaw_angle',
                                     'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate']:
                            ds = f[name]
                            ds.resize((ds.shape[0] + 1,))

                            if name == 'time':
                                ds[-1] = state.time
                            elif name == 'altitude':
                                ds[-1] = state.altitude
                            elif name == 'airspeed':
                                ds[-1] = state.airspeed
                            elif name == 'roll':
                                ds[-1] = np.degrees(state.roll)
                            elif name == 'pitch':
                                ds[-1] = np.degrees(state.pitch)
                            elif name == 'yaw':
                                ds[-1] = np.degrees(state.yaw)
                            elif name == 'north':
                                ds[-1] = state.north
                            elif name == 'east':
                                ds[-1] = state.east
                            elif name == 'roll_rate':
                                ds[-1] = np.degrees(state.p)
                            elif name == 'pitch_rate':
                                ds[-1] = np.degrees(state.q)
                            elif name == 'yaw_rate':
                                ds[-1] = np.degrees(state.r)
                            # Commanded values (from state dict, may be None)
                            elif name.startswith('cmd_'):
                                val = state_dict.get(name)
                                ds[-1] = val if val is not None else np.nan

                # Log at 10 Hz
                time.sleep(0.1)

            except Exception as e:
                print(f"Logging error: {e}")
                break

    def run(self):
        """Run GUI with logging and auto-start plotter."""
        # Start the live plotter in a separate process
        plotter_script = Path(__file__).parent / "plot_telemetry_live.py"
        plotter_process = None

        try:
            plotter_process = subprocess.Popen(
                [sys.executable, str(plotter_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Started live plotter (PID: {plotter_process.pid})")
        except Exception as e:
            print(f"Warning: Could not start live plotter: {e}")
            print("You can manually start it with: python examples/plot_telemetry_live.py")

        # Start logging thread
        log_thread = threading.Thread(target=self.logging_loop, daemon=True)
        log_thread.start()

        print("\n" + "="*70)
        print("PYGAME GUI + TELEMETRY LOGGING + LIVE PLOTS")
        print("="*70)
        print(f"\nLogging to: {self.log_file}")
        print("\nGUI Controls:")
        print("  - Use joystick to control aircraft")
        print("  - Mode buttons to switch control levels")
        print("  - 'D' key to toggle debug panel")
        print("  - 'R' to reset simulation")
        print("\nLive telemetry plotter is running in a separate window!")
        print("\nClose GUI window to exit both GUI and plotter.")
        print("="*70 + "\n")

        try:
            # Run GUI (blocking until window closed)
            self.gui.run()
        finally:
            # Stop logging thread
            self.running = False
            log_thread.join(timeout=1.0)

            # Terminate plotter process
            if plotter_process:
                plotter_process.terminate()
                try:
                    plotter_process.wait(timeout=2.0)
                    print("Live plotter stopped.")
                except subprocess.TimeoutExpired:
                    plotter_process.kill()
                    print("Live plotter forcefully terminated.")


if __name__ == '__main__':
    gui = GUIWithLogging(log_file="live_telemetry.h5")
    gui.run()
