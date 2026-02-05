#!/usr/bin/env python3
"""Launch the Pygame GUI with Learned Rate Controller option.

This script allows testing the learned RL rate controller in the pygame GUI
with the ability to toggle between learned and PID controllers in real-time.

Usage:
    python examples/launch_pygame_gui_with_learned_rate.py [--model PATH]
"""

import sys
from pathlib import Path
import argparse
import h5py
import numpy as np
import time
import threading
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GUI and visualization
from gui.flight_gui_pygame_v2 import FlightControlGUI


class GUIWithLearnedRate:
    """Wrapper that runs GUI with learned rate controller option."""

    def __init__(
        self,
        model_path: str,
        aircraft_model_path=None,
        log_file="learned_rate_telemetry.h5"
    ):
        """Initialize GUI with learned rate controller.

        Args:
            model_path: Path to trained RL model (.zip)
            aircraft_model_path: Optional path to .obj aircraft model
            log_file: HDF5 file for telemetry logging
        """
        # Import here to avoid circular dependencies
        from gui.simulation_worker_learned import SimulationWorkerWithLearned

        self.model_path = model_path
        self.log_file = log_file
        self.running = True

        # Create GUI with custom simulation worker
        self.gui = FlightControlGUI(aircraft_model_path=aircraft_model_path)

        # Replace the simulation worker with learned controller version
        self.gui.sim_worker.stop()
        self.gui.sim_worker = SimulationWorkerWithLearned(
            model_path=model_path,
            use_learned=True  # Start with learned controller
        )
        self.gui.sim_worker.start()

        # Initialize HDF5 file for logging
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
            f.create_dataset('cmd_roll_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_pitch_rate', (0,), maxshape=(None,), dtype='f')
            f.create_dataset('cmd_yaw_rate', (0,), maxshape=(None,), dtype='f')
            # Controller type (0=PID, 1=Learned)
            f.create_dataset('controller_type', (0,), maxshape=(None,), dtype='i')

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
                                     'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate', 'controller_type']:
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
                            elif name == 'controller_type':
                                # 1 if using learned, 0 if using PID
                                is_learned = getattr(self.gui.sim_worker, 'use_learned', True)
                                ds[-1] = 1 if is_learned else 0
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
        """Run GUI with learned controller and logging."""
        # Start the live plotter in a separate process (optional)
        plotter_script = Path(__file__).parent / "plot_telemetry_live.py"
        plotter_process = None

        try:
            if plotter_script.exists():
                plotter_process = subprocess.Popen(
                    [sys.executable, str(plotter_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"Started live plotter (PID: {plotter_process.pid})")
        except Exception as e:
            print(f"Warning: Could not start live plotter: {e}")

        # Start logging thread
        log_thread = threading.Thread(target=self.logging_loop, daemon=True)
        log_thread.start()

        print("\n" + "="*80)
        print("PYGAME GUI WITH LEARNED RATE CONTROLLER")
        print("="*80)
        print(f"\nModel: {self.model_path}")
        print(f"Logging to: {self.log_file}")
        print("\nGUI Controls:")
        print("  - Use joystick to control aircraft")
        print("  - Mode buttons to switch control levels")
        print("  - 'D' key to toggle debug panel")
        print("  - 'L' key to toggle Learned/PID rate controller")
        print("  - 'R' to reset simulation")
        print("\nIn RATE mode:")
        print("  - Controller will use RL policy (or PID if toggled)")
        print("  - Watch the telemetry panel for 'Rate Controller: Learned/PID'")
        print("\nClose GUI window to exit.")
        print("="*80 + "\n")

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
    parser = argparse.ArgumentParser(
        description="Launch Pygame GUI with Learned Rate Controller"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='learned_controllers/models/checkpoints/final_model.zip',
        help='Path to trained model (default: checkpoints/final_model.zip)'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='learned_rate_telemetry.h5',
        help='Log file path (default: learned_rate_telemetry.h5)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nAvailable models:")
        models_dir = Path("learned_controllers/models")
        if models_dir.exists():
            for model_file in models_dir.rglob("*.zip"):
                print(f"  - {model_file}")
        sys.exit(1)

    gui = GUIWithLearnedRate(
        model_path=args.model,
        log_file=args.log
    )
    gui.run()
