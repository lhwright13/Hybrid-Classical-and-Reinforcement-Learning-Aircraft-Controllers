#!/usr/bin/env python3
"""Simple live telemetry plotter that reads from simulation worker queue.

This creates a matplotlib window that displays real-time telemetry.
Works by monitoring a shared memory file updated by the GUI.

Usage:
    python examples/plot_telemetry_live.py

Press Ctrl+C to exit.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
from pathlib import Path
import json
import time

# Shared state file (updated by GUI)
STATE_FILE = Path("live_state.json")
WINDOW_SIZE = 300  # Show last 30 seconds (at 10 Hz)


class LiveTelemetryPlotter:
    """Live telemetry plotter using matplotlib animation."""

    def __init__(self):
        # Data buffers (rolling windows)
        self.time_data = deque(maxlen=WINDOW_SIZE)
        self.altitude = deque(maxlen=WINDOW_SIZE)
        self.airspeed = deque(maxlen=WINDOW_SIZE)
        self.roll = deque(maxlen=WINDOW_SIZE)
        self.pitch = deque(maxlen=WINDOW_SIZE)
        self.yaw = deque(maxlen=WINDOW_SIZE)
        self.roll_rate = deque(maxlen=WINDOW_SIZE)
        self.pitch_rate = deque(maxlen=WINDOW_SIZE)
        self.yaw_rate = deque(maxlen=WINDOW_SIZE)

        # Commanded value buffers
        self.cmd_roll_angle = deque(maxlen=WINDOW_SIZE)
        self.cmd_pitch_angle = deque(maxlen=WINDOW_SIZE)
        self.cmd_yaw_angle = deque(maxlen=WINDOW_SIZE)
        self.cmd_roll_rate = deque(maxlen=WINDOW_SIZE)
        self.cmd_pitch_rate = deque(maxlen=WINDOW_SIZE)
        self.cmd_yaw_rate = deque(maxlen=WINDOW_SIZE)

        # Track last time to detect resets
        self.last_time = 0.0

        # Create figure - 4 rows x 2 cols to include yaw
        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 14))
        self.fig.suptitle('Live Telemetry - PID Tuning View', fontsize=16)

        # Setup axes
        titles = [
            ('Altitude (m)', 'Time (s)', 'Altitude'),
            ('Airspeed (m/s)', 'Time (s)', 'Airspeed'),
            ('Roll (deg)', 'Time (s)', 'Roll Angle Warning: Watch for Oscillations!'),
            ('Pitch (deg)', 'Time (s)', 'Pitch Angle Warning: Watch for Oscillations!'),
            ('Yaw (deg)', 'Time (s)', 'Yaw Angle Warning: Watch for Oscillations!'),
            ('Yaw Rate (deg/s)', 'Time (s)', 'Yaw Rate (Rate Mode Tuning)'),
            ('Roll Rate (deg/s)', 'Time (s)', 'Roll Rate (Rate Mode Tuning)'),
            ('Pitch Rate (deg/s)', 'Time (s)', 'Pitch Rate (Rate Mode Tuning)'),
        ]

        for ax, (ylabel, xlabel, title) in zip(self.axes.flat, titles):
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Color the critical plots
        self.axes[1, 0].title.set_color('red')
        self.axes[1, 1].title.set_color('green')
        self.axes[2, 0].title.set_color('blue')

        plt.tight_layout()

    def update(self, frame):
        """Animation update function."""
        # Read state from file
        if not STATE_FILE.exists():
            return

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

            # Check for reset (time went backwards)
            current_time = state.get('time', 0.0)
            if current_time < self.last_time:
                print("Reset detected! Clearing plots...")
                self.time_data.clear()
                self.altitude.clear()
                self.airspeed.clear()
                self.roll.clear()
                self.pitch.clear()
                self.yaw.clear()
                self.roll_rate.clear()
                self.pitch_rate.clear()
                self.yaw_rate.clear()
                self.cmd_roll_angle.clear()
                self.cmd_pitch_angle.clear()
                self.cmd_yaw_angle.clear()
                self.cmd_roll_rate.clear()
                self.cmd_pitch_rate.clear()
                self.cmd_yaw_rate.clear()

            self.last_time = current_time

            # Append new data
            self.time_data.append(state.get('time', 0.0))
            self.altitude.append(state.get('altitude', 0.0))
            self.airspeed.append(state.get('airspeed', 0.0))
            self.roll.append(state.get('roll', 0.0))
            self.pitch.append(state.get('pitch', 0.0))
            self.yaw.append(state.get('yaw', 0.0))
            self.roll_rate.append(state.get('roll_rate', 0.0))
            self.pitch_rate.append(state.get('pitch_rate', 0.0))
            self.yaw_rate.append(state.get('yaw_rate', 0.0))

            # Append commanded values (may be None)
            self.cmd_roll_angle.append(state.get('cmd_roll_angle'))
            self.cmd_pitch_angle.append(state.get('cmd_pitch_angle'))
            self.cmd_yaw_angle.append(state.get('cmd_yaw_angle'))
            self.cmd_roll_rate.append(state.get('cmd_roll_rate'))
            self.cmd_pitch_rate.append(state.get('cmd_pitch_rate'))
            self.cmd_yaw_rate.append(state.get('cmd_yaw_rate'))

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return

        # Only plot if we have data
        if len(self.time_data) < 2:
            return

        # Convert to arrays
        t = np.array(self.time_data)
        alt = np.array(self.altitude)
        spd = np.array(self.airspeed)
        roll = np.array(self.roll)
        pitch = np.array(self.pitch)
        yaw = np.array(self.yaw)
        roll_rate = np.array(self.roll_rate)
        pitch_rate = np.array(self.pitch_rate)
        yaw_rate = np.array(self.yaw_rate)

        # Convert commanded values to arrays (filter out None values)
        cmd_roll_angle_data = list(self.cmd_roll_angle)
        cmd_pitch_angle_data = list(self.cmd_pitch_angle)
        cmd_yaw_angle_data = list(self.cmd_yaw_angle)
        cmd_roll_rate_data = list(self.cmd_roll_rate)
        cmd_pitch_rate_data = list(self.cmd_pitch_rate)
        cmd_yaw_rate_data = list(self.cmd_yaw_rate)

        # Clear and redraw all plots
        for ax in self.axes.flat:
            ax.clear()

        # Altitude
        self.axes[0, 0].plot(t, alt, 'b-', linewidth=2)
        self.axes[0, 0].set_ylabel('Altitude (m)', fontsize=10)
        self.axes[0, 0].set_xlabel('Time (s)', fontsize=9)
        self.axes[0, 0].set_title('Altitude', fontsize=11, fontweight='bold')
        self.axes[0, 0].grid(True, alpha=0.3)

        # Airspeed
        self.axes[0, 1].plot(t, spd, 'r-', linewidth=2)
        self.axes[0, 1].set_ylabel('Airspeed (m/s)', fontsize=10)
        self.axes[0, 1].set_xlabel('Time (s)', fontsize=9)
        self.axes[0, 1].set_title('Airspeed', fontsize=11, fontweight='bold')
        self.axes[0, 1].grid(True, alpha=0.3)

        # Roll (CRITICAL)
        self.axes[1, 0].plot(t, roll, 'r-', linewidth=2.5, label='Actual')
        # Add commanded roll if available (attitude mode)
        if any(v is not None for v in cmd_roll_angle_data):
            cmd_roll = np.array([v if v is not None else np.nan for v in cmd_roll_angle_data])
            self.axes[1, 0].plot(t, cmd_roll, 'r--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[1, 0].set_ylabel('Roll (deg)', fontsize=10)
        self.axes[1, 0].set_xlabel('Time (s)', fontsize=9)
        self.axes[1, 0].set_title('Roll Angle Warning: Watch for Oscillations!',
                                   fontsize=11, fontweight='bold', color='red')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend(loc='upper right', fontsize=8)

        # Pitch (CRITICAL)
        self.axes[1, 1].plot(t, pitch, 'g-', linewidth=2.5, label='Actual')
        # Add commanded pitch if available (attitude mode)
        if any(v is not None for v in cmd_pitch_angle_data):
            cmd_pitch = np.array([v if v is not None else np.nan for v in cmd_pitch_angle_data])
            self.axes[1, 1].plot(t, cmd_pitch, 'g--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[1, 1].set_ylabel('Pitch (deg)', fontsize=10)
        self.axes[1, 1].set_xlabel('Time (s)', fontsize=9)
        self.axes[1, 1].set_title('Pitch Angle Warning: Watch for Oscillations!',
                                   fontsize=11, fontweight='bold', color='green')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].legend(loc='upper right', fontsize=8)

        # Yaw (CRITICAL)
        self.axes[2, 0].plot(t, yaw, 'b-', linewidth=2.5, label='Actual')
        # Add commanded yaw if available (attitude mode)
        if any(v is not None for v in cmd_yaw_angle_data):
            cmd_yaw = np.array([v if v is not None else np.nan for v in cmd_yaw_angle_data])
            self.axes[2, 0].plot(t, cmd_yaw, 'b--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[2, 0].set_ylabel('Yaw (deg)', fontsize=10)
        self.axes[2, 0].set_xlabel('Time (s)', fontsize=9)
        self.axes[2, 0].set_title('Yaw Angle Warning: Watch for Oscillations!',
                                   fontsize=11, fontweight='bold', color='blue')
        self.axes[2, 0].grid(True, alpha=0.3)
        self.axes[2, 0].legend(loc='upper right', fontsize=8)

        # Yaw Rate
        self.axes[2, 1].plot(t, yaw_rate, 'cyan', linewidth=2, label='Actual')
        # Add commanded yaw rate if available (rate mode)
        if any(v is not None for v in cmd_yaw_rate_data):
            cmd_yaw_r = np.array([v if v is not None else np.nan for v in cmd_yaw_rate_data])
            self.axes[2, 1].plot(t, cmd_yaw_r, 'c--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[2, 1].set_ylabel('Yaw Rate (deg/s)', fontsize=10)
        self.axes[2, 1].set_xlabel('Time (s)', fontsize=9)
        self.axes[2, 1].set_title('Yaw Rate (Rate Mode Tuning)',
                                   fontsize=11, fontweight='bold')
        self.axes[2, 1].grid(True, alpha=0.3)
        self.axes[2, 1].legend(loc='upper right', fontsize=8)

        # Roll Rate
        self.axes[3, 0].plot(t, roll_rate, 'orange', linewidth=2, label='Actual')
        # Add commanded roll rate if available (rate mode)
        if any(v is not None for v in cmd_roll_rate_data):
            cmd_roll_r = np.array([v if v is not None else np.nan for v in cmd_roll_rate_data])
            self.axes[3, 0].plot(t, cmd_roll_r, 'orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[3, 0].set_ylabel('Roll Rate (deg/s)', fontsize=10)
        self.axes[3, 0].set_xlabel('Time (s)', fontsize=9)
        self.axes[3, 0].set_title('Roll Rate (Rate Mode Tuning)',
                                   fontsize=11, fontweight='bold')
        self.axes[3, 0].grid(True, alpha=0.3)
        self.axes[3, 0].legend(loc='upper right', fontsize=8)

        # Pitch Rate
        self.axes[3, 1].plot(t, pitch_rate, 'purple', linewidth=2, label='Actual')
        # Add commanded pitch rate if available (rate mode)
        if any(v is not None for v in cmd_pitch_rate_data):
            cmd_pitch_r = np.array([v if v is not None else np.nan for v in cmd_pitch_rate_data])
            self.axes[3, 1].plot(t, cmd_pitch_r, 'purple', linestyle='--', linewidth=1.5, alpha=0.7, label='Commanded')
        self.axes[3, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.axes[3, 1].set_ylabel('Pitch Rate (deg/s)', fontsize=10)
        self.axes[3, 1].set_xlabel('Time (s)', fontsize=9)
        self.axes[3, 1].set_title('Pitch Rate (Rate Mode Tuning)',
                                   fontsize=11, fontweight='bold')
        self.axes[3, 1].grid(True, alpha=0.3)
        self.axes[3, 1].legend(loc='upper right', fontsize=8)

    def run(self):
        """Start animation."""
        print("\n" + "="*70)
        print("LIVE TELEMETRY VIEWER - PID TUNING")
        print("="*70)
        print("\nWaiting for GUI to start...")
        print("\nWhat to look for:")
        print("  • Roll/Pitch plots - Main indicators for PID tuning")
        print("  • Smooth curves = Good tuning")
        print("  • Fast oscillations (1-2 Hz) = P gain too high")
        print("  • Slow oscillations (0.1-0.5 Hz) = I gain too high")
        print("  • Overshoot = Need more D gain or less P gain")
        print("\nPlots automatically reset when you press 'R' in GUI")
        print("\nPress Ctrl+C to stop.")
        print("="*70 + "\n")

        # Wait for state file
        while not STATE_FILE.exists():
            time.sleep(0.5)

        print("GUI detected! Starting plots...")

        # Start animation (updates every 100ms)
        # Note: Must keep reference to prevent garbage collection
        _ani = animation.FuncAnimation(
            self.fig, self.update, interval=100, cache_frame_data=False
        )

        plt.show()


if __name__ == '__main__':
    plotter = LiveTelemetryPlotter()
    plotter.run()
