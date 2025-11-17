"""Control Hierarchy Diagnostics - Debug each control level independently.

This tool helps identify which control level is causing issues by:
1. Showing commanded vs actual values at each level
2. Step response testing for each level
3. Performance metrics (rise time, overshoot, settling time)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.simplified_6dof import Simplified6DOF
from controllers.rate_agent import RateAgent
from controllers.attitude_agent import AttitudeAgent
from controllers.hsa_agent import HSAAgent
from controllers.waypoint_agent import WaypointAgent
from controllers.types import ControlCommand, ControlMode, Waypoint, load_config_from_yaml


def test_rate_controller(duration=5.0, dt=0.01):
    """Test Level 5: Rate controller with step inputs."""
    print("\n" + "="*70)
    print("LEVEL 5: RATE CONTROLLER TEST")
    print("="*70)

    config = load_config_from_yaml("controllers/config/pid_gains.yaml")
    aircraft = Simplified6DOF()
    agent = RateAgent(config)

    # Initialize at trim
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.0, 0.0, -100.0]  # 100m altitude
    initial_state[3:6] = [12.0, 0.0, 0.0]    # 12 m/s forward
    aircraft.state = initial_state

    # Test: Step input in roll rate
    steps = int(duration / dt)
    time = np.arange(steps) * dt

    # Data storage
    commanded_p = np.zeros(steps)
    actual_p = np.zeros(steps)
    commanded_q = np.zeros(steps)
    actual_q = np.zeros(steps)
    commanded_r = np.zeros(steps)
    actual_r = np.zeros(steps)

    for i in range(steps):
        state = aircraft.get_state()

        # Step commands at different times
        if time[i] < 1.0:
            # Steady state
            cmd = ControlCommand(mode=ControlMode.RATE, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, throttle=0.3)
        elif time[i] < 2.5:
            # Roll rate step
            cmd = ControlCommand(mode=ControlMode.RATE, roll_rate=np.radians(20), pitch_rate=0.0, yaw_rate=0.0, throttle=0.3)
        elif time[i] < 4.0:
            # Pitch rate step
            cmd = ControlCommand(mode=ControlMode.RATE, roll_rate=0.0, pitch_rate=np.radians(10), yaw_rate=0.0, throttle=0.3)
        else:
            # Yaw rate step
            cmd = ControlCommand(mode=ControlMode.RATE, roll_rate=0.0, pitch_rate=0.0, yaw_rate=np.radians(5), throttle=0.3)

        commanded_p[i] = cmd.roll_rate
        commanded_q[i] = cmd.pitch_rate
        commanded_r[i] = cmd.yaw_rate
        actual_p[i] = state.p
        actual_q[i] = state.q
        actual_r[i] = state.r

        surfaces = agent.compute_action(cmd, state, dt)
        aircraft.set_controls(surfaces)
        aircraft.step(dt)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Level 5: Rate Controller Step Response', fontsize=14, fontweight='bold')

    # Roll rate
    axes[0].plot(time, np.degrees(commanded_p), 'r--', label='Commanded p', linewidth=2)
    axes[0].plot(time, np.degrees(actual_p), 'b-', label='Actual p', linewidth=1.5)
    axes[0].set_ylabel('Roll Rate (deg/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Pitch rate
    axes[1].plot(time, np.degrees(commanded_q), 'r--', label='Commanded q', linewidth=2)
    axes[1].plot(time, np.degrees(actual_q), 'b-', label='Actual q', linewidth=1.5)
    axes[1].set_ylabel('Pitch Rate (deg/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Yaw rate
    axes[2].plot(time, np.degrees(commanded_r), 'r--', label='Commanded r', linewidth=2)
    axes[2].plot(time, np.degrees(actual_r), 'b-', label='Actual r', linewidth=1.5)
    axes[2].set_ylabel('Yaw Rate (deg/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs/diagnostics").mkdir(parents=True, exist_ok=True)
    plot_path = f"logs/diagnostics/rate_controller_test_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    # Compute metrics
    print("\nPerformance Metrics:")
    print(f"Roll rate tracking error (RMS): {np.sqrt(np.mean((commanded_p - actual_p)**2)):.4f} rad/s")
    print(f"Pitch rate tracking error (RMS): {np.sqrt(np.mean((commanded_q - actual_q)**2)):.4f} rad/s")
    print(f"Yaw rate tracking error (RMS): {np.sqrt(np.mean((commanded_r - actual_r)**2)):.4f} rad/s")

    return fig


def test_attitude_controller(duration=10.0, dt=0.01):
    """Test Level 4: Attitude controller with step inputs."""
    print("\n" + "="*70)
    print("LEVEL 4: ATTITUDE CONTROLLER TEST")
    print("="*70)

    config = load_config_from_yaml("controllers/config/pid_gains.yaml")
    aircraft = Simplified6DOF()
    agent = AttitudeAgent(config)

    # Initialize at trim
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.0, 0.0, -100.0]
    initial_state[3:6] = [12.0, 0.0, 0.0]
    aircraft.state = initial_state

    steps = int(duration / dt)
    time = np.arange(steps) * dt

    # Data storage
    commanded_roll = np.zeros(steps)
    actual_roll = np.zeros(steps)
    commanded_pitch = np.zeros(steps)
    actual_pitch = np.zeros(steps)
    commanded_yaw = np.zeros(steps)
    actual_yaw = np.zeros(steps)

    for i in range(steps):
        state = aircraft.get_state()

        # Step commands
        if time[i] < 2.0:
            cmd = ControlCommand(mode=ControlMode.ATTITUDE, roll_angle=0.0, pitch_angle=0.0, yaw_angle=0.0, throttle=0.3)
        elif time[i] < 5.0:
            # Roll step
            cmd = ControlCommand(mode=ControlMode.ATTITUDE, roll_angle=np.radians(10), pitch_angle=0.0, yaw_angle=0.0, throttle=0.3)
        elif time[i] < 8.0:
            # Pitch step
            cmd = ControlCommand(mode=ControlMode.ATTITUDE, roll_angle=0.0, pitch_angle=np.radians(5), yaw_angle=0.0, throttle=0.3)
        else:
            # Return to level
            cmd = ControlCommand(mode=ControlMode.ATTITUDE, roll_angle=0.0, pitch_angle=0.0, yaw_angle=0.0, throttle=0.3)

        commanded_roll[i] = cmd.roll_angle
        commanded_pitch[i] = cmd.pitch_angle
        commanded_yaw[i] = cmd.yaw_angle
        actual_roll[i] = state.roll
        actual_pitch[i] = state.pitch
        actual_yaw[i] = state.yaw

        surfaces = agent.compute_action(cmd, state, dt)
        aircraft.set_controls(surfaces)
        aircraft.step(dt)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Level 4: Attitude Controller Step Response', fontsize=14, fontweight='bold')

    axes[0].plot(time, np.degrees(commanded_roll), 'r--', label='Commanded', linewidth=2)
    axes[0].plot(time, np.degrees(actual_roll), 'b-', label='Actual', linewidth=1.5)
    axes[0].set_ylabel('Roll (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, np.degrees(commanded_pitch), 'r--', label='Commanded', linewidth=2)
    axes[1].plot(time, np.degrees(actual_pitch), 'b-', label='Actual', linewidth=1.5)
    axes[1].set_ylabel('Pitch (deg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, np.degrees(commanded_yaw), 'r--', label='Commanded', linewidth=2)
    axes[2].plot(time, np.degrees(actual_yaw), 'b-', label='Actual', linewidth=1.5)
    axes[2].set_ylabel('Yaw (deg)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/diagnostics/attitude_controller_test_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    print("\nPerformance Metrics:")
    print(f"Roll tracking error (RMS): {np.sqrt(np.mean((commanded_roll - actual_roll)**2)):.4f} rad")
    print(f"Pitch tracking error (RMS): {np.sqrt(np.mean((commanded_pitch - actual_pitch)**2)):.4f} rad")

    return fig


def test_hsa_controller(duration=60.0, dt=0.01):
    """Test Level 3: HSA controller with step inputs."""
    print("\n" + "="*70)
    print("LEVEL 3: HSA CONTROLLER TEST")
    print("="*70)

    config = load_config_from_yaml("controllers/config/pid_gains.yaml")
    aircraft = Simplified6DOF()
    agent = HSAAgent(config)

    # Initialize at trim
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.0, 0.0, -100.0]
    initial_state[3:6] = [12.0, 0.0, 0.0]
    aircraft.state = initial_state

    steps = int(duration / dt)
    time = np.arange(steps) * dt

    # Data storage
    commanded_heading = np.zeros(steps)
    actual_heading = np.zeros(steps)
    commanded_speed = np.zeros(steps)
    actual_speed = np.zeros(steps)
    commanded_altitude = np.zeros(steps)
    actual_altitude = np.zeros(steps)
    throttle = np.zeros(steps)
    roll_angle = np.zeros(steps)
    pitch_angle = np.zeros(steps)

    for i in range(steps):
        state = aircraft.get_state()

        # Step commands
        if time[i] < 10.0:
            # Steady state
            cmd = ControlCommand(mode=ControlMode.HSA, heading=0.0, speed=12.0, altitude=100.0)
        elif time[i] < 25.0:
            # Heading change (45°)
            cmd = ControlCommand(mode=ControlMode.HSA, heading=np.radians(45), speed=12.0, altitude=100.0)
        elif time[i] < 40.0:
            # Altitude change (+20m)
            cmd = ControlCommand(mode=ControlMode.HSA, heading=np.radians(45), speed=12.0, altitude=120.0)
        else:
            # Speed change (+3 m/s)
            cmd = ControlCommand(mode=ControlMode.HSA, heading=np.radians(45), speed=15.0, altitude=120.0)

        commanded_heading[i] = cmd.heading
        commanded_speed[i] = cmd.speed
        commanded_altitude[i] = cmd.altitude
        actual_heading[i] = state.heading
        actual_speed[i] = state.airspeed
        actual_altitude[i] = state.altitude
        roll_angle[i] = state.roll
        pitch_angle[i] = state.pitch

        surfaces = agent.compute_action(cmd, state, dt)
        throttle[i] = surfaces.throttle
        aircraft.set_controls(surfaces)
        aircraft.step(dt)

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('Level 3: HSA Controller Step Response', fontsize=14, fontweight='bold')

    axes[0].plot(time, np.degrees(commanded_heading), 'r--', label='Commanded', linewidth=2)
    axes[0].plot(time, np.degrees(actual_heading), 'b-', label='Actual', linewidth=1.5)
    axes[0].set_ylabel('Heading (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, commanded_altitude, 'r--', label='Commanded', linewidth=2)
    axes[1].plot(time, actual_altitude, 'b-', label='Actual', linewidth=1.5)
    axes[1].set_ylabel('Altitude (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, commanded_speed, 'r--', label='Commanded', linewidth=2)
    axes[2].plot(time, actual_speed, 'b-', label='Actual', linewidth=1.5)
    axes[2].set_ylabel('Airspeed (m/s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time, np.degrees(roll_angle), 'g-', label='Roll', linewidth=1.5)
    axes[3].plot(time, np.degrees(pitch_angle), 'm-', label='Pitch', linewidth=1.5)
    axes[3].set_ylabel('Attitude (deg)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time, throttle, 'k-', linewidth=1.5)
    axes[4].set_ylabel('Throttle')
    axes[4].set_xlabel('Time (s)')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([0, 1])

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/diagnostics/hsa_controller_test_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    print("\nPerformance Metrics:")
    print(f"Heading tracking error (RMS): {np.degrees(np.sqrt(np.mean((commanded_heading - actual_heading)**2))):.2f} deg")
    print(f"Altitude tracking error (RMS): {np.sqrt(np.mean((commanded_altitude - actual_altitude)**2)):.2f} m")
    print(f"Speed tracking error (RMS): {np.sqrt(np.mean((commanded_speed - actual_speed)**2)):.2f} m/s")
    print(f"Max roll angle: {np.degrees(np.max(np.abs(roll_angle))):.2f} deg")
    print(f"Max pitch angle: {np.degrees(np.max(np.abs(pitch_angle))):.2f} deg")
    print(f"Throttle range: [{np.min(throttle):.3f}, {np.max(throttle):.3f}]")

    return fig


if __name__ == "__main__":
    print("="*70)
    print("CONTROL HIERARCHY DIAGNOSTICS")
    print("="*70)
    print("\nTesting each control level independently...")
    print("This will help identify which level needs tuning.\n")

    # Test each level
    test_rate_controller()
    test_attitude_controller()
    test_hsa_controller()

    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\nCheck logs/diagnostics/ for detailed plots.")
    print("\nNext steps:")
    print("1. Review step responses for each level")
    print("2. Identify which level has poor tracking performance")
    print("3. Adjust PID gains for that level in controllers/config/pid_gains.yaml")
    print("4. Re-run diagnostics to verify improvements")
