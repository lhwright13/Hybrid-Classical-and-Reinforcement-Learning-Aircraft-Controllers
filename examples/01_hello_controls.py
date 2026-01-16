#!/usr/bin/env python3
"""Hello Controls - The Simplest Flight Control Demo

This is the absolute simplest example demonstrating:
- Creating a simulation backend
- Using the RateAgent (Level 4: Rate control)
- Commanding angular rates
- Visualizing results

Expected output: Aircraft tracks a 30°/s roll rate command in ~1-2 seconds.

Usage:
    python examples/01_hello_controls.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from controllers import (
    RateAgent,
    ControlCommand,
    ControlMode,
    AircraftState,
    ControllerConfig,
)


def create_simple_config():
    """Create a basic controller configuration.

    Uses default gains from ControllerConfig which are tuned for the
    stable aircraft model.
    """
    config = ControllerConfig()
    # Use default tuned PID gains from types.py
    # No need to override - they're already tuned for the stable aircraft model
    return config


def main():
    print("=" * 60)
    print("Hello Controls - Your First Flight Simulation!")
    print("=" * 60)
    print()
    print("This demo commands a 30°/s roll rate and watches the aircraft respond.")
    print("The RateAgent (Level 4) uses a C++ PID controller running at 1000 Hz.")
    print()

    # 1. Create configuration
    config = create_simple_config()

    # 2. Create simulation backend
    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})

    # 3. Create rate controller (Level 4)
    agent = RateAgent(config)

    # 4. Initialize aircraft state (level flight at 20 m/s, 100m altitude)
    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),  # NED: z is negative
        velocity=np.array([20.0, 0.0, 0.0]),    # Flying north at 20 m/s
        attitude=np.zeros(3),                    # Level flight (roll=pitch=yaw=0)
        angular_rate=np.zeros(3),                # No rotation initially
        airspeed=20.0,
        altitude=100.0
    )
    backend.reset(initial_state)

    # 5. Simulation parameters
    sim_time = 5.0  # seconds
    dt = 0.01       # 100 Hz simulation timestep
    steps = int(sim_time / dt)

    # 6. Create control command: 30°/s roll rate
    roll_rate_cmd = np.radians(30)  # Convert to rad/s

    command = ControlCommand(
        mode=ControlMode.RATE,
        roll_rate=roll_rate_cmd,
        pitch_rate=0.0,  # No pitch
        yaw_rate=0.0,    # No yaw
        throttle=0.7     # Moderate throttle
    )

    # 7. Storage for plotting
    history = {
        'time': [],
        'roll_rate_cmd': [],
        'roll_rate_actual': [],
        'pitch_rate': [],
        'yaw_rate': [],
        'roll_angle': [],
        'aileron': [],
        'elevator': [],
        'rudder': []
    }

    # 8. Run simulation
    print(f"Running {sim_time}s simulation...")
    print(f"Commanding: {np.degrees(roll_rate_cmd):.1f}°/s roll rate")
    print()

    state = backend.get_state()

    for i in range(steps):
        t = i * dt

        # Compute control action (pass dt for correct PID behavior)
        surfaces = agent.compute_action(command, state, dt=dt)

        # Step simulation
        backend.set_controls(surfaces)
        backend.step(dt)
        state = backend.get_state()

        # Log data
        history['time'].append(t)
        history['roll_rate_cmd'].append(np.degrees(roll_rate_cmd))
        history['roll_rate_actual'].append(np.degrees(state.p))
        history['pitch_rate'].append(np.degrees(state.q))
        history['yaw_rate'].append(np.degrees(state.r))
        history['roll_angle'].append(np.degrees(state.roll))
        history['aileron'].append(surfaces.aileron)
        history['elevator'].append(surfaces.elevator)
        history['rudder'].append(surfaces.rudder)

        # Print progress every second
        if i % 100 == 0:
            print(f"t={t:.1f}s: Roll rate = {np.degrees(state.p):6.1f}°/s "
                  f"(target: {np.degrees(roll_rate_cmd):6.1f}°/s), "
                  f"Roll angle = {np.degrees(state.roll):6.1f}°")

    print()
    print("Simulation complete!")
    print()

    # 9. Calculate performance metrics
    final_error = abs(history['roll_rate_actual'][-1] - history['roll_rate_cmd'][-1])
    print(f"Final tracking error: {final_error:.2f}°/s")

    # Find settling time (within 5% of command)
    threshold = 0.05 * np.degrees(roll_rate_cmd)
    errors = [abs(actual - cmd) for actual, cmd in
              zip(history['roll_rate_actual'], history['roll_rate_cmd'])]

    settling_idx = None
    for i in range(len(errors) - 50):  # Check if stays within threshold for 0.5s
        if all(e < threshold for e in errors[i:i+50]):
            settling_idx = i
            break

    if settling_idx:
        settling_time = history['time'][settling_idx]
        print(f"Settling time: {settling_time:.2f}s (within 5% of command)")

    print()
    print("=" * 60)

    # 10. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Rate Control Demo - Level 4 (Inner Loop)', fontsize=16, fontweight='bold')

    # Plot 1: Angular Rates
    ax1 = axes[0]
    ax1.plot(history['time'], history['roll_rate_cmd'], 'k--', label='Roll Rate Command', linewidth=2)
    ax1.plot(history['time'], history['roll_rate_actual'], 'r-', label='Roll Rate Actual', linewidth=1.5)
    ax1.plot(history['time'], history['pitch_rate'], 'g-', label='Pitch Rate', linewidth=1, alpha=0.7)
    ax1.plot(history['time'], history['yaw_rate'], 'b-', label='Yaw Rate', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Angular Rate (°/s)', fontsize=12)
    ax1.set_title('Angular Rates: Command vs Actual', fontsize=13)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 2: Roll Angle (Integration of roll rate)
    ax2 = axes[1]
    ax2.plot(history['time'], history['roll_angle'], 'r-', label='Roll Angle', linewidth=1.5)
    ax2.set_ylabel('Roll Angle (°)', fontsize=12)
    ax2.set_title('Roll Angle (Integration of Roll Rate)', fontsize=13)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 3: Control Surfaces
    ax3 = axes[2]
    ax3.plot(history['time'], history['aileron'], 'r-', label='Aileron', linewidth=1.5)
    ax3.plot(history['time'], history['elevator'], 'g-', label='Elevator', linewidth=1.5, alpha=0.7)
    ax3.plot(history['time'], history['rudder'], 'b-', label='Rudder', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Deflection (-1 to 1)', fontsize=12)
    ax3.set_title('Control Surface Deflections', fontsize=13)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylim(-1.1, 1.1)

    plt.tight_layout()

    print("\nDisplaying results plot...")
    print("   - Top: Angular rates (red line should track black dashed command)")
    print("   - Middle: Roll angle increases as aircraft rolls continuously")
    print("   - Bottom: Control surfaces (mainly aileron for roll control)")
    print()
    print("Success! The PID controller tracks the commanded roll rate.")
    print()
    print("Next steps:")
    print("  - Try examples/launch_pygame_gui.py for interactive control")
    print("  - Try examples/02_rl_vs_pid_demo.py to compare RL vs PID")
    print()

    plt.show()


if __name__ == "__main__":
    main()
