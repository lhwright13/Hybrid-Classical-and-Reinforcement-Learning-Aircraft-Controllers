#!/usr/bin/env python3
"""PID Tuning Script - Systematically test gain combinations.

This script tests different PID gain sets to find stable control parameters.
Focuses on the most critical loops: altitude and heading control.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from controllers import (
    ControlCommand, ControlMode, AircraftState,
    ControllerConfig, PIDGains,
)


def create_tuned_config(
    altitude_kp=0.2,
    altitude_ki=0.02,
    altitude_kd=0.3,
    heading_kp=1.5,
    roll_rate_kp=0.25,
    pitch_rate_kp=0.25
):
    """Create controller config with specified gains."""
    config = ControllerConfig()

    # Keep default angle and rate gains from YAML
    # (These will be overridden in HSA agent, but needed for structure)
    config.roll_rate_gains = PIDGains(kp=roll_rate_kp, ki=0.20, kd=0.0002, i_limit=25.0)
    config.pitch_rate_gains = PIDGains(kp=pitch_rate_kp, ki=0.20, kd=0.0002, i_limit=25.0)
    config.yaw_gains = PIDGains(kp=0.20, ki=0.05, kd=0.00015, i_limit=25.0)  # Reduced from 0.30

    config.roll_angle_gains = PIDGains(kp=0.30, ki=0.20, kd=0.08, i_limit=25.0)  # Increased from 0.20
    config.pitch_angle_gains = PIDGains(kp=0.30, ki=0.20, kd=0.08, i_limit=25.0)

    config.max_roll_rate = 90.0  # Reduced from 180
    config.max_pitch_rate = 90.0
    config.max_yaw_rate = 60.0  # Reduced from 90

    config.max_roll = 35.0  # Reduced from 45
    config.max_pitch = 25.0  # Reduced from 30

    config.dt = 0.01

    # Store HSA gains for modification
    config.altitude_kp = altitude_kp
    config.altitude_ki = altitude_ki
    config.altitude_kd = altitude_kd
    config.heading_kp = heading_kp

    return config


def test_altitude_hold(config, duration=30.0):
    """Test altitude hold performance."""
    from controllers.hsa_agent import HSAAgent
    import aircraft_controls_bindings as acb

    # Create HSA agent with custom altitude gains
    agent = HSAAgent(config)

    # Override altitude PID with tuned gains
    altitude_config = acb.PIDConfig()
    altitude_config.gains = acb.PIDGains(
        kp=config.altitude_kp,
        ki=config.altitude_ki,
        kd=config.altitude_kd
    )
    altitude_config.integral_min = -10.0
    altitude_config.integral_max = 10.0
    altitude_config.output_min = -np.radians(20)  # Increased from 15°
    altitude_config.output_max = np.radians(20)
    agent.altitude_pid = acb.PIDController(altitude_config)

    # Override heading PID
    heading_config = acb.PIDConfig()
    heading_config.gains = acb.PIDGains(kp=config.heading_kp, ki=0.1, kd=0.5)
    heading_config.integral_min = -10.0
    heading_config.integral_max = 10.0
    heading_config.output_min = -np.radians(25)  # Reduced from 30°
    heading_config.output_max = np.radians(25)
    agent.heading_pid = acb.PIDController(heading_config)

    # Setup aircraft
    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})
    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),
        velocity=np.array([20.0, 0.0, 0.0]),
        attitude=np.zeros(3),
        angular_rate=np.zeros(3),
        airspeed=20.0,
        altitude=100.0
    )
    backend.reset(initial_state)

    # Command: hold altitude 100m, heading 0°, speed 20 m/s
    command = ControlCommand(
        mode=ControlMode.HSA,
        heading=0.0,
        speed=20.0,
        altitude=100.0
    )

    dt = 0.01
    num_steps = int(duration / dt)

    altitudes = []
    rolls = []
    pitches = []
    max_roll = 0
    max_pitch = 0

    for step in range(num_steps):
        state = backend.get_state()

        # Compute control
        surfaces = agent.compute_action(command, state, dt=dt)

        # Track metrics
        altitudes.append(state.altitude)
        rolls.append(np.degrees(state.roll))
        pitches.append(np.degrees(state.pitch))
        max_roll = max(max_roll, abs(state.roll))
        max_pitch = max(max_pitch, abs(state.pitch))

        backend.set_controls(surfaces)
        backend.step(dt)

    # Calculate performance metrics
    alt_final = altitudes[-1]
    alt_mean_last_5s = np.mean(altitudes[-500:])  # Last 5 seconds
    alt_std_last_5s = np.std(altitudes[-500:])
    alt_max = max(altitudes)
    alt_min = min(altitudes)
    alt_error = abs(alt_mean_last_5s - 100.0)

    # Check for divergence
    diverged = alt_min < 50 or alt_max > 150 or max_roll > np.radians(60) or max_pitch > np.radians(45)

    return {
        'altitude_error': alt_error,
        'altitude_std': alt_std_last_5s,
        'altitude_final': alt_final,
        'altitude_range': (alt_min, alt_max),
        'max_roll_deg': np.degrees(max_roll),
        'max_pitch_deg': np.degrees(max_pitch),
        'diverged': diverged,
        'stable': alt_error < 10.0 and alt_std_last_5s < 5.0 and not diverged
    }


def main():
    """Run PID tuning tests."""
    print("=" * 70)
    print("PID Tuning Script")
    print("=" * 70)

    print("\nTesting altitude hold with different gain sets...")
    print("Target: Hold 100m altitude for 30s\n")

    # Test configurations (altitude_kp, altitude_ki, altitude_kd, heading_kp)
    test_configs = [
        # (altitude_kp, ki, kd, heading_kp, roll_rate_kp, pitch_rate_kp, name)
        (0.05, 0.01, 0.2, 2.0, 0.15, 0.15, "Original (WEAK)"),
        (0.15, 0.02, 0.3, 1.5, 0.20, 0.20, "Conservative"),
        (0.20, 0.02, 0.3, 1.5, 0.25, 0.25, "Moderate"),
        (0.25, 0.03, 0.4, 1.5, 0.30, 0.30, "Aggressive"),
        (0.30, 0.03, 0.4, 1.2, 0.30, 0.30, "Very Aggressive"),
    ]

    results = []

    for alt_kp, alt_ki, alt_kd, hdg_kp, roll_kp, pitch_kp, name in test_configs:
        print(f"\nTesting: {name}")
        print(f"  Altitude PID: kp={alt_kp:.2f}, ki={alt_ki:.3f}, kd={alt_kd:.1f}")
        print(f"  Heading PID: kp={hdg_kp:.1f}")
        print(f"  Rate kp: roll={roll_kp:.2f}, pitch={pitch_kp:.2f}")

        config = create_tuned_config(
            altitude_kp=alt_kp,
            altitude_ki=alt_ki,
            altitude_kd=alt_kd,
            heading_kp=hdg_kp,
            roll_rate_kp=roll_kp,
            pitch_rate_kp=pitch_kp
        )

        result = test_altitude_hold(config, duration=30.0)
        results.append((name, result))

        status = "✓ STABLE" if result['stable'] else ("✗ DIVERGED" if result['diverged'] else "~ MARGINAL")
        print(f"  Result: {status}")
        print(f"    Altitude error: {result['altitude_error']:.2f}m")
        print(f"    Altitude std: {result['altitude_std']:.2f}m")
        print(f"    Altitude range: {result['altitude_range'][0]:.1f}m to {result['altitude_range'][1]:.1f}m")
        print(f"    Max roll: {result['max_roll_deg']:.1f}°")
        print(f"    Max pitch: {result['max_pitch_deg']:.1f}°")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Status':<15} {'Alt Error':<12} {'Alt Std':<10}")
    print("-" * 70)

    for name, result in results:
        status = "✓ STABLE" if result['stable'] else ("✗ DIVERGED" if result['diverged'] else "~ MARGINAL")
        print(f"{name:<25} {status:<15} {result['altitude_error']:>6.2f}m      {result['altitude_std']:>6.2f}m")

    # Find best
    stable_results = [(name, r) for name, r in results if r['stable']]
    if stable_results:
        best_name, best_result = min(stable_results, key=lambda x: x[1]['altitude_error'])
        print(f"\n✓ Best stable configuration: {best_name}")
        print(f"  Altitude error: {best_result['altitude_error']:.2f}m")
        print(f"  Altitude std: {best_result['altitude_std']:.2f}m")
    else:
        print("\n✗ No stable configurations found. Need more aggressive tuning.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
