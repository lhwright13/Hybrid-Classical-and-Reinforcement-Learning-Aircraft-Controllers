"""Gain Tuner - Interactive tool for tuning PID gains.

Usage:
    python validation/gain_tuner.py --controller hsa --param baseline_throttle --values 0.1,0.2,0.3
    python validation/gain_tuner.py --controller hsa --param heading_kp --values 0.8,1.0,1.2,1.5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from validation.comprehensive_waypoint_analysis import run_waypoint_mission


def test_parameter_sweep(controller, param_name, values, mission_duration=300.0):
    """Test a range of values for a specific parameter.

    Args:
        controller: Which controller to tune ('rate', 'attitude', 'hsa', 'waypoint')
        param_name: Parameter name (e.g., 'heading_kp', 'baseline_throttle')
        values: List of values to test
        mission_duration: How long to run each test
    """
    print("="*70)
    print(f"PARAMETER SWEEP: {controller}.{param_name}")
    print("="*70)
    print(f"Testing {len(values)} values: {values}\n")

    results = []

    for value in values:
        print(f"\nTesting {param_name} = {value}")
        print("-"*50)

        # TODO: Modify the controller parameter
        # This requires a way to inject parameters into the controller
        # For now, user must manually edit the code

        # Run mission
        result = run_waypoint_mission(duration=mission_duration)
        results.append({
            'value': value,
            'waypoints': result['waypoints_reached'],
            'altitude_error': result['altitude_error_rms'],
            'speed_error': result['speed_error_rms'],
            'max_altitude': result['max_altitude'],
            'min_altitude': result['min_altitude'],
            'max_speed': result['max_speed'],
        })

        print(f"  Waypoints: {result['waypoints_reached']}/5")
        print(f"  Altitude error (RMS): {result['altitude_error_rms']:.2f} m")
        print(f"  Speed error (RMS): {result['speed_error_rms']:.2f} m/s")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Parameter Sweep: {controller}.{param_name}', fontsize=14, fontweight='bold')

    x = [r['value'] for r in results]

    # Waypoints completed
    axes[0, 0].plot(x, [r['waypoints'] for r in results], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_ylabel('Waypoints Completed')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 5.5])

    # Altitude tracking
    axes[0, 1].plot(x, [r['altitude_error'] for r in results], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_ylabel('Altitude Error RMS (m)')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].grid(True, alpha=0.3)

    # Speed tracking
    axes[1, 0].plot(x, [r['speed_error'] for r in results], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_ylabel('Speed Error RMS (m/s)')
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].grid(True, alpha=0.3)

    # Altitude range
    axes[1, 1].fill_between(x, [r['min_altitude'] for r in results], [r['max_altitude'] for r in results], alpha=0.3)
    axes[1, 1].plot(x, [100]*len(x), 'k--', label='Target')
    axes[1, 1].set_ylabel('Altitude Range (m)')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs/tuning").mkdir(parents=True, exist_ok=True)
    plot_path = f"logs/tuning/{controller}_{param_name}_sweep_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n\nSaved sweep results: {plot_path}")

    # Print summary
    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print("="*70)
    best_waypoints = max(results, key=lambda r: r['waypoints'])
    best_altitude = min(results, key=lambda r: r['altitude_error'])
    best_speed = min(results, key=lambda r: r['speed_error'])

    print(f"\nBest waypoint completion: {param_name} = {best_waypoints['value']}")
    print(f"  → {best_waypoints['waypoints']}/5 waypoints")

    print(f"\nBest altitude tracking: {param_name} = {best_altitude['value']}")
    print(f"  → {best_altitude['altitude_error']:.2f} m RMS error")

    print(f"\nBest speed tracking: {param_name} = {best_speed['value']}")
    print(f"  → {best_speed['speed_error']:.2f} m/s RMS error")

    return results


def print_current_gains():
    """Print all current controller gains from hsa_agent.py"""
    print("\n" + "="*70)
    print("CURRENT HSA CONTROLLER PARAMETERS")
    print("="*70)

    try:
        with open("controllers/hsa_agent.py", 'r') as f:
            lines = f.readlines()

        print("\nSearching for tunable parameters...")
        print("-"*70)

        for i, line in enumerate(lines, 1):
            if "TUNED:" in line or "Baseline throttle" in line or "max bank" in line:
                # Print context
                print(f"\nLine {i}:")
                print(line.rstrip())

    except FileNotFoundError:
        print("Error: controllers/hsa_agent.py not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune controller gains')
    parser.add_argument('--controller', type=str, default='hsa',
                        choices=['rate', 'attitude', 'hsa', 'waypoint'],
                        help='Which controller to tune')
    parser.add_argument('--param', type=str,
                        help='Parameter name (e.g., heading_kp, baseline_throttle)')
    parser.add_argument('--values', type=str,
                        help='Comma-separated values to test (e.g., 0.1,0.2,0.3)')
    parser.add_argument('--show-gains', action='store_true',
                        help='Show current gain values')

    args = parser.parse_args()

    if args.show_gains:
        print_current_gains()
    elif args.param and args.values:
        values = [float(v) for v in args.values.split(',')]
        test_parameter_sweep(args.controller, args.param, values)
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("QUICK START EXAMPLES")
        print("="*70)
        print("\n1. Show current gains:")
        print("   python validation/gain_tuner.py --show-gains")
        print("\n2. Sweep baseline throttle:")
        print("   python validation/gain_tuner.py --controller hsa --param baseline_throttle --values 0.15,0.20,0.25,0.30")
        print("\n3. Sweep bank angle:")
        print("   python validation/gain_tuner.py --controller hsa --param max_bank_deg --values 8,10,12,15")
