#!/usr/bin/env python3
"""Simple validation runner - compares simplified 6-DOF against JSBSim."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from validation.jsbsim_backend import JSBSimBackend
from validation.scenarios.level_flight import LevelFlightScenario
from validation.metrics.trajectory_metrics import compare_trajectories, format_metrics_summary


def main():
    """Run simple validation comparison."""
    print("=" * 70)
    print("PHYSICS VALIDATION: Simplified 6-DOF vs JSBSim")
    print("=" * 70)

    # Create scenario
    print("\n1. Creating scenario...")
    scenario = LevelFlightScenario()
    print(f"   Scenario: {scenario.get_name()}")
    print(f"   Description: {scenario.get_description()}")
    print(f"   Duration: {scenario.duration}s at {1/scenario.dt} Hz")

    # Create backends
    print("\n2. Initializing simulation backends...")
    simplified = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})
    jsbsim = JSBSimBackend({'aircraft': 'rc_plane'})  # Now using matched RC plane model!
    print(f"   Simplified 6-DOF: {simplified}")
    print(f"   JSBSim: {jsbsim}")

    # Run on simplified model
    print("\n3. Running scenario on Simplified 6-DOF...")
    df_simplified = scenario.run_simulation(simplified)
    print(f"   Generated {len(df_simplified)} data points")
    print(f"   Final altitude: {df_simplified['altitude'].iloc[-1]:.2f}m")
    print(f"   Final airspeed: {df_simplified['airspeed'].iloc[-1]:.2f}m/s")

    # Run on JSBSim
    print("\n4. Running scenario on JSBSim...")
    df_jsbsim = scenario.run_simulation(jsbsim)
    print(f"   Generated {len(df_jsbsim)} data points")
    print(f"   Final altitude: {df_jsbsim['altitude'].iloc[-1]:.2f}m")
    print(f"   Final airspeed: {df_jsbsim['airspeed'].iloc[-1]:.2f}m/s")

    # Compare trajectories
    print("\n5. Computing comparison metrics...")
    metrics = compare_trajectories(df_simplified, df_jsbsim)

    # Print summary
    print("\n")
    print(format_metrics_summary(metrics))

    # Check against expected thresholds
    print("\n6. Validating against expected criteria...")
    expected = scenario.get_expected_metrics()

    position_pass = metrics['position_3d_rmse'] < expected['position_rmse_threshold']
    attitude_pass = metrics['attitude_roll_rmse_deg'] < expected['attitude_rmse_threshold']
    corr_pass = metrics['overall_correlation'] > expected['min_correlation']

    print(f"   Position RMSE: {metrics['position_3d_rmse']:.2f}m "
          f"(threshold: {expected['position_rmse_threshold']}m) "
          f"{'PASS' if position_pass else 'FAIL'}")

    print(f"   Attitude RMSE: {metrics['attitude_roll_rmse_deg']:.2f}° "
          f"(threshold: {expected['attitude_rmse_threshold']}°) "
          f"{'PASS' if attitude_pass else 'FAIL'}")

    print(f"   Correlation: {metrics['overall_correlation']:.3f} "
          f"(threshold: {expected['min_correlation']}) "
          f"{'PASS' if corr_pass else 'FAIL'}")

    all_pass = position_pass and attitude_pass and corr_pass

    print("\n" + "=" * 70)
    if all_pass:
        print("VALIDATION PASSED - Simplified model matches JSBSim!")
    else:
        print("Warning: VALIDATION INCOMPLETE - Some metrics outside expected range")
        print("   This may indicate tuning needed or limitations of simplified model")
    print("=" * 70)

    # Save results
    print("\n7. Saving results...")
    output_dir = Path(__file__).parent / 'results' / 'raw_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    df_simplified.to_csv(output_dir / 'level_flight_simplified.csv', index=False)
    df_jsbsim.to_csv(output_dir / 'level_flight_jsbsim.csv', index=False)

    print(f"   Saved trajectories to: {output_dir}")
    print("\nValidation complete!")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
