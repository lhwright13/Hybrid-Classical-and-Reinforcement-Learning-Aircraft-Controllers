#!/usr/bin/env python3
"""Waypoint mission demonstration with trajectory visualization.

This script demonstrates waypoint navigation with the full control hierarchy:
- Single aircraft following a square waypoint pattern
- MissionPlanner sequencing through waypoints
- WaypointAgent (Level 1) cascading to HSA (Level 2)
- Telemetry logging and trajectory visualization
- PNG output showing actual path vs planned waypoints

This showcases the complete 5-level control hierarchy in action.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from validation.jsbsim_backend import JSBSimBackend
from interfaces.sensor import PerfectSensorInterface
from interfaces.aircraft_registry import AircraftRegistry
from controllers import (
    ControlCommand, ControlMode, AircraftState, Waypoint,
    ControllerConfig, PIDGains, WaypointAgent, MissionPlanner
)
from visualization.logger import TelemetryLogger
from visualization.plotter import MultiAircraftPlotter
from visualization.replay import MultiAircraftReplay


def load_controller_config(use_jsbsim=False):
    """Load PID gains from config file.

    Args:
        use_jsbsim: If True, load JSBSim-specific gains. Otherwise load default.
    """
    import yaml

    if use_jsbsim:
        config_file = "jsbsim_gains.yaml"
    else:
        config_file = "default_gains.yaml"

    config_path = Path(__file__).parent.parent / "configs" / "controllers" / config_file

    with open(config_path, 'r') as f:
        gains_config = yaml.safe_load(f)

    config = ControllerConfig()
    config.roll_rate_gains = PIDGains(**gains_config['roll_rate'])
    config.pitch_rate_gains = PIDGains(**gains_config['pitch_rate'])
    config.yaw_gains = PIDGains(**gains_config['yaw_rate'])
    config.roll_angle_gains = PIDGains(**gains_config['roll_angle'])
    config.pitch_angle_gains = PIDGains(**gains_config['pitch_angle'])
    config.heading_gains = PIDGains(**gains_config['heading'])
    config.speed_gains = PIDGains(**gains_config['speed'])
    config.altitude_gains = PIDGains(**gains_config['altitude'])
    config.max_roll_rate = gains_config['max_roll_rate']
    config.max_pitch_rate = gains_config['max_pitch_rate']
    config.max_yaw_rate = gains_config['max_yaw_rate']
    config.max_roll = gains_config['max_roll']
    config.max_pitch = gains_config['max_pitch']
    config.dt = gains_config['dt']

    return config


def create_square_mission(side_length: float = 1000.0, altitude: float = 100.0) -> list:
    """Create a square waypoint pattern.

    Args:
        side_length: Length of each side (meters)
        altitude: Flight altitude (meters, positive up)

    Returns:
        List of Waypoint objects forming a square
    """
    # Simple square: Four equal sides with 90° turns
    # Start at origin, fly counter-clockwise

    waypoints = [
        Waypoint.from_ned(0, 0, -altitude, speed=12.0),                    # WP1: Start (SW corner)
        Waypoint.from_ned(side_length, 0, -altitude, speed=12.0),          # WP2: North (NW corner)
        Waypoint.from_ned(side_length, side_length, -altitude, speed=12.0), # WP3: East (NE corner)
        Waypoint.from_ned(0, side_length, -altitude, speed=12.0),          # WP4: South (SE corner)
        Waypoint.from_ned(0, 0, -altitude, speed=12.0),                    # WP5: West back to start
    ]

    return waypoints


def main():
    """Run waypoint mission demonstration."""
    print("=" * 70)
    print("Waypoint Mission Demonstration")
    print("=" * 70)

    # Configuration
    aircraft_id = "001"
    duration = 300.0  # seconds (5 minutes)
    dt = 0.01  # 100 Hz simulation
    acceptance_radius = 300.0  # meters - wide radius to avoid triggering phugoid
    mission_size = 2000.0  # meters (2km square - good for demonstrating straight-line nav)
    altitude = 100.0  # meters

    print(f"\nConfiguration:")
    print(f"  Aircraft ID: {aircraft_id}")
    print(f"  Mission: Square pattern ({mission_size}m sides)")
    print(f"  Altitude: {altitude}m")
    print(f"  Acceptance radius: {acceptance_radius}m")
    print(f"  Max duration: {duration}s")
    print(f"  Simulation rate: {1/dt} Hz")

    # Create aircraft registry
    print("\n1. Initializing aircraft...")
    registry = AircraftRegistry()
    registry.register(aircraft_id, aircraft_type="rc_plane")
    aircraft_ids = [aircraft_id]

    # Create simulation backend - Use simplified 6DOF (JSBSim has stability issues)
    print("   Using simplified 6DOF simulator...")
    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})

    # Set initial state: level flight at 100m altitude
    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -altitude]),  # NED frame
        velocity=np.array([12.0, 0.0, 0.0]),       # 12 m/s forward for tighter turn radius
        attitude=np.zeros(3),                       # Level
        angular_rate=np.zeros(3),
        airspeed=12.0,
        altitude=altitude
    )
    backend.reset(initial_state)
    print(f"   Initial position: (N={initial_state.north:.1f}, E={initial_state.east:.1f}, Alt={altitude:.1f}m)")

    # Create sensor
    sensor = PerfectSensorInterface()
    sensor.update(initial_state)

    # Create controller (WaypointAgent - Level 1)
    # Load default PID gains (tuned for simplified sim)
    config = load_controller_config(use_jsbsim=False)
    agent = WaypointAgent(config, guidance_type='PP')  # Pure Pursuit guidance
    print(f"   Controller: {agent}")

    # Create mission
    print("\n2. Creating mission...")
    waypoints = create_square_mission(mission_size, altitude)
    planner = MissionPlanner(waypoints, acceptance_radius=acceptance_radius)
    print(f"   Mission: {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"     WP{i+1}: N={wp.north:.1f}, E={wp.east:.1f}, Alt={-wp.down:.1f}m")
    print(f"   Total distance: {planner.get_total_mission_distance():.1f}m")

    # Start mission
    planner.start()

    # Create telemetry logger
    print("\n3. Initializing telemetry logger...")
    log_dir = Path(__file__).parent.parent / "logs" / "waypoint_missions"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TelemetryLogger(str(log_dir / "waypoint_mission.hdf5"))
    logger.register_aircraft(aircraft_id, {"mission": "square", "waypoints": len(waypoints)})

    # Create plotter (disable rolling window to show full mission)
    print("\n4. Running simulation...")
    plotter = MultiAircraftPlotter(aircraft_ids, window_size=100000)  # Large window for full mission

    # Simulation loop
    num_steps = int(duration / dt)
    step = 0
    last_wp_index = -1

    while step < num_steps:
        t = step * dt

        # Get current state
        state = backend.get_state()
        state.time = t

        # Update mission planner
        waypoint_reached = planner.update(state)
        if waypoint_reached:
            current_wp = planner.current_waypoint_index
            if current_wp <= len(waypoints):
                print(f"   [{t:6.2f}s] Waypoint {current_wp}/{len(waypoints)} reached! "
                      f"Distance: {planner.waypoint_distances[-1]:.2f}m")
                # Debug: print next waypoint
                next_wp = planner.get_current_waypoint()
                if next_wp:
                    print(f"   [{t:6.2f}s] Advancing to WP{current_wp+1}: N={next_wp.north:.1f}, E={next_wp.east:.1f}")

        # Check if mission complete
        if planner.is_complete():
            print(f"\n   Mission complete at t={t:.2f}s!")
            print(f"   Duration: {planner.get_mission_duration():.2f}s")
            break

        # Get waypoint command
        command = planner.get_waypoint_command()
        if command is None:
            print("   Warning: No active waypoint, holding position")
            break

        # Compute control (cascades through all levels)
        surfaces = agent.compute_action(command, state, dt)

        # Apply control and step simulation
        backend.set_controls(surfaces)
        backend.step(dt)

        # Log telemetry
        next_state = backend.get_state()
        logger.log_state(aircraft_id, state)
        logger.log_command(aircraft_id, command, t)
        logger.log_surfaces(aircraft_id, surfaces, t)

        # Update plotter every 10 steps (for performance)
        if step % 10 == 0:
            plotter.update(aircraft_id, state)

        # Progress indicator
        if step % 1000 == 0:
            distance = planner.get_distance_to_current_waypoint(state)
            progress = planner.get_progress_percentage()
            print(f"   [{t:6.2f}s] Progress: {progress:5.1f}% | "
                  f"WP{planner.current_waypoint_index+1} distance: {distance:.1f}m")

        step += 1

    # Close logger
    logger.close()

    # Mission summary
    print("\n5. Mission Summary:")
    summary = planner.get_summary()
    print(f"   Status: {summary['state']}")
    print(f"   Waypoints reached: {summary['waypoints_reached']}/{summary['total_waypoints']}")
    print(f"   Progress: {summary['progress_percent']:.1f}%")
    if summary['mission_duration_s']:
        print(f"   Duration: {summary['mission_duration_s']:.2f}s")
    print(f"   Planned distance: {summary['total_distance_m']:.1f}m")

    if summary['waypoint_distances']:
        print(f"\n   Waypoint accuracy:")
        for i, dist in enumerate(summary['waypoint_distances']):
            print(f"     WP{i+1}: {dist:.2f}m error")
        avg_error = np.mean(summary['waypoint_distances'])
        max_error = np.max(summary['waypoint_distances'])
        print(f"     Average error: {avg_error:.2f}m")
        print(f"     Maximum error: {max_error:.2f}m")

    # Generate plots
    print("\n6. Generating trajectory visualization...")

    # Note: The plotter was updated in real-time during simulation
    # So we just need to set waypoints and save
    plotter.set_waypoints(waypoints)

    # Save plot
    output_path = Path(__file__).parent.parent / "logs" / "waypoint_missions" / "trajectory.png"
    plotter.save(str(output_path))
    print(f"   Saved trajectory plot: {output_path}")

    print("\n" + "=" * 70)
    print("Mission demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
