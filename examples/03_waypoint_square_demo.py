#!/usr/bin/env python3
"""Waypoint Navigation Demo - 300m Square Pattern.

This demo flies a square waypoint pattern and generates trajectory plots.
All controllers run at 1kHz for optimal performance.

Configuration is loaded from YAML files in configs/:
    - configs/missions/square_pattern.yaml
    - configs/controllers/cascaded_pid.yaml

Usage:
    python examples/03_waypoint_square_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from simulation.simplified_6dof import Simplified6DOF, AircraftParams
from controllers.waypoint_agent import WaypointAgent
from controllers.mission_planner import MissionPlanner
from controllers.types import ControllerConfig, Waypoint, AircraftState
from controllers.config_loader import (
    load_controller_config,
    load_mission_config,
)


# =============================================================================
# Load Configuration from YAML
# =============================================================================

# Load configs from YAML files
MISSION_CONFIG = load_mission_config("square_pattern.yaml")
FLIGHT_CONFIG = load_controller_config("cascaded_pid.yaml")

# Extract mission parameters
SQUARE_SIZE = MISSION_CONFIG.pattern_size
ALTITUDE = MISSION_CONFIG.altitude
SPEED = MISSION_CONFIG.speed
GUIDANCE = MISSION_CONFIG.guidance
MAX_DURATION = MISSION_CONFIG.max_duration
DT = MISSION_CONFIG.dt

# Acceptance radius from guidance config
ACCEPTANCE_RADIUS = FLIGHT_CONFIG.guidance.acceptance_radius

OUTPUT_DIR = Path(__file__).parent.parent / MISSION_CONFIG.output_dir


# =============================================================================
# Mission Definition
# =============================================================================

def create_square_mission():
    """Create a counter-clockwise square waypoint pattern."""
    waypoints = [
        Waypoint.from_altitude(0, 0, ALTITUDE, speed=SPEED),                      # WP1: Start
        Waypoint.from_altitude(SQUARE_SIZE, 0, ALTITUDE, speed=SPEED),            # WP2: North
        Waypoint.from_altitude(SQUARE_SIZE, SQUARE_SIZE, ALTITUDE, speed=SPEED),  # WP3: NE
        Waypoint.from_altitude(0, SQUARE_SIZE, ALTITUDE, speed=SPEED),            # WP4: East
        Waypoint.from_altitude(0, 0, ALTITUDE, speed=SPEED),                      # WP5: Back
    ]
    return waypoints


# =============================================================================
# Simulation
# =============================================================================

def run_mission():
    """Run the waypoint mission and collect telemetry."""

    print("=" * 80)
    print(f"Waypoint Navigation Demo - {MISSION_CONFIG.name}")
    print("=" * 80)
    print()
    print("Configuration (from YAML):")
    print(f"  Square size: {SQUARE_SIZE}m")
    print(f"  Altitude: {ALTITUDE}m")
    print(f"  Speed: {SPEED} m/s")
    print(f"  Acceptance radius: {ACCEPTANCE_RADIUS}m")
    print(f"  Guidance: {GUIDANCE}")
    print(f"  Lookahead time: {FLIGHT_CONFIG.guidance.lookahead_time}s")
    print(f"  Max bank angle: {FLIGHT_CONFIG.hsa.max_bank_angle} deg")
    print()

    # Initialize simulation and controllers
    aircraft_config = AircraftParams()
    sim = Simplified6DOF(aircraft_config)

    # Set initial state: at origin, 100m altitude, flying north at SPEED m/s
    initial_state = AircraftState(
        position=np.array([0.0, 0.0, -ALTITUDE]),  # NED: 100m altitude
        velocity=np.array([SPEED, 0.0, 0.0]),      # Flying forward at cruise speed
        attitude=np.zeros(3),                       # Level flight, heading north
        angular_rate=np.zeros(3),
        airspeed=SPEED,
        altitude=ALTITUDE
    )
    sim.reset(initial_state)

    # Legacy config for backwards compatibility
    ctrl_config = ControllerConfig()
    ctrl_config.waypoint_acceptance_radius = ACCEPTANCE_RADIUS

    # Create waypoint agent with both configs
    waypoint_agent = WaypointAgent(
        ctrl_config,
        guidance_type=GUIDANCE,
        flight_config=FLIGHT_CONFIG
    )

    # Create mission
    waypoints = create_square_mission()
    mission = MissionPlanner(waypoints, acceptance_radius=ACCEPTANCE_RADIUS)

    # Telemetry storage
    history = {
        'time': [],
        'north': [],
        'east': [],
        'altitude': [],
        'heading': [],
        'airspeed': [],
        'roll': [],
        'pitch': [],
        'waypoint_index': [],
        'distance_to_wp': [],
    }

    # Waypoint events
    wp_events = []

    # Start mission
    mission.start()
    print("Running simulation...")
    print(f"[{0:7.2f}s] Started mission, heading to WP1")

    t = 0.0
    last_wp_index = 0

    while t < MAX_DURATION:
        state = sim.get_state()

        # Record telemetry every step (DT=0.01 is already 10ms)
        # This ensures the final position is recorded when mission completes
        current_wp = mission.get_current_waypoint()
        dist_to_wp = np.sqrt(
            (state.north - current_wp.north)**2 +
            (state.east - current_wp.east)**2
        ) if current_wp else 0

        history['time'].append(t)
        history['north'].append(state.north)
        history['east'].append(state.east)
        history['altitude'].append(state.altitude)
        history['heading'].append(np.degrees(state.heading))
        history['airspeed'].append(state.airspeed)
        history['roll'].append(np.degrees(state.roll))
        history['pitch'].append(np.degrees(state.pitch))
        history['waypoint_index'].append(mission.current_waypoint_index)
        history['distance_to_wp'].append(dist_to_wp)

        # Update mission state
        mission.update(state)

        # Check for waypoint transitions
        current_wp_index = mission.current_waypoint_index
        if current_wp_index != last_wp_index:
            # Calculate error at waypoint
            wp = waypoints[last_wp_index]
            error = np.sqrt(
                (state.north - wp.north)**2 +
                (state.east - wp.east)**2 +
                (state.altitude - wp.altitude)**2
            )
            wp_events.append({
                'time': t,
                'wp_index': last_wp_index,
                'error': error
            })
            if current_wp_index < len(waypoints):
                print(f"[{t:7.2f}s] WP{last_wp_index+1} reached (error: {error:5.1f}m), heading to WP{current_wp_index+1}")
            else:
                print(f"[{t:7.2f}s] WP{last_wp_index+1} reached (error: {error:5.1f}m)")
            last_wp_index = current_wp_index

        # Check mission complete
        if mission.is_complete():
            print(f"[{t:7.2f}s] Mission complete!")
            break

        # Get waypoint command
        wp_cmd = mission.get_waypoint_command()
        if wp_cmd is None:
            break

        # Compute control
        surfaces = waypoint_agent.compute_action(wp_cmd, state, DT)
        sim.set_controls(surfaces)
        sim.step(DT)

        t += DT

    # Convert to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    return history, waypoints, wp_events, mission


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(history, waypoints, wp_events):
    """Generate trajectory and performance plots."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Extract data
    times = history['time']
    north = history['north']
    east = history['east']
    altitude = history['altitude']
    heading = history['heading']
    airspeed = history['airspeed']

    # Waypoint positions
    wp_north = [wp.north for wp in waypoints]
    wp_east = [wp.east for wp in waypoints]

    # --- Plot 1: 2D Trajectory ---
    fig1, ax1 = plt.subplots(figsize=(10, 10))

    # Planned path
    ax1.plot(wp_east + [wp_east[0]], wp_north + [wp_north[0]],
             'r--', linewidth=2, label='Planned Path', zorder=1)

    # Actual trajectory
    ax1.plot(east, north, 'b-', linewidth=1.5, label='Actual Trajectory', zorder=2)

    # Waypoint markers
    for i, (e, n) in enumerate(zip(wp_east, wp_north)):
        ax1.plot(e, n, 'rx', markersize=15, markeredgewidth=3, zorder=3)
        ax1.annotate(f'WP{i+1}', (e, n), xytext=(10, 10),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    # Start and end markers
    ax1.plot(east[0], north[0], 'go', markersize=12, label='Start', zorder=4)
    ax1.plot(east[-1], north[-1], 'rs', markersize=12, label='End', zorder=4)

    # Acceptance radius circles
    for e, n in zip(wp_east, wp_north):
        circle = plt.Circle((e, n), ACCEPTANCE_RADIUS, fill=False,
                           color='gray', linestyle=':', alpha=0.5)
        ax1.add_patch(circle)

    ax1.set_xlabel('East (m)', fontsize=12)
    ax1.set_ylabel('North (m)', fontsize=12)
    ax1.set_title('Waypoint Navigation - 2D Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-50, SQUARE_SIZE + 100)
    ax1.set_ylim(-50, SQUARE_SIZE + 100)

    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / 'waypoint_trajectory_2d.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waypoint_trajectory_2d.png'}")

    # --- Plot 2: Altitude Profile ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.axhline(y=ALTITUDE, color='r', linestyle='--', linewidth=2, label=f'Target ({ALTITUDE}m)')
    ax2.fill_between(times, ALTITUDE - 5, ALTITUDE + 5, alpha=0.2, color='green', label='+/- 5m band')
    ax2.plot(times, altitude, 'b-', linewidth=1.5, label='Actual')

    # Mark waypoint captures
    for event in wp_events:
        ax2.axvline(x=event['time'], color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title('Altitude Profile', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ALTITUDE - 20, ALTITUDE + 20)

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'waypoint_altitude_profile.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waypoint_altitude_profile.png'}")

    # --- Plot 3: Heading Tracking ---
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.plot(times, heading, 'b-', linewidth=1.5, label='Actual Heading')

    # Mark waypoint captures
    for event in wp_events:
        ax3.axvline(x=event['time'], color='gray', linestyle=':', alpha=0.5,
                   label='WP Capture' if event == wp_events[0] else '')

    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Heading (deg)', fontsize=12)
    ax3.set_title('Heading During Mission', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'waypoint_heading_tracking.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waypoint_heading_tracking.png'}")

    # --- Plot 4: Mission Summary (2x2) ---
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Waypoint Mission Summary - 300m Square', fontsize=14, fontweight='bold')

    # Top-left: Trajectory
    ax = axes[0, 0]
    ax.plot(wp_east + [wp_east[0]], wp_north + [wp_north[0]], 'r--', linewidth=2)
    ax.plot(east, north, 'b-', linewidth=1.5)
    for i, (e, n) in enumerate(zip(wp_east, wp_north)):
        ax.plot(e, n, 'rx', markersize=12, markeredgewidth=2)
        ax.annotate(f'WP{i+1}', (e, n), xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Trajectory')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Top-right: Altitude
    ax = axes[0, 1]
    ax.axhline(y=ALTITUDE, color='r', linestyle='--', linewidth=2)
    ax.plot(times, altitude, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)

    # Bottom-left: Heading
    ax = axes[1, 0]
    ax.plot(times, heading, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (deg)')
    ax.set_title('Heading')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Airspeed
    ax = axes[1, 1]
    ax.axhline(y=SPEED, color='r', linestyle='--', linewidth=2, label=f'Target ({SPEED} m/s)')
    ax.plot(times, airspeed, 'b-', linewidth=1.5, label='Actual')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Airspeed (m/s)')
    ax.set_title('Airspeed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / 'waypoint_mission_summary.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waypoint_mission_summary.png'}")

    plt.close('all')


def print_summary(history, waypoints, wp_events, mission):
    """Print mission summary statistics."""

    print()
    print("=" * 80)
    print("Mission Summary")
    print("=" * 80)
    print()

    # Completion
    num_wps = len(waypoints)
    num_reached = len(wp_events)
    duration = history['time'][-1]

    print("Completion:")
    print(f"  Waypoints: {num_reached}/{num_wps} ({100*num_reached/num_wps:.0f}%)")
    print(f"  Duration: {duration:.2f} seconds")

    # Planned vs actual distance
    planned_dist = 4 * SQUARE_SIZE

    # Calculate actual distance traveled
    north = history['north']
    east = history['east']
    actual_dist = 0
    for i in range(1, len(north)):
        actual_dist += np.sqrt((north[i]-north[i-1])**2 + (east[i]-east[i-1])**2)

    print(f"  Distance: {planned_dist:.0f}m (planned) / {actual_dist:.0f}m (actual)")
    print()

    # Waypoint accuracy
    print("Waypoint Accuracy:")
    errors = [e['error'] for e in wp_events]
    for i, event in enumerate(wp_events):
        print(f"  WP{event['wp_index']+1}: {event['error']:5.1f}m error")
    print(f"  Average: {np.mean(errors):.1f}m")
    print(f"  Maximum: {np.max(errors):.1f}m")
    print()

    # Tracking performance
    print("Tracking Performance:")
    alt_error = history['altitude'] - ALTITUDE
    alt_rmse = np.sqrt(np.mean(alt_error**2))
    print(f"  Altitude RMSE: {alt_rmse:.2f}m")

    speed_error = history['airspeed'] - SPEED
    speed_rmse = np.sqrt(np.mean(speed_error**2))
    print(f"  Speed RMSE: {speed_rmse:.2f} m/s")
    print()

    print(f"Output files saved to: {OUTPUT_DIR}/")
    print("  - waypoint_trajectory_2d.png")
    print("  - waypoint_altitude_profile.png")
    print("  - waypoint_heading_tracking.png")
    print("  - waypoint_mission_summary.png")
    print()
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run mission
    history, waypoints, wp_events, mission = run_mission()

    # Generate plots
    generate_plots(history, waypoints, wp_events)

    # Print summary
    print_summary(history, waypoints, wp_events, mission)
