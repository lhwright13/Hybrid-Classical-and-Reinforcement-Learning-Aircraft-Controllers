"""Comprehensive waypoint mission analysis with detailed event tracking."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

from controllers.waypoint_agent import WaypointAgent
from controllers.mission_planner import MissionPlanner, Waypoint
from controllers.types import ControlCommand, ControlMode, ControlSurfaces
from controllers.config import load_config_from_yaml
from simulation.simplified_6dof import Simplified6DOF


def run_comprehensive_analysis():
    """Run waypoint mission with comprehensive event tracking and visualization."""

    # Configuration
    config = load_config_from_yaml("controllers/config/pid_gains.yaml")

    # Create waypoints for square pattern (2000m sides)
    waypoints = [
        Waypoint.from_altitude(north=0.0, east=0.0, altitude=100.0),      # WP1: Origin
        Waypoint.from_altitude(north=2000.0, east=0.0, altitude=100.0),   # WP2: North
        Waypoint.from_altitude(north=2000.0, east=2000.0, altitude=100.0), # WP3: Northeast
        Waypoint.from_altitude(north=0.0, east=2000.0, altitude=100.0),   # WP4: East
        Waypoint.from_altitude(north=0.0, east=0.0, altitude=100.0),      # WP5: Return to origin
    ]

    # Initialize mission planner and controller
    planner = MissionPlanner(waypoints, acceptance_radius=300.0)
    agent = WaypointAgent(config, guidance_type="pure_pursuit")

    # Initialize aircraft at cruise condition: 100m altitude, 12 m/s airspeed
    aircraft = Simplified6DOF()

    # Set proper initial state (matching target flight condition to avoid energy deficit)
    # State: [position(3), velocity(3), attitude(3), angular_rate(3)] = 12 elements
    initial_state = np.zeros(12)
    initial_state[0:3] = [0.0, 0.0, -100.0]  # Position: origin at 100m altitude (NED: down=-100)
    initial_state[3:6] = [12.0, 0.0, 0.0]    # Velocity (body frame): 12 m/s forward (u, v, w)
    initial_state[6:9] = [0.0, 0.0, 0.0]     # Attitude: level flight (roll, pitch, yaw)
    initial_state[9:12] = [0.0, 0.0, 0.0]    # Angular rates: zero (p, q, r)
    aircraft.state = initial_state

    # Set initial controls to trim condition
    trim_surfaces = ControlSurfaces(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.5)
    aircraft.set_controls(trim_surfaces)

    state = aircraft.get_state()
    print(f"Initial state: Alt={state.altitude:.1f}m, Speed={state.airspeed:.1f}m/s, "
          f"Pos=({state.north:.1f}, {state.east:.1f})\n")

    # Simulation parameters
    dt = 0.01  # 100 Hz
    duration = 300.0  # 5 minutes
    steps = int(duration / dt)

    # Data storage
    data = {
        'time': [],
        'north': [], 'east': [], 'altitude': [],
        'airspeed': [], 'heading': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'throttle': [],
        'cmd_north': [], 'cmd_east': [], 'cmd_altitude': [], 'cmd_speed': [], 'cmd_heading': [],
        'waypoint_index': [],
        'waypoint_distance': [],
        'energy_total': [],
        'energy_balance': [],
        'energy_cmd_total': [],
        'energy_cmd_balance': [],
    }

    # Event log
    events = []

    print("=" * 70)
    print("COMPREHENSIVE WAYPOINT MISSION ANALYSIS")
    print("=" * 70)
    print(f"\nMission: {len(waypoints)} waypoints, 8000m total distance")
    print(f"Acceptance radius: {planner.acceptance_radius}m")
    print(f"Bank angle limit: 10°")
    print(f"TECS enabled\n")

    # Start mission
    planner.start()

    # Run simulation
    for step in range(steps):
        t = step * dt

        # Update mission planner
        waypoint_reached = planner.update(state)
        if waypoint_reached:
            wp_idx = planner.current_waypoint_index
            dist = planner.waypoint_distances[-1]
            events.append({
                'time': t,
                'type': 'waypoint_reached',
                'waypoint': wp_idx,
                'distance': dist,
                'altitude': state.altitude,
                'airspeed': state.airspeed,
            })
            print(f"[{t:6.2f}s] ✓ Waypoint {wp_idx}/{len(waypoints)} reached! "
                  f"Distance: {dist:.2f}m, Alt: {state.altitude:.1f}m, Speed: {state.airspeed:.1f}m/s")

            # Get next waypoint
            current_wp = planner.get_current_waypoint()
            if current_wp:
                print(f"           → Advancing to WP{wp_idx+1}: "
                      f"N={current_wp.north:.1f}, E={current_wp.east:.1f}, Alt={current_wp.altitude:.1f}")

        # Get waypoint command
        command = planner.get_waypoint_command()
        if command is None:
            print(f"[{t:6.2f}s] Warning: No active waypoint, stopping simulation")
            break

        surfaces = agent.compute_action(command, state, dt)

        # Step simulation
        aircraft.set_controls(surfaces)
        aircraft.step(dt)
        state = aircraft.get_state()

        # Get waypoint for logging
        current_wp = command.waypoint

        # Calculate distance to current waypoint
        wp_dist = np.sqrt((current_wp.north - state.position[0])**2 +
                         (current_wp.east - state.position[1])**2)

        # Calculate energy states (for TECS analysis)
        g = 9.81
        E_total = g * state.altitude + 0.5 * state.airspeed**2
        E_balance = g * state.altitude - 0.5 * state.airspeed**2
        E_cmd_total = g * current_wp.altitude + 0.5 * (current_wp.speed if current_wp.speed else 12.0)**2
        E_cmd_balance = g * current_wp.altitude - 0.5 * (current_wp.speed if current_wp.speed else 12.0)**2

        # Log data
        data['time'].append(t)
        data['north'].append(state.position[0])
        data['east'].append(state.position[1])
        data['altitude'].append(state.altitude)
        data['airspeed'].append(state.airspeed)
        data['heading'].append(np.degrees(state.heading))
        data['roll'].append(np.degrees(state.roll))
        data['pitch'].append(np.degrees(state.pitch))
        data['yaw'].append(np.degrees(state.yaw))
        data['throttle'].append(surfaces.throttle)
        data['cmd_north'].append(current_wp.north)
        data['cmd_east'].append(current_wp.east)
        data['cmd_altitude'].append(current_wp.altitude)
        data['cmd_speed'].append(current_wp.speed if current_wp.speed else 12.0)
        data['cmd_heading'].append(np.degrees(command.heading) if hasattr(command, 'heading') and command.heading is not None else 0)
        data['waypoint_index'].append(planner.current_waypoint_index)
        data['waypoint_distance'].append(wp_dist)
        data['energy_total'].append(E_total)
        data['energy_balance'].append(E_balance)
        data['energy_cmd_total'].append(E_cmd_total)
        data['energy_cmd_balance'].append(E_cmd_balance)

        # Log significant events
        if step > 0:
            # Check for large altitude deviations
            if abs(state.altitude - current_wp.altitude) > 100:
                if step % 1000 == 0:  # Log every 10 seconds
                    events.append({
                        'time': t,
                        'type': 'altitude_deviation',
                        'altitude': state.altitude,
                        'target': current_wp.altitude,
                        'error': state.altitude - current_wp.altitude,
                    })

            # Check for speed deviations
            target_speed = current_wp.speed if current_wp.speed else 12.0
            if abs(state.airspeed - target_speed) > 20:
                if step % 1000 == 0:
                    events.append({
                        'time': t,
                        'type': 'speed_deviation',
                        'airspeed': state.airspeed,
                        'target': target_speed,
                        'error': state.airspeed - target_speed,
                    })

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    # Find waypoint transition times
    wp_changes = []
    for i in range(1, len(data['waypoint_index'])):
        if data['waypoint_index'][i] != data['waypoint_index'][i-1]:
            wp_changes.append((data['time'][i], data['waypoint_index'][i]))

    # Create comprehensive visualization
    print(f"\n{'=' * 70}")
    print("Creating comprehensive analysis plots...")

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Color for waypoint transitions
    wp_color = 'red'
    wp_alpha = 0.3

    # Row 1: Position and trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['time'], data['north'], 'b-', label='North', linewidth=1.5)
    ax1.plot(data['time'], data['cmd_north'], 'r--', label='Cmd North', alpha=0.7)
    for t, wp_idx in wp_changes:
        ax1.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
        ax1.text(t, ax1.get_ylim()[1]*0.95, f'WP{wp_idx}', rotation=90,
                va='top', ha='right', fontsize=8, color=wp_color)
    ax1.set_ylabel('North (m)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('North Position')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data['time'], data['east'], 'b-', label='East', linewidth=1.5)
    ax2.plot(data['time'], data['cmd_east'], 'r--', label='Cmd East', alpha=0.7)
    for t, wp_idx in wp_changes:
        ax2.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax2.set_ylabel('East (m)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('East Position')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(data['east'], data['north'], 'b-', linewidth=2, label='Trajectory')
    # Plot waypoints
    for i, wp in enumerate(waypoints, 1):
        ax3.plot(wp.east, wp.north, 'r*', markersize=15)
        ax3.text(wp.east, wp.north+100, f'WP{i}', ha='center', fontsize=10, color='red')
    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Trajectory (Top View)')
    ax3.legend()

    # Row 2: Altitude and airspeed (with commands)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(data['time'], data['altitude'], 'b-', label='Altitude', linewidth=1.5)
    ax4.plot(data['time'], data['cmd_altitude'], 'r--', label='Cmd Altitude', alpha=0.7)
    ax4.axhline(100, color='g', linestyle=':', alpha=0.5, label='Target (100m)')
    for t, wp_idx in wp_changes:
        ax4.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
        if wp_idx <= len(wp_changes):
            ax4.text(t, ax4.get_ylim()[1]*0.95, f'WP{wp_idx}', rotation=90,
                    va='top', ha='right', fontsize=8, color=wp_color)
    ax4.set_ylabel('Altitude (m)')
    ax4.set_xlabel('Time (s)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Altitude Control')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(data['time'], data['airspeed'], 'b-', label='Airspeed', linewidth=1.5)
    ax5.plot(data['time'], data['cmd_speed'], 'r--', label='Cmd Speed', alpha=0.7)
    ax5.axhline(12, color='g', linestyle=':', alpha=0.5, label='Target (12m/s)')
    for t, wp_idx in wp_changes:
        ax5.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax5.set_ylabel('Airspeed (m/s)')
    ax5.set_xlabel('Time (s)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Airspeed Control')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(data['time'], data['waypoint_distance'], 'b-', linewidth=1.5)
    ax6.axhline(300, color='r', linestyle='--', alpha=0.7, label='Acceptance (300m)')
    for t, wp_idx in wp_changes:
        ax6.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax6.set_ylabel('Distance (m)')
    ax6.set_xlabel('Time (s)')
    ax6.set_yscale('log')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Distance to Current Waypoint')

    # Row 3: Attitude
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(data['time'], data['roll'], 'b-', label='Roll', linewidth=1.5)
    ax7.axhline(8, color='r', linestyle='--', alpha=0.5, label='Limit (±8°)')
    ax7.axhline(-8, color='r', linestyle='--', alpha=0.5)
    for t, wp_idx in wp_changes:
        ax7.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax7.set_ylabel('Roll (deg)')
    ax7.set_xlabel('Time (s)')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Roll Angle')

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(data['time'], data['pitch'], 'b-', label='Pitch', linewidth=1.5)
    ax8.axhline(10, color='r', linestyle='--', alpha=0.5, label='Limit (±10°)')
    ax8.axhline(-10, color='r', linestyle='--', alpha=0.5)
    for t, wp_idx in wp_changes:
        ax8.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax8.set_ylabel('Pitch (deg)')
    ax8.set_xlabel('Time (s)')
    ax8.legend(loc='best')
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Pitch Angle')

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(data['time'], data['heading'], 'b-', label='Heading', linewidth=1.5)
    for t, wp_idx in wp_changes:
        ax9.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax9.set_ylabel('Heading (deg)')
    ax9.set_xlabel('Time (s)')
    ax9.legend(loc='best')
    ax9.grid(True, alpha=0.3)
    ax9.set_title('Heading')

    # Row 4: TECS Energy Analysis
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(data['time'], data['energy_total'], 'b-', label='Total Energy', linewidth=1.5)
    ax10.plot(data['time'], data['energy_cmd_total'], 'r--', label='Cmd Total Energy', alpha=0.7)
    for t, wp_idx in wp_changes:
        ax10.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax10.set_ylabel('Specific Energy (J/kg)')
    ax10.set_xlabel('Time (s)')
    ax10.legend(loc='best')
    ax10.grid(True, alpha=0.3)
    ax10.set_title('TECS: Total Energy (E = g*h + 0.5*V²)')

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(data['time'], data['energy_balance'], 'b-', label='Energy Balance', linewidth=1.5)
    ax11.plot(data['time'], data['energy_cmd_balance'], 'r--', label='Cmd Balance', alpha=0.7)
    for t, wp_idx in wp_changes:
        ax11.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax11.set_ylabel('Energy Balance (J/kg)')
    ax11.set_xlabel('Time (s)')
    ax11.legend(loc='best')
    ax11.grid(True, alpha=0.3)
    ax11.set_title('TECS: Energy Balance (E = g*h - 0.5*V²)')

    ax12 = fig.add_subplot(gs[3, 2])
    energy_total_error = data['energy_total'] - data['energy_cmd_total']
    energy_balance_error = data['energy_balance'] - data['energy_cmd_balance']
    ax12.plot(data['time'], energy_total_error, 'b-', label='Total Energy Error', linewidth=1.5)
    ax12.plot(data['time'], energy_balance_error, 'r-', label='Balance Error', linewidth=1.5)
    ax12.axhline(0, color='k', linestyle=':', alpha=0.5)
    for t, wp_idx in wp_changes:
        ax12.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax12.set_ylabel('Energy Error (J/kg)')
    ax12.set_xlabel('Time (s)')
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)
    ax12.set_title('TECS: Energy Errors')

    # Row 5: Control outputs and errors
    ax13 = fig.add_subplot(gs[4, 0])
    ax13.plot(data['time'], data['throttle'], 'b-', linewidth=1.5)
    ax13.axhline(0.5, color='g', linestyle=':', alpha=0.5, label='Baseline (0.5)')
    for t, wp_idx in wp_changes:
        ax13.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax13.set_ylabel('Throttle')
    ax13.set_xlabel('Time (s)')
    ax13.legend(loc='best')
    ax13.grid(True, alpha=0.3)
    ax13.set_title('Throttle Command')
    ax13.set_ylim([0, 1])

    ax14 = fig.add_subplot(gs[4, 1])
    altitude_error = data['altitude'] - data['cmd_altitude']
    speed_error = data['airspeed'] - data['cmd_speed']
    ax14.plot(data['time'], altitude_error, 'b-', label='Altitude Error', linewidth=1.5)
    ax14.axhline(0, color='k', linestyle=':', alpha=0.5)
    for t, wp_idx in wp_changes:
        ax14.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax14.set_ylabel('Altitude Error (m)')
    ax14.set_xlabel('Time (s)')
    ax14.legend(loc='best')
    ax14.grid(True, alpha=0.3)
    ax14.set_title('Altitude Tracking Error')

    ax15 = fig.add_subplot(gs[4, 2])
    ax15.plot(data['time'], speed_error, 'r-', label='Speed Error', linewidth=1.5)
    ax15.axhline(0, color='k', linestyle=':', alpha=0.5)
    for t, wp_idx in wp_changes:
        ax15.axvline(t, color=wp_color, alpha=wp_alpha, linestyle='--')
    ax15.set_ylabel('Speed Error (m/s)')
    ax15.set_xlabel('Time (s)')
    ax15.legend(loc='best')
    ax15.grid(True, alpha=0.3)
    ax15.set_title('Speed Tracking Error')

    plt.suptitle('Comprehensive Waypoint Mission Analysis with TECS', fontsize=16, y=0.995)

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/waypoint_missions/comprehensive_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {filename}")

    # Print event summary
    print(f"\n{'=' * 70}")
    print("EVENT SUMMARY")
    print(f"{'=' * 70}\n")

    waypoint_events = [e for e in events if e['type'] == 'waypoint_reached']
    print(f"Waypoints reached: {len(waypoint_events)}/{len(waypoints)}")
    print(f"\nWaypoint Timeline:")
    for e in waypoint_events:
        print(f"  [{e['time']:6.2f}s] WP{e['waypoint']}: "
              f"dist={e['distance']:.2f}m, alt={e['altitude']:.1f}m, speed={e['airspeed']:.1f}m/s")

    # Analyze phugoid characteristics
    print(f"\n{'=' * 70}")
    print("PHUGOID OSCILLATION ANALYSIS")
    print(f"{'=' * 70}\n")

    alt_min, alt_max = data['altitude'].min(), data['altitude'].max()
    speed_min, speed_max = data['airspeed'].min(), data['airspeed'].max()

    print(f"Altitude range: {alt_min:.1f}m to {alt_max:.1f}m (amplitude: ±{(alt_max-alt_min)/2:.1f}m)")
    print(f"Airspeed range: {speed_min:.1f} to {speed_max:.1f} m/s (amplitude: ±{(speed_max-speed_min)/2:.1f}m/s)")

    # Find oscillation period (rough estimate from altitude zero-crossings)
    alt_centered = data['altitude'] - 100.0
    zero_crossings = []
    for i in range(1, len(alt_centered)):
        if alt_centered[i-1] * alt_centered[i] < 0:
            zero_crossings.append(data['time'][i])

    if len(zero_crossings) >= 4:
        periods = []
        for i in range(2, len(zero_crossings), 2):
            periods.append(zero_crossings[i] - zero_crossings[i-2])
        avg_period = np.mean(periods)
        print(f"Phugoid period: ~{avg_period:.1f}s ({60/avg_period:.2f} cycles/min)")

    print(f"\n{'=' * 70}")

    return data, events


if __name__ == "__main__":
    data, events = run_comprehensive_analysis()
