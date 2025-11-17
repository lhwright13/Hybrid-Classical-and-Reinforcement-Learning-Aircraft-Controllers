#!/usr/bin/env python3
"""
Fly waypoint missions with FlightGear visualization

This script runs YOUR Python 6DOF simulation and displays the results in FlightGear.
FlightGear is used ONLY for visualization - all physics are computed by your Python sim.
"""

import socket
import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simplified_6dof import AircraftState, Simplified6DOF
from controllers.rate_agent import RateAgent
from controllers.attitude_agent import AttitudeAgent
from controllers.hsa_agent import HSAAgent
from controllers.waypoint_agent import WaypointAgent


class FlightGearVisualizer:
    """
    Visualizes Python simulation in FlightGear

    FlightGear is used ONLY as a 3D renderer - all physics computed by Python sim
    """

    def __init__(self, host='localhost', port=5401):
        """Initialize connection to FlightGear property server"""
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False

    def connect(self):
        """Connect to FlightGear"""
        try:
            print(f"Connecting to FlightGear at {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.1)
            self.connected = True
            print("✓ Connected to FlightGear!")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to FlightGear: {e}")
            print("  FlightGear must be running with --telnet=5401")
            self.connected = False
            return False

    def update_state(self, state: AircraftState):
        """
        Update FlightGear display to match Python simulation state

        This sends position/orientation from YOUR simulation to FlightGear
        """
        if not self.connected:
            return

        try:
            # Convert NED position to lat/lon (using KSFO as reference)
            # Reference point: KSFO (37.6177° N, 122.3750° W)
            ref_lat = 37.6177
            ref_lon = -122.3750

            # Convert north/east (meters) to lat/lon degrees
            # 1 degree latitude ≈ 111,111 meters
            # 1 degree longitude ≈ 111,111 * cos(latitude) meters
            meters_per_deg_lat = 111111.0
            meters_per_deg_lon = 111111.0 * np.cos(np.radians(ref_lat))

            lat_deg = ref_lat + (state.north / meters_per_deg_lat)
            lon_deg = ref_lon + (state.east / meters_per_deg_lon)
            alt_ft = state.altitude * 3.28084  # meters to feet

            roll_deg = np.degrees(state.roll)
            pitch_deg = np.degrees(state.pitch)
            heading_deg = np.degrees(state.heading)

            # Velocities
            airspeed_kt = state.airspeed * 1.94384  # m/s to knots

            # Send commands to FlightGear (disable its autopilot/physics)
            commands = [
                f"set /position/latitude-deg {lat_deg}",
                f"set /position/longitude-deg {lon_deg}",
                f"set /position/altitude-ft {alt_ft}",
                f"set /orientation/roll-deg {roll_deg}",
                f"set /orientation/pitch-deg {pitch_deg}",
                f"set /orientation/heading-deg {heading_deg}",
                f"set /velocities/airspeed-kt {airspeed_kt}",
            ]

            for cmd in commands:
                self.sock.send(f"{cmd}\r\n".encode())

        except Exception as e:
            # Don't spam errors, just silently fail
            if "Broken pipe" in str(e):
                self.connected = False

    def close(self):
        """Close connection"""
        if self.sock:
            self.sock.close()
        self.connected = False


def fly_mission_with_visualization(waypoints, duration=300.0, dt=0.01):
    """
    Fly waypoint mission with FlightGear visualization

    Args:
        waypoints: List of waypoints [(N, E, alt), ...]
        duration: Mission duration in seconds
        dt: Simulation timestep
    """

    print("\n" + "="*70)
    print("WAYPOINT MISSION WITH FLIGHTGEAR VISUALIZATION")
    print("="*70)
    print(f"\nMission: {len(waypoints)} waypoints")
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print("\n" + "="*70 + "\n")

    # Initialize FlightGear visualizer
    fg_viz = FlightGearVisualizer()
    if not fg_viz.connect():
        print("\n⚠ WARNING: FlightGear not connected - simulation will run without visualization")
        print("  To see visualization, make sure FlightGear is running and try again\n")
        time.sleep(2)

    # Initial state - start at first waypoint
    initial_state = AircraftState(
        time=0.0,
        position=np.array([waypoints[0][0], waypoints[0][1], -waypoints[0][2]]),  # NED: down is negative altitude
        velocity=np.array([12.0, 0.0, 0.0]),  # 12 m/s forward
        attitude=np.array([0.0, 0.0, 0.0]),  # level flight
        angular_rate=np.array([0.0, 0.0, 0.0]),  # no rotation
        airspeed=12.0,
        altitude=waypoints[0][2],
        ground_speed=12.0,
        heading=0.0
    )

    # Initialize simulation
    sim = Simplified6DOF()
    sim.reset(initial_state)

    # Initialize control hierarchy
    rate_agent = RateAgent()
    attitude_agent = AttitudeAgent()
    hsa_agent = HSAAgent()
    waypoint_agent = WaypointAgent(waypoints=waypoints)

    # Simulation loop
    print("Starting simulation...")
    print("Watch FlightGear window for visualization!\n")

    start_time = time.time()
    sim_time = 0.0
    steps = 0
    last_update = 0.0
    update_interval = 1.0  # Print status every second

    try:
        while sim_time < duration:
            # Get current state
            state = sim.get_state()

            # Control hierarchy (bottom to top)
            # Level 5: Waypoint → HSA commands
            hsa_cmd = waypoint_agent.compute_control(state)

            # Level 4: HSA → Attitude commands
            att_cmd = hsa_agent.compute_control(state, hsa_cmd)

            # Level 3: Attitude → Rate commands
            rate_cmd = attitude_agent.compute_control(state, att_cmd)

            # Level 2: Rate → Surface commands
            surfaces = rate_agent.compute_control(state, rate_cmd)

            # Set control surfaces and step simulation (YOUR physics, not FlightGear's)
            sim.set_controls(surfaces)
            state = sim.step(dt)

            # Update FlightGear visualization
            if fg_viz.connected:
                fg_viz.update_state(state)

            # Print status periodically
            if sim_time - last_update >= update_interval:
                wp_info = waypoint_agent.get_status()
                print(f"[{sim_time:6.1f}s] WP {wp_info['current_waypoint_index']+1}/{len(waypoints)} | "
                      f"Dist: {wp_info['distance_to_waypoint']:.0f}m | "
                      f"Alt: {state.altitude:.0f}m | "
                      f"Speed: {np.linalg.norm([state.u, state.v, state.w]):.1f}m/s | "
                      f"Hdg: {np.degrees(state.heading):.0f}°")
                last_update = sim_time

            # Check if mission complete
            if waypoint_agent.mission_complete():
                print(f"\n✓ Mission complete! All {len(waypoints)} waypoints reached in {sim_time:.1f}s")
                break

            sim_time += dt
            steps += 1

    except KeyboardInterrupt:
        print("\n\nMission interrupted by user")

    finally:
        # Cleanup
        fg_viz.close()

        # Final status
        print("\n" + "="*70)
        print("MISSION SUMMARY")
        print("="*70)
        wp_info = waypoint_agent.get_status()
        print(f"Waypoints reached: {wp_info['waypoints_reached']}/{len(waypoints)}")
        print(f"Simulation time: {sim_time:.1f}s")
        print(f"Steps: {steps}")
        print(f"Final altitude: {state.altitude:.1f}m")
        print(f"Final speed: {np.linalg.norm([state.u, state.v, state.w]):.1f}m/s")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Define waypoint mission (same as your validation tests)
    waypoints = [
        (0, 0, 100),        # WP1: Start
        (2000, 0, 100),     # WP2: North 2km
        (2000, 2000, 100),  # WP3: East 2km
        (0, 2000, 100),     # WP4: South 2km
        (0, 0, 100),        # WP5: Back to start
    ]

    print("\nStarting waypoint mission with FlightGear visualization...")
    print("\nIMPORTANT:")
    print("  - FlightGear is ONLY for visualization")
    print("  - All physics computed by YOUR Python 6DOF simulation")
    print("  - Watch FlightGear window to see your aircraft fly!\n")

    fly_mission_with_visualization(waypoints, duration=300.0, dt=0.01)
