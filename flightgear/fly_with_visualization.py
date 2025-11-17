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

from validation.jsbsim_backend import JSBSimBackend
from controllers.waypoint_agent import WaypointAgent
from controllers.mission_planner import MissionPlanner, Waypoint
from controllers.types import ControllerConfig, AircraftState, ControlMode, ControlCommand, PIDGains
import yaml
from pathlib import Path


def load_controller_config(config_file="jsbsim_gains.yaml"):
    """Load PID gains from config file.

    Args:
        config_file: Name of config file in configs/controllers/
    """
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
    config.max_roll = gains_config['max_roll']
    config.max_pitch = gains_config['max_pitch']
    config.max_roll_rate = gains_config['max_roll_rate']
    config.max_pitch_rate = gains_config['max_pitch_rate']
    config.max_yaw_rate = gains_config['max_yaw_rate']

    return config


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

            # Disable FlightGear's flight dynamics - we control everything!
            print("  Freezing FlightGear's physics engine...")
            self.sock.send(b"set /sim/freeze/flight-model 1\r\n")
            self.sock.send(b"set /controls/engines/engine[0]/cutoff 0\r\n")  # Keep engine on for realism
            time.sleep(0.1)
            print("  ✓ FlightGear physics disabled - full Python control!")

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

    # Convert waypoints to Waypoint objects
    waypoint_objects = [Waypoint.from_altitude(north=wp[0], east=wp[1], altitude=wp[2]) for wp in waypoints]

    # Initialize mission planner and controller with JSBSim-specific gains
    config = load_controller_config("jsbsim_gains.yaml")
    planner = MissionPlanner(waypoint_objects, acceptance_radius=300.0)
    planner.start()  # Start the mission
    agent = WaypointAgent(config, guidance_type="pure_pursuit")

    # Initialize JSBSim simulation (much more realistic than simplified 6DOF!)
    print("\n🚀 Using JSBSim for realistic flight dynamics...")
    jsbsim_config = {
        'aircraft': 'c172p',  # Cessna 172
        'initial_lat': 37.6177,  # KSFO
        'initial_lon': -122.3750,
        'initial_altitude': waypoints[0][2],  # Start altitude
        'dt_physics': 0.01  # 100 Hz physics
    }
    sim = JSBSimBackend(jsbsim_config)

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

            # Update mission planner (checks waypoint reached and advances)
            planner.update(state)

            # Get current waypoint command
            waypoint_cmd = planner.get_waypoint_command()

            # If no waypoint (mission complete or not started), break
            if waypoint_cmd is None:
                break

            # Compute control surfaces from waypoint command
            surfaces = agent.compute_action(waypoint_cmd, state, dt=dt)

            # Set control surfaces and step simulation (YOUR physics, not FlightGear's)
            sim.set_controls(surfaces)
            state = sim.step(dt)

            # Update FlightGear visualization
            if fg_viz.connected:
                fg_viz.update_state(state)

            # Print status periodically
            if sim_time - last_update >= update_interval:
                summary = planner.get_summary()
                current_wp = summary['current_waypoint_index']
                dist = planner.get_distance_to_current_waypoint(state)
                print(f"[{sim_time:6.1f}s] WP {current_wp+1}/{len(waypoints)} | "
                      f"Dist: {dist:.0f}m | "
                      f"Alt: {state.altitude:.0f}m | "
                      f"Speed: {state.airspeed:.1f}m/s | "
                      f"Hdg: {np.degrees(state.heading):.0f}°")
                last_update = sim_time

            # Check if mission complete
            if planner.is_complete():
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
        summary = planner.get_summary()
        print(f"Waypoints reached: {summary['waypoints_reached']}/{len(waypoints)}")
        print(f"Simulation time: {sim_time:.1f}s")
        print(f"Steps: {steps}")
        final_state = sim.get_state()
        print(f"Final altitude: {final_state.altitude:.1f}m")
        print(f"Final speed: {final_state.airspeed:.1f}m/s")
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
