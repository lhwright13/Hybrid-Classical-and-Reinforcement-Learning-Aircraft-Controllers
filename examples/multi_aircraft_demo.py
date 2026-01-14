#!/usr/bin/env python3
"""Multi-aircraft demonstration with visualization and logging.

This script demonstrates the multi-aircraft visualization capabilities:
- 3 aircraft flying in formation
- Real-time telemetry logging to HDF5
- Multi-aircraft matplotlib plotter
- 3D fleet visualization (optional)
- Replay from saved logs

This showcases Phase 4: Visualization & Monitoring
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from interfaces.sensor import PerfectSensorInterface
from interfaces.aircraft_registry import AircraftRegistry, AircraftStatus
from controllers import (
    ControlCommand, ControlMode, AircraftState,
    ControllerConfig, PIDGains, HSAAgent
)
from visualization.logger import TelemetryLogger
from visualization.plotter import MultiAircraftPlotter
from visualization.replay import MultiAircraftReplay

# Optional: 3D visualization (requires PyVista)
try:
    from visualization.aircraft_3d import FleetVisualizer3D
    HAS_PYVISTA = True
except ImportError:
    from visualization.aircraft_3d import SimpleFleetVisualizer3D as FleetVisualizer3D
    HAS_PYVISTA = False


def load_controller_config():
    """Load PID gains from config file."""
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "default_gains.yaml"

    with open(config_path, 'r') as f:
        gains_config = yaml.safe_load(f)

    config = ControllerConfig()
    config.roll_rate_gains = PIDGains(**gains_config['roll_rate'])
    config.pitch_rate_gains = PIDGains(**gains_config['pitch_rate'])
    config.yaw_gains = PIDGains(**gains_config['yaw_rate'])
    config.roll_angle_gains = PIDGains(**gains_config['roll_angle'])
    config.pitch_angle_gains = PIDGains(**gains_config['pitch_angle'])
    config.max_roll_rate = gains_config['max_roll_rate']
    config.max_pitch_rate = gains_config['max_pitch_rate']
    config.max_yaw_rate = gains_config['max_yaw_rate']
    config.max_roll = gains_config['max_roll']
    config.max_pitch = gains_config['max_pitch']
    config.dt = gains_config['dt']

    return config


def main():
    """Run multi-aircraft demonstration."""
    print("=" * 70)
    print("Multi-Aircraft Demonstration (Phase 4: Visualization & Monitoring)")
    print("=" * 70)

    # Configuration
    num_aircraft = 3
    duration = 30.0  # seconds
    dt = 0.01  # 100 Hz simulation
    plot_interval = 10  # Update plots every 10 steps (10 Hz)
    viz_3d_enabled = False  # Set to True to enable 3D visualization

    aircraft_ids = [f"{i:03d}" for i in range(1, num_aircraft + 1)]

    print(f"\nConfiguration:")
    print(f"  Aircraft count: {num_aircraft}")
    print(f"  Aircraft IDs: {aircraft_ids}")
    print(f"  Duration: {duration}s")
    print(f"  Simulation rate: {1/dt} Hz")
    print(f"  Plot update rate: {1/(dt*plot_interval)} Hz")

    # Create aircraft registry
    print("\n1. Initializing aircraft registry...")
    registry = AircraftRegistry()
    for aircraft_id in aircraft_ids:
        registry.register(aircraft_id, aircraft_type="rc_plane")
        print(f"   Registered aircraft {aircraft_id} (color: {registry.get_color(aircraft_id)})")

    # Create simulation backends (one per aircraft)
    print("\n2. Creating simulation backends...")
    backends = {}
    sensors = {}
    agents = {}
    config = load_controller_config()

    for aircraft_id in aircraft_ids:
        backends[aircraft_id] = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})
        sensors[aircraft_id] = PerfectSensorInterface()
        agents[aircraft_id] = HSAAgent(config)

    # Formation initial positions (line formation)
    formation_spacing = 50.0  # meters between aircraft
    print("\n3. Initializing formation (line formation)...")
    for i, aircraft_id in enumerate(aircraft_ids):
        # Offset in East direction
        east_offset = (i - num_aircraft // 2) * formation_spacing
        initial = AircraftState(
            altitude=100.0,
            airspeed=20.0,
            position=np.array([0.0, east_offset, -100.0]),
            velocity=np.array([20.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3)
        )
        backends[aircraft_id].reset(initial)
        sensors[aircraft_id].update(backends[aircraft_id].get_state())
        registry.update_status(aircraft_id, AircraftStatus.ACTIVE)
        print(f"   Aircraft {aircraft_id} at position (N={0:.1f}, E={east_offset:.1f}, Alt=100m)")

    # Create telemetry logger
    log_file = "examples/multi_aircraft_flight.h5"
    print(f"\n4. Creating telemetry logger ({log_file})...")
    logger = TelemetryLogger(log_file)
    for aircraft_id in aircraft_ids:
        logger.register_aircraft(aircraft_id, metadata={'type': 'rc_plane', 'formation_index': aircraft_ids.index(aircraft_id)})

    # Create visualizers
    print("\n5. Creating visualizers...")
    plotter = MultiAircraftPlotter(aircraft_ids, window_size=500)
    print(f"   Multi-aircraft matplotlib plotter: ready")

    viz_3d = None
    if viz_3d_enabled:
        viz_3d = FleetVisualizer3D(aircraft_ids, trajectory_length=500)
        print(f"   3D fleet visualizer ({'PyVista' if HAS_PYVISTA else 'matplotlib'}): ready")

    # Flight plan: Formation flight with waypoint navigation
    print("\n6. Flight plan:")
    print("   0-10s:  Maintain altitude, heading North, formation line")
    print("   10-20s: Turn East (heading 90°), climb to 120m")
    print("   20-30s: Turn South (heading 180°), descend to 100m")

    # Run simulation
    print("\n7. Running simulation...")
    print(f"   Progress: ", end='', flush=True)

    step_count = int(duration / dt)
    progress_interval = step_count // 20  # 5% intervals

    for i in range(step_count):
        t = i * dt

        # Progress indicator
        if i % progress_interval == 0:
            print(f"{int(100*i/step_count)}%... ", end='', flush=True)

        # Flight plan commands (same for all aircraft in formation)
        if t < 10.0:
            # Heading North, altitude 100m
            heading = np.radians(0)
            altitude = 100.0
            speed = 20.0
        elif t < 20.0:
            # Heading East, climb to 120m
            heading = np.radians(90)
            altitude = 120.0
            speed = 22.0
        else:
            # Heading South, descend to 100m
            heading = np.radians(180)
            altitude = 100.0
            speed = 20.0

        # Update each aircraft
        for j, aircraft_id in enumerate(aircraft_ids):
            # Add small formation offset to heading for spacing
            formation_offset = (j - num_aircraft // 2) * 0.05  # Slight heading variation
            aircraft_heading = heading + formation_offset

            # HSA command
            command = ControlCommand(
                mode=ControlMode.HSA,
                heading=aircraft_heading,
                speed=speed,
                altitude=altitude,
                throttle=0.7
            )

            # Get state, compute control, step simulation
            state = backends[aircraft_id].get_state()
            surfaces = agents[aircraft_id].compute_action(command, state, dt=dt)
            backends[aircraft_id].set_controls(surfaces)
            true_state = backends[aircraft_id].step(dt)
            sensors[aircraft_id].update(true_state)
            state = sensors[aircraft_id].get_state()

            # Log data
            logger.log_state(aircraft_id, state, t)
            logger.log_command(aircraft_id, command, t)
            logger.log_surfaces(aircraft_id, surfaces, t)

            # Update visualizers
            if i % plot_interval == 0:
                plotter.update(aircraft_id, state)
                if viz_3d:
                    viz_3d.update(aircraft_id, state)

        # Render visualizations
        if i % plot_interval == 0:
            plotter.plot()
            if viz_3d:
                viz_3d.render(camera_mode='fleet_center')

    print("100% Done!")

    # Close logger
    print("\n8. Closing telemetry logger and saving data...")
    logger.close()

    # Final statistics
    print("\n9. Flight statistics:")
    for aircraft_id in aircraft_ids:
        final_state = sensors[aircraft_id].get_state()
        print(f"   Aircraft {aircraft_id}:")
        print(f"     Final altitude: {final_state.altitude:.1f}m")
        print(f"     Final airspeed: {final_state.airspeed:.1f}m/s")
        print(f"     Final position: N={final_state.position[0]:.1f}m, E={final_state.position[1]:.1f}m")
        print(f"     Distance traveled: {np.linalg.norm(final_state.position[:2]):.1f}m")

    # Registry summary
    print(f"\n10. Registry summary:")
    summary = registry.get_summary()
    print(f"   Total aircraft: {summary['total']}")
    print(f"   Active aircraft: {summary['active']}")

    # Save final plot
    print("\n11. Saving visualization...")
    output_plot = "examples/multi_aircraft_telemetry.png"
    plotter.save(output_plot)
    print(f"   Saved plot to: {output_plot}")

    if viz_3d:
        viz_3d.screenshot("examples/multi_aircraft_3d.png")
        viz_3d.close()

    # Demonstrate replay
    print("\n12. Demonstrating replay system...")
    print(f"   Loading log file: {log_file}")
    replay = MultiAircraftReplay(log_file)
    print(f"   Available aircraft: {replay.get_available_aircraft()}")
    replay.load_aircraft(aircraft_ids[:2])  # Load first 2 aircraft
    print(f"   {replay}")
    print(f"   Summary: {replay.get_summary()}")

    # Export one aircraft to CSV
    csv_file = f"examples/aircraft_{aircraft_ids[0]}.csv"
    replay.export_csv(aircraft_ids[0], csv_file)
    print(f"   Exported aircraft {aircraft_ids[0]} to: {csv_file}")

    # Show replay playback (first 5 seconds)
    print("\n13. Replay playback (first 5 seconds):")
    replay_plotter = MultiAircraftPlotter(aircraft_ids[:2], window_size=500)
    for t in replay.play(speed=10.0, end_time=5.0):  # 10x speed
        states = replay.get_states_at_time(t)
        for aircraft_id, state in states.items():
            replay_plotter.update(aircraft_id, state)
        if int(t * 10) % 10 == 0:  # Every 0.1s
            replay_plotter.plot()
            print(f"   Replay time: {t:.2f}s", end='\r')

    print("\n   Replay complete!")

    print("\n" + "=" * 70)
    print("Multi-Aircraft Demonstration Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Flight log: {log_file}")
    print(f"  - Telemetry plot: {output_plot}")
    print(f"  - CSV export: {csv_file}")
    if viz_3d:
        print(f"  - 3D screenshot: examples/multi_aircraft_3d.png")

    print(f"\nPhase 4: Visualization & Monitoring system working!")
    print(f"   - Multi-aircraft logging")
    print(f"   - Real-time visualization")
    print(f"   - Aircraft registry")
    print(f"   - Replay system")
    print(f"   - Data export")

    # Keep plots open
    print("\nClose plots to exit...")
    replay_plotter.show()


if __name__ == "__main__":
    main()
