#!/usr/bin/env python3
"""Systematic PID controller analysis with doublet inputs.

This script tests each PID controller individually with representative
doublet inputs to analyze their response characteristics:
- Step response
- Overshoot
- Settling time
- Steady-state error
- PID term contributions (P, I, D)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationAircraftBackend
from controllers import (
    ControlCommand, ControlMode, AircraftState,
    ControllerConfig, PIDGains, RateAgent, AttitudeAgent, HSAAgent
)
import yaml


def load_controller_config():
    """Load PID gains from default config file."""
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "default_gains.yaml"

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


def test_rate_controllers(config, duration=10.0, dt=0.001):
    """Test Level 4 rate controllers with doublet inputs.

    Args:
        config: Controller configuration
        duration: Test duration in seconds
        dt: Time step (1 kHz for rate loop)

    Returns:
        dict: Test results for each rate controller
    """
    print("\n" + "="*70)
    print("Testing Level 4: Rate Controllers (1000 Hz)")
    print("="*70)

    # Create simulation backend
    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})

    # Initial state: level flight at 100m
    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),
        velocity=np.array([12.0, 0.0, 0.0]),
        attitude=np.zeros(3),
        angular_rate=np.zeros(3),
        airspeed=12.0,
        altitude=100.0
    )

    # Create rate agent
    agent = RateAgent(config)

    results = {}

    # Test roll rate controller
    print("\n1. Roll Rate Controller (p → aileron)")
    print("   Doublet: 0 → 30 deg/s → 0 deg/s")
    backend.reset(initial_state)
    agent.reset()

    num_steps = int(duration / dt)
    times = np.zeros(num_steps)
    commands = np.zeros(num_steps)
    actuals = np.zeros(num_steps)
    outputs = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt
        times[i] = t

        # Doublet: step at 2s, return to zero at 6s
        if t < 2.0:
            roll_rate_cmd = 0.0
        elif t < 6.0:
            roll_rate_cmd = np.radians(30)  # 30 deg/s
        else:
            roll_rate_cmd = 0.0

        commands[i] = np.degrees(roll_rate_cmd)

        # Get state
        state = backend.get_state()
        state.time = t
        actuals[i] = np.degrees(state.p)

        # Compute control
        cmd = ControlCommand(
            mode=ControlMode.RATE,
            roll_rate=roll_rate_cmd,
            pitch_rate=0.0,
            yaw_rate=0.0,
            throttle=0.5
        )
        surfaces = agent.compute_action(cmd, state)
        outputs[i] = surfaces.aileron

        # Apply and step
        backend.set_controls(surfaces)
        backend.step(dt)

    results['roll_rate'] = {
        'times': times,
        'commands': commands,
        'actuals': actuals,
        'outputs': outputs,
        'name': 'Roll Rate (p → aileron)',
        'units': 'deg/s'
    }

    # Test pitch rate controller
    print("\n2. Pitch Rate Controller (q → elevator)")
    print("   Doublet: 0 → 30 deg/s → 0 deg/s")
    backend.reset(initial_state)
    agent.reset()

    commands = np.zeros(num_steps)
    actuals = np.zeros(num_steps)
    outputs = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt

        # Doublet
        if t < 2.0:
            pitch_rate_cmd = 0.0
        elif t < 6.0:
            pitch_rate_cmd = np.radians(30)
        else:
            pitch_rate_cmd = 0.0

        commands[i] = np.degrees(pitch_rate_cmd)

        state = backend.get_state()
        state.time = t
        actuals[i] = np.degrees(state.q)

        cmd = ControlCommand(
            mode=ControlMode.RATE,
            roll_rate=0.0,
            pitch_rate=pitch_rate_cmd,
            yaw_rate=0.0,
            throttle=0.5
        )
        surfaces = agent.compute_action(cmd, state)
        outputs[i] = surfaces.elevator

        backend.set_controls(surfaces)
        backend.step(dt)

    results['pitch_rate'] = {
        'times': times,
        'commands': commands,
        'actuals': actuals,
        'outputs': outputs,
        'name': 'Pitch Rate (q → elevator)',
        'units': 'deg/s'
    }

    # Test yaw rate controller
    print("\n3. Yaw Rate Controller (r → rudder)")
    print("   Doublet: 0 → 20 deg/s → 0 deg/s")
    backend.reset(initial_state)
    agent.reset()

    commands = np.zeros(num_steps)
    actuals = np.zeros(num_steps)
    outputs = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt

        # Doublet (smaller for yaw)
        if t < 2.0:
            yaw_rate_cmd = 0.0
        elif t < 6.0:
            yaw_rate_cmd = np.radians(20)
        else:
            yaw_rate_cmd = 0.0

        commands[i] = np.degrees(yaw_rate_cmd)

        state = backend.get_state()
        state.time = t
        actuals[i] = np.degrees(state.r)

        cmd = ControlCommand(
            mode=ControlMode.RATE,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=yaw_rate_cmd,
            throttle=0.5
        )
        surfaces = agent.compute_action(cmd, state)
        outputs[i] = surfaces.rudder

        backend.set_controls(surfaces)
        backend.step(dt)

    results['yaw_rate'] = {
        'times': times,
        'commands': commands,
        'actuals': actuals,
        'outputs': outputs,
        'name': 'Yaw Rate (r → rudder)',
        'units': 'deg/s'
    }

    return results


def test_attitude_controllers(config, duration=20.0, dt=0.01):
    """Test Level 3 attitude controllers with doublet inputs.

    Args:
        config: Controller configuration
        duration: Test duration in seconds
        dt: Time step (100 Hz for attitude loop)

    Returns:
        dict: Test results for each attitude controller
    """
    print("\n" + "="*70)
    print("Testing Level 3: Attitude Controllers (100 Hz)")
    print("="*70)

    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})

    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),
        velocity=np.array([12.0, 0.0, 0.0]),
        attitude=np.zeros(3),
        angular_rate=np.zeros(3),
        airspeed=12.0,
        altitude=100.0
    )

    agent = AttitudeAgent(config)

    results = {}

    # Test roll angle controller
    print("\n1. Roll Angle Controller (roll → p_cmd)")
    print("   Doublet: 0 → 20° → 0°")
    backend.reset(initial_state)
    agent.reset()

    num_steps = int(duration / dt)
    times = np.zeros(num_steps)
    commands = np.zeros(num_steps)
    actuals = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt
        times[i] = t

        # Doublet
        if t < 5.0:
            roll_cmd = 0.0
        elif t < 15.0:
            roll_cmd = np.radians(20)
        else:
            roll_cmd = 0.0

        commands[i] = np.degrees(roll_cmd)

        state = backend.get_state()
        state.time = t
        actuals[i] = np.degrees(state.roll)

        cmd = ControlCommand(
            mode=ControlMode.ATTITUDE,
            roll_angle=roll_cmd,
            pitch_angle=0.0,
            yaw_angle=0.0,
            throttle=0.5
        )
        surfaces = agent.compute_action(cmd, state)

        backend.set_controls(surfaces)
        backend.step(dt)

    results['roll_angle'] = {
        'times': times,
        'commands': commands,
        'actuals': actuals,
        'name': 'Roll Angle (roll → p)',
        'units': 'deg'
    }

    # Test pitch angle controller
    print("\n2. Pitch Angle Controller (pitch → q_cmd)")
    print("   Doublet: 0 → 15° → 0°")
    backend.reset(initial_state)
    agent.reset()

    commands = np.zeros(num_steps)
    actuals = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt

        # Doublet
        if t < 5.0:
            pitch_cmd = 0.0
        elif t < 15.0:
            pitch_cmd = np.radians(15)
        else:
            pitch_cmd = 0.0

        commands[i] = np.degrees(pitch_cmd)

        state = backend.get_state()
        state.time = t
        actuals[i] = np.degrees(state.pitch)

        cmd = ControlCommand(
            mode=ControlMode.ATTITUDE,
            roll_angle=0.0,
            pitch_angle=pitch_cmd,
            yaw_angle=0.0,
            throttle=0.5
        )
        surfaces = agent.compute_action(cmd, state)

        backend.set_controls(surfaces)
        backend.step(dt)

    results['pitch_angle'] = {
        'times': times,
        'commands': commands,
        'actuals': actuals,
        'name': 'Pitch Angle (pitch → q)',
        'units': 'deg'
    }

    return results


def test_hsa_controllers(config, duration=60.0, dt=0.01):
    """Test Level 2 HSA controllers with doublet inputs.

    Args:
        config: Controller configuration
        duration: Test duration in seconds
        dt: Time step (100 Hz)

    Returns:
        dict: Test results for each HSA controller
    """
    print("\n" + "="*70)
    print("Testing Level 2: HSA Controllers (100 Hz)")
    print("="*70)

    backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})

    agent = HSAAgent(config)

    results = {}

    # Test altitude controller
    print("\n1. Altitude Controller (altitude → pitch_cmd)")
    print("   Doublet: 100m → 150m → 100m")

    initial_state = AircraftState(
        time=0.0,
        position=np.array([0.0, 0.0, -100.0]),
        velocity=np.array([12.0, 0.0, 0.0]),
        attitude=np.zeros(3),
        angular_rate=np.zeros(3),
        airspeed=12.0,
        altitude=100.0
    )
    backend.reset(initial_state)
    agent.reset()

    num_steps = int(duration / dt)
    times = np.zeros(num_steps)
    alt_commands = np.zeros(num_steps)
    alt_actuals = np.zeros(num_steps)
    speed_actuals = np.zeros(num_steps)
    pitch_actuals = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt
        times[i] = t

        # Doublet
        if t < 10.0:
            alt_cmd = 100.0
        elif t < 40.0:
            alt_cmd = 150.0
        else:
            alt_cmd = 100.0

        alt_commands[i] = alt_cmd

        state = backend.get_state()
        state.time = t
        alt_actuals[i] = state.altitude
        speed_actuals[i] = state.airspeed
        pitch_actuals[i] = np.degrees(state.pitch)

        cmd = ControlCommand(
            mode=ControlMode.HSA,
            heading=0.0,
            speed=12.0,
            altitude=alt_cmd
        )
        surfaces = agent.compute_action(cmd, state)

        backend.set_controls(surfaces)
        backend.step(dt)

    results['altitude'] = {
        'times': times,
        'commands': alt_commands,
        'actuals': alt_actuals,
        'speed_actuals': speed_actuals,
        'pitch_actuals': pitch_actuals,
        'name': 'Altitude (altitude → pitch)',
        'units': 'm'
    }

    # Test speed controller
    print("\n2. Speed Controller (speed → throttle)")
    print("   Doublet: 12 m/s → 15 m/s → 12 m/s")

    backend.reset(initial_state)
    agent.reset()

    speed_commands = np.zeros(num_steps)
    speed_actuals = np.zeros(num_steps)
    throttle_actuals = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt

        # Doublet
        if t < 10.0:
            speed_cmd = 12.0
        elif t < 40.0:
            speed_cmd = 15.0
        else:
            speed_cmd = 12.0

        speed_commands[i] = speed_cmd

        state = backend.get_state()
        state.time = t
        speed_actuals[i] = state.airspeed

        cmd = ControlCommand(
            mode=ControlMode.HSA,
            heading=0.0,
            speed=speed_cmd,
            altitude=100.0
        )
        surfaces = agent.compute_action(cmd, state)
        throttle_actuals[i] = surfaces.throttle

        backend.set_controls(surfaces)
        backend.step(dt)

    results['speed'] = {
        'times': times,
        'commands': speed_commands,
        'actuals': speed_actuals,
        'throttle': throttle_actuals,
        'name': 'Speed (speed → throttle)',
        'units': 'm/s'
    }

    # Test heading controller
    print("\n3. Heading Controller (heading → roll_cmd)")
    print("   Doublet: 0° → 45° → 0°")

    backend.reset(initial_state)
    agent.reset()

    heading_commands = np.zeros(num_steps)
    heading_actuals = np.zeros(num_steps)
    roll_actuals = np.zeros(num_steps)

    for i in range(num_steps):
        t = i * dt

        # Doublet
        if t < 10.0:
            heading_cmd = 0.0
        elif t < 40.0:
            heading_cmd = np.radians(45)
        else:
            heading_cmd = 0.0

        heading_commands[i] = np.degrees(heading_cmd)

        state = backend.get_state()
        state.time = t
        heading_actuals[i] = np.degrees(state.heading)
        roll_actuals[i] = np.degrees(state.roll)

        cmd = ControlCommand(
            mode=ControlMode.HSA,
            heading=heading_cmd,
            speed=12.0,
            altitude=100.0
        )
        surfaces = agent.compute_action(cmd, state)

        backend.set_controls(surfaces)
        backend.step(dt)

    results['heading'] = {
        'times': times,
        'commands': heading_commands,
        'actuals': heading_actuals,
        'roll_actuals': roll_actuals,
        'name': 'Heading (heading → roll)',
        'units': 'deg'
    }

    return results


def plot_results(rate_results, attitude_results, hsa_results, output_dir):
    """Generate comprehensive plots for all PID controllers.

    Args:
        rate_results: Rate controller test results
        attitude_results: Attitude controller test results
        hsa_results: HSA controller test results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70)

    # Plot rate controllers
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Level 4: Rate Controllers (1000 Hz)', fontsize=16, fontweight='bold')

    for idx, (key, data) in enumerate(rate_results.items()):
        # Command vs Actual
        ax = axes[idx, 0]
        ax.plot(data['times'], data['commands'], 'r--', label='Command', linewidth=2)
        ax.plot(data['times'], data['actuals'], 'b-', label='Actual', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{data["units"]}')
        ax.set_title(data['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Control Output
        ax = axes[idx, 1]
        ax.plot(data['times'], data['outputs'], 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surface Deflection')
        ax.set_title(f'{data["name"]} - Control Output')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'rate_controllers_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Plot attitude controllers
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Level 3: Attitude Controllers (100 Hz)', fontsize=16, fontweight='bold')

    for idx, (key, data) in enumerate(attitude_results.items()):
        ax = axes[idx, 0]
        ax.plot(data['times'], data['commands'], 'r--', label='Command', linewidth=2)
        ax.plot(data['times'], data['actuals'], 'b-', label='Actual', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{data["units"]}')
        ax.set_title(data['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error
        ax = axes[idx, 1]
        error = data['commands'] - data['actuals']
        ax.plot(data['times'], error, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error ({data["units"]})')
        ax.set_title(f'{data["name"]} - Tracking Error')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'attitude_controllers_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Plot HSA controllers (altitude with coupled speed)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Level 2: Altitude Controller with Phugoid Analysis', fontsize=16, fontweight='bold')

    data = hsa_results['altitude']

    # Altitude tracking
    ax = axes[0, 0]
    ax.plot(data['times'], data['commands'], 'r--', label='Command', linewidth=2)
    ax.plot(data['times'], data['actuals'], 'b-', label='Actual', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Altitude error
    ax = axes[0, 1]
    error = data['commands'] - data['actuals']
    ax.plot(data['times'], error, 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude Error (m)')
    ax.set_title('Altitude Tracking Error')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Coupled airspeed (phugoid indicator)
    ax = axes[1, 0]
    ax.plot(data['times'], data['speed_actuals'], 'g-', linewidth=1.5)
    ax.axhline(12.0, color='k', linestyle='--', alpha=0.5, label='Target (12 m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Airspeed (m/s)')
    ax.set_title('Airspeed (Phugoid Indicator)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pitch angle
    ax = axes[1, 1]
    ax.plot(data['times'], data['pitch_actuals'], 'm-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Pitch Angle')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'altitude_phugoid_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Plot speed and heading controllers
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Level 2: Speed and Heading Controllers', fontsize=16, fontweight='bold')

    # Speed response
    data = hsa_results['speed']
    ax = axes[0, 0]
    ax.plot(data['times'], data['commands'], 'r--', label='Command', linewidth=2)
    ax.plot(data['times'], data['actuals'], 'b-', label='Actual', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throttle output
    ax = axes[0, 1]
    ax.plot(data['times'], data['throttle'], 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throttle')
    ax.set_title('Throttle Output')
    ax.grid(True, alpha=0.3)

    # Heading response
    data = hsa_results['heading']
    ax = axes[1, 0]
    ax.plot(data['times'], data['commands'], 'r--', label='Command', linewidth=2)
    ax.plot(data['times'], data['actuals'], 'b-', label='Actual', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (deg)')
    ax.set_title('Heading Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Roll angle (heading control)
    ax = axes[1, 1]
    ax.plot(data['times'], data['roll_actuals'], 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Roll (deg)')
    ax.set_title('Roll Angle (Heading Control)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'speed_heading_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()


def main():
    """Run systematic PID controller analysis."""
    print("="*70)
    print("Systematic PID Controller Analysis")
    print("="*70)

    # Load configuration
    config = load_controller_config()

    # Test each level
    rate_results = test_rate_controllers(config)
    attitude_results = test_attitude_controllers(config)
    hsa_results = test_hsa_controllers(config)

    # Generate plots
    output_dir = Path(__file__).parent.parent / "logs" / "pid_analysis"
    plot_results(rate_results, attitude_results, hsa_results, output_dir)

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
