"""Configuration loader for flight control system.

Loads YAML configuration files and converts them to dataclass structures.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

# Default config directory
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary with configuration data
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'PIDGains':
        return cls(kp=d.get('kp', 0.0), ki=d.get('ki', 0.0), kd=d.get('kd', 0.0))


@dataclass
class GuidanceConfig:
    """Waypoint navigation and guidance configuration."""
    # Acceptance criteria
    acceptance_radius: float = 40.0

    # Pure Pursuit
    lookahead_time: float = 1.2
    lookahead_min: float = 12.0
    lookahead_max: float = 30.0
    proximity_scale_distance: float = 2.0

    # LOS guidance
    los_max_bank: float = 20.0
    los_lead_angle: float = 30.0

    # Speed management
    turn_threshold_angle: float = 60.0
    turn_threshold_distance: float = 60.0
    max_speed_reduction: float = 0.15
    min_speed: float = 14.0

    # Proportional navigation
    pn_gain: float = 3.0


@dataclass
class TECSConfig:
    """TECS (Total Energy Control System) configuration."""
    energy_gains: PIDGains = field(default_factory=lambda: PIDGains(0.12, 0.03, 0.03))
    balance_gains: PIDGains = field(default_factory=lambda: PIDGains(0.06, 0.005, 0.03))
    max_pitch_command: float = 15.0
    baseline_throttle: float = 0.1
    load_factor_gain: float = 0.05


@dataclass
class HSAConfig:
    """HSA (Heading/Speed/Altitude) controller configuration."""
    heading_gains: PIDGains = field(default_factory=lambda: PIDGains(1.0, 0.05, 0.2))
    max_bank_angle: float = 25.0
    tecs: TECSConfig = field(default_factory=TECSConfig)


@dataclass
class FlightControlConfig:
    """Complete flight control system configuration."""
    # Timing
    outer_loop_dt: float = 0.01
    inner_loop_dt: float = 0.001

    # Rate control (Level 4)
    roll_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(1.3, 0.4, 0.012))
    pitch_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(0.6, 0.2, 0.008))
    yaw_rate_gains: PIDGains = field(default_factory=lambda: PIDGains(1.6, 0.15, 0.01))
    max_roll_rate: float = 200.0
    max_pitch_rate: float = 100.0
    max_yaw_rate: float = 60.0

    # Attitude control (Level 3)
    roll_angle_gains: PIDGains = field(default_factory=lambda: PIDGains(8.0, 2.0, 0.3))
    pitch_angle_gains: PIDGains = field(default_factory=lambda: PIDGains(6.0, 1.5, 0.2))
    max_roll: float = 30.0
    max_pitch: float = 20.0

    # HSA control (Level 2)
    hsa: HSAConfig = field(default_factory=HSAConfig)

    # Guidance (Level 1)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)


def load_controller_config(
    config_file: str = "cascaded_pid.yaml"
) -> FlightControlConfig:
    """Load controller configuration from YAML file.

    Args:
        config_file: Name of config file in configs/controllers/

    Returns:
        FlightControlConfig with all settings
    """
    filepath = CONFIG_DIR / "controllers" / config_file
    if not filepath.exists():
        print(f"Warning: Config file {filepath} not found, using defaults")
        return FlightControlConfig()

    data = load_yaml(filepath)
    config = FlightControlConfig()

    # Timing
    if 'timing' in data:
        config.outer_loop_dt = data['timing'].get('outer_loop_dt', 0.01)
        config.inner_loop_dt = data['timing'].get('inner_loop_dt', 0.001)

    # Rate control
    if 'rate_control' in data:
        rc = data['rate_control']
        if 'roll' in rc:
            config.roll_rate_gains = PIDGains.from_dict(rc['roll'])
        if 'pitch' in rc:
            config.pitch_rate_gains = PIDGains.from_dict(rc['pitch'])
        if 'yaw' in rc:
            config.yaw_rate_gains = PIDGains.from_dict(rc['yaw'])
        if 'limits' in rc:
            config.max_roll_rate = rc['limits'].get('max_roll_rate', 200.0)
            config.max_pitch_rate = rc['limits'].get('max_pitch_rate', 100.0)
            config.max_yaw_rate = rc['limits'].get('max_yaw_rate', 60.0)

    # Attitude control
    if 'attitude_control' in data:
        ac = data['attitude_control']
        if 'roll' in ac:
            config.roll_angle_gains = PIDGains.from_dict(ac['roll'])
        if 'pitch' in ac:
            config.pitch_angle_gains = PIDGains.from_dict(ac['pitch'])
        if 'limits' in ac:
            config.max_roll = ac['limits'].get('max_roll', 30.0)
            config.max_pitch = ac['limits'].get('max_pitch', 20.0)

    # HSA control
    if 'hsa_control' in data:
        hc = data['hsa_control']
        if 'heading' in hc:
            config.hsa.heading_gains = PIDGains.from_dict(hc['heading'])
        if 'limits' in hc:
            config.hsa.max_bank_angle = hc['limits'].get('max_bank_angle', 25.0)
            config.hsa.tecs.baseline_throttle = hc['limits'].get('baseline_throttle', 0.1)
            config.hsa.tecs.max_pitch_command = hc['limits'].get('max_pitch_command', 15.0)
        if 'tecs' in hc:
            tecs = hc['tecs']
            if 'energy' in tecs:
                config.hsa.tecs.energy_gains = PIDGains.from_dict(tecs['energy'])
            if 'balance' in tecs:
                config.hsa.tecs.balance_gains = PIDGains.from_dict(tecs['balance'])
        if 'turn_compensation' in hc:
            tc = hc['turn_compensation']
            config.hsa.tecs.load_factor_gain = tc.get('load_factor_gain', 0.05)

    # Waypoint navigation
    if 'waypoint_navigation' in data:
        wn = data['waypoint_navigation']
        config.guidance.acceptance_radius = wn.get('acceptance_radius', 40.0)

        if 'pure_pursuit' in wn:
            pp = wn['pure_pursuit']
            config.guidance.lookahead_time = pp.get('lookahead_time', 1.2)
            config.guidance.lookahead_min = pp.get('lookahead_min', 12.0)
            config.guidance.lookahead_max = pp.get('lookahead_max', 30.0)
            config.guidance.proximity_scale_distance = pp.get('proximity_scale_distance', 2.0)

        if 'los' in wn:
            los = wn['los']
            config.guidance.los_max_bank = los.get('max_bank_for_turn_calc', 20.0)
            config.guidance.los_lead_angle = los.get('lead_angle', 30.0)

        if 'speed_control' in wn:
            sc = wn['speed_control']
            config.guidance.turn_threshold_angle = sc.get('turn_threshold_angle', 60.0)
            config.guidance.turn_threshold_distance = sc.get('turn_threshold_distance', 60.0)
            config.guidance.max_speed_reduction = sc.get('max_speed_reduction', 0.15)
            config.guidance.min_speed = sc.get('min_speed', 14.0)

        if 'proportional_nav' in wn:
            config.guidance.pn_gain = wn['proportional_nav'].get('gain', 3.0)

    return config


@dataclass
class MissionConfig:
    """Mission configuration."""
    name: str = "Unnamed Mission"
    pattern_type: str = "square"
    pattern_size: float = 300.0
    altitude: float = 100.0
    speed: float = 15.0
    guidance: str = "PP"
    max_duration: float = 180.0
    dt: float = 0.01
    output_dir: str = "final_figures"


def load_mission_config(config_file: str = "square_pattern.yaml") -> MissionConfig:
    """Load mission configuration from YAML file.

    Args:
        config_file: Name of config file in configs/missions/

    Returns:
        MissionConfig with mission settings
    """
    filepath = CONFIG_DIR / "missions" / config_file
    if not filepath.exists():
        print(f"Warning: Config file {filepath} not found, using defaults")
        return MissionConfig()

    data = load_yaml(filepath)
    config = MissionConfig()

    if 'mission' in data:
        m = data['mission']
        config.name = m.get('name', 'Unnamed Mission')
        if 'pattern' in m:
            config.pattern_type = m['pattern'].get('type', 'square')
            config.pattern_size = m['pattern'].get('size', 300.0)
        if 'flight' in m:
            config.altitude = m['flight'].get('altitude', 100.0)
            config.speed = m['flight'].get('speed', 15.0)
            config.guidance = m['flight'].get('guidance', 'PP')

    if 'simulation' in data:
        s = data['simulation']
        config.max_duration = s.get('max_duration', 180.0)
        config.dt = s.get('dt', 0.01)

    if 'output' in data:
        config.output_dir = data['output'].get('directory', 'final_figures')

    return config


# Convenience function to print config summary
def print_config_summary(config: FlightControlConfig):
    """Print a summary of the configuration."""
    print("=" * 60)
    print("Flight Control Configuration Summary")
    print("=" * 60)
    print(f"\nTiming:")
    print(f"  Outer loop: {config.outer_loop_dt * 1000:.1f} ms ({1/config.outer_loop_dt:.0f} Hz)")
    print(f"  Inner loop: {config.inner_loop_dt * 1000:.1f} ms ({1/config.inner_loop_dt:.0f} Hz)")

    print(f"\nRate Control (Level 4):")
    print(f"  Roll:  kp={config.roll_rate_gains.kp}, ki={config.roll_rate_gains.ki}, kd={config.roll_rate_gains.kd}")
    print(f"  Pitch: kp={config.pitch_rate_gains.kp}, ki={config.pitch_rate_gains.ki}, kd={config.pitch_rate_gains.kd}")
    print(f"  Yaw:   kp={config.yaw_rate_gains.kp}, ki={config.yaw_rate_gains.ki}, kd={config.yaw_rate_gains.kd}")

    print(f"\nAttitude Control (Level 3):")
    print(f"  Roll:  kp={config.roll_angle_gains.kp}, ki={config.roll_angle_gains.ki}, kd={config.roll_angle_gains.kd}")
    print(f"  Pitch: kp={config.pitch_angle_gains.kp}, ki={config.pitch_angle_gains.ki}, kd={config.pitch_angle_gains.kd}")
    print(f"  Limits: roll={config.max_roll} deg, pitch={config.max_pitch} deg")

    print(f"\nHSA Control (Level 2):")
    print(f"  Heading: kp={config.hsa.heading_gains.kp}, ki={config.hsa.heading_gains.ki}, kd={config.hsa.heading_gains.kd}")
    print(f"  Max bank: {config.hsa.max_bank_angle} deg")
    print(f"  TECS baseline throttle: {config.hsa.tecs.baseline_throttle}")

    print(f"\nGuidance (Level 1):")
    print(f"  Acceptance radius: {config.guidance.acceptance_radius} m")
    print(f"  Lookahead time: {config.guidance.lookahead_time} s")
    print(f"  Min speed: {config.guidance.min_speed} m/s")
    print("=" * 60)


if __name__ == "__main__":
    # Test loading configs
    print("Loading controller config...")
    ctrl_config = load_controller_config()
    print_config_summary(ctrl_config)

    print("\nLoading mission config...")
    mission_config = load_mission_config()
    print(f"Mission: {mission_config.name}")
    print(f"  Pattern: {mission_config.pattern_size}m {mission_config.pattern_type}")
    print(f"  Altitude: {mission_config.altitude}m, Speed: {mission_config.speed} m/s")
