"""Configuration loading for flight controllers."""

import yaml
from pathlib import Path
from controllers.types import ControllerConfig, PIDGains


def load_config_from_yaml(config_path: str = None) -> ControllerConfig:
    """Load controller configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default.

    Returns:
        ControllerConfig with PID gains loaded from YAML
    """
    if config_path is None:
        # Default to pid_gains.yaml in this directory
        config_path = Path(__file__).parent / "pid_gains.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return ControllerConfig()

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Parse PID gains
    config = ControllerConfig(
        # Attitude (angle) gains
        roll_angle_gains=PIDGains(
            kp=data['attitude']['roll']['kp'],
            ki=data['attitude']['roll']['ki'],
            kd=data['attitude']['roll']['kd'],
            i_limit=data['attitude']['roll']['i_limit']
        ),
        pitch_angle_gains=PIDGains(
            kp=data['attitude']['pitch']['kp'],
            ki=data['attitude']['pitch']['ki'],
            kd=data['attitude']['pitch']['kd'],
            i_limit=data['attitude']['pitch']['i_limit']
        ),

        # Rate gains
        roll_rate_gains=PIDGains(
            kp=data['rate']['roll']['kp'],
            ki=data['rate']['roll']['ki'],
            kd=data['rate']['roll']['kd'],
            i_limit=data['rate']['roll']['i_limit']
        ),
        pitch_rate_gains=PIDGains(
            kp=data['rate']['pitch']['kp'],
            ki=data['rate']['pitch']['ki'],
            kd=data['rate']['pitch']['kd'],
            i_limit=data['rate']['pitch']['i_limit']
        ),

        # Yaw gains (used in both rate and attitude)
        yaw_gains=PIDGains(
            kp=data['attitude']['yaw']['kp'],
            ki=data['attitude']['yaw']['ki'],
            kd=data['attitude']['yaw']['kd'],
            i_limit=data['attitude']['yaw']['i_limit']
        ),

        # Limits
        max_roll=data['limits']['max_roll_deg'],
        max_pitch=data['limits']['max_pitch_deg'],
        max_roll_rate=data['limits']['max_roll_rate_deg'],
        max_pitch_rate=data['limits']['max_pitch_rate_deg'],
        max_yaw_rate=data['limits']['max_yaw_rate_deg']
    )

    return config


def print_current_gains(config: ControllerConfig):
    """Print current PID gains in readable format.

    Args:
        config: Controller configuration to print
    """
    print("\n" + "="*60)
    print("CURRENT PID GAINS")
    print("="*60)

    print("\nATTITUDE (Outer Loop):")
    print(f"  Roll:  kp={config.roll_angle_gains.kp:.3f}, "
          f"ki={config.roll_angle_gains.ki:.3f}, "
          f"kd={config.roll_angle_gains.kd:.5f}")
    print(f"  Pitch: kp={config.pitch_angle_gains.kp:.3f}, "
          f"ki={config.pitch_angle_gains.ki:.3f}, "
          f"kd={config.pitch_angle_gains.kd:.5f}")
    print(f"  Yaw:   kp={config.yaw_gains.kp:.3f}, "
          f"ki={config.yaw_gains.ki:.3f}, "
          f"kd={config.yaw_gains.kd:.5f}")

    print("\nRATE (Inner Loop):")
    print(f"  Roll:  kp={config.roll_rate_gains.kp:.3f}, "
          f"ki={config.roll_rate_gains.ki:.3f}, "
          f"kd={config.roll_rate_gains.kd:.5f}")
    print(f"  Pitch: kp={config.pitch_rate_gains.kp:.3f}, "
          f"ki={config.pitch_rate_gains.ki:.3f}, "
          f"kd={config.pitch_rate_gains.kd:.5f}")

    print("\nLIMITS:")
    print(f"  Max Roll:  {config.max_roll:.1f}°")
    print(f"  Max Pitch: {config.max_pitch:.1f}°")
    print(f"  Max Roll Rate:  {config.max_roll_rate:.1f}°/s")
    print(f"  Max Pitch Rate: {config.max_pitch_rate:.1f}°/s")
    print(f"  Max Yaw Rate:   {config.max_yaw_rate:.1f}°/s")
    print("="*60 + "\n")
