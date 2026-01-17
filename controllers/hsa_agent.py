"""Level 2: HSA Control Agent - Heading, Speed, Altitude control."""

import numpy as np
from controllers.base_agent import BaseAgent
from controllers.attitude_agent import AttitudeAgent
from controllers.utils import wrap_angle, validate_command
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)
from controllers.config_loader import FlightControlConfig, TECSConfig, HSAConfig

# Import C++ PID controller
import aircraft_controls_bindings as acb


class HSAAgent(BaseAgent):
    """Level 2: HSA (Heading/Speed/Altitude) controller.

    Commands HSA state variables, outputs attitude commands to Level 3.

    HSA Control:
    - Heading: Yaw angle to point aircraft in desired direction
    - Speed: Airspeed control via throttle
    - Altitude: Height above ground

    Architecture:
        HSA error → PID → attitude command → Level 3 → surfaces

    Use cases:
    - Formation flight (maintain relative HSA)
    - Loitering (circle at constant altitude)
    - Area coverage patterns
    - Terrain following
    - Traffic avoidance (altitude separation)
    """

    def __init__(self, config: ControllerConfig, flight_config: FlightControlConfig = None):
        """Initialize HSA agent with PID controllers.

        Args:
            config: Controller configuration (legacy)
            flight_config: Flight control configuration with TECS params
        """
        self.config = config

        # Use flight_config if provided, otherwise use defaults
        if flight_config is not None:
            hsa_cfg = flight_config.hsa
            tecs_cfg = hsa_cfg.tecs
        else:
            hsa_cfg = HSAConfig()
            tecs_cfg = TECSConfig()

        # Last commanded yaw for rate limiting
        self.last_yaw_cmd = None

        # HSA controllers (simple PIDs in Python - not performance critical)
        # Heading controller
        heading_config = acb.PIDConfig()
        heading_config.gains = acb.PIDGains(
            kp=hsa_cfg.heading_gains.kp,
            ki=hsa_cfg.heading_gains.ki,
            kd=hsa_cfg.heading_gains.kd
        )
        heading_config.integral_min = -config.heading_gains.i_limit
        heading_config.integral_max = config.heading_gains.i_limit
        # Use configured max bank angle for coordinated turns
        max_bank_rad = np.radians(hsa_cfg.max_bank_angle)
        heading_config.output_min = -max_bank_rad
        heading_config.output_max = max_bank_rad
        self.heading_pid = acb.PIDController(heading_config)

        # === TECS (Total Energy Control System) ===
        # Total energy PID (controls throttle to add/remove energy)
        energy_config = acb.PIDConfig()
        energy_config.gains = acb.PIDGains(
            kp=tecs_cfg.energy_gains.kp,
            ki=tecs_cfg.energy_gains.ki,
            kd=tecs_cfg.energy_gains.kd
        )
        energy_config.integral_min = -10.0
        energy_config.integral_max = 10.0
        energy_config.output_min = -0.5
        energy_config.output_max = 0.5
        self.energy_pid = acb.PIDController(energy_config)

        # Energy distribution PID (controls pitch to exchange kinetic and potential energy)
        balance_config = acb.PIDConfig()
        balance_config.gains = acb.PIDGains(
            kp=tecs_cfg.balance_gains.kp,
            ki=tecs_cfg.balance_gains.ki,
            kd=tecs_cfg.balance_gains.kd
        )
        balance_config.integral_min = -5.0
        balance_config.integral_max = 5.0
        # Limit pitch command from config
        max_pitch_rad = np.radians(tecs_cfg.max_pitch_command)
        balance_config.output_min = -max_pitch_rad
        balance_config.output_max = max_pitch_rad
        self.balance_pid = acb.PIDController(balance_config)

        # Inner loop: Attitude controller (Level 3)
        self.attitude_agent = AttitudeAgent(config)

        # Store config values for use in compute_action
        self.max_bank_rad = max_bank_rad
        self.baseline_throttle = tecs_cfg.baseline_throttle
        self.load_factor_gain = tecs_cfg.load_factor_gain
        self.max_pitch_command = tecs_cfg.max_pitch_command

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.HSA
        """
        return ControlMode.HSA

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = 0.01
    ) -> ControlSurfaces:
        """Compute surfaces from HSA commands.

        Control strategy:
        - Heading → roll angle (coordinated turn)
        - Altitude → pitch angle (climb/descend)
        - Speed → throttle

        Args:
            command: HSA control command
            state: Current aircraft state
            dt: Time step in seconds (default 0.01 for 100 Hz)

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not HSA
        """
        assert command.mode == ControlMode.HSA, \
            f"HSA agent expects HSA mode, got {command.mode}"

        # Validate required command fields
        validate_command(command, "HSA", ["heading", "altitude", "speed"])

        # === Heading Control ===
        # Compute heading error (wrap to [-pi, pi])
        heading_error = wrap_angle(command.heading - state.heading)

        # Heading PID -> roll angle (coordinated turn)
        # Use wrapped error by computing virtual setpoint = state + wrapped_error
        # This ensures the PID sees the correct (wrapped) error magnitude
        virtual_setpoint = state.heading + heading_error
        roll_angle = self.heading_pid.compute(virtual_setpoint, state.heading, dt)

        # Limit roll angle using configured max bank
        roll_angle = np.clip(roll_angle, -self.max_bank_rad, self.max_bank_rad)

        # === TECS (Total Energy Control System) ===
        # Treats altitude and speed as a coupled energy problem
        # Total energy: E = m*g*h + 0.5*m*V²
        # Energy distribution: Balance between potential and kinetic energy

        # Current state
        g = 9.81  # m/s²
        h = state.altitude  # m
        V = state.airspeed  # m/s

        # Commanded state
        h_cmd = command.altitude  # m
        V_cmd = command.speed  # m/s

        # Specific energy (energy per unit mass)
        E_specific = g * h + 0.5 * V**2  # J/kg
        E_specific_cmd = g * h_cmd + 0.5 * V_cmd**2  # J/kg
        E_specific_error = E_specific_cmd - E_specific  # J/kg

        # Energy distribution (balance between altitude and speed)
        # Positive when we prioritize altitude, negative when we prioritize speed
        E_balance = g * h - 0.5 * V**2  # J/kg
        E_balance_cmd = g * h_cmd - 0.5 * V_cmd**2  # J/kg
        E_balance_error = E_balance_cmd - E_balance  # J/kg

        # === Total Energy Control → Throttle ===
        # Throttle adds/removes energy from the system
        # Positive error (need more energy) → increase throttle
        # Negative error (too much energy) → decrease throttle
        throttle_adjustment = self.energy_pid.compute(E_specific_cmd, E_specific, dt)
        throttle = self.baseline_throttle + throttle_adjustment

        # Note: Turn compensation removed - TECS energy controller handles speed management
        # Previous turn compensation caused altitude loss during turns
        throttle = np.clip(throttle, 0.0, 1.0)

        # === Energy Distribution Control → Pitch ===
        # Pitch exchanges kinetic ↔ potential energy
        # Positive error (need more altitude relative to speed) → pitch up
        # Negative error (need more speed relative to altitude) → pitch down
        pitch_angle = self.balance_pid.compute(E_balance_cmd, E_balance, dt)

        # Load factor compensation during turns
        cos_roll = np.cos(roll_angle)
        if abs(cos_roll) > 0.01:
            load_factor = 1.0 / cos_roll
            pitch_angle += self.load_factor_gain * (load_factor - 1.0)

        # Limit pitch angle to prevent excessive pitch during turns
        # Use tighter limit (10 deg) than TECS output (15 deg) for stability
        max_pitch_cmd = np.radians(10.0)
        pitch_angle = np.clip(pitch_angle, -max_pitch_cmd, max_pitch_cmd)

        # === Turn Coordination ===
        # Track current yaw - let bank handle the turn
        yaw_angle = state.yaw

        # Create attitude command (Level 3)
        attitude_cmd = ControlCommand(
            mode=ControlMode.ATTITUDE,
            roll_angle=roll_angle,
            pitch_angle=pitch_angle,
            yaw_angle=yaw_angle,
            throttle=throttle
        )

        # Inner loop: attitude → surfaces (Level 3)
        # Pass dt to inner loop for consistent timing
        surfaces = self.attitude_agent.compute_action(attitude_cmd, state, dt)

        return surfaces

    def reset(self):
        """Reset PID controllers."""
        self.heading_pid.reset()
        self.energy_pid.reset()
        self.balance_pid.reset()
        self.attitude_agent.reset()
        self.last_yaw_cmd = None

    def __repr__(self) -> str:
        """String representation."""
        return (f"HSAAgent(TECS: heading_kp={self.heading_pid.get_gains().kp:.3f}, "
                f"energy_kp={self.energy_pid.get_gains().kp:.3f}, "
                f"balance_kp={self.balance_pid.get_gains().kp:.3f})")
