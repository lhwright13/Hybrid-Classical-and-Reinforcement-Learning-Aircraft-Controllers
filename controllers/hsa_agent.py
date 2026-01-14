"""Level 2: HSA Control Agent - Heading, Speed, Altitude control."""

import numpy as np
import logging
from controllers.base_agent import BaseAgent
from controllers.attitude_agent import AttitudeAgent
from controllers.utils import wrap_angle, validate_command
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)

# Import C++ PID controller
import aircraft_controls_bindings as acb

logger = logging.getLogger(__name__)


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

    def __init__(self, config: ControllerConfig):
        """Initialize HSA agent with PID controllers.

        Args:
            config: Controller configuration
        """
        self.config = config

        # Last commanded yaw for rate limiting
        self.last_yaw_cmd = None

        # HSA controllers (simple PIDs in Python - not performance critical)
        # Heading controller
        heading_config = acb.PIDConfig()
        heading_config.gains = acb.PIDGains(
            kp=config.heading_gains.kp,
            ki=config.heading_gains.ki,
            kd=config.heading_gains.kd
        )
        heading_config.integral_min = -config.heading_gains.i_limit
        heading_config.integral_max = config.heading_gains.i_limit
        # Use configured max bank angle for coordinated turns
        max_bank_rad = np.radians(config.max_bank_angle_hsa)
        heading_config.output_min = -max_bank_rad
        heading_config.output_max = max_bank_rad
        self.heading_pid = acb.PIDController(heading_config)

        # === TECS (Total Energy Control System) ===
        # Instead of separate altitude and speed PIDs, use energy-based control
        # Total energy PID (controls throttle to add/remove energy)
        energy_config = acb.PIDConfig()
        energy_config.gains = acb.PIDGains(
            kp=0.08,   # Total energy error → throttle (TUNED: 0.05→0.08, moderate increase)
            ki=0.02,   # Integral for steady-state (TUNED: 0.01→0.02, moderate increase)
            kd=0.02    # Derivative for damping (added for speed spike control)
        )
        energy_config.integral_min = -10.0
        energy_config.integral_max = 10.0
        energy_config.output_min = -0.5
        energy_config.output_max = 0.5
        self.energy_pid = acb.PIDController(energy_config)

        # Energy distribution PID (controls pitch to exchange kinetic ↔ potential energy)
        balance_config = acb.PIDConfig()
        balance_config.gains = acb.PIDGains(
            kp=0.10,   # Energy balance error → pitch (TUNED: 0.02→0.10, 5x for aggressive speed→altitude trading)
            ki=0.002,  # Weak integral
            kd=0.05    # Damping (TUNED: 0.02→0.05, 2.5x for better damping)
        )
        balance_config.integral_min = -10.0
        balance_config.integral_max = 10.0
        balance_config.output_min = -max_bank_rad
        balance_config.output_max = max_bank_rad
        self.balance_pid = acb.PIDController(balance_config)

        # Inner loop: Attitude controller (Level 3)
        self.attitude_agent = AttitudeAgent(config)

        # Store config values for use in compute_action
        self.max_bank_rad = max_bank_rad
        self.baseline_throttle = config.baseline_throttle

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
        # Compute heading error (wrap to [-π, π])
        heading_error = wrap_angle(command.heading - state.heading)

        # Heading PID → roll angle (coordinated turn)
        # For small bank angles: roll ≈ (V²/g) * yaw_rate
        # Simplified: use heading error to command roll
        roll_angle = self.heading_pid.compute(command.heading, state.heading, dt)

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

        # Turn compensation for throttle
        # During turns, reduce throttle to prevent speed buildup
        # Speed increases during turns due to reduced induced drag in coordinated flight
        # Reduce throttle proportionally to bank angle
        roll_angle_deg = np.abs(np.degrees(roll_angle))
        if roll_angle_deg > 1.0:  # Only apply during meaningful turns
            # Reduce throttle by 3.0% per degree of bank (30% reduction at 10° bank)
            # Balance between speed control and altitude stability
            throttle -= 0.030 * roll_angle_deg

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
            pitch_angle += 0.05 * (load_factor - 1.0)  # Gentle compensation

        # Limit pitch angle using configured max bank (symmetrical limit)
        pitch_angle = np.clip(pitch_angle, -self.max_bank_rad, self.max_bank_rad)

        # === Yaw Coordination ===
        # Command yaw to track current yaw (effectively disables active yaw control)
        # This allows yaw to naturally follow aircraft dynamics during turns
        # Heading control comes entirely from ROLL (banking), not yaw
        # The aircraft naturally coordinates itself through aerodynamics
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
