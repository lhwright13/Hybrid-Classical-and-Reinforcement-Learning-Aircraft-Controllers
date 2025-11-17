"""Background simulation worker with Learned Rate Controller support.

This extends the standard simulation worker to support both PID and
learned RL rate controllers, with the ability to toggle between them.
"""

import threading
import time
import numpy as np
from queue import Queue, Empty
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from simulation import SimulationAircraftBackend
from interfaces.sensor import PerfectSensorInterface
from controllers.types import (
    AircraftState, ControlCommand, ControlSurfaces, ControlMode
)
from controllers import HSAAgent, AttitudeAgent, RateAgent, SurfaceAgent, ControllerConfig
from controllers.learned_rate_agent import LearnedRateAgent
from gui.simulation_worker import FlightCommand


class SimulationWorkerWithLearned:
    """Background worker that runs flight simulation with learned rate controller.

    Runs simulation at 100 Hz in separate thread.
    Supports toggling between learned RL and PID rate controllers.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        use_learned: bool = True
    ):
        """Initialize simulation worker with learned controller.

        Args:
            model_path: Path to trained RL model
            config_path: Path to controller config YAML
            use_learned: Start with learned controller (vs PID)
        """
        # Simulation components
        self.backend = None
        self.sensor = None
        self.agents = {}
        self.rate_agent_pid = None
        self.rate_agent_learned = None

        # Threading
        self.thread = None
        self.running = False
        self.command_queue = Queue(maxsize=10)
        self.state_queue = Queue(maxsize=100)

        # Simulation parameters
        self.dt = 0.01  # 100 Hz
        self.current_command = FlightCommand(mode=ControlMode.SURFACE)

        # Load controller config
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Learned controller settings
        self.model_path = model_path
        self.use_learned = use_learned

    def _load_config(self, config_path: Optional[str]) -> ControllerConfig:
        """Load controller configuration from pid_gains.yaml."""
        # Try to load from our tuning config file
        if config_path is None:
            config_path = Path(__file__).parent.parent / "controllers" / "config" / "pid_gains.yaml"

        config_path = Path(config_path)

        # If config file exists, load it
        if config_path.exists():
            try:
                from controllers.config import load_config_from_yaml
                config = load_config_from_yaml(str(config_path))
                print(f"Loaded PID config from: {config_path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                print("Using default configuration from types.py")
                return ControllerConfig()
        else:
            print(f"Warning: Config file not found at {config_path}")
            print("Using default configuration from types.py")
            return ControllerConfig()

    def toggle_controller(self):
        """Toggle between learned and PID rate controllers."""
        self.use_learned = not self.use_learned
        controller_type = "Learned RL" if self.use_learned else "PID"
        print(f"\n🔄 Switched to {controller_type} rate controller")

        # Reset the controller states
        if self.rate_agent_learned:
            self.rate_agent_learned.reset()
        if self.rate_agent_pid:
            self.rate_agent_pid.reset()

    def start(self):
        """Start simulation worker thread."""
        if self.running:
            return

        # Initialize simulation
        self.backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})
        self.sensor = PerfectSensorInterface()

        # Initialize at stable flight - high altitude for testing
        initial = AircraftState(
            altitude=3000.0,
            airspeed=20.0,
            position=np.array([0.0, 0.0, -3000.0]),
            velocity=np.array([20.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3)
        )
        self.backend.reset(initial)
        self.sensor.update(self.backend.get_state())

        # Initialize PID rate agent
        self.rate_agent_pid = RateAgent(self.config)

        # Initialize learned rate agent
        try:
            self.rate_agent_learned = LearnedRateAgent(
                model_path=self.model_path,
                config=self.config,
                fallback_to_pid=True,
                device="auto"
            )
            print(f"✅ Loaded learned rate controller from: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load learned controller: {e}")
            print("   Falling back to PID only")
            self.use_learned = False
            self.rate_agent_learned = None

        # Initialize agents for all control levels
        # Use learned or PID rate agent based on toggle
        self.agents = {
            ControlMode.SURFACE: SurfaceAgent(),
            ControlMode.RATE: self.rate_agent_learned if self.use_learned else self.rate_agent_pid,
            ControlMode.ATTITUDE: AttitudeAgent(self.config),
            ControlMode.HSA: HSAAgent(self.config),
        }

        # Start thread
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop simulation worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def reset(self):
        """Reset aircraft to initial state."""
        if self.backend:
            initial = AircraftState(
                altitude=3000.0,
                airspeed=20.0,
                position=np.array([0.0, 0.0, -3000.0]),
                velocity=np.array([20.0, 0.0, 0.0]),
                attitude=np.zeros(3),
                angular_rate=np.zeros(3)
            )
            self.backend.reset(initial)
            self.sensor.update(self.backend.get_state())

            # Reset controller states
            if self.rate_agent_learned:
                self.rate_agent_learned.reset()
            if self.rate_agent_pid:
                self.rate_agent_pid.reset()

    def send_command(self, command: FlightCommand):
        """Send command to simulation.

        Args:
            command: Flight command from GUI
        """
        try:
            self.command_queue.put_nowait(command)
        except:
            pass  # Queue full, drop command

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get latest state from simulation.

        Returns:
            State dictionary or None if no update available
        """
        try:
            return self.state_queue.get_nowait()
        except Empty:
            return None

    def _simulation_loop(self):
        """Main simulation loop (runs in background thread)."""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_time

            # Maintain 100 Hz rate
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
                continue

            last_time = current_time

            # Process commands from queue
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self.current_command = cmd
            except Empty:
                pass  # No new commands

            # Get current state
            state = self.sensor.get_state()

            # Update rate agent based on toggle (only in rate mode)
            if self.current_command.mode == ControlMode.RATE:
                if self.use_learned and self.rate_agent_learned:
                    self.agents[ControlMode.RATE] = self.rate_agent_learned
                else:
                    self.agents[ControlMode.RATE] = self.rate_agent_pid

            # Compute control based on mode
            command_obj = self._command_to_control_command(self.current_command)
            agent = self.agents.get(self.current_command.mode)

            if agent:
                surfaces = agent.compute_action(command_obj, state)
            else:
                # Fallback: direct surface control
                surfaces = ControlSurfaces(
                    elevator=self.current_command.elevator,
                    aileron=self.current_command.aileron,
                    rudder=self.current_command.rudder,
                    throttle=self.current_command.throttle
                )

            # Apply control and step simulation
            self.backend.set_controls(surfaces)
            true_state = self.backend.step(self.dt)
            self.sensor.update(true_state)
            state = self.sensor.get_state()

            # Check for crash (auto-reset)
            if state.altitude < 0:
                self.reset()
                state = self.sensor.get_state()

            # Send state update to GUI
            state_dict = self._state_to_dict(state, surfaces)
            try:
                self.state_queue.put_nowait(state_dict)
            except:
                # Queue full, drop oldest
                try:
                    self.state_queue.get_nowait()
                    self.state_queue.put_nowait(state_dict)
                except:
                    pass

    def _command_to_control_command(self, cmd: FlightCommand) -> ControlCommand:
        """Convert flight command to ControlCommand object."""
        if cmd.mode == ControlMode.SURFACE:
            return ControlCommand(
                mode=ControlMode.SURFACE,
                elevator=cmd.elevator,
                aileron=cmd.aileron,
                rudder=cmd.rudder,
                throttle=cmd.throttle
            )
        elif cmd.mode == ControlMode.RATE:
            return ControlCommand(
                mode=ControlMode.RATE,
                roll_rate=cmd.roll_rate,
                pitch_rate=cmd.pitch_rate,
                yaw_rate=cmd.yaw_rate,
                throttle=cmd.throttle
            )
        elif cmd.mode == ControlMode.ATTITUDE:
            return ControlCommand(
                mode=ControlMode.ATTITUDE,
                roll_angle=cmd.roll_angle,
                pitch_angle=cmd.pitch_angle,
                yaw_angle=cmd.yaw_angle,
                throttle=cmd.throttle
            )
        elif cmd.mode == ControlMode.HSA:
            return ControlCommand(
                mode=ControlMode.HSA,
                heading=cmd.heading,
                speed=cmd.speed,
                altitude=cmd.altitude,
                throttle=cmd.throttle
            )
        else:
            # Fallback
            return ControlCommand(mode=ControlMode.SURFACE, throttle=0.5)

    def _state_to_dict(self, state: AircraftState, surfaces: ControlSurfaces) -> Dict[str, Any]:
        """Convert aircraft state to dictionary for GUI."""
        state_dict = {
            # Raw state object (for external plotting/logging)
            'state_object': state,
            # Position
            'time': state.time,
            'north': state.position[0],
            'east': state.position[1],
            'altitude': state.altitude,
            # Velocity
            'airspeed': state.airspeed,
            'groundspeed': np.linalg.norm(state.velocity[:2]),
            'vertical_speed': -state.velocity[2],
            # Attitude (degrees)
            'roll': np.degrees(state.roll),
            'pitch': np.degrees(state.pitch),
            'yaw': np.degrees(state.yaw),
            'heading': np.degrees(state.yaw) % 360,
            # Rates (degrees/s)
            'roll_rate': np.degrees(state.p),
            'pitch_rate': np.degrees(state.q),
            'yaw_rate': np.degrees(state.r),
            # Control surfaces
            'elevator': surfaces.elevator,
            'aileron': surfaces.aileron,
            'rudder': surfaces.rudder,
            'throttle': surfaces.throttle,
            # Derived
            'g_force': 1.0 + state.velocity[2] / 9.81,  # Simplified
            # Control mode
            'mode': self.current_command.mode.name,
            # Controller type (for display)
            'rate_controller': 'Learned RL' if self.use_learned else 'PID',
        }

        # Add commanded values based on current mode
        if self.current_command.mode == ControlMode.ATTITUDE:
            # Commanded angles in degrees
            state_dict['cmd_roll_angle'] = np.degrees(self.current_command.roll_angle)
            state_dict['cmd_pitch_angle'] = np.degrees(self.current_command.pitch_angle)
            state_dict['cmd_yaw_angle'] = np.degrees(self.current_command.yaw_angle)
            state_dict['cmd_roll_rate'] = None
            state_dict['cmd_pitch_rate'] = None
            state_dict['cmd_yaw_rate'] = None
        elif self.current_command.mode == ControlMode.RATE:
            # Commanded rates in deg/s
            state_dict['cmd_roll_rate'] = np.degrees(self.current_command.roll_rate)
            state_dict['cmd_pitch_rate'] = np.degrees(self.current_command.pitch_rate)
            state_dict['cmd_yaw_rate'] = np.degrees(self.current_command.yaw_rate)
            state_dict['cmd_roll_angle'] = None
            state_dict['cmd_pitch_angle'] = None
            state_dict['cmd_yaw_angle'] = None
        else:
            # For other modes, set all to None
            state_dict['cmd_roll_angle'] = None
            state_dict['cmd_pitch_angle'] = None
            state_dict['cmd_yaw_angle'] = None
            state_dict['cmd_roll_rate'] = None
            state_dict['cmd_pitch_rate'] = None
            state_dict['cmd_yaw_rate'] = None

        return state_dict
