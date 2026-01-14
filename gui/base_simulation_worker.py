"""Base simulation worker for real-time flight control.

This module provides the common functionality for simulation workers,
processing commands from the GUI and providing state updates.
"""

import threading
import time
import numpy as np
from abc import ABC, abstractmethod
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from simulation import SimulationAircraftBackend
from interfaces.sensor import PerfectSensorInterface
from controllers.types import (
    AircraftState, ControlCommand, ControlSurfaces, ControlMode, ControllerConfig
)
from controllers import HSAAgent, AttitudeAgent, RateAgent, SurfaceAgent

# Constants
COMMAND_QUEUE_SIZE = 10
STATE_QUEUE_SIZE = 100
SIMULATION_DT = 0.01  # 100 Hz
INITIAL_ALTITUDE = 3000.0  # meters
INITIAL_AIRSPEED = 20.0  # m/s


@dataclass
class FlightCommand:
    """Command from GUI to simulation."""
    mode: ControlMode
    # Direct surface control (Level 5)
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.5
    # Rate control (Level 4)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    # Attitude control (Level 3)
    roll_angle: float = 0.0
    pitch_angle: float = 0.0
    yaw_angle: float = 0.0
    # HSA control (Level 2)
    heading: float = 0.0
    speed: float = 20.0
    altitude: float = 100.0
    # Waypoint control (Level 1)
    waypoint_north: float = 0.0
    waypoint_east: float = 0.0
    waypoint_altitude: float = 100.0


class BaseSimulationWorker(ABC):
    """Base class for simulation workers.

    Provides common functionality for running flight simulation in a
    background thread, processing commands, and sending state updates.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize simulation worker.

        Args:
            config_path: Path to controller config YAML
        """
        # Simulation components
        self.backend: Optional[SimulationAircraftBackend] = None
        self.sensor: Optional[PerfectSensorInterface] = None
        self.agents: Dict[ControlMode, Any] = {}

        # Threading
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.command_queue: Queue = Queue(maxsize=COMMAND_QUEUE_SIZE)
        self.state_queue: Queue = Queue(maxsize=STATE_QUEUE_SIZE)

        # Simulation parameters
        self.dt = SIMULATION_DT
        self.current_command = FlightCommand(mode=ControlMode.SURFACE)

        # Load controller config
        self.config_path = config_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> ControllerConfig:
        """Load controller configuration from pid_gains.yaml.

        Args:
            config_path: Path to config file, or None for default

        Returns:
            Loaded or default ControllerConfig
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "controllers" / "config" / "pid_gains.yaml"

        config_path = Path(config_path)

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

    def _create_initial_state(self) -> AircraftState:
        """Create initial aircraft state for simulation start/reset.

        Returns:
            Initial AircraftState at high altitude for testing
        """
        return AircraftState(
            altitude=INITIAL_ALTITUDE,
            airspeed=INITIAL_AIRSPEED,
            position=np.array([0.0, 0.0, -INITIAL_ALTITUDE]),
            velocity=np.array([INITIAL_AIRSPEED, 0.0, 0.0]),
            attitude=np.zeros(3),
            angular_rate=np.zeros(3)
        )

    @abstractmethod
    def _create_agents(self) -> Dict[ControlMode, Any]:
        """Create agents for all control levels.

        Subclasses override this to provide their specific agent configuration.

        Returns:
            Dictionary mapping ControlMode to agent instances
        """
        pass

    @abstractmethod
    def toggle_controller(self):
        """Toggle between controller types (if applicable)."""
        pass

    def start(self):
        """Start simulation worker thread."""
        if self.running:
            return

        # Initialize simulation
        self.backend = SimulationAircraftBackend({'aircraft_type': 'rc_plane'})
        self.sensor = PerfectSensorInterface()

        # Initialize at stable flight
        initial = self._create_initial_state()
        self.backend.reset(initial)
        self.sensor.update(self.backend.get_state())

        # Initialize agents (subclass-specific)
        self.agents = self._create_agents()

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
            initial = self._create_initial_state()
            self.backend.reset(initial)
            self.sensor.update(self.backend.get_state())
            self._on_reset()

    def _on_reset(self):
        """Hook for subclasses to perform additional reset actions."""
        pass

    def send_command(self, command: FlightCommand):
        """Send command to simulation.

        Args:
            command: Flight command from GUI
        """
        try:
            self.command_queue.put_nowait(command)
        except Full:
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

    def _get_rate_agent(self) -> Any:
        """Get the current rate agent to use.

        Subclasses can override to implement controller switching.

        Returns:
            Rate agent instance
        """
        return self.agents.get(ControlMode.RATE)

    def _pre_simulation_step(self):
        """Hook called before each simulation step.

        Subclasses can override for periodic checks (e.g., config reload).
        """
        pass

    def _simulation_loop(self):
        """Main simulation loop (runs in background thread)."""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_time

            # Maintain simulation rate
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
                continue

            last_time = current_time

            # Pre-step hook
            self._pre_simulation_step()

            # Process commands from queue
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self.current_command = cmd
            except Empty:
                pass

            # Get current state
            state = self.sensor.get_state()

            # Update rate agent if in rate mode (for controller switching)
            if self.current_command.mode == ControlMode.RATE:
                rate_agent = self._get_rate_agent()
                if rate_agent:
                    self.agents[ControlMode.RATE] = rate_agent

            # Compute control based on mode
            command_obj = self._command_to_control_command(self.current_command)
            agent = self.agents.get(self.current_command.mode)

            if agent:
                surfaces = agent.compute_action(command_obj, state, dt=self.dt)
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
            except Full:
                # Queue full, drop oldest
                try:
                    self.state_queue.get_nowait()
                    self.state_queue.put_nowait(state_dict)
                except (Empty, Full):
                    pass

    def _command_to_control_command(self, cmd: FlightCommand) -> ControlCommand:
        """Convert flight command to ControlCommand object.

        Args:
            cmd: Flight command from GUI

        Returns:
            ControlCommand for the controller
        """
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
            return ControlCommand(mode=ControlMode.SURFACE, throttle=0.5)

    def _state_to_dict(self, state: AircraftState, surfaces: ControlSurfaces) -> Dict[str, Any]:
        """Convert aircraft state to dictionary for GUI.

        Args:
            state: Current aircraft state
            surfaces: Current control surface positions

        Returns:
            Dictionary with state values for GUI display
        """
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
            'g_force': 1.0 + state.velocity[2] / 9.81,
            # Control mode
            'mode': self.current_command.mode.name,
        }

        # Add subclass-specific fields
        self._add_extra_state_fields(state_dict)

        # Add commanded values based on current mode
        self._add_commanded_values(state_dict)

        return state_dict

    def _add_extra_state_fields(self, state_dict: Dict[str, Any]):
        """Hook for subclasses to add extra fields to state dict.

        Args:
            state_dict: State dictionary to modify
        """
        pass

    def _add_commanded_values(self, state_dict: Dict[str, Any]):
        """Add commanded values to state dictionary based on mode.

        Args:
            state_dict: State dictionary to modify
        """
        if self.current_command.mode == ControlMode.ATTITUDE:
            state_dict['cmd_roll_angle'] = np.degrees(self.current_command.roll_angle)
            state_dict['cmd_pitch_angle'] = np.degrees(self.current_command.pitch_angle)
            state_dict['cmd_yaw_angle'] = np.degrees(self.current_command.yaw_angle)
            state_dict['cmd_roll_rate'] = None
            state_dict['cmd_pitch_rate'] = None
            state_dict['cmd_yaw_rate'] = None
        elif self.current_command.mode == ControlMode.RATE:
            state_dict['cmd_roll_rate'] = np.degrees(self.current_command.roll_rate)
            state_dict['cmd_pitch_rate'] = np.degrees(self.current_command.pitch_rate)
            state_dict['cmd_yaw_rate'] = np.degrees(self.current_command.yaw_rate)
            state_dict['cmd_roll_angle'] = None
            state_dict['cmd_pitch_angle'] = None
            state_dict['cmd_yaw_angle'] = None
        else:
            state_dict['cmd_roll_angle'] = None
            state_dict['cmd_pitch_angle'] = None
            state_dict['cmd_yaw_angle'] = None
            state_dict['cmd_roll_rate'] = None
            state_dict['cmd_pitch_rate'] = None
            state_dict['cmd_yaw_rate'] = None
