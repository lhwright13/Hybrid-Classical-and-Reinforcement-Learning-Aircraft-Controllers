"""Background simulation worker for real-time flight control (PID only).

This module provides a PID-only simulation worker with hot-reload support
for PID gains configuration.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from controllers.types import ControlMode
from controllers import HSAAgent, AttitudeAgent, RateAgent, SurfaceAgent

from gui.base_simulation_worker import (
    BaseSimulationWorker, FlightCommand, SIMULATION_DT
)

# Re-export FlightCommand for backwards compatibility
__all__ = ['SimulationWorker', 'FlightCommand']

# Config reload interval (number of simulation steps)
CONFIG_CHECK_INTERVAL = 100  # 100 * 0.01s = 1 second


class SimulationWorker(BaseSimulationWorker):
    """Simulation worker with PID controllers and hot-reload support.

    Runs simulation at 100 Hz in separate thread.
    Supports hot-reloading of PID gains from config file.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize simulation worker.

        Args:
            config_path: Path to controller config YAML
        """
        super().__init__(config_path)
        self.last_config_mtime = 0.0
        self._config_check_counter = 0

    def _load_config(self, config_path):
        """Load config and track modification time for hot-reload."""
        config = super()._load_config(config_path)

        # Track file modification time
        if config_path is None:
            config_path = Path(__file__).parent.parent / "controllers" / "config" / "pid_gains.yaml"
        config_path = Path(config_path)

        if config_path.exists():
            self.last_config_mtime = config_path.stat().st_mtime

        return config

    def _create_agents(self) -> Dict[ControlMode, Any]:
        """Create PID agents for all control levels."""
        return {
            ControlMode.SURFACE: SurfaceAgent(),
            ControlMode.RATE: RateAgent(self.config),
            ControlMode.ATTITUDE: AttitudeAgent(self.config),
            ControlMode.HSA: HSAAgent(self.config),
        }

    def toggle_controller(self):
        """Toggle controller (PID-only version - shows info message)."""
        print("\nWarning: This simulation uses PID controllers only")
        print("   To toggle between RL and PID, use:")
        print("   python examples/launch_pygame_gui_with_learned_rate.py")
        print("   (Requires trained RL model)\n")

    def _pre_simulation_step(self):
        """Check for config file changes periodically."""
        self._config_check_counter += 1
        if self._config_check_counter >= CONFIG_CHECK_INTERVAL:
            self._check_and_reload_config()
            self._config_check_counter = 0

    def _check_and_reload_config(self):
        """Check if config file changed and reload if needed."""
        if self.config_path is None:
            config_path = Path(__file__).parent.parent / "controllers" / "config" / "pid_gains.yaml"
        else:
            config_path = Path(self.config_path)

        if config_path.exists():
            current_mtime = config_path.stat().st_mtime
            if current_mtime > self.last_config_mtime:
                print("\n Config file changed! Reloading PID gains...")
                self.config = self._load_config(self.config_path)
                self.agents = self._create_agents()
                print("PID gains updated! New settings active.\n")
