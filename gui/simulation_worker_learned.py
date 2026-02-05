"""Background simulation worker with Learned Rate Controller support.

This extends the base simulation worker to support both PID and
learned RL rate controllers, with the ability to toggle between them.
"""

from typing import Dict, Any, Optional

from controllers.types import ControlMode
from controllers import HSAAgent, AttitudeAgent, RateAgent, SurfaceAgent
from controllers.learned_rate_agent import LearnedRateAgent

from gui.base_simulation_worker import BaseSimulationWorker, FlightCommand

# Re-export FlightCommand for backwards compatibility
__all__ = ['SimulationWorkerWithLearned', 'FlightCommand']


class SimulationWorkerWithLearned(BaseSimulationWorker):
    """Simulation worker with learned RL rate controller support.

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
        self._model_path = model_path
        self._use_learned = use_learned
        self._rate_agent_pid: Optional[RateAgent] = None
        self._rate_agent_learned: Optional[LearnedRateAgent] = None

        super().__init__(config_path)

    @property
    def use_learned(self) -> bool:
        """Whether the learned controller is currently active."""
        return self._use_learned

    def _create_agents(self) -> Dict[ControlMode, Any]:
        """Create agents with learned rate controller support."""
        # Create PID rate agent
        self._rate_agent_pid = RateAgent(self._config)

        # Try to create learned rate agent
        try:
            self._rate_agent_learned = LearnedRateAgent(
                model_path=self._model_path,
                config=self._config,
                fallback_to_pid=True,
                device="auto"
            )
            print(f"Loaded learned rate controller from: {self._model_path}")
        except Exception as e:
            print(f"Failed to load learned controller: {e}")
            print("   Falling back to PID only")
            self._use_learned = False
            self._rate_agent_learned = None

        # Select initial rate agent based on toggle
        rate_agent = self._rate_agent_learned if self._use_learned else self._rate_agent_pid

        return {
            ControlMode.SURFACE: SurfaceAgent(),
            ControlMode.RATE: rate_agent,
            ControlMode.ATTITUDE: AttitudeAgent(self._config),
            ControlMode.HSA: HSAAgent(self._config),
        }

    def toggle_controller(self):
        """Toggle between learned and PID rate controllers."""
        self._use_learned = not self._use_learned
        controller_type = "Learned RL" if self._use_learned else "PID"
        print(f"\n Switched to {controller_type} rate controller")

        # Reset controller states
        if self._rate_agent_learned:
            self._rate_agent_learned.reset()
        if self._rate_agent_pid:
            self._rate_agent_pid.reset()

    def _get_rate_agent(self) -> Any:
        """Get the current rate agent based on toggle state."""
        if self._use_learned and self._rate_agent_learned:
            return self._rate_agent_learned
        return self._rate_agent_pid

    def _on_reset(self):
        """Reset controller states on simulation reset."""
        if self._rate_agent_learned:
            self._rate_agent_learned.reset()
        if self._rate_agent_pid:
            self._rate_agent_pid.reset()

    def _add_extra_state_fields(self, state_dict: Dict[str, Any]):
        """Add controller type to state dict."""
        state_dict['rate_controller'] = 'Learned RL' if self._use_learned else 'PID'
