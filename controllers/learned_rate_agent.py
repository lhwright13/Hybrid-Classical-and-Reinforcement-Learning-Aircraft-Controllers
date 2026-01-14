"""Level 4: Learned Rate Control Agent - RL-based rate controller."""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from controllers.base_agent import BaseAgent
from controllers.types import (
    ControlMode, ControlCommand, AircraftState,
    ControlSurfaces, ControllerConfig
)

# Import RL models
try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RecurrentPPO = None
    PPO = None


class LearnedRateAgent(BaseAgent):
    """Level 4: Learned rate mode controller (inner loop).

    Commands angular rates (p, q, r), outputs surface deflections.
    Uses a trained LSTM policy network instead of PID control.

    This is an RL-based alternative to the PID RateAgent:
    - Level 3 (Attitude) commands desired rates
    - Level 4 (Rate) tracks those rates → outputs surfaces

    The agent maintains an observation buffer for the recurrent network
    and can fall back to PID control if needed.
    """

    def __init__(
        self,
        model_path: str,
        config: ControllerConfig,
        fallback_to_pid: bool = True,
        device: str = "auto",
    ):
        """Initialize learned rate agent.

        Args:
            model_path: Path to trained model (.zip file)
            config: Controller configuration (for PID fallback)
            fallback_to_pid: Use PID fallback on model failure
            device: Device for inference ("auto", "cpu", "cuda")
        """
        if not RL_AVAILABLE:
            raise ImportError(
                "Stable-Baselines3 not available. Install with: "
                "pip install stable-baselines3[extra] sb3-contrib"
            )

        self.config = config
        self.fallback_to_pid = fallback_to_pid
        self.model_path = model_path

        # Load trained model
        self.model = self._load_model(model_path, device)

        # LSTM hidden state (if using RecurrentPPO)
        self.lstm_state = None
        self.is_recurrent = isinstance(self.model, RecurrentPPO)

        # Previous action for observation
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.5])

        # Observation buffer (18-dim)
        self.obs = np.zeros(18, dtype=np.float32)

        # Rate limits (rad/s)
        self.max_roll_rate = np.radians(config.max_roll_rate)
        self.max_pitch_rate = np.radians(config.max_pitch_rate)
        self.max_yaw_rate = np.radians(config.max_yaw_rate)

        # PID fallback (lazy initialization)
        self._pid_fallback = None
        self.using_fallback = False

    def _load_model(self, model_path: str, device: str):
        """Load trained model.

        Args:
            model_path: Path to model
            device: Device for inference

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Try loading as RecurrentPPO first
        try:
            model = RecurrentPPO.load(model_path, device=device)
            print(f"Loaded RecurrentPPO model from {model_path}")
            return model
        except Exception as e:
            # Log for debugging but continue to try PPO
            logger.debug(f"RecurrentPPO load failed (trying PPO): {e}")

        # Try loading as PPO
        try:
            model = PPO.load(model_path, device=device)
            print(f"Loaded PPO model from {model_path}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    def get_control_level(self) -> ControlMode:
        """Return control level.

        Returns:
            ControlMode.RATE
        """
        return ControlMode.RATE

    def compute_action(
        self,
        command: ControlCommand,
        state: AircraftState,
        dt: float = None
    ) -> ControlSurfaces:
        """Compute surface deflections from rate commands.

        Args:
            command: Rate control command (p, q, r desired)
            state: Current aircraft state
            dt: Time step (unused by RL agent, included for API compatibility)

        Returns:
            ControlSurfaces: Control surface deflections

        Raises:
            AssertionError: If command mode is not RATE
        """
        assert command.mode == ControlMode.RATE, \
            f"Learned rate agent expects RATE mode, got {command.mode}"

        # Rate setpoint (clip to rate limits)
        p_cmd = np.clip(command.roll_rate, -self.max_roll_rate, self.max_roll_rate)
        q_cmd = np.clip(command.pitch_rate, -self.max_pitch_rate, self.max_pitch_rate)
        r_cmd = np.clip(command.yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Build observation vector
        # [p, q, r, p_cmd, q_cmd, r_cmd, p_err, q_err, r_err,
        #  airspeed, altitude, roll, pitch, yaw,
        #  prev_aileron, prev_elevator, prev_rudder, prev_throttle]
        p_err = p_cmd - state.p
        q_err = q_cmd - state.q
        r_err = r_cmd - state.r

        self.obs = np.array([
            state.p, state.q, state.r,
            p_cmd, q_cmd, r_cmd,
            p_err, q_err, r_err,
            state.airspeed, state.altitude,
            state.roll, state.pitch, state.yaw,
            self.prev_action[0], self.prev_action[1],
            self.prev_action[2], self.prev_action[3],
        ], dtype=np.float32)

        # Get action from model
        try:
            action = self._predict(self.obs)
            self.using_fallback = False
        except Exception as e:
            if self.fallback_to_pid:
                print(f"Model prediction failed, using PID fallback: {e}")
                action = self._pid_fallback_action(command, state)
                self.using_fallback = True
            else:
                raise

        # Store action for next step
        self.prev_action = action.copy()

        # Map to surfaces
        surfaces = ControlSurfaces(
            aileron=float(action[0]),
            elevator=float(action[1]),
            rudder=float(action[2]),
            throttle=float(action[3]),
        )

        # Saturate (should already be done by model, but double-check)
        surfaces.aileron = np.clip(surfaces.aileron, -1.0, 1.0)
        surfaces.elevator = np.clip(surfaces.elevator, -1.0, 1.0)
        surfaces.rudder = np.clip(surfaces.rudder, -1.0, 1.0)
        surfaces.throttle = np.clip(surfaces.throttle, 0.0, 1.0)

        return surfaces

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        """Get action from model.

        Args:
            obs: Observation vector

        Returns:
            Action array [aileron, elevator, rudder, throttle]
        """
        if self.is_recurrent:
            # RecurrentPPO: maintain LSTM state
            action, self.lstm_state = self.model.predict(
                obs,
                state=self.lstm_state,
                deterministic=True,
            )
        else:
            # Standard PPO
            action, _ = self.model.predict(obs, deterministic=True)

        return action

    def _pid_fallback_action(
        self,
        command: ControlCommand,
        state: AircraftState
    ) -> np.ndarray:
        """Compute action using PID fallback.

        Args:
            command: Rate command
            state: Aircraft state

        Returns:
            Action array
        """
        # Lazy initialization of PID fallback
        if self._pid_fallback is None:
            from controllers.rate_agent import RateAgent
            self._pid_fallback = RateAgent(self.config)

        # Get PID action
        surfaces = self._pid_fallback.compute_action(command, state)

        action = np.array([
            surfaces.aileron,
            surfaces.elevator,
            surfaces.rudder,
            surfaces.throttle,
        ])

        return action

    def reset(self):
        """Reset agent state."""
        # Reset LSTM state
        self.lstm_state = None

        # Reset previous action
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.5])

        # Reset observation
        self.obs = np.zeros(18, dtype=np.float32)

        # Reset PID fallback if used
        if self._pid_fallback is not None:
            self._pid_fallback.reset()

        self.using_fallback = False

    def __repr__(self) -> str:
        """String representation."""
        model_type = "RecurrentPPO" if self.is_recurrent else "PPO"
        return (f"LearnedRateAgent(model={model_type}, "
                f"path={self.model_path}, "
                f"fallback={self.fallback_to_pid})")
