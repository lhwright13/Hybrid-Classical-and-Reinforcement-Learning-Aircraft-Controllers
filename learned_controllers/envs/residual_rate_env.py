"""Residual RL environment for rate control.

The RL agent learns corrections on top of a PID baseline controller.
This makes learning much easier since the PID handles most of the work.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from controllers.rate_agent import RateAgent
from controllers.types import ControlMode, ControlCommand, ControllerConfig
from learned_controllers.envs.rate_env import RateControlEnv


class ResidualRateControlEnv(gym.Env):
    """Residual RL environment where agent learns corrections to PID output.

    The agent's action is ADDED to the PID controller's output, allowing it
    to learn small corrections rather than the full control policy.

    Benefits:
    - Much easier to learn (PID provides strong baseline)
    - More stable training (bounded corrections)
    - Faster convergence
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        difficulty: str = "medium",
        episode_length: float = 10.0,
        dt: float = 0.02,
        command_type: str = "step",
        render_mode: Optional[str] = None,
        rng_seed: Optional[int] = None,
        residual_scale: float = 0.3,  # Max correction is 30% of full range
    ):
        """Initialize residual rate control environment.

        Args:
            difficulty: Difficulty level
            episode_length: Episode duration in seconds
            dt: Control timestep
            command_type: Type of rate command
            render_mode: Rendering mode
            rng_seed: Random seed
            residual_scale: Scale for residual actions (0-1)
        """
        super().__init__()

        self.residual_scale = residual_scale

        # Create base environment
        self.base_env = RateControlEnv(
            difficulty=difficulty,
            episode_length=episode_length,
            dt=dt,
            command_type=command_type,
            render_mode=render_mode,
            rng_seed=rng_seed
        )

        # Create PID controller
        self.pid_config = ControllerConfig()
        self.pid_agent = RateAgent(self.pid_config)

        # Same observation space as base env
        self.observation_space = self.base_env.observation_space

        # Residual action space: small corrections to PID output
        # Range is [-residual_scale, residual_scale] for surfaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Store last PID action for observation augmentation (optional)
        self.last_pid_action = np.zeros(4)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.pid_agent.reset()
        self.last_pid_action = np.zeros(4)

        # Add PID action to info
        info['pid_action'] = self.last_pid_action.copy()

        return obs, info

    def step(
        self,
        residual_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with PID + residual action.

        Args:
            residual_action: Correction to add to PID output [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current state
        state = self.base_env.sim.get_state()

        # Get rate command from base environment
        p_cmd, q_cmd, r_cmd = self.base_env.rate_command

        # Create command for PID
        command = ControlCommand(
            mode=ControlMode.RATE,
            roll_rate=p_cmd,
            pitch_rate=q_cmd,
            yaw_rate=r_cmd,
            throttle=0.6
        )

        # Get PID action
        surfaces = self.pid_agent.compute_action(command, state, dt=self.base_env.dt)
        pid_action = np.array([
            surfaces.aileron,
            surfaces.elevator,
            surfaces.rudder,
            surfaces.throttle
        ], dtype=np.float32)

        self.last_pid_action = pid_action.copy()

        # Scale residual and add to PID
        scaled_residual = residual_action * self.residual_scale
        combined_action = pid_action + scaled_residual

        # Clip to valid range
        combined_action[:3] = np.clip(combined_action[:3], -1.0, 1.0)
        combined_action[3] = np.clip(combined_action[3], 0.0, 1.0)

        # Step base environment
        obs, reward, terminated, truncated, info = self.base_env.step(combined_action)

        # Add extra info
        info['pid_action'] = pid_action
        info['residual_action'] = scaled_residual
        info['combined_action'] = combined_action

        # Bonus reward for small residuals (encourage minimal corrections)
        residual_magnitude = np.sum(residual_action[:3]**2)
        reward += 0.05 * (1.0 - residual_magnitude / 3.0)  # Small bonus for small corrections

        return obs, reward, terminated, truncated, info

    @property
    def sim(self):
        """Access to simulation backend."""
        return self.base_env.sim

    @property
    def rate_command(self):
        """Current rate command."""
        return self.base_env.rate_command

    def render(self):
        """Render environment."""
        return self.base_env.render()

    def close(self):
        """Clean up."""
        self.base_env.close()


# Register the environment
gym.register(
    id='ResidualRateControl-v0',
    entry_point='learned_controllers.envs.residual_rate_env:ResidualRateControlEnv',
)
