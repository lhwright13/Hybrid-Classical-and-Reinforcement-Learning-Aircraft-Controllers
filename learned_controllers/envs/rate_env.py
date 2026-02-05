"""Gymnasium environment for rate control training."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from controllers.types import AircraftState, ControlSurfaces
from simulation.simulation_backend import SimulationAircraftBackend
from learned_controllers.envs.rewards import RateTrackingReward, SettlingTimeBonus
from learned_controllers.data.generators import (
    RateCommandGenerator,
    FlightEnvelopeSampler
)


class RateControlEnv(gym.Env):
    """Gymnasium environment for aircraft rate control.

    The agent must track commanded angular rates (p, q, r) by outputting
    control surface commands to the aircraft simulator.

    Observation Space:
        - Current angular rates: [p, q, r]
        - Commanded rates: [p_cmd, q_cmd, r_cmd]
        - Rate errors: [p_err, q_err, r_err]
        - Flight state: [airspeed, altitude, roll, pitch, yaw]
        - Previous action: [aileron_prev, elevator_prev, rudder_prev, throttle_prev]
        Total: 18 dimensions

    Action Space:
        - Continuous: [aileron, elevator, rudder, throttle]
        - Range: aileron/elevator/rudder in [-1, 1], throttle in [0, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        difficulty: str = "medium",
        episode_length: float = 10.0,  # seconds
        dt: float = 0.02,  # 50 Hz control rate
        command_type: str = "step",   # step, ramp, sine, random
        render_mode: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ):
        """Initialize rate control environment.

        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            episode_length: Episode duration in seconds
            dt: Control timestep (seconds)
            command_type: Type of rate command ("step", "ramp", "sine", "random")
            render_mode: Rendering mode (not implemented yet)
            rng_seed: Random seed for reproducibility
        """
        super().__init__()

        self.difficulty = difficulty
        self.episode_length = episode_length
        self.dt = dt
        self.command_type = command_type
        self.render_mode = render_mode
        self.rng = np.random.RandomState(rng_seed)

        # Simulation backend
        self.sim = SimulationAircraftBackend({
            'aircraft_type': 'rc_plane',
            'dt_physics': 0.001,  # 1ms physics step
        })

        # Command and envelope generators
        self.cmd_generator = RateCommandGenerator(
            difficulty=difficulty,
            rng_seed=rng_seed
        )
        self.envelope_sampler = FlightEnvelopeSampler(rng_seed=rng_seed)

        # Reward functions
        self.reward_tracker = RateTrackingReward()
        self.settle_bonus = SettlingTimeBonus()

        # State tracking
        self.current_time = 0.0
        self.max_steps = int(episode_length / dt)
        self.step_count = 0

        # Command tracking
        self.rate_command = np.zeros(3)  # [p_cmd, q_cmd, r_cmd]
        self.prev_action = np.zeros(4)   # [aileron, elevator, rudder, throttle]

        # Command schedule for ramp/sine
        self.command_schedule = None
        self.sine_params = None

        # Pre-allocated buffers for hot-path (avoid per-step allocations)
        self._obs_buf = np.zeros(18, dtype=np.float32)
        self._rate_error_buf = np.zeros(3)
        self._max_rates = np.array([
            self.cmd_generator.max_roll_rate,
            self.cmd_generator.max_pitch_rate,
            self.cmd_generator.max_yaw_rate,
        ])
        self._cached_state = None

        # Define observation and action spaces
        # Observation: [p, q, r, p_cmd, q_cmd, r_cmd, p_err, q_err, r_err,
        #               airspeed, altitude, roll, pitch, yaw,
        #               prev_aileron, prev_elevator, prev_rudder, prev_throttle]
        obs_low = np.array([
            -10.0, -10.0, -10.0,  # Current rates (rad/s)
            -10.0, -10.0, -10.0,  # Commanded rates
            -20.0, -20.0, -20.0,  # Rate errors
            5.0, 0.0,              # Airspeed (m/s), altitude (m)
            -np.pi, -np.pi/2, -np.pi,  # Roll, pitch, yaw
            -1.0, -1.0, -1.0, 0.0,     # Previous action
        ])
        obs_high = np.array([
            10.0, 10.0, 10.0,
            10.0, 10.0, 10.0,
            20.0, 20.0, 20.0,
            50.0, 500.0,
            np.pi, np.pi/2, np.pi,
            1.0, 1.0, 1.0, 1.0,
        ])

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Action: [aileron, elevator, rudder, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Info dict for logging
        self.episode_rewards = {
            "tracking": 0.0,
            "smoothness": 0.0,
            "stability": 0.0,
            "oscillation": 0.0,
            "survival": 0.0,
            "settle_bonus": 0.0,
            "crash_penalty": 0.0,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Sample initial conditions
        init_cond = self.envelope_sampler.sample()

        # Create initial state
        initial_state = AircraftState(
            time=0.0,
            position=np.array([0.0, 0.0, -init_cond["altitude"]]),
            velocity=np.array([init_cond["airspeed"], 0.0, 0.0]),
            attitude=init_cond["attitude"],
            angular_rate=init_cond["angular_rate"],
            airspeed=init_cond["airspeed"],
            altitude=init_cond["altitude"],
        )

        # Reset simulation
        self.sim.reset(initial_state)

        # Generate rate command
        self._generate_new_command()

        # Reset state tracking
        self.current_time = 0.0
        self.step_count = 0
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.5])  # Neutral + mid throttle

        # Reset reward trackers
        self.reward_tracker.reset()
        self.settle_bonus.reset()
        self.episode_rewards = {k: 0.0 for k in self.episode_rewards}

        # Cache initial state for _get_observation and _get_info
        self._cached_state = self.sim.get_state()
        self._rate_error_buf[0] = self.rate_command[0] - self._cached_state.p
        self._rate_error_buf[1] = self.rate_command[1] - self._cached_state.q
        self._rate_error_buf[2] = self.rate_command[2] - self._cached_state.r

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Control action [aileron, elevator, rudder, throttle]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action to simulator
        surfaces = ControlSurfaces(
            aileron=float(action[0]),
            elevator=float(action[1]),
            rudder=float(action[2]),
            throttle=float(action[3]),
        )
        self.sim.set_controls(surfaces)

        # Step simulation
        state = self.sim.step(self.dt)
        self._cached_state = state

        # Update time
        self.current_time += self.dt
        self.step_count += 1

        # Update command if needed (for time-varying commands)
        self._update_command()

        # Get current rates and errors
        p, q, r = state.p, state.q, state.r
        p_cmd, q_cmd, r_cmd = self.rate_command
        p_err = p_cmd - p
        q_err = q_cmd - q
        r_err = r_cmd - r

        # Cache rate errors for _get_info
        self._rate_error_buf[0] = p_err
        self._rate_error_buf[1] = q_err
        self._rate_error_buf[2] = r_err

        # Compute reward
        reward, reward_components = self.reward_tracker.compute(
            p_err, q_err, r_err,
            action, self.prev_action,
            state.airspeed, state.altitude,
            state.roll, state.pitch
        )

        # Add settling bonus
        settle_reward = self.settle_bonus.compute(
            p_err, q_err, r_err,
            p_cmd, q_cmd, r_cmd,
            self.dt
        )
        reward += settle_reward

        # Track rewards
        for key, value in reward_components.items():
            if key in self.episode_rewards:
                self.episode_rewards[key] += value
        self.episode_rewards["settle_bonus"] += settle_reward

        # Update previous action
        self.prev_action = action.copy()

        # Check termination conditions
        terminated = self._check_termination(state)
        truncated = self.step_count >= self.max_steps

        # Apply crash penalty if terminated early
        if terminated and not truncated:
            # Heavy penalty for crashing (prevents exploit of crashing early)
            crash_penalty = -100.0
            reward += crash_penalty
            if "crash_penalty" not in reward_components:
                reward_components["crash_penalty"] = crash_penalty

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info(reward_components)

        return obs, reward, terminated, truncated, info

    def _generate_new_command(self):
        """Generate new rate command based on command type."""
        if self.command_type == "step":
            # Step command
            num_axes = self.rng.choice([1, 2, 3])
            self.rate_command, desc = self.cmd_generator.generate_step_command(
                num_axes=num_axes
            )
            self.command_schedule = None

        elif self.command_type == "ramp":
            # Ramp command
            start, end, desc = self.cmd_generator.generate_ramp_command()
            self.command_schedule = {
                "type": "ramp",
                "start": start,
                "end": end,
                "duration": 3.0,
                "start_time": 0.0,
            }
            self.rate_command = start

        elif self.command_type == "sine":
            # Sinusoidal command
            freq, amps, desc = self.cmd_generator.generate_sine_command()
            self.sine_params = {
                "frequency": freq,
                "amplitudes": amps,
            }
            self.rate_command = np.zeros(3)

        elif self.command_type == "random":
            # Random walk
            self.rate_command = np.zeros(3)
            self.command_schedule = {"type": "random_walk"}

        else:
            # Default to step
            self.rate_command = np.zeros(3)

    def _update_command(self):
        """Update rate command for time-varying commands."""
        if self.command_schedule is not None:
            if self.command_schedule["type"] == "ramp":
                # Linear interpolation
                t = self.current_time - self.command_schedule["start_time"]
                duration = self.command_schedule["duration"]
                if t < duration:
                    alpha = t / duration
                    self.rate_command = (
                        (1 - alpha) * self.command_schedule["start"] +
                        alpha * self.command_schedule["end"]
                    )
                else:
                    self.rate_command = self.command_schedule["end"]

            elif self.command_schedule["type"] == "random_walk":
                # Random walk update
                delta, _ = self.cmd_generator.generate_random_walk(dt=self.dt)
                self.rate_command += delta
                np.clip(
                    self.rate_command, -self._max_rates, self._max_rates,
                    out=self.rate_command,
                )

        elif self.sine_params is not None:
            # Sinusoidal command
            t = self.current_time
            freq = self.sine_params["frequency"]
            amps = self.sine_params["amplitudes"]
            self.rate_command = amps * np.sin(2 * np.pi * freq * t)

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.

        Returns:
            Observation array
        """
        state = self._cached_state if self._cached_state is not None else self.sim.get_state()

        # Fill pre-allocated buffer instead of creating a new array
        buf = self._obs_buf
        # Current rates
        buf[0] = state.p
        buf[1] = state.q
        buf[2] = state.r
        # Commanded rates
        buf[3] = self.rate_command[0]
        buf[4] = self.rate_command[1]
        buf[5] = self.rate_command[2]
        # Rate errors
        buf[6] = self.rate_command[0] - state.p
        buf[7] = self.rate_command[1] - state.q
        buf[8] = self.rate_command[2] - state.r
        # Flight state
        buf[9] = state.airspeed
        buf[10] = state.altitude
        buf[11] = state.roll
        buf[12] = state.pitch
        buf[13] = state.yaw
        # Previous action
        buf[14] = self.prev_action[0]
        buf[15] = self.prev_action[1]
        buf[16] = self.prev_action[2]
        buf[17] = self.prev_action[3]

        return buf

    def _get_info(self, reward_components: Optional[Dict] = None) -> Dict[str, Any]:
        """Get info dictionary.

        Args:
            reward_components: Reward breakdown from current step

        Returns:
            Info dictionary
        """
        state = self._cached_state if self._cached_state is not None else self.sim.get_state()

        info = {
            "time": self.current_time,
            "step": self.step_count,
            "position": state.position.copy(),
            "rate_command": self.rate_command.copy(),
            "rate_error": self._rate_error_buf.copy(),
            "airspeed": state.airspeed,
            "altitude": state.altitude,
            "is_settled": self.settle_bonus.is_settled,
        }

        if reward_components is not None:
            info["reward_components"] = reward_components

        return info

    def _check_termination(self, state: AircraftState) -> bool:
        """Check if episode should terminate early.

        Args:
            state: Current aircraft state

        Returns:
            True if episode should terminate
        """
        # Crash detection
        if state.altitude < 5.0:
            return True

        # Excessive attitude - relaxed to give more room to recover
        if abs(state.roll) > np.radians(120):  # Was 80, now 120 (allow inverted attempts)
            return True
        if abs(state.pitch) > np.radians(80):  # Was 60, now 80
            return True

        # Stall - relaxed slightly
        if state.airspeed < 8.0:  # Was 10, now 8
            return True

        return False

    def render(self):
        """Render environment (not implemented)."""
        if self.render_mode == "human":
            # TODO: Implement visualization
            pass

    def close(self):
        """Clean up environment resources."""
        pass
