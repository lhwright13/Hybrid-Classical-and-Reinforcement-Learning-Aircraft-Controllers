"""RL Agent interface for learning-based control agents.

This module defines the interface for RL/learning agents, which differs from
the classical controller interface in controllers/base_agent.py:

- RLAgentInterface: For RL agents that take observations and return commands
- controllers.base_agent.BaseAgent: For classical controllers that take commands
  and return surface deflections
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from controllers.types import (
    AircraftState,
    ControlMode,
    ControlCommand,
)


class RLAgentInterface(ABC):
    """Abstract base class for RL/learning agent types.

    This interface is designed for agents that:
    - Take raw observations (numpy arrays) as input
    - Return ControlCommands as output
    - May learn from experience (update method)
    - May need persistence (save/load methods)

    For classical controllers that take ControlCommands and return
    ControlSurfaces, use controllers.base_agent.BaseAgent instead.
    """

    @abstractmethod
    def get_control_level(self) -> ControlMode:
        """Return which control level this agent commands at.

        Returns:
            ControlMode: One of WAYPOINT, HSA, STICK_THROTTLE, SURFACE

        Example:
            >>> agent.get_control_level()
            ControlMode.STICK_THROTTLE
        """
        pass

    @abstractmethod
    def reset(self, initial_state: AircraftState) -> None:
        """Reset agent to initial state.

        Called at the beginning of each episode or when simulation resets.

        Args:
            initial_state: Initial aircraft state

        Example:
            >>> agent.reset(initial_state)
        """
        pass

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> ControlCommand:
        """Compute action given observation.

        Args:
            observation: State observation (format depends on control level)
                - Level 1 (Waypoint): 12D [pos(3), vel(3), att(3), wp_error(3)]
                - Level 2 (HSA): 12D [pos(3), vel(3), att(3), target_HSA(3)]
                - Level 3 (Stick): 10D [vel(3), att(3), rates(3), airspeed]
                - Level 4 (Surface): 14D [vel(3), att(3), rates(3), airspeed, aoa, sideslip, load]

        Returns:
            ControlCommand with fields populated for the agent's control level

        Example:
            >>> obs = np.array([...])  # 10D observation for Level 3
            >>> command = agent.get_action(obs)
            >>> print(command.mode)  # ControlMode.STICK_THROTTLE
            >>> print(command.roll_cmd)  # 0.15
        """
        pass

    def update(self, transition: Dict[str, Any]) -> None:
        """Update agent with transition (optional, for learning agents).

        This method is called after each step to provide feedback to the agent.
        Non-learning agents (e.g., classical controllers) can leave this empty.

        Args:
            transition: Dictionary with keys:
                - 'state': Previous state (np.ndarray)
                - 'action': Action taken (ControlCommand)
                - 'reward': Reward received (float)
                - 'next_state': Resulting state (np.ndarray)
                - 'done': Episode terminated (bool)
                - 'info': Additional information (dict)

        Example:
            >>> transition = {
            ...     'state': prev_obs,
            ...     'action': command,
            ...     'reward': -0.5,
            ...     'next_state': next_obs,
            ...     'done': False,
            ...     'info': {}
            ... }
            >>> agent.update(transition)
        """
        pass  # Optional: only learning agents need this

    def save(self, path: str) -> None:
        """Save agent state/policy to disk.

        Args:
            path: File path to save to

        Example:
            >>> agent.save("models/my_agent.pth")
        """
        pass  # Optional

    def load(self, path: str) -> None:
        """Load agent state/policy from disk.

        Args:
            path: File path to load from

        Example:
            >>> agent.load("models/my_agent.pth")
        """
        pass  # Optional

    # Optional methods for advanced agents

    def switch_control_level(self, level: ControlMode) -> None:
        """Switch control level (for adaptive agents).

        Only adaptive agents that can dynamically change control levels
        need to implement this.

        Args:
            level: New control level to operate at

        Raises:
            NotImplementedError: If agent does not support level switching

        Example:
            >>> agent.switch_control_level(ControlMode.HSA)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support level switching"
        )

    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space definition for this agent's control level.

        Returns:
            dict with 'shape', 'low', 'high' keys

        Example:
            >>> obs_space = agent.get_observation_space()
            >>> print(obs_space)
            {'shape': (10,), 'low': -np.inf, 'high': np.inf}
        """
        level = self.get_control_level()

        # Default observation spaces per level
        spaces = {
            ControlMode.WAYPOINT: {
                'shape': (12,),
                'low': -np.inf,
                'high': np.inf,
                'description': '[pos(3), vel(3), att(3), wp_error(3)]'
            },
            ControlMode.HSA: {
                'shape': (12,),
                'low': -np.inf,
                'high': np.inf,
                'description': '[pos(3), vel(3), att(3), target_HSA(3)]'
            },
            ControlMode.RATE: {
                'shape': (10,),
                'low': -np.inf,
                'high': np.inf,
                'description': '[vel(3), att(3), rates(3), airspeed]'
            },
            ControlMode.ATTITUDE: {
                'shape': (10,),
                'low': -np.inf,
                'high': np.inf,
                'description': '[vel(3), att(3), rates(3), airspeed]'
            },
            ControlMode.SURFACE: {
                'shape': (14,),
                'low': -np.inf,
                'high': np.inf,
                'description': '[vel(3), att(3), rates(3), airspeed, aoa, sideslip, load]'
            },
        }

        return spaces.get(level, {'shape': (0,), 'low': 0, 'high': 0})

    def get_action_space(self) -> Dict[str, Any]:
        """Get action space definition for this agent's control level.

        Returns:
            dict with 'shape', 'low', 'high' keys

        Example:
            >>> action_space = agent.get_action_space()
            >>> print(action_space)
            {'shape': (4,), 'low': [-1, -1, -1, 0], 'high': [1, 1, 1, 1]}
        """
        level = self.get_control_level()

        # Default action spaces per level
        spaces = {
            ControlMode.WAYPOINT: {
                'shape': (4,),
                'low': np.array([-np.inf, -np.inf, -np.inf, 0]),
                'high': np.array([np.inf, np.inf, np.inf, 100]),
                'description': '[N, E, D, speed]'
            },
            ControlMode.HSA: {
                'shape': (3,),
                'low': np.array([0, 0, 0]),
                'high': np.array([2*np.pi, 100, 1000]),
                'description': '[heading(rad), speed(m/s), altitude(m)]'
            },
            ControlMode.RATE: {
                'shape': (4,),
                'low': np.array([-1, -1, -1, 0]),
                'high': np.array([1, 1, 1, 1]),
                'description': '[roll_rate, pitch_rate, yaw_rate, throttle]'
            },
            ControlMode.ATTITUDE: {
                'shape': (4,),
                'low': np.array([-1, -1, -1, 0]),
                'high': np.array([1, 1, 1, 1]),
                'description': '[roll_angle, pitch_angle, yaw_angle, throttle]'
            },
            ControlMode.SURFACE: {
                'shape': (4,),
                'low': np.array([-1, -1, -1, 0]),
                'high': np.array([1, 1, 1, 1]),
                'description': '[elevator, aileron, rudder, throttle]'
            },
        }

        return spaces.get(level, {'shape': (0,), 'low': 0, 'high': 0})

    def preprocess_observation(self, state: AircraftState) -> np.ndarray:
        """Convert AircraftState to observation vector for this agent's level.

        This is a helper method that extracts the relevant features from
        the full aircraft state based on the agent's control level.

        Args:
            state: Complete aircraft state

        Returns:
            Observation vector matching get_observation_space()

        Example:
            >>> state = AircraftState(...)
            >>> obs = agent.preprocess_observation(state)
            >>> print(obs.shape)  # (10,) for Level 3
        """
        level = self.get_control_level()

        if level == ControlMode.WAYPOINT:
            # TODO: Needs target waypoint from task
            return np.concatenate([
                state.position,
                state.velocity,
                state.attitude,
                np.zeros(3),  # Waypoint error (needs target)
            ])

        elif level == ControlMode.HSA:
            # TODO: Needs target HSA from task
            return np.concatenate([
                state.position,
                state.velocity,
                state.attitude,
                np.zeros(3),  # Target HSA (needs target)
            ])

        elif level == ControlMode.RATE or level == ControlMode.ATTITUDE:
            return np.array([
                state.velocity[0],
                state.velocity[1],
                state.velocity[2],
                state.attitude[0],
                state.attitude[1],
                state.attitude[2],
                state.angular_rate[0],
                state.angular_rate[1],
                state.angular_rate[2],
                state.airspeed,
            ])

        elif level == ControlMode.SURFACE:
            # TODO: Calculate AoA, sideslip, load factor
            aoa = 0.0  # Placeholder
            sideslip = 0.0  # Placeholder
            load_factor = 1.0  # Placeholder

            return np.array([
                state.velocity[0],
                state.velocity[1],
                state.velocity[2],
                state.attitude[0],
                state.attitude[1],
                state.attitude[2],
                state.angular_rate[0],
                state.angular_rate[1],
                state.angular_rate[2],
                state.airspeed,
                aoa,
                sideslip,
                load_factor,
                state.altitude,
            ])

        else:
            raise ValueError(f"Unknown control level: {level}")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(level={self.get_control_level().name})"
