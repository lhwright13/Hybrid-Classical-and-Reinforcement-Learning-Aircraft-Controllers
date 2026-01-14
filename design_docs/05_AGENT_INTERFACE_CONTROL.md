# Agent Interface & Control - Integration Guide

## Overview

This document is a **step-by-step guide** for integrating ANY agent type into the system. Whether you're implementing a classical PID controller, an RL agent, a hybrid approach, or a hierarchical multi-level agent, this guide walks you through the complete integration process.

**Goal**: Enable a developer to integrate a new agent in **< 1 day**.

## Agent Integration Checklist

- [ ] Implement `BaseAgent` interface
- [ ] Define control level (1, 2, 3, or 4)
- [ ] Define observation space for chosen level
- [ ] Define action space for chosen level
- [ ] Implement observation preprocessing
- [ ] Implement action postprocessing
- [ ] Create agent configuration file
- [ ] Add agent to registry
- [ ] Write unit tests
- [ ] Create example usage script
- [ ] Document hyperparameters (if applicable)
- [ ] Test in simulation
- [ ] Benchmark performance
- [ ] Deploy to hardware (optional)

## Step-by-Step Integration Guide

### Step 1: Choose Control Level

Decide which control level your agent will command at:

| Level | Best For | Difficulty |
|-------|----------|----------|
| 1: Waypoint | Long-horizon navigation | Hardest to train |
| 2: HSA | Trajectory tracking, formation | Medium |
| 3: Stick | Agile maneuvers, acrobatics | Easier |
| 4: Surface | Novel aircraft, optimal control | Easiest to train, hardest to deploy safely |

**Decision Criteria**:
- **Sparse rewards, long horizon?** → Start at Level 4, work up to Level 1
- **Existing demonstrations available?** → Match level of demonstrations
- **Safety critical?** → Higher level (1 or 2) has more safety layers
- **Novel aircraft?** → Level 4 for maximum flexibility

### Step 2: Create Agent Class

Create a new file `agents/my_agent.py`:

```python
from interfaces.agent_interface import BaseAgent
from controllers.types import (
    ControlMode,
    AircraftState,
    ControlCommand,
    Waypoint,
)
import numpy as np


class MyAgent(BaseAgent):
    """My custom agent implementation.

    Args:
        config: Agent configuration dict
    """

    def __init__(self, config: dict):
        self.config = config
        self.control_level = ControlMode[config["control_level"]]

        # Initialize your policy/model here
        # For RL: self.policy = load_policy(config["model_path"])
        # For classical: self.pid_gains = config["pid_gains"]

        # Internal state
        self.prev_observation = None
        self.step_count = 0

    def get_control_level(self) -> ControlMode:
        """Return control level."""
        return self.control_level

    def reset(self, initial_state: AircraftState) -> None:
        """Reset agent."""
        self.prev_observation = None
        self.step_count = 0

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        """Compute action from observation.

        Args:
            observation: State observation (format depends on control level)

        Returns:
            ControlCommand with appropriate fields populated
        """
        # Your agent logic here
        # Example for Level 3 (Stick & Throttle):
        action_raw = self._compute_action(observation)

        # Create command
        command = ControlCommand(
            mode=self.control_level,
            roll_cmd=action_raw[0],
            pitch_cmd=action_raw[1],
            yaw_cmd=action_raw[2],
            throttle=action_raw[3],
        )

        self.prev_observation = observation
        self.step_count += 1

        return command

    def _compute_action(self, observation: np.ndarray) -> np.ndarray:
        """Internal action computation (override this)."""
        # Placeholder: your algorithm here
        return np.zeros(4)

    def save(self, path: str) -> None:
        """Save agent to disk."""
        # For RL: save model weights
        # For classical: save configuration
        pass

    def load(self, path: str) -> None:
        """Load agent from disk."""
        # For RL: load model weights
        # For classical: load configuration
        pass
```

### Step 3: Implement Observation Processing

Each control level has a standard observation space (see `02_ABSTRACTION_LAYERS.md`).

```python
def get_observation_space(self, level: ControlMode) -> dict:
    """Define observation space for this agent's level."""
    from design_docs.observation_spaces import OBSERVATION_SPACES
    return OBSERVATION_SPACES[level]


def preprocess_observation(
    self,
    state: AircraftState,
    context: dict
) -> np.ndarray:
    """Convert AircraftState to observation vector.

    Args:
        state: Current aircraft state
        context: Additional context (e.g., waypoint, target HSA)

    Returns:
        Observation vector matching observation space
    """
    if self.control_level == ControlMode.WAYPOINT:
        # Level 1: Waypoint navigation
        waypoint = context.get("waypoint")
        obs = np.array([
            waypoint.north - state.north,          # position_error_north
            waypoint.east - state.east,            # position_error_east
            waypoint.down - state.down,            # position_error_down
            np.linalg.norm([waypoint.north - state.north,
                           waypoint.east - state.east,
                           waypoint.down - state.down]),  # distance
            np.arctan2(waypoint.east - state.east,
                      waypoint.north - state.north),  # heading_to_waypoint
            state.heading,                         # current_heading
            state.airspeed,                        # current_airspeed
            state.altitude,                        # current_altitude
            state.velocity[0],                     # velocity_north
            state.velocity[1],                     # velocity_east
            state.velocity[2],                     # velocity_down
            # ... etc
        ])

    elif self.control_level == ControlMode.HSA:
        # Level 2: HSA control
        target_hsa = context.get("target_hsa")
        obs = np.array([
            state.heading,
            state.airspeed,
            state.altitude,
            target_hsa["heading"] - state.heading,  # heading_error
            target_hsa["speed"] - state.airspeed,   # speed_error
            target_hsa["altitude"] - state.altitude, # altitude_error
            state.roll,
            state.pitch,
            state.p,
            state.q,
            state.r,
            state.velocity[2],  # vertical_speed
        ])

    elif self.control_level == ControlMode.STICK_THROTTLE:
        # Level 3: Stick & throttle
        target_attitude = context.get("target_attitude", {})
        obs = np.array([
            state.roll,
            state.pitch,
            state.yaw,
            state.p,
            state.q,
            state.r,
            state.airspeed,
            state.velocity[2],  # vertical_speed
            target_attitude.get("roll", 0) - state.roll,  # roll_error
            target_attitude.get("pitch", 0) - state.pitch, # pitch_error
        ])

    elif self.control_level == ControlMode.SURFACE:
        # Level 4: Direct surface control
        obs = np.array([
            state.roll,
            state.pitch,
            state.yaw,
            state.p,
            state.q,
            state.r,
            state.velocity[0],  # u
            state.velocity[1],  # v
            state.velocity[2],  # w
            state.airspeed,
            # Add actuator positions if available
            # ...
        ])

    # Normalization (optional but recommended)
    if self.config.get("normalize_observations", True):
        obs = self._normalize_observation(obs)

    return obs
```

### Step 4: Implement Action Processing

Convert your agent's raw output to a `ControlCommand`.

```python
def postprocess_action(
    self,
    action_raw: np.ndarray,
    state: AircraftState
) -> ControlCommand:
    """Convert raw action to ControlCommand.

    Args:
        action_raw: Raw action from policy
        state: Current state (for context)

    Returns:
        ControlCommand with appropriate fields
    """
    # Clip actions to valid range
    if self.config.get("clip_actions", True):
        action_space = self.get_action_space(self.control_level)
        action = np.clip(action_raw, action_space["low"], action_space["high"])
    else:
        action = action_raw

    # Create command based on level
    command = ControlCommand(mode=self.control_level, timestamp=state.time)

    if self.control_level == ControlMode.WAYPOINT:
        command.waypoint = Waypoint(
            north=action[0],
            east=action[1],
            down=action[2],
            speed=action[3],
        )

    elif self.control_level == ControlMode.HSA:
        command.heading = action[0]
        command.speed = action[1]
        command.altitude = action[2]

    elif self.control_level == ControlMode.STICK_THROTTLE:
        command.roll_cmd = action[0]
        command.pitch_cmd = action[1]
        command.yaw_cmd = action[2]
        command.throttle = action[3]

    elif self.control_level == ControlMode.SURFACE:
        command.elevator = action[0]
        command.aileron = action[1]
        command.rudder = action[2]
        command.throttle = action[3]

    return command
```

### Step 5: Create Configuration File

Create `configs/agents/my_agent.yaml`:

```yaml
agent:
  type: "my_agent"
  control_level: "STICK_THROTTLE"  # or WAYPOINT, HSA, SURFACE

  # Observation/action processing
  normalize_observations: true
  clip_actions: true

  # Agent-specific parameters
  model_path: null  # For RL agents
  hyperparameters:
    learning_rate: 0.0003
    gamma: 0.99
    # ... etc

  # For classical controllers
  pid_gains:
    roll: {kp: 0.2, ki: 0.1, kd: 0.05}
    pitch: {kp: 0.2, ki: 0.1, kd: 0.05}
    yaw: {kp: 0.3, ki: 0.05, kd: 0.0}
```

### Step 6: Register Agent

Add your agent to the registry in `interfaces/agent_interface.py`:

```python
AGENT_REGISTRY = {
    "classical": ClassicalAgent,
    "rl_ppo": RLAgentPPO,
    "rl_sac": RLAgentSAC,
    "my_agent": MyAgent,  # <-- Add your agent here
    "hybrid": HybridAgent,
    "hierarchical": HierarchicalAgent,
}


def create_agent(config: dict) -> BaseAgent:
    """Factory function to create agents."""
    agent_type = config["agent"]["type"]

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(config["agent"])
```

### Step 7: Write Unit Tests

Create `tests/test_my_agent.py`:

```python
import pytest
import numpy as np
from agents.my_agent import MyAgent
from controllers.types import AircraftState, ControlMode


def test_agent_initialization():
    """Test agent initializes correctly."""
    config = {
        "control_level": "STICK_THROTTLE",
        "normalize_observations": True,
    }
    agent = MyAgent(config)
    assert agent.get_control_level() == ControlMode.STICK_THROTTLE


def test_agent_reset():
    """Test agent resets correctly."""
    config = {"control_level": "STICK_THROTTLE"}
    agent = MyAgent(config)

    initial_state = AircraftState()
    agent.reset(initial_state)

    assert agent.step_count == 0
    assert agent.prev_observation is None


def test_agent_get_action():
    """Test agent produces valid actions."""
    config = {"control_level": "STICK_THROTTLE"}
    agent = MyAgent(config)

    # Create observation
    observation = np.random.randn(10)

    # Get action
    command = agent.get_action(observation)

    # Validate command
    assert command.mode == ControlMode.STICK_THROTTLE
    assert command.roll_cmd is not None
    assert -1.0 <= command.roll_cmd <= 1.0
    assert 0.0 <= command.throttle <= 1.0


def test_observation_space():
    """Test observation space is correctly defined."""
    config = {"control_level": "STICK_THROTTLE"}
    agent = MyAgent(config)

    obs_space = agent.get_observation_space(ControlMode.STICK_THROTTLE)

    assert "shape" in obs_space
    assert "low" in obs_space
    assert "high" in obs_space


def test_action_clipping():
    """Test actions are clipped to valid range."""
    config = {"control_level": "STICK_THROTTLE", "clip_actions": True}
    agent = MyAgent(config)

    # Force out-of-bounds action
    action_raw = np.array([10.0, -10.0, 5.0, 2.0])  # Out of bounds
    state = AircraftState()

    command = agent.postprocess_action(action_raw, state)

    # Should be clipped
    assert -1.0 <= command.roll_cmd <= 1.0
    assert -1.0 <= command.pitch_cmd <= 1.0
    assert 0.0 <= command.throttle <= 1.0
```

### Step 8: Create Example Usage

Create `examples/my_agent_demo.py`:

```python
#!/usr/bin/env python3
"""Demo of MyAgent controlling aircraft in simulation."""

import yaml
from interfaces.agent_interface import create_agent
from interfaces.aircraft_interface import create_aircraft_backend
from visualization.realtime_plotter import RealtimePlotter


def main():
    # Load configs
    with open("configs/agents/my_agent.yaml") as f:
        agent_config = yaml.safe_load(f)

    with open("configs/backends/simulation.yaml") as f:
        backend_config = yaml.safe_load(f)

    # Create agent and backend
    agent = create_agent(agent_config)
    backend = create_aircraft_backend(backend_config)

    # Optional: visualization
    plotter = RealtimePlotter(enabled=True)

    # Control loop
    state = backend.reset()
    agent.reset(state)

    for step in range(1000):
        # Get observation
        observation = agent.preprocess_observation(state, context={})

        # Get action
        command = agent.get_action(observation)

        # Execute action
        backend.set_controls(command)
        state = backend.step(dt=0.01)

        # Visualize
        plotter.update(state, command)

        # Check termination
        if state.altitude < 0:
            print("Crashed!")
            break

    print(f"Completed {step} steps")
    backend.close()


if __name__ == "__main__":
    main()
```

## Agent Type Examples

### Example 1: Classical PID Agent (Level 3)

```python
class ClassicalPIDAgent(BaseAgent):
    """Classical PID controller for attitude control."""

    def __init__(self, config: dict):
        self.control_level = ControlMode.STICK_THROTTLE
        self.gains = config["pid_gains"]

        # PID state
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        # Extract state from observation
        roll, pitch, yaw, p, q, r, _, _, roll_error, pitch_error = observation

        # PID control
        self.integral_roll += roll_error * 0.01
        self.integral_pitch += pitch_error * 0.01

        d_roll = (roll_error - self.prev_roll_error) / 0.01
        d_pitch = (pitch_error - self.prev_pitch_error) / 0.01

        roll_cmd = (
            self.gains["roll"]["kp"] * roll_error +
            self.gains["roll"]["ki"] * self.integral_roll +
            self.gains["roll"]["kd"] * d_roll
        )

        pitch_cmd = (
            self.gains["pitch"]["kp"] * pitch_error +
            self.gains["pitch"]["ki"] * self.integral_pitch +
            self.gains["pitch"]["kd"] * d_pitch
        )

        # Update state
        self.prev_roll_error = roll_error
        self.prev_pitch_error = pitch_error

        return ControlCommand(
            mode=self.control_level,
            roll_cmd=np.clip(roll_cmd, -1, 1),
            pitch_cmd=np.clip(pitch_cmd, -1, 1),
            yaw_cmd=0.0,
            throttle=0.5,
        )
```

### Example 2: RL Agent (Level 4, Stable-Baselines3)

```python
from stable_baselines3 import PPO


class RLAgentPPO(BaseAgent):
    """RL agent using PPO from Stable-Baselines3."""

    def __init__(self, config: dict):
        self.control_level = ControlMode.SURFACE
        self.config = config

        # Load trained model
        if config.get("model_path"):
            self.policy = PPO.load(config["model_path"])
        else:
            # Create new model for training
            self.policy = PPO("MlpPolicy", env=None, **config["hyperparameters"])

        self.training_mode = config.get("training_mode", False)

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        # Get action from policy
        action, _ = self.policy.predict(observation, deterministic=not self.training_mode)

        # Convert to command
        return ControlCommand(
            mode=self.control_level,
            elevator=float(action[0]),
            aileron=float(action[1]),
            rudder=float(action[2]),
            throttle=float(action[3]),
        )

    def update(self, transition):
        """Update policy (for online learning)."""
        if self.training_mode:
            # In training mode, transitions are collected by SB3 internally
            pass

    def save(self, path: str):
        self.policy.save(path)

    def load(self, path: str):
        self.policy = PPO.load(path)
```

### Example 3: Hierarchical Agent (Levels 1 + 3)

```python
class HierarchicalAgent(BaseAgent):
    """Hierarchical agent: Level 1 planner + Level 3 executor."""

    def __init__(self, config: dict):
        # High-level: Waypoint planner (runs at 1 Hz)
        self.high_level = WaypointPlannerAgent(config["high_level"])

        # Low-level: Stick controller (runs at 50 Hz)
        self.low_level = StickControllerAgent(config["low_level"])

        self.control_level = self.low_level.get_control_level()

        # Timing
        self.high_level_period = 1.0  # seconds
        self.time_since_high_level = 0.0
        self.current_waypoint = None

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        # Update high-level periodically
        if self.time_since_high_level >= self.high_level_period:
            high_level_obs = self._extract_high_level_obs(observation)
            waypoint_cmd = self.high_level.get_action(high_level_obs)
            self.current_waypoint = waypoint_cmd.waypoint
            self.time_since_high_level = 0.0

        # Low-level executes every step
        context = {"waypoint": self.current_waypoint}
        low_level_obs = self._extract_low_level_obs(observation, context)
        stick_cmd = self.low_level.get_action(low_level_obs)

        self.time_since_high_level += 0.02  # Assuming 50 Hz

        return stick_cmd
```

### Example 4: Adaptive Level-Switching Agent

```python
class AdaptiveLevelAgent(BaseAgent):
    """Agent that switches control levels based on situation."""

    def __init__(self, config: dict):
        # Meta-policy that selects level
        self.meta_policy = MetaLevelSelector(config["meta"])

        # Sub-policies for each level
        self.level_policies = {
            ControlMode.WAYPOINT: WaypointAgent(config["waypoint"]),
            ControlMode.HSA: HSAAgent(config["hsa"]),
            ControlMode.STICK_THROTTLE: StickAgent(config["stick"]),
            ControlMode.SURFACE: SurfaceAgent(config["surface"]),
        }

        self.current_level = ControlMode.STICK_THROTTLE  # Default

    def get_control_level(self) -> ControlMode:
        return self.current_level

    def get_action(self, observation: np.ndarray) -> ControlCommand:
        # Meta-policy decides which level to use
        situation_features = self._extract_situation_features(observation)
        selected_level = self.meta_policy.select_level(situation_features)

        # Switch level if needed
        if selected_level != self.current_level:
            print(f"Switching from {self.current_level} to {selected_level}")
            self.current_level = selected_level

        # Execute policy for selected level
        policy = self.level_policies[selected_level]
        return policy.get_action(observation)
```

## Performance Benchmarking

Always benchmark your agent against baselines.

```python
def benchmark_agent(agent, backend, num_episodes=100):
    """Benchmark agent performance."""
    metrics = {
        "success_rate": [],
        "average_reward": [],
        "average_time": [],
        "tracking_error": [],
    }

    for episode in range(num_episodes):
        state = backend.reset()
        agent.reset(state)

        episode_reward = 0.0
        episode_time = 0.0
        tracking_errors = []

        for step in range(1000):
            obs = agent.preprocess_observation(state, context={})
            command = agent.get_action(obs)
            state = backend.step(dt=0.01)

            reward = compute_reward(state, command)  # Your reward function
            episode_reward += reward
            episode_time += 0.01

            tracking_errors.append(compute_tracking_error(state, target))

            if is_done(state):
                break

        metrics["success_rate"].append(is_success(state))
        metrics["average_reward"].append(episode_reward)
        metrics["average_time"].append(episode_time)
        metrics["tracking_error"].append(np.mean(tracking_errors))

    return {
        "success_rate": np.mean(metrics["success_rate"]),
        "average_reward": np.mean(metrics["average_reward"]),
        "average_time": np.mean(metrics["average_time"]),
        "tracking_error_rms": np.sqrt(np.mean(np.array(metrics["tracking_error"])**2)),
    }
```

## Common Pitfalls & Solutions

### Pitfall 1: Observation Space Mismatch

**Problem**: Observation dimension doesn't match expected shape.

**Solution**: Always validate observation shape:
```python
assert observation.shape == self.get_observation_space(self.control_level)["shape"]
```

### Pitfall 2: Action Clipping Ignored

**Problem**: Actions outside valid range cause crashes.

**Solution**: Always clip actions:
```python
action = np.clip(action_raw, action_space["low"], action_space["high"])
```

### Pitfall 3: Forgetting to Reset Internal State

**Problem**: Agent carries over state between episodes.

**Solution**: Reset all internal variables in `reset()`:
```python
def reset(self, initial_state):
    self.integral = 0.0
    self.prev_error = 0.0
    self.prev_observation = None
```

### Pitfall 4: Wrong Control Level

**Problem**: Agent commands at wrong level.

**Solution**: Explicitly set and check control level:
```python
assert command.mode == self.get_control_level()
```

---

**Document Status**: Complete
**Last Updated**: 2025-10-09
**Related Documents**:
- 02_ABSTRACTION_LAYERS.md (interfaces)
- 03_CONTROL_HIERARCHY.md (control levels)
- 06_RL_AGENT_TRAINING.md (training RL agents)
