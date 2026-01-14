# RL Agent Training - Multi-Level Training Infrastructure

## Overview

This document specifies the RL training infrastructure for training agents at all 4 control levels. The system supports multiple RL frameworks and provides level-specific training strategies.

## RL Framework Integration

### Supported Frameworks

1. **Stable-Baselines3** (Primary)
   - Well-tested, easy to use
   - PPO, SAC, TD3 algorithms
   - Great for single-machine training

2. **RLlib** (Distributed)
   - Ray-based distributed training
   - Scales to 100+ parallel environments
   - Advanced algorithms (APPO, IMPALA)

3. **CleanRL** (Research)
   - Single-file implementations
   - Easy to modify and experiment
   - Great for understanding algorithms

## Gymnasium Environment Wrapper

### Base Environment

**File**: `training/aircraft_env.py`

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from controllers.types import ControlMode, AircraftState
from interfaces.aircraft_interface import create_aircraft_backend


class AircraftEnv(gym.Env):
    """Gymnasium environment for aircraft control.
    
    Supports all 4 control levels via configuration.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.control_level = ControlMode[config["control_level"]]
        
        # Create simulation backend
        self.backend = create_aircraft_backend(config["backend"])
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode state
        self.current_step = 0
        self.max_steps = config.get("max_steps", 1000)
        
        # For curriculum learning
        self.curriculum_level = 0
    
    def _setup_spaces(self):
        """Define observation and action spaces based on control level."""
        from design_docs.observation_spaces import OBSERVATION_SPACES, ACTION_SPACES
        
        obs_spec = OBSERVATION_SPACES[self.control_level]
        act_spec = ACTION_SPACES[self.control_level]
        
        self.observation_space = spaces.Box(
            low=obs_spec["low"],
            high=obs_spec["high"],
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=act_spec["low"],
            high=act_spec["high"],
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random initial state (with curriculum)
        initial_state = self._get_initial_state()
        self.state = self.backend.reset(initial_state)
        
        self.current_step = 0
        self.prev_action = np.zeros(self.action_space.shape)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one timestep."""
        # Apply action through flight controller
        command = self._action_to_command(action)
        self.backend.set_controls(command)
        
        # Step simulation
        self.state = self.backend.step(dt=self.config.get("dt", 0.01))
        
        # Get observation, reward, termination
        observation = self._get_observation()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        self.current_step += 1
        self.prev_action = action
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Convert state to observation vector."""
        if self.control_level == ControlMode.WAYPOINT:
            # Level 1: Position errors, distance to waypoint
            waypoint = self.target_waypoint
            obs = np.array([
                waypoint.north - self.state.north,
                waypoint.east - self.state.east,
                waypoint.down - self.state.down,
                np.linalg.norm([waypoint.north - self.state.north,
                               waypoint.east - self.state.east,
                               waypoint.down - self.state.down]),
                np.arctan2(waypoint.east - self.state.east,
                          waypoint.north - self.state.north),
                self.state.heading,
                self.state.airspeed,
                self.state.altitude,
                self.state.velocity[0],
                self.state.velocity[1],
                self.state.velocity[2],
                self._wrap_angle(np.arctan2(waypoint.east - self.state.east,
                                waypoint.north - self.state.north) - self.state.heading)
            ], dtype=np.float32)
        
        elif self.control_level == ControlMode.HSA:
            # Level 2: Current HSA + errors
            obs = np.array([
                self.state.heading,
                self.state.airspeed,
                self.state.altitude,
                self._wrap_angle(self.target_hsa["heading"] - self.state.heading),
                self.target_hsa["speed"] - self.state.airspeed,
                self.target_hsa["altitude"] - self.state.altitude,
                self.state.roll,
                self.state.pitch,
                self.state.p,
                self.state.q,
                self.state.r,
                self.state.velocity[2],  # vertical speed
            ], dtype=np.float32)
        
        elif self.control_level == ControlMode.STICK_THROTTLE:
            # Level 3: Attitude and rates
            obs = np.array([
                self.state.roll,
                self.state.pitch,
                self.state.yaw,
                self.state.p,
                self.state.q,
                self.state.r,
                self.state.airspeed,
                self.state.velocity[2],
                self.target_attitude["roll"] - self.state.roll,
                self.target_attitude["pitch"] - self.state.pitch,
            ], dtype=np.float32)
        
        elif self.control_level == ControlMode.SURFACE:
            # Level 4: Full state
            obs = np.array([
                self.state.roll,
                self.state.pitch,
                self.state.yaw,
                self.state.p,
                self.state.q,
                self.state.r,
                self.state.velocity[0],
                self.state.velocity[1],
                self.state.velocity[2],
                self.state.airspeed,
                self.prev_action[0],  # elevator position
                self.prev_action[1],  # aileron position
                self.prev_action[2],  # rudder position
                self.prev_action[3],  # throttle position
            ], dtype=np.float32)
        
        # Normalize if configured
        if self.config.get("normalize_observations", True):
            obs = self._normalize_observation(obs)
        
        return obs
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on control level and task."""
        reward = 0.0
        
        if self.control_level == ControlMode.WAYPOINT:
            # Level 1: Reward for reaching waypoint
            distance = np.linalg.norm([
                self.target_waypoint.north - self.state.north,
                self.target_waypoint.east - self.state.east,
                self.target_waypoint.down - self.state.down
            ])
            
            reward -= distance / 100.0  # Distance penalty
            
            if distance < 5.0:  # Goal reached
                reward += 1000.0
            
            # Progress reward
            if hasattr(self, 'prev_distance'):
                progress = self.prev_distance - distance
                reward += progress * 10.0
            self.prev_distance = distance
            
            # Energy penalty
            reward -= 0.01 * action[3]  # Speed cost
        
        elif self.control_level == ControlMode.HSA:
            # Level 2: Reward for tracking HSA
            heading_error = abs(self._wrap_angle(
                self.target_hsa["heading"] - self.state.heading
            ))
            speed_error = abs(self.target_hsa["speed"] - self.state.airspeed)
            altitude_error = abs(self.target_hsa["altitude"] - self.state.altitude)
            
            reward -= heading_error * 10.0
            reward -= speed_error * 5.0
            reward -= altitude_error * 2.0
            
            # Smoothness bonus
            reward -= 0.1 * (abs(self.state.p) + abs(self.state.q))
            
            # Fuel efficiency
            reward -= 0.01 * action[1]  # Speed command cost
        
        elif self.control_level == ControlMode.STICK_THROTTLE:
            # Level 3: Reward for attitude tracking
            roll_error = abs(self.target_attitude["roll"] - self.state.roll)
            pitch_error = abs(self.target_attitude["pitch"] - self.state.pitch)
            
            reward -= (roll_error**2 + pitch_error**2) * 10.0
            
            # Rate penalty (smoothness)
            reward -= (self.state.p**2 + self.state.q**2 + self.state.r**2) * 0.1
            
            # Action smoothness
            if hasattr(self, 'prev_action'):
                action_diff = np.sum(np.abs(action - self.prev_action))
                reward -= action_diff * 0.5
            
            # Altitude maintenance
            reward -= abs(self.state.altitude - self.target_altitude) * 0.5
        
        elif self.control_level == ControlMode.SURFACE:
            # Level 4: Task-specific reward (maintain level flight)
            reward -= (abs(self.state.roll) + abs(self.state.pitch)) * 10.0
            reward -= abs(self.state.altitude - self.target_altitude) * 5.0
            
            # Control effort penalty
            reward -= np.sum(np.abs(action)) * 0.1
            
            # Efficiency bonus
            if hasattr(self.state, 'drag_coefficient'):
                reward -= self.state.drag_coefficient * 10.0
        
        # Crash penalty
        if self.state.altitude < 0:
            reward -= 1000.0
        
        # Angle limits
        if abs(self.state.roll) > np.radians(60) or abs(self.state.pitch) > np.radians(45):
            reward -= 100.0
        
        return float(reward)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Crash
        if self.state.altitude < 0:
            return True
        
        # Extreme attitudes
        if abs(self.state.roll) > np.radians(90) or abs(self.state.pitch) > np.radians(60):
            return True
        
        # Out of bounds (position)
        if np.linalg.norm(self.state.position[:2]) > 1000:
            return True
        
        # Level-specific termination
        if self.control_level == ControlMode.WAYPOINT:
            # Success: reached waypoint
            distance = np.linalg.norm([
                self.target_waypoint.north - self.state.north,
                self.target_waypoint.east - self.state.east,
                self.target_waypoint.down - self.state.down
            ])
            if distance < 5.0:
                return True
        
        return False
    
    def _get_initial_state(self) -> AircraftState:
        """Generate initial state (with curriculum)."""
        # Start simple, progressively harder
        if self.curriculum_level == 0:
            # Easy: level flight, slight perturbation
            state = AircraftState(
                position=np.array([0, 0, -100]) + np.random.randn(3) * 5,
                velocity=np.array([20, 0, 0]) + np.random.randn(3) * 1,
                attitude=np.random.randn(3) * 0.1,
                airspeed=20.0
            )
        elif self.curriculum_level == 1:
            # Medium: larger perturbations
            state = AircraftState(
                position=np.array([0, 0, -100]) + np.random.randn(3) * 20,
                velocity=np.array([20, 0, 0]) + np.random.randn(3) * 5,
                attitude=np.random.randn(3) * 0.3,
                airspeed=20.0
            )
        else:
            # Hard: random states
            state = AircraftState(
                position=np.random.randn(3) * 50 + np.array([0, 0, -100]),
                velocity=np.random.randn(3) * 10 + np.array([20, 0, 0]),
                attitude=np.random.randn(3) * 0.5,
                airspeed=15 + np.random.rand() * 10
            )
        
        return state
    
    def update_curriculum(self, success_rate: float):
        """Update curriculum level based on performance."""
        if success_rate > 0.8 and self.curriculum_level < 2:
            self.curriculum_level += 1
            print(f"Curriculum advanced to level {self.curriculum_level}")
```

## Training Scripts

### Level 3 Training (Stick & Throttle)

**File**: `training/train_level3_ppo.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import yaml


def make_env(config, rank):
    """Create environment for parallel training."""
    def _init():
        env = AircraftEnv(config)
        env.seed(config["seed"] + rank)
        return env
    return _init


def train_level3_agent():
    """Train Level 3 (Stick & Throttle) agent with PPO."""
    
    # Load config
    with open("configs/training/level3_ppo.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create vectorized environments (parallel)
    num_envs = config.get("num_envs", 8)
    env = SubprocVecEnv([make_env(config["env"], i) for i in range(num_envs)])
    
    # Create eval environment
    eval_env = AircraftEnv(config["env"])
    
    # PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["hyperparameters"]["learning_rate"],
        n_steps=config["hyperparameters"]["n_steps"],
        batch_size=config["hyperparameters"]["batch_size"],
        gamma=config["hyperparameters"]["gamma"],
        gae_lambda=config["hyperparameters"]["gae_lambda"],
        ent_coef=config["hyperparameters"]["ent_coef"],
        vf_coef=config["hyperparameters"]["vf_coef"],
        max_grad_norm=config["hyperparameters"]["max_grad_norm"],
        verbose=1,
        tensorboard_log="./logs/level3_ppo"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/level3_ppo",
        name_prefix="ppo_level3"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/level3_ppo",
        log_path="./logs/level3_ppo",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True
    )
    
    # Train
    total_timesteps = config.get("total_timesteps", 1_000_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    model.save("models/level3_ppo_final")
    
    return model


if __name__ == "__main__":
    train_level3_agent()
```

### Training Configuration

**File**: `configs/training/level3_ppo.yaml`

```yaml
env:
  control_level: "STICK_THROTTLE"
  max_steps: 1000
  dt: 0.02  # 50 Hz
  normalize_observations: true
  
  backend:
    backend_type: "simulation"
    simulator: "simplified"
    sensor_noise: true
    
  task:
    type: "attitude_tracking"
    target_roll: 0.0
    target_pitch: 0.0
    target_altitude: 100.0

hyperparameters:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

training:
  total_timesteps: 1_000_000
  num_envs: 8
  seed: 42

curriculum:
  enabled: true
  levels: 3
  success_threshold: 0.8
```

## Domain Randomization

**File**: `training/domain_randomization.py`

```python
class DomainRandomizer:
    """Randomize simulation parameters for robustness."""
    
    def __init__(self, config):
        self.config = config
    
    def randomize_aircraft_params(self):
        """Randomize aircraft physical parameters."""
        return {
            "mass": np.random.uniform(0.9, 1.1),  # ±10%
            "inertia_xx": np.random.uniform(0.8, 1.2),
            "inertia_yy": np.random.uniform(0.8, 1.2),
            "inertia_zz": np.random.uniform(0.8, 1.2),
            "cg_offset": np.random.randn(3) * 0.01,  # ±1cm
        }
    
    def randomize_environment(self):
        """Randomize environmental conditions."""
        return {
            "wind_speed": np.random.uniform(0, 5),  # 0-5 m/s
            "wind_direction": np.random.uniform(0, 2*np.pi),
            "turbulence_intensity": np.random.uniform(0, 0.3),
            "air_density": np.random.uniform(0.95, 1.05),
        }
    
    def randomize_sensors(self):
        """Randomize sensor noise parameters."""
        return {
            "imu_noise_stddev": np.random.uniform(0.001, 0.01),
            "gps_noise_stddev": np.random.uniform(0.5, 2.0),
            "delay_ms": np.random.uniform(0, 20),
        }
    
    def randomize_actuators(self):
        """Randomize actuator dynamics."""
        return {
            "servo_delay_ms": np.random.uniform(5, 20),
            "servo_noise": np.random.uniform(0, 0.02),
            "throttle_response_time": np.random.uniform(0.05, 0.2),
        }
```

## Level-Specific Training Strategies

### Level 4 (Surfaces) - Easiest

```python
# Fast learning, dense rewards
config_level4 = {
    "algorithm": "SAC",  # Good for continuous control
    "total_timesteps": 100_000,  # Learns quickly
    "curriculum": False,  # Not needed
    "domain_randomization": True,  # Essential for sim-to-real
}
```

### Level 3 (Stick) - Easy

```python
config_level3 = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "curriculum": True,
    "reward_shaping": "dense",
}
```

### Level 2 (HSA) - Medium

```python
config_level2 = {
    "algorithm": "TD3",
    "total_timesteps": 1_000_000,
    "curriculum": True,
    "hierarchical": False,  # or use Level 3 as sub-policy
}
```

### Level 1 (Waypoint) - Hardest

```python
config_level1 = {
    "algorithm": "PPO + HER",  # Hindsight Experience Replay for sparse rewards
    "total_timesteps": 2_000_000,
    "curriculum": True,
    "hierarchical": True,  # Use Level 3 for tracking
}
```

## Hyperparameter Tuning

**Using Optuna**:

```python
import optuna
from optuna.pruners import MedianPruner


def objective(trial):
    """Optuna objective for hyperparameter tuning."""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    
    # Train agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0
    )
    
    model.learn(total_timesteps=50000)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    return mean_reward


# Run optimization
study = optuna.create_study(direction="maximize", pruner=MedianPruner())
study.optimize(objective, n_trials=100)

print(f"Best hyperparameters: {study.best_params}")
```

---

**Document Status**: Complete
**Last Updated**: 2025-10-09
**Related Documents**: 03_CONTROL_HIERARCHY.md, 05_AGENT_INTERFACE_CONTROL.md
