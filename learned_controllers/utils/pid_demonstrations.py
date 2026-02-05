"""Collect demonstrations from PID controller for imitation learning."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pickle

from controllers.rate_agent import RateAgent
from controllers.types import ControlMode, ControlCommand, ControllerConfig
from learned_controllers.envs.rate_env import RateControlEnv


def collect_pid_demonstrations(
    n_episodes: int = 100,
    difficulty: str = "medium",
    save_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect demonstrations from PID rate controller.

    Args:
        n_episodes: Number of episodes to collect
        difficulty: Environment difficulty
        save_path: Path to save demonstrations (optional)
        seed: Random seed

    Returns:
        Tuple of (observations, actions) arrays
    """
    # Create environment
    env = RateControlEnv(
        difficulty=difficulty,
        episode_length=10.0,
        dt=0.02,
        rng_seed=seed
    )

    # Create PID controller
    config = ControllerConfig()
    pid_agent = RateAgent(config)

    observations = []
    actions = []

    print(f"Collecting {n_episodes} episodes of PID demonstrations...")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        pid_agent.reset()

        ep_obs = []
        ep_actions = []
        steps = 0

        while True:
            # Get state from environment
            state = env.sim.get_state()

            # Get rate command from environment
            p_cmd, q_cmd, r_cmd = env.rate_command

            # Create command for PID controller
            command = ControlCommand(
                mode=ControlMode.RATE,
                roll_rate=p_cmd,
                pitch_rate=q_cmd,
                yaw_rate=r_cmd,
                throttle=0.6  # Cruise throttle
            )

            # Get PID action
            surfaces = pid_agent.compute_action(command, state, dt=0.02)
            action = np.array([
                surfaces.aileron,
                surfaces.elevator,
                surfaces.rudder,
                surfaces.throttle
            ], dtype=np.float32)

            # Store demonstration
            ep_obs.append(obs)
            ep_actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        observations.extend(ep_obs)
        actions.extend(ep_actions)

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}, steps this ep: {steps}, total samples: {len(observations)}")

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    print(f"Collected {len(observations)} demonstration samples")

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'observations': observations, 'actions': actions}, f)
        print(f"Saved demonstrations to {save_path}")

    return observations, actions


def load_demonstrations(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load demonstrations from file.

    Args:
        path: Path to saved demonstrations

    Returns:
        Tuple of (observations, actions) arrays
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['observations'], data['actions']


if __name__ == "__main__":
    # Collect and save demonstrations
    obs, acts = collect_pid_demonstrations(
        n_episodes=200,
        difficulty="medium",
        save_path="learned_controllers/data/pid_demos.pkl"
    )
    print(f"Observations shape: {obs.shape}")
    print(f"Actions shape: {acts.shape}")
