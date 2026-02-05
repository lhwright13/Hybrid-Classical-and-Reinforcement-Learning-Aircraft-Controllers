"""Shared training utilities for RL controllers.

This module consolidates common functions used across training scripts to eliminate
code duplication and provide a single source of truth.
"""

import os
import yaml
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from learned_controllers.envs.rate_env import RateControlEnv


def make_env(config: dict, rank: int = 0, seed: Optional[int] = None):
    """Create and wrap environment.

    Args:
        config: Configuration dict with 'environment' section
        rank: Environment ID for parallel envs (default: 0)
        seed: Random seed (default: None)

    Returns:
        Callable that returns wrapped environment
    """
    def _init():
        env_config = config["environment"]
        env = RateControlEnv(
            difficulty=env_config["difficulty"],
            episode_length=env_config["episode_length"],
            dt=env_config["dt"],
            command_type=env_config["command_type"],
            rng_seed=seed + rank if seed is not None else None,
        )
        env = Monitor(env)
        return env

    return _init


def create_vec_env(config: dict, n_envs: int = 4, seed: Optional[int] = None):
    """Create vectorized environment.

    Args:
        config: Configuration dict
        n_envs: Number of parallel environments (default: 4)
        seed: Random seed (default: None)

    Returns:
        Vectorized environment (DummyVecEnv or SubprocVecEnv)
    """
    if n_envs == 1:
        # Single environment (no parallelization)
        env = DummyVecEnv([make_env(config, 0, seed)])
    else:
        # Parallel environments (faster training)
        env = SubprocVecEnv([
            make_env(config, i, seed) for i in range(n_envs)
        ])

    return env


def create_callbacks(config: dict, eval_env, flight_logger=None):
    """Create training callbacks with optional flight logging.

    Args:
        config: Configuration dict with 'paths', 'training', and 'evaluation' sections
        eval_env: Evaluation environment
        flight_logger: Optional FlightLogger for recording episodes (default: None)

    Returns:
        CallbackList with evaluation, checkpoint, and optional flight logging
    """
    paths = config["paths"]
    eval_config = config["evaluation"]

    # Create directories
    os.makedirs(paths["model_save_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard_log"], exist_ok=True)
    os.makedirs(os.path.dirname(paths["best_model_path"]), exist_ok=True)

    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=paths["best_model_path"],
        log_path=paths["best_model_path"],
        eval_freq=config["training"]["eval_freq"],
        n_eval_episodes=eval_config["n_eval_episodes"],
        deterministic=eval_config["deterministic"],
        render=eval_config["render"],
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=paths["model_save_dir"],
        name_prefix="rate_controller",
    )
    callbacks.append(checkpoint_callback)

    # Flight logging callback (if flight_logger provided)
    if flight_logger is not None:
        try:
            # Import from tensorboard_flight plugin
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root / "tensorboard_flight_plugin" / "src"))
            from tensorboard_flight.callbacks import FlightLoggerCallback

            # Get logging frequency from config (default: every 50 episodes)
            flight_config = config.get("flight_logging", {})
            log_every_n = flight_config.get("log_every_n_episodes", 50)

            flight_callback = FlightLoggerCallback(
                logger=flight_logger,
                log_every_n_episodes=log_every_n,
                agent_id="ppo_rate_controller",
                verbose=1,
            )
            callbacks.append(flight_callback)
            print("Flight trajectory logging ENABLED")
            print(f"  Logging every {log_every_n} episodes")
            print(f"  Log dir: {flight_logger.log_dir}")
        except ImportError as e:
            print(f"Warning: FlightLoggerCallback not available ({e}), skipping flight logging")

    # Combine all callbacks
    return CallbackList(callbacks)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def behavior_cloning_pretrain(model, observations, actions, epochs=10,
                              batch_size=256, lr=1e-3):
    """Pretrain policy network with behavior cloning.

    Args:
        model: PPO model to pretrain
        observations: Expert observations array
        actions: Expert actions array
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    print(f"\n{'='*60}")
    print(f"Behavior Cloning Pretraining ({epochs} epochs)")
    print(f"{'='*60}")

    policy = model.policy

    obs_tensor = torch.FloatTensor(observations)
    act_tensor = torch.FloatTensor(actions)
    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.mlp_extractor.parameters(), lr=lr)
    optimizer.add_param_group({'params': policy.action_net.parameters()})
    criterion = nn.MSELoss()

    policy.train()
    device = next(policy.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            features = policy.extract_features(obs_batch)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)

            loss = criterion(mean_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    policy.eval()
    print("Behavior cloning complete!\n")


def run_final_evaluation(model, difficulty='hard', n_episodes=10, dt=0.02):
    """Run final evaluation episodes and print results.

    Args:
        model: Trained model
        difficulty: Environment difficulty
        n_episodes: Number of evaluation episodes
        dt: Control timestep
    """
    print(f"\nFinal evaluation ({n_episodes} episodes on {difficulty}):")
    test_env = RateControlEnv(difficulty=difficulty, episode_length=10.0, dt=dt)

    rewards = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = test_env.reset(seed=ep * 100)
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep + 1}: {steps} steps ({steps * dt:.1f}s), reward: {total_reward:.1f}")

    print(f"\nAverage: {np.mean(lengths):.0f} steps ({np.mean(lengths) * dt:.1f}s), reward: {np.mean(rewards):.1f}")

    test_env.close()
