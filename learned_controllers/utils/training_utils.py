"""Shared training utilities for RL controllers.

This module consolidates common functions used across training scripts to eliminate
code duplication and provide a single source of truth.
"""

import os
from typing import Optional
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

            # Log every 10 episodes to avoid excessive data
            flight_callback = FlightLoggerCallback(
                logger=flight_logger,
                log_every_n_episodes=10,
                agent_id="ppo_rate_controller",
                verbose=1,
            )
            callbacks.append(flight_callback)
            print("✓ Flight trajectory logging ENABLED")
            print(f"  Logging to: {flight_logger.log_dir}")
            print(f"  View with: tensorboard --logdir {flight_logger.log_dir}")
        except ImportError as e:
            print(f"Warning: FlightLoggerCallback not available ({e}), skipping flight logging")

    # Combine all callbacks
    return CallbackList(callbacks)
