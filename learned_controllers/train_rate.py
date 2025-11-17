#!/usr/bin/env python3
"""Training script for learned rate controller using PPO + LSTM."""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from learned_controllers.envs.rate_env import RateControlEnv
from learned_controllers.networks.lstm_policy import LSTMPolicy, SimpleMLPPolicy
from learned_controllers.utils.training_utils import make_env, create_vec_env, create_callbacks


def load_config(config_path: str = "learned_controllers/config/ppo_lstm.yaml") -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_phase(
    model,
    phase_config: dict,
    env,
    eval_env,
    callbacks,
    total_timesteps: int,
):
    """Train a single curriculum phase.

    Args:
        model: PPO/RecurrentPPO model
        phase_config: Phase configuration
        env: Training environment
        eval_env: Evaluation environment
        callbacks: Training callbacks
        total_timesteps: Number of timesteps to train

    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training Phase: {phase_config['name']}")
    print(f"Difficulty: {phase_config['difficulty']}")
    print(f"Command Type: {phase_config['command_type']}")
    print(f"Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    # Update environment difficulty
    # Note: This requires recreating environments for vectorized envs
    # For simplicity, we'll train with the phase's timesteps

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=False,  # Continue from previous phase
    )

    return model


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train learned rate controller")
    parser.add_argument(
        "--config",
        type=str,
        default="learned_controllers/config/ppo_lstm.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Use MLP instead of LSTM",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="TensorBoard log directory (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # Override TensorBoard path if specified
    if args.tensorboard:
        config["paths"]["tensorboard_log"] = args.tensorboard

    # Create environments
    print("Creating environments...")
    n_envs = config["training"]["n_envs"]
    env = create_vec_env(config, n_envs=n_envs, seed=seed)
    eval_env = create_vec_env(config, n_envs=1, seed=seed + 1000)

    # Create callbacks
    callbacks = create_callbacks(config, eval_env)

    # Policy kwargs
    if args.no_lstm or not config["lstm"]["enabled"]:
        print("Using MLP policy")
        policy_kwargs = SimpleMLPPolicy.get_policy_kwargs(
            net_arch=config["mlp"]["net_arch"]
        )
        model_class = PPO
        policy_name = "MlpPolicy"
    else:
        print("Using LSTM policy")
        policy_kwargs = LSTMPolicy.get_policy_kwargs(
            lstm_hidden_size=config["lstm"]["lstm_hidden_size"],
            n_lstm_layers=config["lstm"]["n_lstm_layers"],
            features_dim=config["lstm"]["features_dim"],
        )
        model_class = RecurrentPPO
        policy_name = "MlpLstmPolicy"

    # PPO hyperparameters
    ppo_config = config["ppo"]

    # Create model
    print(f"Creating {model_class.__name__} model...")
    model = model_class(
        policy_name,
        env,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        clip_range_vf=ppo_config["clip_range_vf"],
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        use_sde=ppo_config["use_sde"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=config["paths"]["tensorboard_log"],
        verbose=1,
        seed=seed,
    )

    # Curriculum learning
    if config["curriculum"]["enabled"]:
        print("\n" + "="*60)
        print("CURRICULUM LEARNING ENABLED")
        print("="*60)

        for phase in config["curriculum"]["phases"]:
            # Update environment config for phase
            config["environment"]["difficulty"] = phase["difficulty"]
            config["environment"]["command_type"] = phase["command_type"]

            # Recreate environments for new difficulty
            env.close()
            eval_env.close()
            env = create_vec_env(config, n_envs=n_envs, seed=seed)
            eval_env = create_vec_env(config, n_envs=1, seed=seed + 1000)

            # Update model environment
            model.set_env(env)

            # Recreate callbacks with new eval env
            callbacks = create_callbacks(config, eval_env)

            # Train phase
            model = train_phase(
                model,
                phase,
                env,
                eval_env,
                callbacks,
                phase["timesteps"],
            )

    else:
        # Standard training (no curriculum)
        print("\n" + "="*60)
        print("STANDARD TRAINING (no curriculum)")
        print("="*60)

        total_timesteps = config["training"]["total_timesteps"]
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config["training"]["log_interval"],
        )

    # Save final model
    final_path = f"{config['paths']['model_save_dir']}/final_model"
    print(f"\nSaving final model to: {final_path}")
    model.save(final_path)

    # Cleanup
    env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {config['paths']['best_model_path']}")
    print(f"Final model saved to: {final_path}")
    print(f"TensorBoard logs: {config['paths']['tensorboard_log']}")
    print("="*60)


if __name__ == "__main__":
    main()
