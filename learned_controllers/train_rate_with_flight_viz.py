#!/usr/bin/env python3
"""Training script with Flight Visualization logging.

This is a modified version of train_rate.py that includes automatic
flight trajectory logging to the TensorBoard Flight plugin.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from learned_controllers.networks.lstm_policy import LSTMPolicy, SimpleMLPPolicy
from learned_controllers.utils.training_utils import create_vec_env, create_callbacks, load_config

# Import TensorBoard Flight plugin
sys.path.insert(0, str(project_root / "tensorboard_flight_plugin" / "src"))
from tensorboard_flight import FlightLogger


def train_phase(
    model,
    phase_config: dict,
    env,
    eval_env,
    callbacks,
    total_timesteps: int,
):
    """Train a single curriculum phase."""
    print(f"\n{'='*60}")
    print(f"Training Phase: {phase_config['name']}")
    print(f"Difficulty: {phase_config['difficulty']}")
    print(f"Command Type: {phase_config['command_type']}")
    print(f"Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=False,
    )

    return model


def main():
    """Main training loop with flight visualization."""
    parser = argparse.ArgumentParser(
        description="Train learned rate controller with flight visualization"
    )
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
    parser.add_argument(
        "--flight-viz",
        action="store_true",
        default=True,
        help="Enable flight trajectory visualization (default: True)",
    )
    parser.add_argument(
        "--no-flight-viz",
        action="store_false",
        dest="flight_viz",
        help="Disable flight trajectory visualization",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
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

    # Create flight logger if enabled
    flight_logger = None
    if args.flight_viz:
        flight_log_dir = f"{config['paths']['tensorboard_log']}_flight"
        flight_logger = FlightLogger(
            log_dir=flight_log_dir,
            max_buffer_size=100,
        )

    # Create environments
    print("Creating environments...")
    n_envs = config["training"]["n_envs"]
    env = create_vec_env(config, n_envs=n_envs, seed=seed)
    eval_env = create_vec_env(config, n_envs=1, seed=seed + 1000)

    # Create callbacks
    callbacks = create_callbacks(config, eval_env, flight_logger)

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

    # Create or load model
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = model_class.load(
            args.resume,
            env=env,
            tensorboard_log=config["paths"]["tensorboard_log"],
        )
        # Update learning rate and other params from config
        model.learning_rate = ppo_config["learning_rate"]
        model.ent_coef = ppo_config["ent_coef"]
    else:
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

        # Calculate starting point for resume
        start_steps = model.num_timesteps if args.resume else 0
        cumulative_steps = 0

        for phase in config["curriculum"]["phases"]:
            phase_end = cumulative_steps + phase["timesteps"]

            # Skip completed phases when resuming
            if start_steps >= phase_end:
                print(f"\nSkipping completed phase: {phase['name']} (ends at {phase_end} steps)")
                cumulative_steps = phase_end
                continue

            # Calculate remaining steps for this phase
            if start_steps > cumulative_steps:
                remaining = phase_end - start_steps
                print(f"\nResuming phase: {phase['name']} with {remaining} steps remaining")
            else:
                remaining = phase["timesteps"]

            cumulative_steps = phase_end
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
            callbacks = create_callbacks(config, eval_env, flight_logger)

            # Train phase
            model = train_phase(
                model,
                phase,
                env,
                eval_env,
                callbacks,
                remaining,
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

    # Close flight logger (writes final data)
    if flight_logger:
        flight_logger.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {config['paths']['best_model_path']}")
    print(f"Final model saved to: {final_path}")
    print(f"TensorBoard logs: {config['paths']['tensorboard_log']}")
    if flight_logger:
        print(f"Flight trajectories: {flight_logger.log_dir}")
        print("\nView flight data with:")
        print(f"  tensorboard --logdir {flight_logger.log_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
