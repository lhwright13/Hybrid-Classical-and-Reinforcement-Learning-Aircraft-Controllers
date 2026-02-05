#!/usr/bin/env python3
"""Recovery script to load checkpoint from early training and continue with better config."""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sb3_contrib import RecurrentPPO


def find_best_checkpoint(checkpoint_dir: str = "learned_controllers/models/checkpoints"):
    """Find the best checkpoint based on evaluation data.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint
    """
    eval_file = "learned_controllers/models/best_rate_controller/evaluations.npz"

    if os.path.exists(eval_file):
        data = np.load(eval_file)
        timesteps = data['timesteps']
        results = data['results']

        # Calculate mean reward for each evaluation
        mean_rewards = results.mean(axis=1)

        print("\nEvaluation History:")
        print("=" * 60)
        for i, (step, reward) in enumerate(zip(timesteps, mean_rewards)):
            print(f"Step {step:7d}: Mean Reward = {reward:8.2f}")

        # Find best checkpoint
        best_idx = np.argmax(mean_rewards)
        best_step = timesteps[best_idx]
        best_reward = mean_rewards[best_idx]

        print("=" * 60)
        print(f"Best checkpoint: step {best_step} (reward: {best_reward:.2f})")
        print("=" * 60)

        return best_step, best_reward
    else:
        print(f"No evaluation file found at {eval_file}")
        print("Using earliest checkpoint (10k steps)...")
        return 10000, None


def recover_checkpoint(checkpoint_step: int, output_path: str):
    """Copy and verify checkpoint.

    Args:
        checkpoint_step: Timestep of checkpoint to recover
        output_path: Where to save recovered model
    """
    checkpoint_file = f"learned_controllers/models/checkpoints/rate_controller_{checkpoint_step}_steps.zip"

    if not os.path.exists(checkpoint_file):
        print(f"ERROR: Checkpoint not found: {checkpoint_file}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = "learned_controllers/models/checkpoints"
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.zip'):
                print(f"  - {f}")
        return False

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Copy checkpoint
    print(f"\nCopying checkpoint from {checkpoint_file}")
    print(f"             to {output_path}")
    shutil.copy2(checkpoint_file, output_path)

    # Verify checkpoint loads
    print("\nVerifying checkpoint...")
    try:
        model = RecurrentPPO.load(output_path)
        print("✓ Checkpoint loaded successfully!")
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  Policy: {model.policy.__class__.__name__}")
        return True
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Recover training from best checkpoint"
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific checkpoint step to recover (default: auto-select best)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="learned_controllers/models/recovered_model.zip",
        help="Output path for recovered model",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRAINING RECOVERY SCRIPT")
    print("=" * 60)

    # Find best checkpoint
    if args.checkpoint_step is None:
        checkpoint_step, best_reward = find_best_checkpoint()
    else:
        checkpoint_step = args.checkpoint_step
        print(f"\nUsing specified checkpoint: {checkpoint_step} steps")

    # Recover checkpoint
    success = recover_checkpoint(checkpoint_step, args.output)

    if success:
        print("\n" + "=" * 60)
        print("RECOVERY COMPLETE!")
        print("=" * 60)
        print(f"\nRecovered model saved to: {args.output}")
        print("\nNext steps:")
        print("1. Review the improved config: learned_controllers/config/ppo_lstm_improved.yaml")
        print("2. Continue training with:")
        print(f"   ./venv/bin/python learned_controllers/train_rate.py \\")
        print(f"     --config learned_controllers/config/ppo_lstm_improved.yaml \\")
        print(f"     --resume {args.output}")
        print("\n   OR start fresh with the improved config:")
        print(f"   ./venv/bin/python learned_controllers/train_rate.py \\")
        print(f"     --config learned_controllers/config/ppo_lstm_improved.yaml")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("RECOVERY FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
