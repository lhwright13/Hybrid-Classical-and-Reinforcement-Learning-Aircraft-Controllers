#!/usr/bin/env python3
"""Training script with imitation learning pretraining.

This script:
1. Collects demonstrations from PID controller
2. Pretrains policy with behavior cloning
3. Fine-tunes with RL

This approach typically converges much faster than pure RL.
"""

import sys
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from learned_controllers.envs.rate_env import RateControlEnv
from learned_controllers.envs.residual_rate_env import ResidualRateControlEnv
from learned_controllers.utils.training_utils import behavior_cloning_pretrain
from learned_controllers.utils.pid_demonstrations import collect_pid_demonstrations, load_demonstrations


def _make_env(difficulty: str, seed: int, residual: bool = False):
    """Create environment factory."""
    def _init():
        if residual:
            env = ResidualRateControlEnv(
                difficulty=difficulty,
                episode_length=10.0,
                dt=0.02,
                rng_seed=seed
            )
        else:
            env = RateControlEnv(
                difficulty=difficulty,
                episode_length=10.0,
                dt=0.02,
                rng_seed=seed
            )
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train with imitation learning")
    parser.add_argument("--n-demos", type=int, default=200, help="Number of demo episodes")
    parser.add_argument("--bc-epochs", type=int, default=15, help="Behavior cloning epochs")
    parser.add_argument("--rl-steps", type=int, default=500000, help="RL fine-tuning steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--residual", action="store_true", help="Use residual RL")
    parser.add_argument("--difficulty", type=str, default="medium", help="Difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavior cloning")
    args = parser.parse_args()

    print("=" * 60)
    print("Training with Imitation Learning")
    print("=" * 60)
    print(f"  Demonstrations: {args.n_demos} episodes")
    print(f"  BC epochs: {args.bc_epochs}")
    print(f"  RL steps: {args.rl_steps}")
    print(f"  Residual mode: {args.residual}")
    print(f"  Difficulty: {args.difficulty}")
    print()

    # Paths
    demo_path = Path("learned_controllers/data/pid_demos.pkl")
    model_dir = Path("learned_controllers/models/imitation_trained")
    log_dir = Path("learned_controllers/logs/tensorboard")

    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect or load demonstrations
    if not args.skip_bc:
        if demo_path.exists():
            print(f"Loading existing demonstrations from {demo_path}")
            observations, actions = load_demonstrations(str(demo_path))
        else:
            print("Collecting PID demonstrations...")
            observations, actions = collect_pid_demonstrations(
                n_episodes=args.n_demos,
                difficulty=args.difficulty,
                save_path=str(demo_path),
                seed=args.seed
            )

        print(f"Demonstrations: {len(observations)} samples")

    # Step 2: Create environments
    print("\nCreating environments...")
    env_fns = [_make_env(args.difficulty, args.seed + i, args.residual) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)

    eval_env = DummyVecEnv([_make_env(args.difficulty, args.seed + 1000, args.residual)])

    # Step 3: Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )

    # Step 4: Behavior cloning pretraining
    if not args.skip_bc:
        behavior_cloning_pretrain(
            model,
            observations,
            actions,
            epochs=args.bc_epochs,
            batch_size=256,
            lr=1e-3
        )

        # Save BC pretrained model
        bc_path = model_dir / "bc_pretrained"
        model.save(str(bc_path))
        print(f"Saved BC pretrained model to {bc_path}")

    # Step 5: RL fine-tuning
    print("\nStarting RL fine-tuning...")

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir),
            eval_freq=25000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=100000,
            save_path=str(model_dir / "checkpoints"),
            name_prefix="imitation_model"
        )
    ]

    model.learn(
        total_timesteps=args.rl_steps,
        callback=callbacks,
        log_interval=10,
    )

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Final model: {final_path}")
    print(f"  Best model: {model_dir / 'best'}")
    print()

    # Quick evaluation
    print("Final evaluation (5 episodes):")
    test_env = RateControlEnv(difficulty=args.difficulty, episode_length=10.0, dt=0.02)

    for ep in range(5):
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

        print(f"  Episode {ep + 1}: {steps} steps ({steps * 0.02:.1f}s), reward: {total_reward:.1f}")

    env.close()
    eval_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
