#!/usr/bin/env python3
"""Overnight training script with imitation learning and curriculum.

Usage:
    python learned_controllers/train_overnight.py --config learned_controllers/config/overnight_v2.yaml
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from learned_controllers.envs.rate_env import RateControlEnv
from learned_controllers.utils.training_utils import (
    load_config,
    behavior_cloning_pretrain,
    run_final_evaluation,
)
from learned_controllers.utils.pid_demonstrations import collect_pid_demonstrations, load_demonstrations


def _make_env(difficulty: str, command_type: str, seed: int, dt: float = 0.02):
    """Create environment factory."""
    def _init():
        return RateControlEnv(
            difficulty=difficulty,
            episode_length=10.0,
            dt=dt,
            command_type=command_type,
            rng_seed=seed
        )
    return _init


def train_phase(model, phase, env, eval_env, config, model_dir):
    """Train a single curriculum phase."""
    print(f"\n{'='*60}")
    print(f"Phase: {phase['name']}")
    print(f"Difficulty: {phase['difficulty']}")
    print(f"Command Type: {phase['command_type']}")
    print(f"Timesteps: {phase['timesteps']:,}")
    print(f"{'='*60}\n")

    eval_config = config['evaluation']
    ckpt_config = config['checkpointing']

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(model_dir / "eval_logs"),
            eval_freq=eval_config['eval_freq'],
            n_eval_episodes=eval_config['n_eval_episodes'],
            deterministic=eval_config['deterministic'],
        ),
        CheckpointCallback(
            save_freq=ckpt_config['save_freq'],
            save_path=str(model_dir / "checkpoints"),
            name_prefix=f"phase_{phase['name']}"
        )
    ]

    model.learn(
        total_timesteps=phase['timesteps'],
        callback=callbacks,
        log_interval=config['logging']['log_interval'],
        reset_num_timesteps=False,
    )

    # Save phase checkpoint
    phase_path = model_dir / f"phase_{phase['name']}_complete"
    model.save(str(phase_path))
    print(f"Saved phase checkpoint: {phase_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Overnight training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--skip-demos", action="store_true", help="Skip demo collection (use existing)")
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavior cloning")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("="*60)
    print("OVERNIGHT TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Calculate total steps
    total_steps = sum(p['timesteps'] for p in config['curriculum']['phases'])
    print(f"\nTotal steps: {total_steps:,}")
    print(f"Estimated time: {total_steps / 1100 / 3600:.1f} hours (at 1100 FPS)")

    # Setup paths
    model_dir = Path(config['paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['paths']['tensorboard_log'])
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = config['seed']
    np.random.seed(seed)

    # Step 1: Collect or load demonstrations
    demo_path = Path(config['demonstrations']['save_path'])
    if config['approach']['use_imitation'] and not args.skip_bc:
        if demo_path.exists() and args.skip_demos:
            print(f"\nLoading existing demos from {demo_path}")
            observations, actions = load_demonstrations(str(demo_path))
        else:
            print(f"\nCollecting {config['demonstrations']['n_episodes']} PID demonstrations...")
            observations, actions = collect_pid_demonstrations(
                n_episodes=config['demonstrations']['n_episodes'],
                difficulty=config['demonstrations']['difficulty'],
                save_path=str(demo_path),
                seed=seed
            )
        print(f"Demonstrations: {len(observations):,} samples")

    # Step 2: Create initial environment (first phase)
    first_phase = config['curriculum']['phases'][0]
    n_envs = config['parallel']['n_envs']

    print(f"\nCreating {n_envs} parallel environments...")
    env_fns = [
        _make_env(first_phase['difficulty'], first_phase['command_type'], seed + i)
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns)
    eval_env = DummyVecEnv([_make_env(first_phase['difficulty'], first_phase['command_type'], seed + 1000)])

    # Step 3: Create or load model
    net_config = config['network']
    ppo_config = config['ppo']

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=str(log_dir))
    else:
        print("\nCreating PPO model...")
        policy_kwargs = {"net_arch": net_config['mlp']['net_arch']}

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=config['logging']['verbose'],
            seed=seed,
        )

    # Step 4: Behavior cloning pretraining
    if config['approach']['use_imitation'] and not args.skip_bc and not args.resume:
        bc_config = config['behavior_cloning']
        behavior_cloning_pretrain(
            model, observations, actions,
            epochs=bc_config['epochs'],
            batch_size=bc_config['batch_size'],
            lr=bc_config['learning_rate'],
        )
        bc_path = model_dir / "bc_pretrained"
        model.save(str(bc_path))
        print(f"Saved BC model: {bc_path}")

    # Step 5: Curriculum training
    print(f"\n{'='*60}")
    print("CURRICULUM TRAINING")
    print(f"{'='*60}")

    for i, phase in enumerate(config['curriculum']['phases']):
        # Recreate environments for new phase
        if i > 0:
            env.close()
            eval_env.close()

            env_fns = [
                _make_env(phase['difficulty'], phase['command_type'], seed + i * 100 + j)
                for j in range(n_envs)
            ]
            env = SubprocVecEnv(env_fns)
            eval_env = DummyVecEnv([_make_env(phase['difficulty'], phase['command_type'], seed + 1000 + i)])
            model.set_env(env)

        model = train_phase(model, phase, env, eval_env, config, model_dir)

    # Step 6: Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {model_dir / 'best'}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")

    run_final_evaluation(model, difficulty='hard', n_episodes=10, dt=0.02)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
