#!/usr/bin/env python3
"""Example usage of the learned rate controller infrastructure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from learned_controllers.envs.rate_env import RateControlEnv
from learned_controllers.data.generators import RateCommandGenerator, FlightEnvelopeSampler


def example_1_manual_environment_interaction():
    """Example 1: Manually interact with the rate control environment."""
    print("\n" + "="*60)
    print("Example 1: Manual Environment Interaction")
    print("="*60)

    # Create environment
    env = RateControlEnv(
        difficulty="easy",
        episode_length=5.0,
        dt=0.02,
        command_type="step",
    )

    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Rate command: {info['rate_command']}")

    # Run episode with random actions
    total_reward = 0.0
    for step in range(50):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.3f}, "
                  f"rate_error={np.linalg.norm(info['rate_error']):.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()


def example_2_command_generation():
    """Example 2: Generate various rate commands."""
    print("\n" + "="*60)
    print("Example 2: Command Generation")
    print("="*60)

    generator = RateCommandGenerator(difficulty="medium", rng_seed=42)

    # Step command
    print("\n--- Step Commands ---")
    for i in range(3):
        cmd, desc = generator.generate_step_command(num_axes=i+1)
        print(f"{desc}: p={cmd[0]:.2f}, q={cmd[1]:.2f}, r={cmd[2]:.2f} rad/s")

    # Sine command
    print("\n--- Sine Commands ---")
    for i in range(3):
        freq, amps, desc = generator.generate_sine_command()
        print(f"{desc}: Amplitudes p={amps[0]:.2f}, q={amps[1]:.2f}, r={amps[2]:.2f} rad/s")

    # Multi-axis command
    print("\n--- Multi-Axis Command ---")
    cmd, desc = generator.generate_multi_axis_command()
    print(f"{desc}: p={cmd[0]:.2f}, q={cmd[1]:.2f}, r={cmd[2]:.2f} rad/s")


def example_3_flight_envelope_sampling():
    """Example 3: Sample initial conditions from flight envelope."""
    print("\n" + "="*60)
    print("Example 3: Flight Envelope Sampling")
    print("="*60)

    sampler = FlightEnvelopeSampler(rng_seed=42)

    print("\n--- Sampled Initial Conditions ---")
    for i in range(5):
        init = sampler.sample()
        print(f"\nSample {i+1}:")
        print(f"  Airspeed: {init['airspeed']:.1f} m/s")
        print(f"  Altitude: {init['altitude']:.1f} m")
        print(f"  Attitude: roll={np.degrees(init['attitude'][0]):.1f}°, "
              f"pitch={np.degrees(init['attitude'][1]):.1f}°, "
              f"yaw={np.degrees(init['attitude'][2]):.1f}°")


def example_4_reward_calculation():
    """Example 4: Reward function behavior."""
    print("\n" + "="*60)
    print("Example 4: Reward Function Behavior")
    print("="*60)

    from learned_controllers.envs.rewards import RateTrackingReward

    reward_fn = RateTrackingReward(
        w_tracking=1.0,
        w_smoothness=0.01,
        w_stability=0.1,
        w_oscillation=0.5,
    )

    # Simulate some error scenarios
    scenarios = [
        ("Perfect tracking", 0.0, 0.0, 0.0),
        ("Small error", 0.1, 0.1, 0.05),
        ("Large error", 1.0, 0.8, 0.6),
        ("Very large error", 3.0, 2.5, 2.0),
    ]

    action = np.array([0.1, 0.05, -0.02, 0.5])
    prev_action = np.array([0.08, 0.04, -0.03, 0.5])

    print("\n--- Reward Components for Different Errors ---")
    for name, p_err, q_err, r_err in scenarios:
        reward, components = reward_fn.compute(
            p_err, q_err, r_err,
            action, prev_action,
            airspeed=20.0, altitude=100.0,
            roll=0.1, pitch=0.05,
        )
        print(f"\n{name} (errors: p={p_err:.2f}, q={q_err:.2f}, r={r_err:.2f}):")
        print(f"  Total reward: {reward:.4f}")
        print(f"  Tracking:     {components['tracking']:.4f}")
        print(f"  Smoothness:   {components['smoothness']:.4f}")
        print(f"  Stability:    {components['stability']:.4f}")
        print(f"  Oscillation:  {components['oscillation']:.4f}")


def example_5_training_preview():
    """Example 5: Preview training setup (without actually training)."""
    print("\n" + "="*60)
    print("Example 5: Training Setup Preview")
    print("="*60)

    import yaml

    # Load training config
    config_path = "learned_controllers/config/ppo_lstm.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n--- Training Configuration ---")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Parallel envs: {config['training']['n_envs']}")
    print(f"Curriculum learning: {config['curriculum']['enabled']}")

    if config['curriculum']['enabled']:
        print("\nCurriculum phases:")
        for phase in config['curriculum']['phases']:
            print(f"  - {phase['name']}: {phase['difficulty']}, "
                  f"{phase['timesteps']:,} steps, "
                  f"command_type={phase['command_type']}")

    print("\n--- PPO Hyperparameters ---")
    for key, value in config['ppo'].items():
        print(f"  {key}: {value}")

    print("\n--- LSTM Architecture ---")
    if config['lstm']['enabled']:
        print(f"  Hidden size: {config['lstm']['lstm_hidden_size']}")
        print(f"  Layers: {config['lstm']['n_lstm_layers']}")
        print(f"  Features dim: {config['lstm']['features_dim']}")
    else:
        print("  Using MLP (LSTM disabled)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LEARNED RATE CONTROLLER - EXAMPLE USAGE")
    print("="*70)

    # Run examples
    example_1_manual_environment_interaction()
    example_2_command_generation()
    example_3_flight_envelope_sampling()
    example_4_reward_calculation()
    example_5_training_preview()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("\nNext steps:")
    print("  1. Train a model: python learned_controllers/train_rate.py")
    print("  2. Evaluate model: python learned_controllers/eval_rate.py --model <path>")
    print("  3. Deploy in control system (see README.md)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
