#!/usr/bin/env python3
"""Quick test of FlightLogger with RateControlEnv.

This script runs a few episodes with a random policy and logs
flight data to test the TensorBoard Flight plugin integration.
"""

import sys
from pathlib import Path

# Add tensorboard_flight_plugin to path
sys.path.insert(0, str(Path(__file__).parent / "tensorboard_flight_plugin" / "src"))

import numpy as np
from learned_controllers.envs.rate_env import RateControlEnv
from tensorboard_flight import FlightLogger


def main():
    """Run quick test."""
    print("="*60)
    print("Testing FlightLogger with RateControlEnv")
    print("="*60)

    # Create logger
    log_dir = "learned_controllers/logs/flight_test"
    print(f"\nCreating FlightLogger at: {log_dir}")
    logger = FlightLogger(
        log_dir=log_dir,
        max_buffer_size=50,
    )

    # Create environment
    print("Creating RateControlEnv...")
    env = RateControlEnv(
        difficulty="easy",
        episode_length=5.0,  # Short episodes for testing
        dt=0.02,
        command_type="step",
    )

    # Run a few episodes
    num_episodes = 3
    print(f"\nRunning {num_episodes} test episodes...")

    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}")

        # Start episode logging
        logger.start_episode(agent_id="random_policy")

        # Reset environment
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            # Random action
            action = env.action_space.sample()

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Extract flight data
            # Parse observation
            p, q, r = obs[0], obs[1], obs[2]
            airspeed = float(obs[9])
            altitude = float(obs[10])
            roll = float(obs[11])
            pitch = float(obs[12])
            yaw = float(obs[13])

            # Position (approximate)
            position = (0.0, 0.0, altitude)

            # Velocity (approximate from airspeed and attitude)
            vx = airspeed * np.cos(pitch) * np.cos(yaw)
            vy = airspeed * np.cos(pitch) * np.sin(yaw)
            vz = airspeed * np.sin(pitch)
            velocity = (float(vx), float(vy), float(vz))

            # Orientation in degrees
            orientation = (
                float(np.degrees(roll)),
                float(np.degrees(pitch)),
                float(np.degrees(yaw)),
            )

            # Angular velocity
            angular_velocity = (float(p), float(q), float(r))

            # Telemetry
            telemetry = {
                'airspeed': airspeed,
                'altitude': altitude,
                'g_force': 1.0,
                'throttle': float(action[3]),
                'aoa': 0.0,
                'aos': 0.0,
                'heading': float(np.degrees(yaw)),
                'vertical_speed': float(vz),
                'turn_rate': float(np.degrees(r)),
                'bank_angle': float(np.degrees(roll)),
                'aileron': float(action[0]),
                'elevator': float(action[1]),
                'rudder': float(action[2]),
            }

            # RL metrics
            rl_metrics = {
                'reward': float(reward),
                'action': action.tolist(),
            }

            # Add reward components if available
            if 'reward_components' in info:
                rl_metrics['reward_components'] = info['reward_components']

            # Log flight data
            logger.log_flight_data(
                step=step,
                agent_id="random_policy",
                position=position,
                orientation=orientation,
                velocity=velocity,
                angular_velocity=angular_velocity,
                telemetry=telemetry,
                rl_metrics=rl_metrics,
                timestamp=info.get('time', step * 0.02),
            )

            obs = next_obs
            step += 1

            if step % 50 == 0:
                print(f"    Step {step}, reward={reward:.3f}")

        # End episode
        success = not done  # Not terminated early
        termination_reason = "completed" if success else "crash"
        logger.end_episode(
            success=success,
            termination_reason=termination_reason,
            tags=[f"episode_{episode}", "test"],
        )

        print(f"    Logged {step} steps")

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\nLogged {num_episodes} episodes to: {log_dir}")
    print("\nTo view the data:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nThen open: http://localhost:6006")
    print("(Note: 3D visualization not implemented yet, but data is logged)")
    print("="*60)


if __name__ == "__main__":
    main()
