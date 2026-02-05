#!/usr/bin/env python3
"""Evaluation script for learned rate controller vs PID baseline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from typing import List

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from learned_controllers.envs.rate_env import RateControlEnv
from learned_controllers.eval.metrics import MetricsCalculator, compare_metrics, RateControlMetrics
from controllers.rate_agent import RateAgent
from controllers.types import ControlCommand, ControlMode
from controllers.config import load_config_from_yaml as load_controller_config


def evaluate_learned_controller(
    model_path: str,
    n_episodes: int = 10,
    difficulty: str = "medium",
    episode_length: float = 10.0,
    command_type: str = "step",
    deterministic: bool = True,
) -> List[RateControlMetrics]:
    """Evaluate learned controller.

    Args:
        model_path: Path to trained model
        n_episodes: Number of evaluation episodes
        difficulty: Environment difficulty
        episode_length: Episode duration in seconds
        command_type: Type of rate command
        deterministic: Use deterministic policy

    Returns:
        List of metrics for each episode
    """
    print(f"\nEvaluating Learned Controller: {model_path}")
    print(f"Episodes: {n_episodes}, Difficulty: {difficulty}, Command: {command_type}")

    # Load model
    try:
        model = RecurrentPPO.load(model_path)
        print("Loaded RecurrentPPO model")
    except:
        model = PPO.load(model_path)
        print("Loaded PPO model")

    # Create environment (match training config)
    env = RateControlEnv(
        difficulty=difficulty,
        episode_length=episode_length,
        dt=0.02,
        command_type=command_type,
    )

    # Metrics calculator
    calculator = MetricsCalculator()

    # Run episodes
    all_metrics = []

    for ep in range(n_episodes):
        print(f"  Episode {ep+1}/{n_episodes}...", end="")

        # Reset environment
        obs, info = env.reset()
        done = False

        # Episode data collection
        times = []
        rates = []
        commands = []
        actions_list = []
        rewards_list = []

        # Reset LSTM state if RecurrentPPO
        if isinstance(model, RecurrentPPO):
            lstm_states = None

        while not done:
            # Get action from model
            if isinstance(model, RecurrentPPO):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    deterministic=deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Collect data
            state = env.sim.get_state()
            times.append(info["time"])
            rates.append([state.p, state.q, state.r])
            commands.append(info["rate_command"])
            actions_list.append(action)
            rewards_list.append(reward)

        # Compute metrics
        metrics = calculator.compute_metrics(
            times=np.array(times),
            rates=np.array(rates),
            commands=np.array(commands),
            actions=np.array(actions_list),
            rewards=np.array(rewards_list),
        )

        all_metrics.append(metrics)
        print(f" Reward: {metrics.total_reward:.2f}, Success: {metrics.success}")

    env.close()

    return all_metrics


def evaluate_pid_controller(
    n_episodes: int = 10,
    difficulty: str = "medium",
    episode_length: float = 10.0,
    command_type: str = "step",
) -> List[RateControlMetrics]:
    """Evaluate PID baseline controller.

    Args:
        n_episodes: Number of evaluation episodes
        difficulty: Environment difficulty
        episode_length: Episode duration in seconds
        command_type: Type of rate command

    Returns:
        List of metrics for each episode
    """
    print(f"\nEvaluating PID Baseline Controller")
    print(f"Episodes: {n_episodes}, Difficulty: {difficulty}, Command: {command_type}")

    # Load PID controller config
    controller_config = load_controller_config()

    # Create PID rate agent
    pid_agent = RateAgent(controller_config)

    # Create environment (match training config)
    env = RateControlEnv(
        difficulty=difficulty,
        episode_length=episode_length,
        dt=0.02,
        command_type=command_type,
    )

    # Metrics calculator
    calculator = MetricsCalculator()

    # Run episodes
    all_metrics = []

    for ep in range(n_episodes):
        print(f"  Episode {ep+1}/{n_episodes}...", end="")

        # Reset environment
        obs, info = env.reset()
        done = False

        # Reset PID controller
        pid_agent.reset()

        # Episode data collection
        times = []
        rates = []
        commands = []
        actions_list = []
        rewards_list = []

        while not done:
            # Get state
            state = env.sim.get_state()

            # Create rate command
            rate_cmd = info["rate_command"]
            command = ControlCommand(
                mode=ControlMode.RATE,
                roll_rate=rate_cmd[0],
                pitch_rate=rate_cmd[1],
                yaw_rate=rate_cmd[2],
                throttle=0.5,  # Maintain throttle
            )

            # Compute PID action
            surfaces = pid_agent.compute_action(command, state)

            # Convert to action array
            action = np.array([
                surfaces.aileron,
                surfaces.elevator,
                surfaces.rudder,
                surfaces.throttle,
            ])

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Collect data
            times.append(info["time"])
            rates.append([state.p, state.q, state.r])
            commands.append(rate_cmd)
            actions_list.append(action)
            rewards_list.append(reward)

        # Compute metrics
        metrics = calculator.compute_metrics(
            times=np.array(times),
            rates=np.array(rates),
            commands=np.array(commands),
            actions=np.array(actions_list),
            rewards=np.array(rewards_list),
        )

        all_metrics.append(metrics)
        print(f" Reward: {metrics.total_reward:.2f}, Success: {metrics.success}")

    env.close()

    return all_metrics


def aggregate_metrics(metrics_list: List[RateControlMetrics]) -> RateControlMetrics:
    """Aggregate metrics across episodes.

    Args:
        metrics_list: List of metrics from multiple episodes

    Returns:
        Averaged metrics
    """
    # Compute mean of each metric
    n = len(metrics_list)

    avg_metrics = RateControlMetrics()

    # Average all numeric fields
    for field in avg_metrics.__dataclass_fields__:
        if field == "success":
            # Success rate
            avg_metrics.success = sum(m.success for m in metrics_list) / n
        else:
            # Average value
            values = [getattr(m, field) for m in metrics_list]
            setattr(avg_metrics, field, np.mean(values))

    return avg_metrics


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate learned rate controller")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to learned model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Environment difficulty",
    )
    parser.add_argument(
        "--compare-pid",
        action="store_true",
        help="Compare with PID baseline",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--episode-length",
        type=float,
        default=10.0,
        help="Episode length in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--command-type",
        type=str,
        default="step",
        choices=["step", "ramp", "sine", "random"],
        help="Command type (default: step)",
    )

    args = parser.parse_args()

    # Evaluate learned controller
    learned_metrics = evaluate_learned_controller(
        model_path=args.model,
        n_episodes=args.n_episodes,
        difficulty=args.difficulty,
        episode_length=args.episode_length,
        command_type=args.command_type,
        deterministic=not args.stochastic,
    )

    # Aggregate metrics
    learned_avg = aggregate_metrics(learned_metrics)
    learned_avg.print_summary("Learned Controller")

    # Compare with PID if requested
    if args.compare_pid:
        pid_metrics = evaluate_pid_controller(
            n_episodes=args.n_episodes,
            difficulty=args.difficulty,
            episode_length=args.episode_length,
            command_type=args.command_type,
        )

        # Aggregate PID metrics
        pid_avg = aggregate_metrics(pid_metrics)
        pid_avg.print_summary("PID Controller")

        # Print comparison
        compare_metrics(
            learned_avg,
            pid_avg,
            name_a="Learned",
            name_b="PID",
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
