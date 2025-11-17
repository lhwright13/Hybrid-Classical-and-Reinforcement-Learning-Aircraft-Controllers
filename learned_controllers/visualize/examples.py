#!/usr/bin/env python3
"""Example usage scripts for visualization tools."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from learned_controllers.visualize import ReplayVisualizer, LearningCurveVisualizer


def example_replay_single_episode():
    """Example: Visualize a single episode with detailed dynamics."""
    print("\n" + "="*60)
    print("Example 1: Single Episode Replay")
    print("="*60)

    # Create visualizer
    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    # Run episode
    print("\nRunning episode...")
    data = viz.run_episode(
        difficulty="medium",
        command_type="step",
        seed=42
    )

    # Plot dynamics
    print("Plotting dynamics...")
    viz.plot_episode(
        data,
        title="Best Model - Medium Difficulty - Step Commands",
        save_path="example_episode.png",
        show=True
    )

    # Print summary
    import numpy as np
    total_reward = np.sum(data["rewards"])
    mean_error = np.mean(np.linalg.norm(data["errors"], axis=1))
    print(f"\nEpisode Summary:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Mean Tracking Error: {mean_error:.4f} rad/s")


def example_compare_difficulties():
    """Example: Compare performance across difficulty levels."""
    print("\n" + "="*60)
    print("Example 2: Compare Difficulty Levels")
    print("="*60)

    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    difficulties = ["easy", "medium", "hard"]
    episodes = []
    labels = []

    for difficulty in difficulties:
        print(f"\nRunning {difficulty} episode...")
        data = viz.run_episode(
            difficulty=difficulty,
            command_type="step",
            seed=42
        )
        episodes.append(data)
        labels.append(difficulty.capitalize())

    # Compare episodes
    print("\nGenerating comparison plot...")
    viz.compare_episodes(
        episodes,
        labels,
        save_path="difficulty_comparison.png",
        show=True
    )


def example_compare_command_types():
    """Example: Compare performance on different command types."""
    print("\n" + "="*60)
    print("Example 3: Compare Command Types")
    print("="*60)

    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    command_types = ["step", "ramp", "sine"]
    episodes = []
    labels = []

    for cmd_type in command_types:
        print(f"\nRunning episode with {cmd_type} commands...")
        data = viz.run_episode(
            difficulty="medium",
            command_type=cmd_type,
            seed=42
        )
        episodes.append(data)
        labels.append(cmd_type.capitalize())

    # Compare episodes
    print("\nGenerating comparison plot...")
    viz.compare_episodes(
        episodes,
        labels,
        save_path="command_type_comparison.png",
        show=True
    )


def example_learning_curves():
    """Example: Visualize learning curves from training."""
    print("\n" + "="*60)
    print("Example 4: Learning Curves")
    print("="*60)

    log_dir = "learned_controllers/logs/tensorboard"
    viz = LearningCurveVisualizer(log_dir)

    # Load data
    print("\nLoading TensorBoard data...")
    data = viz.load_tensorboard_data()

    # Plot reward curve
    print("Plotting reward curve...")
    viz.plot_reward_curve(
        data,
        smooth_window=10,
        save_path="reward_curve.png",
        show=True
    )

    # Plot training diagnostics
    print("Plotting training diagnostics...")
    viz.plot_training_diagnostics(
        data,
        smooth_window=10,
        save_path="training_diagnostics.png",
        show=True
    )

    # Get summary statistics
    print("\nSummary Statistics:")
    summary = viz.get_summary_statistics(data)
    print(summary[["mean", "std", "final"]].round(4))


def example_batch_evaluation():
    """Example: Batch evaluation of multiple seeds."""
    print("\n" + "="*60)
    print("Example 5: Batch Evaluation (Multiple Seeds)")
    print("="*60)

    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    n_seeds = 5
    episodes = []
    labels = []

    print(f"\nRunning {n_seeds} episodes with different seeds...")
    for i in range(n_seeds):
        data = viz.run_episode(
            difficulty="medium",
            command_type="step",
            seed=i
        )
        episodes.append(data)
        labels.append(f"Seed {i}")

        # Save episode data
        viz.save_episode_data(data, f"episode_seed_{i}.pkl")

    # Compare episodes
    print("\nGenerating comparison plot...")
    viz.compare_episodes(
        episodes,
        labels,
        save_path="seed_comparison.png",
        show=True
    )

    # Compute statistics
    import numpy as np
    rewards = [np.sum(ep["rewards"]) for ep in episodes]
    errors = [np.mean(np.linalg.norm(ep["errors"], axis=1)) for ep in episodes]

    print(f"\nStatistics across {n_seeds} seeds:")
    print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Error:  {np.mean(errors):.4f} ± {np.std(errors):.4f} rad/s")


def example_save_and_load():
    """Example: Save episode data and reload for analysis."""
    print("\n" + "="*60)
    print("Example 6: Save and Load Episode Data")
    print("="*60)

    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    # Run and save episode
    print("\nRunning episode...")
    data = viz.run_episode(difficulty="hard", seed=42)

    save_path = "my_episode_data.pkl"
    viz.save_episode_data(data, save_path)

    # Later: load and analyze
    print(f"\nLoading saved data from {save_path}...")
    loaded_data = viz.load_episode_data(save_path)

    # Plot the loaded data
    print("Plotting loaded data...")
    viz.plot_episode(
        loaded_data,
        title="Loaded Episode - Hard Difficulty",
        save_path="loaded_episode.png",
        show=True
    )


def example_custom_analysis():
    """Example: Custom analysis of episode data."""
    print("\n" + "="*60)
    print("Example 7: Custom Data Analysis")
    print("="*60)

    import numpy as np
    import matplotlib.pyplot as plt

    model_path = "learned_controllers/models/best_rate_controller/best_model.zip"
    viz = ReplayVisualizer(model_path)

    # Run episode
    print("\nRunning episode...")
    data = viz.run_episode(difficulty="medium", seed=42)

    # Custom analysis: Settling time for each axis
    print("\nAnalyzing settling time for each axis...")

    threshold = 0.05  # 5% settling threshold
    settling_times = []

    for axis_idx, axis_name in enumerate(["Roll (p)", "Pitch (q)", "Yaw (r)"]):
        errors = np.abs(data["errors"][:, axis_idx])
        commands = np.abs(data["commands"][:, axis_idx])

        # Find when error drops below threshold
        settled_mask = errors < np.maximum(threshold * commands, 0.05)

        # Find first time settled
        if np.any(settled_mask):
            first_settled = np.argmax(settled_mask)
            settling_time = data["time"][first_settled]
            settling_times.append(settling_time)
            print(f"  {axis_name}: {settling_time:.3f}s")
        else:
            print(f"  {axis_name}: Never settled")

    # Custom plot: Control surface usage distribution
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    surface_names = ["Aileron", "Elevator", "Rudder"]

    for idx, (ax, name) in enumerate(zip(axes, surface_names)):
        ax.hist(data["actions"][:, idx], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f"{name} Deflection")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} Usage Distribution")
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("control_surface_distribution.png", dpi=150)
    print("\nControl surface distribution saved to: control_surface_distribution.png")
    plt.show()


def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Run visualization examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 8),
        help="Run specific example (1-7), or run all if not specified"
    )

    args = parser.parse_args()

    examples = [
        example_replay_single_episode,
        example_compare_difficulties,
        example_compare_command_types,
        example_learning_curves,
        example_batch_evaluation,
        example_save_and_load,
        example_custom_analysis,
    ]

    if args.example:
        # Run specific example
        examples[args.example - 1]()
    else:
        # Run all examples
        print("\n" + "="*60)
        print("Running All Examples")
        print("="*60)
        print("\nNote: You can run individual examples with --example N")
        print("      where N is 1-7")

        for i, example in enumerate(examples, 1):
            try:
                example()
            except Exception as e:
                print(f"\nExample {i} failed: {e}")
                print("Continuing to next example...")

    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
