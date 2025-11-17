#!/usr/bin/env python3
"""Learning curve visualization from TensorBoard logs."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class LearningCurveVisualizer:
    """Visualize learning curves from TensorBoard logs."""

    def __init__(self, log_dir: str):
        """Initialize learning curve visualizer.

        Args:
            log_dir: Path to TensorBoard log directory
        """
        self.log_dir = Path(log_dir)

        if not self.log_dir.exists():
            raise ValueError(f"Log directory does not exist: {log_dir}")

        print(f"Loading logs from: {log_dir}")

    def load_tensorboard_data(self, subfolder: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load data from TensorBoard event files.

        Args:
            subfolder: Optional subfolder within log_dir to load from

        Returns:
            Dictionary mapping metric names to DataFrames with columns [step, value, wall_time]
        """
        search_dir = self.log_dir / subfolder if subfolder else self.log_dir

        # Find all event files
        event_files = list(search_dir.rglob("events.out.tfevents.*"))

        if len(event_files) == 0:
            raise ValueError(f"No TensorBoard event files found in {search_dir}")

        print(f"Found {len(event_files)} event file(s)")

        # Load data from all event files
        all_data = {}

        for event_file in event_files:
            print(f"Loading: {event_file.name}")
            ea = EventAccumulator(str(event_file))
            ea.Reload()

            # Get all scalar tags
            tags = ea.Tags()["scalars"]

            for tag in tags:
                events = ea.Scalars(tag)

                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        "step": e.step,
                        "value": e.value,
                        "wall_time": e.wall_time,
                    }
                    for e in events
                ])

                # Merge with existing data for this tag
                if tag in all_data:
                    all_data[tag] = pd.concat([all_data[tag], df]).sort_values("step")
                else:
                    all_data[tag] = df

        print(f"Loaded {len(all_data)} metric(s)")
        return all_data

    def plot_learning_curves(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        metrics: Optional[List[str]] = None,
        smooth_window: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 10),
    ):
        """Plot learning curves for specified metrics.

        Args:
            data: Pre-loaded TensorBoard data (if None, will load from log_dir)
            metrics: List of metric names to plot (if None, plot common metrics)
            smooth_window: Window size for smoothing curves
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        # Load data if not provided
        if data is None:
            data = self.load_tensorboard_data()

        # Default metrics to plot
        if metrics is None:
            # Common RL metrics
            common_metrics = [
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "train/value_loss",
                "train/policy_gradient_loss",
                "train/entropy_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
            ]
            # Filter to available metrics
            metrics = [m for m in common_metrics if m in data]

        if len(metrics) == 0:
            print("No metrics found to plot!")
            print(f"Available metrics: {list(data.keys())}")
            return None

        # Create figure
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            if metric not in data:
                ax.text(0.5, 0.5, f"Metric not found:\n{metric}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric)
                continue

            df = data[metric]

            # Plot raw data
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1, label="Raw")

            # Plot smoothed data
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2, label=f"Smoothed (window={smooth_window})")

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Value")
            ax.set_title(metric)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Learning Curves - {self.log_dir.name}", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_reward_curve(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        smooth_window: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Plot detailed reward curve with confidence intervals.

        Args:
            data: Pre-loaded TensorBoard data
            smooth_window: Window size for smoothing
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        if data is None:
            data = self.load_tensorboard_data()

        # Find reward metrics
        reward_metric = None
        for key in ["rollout/ep_rew_mean", "eval/mean_reward", "reward"]:
            if key in data:
                reward_metric = key
                break

        if reward_metric is None:
            print("No reward metric found in data!")
            return None

        df = data[reward_metric]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot raw data
        ax.plot(df["step"], df["value"], alpha=0.2, linewidth=1, color='blue', label="Raw")

        # Plot smoothed curve
        if len(df) > smooth_window:
            smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
            std = df["value"].rolling(window=smooth_window, center=True).std()

            ax.plot(df["step"], smoothed, linewidth=3, color='blue', label=f"Mean (window={smooth_window})")

            # Confidence interval
            ax.fill_between(
                df["step"],
                smoothed - std,
                smoothed + std,
                alpha=0.3,
                color='blue',
                label="±1 std"
            )

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title(f"Learning Progress - {reward_metric}", fontsize=14, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Reward curve saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_training_diagnostics(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        smooth_window: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 12),
    ):
        """Plot comprehensive training diagnostics.

        Args:
            data: Pre-loaded TensorBoard data
            smooth_window: Window size for smoothing
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        if data is None:
            data = self.load_tensorboard_data()

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Episode Reward
        ax = fig.add_subplot(gs[0, :2])
        if "rollout/ep_rew_mean" in data:
            df = data["rollout/ep_rew_mean"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Mean Episode Reward")
            ax.set_xlabel("Steps")
            ax.set_title("Episode Reward")
            ax.grid(True, alpha=0.3)

        # 2. Episode Length
        ax = fig.add_subplot(gs[0, 2])
        if "rollout/ep_len_mean" in data:
            df = data["rollout/ep_len_mean"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Mean Episode Length")
            ax.set_xlabel("Steps")
            ax.set_title("Episode Length")
            ax.grid(True, alpha=0.3)

        # 3. Value Loss
        ax = fig.add_subplot(gs[1, 0])
        if "train/value_loss" in data:
            df = data["train/value_loss"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Steps")
            ax.set_title("Value Loss")
            ax.grid(True, alpha=0.3)

        # 4. Policy Loss
        ax = fig.add_subplot(gs[1, 1])
        if "train/policy_gradient_loss" in data:
            df = data["train/policy_gradient_loss"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Steps")
            ax.set_title("Policy Gradient Loss")
            ax.grid(True, alpha=0.3)

        # 5. Entropy
        ax = fig.add_subplot(gs[1, 2])
        if "train/entropy_loss" in data:
            df = data["train/entropy_loss"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Entropy")
            ax.set_xlabel("Steps")
            ax.set_title("Entropy Loss")
            ax.grid(True, alpha=0.3)

        # 6. KL Divergence
        ax = fig.add_subplot(gs[2, 0])
        if "train/approx_kl" in data:
            df = data["train/approx_kl"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("KL Divergence")
            ax.set_xlabel("Steps")
            ax.set_title("Approximate KL Divergence")
            ax.grid(True, alpha=0.3)

        # 7. Clip Fraction
        ax = fig.add_subplot(gs[2, 1])
        if "train/clip_fraction" in data:
            df = data["train/clip_fraction"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.set_ylabel("Clip Fraction")
            ax.set_xlabel("Steps")
            ax.set_title("PPO Clip Fraction")
            ax.grid(True, alpha=0.3)

        # 8. Explained Variance
        ax = fig.add_subplot(gs[2, 2])
        if "train/explained_variance" in data:
            df = data["train/explained_variance"]
            ax.plot(df["step"], df["value"], alpha=0.3, linewidth=1)
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2)
            ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label="Perfect")
            ax.set_ylabel("Explained Variance")
            ax.set_xlabel("Steps")
            ax.set_title("Explained Variance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Training Diagnostics", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training diagnostics saved to {save_path}")

        if show:
            plt.show()

        return fig

    def compare_runs(
        self,
        run_dirs: List[str],
        run_labels: List[str],
        metric: str = "rollout/ep_rew_mean",
        smooth_window: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Compare learning curves from multiple training runs.

        Args:
            run_dirs: List of TensorBoard log directories
            run_labels: Labels for each run
            metric: Metric to compare
            smooth_window: Smoothing window size
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))

        for run_dir, label, color in zip(run_dirs, run_labels, colors):
            # Load data for this run
            viz = LearningCurveVisualizer(run_dir)
            data = viz.load_tensorboard_data()

            if metric not in data:
                print(f"Warning: {metric} not found in {run_dir}")
                continue

            df = data[metric]

            # Plot raw data
            ax.plot(df["step"], df["value"], alpha=0.2, linewidth=1, color=color)

            # Plot smoothed curve
            if len(df) > smooth_window:
                smoothed = df["value"].rolling(window=smooth_window, center=True).mean()
                ax.plot(df["step"], smoothed, linewidth=2, color=color, label=label)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Comparison: {metric}", fontsize=14, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def get_summary_statistics(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Get summary statistics for all metrics.

        Args:
            data: Pre-loaded TensorBoard data

        Returns:
            DataFrame with summary statistics
        """
        if data is None:
            data = self.load_tensorboard_data()

        summary = {}

        for metric, df in data.items():
            summary[metric] = {
                "mean": df["value"].mean(),
                "std": df["value"].std(),
                "min": df["value"].min(),
                "max": df["value"].max(),
                "final": df["value"].iloc[-1] if len(df) > 0 else np.nan,
                "n_samples": len(df),
            }

        return pd.DataFrame(summary).T


def main():
    """Example usage of learning curve visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize learning curves from TensorBoard logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="learned_controllers/logs/tensorboard",
        help="Path to TensorBoard log directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="learned_controllers/visualize/learning_curves",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Smoothing window size",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save)",
    )

    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = LearningCurveVisualizer(args.log_dir)

    # Load data
    print("\nLoading TensorBoard data...")
    data = viz.load_tensorboard_data()

    print("\nAvailable metrics:")
    for i, metric in enumerate(data.keys(), 1):
        print(f"  {i}. {metric}")

    # Plot all learning curves
    print("\nGenerating learning curves...")
    viz.plot_learning_curves(
        data=data,
        smooth_window=args.smooth_window,
        save_path=f"{args.save_dir}/all_metrics.png",
        show=not args.no_show,
    )

    # Plot reward curve
    print("\nGenerating reward curve...")
    viz.plot_reward_curve(
        data=data,
        smooth_window=args.smooth_window,
        save_path=f"{args.save_dir}/reward_curve.png",
        show=not args.no_show,
    )

    # Plot training diagnostics
    print("\nGenerating training diagnostics...")
    viz.plot_training_diagnostics(
        data=data,
        smooth_window=args.smooth_window,
        save_path=f"{args.save_dir}/training_diagnostics.png",
        show=not args.no_show,
    )

    # Get summary statistics
    print("\nSummary Statistics:")
    summary = viz.get_summary_statistics(data)
    print(summary.to_string())

    # Save summary to CSV
    summary_path = f"{args.save_dir}/summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to {summary_path}")

    print(f"\n{'='*60}")
    print(f"Visualizations saved to {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
