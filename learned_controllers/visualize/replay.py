#!/usr/bin/env python3
"""Replay and dynamics visualization for learned controllers."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict
import pickle

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from learned_controllers.envs.rate_env import RateControlEnv


class ReplayVisualizer:
    """Visualize episode replays with detailed dynamics plots."""

    def __init__(self, model_path: str, figsize: tuple = (16, 12)):
        """Initialize replay visualizer.

        Args:
            model_path: Path to trained model
            figsize: Figure size for plots
        """
        self.model_path = model_path
        self.figsize = figsize

        # Load model
        try:
            self.model = RecurrentPPO.load(model_path)
            self.is_recurrent = True
            print(f"Loaded RecurrentPPO model from {model_path}")
        except:
            self.model = PPO.load(model_path)
            self.is_recurrent = False
            print(f"Loaded PPO model from {model_path}")

    def run_episode(
        self,
        difficulty: str = "medium",
        episode_length: float = 10.0,
        command_type: str = "step",
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run a single episode and collect data.

        Args:
            difficulty: Environment difficulty
            episode_length: Episode duration in seconds
            command_type: Type of rate command
            deterministic: Use deterministic policy
            seed: Random seed

        Returns:
            Dictionary with episode data
        """
        # Create environment
        env = RateControlEnv(
            difficulty=difficulty,
            episode_length=episode_length,
            dt=0.02,
            command_type=command_type,
            rng_seed=seed,
        )

        # Reset environment
        obs, info = env.reset(seed=seed)
        done = False

        # Data collection
        data = {
            "time": [],
            "rates": [],  # [p, q, r]
            "commands": [],  # [p_cmd, q_cmd, r_cmd]
            "errors": [],  # [p_err, q_err, r_err]
            "actions": [],  # [aileron, elevator, rudder, throttle]
            "attitudes": [],  # [roll, pitch, yaw]
            "airspeed": [],
            "altitude": [],
            "rewards": [],
            "reward_components": [],
        }

        # Reset LSTM state if RecurrentPPO
        lstm_states = None if self.is_recurrent else None

        step = 0
        while not done:
            # Get action from model
            if self.is_recurrent:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    deterministic=deterministic,
                )
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Collect data
            state = env.sim.get_state()
            data["time"].append(info["time"])
            data["rates"].append([state.p, state.q, state.r])
            data["commands"].append(info["rate_command"])
            data["errors"].append(info["rate_error"])
            data["actions"].append(action)
            data["attitudes"].append([state.roll, state.pitch, state.yaw])
            data["airspeed"].append(state.airspeed)
            data["altitude"].append(state.altitude)
            data["rewards"].append(reward)

            if "reward_components" in info:
                data["reward_components"].append(info["reward_components"])

            step += 1

        env.close()

        # Convert to numpy arrays
        for key in data:
            if len(data[key]) > 0:
                data[key] = np.array(data[key])

        return data

    def plot_episode(
        self,
        data: Dict[str, np.ndarray],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot detailed dynamics for an episode.

        Args:
            data: Episode data from run_episode()
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        time = data["time"]

        # 1. Rate Tracking (3 subplots)
        for i, (axis_name, idx) in enumerate([("Roll Rate (p)", 0), ("Pitch Rate (q)", 1), ("Yaw Rate (r)", 2)]):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(time, data["rates"][:, idx], label="Actual", linewidth=2)
            ax.plot(time, data["commands"][:, idx], label="Command", linestyle="--", linewidth=2)
            ax.fill_between(
                time,
                data["commands"][:, idx] - 0.05,
                data["commands"][:, idx] + 0.05,
                alpha=0.2,
                label="±5% settling band"
            )
            ax.set_ylabel("Rate (rad/s)")
            ax.set_xlabel("Time (s)")
            ax.set_title(axis_name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # 2. Control Actions (4 subplots in row 2)
        action_names = ["Aileron", "Elevator", "Rudder", "Throttle"]
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(time, data["actions"][:, i], linewidth=2)
            ax.set_ylabel("Command")
            ax.set_xlabel("Time (s)")
            ax.set_title(action_names[i])
            ax.set_ylim([-1.1, 1.1])
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)

        # 3. Rate Errors
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(time, data["errors"][:, 0], label="p error", linewidth=2)
        ax.plot(time, data["errors"][:, 1], label="q error", linewidth=2)
        ax.plot(time, data["errors"][:, 2], label="r error", linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel("Error (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Rate Tracking Errors")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Attitudes
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(time, np.rad2deg(data["attitudes"][:, 0]), label="Roll", linewidth=2)
        ax.plot(time, np.rad2deg(data["attitudes"][:, 1]), label="Pitch", linewidth=2)
        ax.plot(time, np.rad2deg(data["attitudes"][:, 2]), label="Yaw", linewidth=2)
        ax.set_ylabel("Angle (deg)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Aircraft Attitudes")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5. Flight Envelope
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(time, data["airspeed"], label="Airspeed", linewidth=2, color='blue')
        ax.set_ylabel("Airspeed (m/s)", color='blue')
        ax.set_xlabel("Time (s)")
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(time, data["altitude"], label="Altitude", linewidth=2, color='red')
        ax2.set_ylabel("Altitude (m)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title("Flight Envelope")

        # 6. Rewards
        ax = fig.add_subplot(gs[3, 0])
        cumulative_reward = np.cumsum(data["rewards"])
        ax.plot(time, cumulative_reward, linewidth=2, color='green')
        ax.set_ylabel("Cumulative Reward")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Total Reward: {cumulative_reward[-1]:.2f}")
        ax.grid(True, alpha=0.3)

        # 7. Reward Components (if available)
        if len(data["reward_components"]) > 0 and len(data["reward_components"][0]) > 0:
            ax = fig.add_subplot(gs[3, 1])
            components = data["reward_components"]

            # Extract component names
            comp_keys = ["tracking", "smoothness", "stability", "oscillation"]
            for key in comp_keys:
                values = [comp.get(key, 0) for comp in components]
                ax.plot(time, np.cumsum(values), label=key, linewidth=2)

            ax.set_ylabel("Cumulative Reward")
            ax.set_xlabel("Time (s)")
            ax.set_title("Reward Components")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        # 8. Error Magnitude
        ax = fig.add_subplot(gs[3, 2])
        error_mag = np.linalg.norm(data["errors"], axis=1)
        ax.plot(time, error_mag, linewidth=2, color='red')
        ax.fill_between(time, 0, error_mag, alpha=0.3, color='red')
        ax.set_ylabel("Error Magnitude (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Overall Tracking Error")
        ax.grid(True, alpha=0.3)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f"Episode Replay - Model: {Path(self.model_path).name}", fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        if show:
            plt.show()

        return fig

    def save_episode_data(self, data: Dict[str, np.ndarray], filepath: str):
        """Save episode data to file.

        Args:
            data: Episode data
            filepath: Path to save file (.pkl)
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Episode data saved to {filepath}")

    def load_episode_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load episode data from file.

        Args:
            filepath: Path to data file (.pkl)

        Returns:
            Episode data dictionary
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Episode data loaded from {filepath}")
        return data

    def compare_episodes(
        self,
        episodes: List[Dict[str, np.ndarray]],
        labels: List[str],
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Compare multiple episodes side-by-side.

        Args:
            episodes: List of episode data dictionaries
            labels: Labels for each episode
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. Rate tracking comparison
        for i, (data, label) in enumerate(zip(episodes, labels)):
            time = data["time"]
            error_mag = np.linalg.norm(data["errors"], axis=1)
            axes[0].plot(time, error_mag, label=label, linewidth=2, alpha=0.8)

        axes[0].set_ylabel("Error Magnitude (rad/s)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title("Rate Tracking Error Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Cumulative rewards
        for i, (data, label) in enumerate(zip(episodes, labels)):
            time = data["time"]
            cumulative_reward = np.cumsum(data["rewards"])
            axes[1].plot(time, cumulative_reward, label=f"{label}: {cumulative_reward[-1]:.2f}", linewidth=2, alpha=0.8)

        axes[1].set_ylabel("Cumulative Reward")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Reward Accumulation Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Control effort comparison
        for i, (data, label) in enumerate(zip(episodes, labels)):
            time = data["time"]
            # Control effort as sum of absolute control surface deflections
            control_effort = np.sum(np.abs(data["actions"][:, :3]), axis=1)
            axes[2].plot(time, control_effort, label=label, linewidth=2, alpha=0.8)

        axes[2].set_ylabel("Control Effort")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Control Effort Comparison")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")

        if show:
            plt.show()

        return fig


def main():
    """Example usage of replay visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize learned controller episode replay")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Environment difficulty",
    )
    parser.add_argument(
        "--command-type",
        type=str,
        default="step",
        choices=["step", "ramp", "sine", "random"],
        help="Command type",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="learned_controllers/visualize/replays",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = ReplayVisualizer(args.model)

    # Run and visualize episodes
    for i in range(args.n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{args.n_episodes}")
        print(f"{'='*60}")

        # Run episode
        seed = args.seed + i if args.seed is not None else None
        data = viz.run_episode(
            difficulty=args.difficulty,
            command_type=args.command_type,
            seed=seed,
        )

        # Save data
        data_path = f"{args.save_dir}/episode_{i+1}_data.pkl"
        viz.save_episode_data(data, data_path)

        # Plot episode
        title = f"Episode {i+1} - {args.difficulty.capitalize()} - {args.command_type.capitalize()}"
        save_path = f"{args.save_dir}/episode_{i+1}_dynamics.png"
        viz.plot_episode(
            data,
            title=title,
            save_path=save_path,
            show=False,  # Don't show during batch processing
        )

        # Print summary
        total_reward = np.sum(data["rewards"])
        mean_error = np.mean(np.linalg.norm(data["errors"], axis=1))
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Mean Error: {mean_error:.4f} rad/s")

    print(f"\n{'='*60}")
    print(f"Visualizations saved to {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
