"""Evaluation metrics for rate control performance."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RateControlMetrics:
    """Container for rate control performance metrics."""

    # Settling time metrics
    settling_time_roll: float = 0.0  # seconds
    settling_time_pitch: float = 0.0
    settling_time_yaw: float = 0.0

    # Overshoot metrics (%)
    overshoot_roll: float = 0.0
    overshoot_pitch: float = 0.0
    overshoot_yaw: float = 0.0

    # Steady-state error (rad/s)
    steady_state_error_roll: float = 0.0
    steady_state_error_pitch: float = 0.0
    steady_state_error_yaw: float = 0.0

    # Rise time (seconds to 90% of command)
    rise_time_roll: float = 0.0
    rise_time_pitch: float = 0.0
    rise_time_yaw: float = 0.0

    # Control smoothness (mean absolute action change)
    control_smoothness: float = 0.0

    # Tracking RMSE over episode
    tracking_rmse: float = 0.0

    # Success flag (settled within time limit)
    success: bool = False

    # Episode statistics
    episode_length: float = 0.0  # seconds
    total_reward: float = 0.0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "settling_time_roll": self.settling_time_roll,
            "settling_time_pitch": self.settling_time_pitch,
            "settling_time_yaw": self.settling_time_yaw,
            "overshoot_roll": self.overshoot_roll,
            "overshoot_pitch": self.overshoot_pitch,
            "overshoot_yaw": self.overshoot_yaw,
            "steady_state_error_roll": self.steady_state_error_roll,
            "steady_state_error_pitch": self.steady_state_error_pitch,
            "steady_state_error_yaw": self.steady_state_error_yaw,
            "rise_time_roll": self.rise_time_roll,
            "rise_time_pitch": self.rise_time_pitch,
            "rise_time_yaw": self.rise_time_yaw,
            "control_smoothness": self.control_smoothness,
            "tracking_rmse": self.tracking_rmse,
            "success": self.success,
            "episode_length": self.episode_length,
            "total_reward": self.total_reward,
        }

    def print_summary(self, name: str = "Controller"):
        """Print formatted summary of metrics.

        Args:
            name: Controller name for display
        """
        print(f"\n{'='*60}")
        print(f"{name} Performance Metrics")
        print(f"{'='*60}")
        print(f"Settling Time (s):  Roll={self.settling_time_roll:.3f}, "
              f"Pitch={self.settling_time_pitch:.3f}, "
              f"Yaw={self.settling_time_yaw:.3f}")
        print(f"Overshoot (%):      Roll={self.overshoot_roll:.1f}, "
              f"Pitch={self.overshoot_pitch:.1f}, "
              f"Yaw={self.overshoot_yaw:.1f}")
        print(f"Rise Time (s):      Roll={self.rise_time_roll:.3f}, "
              f"Pitch={self.rise_time_pitch:.3f}, "
              f"Yaw={self.rise_time_yaw:.3f}")
        print(f"Steady-State Error: Roll={self.steady_state_error_roll:.4f}, "
              f"Pitch={self.steady_state_error_pitch:.4f}, "
              f"Yaw={self.steady_state_error_yaw:.4f} rad/s")
        print(f"Tracking RMSE:      {self.tracking_rmse:.4f} rad/s")
        print(f"Control Smoothness: {self.control_smoothness:.4f}")
        print(f"Success:            {self.success}")
        print(f"Total Reward:       {self.total_reward:.2f}")
        print(f"{'='*60}\n")


class MetricsCalculator:
    """Calculate performance metrics from episode data."""

    def __init__(
        self,
        settling_threshold: float = 0.05,  # 5% of commanded rate
        settling_duration: float = 0.2,    # Must stay settled for 0.2s
        dt: float = 0.02,                  # Timestep
    ):
        """Initialize metrics calculator.

        Args:
            settling_threshold: Threshold for "settled" state (fraction)
            settling_duration: Duration to stay settled (seconds)
            dt: Timestep (seconds)
        """
        self.settling_threshold = settling_threshold
        self.settling_duration = settling_duration
        self.dt = dt

    def compute_metrics(
        self,
        times: np.ndarray,
        rates: np.ndarray,  # [N, 3] - [p, q, r]
        commands: np.ndarray,  # [N, 3] - [p_cmd, q_cmd, r_cmd]
        actions: np.ndarray,  # [N, 4] - [aileron, elevator, rudder, throttle]
        rewards: np.ndarray,  # [N,] - total reward per step
    ) -> RateControlMetrics:
        """Compute all metrics from episode trajectory.

        Args:
            times: Time array [N,]
            rates: Measured rates [N, 3]
            commands: Commanded rates [N, 3]
            actions: Control actions [N, 4]
            rewards: Rewards [N,]

        Returns:
            RateControlMetrics object
        """
        metrics = RateControlMetrics()

        # Compute per-axis metrics
        for axis_idx, axis_name in enumerate(["roll", "pitch", "yaw"]):
            rate = rates[:, axis_idx]
            cmd = commands[:, axis_idx]
            error = cmd - rate

            # Skip if command is near zero throughout
            if np.max(np.abs(cmd)) < 0.01:
                continue

            # Settling time
            settling_time = self._compute_settling_time(times, error, cmd)
            setattr(metrics, f"settling_time_{axis_name}", settling_time)

            # Overshoot
            overshoot = self._compute_overshoot(rate, cmd)
            setattr(metrics, f"overshoot_{axis_name}", overshoot)

            # Rise time
            rise_time = self._compute_rise_time(times, rate, cmd)
            setattr(metrics, f"rise_time_{axis_name}", rise_time)

            # Steady-state error
            ss_error = self._compute_steady_state_error(error, settling_time, times)
            setattr(metrics, f"steady_state_error_{axis_name}", ss_error)

        # Control smoothness
        metrics.control_smoothness = self._compute_smoothness(actions)

        # Tracking RMSE
        metrics.tracking_rmse = self._compute_tracking_rmse(rates, commands)

        # Success (all axes settled within episode)
        metrics.success = (
            metrics.settling_time_roll < times[-1] and
            metrics.settling_time_pitch < times[-1] and
            metrics.settling_time_yaw < times[-1]
        )

        # Episode statistics
        metrics.episode_length = times[-1]
        metrics.total_reward = np.sum(rewards)

        return metrics

    def _compute_settling_time(
        self,
        times: np.ndarray,
        error: np.ndarray,
        command: np.ndarray,
    ) -> float:
        """Compute settling time.

        Settling time is when error stays within threshold for settling_duration.

        Args:
            times: Time array
            error: Error signal
            command: Command signal

        Returns:
            Settling time in seconds (or episode length if never settles)
        """
        abs_threshold = 0.05  # Absolute threshold for near-zero commands (rad/s)

        # Compute threshold (adaptive based on command magnitude)
        max_cmd = np.max(np.abs(command))
        threshold = max(self.settling_threshold * max_cmd, abs_threshold)

        # Check if within threshold
        within_threshold = np.abs(error) < threshold

        # Find first time that stays within threshold
        settle_steps = int(self.settling_duration / self.dt)

        for i in range(len(error) - settle_steps):
            if np.all(within_threshold[i:i + settle_steps]):
                return times[i]

        # Never settled
        return times[-1]

    def _compute_overshoot(
        self,
        rate: np.ndarray,
        command: np.ndarray,
    ) -> float:
        """Compute overshoot percentage.

        Args:
            rate: Measured rate
            command: Commanded rate

        Returns:
            Overshoot percentage
        """
        # Find peak overshoot
        # Overshoot is when rate exceeds command in the direction of command
        error = rate - command

        # Direction of command
        cmd_sign = np.sign(command[len(command)//2])  # Take middle value

        if cmd_sign == 0:
            return 0.0

        # Overshoot in direction of command
        overshoot_mask = cmd_sign * error > 0
        if not np.any(overshoot_mask):
            return 0.0

        max_overshoot = np.max(np.abs(error[overshoot_mask]))

        # Command magnitude
        max_cmd = np.max(np.abs(command))

        if max_cmd < 0.01:
            return 0.0

        overshoot_pct = (max_overshoot / max_cmd) * 100.0

        return overshoot_pct

    def _compute_rise_time(
        self,
        times: np.ndarray,
        rate: np.ndarray,
        command: np.ndarray,
    ) -> float:
        """Compute rise time (time to reach 90% of command).

        Args:
            times: Time array
            rate: Measured rate
            command: Commanded rate

        Returns:
            Rise time in seconds
        """
        # Find steady-state command value (average of last 20%)
        n_ss = max(1, len(command) // 5)
        cmd_ss = np.mean(command[-n_ss:])

        if abs(cmd_ss) < 0.01:
            return 0.0

        # 90% threshold
        threshold = 0.9 * cmd_ss

        # Find first crossing
        if cmd_ss > 0:
            crossing = np.where(rate >= threshold)[0]
        else:
            crossing = np.where(rate <= threshold)[0]

        if len(crossing) == 0:
            return times[-1]

        return times[crossing[0]]

    def _compute_steady_state_error(
        self,
        error: np.ndarray,
        settling_time: float,
        times: np.ndarray,
    ) -> float:
        """Compute steady-state error after settling.

        Args:
            error: Error signal
            settling_time: Settling time
            times: Time array

        Returns:
            Mean absolute steady-state error
        """
        # Find index after settling time
        settle_idx = np.searchsorted(times, settling_time)

        if settle_idx >= len(error) - 1:
            # Never settled
            return np.mean(np.abs(error))

        # Mean absolute error after settling
        ss_error = np.mean(np.abs(error[settle_idx:]))

        return ss_error

    def _compute_smoothness(self, actions: np.ndarray) -> float:
        """Compute control smoothness (mean absolute action change).

        Args:
            actions: Control actions [N, 4]

        Returns:
            Mean absolute action change
        """
        # Only consider control surfaces (not throttle)
        surfaces = actions[:, :3]

        # Action changes
        action_diff = np.diff(surfaces, axis=0)

        # Mean absolute change
        smoothness = np.mean(np.abs(action_diff))

        return smoothness

    def _compute_tracking_rmse(
        self,
        rates: np.ndarray,
        commands: np.ndarray,
    ) -> float:
        """Compute tracking RMSE over entire episode.

        Args:
            rates: Measured rates [N, 3]
            commands: Commanded rates [N, 3]

        Returns:
            RMSE (rad/s)
        """
        errors = commands - rates
        rmse = np.sqrt(np.mean(errors ** 2))
        return rmse


def compare_metrics(
    metrics_a: RateControlMetrics,
    metrics_b: RateControlMetrics,
    name_a: str = "Controller A",
    name_b: str = "Controller B",
):
    """Compare two sets of metrics and print comparison.

    Args:
        metrics_a: First controller metrics
        metrics_b: Second controller metrics
        name_a: Name of first controller
        name_b: Name of second controller
    """
    print(f"\n{'='*60}")
    print(f"Performance Comparison: {name_a} vs {name_b}")
    print(f"{'='*60}")

    def pct_diff(a, b):
        """Compute percentage difference."""
        if b == 0:
            return 0.0
        return ((a - b) / b) * 100.0

    # Settling time comparison
    print("\nSettling Time (s):")
    print(f"  Roll:  {metrics_a.settling_time_roll:.3f} vs "
          f"{metrics_b.settling_time_roll:.3f} "
          f"({pct_diff(metrics_a.settling_time_roll, metrics_b.settling_time_roll):+.1f}%)")
    print(f"  Pitch: {metrics_a.settling_time_pitch:.3f} vs "
          f"{metrics_b.settling_time_pitch:.3f} "
          f"({pct_diff(metrics_a.settling_time_pitch, metrics_b.settling_time_pitch):+.1f}%)")
    print(f"  Yaw:   {metrics_a.settling_time_yaw:.3f} vs "
          f"{metrics_b.settling_time_yaw:.3f} "
          f"({pct_diff(metrics_a.settling_time_yaw, metrics_b.settling_time_yaw):+.1f}%)")

    # Overshoot comparison
    print("\nOvershoot (%):")
    print(f"  Roll:  {metrics_a.overshoot_roll:.1f} vs "
          f"{metrics_b.overshoot_roll:.1f}")
    print(f"  Pitch: {metrics_a.overshoot_pitch:.1f} vs "
          f"{metrics_b.overshoot_pitch:.1f}")
    print(f"  Yaw:   {metrics_a.overshoot_yaw:.1f} vs "
          f"{metrics_b.overshoot_yaw:.1f}")

    # Tracking RMSE comparison
    print(f"\nTracking RMSE (rad/s):")
    print(f"  {metrics_a.tracking_rmse:.4f} vs {metrics_b.tracking_rmse:.4f} "
          f"({pct_diff(metrics_a.tracking_rmse, metrics_b.tracking_rmse):+.1f}%)")

    # Control smoothness comparison
    print(f"\nControl Smoothness:")
    print(f"  {metrics_a.control_smoothness:.4f} vs {metrics_b.control_smoothness:.4f} "
          f"({pct_diff(metrics_a.control_smoothness, metrics_b.control_smoothness):+.1f}%)")

    # Success rate comparison
    print(f"\nSuccess:")
    print(f"  {name_a}: {metrics_a.success}")
    print(f"  {name_b}: {metrics_b.success}")

    print(f"{'='*60}\n")
