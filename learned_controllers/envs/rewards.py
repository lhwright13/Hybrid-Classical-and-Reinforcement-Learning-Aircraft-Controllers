"""Reward functions for rate control training."""

import numpy as np
from typing import Dict, Tuple


class RateTrackingReward:
    """Reward function for angular rate tracking.

    Multi-component reward designed to encourage:
    1. Tight rate tracking (primary objective)
    2. Control smoothness (avoid chattering)
    3. Stability within safe flight envelope
    4. Damped responses (avoid excessive oscillation)
    """

    def __init__(
        self,
        w_tracking: float = 0.5,   # Reduced from 1.0 - less harsh on tracking errors
        w_smoothness: float = 0.01,
        w_stability: float = 0.3,  # Increased - reward staying upright
        w_oscillation: float = 0.1,  # Reduced - don't over-penalize oscillation early
        w_survival: float = 1.0,   # Big survival bonus - staying alive is priority
        settling_threshold: float = 0.1,  # Relaxed from 0.05
    ):
        """Initialize reward function.

        Args:
            w_tracking: Weight for rate tracking error
            w_smoothness: Weight for control smoothness penalty
            w_stability: Weight for stability bonus
            w_oscillation: Weight for oscillation penalty
            w_survival: Weight for survival/time bonus (prevents crash exploit)
            settling_threshold: Threshold for "settled" state (fraction)
        """
        self.w_tracking = w_tracking
        self.w_smoothness = w_smoothness
        self.w_stability = w_stability
        self.w_oscillation = w_oscillation
        self.w_survival = w_survival
        self.settling_threshold = settling_threshold

        # For oscillation detection (pre-allocated buffers for hot-path)
        self.prev_errors = np.zeros(3)
        self.sign_changes = np.zeros(3)
        self._errors_buf = np.zeros(3)

    def compute(
        self,
        p_error: float,
        q_error: float,
        r_error: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        airspeed: float,
        altitude: float,
        roll: float,
        pitch: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward for current timestep.

        Args:
            p_error: Roll rate error (rad/s)
            q_error: Pitch rate error (rad/s)
            r_error: Yaw rate error (rad/s)
            action: Current control action [aileron, elevator, rudder, throttle]
            prev_action: Previous control action
            airspeed: Current airspeed (m/s)
            altitude: Current altitude (m)
            roll: Current roll angle (rad)
            pitch: Current pitch angle (rad)

        Returns:
            Tuple of (total_reward, reward_components_dict)
        """
        # 1. Rate tracking error (primary objective)
        # Use MSE for smooth gradients
        tracking_error = (p_error**2 + q_error**2 + r_error**2) / 3.0
        r_tracking = -self.w_tracking * tracking_error

        # 2. Control smoothness (penalize large action changes)
        action_diff = action - prev_action
        control_change = np.sum(action_diff[:3]**2)  # Only surfaces, not throttle
        r_smoothness = -self.w_smoothness * control_change

        # 3. Stability bonus (continuous reward for staying in safe envelope)
        # Use smooth functions instead of binary thresholds
        # Roll stability: max reward at 0, decreases as roll increases
        roll_stability = np.exp(-abs(roll) / np.radians(45))  # Gaussian-like decay
        # Pitch stability: max at 0, decreases as pitch deviates
        pitch_stability = np.exp(-abs(pitch) / np.radians(30))
        # Airspeed stability: reward being above stall
        airspeed_stability = np.clip((airspeed - 8.0) / 12.0, 0, 1)  # 0 at 8m/s, 1 at 20m/s
        # Altitude stability
        altitude_stability = np.clip((altitude - 10.0) / 90.0, 0, 1)  # 0 at 10m, 1 at 100m

        stability_score = (
            roll_stability + pitch_stability + airspeed_stability + altitude_stability
        ) / 4.0
        r_stability = self.w_stability * stability_score

        # 4. Oscillation penalty (detect sign changes in error)
        self._errors_buf[0] = p_error
        self._errors_buf[1] = q_error
        self._errors_buf[2] = r_error

        # Detect sign changes (indicates oscillation around setpoint)
        sign_changes = (np.sign(self._errors_buf) != np.sign(self.prev_errors)) & (
            np.abs(self.prev_errors) > 0.01
        )
        self.sign_changes = 0.9 * self.sign_changes + sign_changes.astype(float)

        # Penalize rapid oscillations
        oscillation_penalty = np.sum(self.sign_changes)
        r_oscillation = -self.w_oscillation * oscillation_penalty

        np.copyto(self.prev_errors, self._errors_buf)

        # 5. Survival bonus (NEW: reward for staying alive)
        # This prevents the exploit where agent crashes early to minimize negative reward
        r_survival = self.w_survival

        # Total reward
        total_reward = r_tracking + r_smoothness + r_stability + r_oscillation + r_survival

        # Component breakdown for logging
        components = {
            "tracking": r_tracking,
            "smoothness": r_smoothness,
            "stability": r_stability,
            "oscillation": r_oscillation,
            "survival": r_survival,
            "total": total_reward,
            "tracking_error_mse": tracking_error,
        }

        return total_reward, components

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_errors = np.zeros(3)
        self.sign_changes = np.zeros(3)


class SettlingTimeBonus:
    """Bonus reward for achieving and maintaining settled state."""

    def __init__(
        self,
        settling_threshold: float = 0.05,  # 5% of commanded rate
        min_settle_time: float = 0.2,      # Must stay settled for 0.2s
        bonus_multiplier: float = 2.0,
    ):
        """Initialize settling time bonus.

        Args:
            settling_threshold: Error threshold for "settled" (fraction)
            min_settle_time: Minimum time to stay settled (seconds)
            bonus_multiplier: Multiplier for bonus reward
        """
        self.settling_threshold = settling_threshold
        self.min_settle_time = min_settle_time
        self.bonus_multiplier = bonus_multiplier

        self.settle_timer = 0.0
        self.is_settled = False

    def compute(
        self,
        p_error: float,
        q_error: float,
        r_error: float,
        p_cmd: float,
        q_cmd: float,
        r_cmd: float,
        dt: float,
    ) -> float:
        """Compute settling bonus.

        Args:
            p_error: Roll rate error (rad/s)
            q_error: Pitch rate error (rad/s)
            r_error: Yaw rate error (rad/s)
            p_cmd: Commanded roll rate (rad/s)
            q_cmd: Commanded pitch rate (rad/s)
            r_cmd: Commanded yaw rate (rad/s)
            dt: Timestep (seconds)

        Returns:
            Bonus reward
        """
        # Check if all axes are within settling threshold
        # Use relative error to handle different command magnitudes
        abs_threshold = 0.05  # Absolute threshold for near-zero commands (rad/s)

        p_settled = (
            abs(p_error) < max(abs(p_cmd) * self.settling_threshold, abs_threshold)
        )
        q_settled = (
            abs(q_error) < max(abs(q_cmd) * self.settling_threshold, abs_threshold)
        )
        r_settled = (
            abs(r_error) < max(abs(r_cmd) * self.settling_threshold, abs_threshold)
        )

        currently_settled = p_settled and q_settled and r_settled

        if currently_settled:
            self.settle_timer += dt

            # Give bonus if settled for minimum time
            if self.settle_timer >= self.min_settle_time:
                self.is_settled = True
                # Continuous bonus for staying settled
                return self.bonus_multiplier * dt
        else:
            # Reset timer if we leave settled state
            self.settle_timer = 0.0
            self.is_settled = False

        return 0.0

    def reset(self):
        """Reset internal state for new episode."""
        self.settle_timer = 0.0
        self.is_settled = False
