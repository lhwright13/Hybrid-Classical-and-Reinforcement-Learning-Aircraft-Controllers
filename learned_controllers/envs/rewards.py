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
        w_tracking: float = 1.0,
        w_smoothness: float = 0.01,
        w_stability: float = 0.1,
        w_oscillation: float = 0.5,
        w_survival: float = 0.1,  # NEW: Reward for each timestep survived
        settling_threshold: float = 0.05,  # 5% of commanded rate
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

        # For oscillation detection
        self.prev_errors = np.zeros(3)
        self.sign_changes = np.zeros(3)

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

        # 3. Stability bonus (reward staying in safe envelope)
        # Penalize excessive attitudes and low airspeed
        safe_roll = abs(roll) < np.radians(60)
        safe_pitch = abs(pitch) < np.radians(30)
        safe_airspeed = airspeed > 12.0  # Above stall speed
        safe_altitude = altitude > 30.0   # Safe margin above ground

        stability_score = (
            safe_roll + safe_pitch + safe_airspeed + safe_altitude
        ) / 4.0
        r_stability = self.w_stability * stability_score

        # 4. Oscillation penalty (detect sign changes in error)
        errors = np.array([p_error, q_error, r_error])

        # Detect sign changes (indicates oscillation around setpoint)
        sign_changes = (np.sign(errors) != np.sign(self.prev_errors)) & (
            np.abs(self.prev_errors) > 0.01
        )
        self.sign_changes = 0.9 * self.sign_changes + sign_changes.astype(float)

        # Penalize rapid oscillations
        oscillation_penalty = np.sum(self.sign_changes)
        r_oscillation = -self.w_oscillation * oscillation_penalty

        self.prev_errors = errors

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
