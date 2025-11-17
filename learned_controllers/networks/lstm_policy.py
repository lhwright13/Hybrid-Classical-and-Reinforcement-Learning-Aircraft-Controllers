"""LSTM policy network for rate control.

This module provides a custom LSTM-based policy for use with
Stable-Baselines3's RecurrentPPO algorithm.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Tuple


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for recurrent policies.

    Processes observations through LSTM layers to capture temporal
    dependencies in the rate control task.

    Architecture:
        Input -> Embedding -> LSTM1 -> LSTM2 -> Output
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 2,
    ):
        """Initialize LSTM feature extractor.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden state size
            n_lstm_layers: Number of LSTM layers
        """
        super().__init__(observation_space, features_dim)

        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        # Input embedding
        n_input = observation_space.shape[0]
        self.embedding = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM.

        Args:
            observations: Batch of observations [batch_size, obs_dim]
                         or [batch_size, seq_len, obs_dim] for sequences

        Returns:
            Features tensor [batch_size, features_dim]
        """
        # Handle both single-step and sequential inputs
        if len(observations.shape) == 2:
            # Add sequence dimension [batch, obs] -> [batch, 1, obs]
            observations = observations.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Embedding
        embedded = self.embedding(observations)

        # LSTM
        lstm_out, _ = self.lstm(embedded)

        # Take last timestep output
        if squeeze_output:
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = lstm_out[:, -1, :]

        # Project to feature space
        features = self.output_proj(lstm_out)

        return features


class LSTMPolicy:
    """LSTM policy configuration for SB3 RecurrentPPO.

    This is a configuration class that specifies the LSTM
    architecture to be used with sb3_contrib.RecurrentPPO.
    """

    @staticmethod
    def get_policy_kwargs(
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 2,
        features_dim: int = 128,
    ) -> dict:
        """Get policy kwargs for RecurrentPPO.

        Args:
            lstm_hidden_size: Size of LSTM hidden states
            n_lstm_layers: Number of LSTM layers
            features_dim: Output feature dimension

        Returns:
            Dictionary of policy kwargs for RecurrentPPO
        """
        policy_kwargs = {
            "features_extractor_class": LSTMFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
            },
            "net_arch": dict(
                pi=[128, 64],  # Actor network
                vf=[128, 64],  # Critic network
            ),
            "activation_fn": nn.ReLU,
        }
        return policy_kwargs


class SimpleMLPPolicy:
    """Simple MLP policy configuration (non-recurrent baseline).

    For comparison, this provides a standard feedforward network.
    """

    @staticmethod
    def get_policy_kwargs(
        net_arch: list = [256, 128, 64],
    ) -> dict:
        """Get policy kwargs for PPO (non-recurrent).

        Args:
            net_arch: Network architecture (shared layers)

        Returns:
            Dictionary of policy kwargs for PPO
        """
        policy_kwargs = {
            "net_arch": dict(
                pi=net_arch,  # Actor network
                vf=net_arch,  # Critic network
            ),
            "activation_fn": nn.ReLU,
        }
        return policy_kwargs
