"""Gamma Arbitration Policy — SB3-compatible PPO policy for BRACE.

Outputs scalar gamma in [0, 1] via tanh squashing on actor mean.
Input: concatenation of [state_features, belief_vector].
Architecture: shared 256x256 ReLU trunk, actor head (1, tanh), critic head (1).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from stable_baselines3.common.policies import ActorCriticPolicy


class GammaArbitrationPolicy(ActorCriticPolicy):
    """PPO-compatible policy that outputs scalar gamma in [0, 1].

    The actor outputs a single value in [-1, 1] (via tanh on the mean).
    Gamma is obtained externally as gamma = 0.5 * (action + 1).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )

        self.log_std = nn.Parameter(torch.zeros(1))

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        obs_dim = self.observation_space.shape[0]
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (action, value, log_prob).

        Action is in [-1, 1] via tanh. Map to gamma externally with 0.5*(a+1).
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)

        mean = torch.tanh(self.action_net(latent_pi))
        std = torch.clamp(self.log_std, -20.0, 2.0).exp().expand_as(mean)

        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()
            action = torch.tanh(action)

        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1)

        value = self.value_net(latent_vf)

        return action, value, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions for PPO update (returns values, log_prob, entropy)."""
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)

        mean = torch.tanh(self.action_net(latent_pi))
        std = torch.clamp(self.log_std, -20.0, 2.0).exp().expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_net(latent_vf)

        return value, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.vf_features_extractor)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)


def action_to_gamma(action: float) -> float:
    """Map raw policy action in [-1, 1] to gamma in [0, 1]."""
    return 0.5 * (max(min(action, 1.0), -1.0) + 1.0)


def gamma_to_action(gamma: float) -> float:
    """Map gamma in [0, 1] to raw policy action in [-1, 1]."""
    return 2.0 * max(min(gamma, 1.0), 0.0) - 1.0
