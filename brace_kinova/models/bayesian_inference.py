"""Bayesian Goal Inference module with learnable parameters.

Maintains a recursive belief distribution over N candidate goals.
Learnable parameters (beta, w_theta, w_dist) are constrained positive via softplus.
Supports pretraining (supervised NLL) and end-to-end training (REINFORCE).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianGoalInference(nn.Module):
    """Bayesian belief update over discrete goals with learnable cost parameters.

    At each timestep, given human action h_t and EE position, computes a
    noisy-rational (Boltzmann) likelihood over goals and updates the belief.

    Parameters are learned via softplus(raw_param) to ensure positivity:
        beta:    rationality coefficient (sharpness of Boltzmann likelihood)
        w_theta: weight on angular deviation cost
        w_dist:  weight on distance deviation cost
    """

    def __init__(
        self,
        n_goals: int,
        initial_beta: float = 2.0,
        initial_w_theta: float = 0.8,
        initial_w_dist: float = 0.2,
        ema_alpha: float = 0.85,
    ):
        super().__init__()
        self.n_goals = n_goals
        self.ema_alpha = ema_alpha

        self.raw_beta = nn.Parameter(torch.tensor(initial_beta))
        self.raw_w_theta = nn.Parameter(torch.tensor(initial_w_theta))
        self.raw_w_dist = nn.Parameter(torch.tensor(initial_w_dist))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

    @property
    def w_theta(self) -> torch.Tensor:
        return F.softplus(self.raw_w_theta)

    @property
    def w_dist(self) -> torch.Tensor:
        return F.softplus(self.raw_w_dist)

    def step_likelihood(
        self,
        human_action: torch.Tensor,
        ee_position: torch.Tensor,
        goal_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-goal likelihoods for a single timestep.

        Args:
            human_action: (batch, 2) human velocity command in XY.
            ee_position: (batch, 2) current EE position.
            goal_positions: (batch, n_goals, 2) candidate goal positions.

        Returns:
            likelihood: (batch, n_goals) un-normalized likelihood per goal.
        """
        batch = human_action.shape[0]
        h = human_action  # (batch, 2)
        h_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        h_unit = h / h_norm  # (batch, 2)

        vec_to_goal = goal_positions - ee_position.unsqueeze(1)  # (batch, n_goals, 2)
        dist_to_goal = torch.linalg.norm(vec_to_goal, dim=-1, keepdim=True).clamp(min=1e-8)
        goal_unit = vec_to_goal / dist_to_goal  # (batch, n_goals, 2)

        cos_sim = (h_unit.unsqueeze(1) * goal_unit).sum(dim=-1)  # (batch, n_goals)
        cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta_dev = torch.acos(cos_sim)  # (batch, n_goals)

        h_opt = dist_to_goal.squeeze(-1)  # optimal magnitude ~ distance to goal
        h_mag = h_norm.squeeze(-1)  # (batch,)
        dist_dev = torch.abs(1.0 - h_mag.unsqueeze(1) / h_opt.clamp(min=1e-8))

        cost = self.w_theta * theta_dev + self.w_dist * dist_dev  # (batch, n_goals)
        log_likelihood = -self.beta * cost

        log_likelihood = log_likelihood - log_likelihood.max(dim=-1, keepdim=True).values
        likelihood = torch.exp(log_likelihood)

        return likelihood

    def update_belief(
        self,
        prior: torch.Tensor,
        likelihood: torch.Tensor,
        tau: float = 1.0,
        prev_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bayesian belief update with temperature and EMA smoothing.

        Args:
            prior: (batch, n_goals) prior belief.
            likelihood: (batch, n_goals) per-goal likelihoods.
            tau: temperature for softmax normalization (>=1 = more uniform).
            prev_belief: optional previous belief for EMA smoothing.

        Returns:
            posterior: (batch, n_goals) updated, normalized belief.
        """
        log_posterior = torch.log(prior.clamp(min=1e-10)) + torch.log(
            likelihood.clamp(min=1e-10)
        )
        posterior = F.softmax(log_posterior / max(tau, 1e-6), dim=-1)

        if prev_belief is not None:
            posterior = self.ema_alpha * posterior + (1.0 - self.ema_alpha) * prev_belief

        return posterior

    def forward(
        self,
        human_action: torch.Tensor,
        ee_position: torch.Tensor,
        goal_positions: torch.Tensor,
        prior_belief: torch.Tensor,
        true_goal_idx: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        prev_belief: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: likelihood → belief update → REINFORCE log-prob.

        Args:
            human_action: (batch, 2) human velocity command.
            ee_position: (batch, 2) EE position.
            goal_positions: (batch, n_goals, 2) goal positions.
            prior_belief: (batch, n_goals) prior belief distribution.
            true_goal_idx: (batch,) optional true goal indices for supervised NLL.
            tau: softmax temperature.
            prev_belief: optional previous belief for EMA.

        Returns:
            updated_belief: (batch, n_goals) posterior belief.
            loss_term: scalar loss. If true_goal_idx given: NLL on true goal.
                       Otherwise: negative log-prob for REINFORCE.
        """
        likelihood = self.step_likelihood(human_action, ee_position, goal_positions)
        updated_belief = self.update_belief(prior_belief, likelihood, tau, prev_belief)

        if true_goal_idx is not None:
            batch_indices = torch.arange(updated_belief.shape[0], device=updated_belief.device)
            p_true = updated_belief[batch_indices, true_goal_idx]
            nll = -torch.log(p_true.clamp(min=1e-10)).mean()
            return updated_belief, nll

        weighted_ll = (prior_belief.detach() * likelihood).sum(dim=-1)
        neg_log_prob = -torch.log(weighted_ll.clamp(min=1e-10)).mean()
        return updated_belief, neg_log_prob

    def get_uniform_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a uniform prior belief over all goals."""
        return torch.ones(batch_size, self.n_goals, device=device) / self.n_goals
