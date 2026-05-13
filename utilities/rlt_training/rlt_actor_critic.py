from __future__ import annotations

import torch
from torch import nn

from rlt_training.rlt_model import SmallImageEncoder


class RLTActorCritic(nn.Module):
    """Simplified RLT-style residual actor-critic.

    The token is computed externally from images, robot state, and the VLA action
    chunk. The actor edits the VLA action chunk with a small residual, and the
    critic scores the proposed residual.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        horizon: int = 8,
        token_dim: int = 256,
        hidden_dim: int = 512,
        max_delta: float = 0.03,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.token_dim = token_dim
        self.max_delta = max_delta

        self.top_encoder = SmallImageEncoder(out_dim=token_dim)
        self.wrist_encoder = SmallImageEncoder(out_dim=token_dim)
        flat_action_dim = horizon * action_dim

        self.token_net = nn.Sequential(
            nn.Linear(token_dim * 2 + state_dim + flat_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(token_dim + state_dim + flat_action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )
        critic_in = token_dim + state_dim + flat_action_dim + flat_action_dim
        self.critic1 = nn.Sequential(
            nn.Linear(critic_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(critic_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(token_dim + state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_token(self, top_image, wrist_image, state, pi_actions):
        batch = top_image.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)
        top_feat = self.top_encoder(top_image)
        wrist_feat = self.wrist_encoder(wrist_image)
        return self.token_net(torch.cat([top_feat, wrist_feat, state, pi_flat], dim=-1))

    def act_from_token(self, token, state, pi_actions, reference_dropout: float = 0.0):
        batch = token.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)
        if self.training and reference_dropout > 0:
            keep = (torch.rand(batch, 1, device=pi_flat.device) > reference_dropout).float()
            pi_flat = pi_flat * keep
        raw = self.actor(torch.cat([token, state, pi_flat], dim=-1))
        delta = torch.tanh(raw).reshape(batch, self.horizon, self.action_dim)
        return delta * self.max_delta

    def q_from_token(self, token, state, pi_actions, delta_actions):
        batch = token.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)
        delta_flat = delta_actions.reshape(batch, self.horizon * self.action_dim)
        x = torch.cat([token, state, pi_flat, delta_flat], dim=-1)
        return self.critic1(x).squeeze(-1), self.critic2(x).squeeze(-1)

    def forward(self, top_image, wrist_image, state, pi_actions, reference_dropout: float = 0.0):
        token = self.encode_token(top_image, wrist_image, state, pi_actions)
        delta = self.act_from_token(token, state, pi_actions, reference_dropout=reference_dropout)
        q1, q2 = self.q_from_token(token, state, pi_actions, delta)
        gate_logit = self.gate_head(torch.cat([token, state], dim=-1)).squeeze(-1)
        return {
            "delta_actions": delta,
            "q1": q1,
            "q2": q2,
            "gate_logit": gate_logit,
            "rlt_token": token,
        }
