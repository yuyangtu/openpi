from __future__ import annotations

import torch
from torch import nn


class SmallImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class RLTActionAdapter(nn.Module):
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        horizon: int = 8,
        token_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.token_dim = token_dim

        self.top_encoder = SmallImageEncoder(out_dim=token_dim)
        self.wrist_encoder = SmallImageEncoder(out_dim=token_dim)
        self.rlt_token = nn.Sequential(
            nn.Linear(token_dim * 2 + state_dim + horizon * action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(token_dim + state_dim + horizon * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(token_dim + state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(token_dim + state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        top_image: torch.Tensor,
        wrist_image: torch.Tensor,
        state: torch.Tensor,
        pi_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = top_image.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)

        top_feat = self.top_encoder(top_image)
        wrist_feat = self.wrist_encoder(wrist_image)
        token_input = torch.cat([top_feat, wrist_feat, state, pi_flat], dim=-1)
        token = self.rlt_token(token_input)

        action_input = torch.cat([token, state, pi_flat], dim=-1)
        delta = self.action_head(action_input).reshape(batch, self.horizon, self.action_dim)

        value_input = torch.cat([token, state], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        risk_logit = self.risk_head(value_input).squeeze(-1)

        return {
            "delta_actions": delta,
            "value": value,
            "risk_logit": risk_logit,
            "rlt_token": token,
        }
