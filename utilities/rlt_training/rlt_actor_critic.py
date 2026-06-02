from __future__ import annotations

import torch
from torch import nn

from rlt_training.rlt_model import SmallImageEncoder


class RLTActorCritic(nn.Module):
    """Semantic RLT actor-critic.

    The actor predicts the same semantic corrections that the human provides:
      - ee_offset: Cartesian end-effector translation [dx, dy, dz]
      - gripper_logits: class logits over {none, close, open}

    Runtime converts ee_offset to joint residuals through the right-arm Jacobian.
    This is safer than directly learning a horizon x action_dim joint residual.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        horizon: int = 12,
        token_dim: int = 256,
        hidden_dim: int = 512,
        max_delta: float = 0.03,
        max_ee_offset: float = 0.03,
        gripper_classes: int = 3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.token_dim = int(token_dim)
        self.max_delta = float(max_delta)
        self.max_ee_offset = float(max_ee_offset)
        self.gripper_classes = int(gripper_classes)

        self.top_encoder = SmallImageEncoder(out_dim=token_dim)
        self.wrist_encoder = SmallImageEncoder(out_dim=token_dim)

        flat_action_dim = self.horizon * self.action_dim
        self.flat_action_dim = flat_action_dim
        self.semantic_dim = 3 + self.gripper_classes

        self.token_net = nn.Sequential(
            nn.Linear(token_dim * 2 + self.state_dim + flat_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        actor_in = token_dim + self.state_dim + flat_action_dim
        self.actor_trunk = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.ee_head = nn.Linear(hidden_dim, 3)
        self.gripper_head = nn.Linear(hidden_dim, self.gripper_classes)

        critic_in = token_dim + self.state_dim + flat_action_dim + self.semantic_dim
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
            nn.Linear(token_dim + self.state_dim, hidden_dim // 2),
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

        h = self.actor_trunk(torch.cat([token, state, pi_flat], dim=-1))
        ee_offset = torch.tanh(self.ee_head(h)) * self.max_ee_offset
        gripper_logits = self.gripper_head(h)
        return ee_offset, gripper_logits

    def q_from_token(self, token, state, pi_actions, ee_offset, gripper_onehot):
        batch = token.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)
        x = torch.cat([token, state, pi_flat, ee_offset, gripper_onehot], dim=-1)
        return self.critic1(x).squeeze(-1), self.critic2(x).squeeze(-1)

    def forward(self, top_image, wrist_image, state, pi_actions, reference_dropout: float = 0.0):
        token = self.encode_token(top_image, wrist_image, state, pi_actions)
        ee_offset, gripper_logits = self.act_from_token(
            token, state, pi_actions, reference_dropout=reference_dropout
        )
        gripper_probs = torch.softmax(gripper_logits, dim=-1)
        gripper_class = torch.argmax(gripper_logits, dim=-1)
        gripper_onehot = torch.nn.functional.one_hot(
            gripper_class, num_classes=self.gripper_classes
        ).float()
        q1, q2 = self.q_from_token(token, state, pi_actions, ee_offset, gripper_onehot)
        gate_logit = self.gate_head(torch.cat([token, state], dim=-1)).squeeze(-1)
        return {
            "ee_offset": ee_offset,
            "gripper_logits": gripper_logits,
            "gripper_probs": gripper_probs,
            "gripper_class": gripper_class,
            "gripper_onehot": gripper_onehot,
            "q1": q1,
            "q2": q2,
            "gate_logit": gate_logit,
            "rlt_token": token,
        }
