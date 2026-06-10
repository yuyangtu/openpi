from __future__ import annotations

import torch
from torch import nn

from rlt_training.rlt_model import SmallImageEncoder


class DINOv2ImageEncoder(nn.Module):
    """Frozen DINOv2-S/14 image encoder with a small projection head.

    Input:  image tensor in RGB, shape [B, 3, 224, 224], value range [0, 1].
    Output: feature tensor [B, out_dim].

    The DINOv2 backbone is kept frozen by default. Only the projection head is
    trained. This is intentionally simple and avoids offline pre-encoding.
    """

    def __init__(self, out_dim: int = 256, model_name: str = "dinov2_vits14", freeze: bool = True):
        super().__init__()
        self.out_dim = int(out_dim)
        self.model_name = str(model_name)
        self.freeze = bool(freeze)

        # torch.hub will download the weights on the first run unless they are
        # already in ~/.cache/torch/hub/checkpoints.
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
        except TypeError:
            self.backbone = torch.hub.load("facebookresearch/dinov2", self.model_name)

        # DINOv2-S/14 output dimension is 384. Keep this explicit for stability.
        if self.model_name == "dinov2_vits14":
            feat_dim = 384
        elif self.model_name == "dinov2_vitb14":
            feat_dim = 768
        else:
            feat_dim = int(getattr(self.backbone, "embed_dim", 384))

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        image = (image - self.mean.to(image.device)) / self.std.to(image.device)
        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(image)
            feat = feat.detach()
        else:
            feat = self.backbone(image)
        return self.proj(feat)


class RLTActorCritic(nn.Module):
    """Semantic RLT actor-critic.

    Actor predicts:
      - ee_offset: Cartesian EE translation [dx, dy, dz]
      - gripper_logits: class logits over {none, close, open}

    Encoder choices:
      - image_encoder="small": old lightweight CNN encoder.
      - image_encoder="dinov2_s": frozen DINOv2-S/14 encoder + trainable projection.
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
        image_encoder: str = "small",
        freeze_image_encoder: bool = True,
        dinov2_model: str = "dinov2_vits14",
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.token_dim = int(token_dim)
        self.max_delta = float(max_delta)
        self.max_ee_offset = float(max_ee_offset)
        self.gripper_classes = int(gripper_classes)
        self.image_encoder_type = str(image_encoder)
        self.freeze_image_encoder = bool(freeze_image_encoder)
        self.dinov2_model = str(dinov2_model)

        if self.image_encoder_type == "dinov2_s":
            # One shared frozen DINOv2 encoder is used for both top and wrist images
            # to save memory. The two calls still produce two independent features.
            self.shared_image_encoder = DINOv2ImageEncoder(
                out_dim=token_dim,
                model_name=self.dinov2_model,
                freeze=self.freeze_image_encoder,
            )
            self.top_encoder = None
            self.wrist_encoder = None
        elif self.image_encoder_type == "small":
            self.shared_image_encoder = None
            self.top_encoder = SmallImageEncoder(out_dim=token_dim)
            self.wrist_encoder = SmallImageEncoder(out_dim=token_dim)
        else:
            raise ValueError(f"Unknown image_encoder={self.image_encoder_type!r}; use 'small' or 'dinov2_s'.")

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

    def _encode_images(self, top_image: torch.Tensor, wrist_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_image_encoder is not None:
            top_feat = self.shared_image_encoder(top_image)
            wrist_feat = self.shared_image_encoder(wrist_image)
        else:
            top_feat = self.top_encoder(top_image)
            wrist_feat = self.wrist_encoder(wrist_image)
        return top_feat, wrist_feat

    def encode_token(self, top_image, wrist_image, state, pi_actions):
        batch = top_image.shape[0]
        pi_flat = pi_actions.reshape(batch, self.horizon * self.action_dim)
        top_feat, wrist_feat = self._encode_images(top_image, wrist_image)
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
