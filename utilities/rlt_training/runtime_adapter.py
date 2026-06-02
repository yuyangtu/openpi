from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rlt_training.rlt_actor_critic import RLTActorCritic
from rlt_training.rlt_model import RLTActionAdapter


def _image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    if image.shape[:2] != (image_size, image_size):
        raise ValueError(f"Expected image size {image_size}, got {image.shape[:2]}")
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()


class RLTActionRuntime:
    """Runtime wrapper.

    Supports two checkpoint kinds:
      - rlt_actor_critic_semantic: predicts ee_offset + gripper class.
      - rlt_actor_critic / legacy adapter: returns corrected joint chunk.

    For semantic checkpoints, correct_observation_chunk() deliberately returns
    pi_actions unchanged and exposes semantic correction in info. The robot
    script should convert info["ee_offset"] to joint residual via Jacobian.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        max_delta: float = 0.12,
        risk_stop_threshold: float | None = None,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        spec = checkpoint.get("spec", {})
        args = checkpoint.get("args", {})
        self.horizon = int(spec.get("horizon", args.get("horizon", 8)))
        self.action_dim = int(spec.get("action_dim", args.get("action_dim", 8)))
        self.image_size = int(spec.get("image_size", args.get("image_size", 224)))
        self.max_delta = float(max_delta)
        self.max_ee_offset = float(args.get("max_ee_offset", max_delta))
        self.risk_stop_threshold = risk_stop_threshold
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.kind = checkpoint.get("kind", "rlt_action_adapter")

        if self.kind in {"rlt_actor_critic", "rlt_actor_critic_semantic"}:
            self.model = RLTActorCritic(
                state_dim=int(args.get("state_dim", 8)),
                action_dim=self.action_dim,
                horizon=self.horizon,
                max_delta=float(args.get("max_delta", max_delta)),
                max_ee_offset=float(args.get("max_ee_offset", max_delta)),
            ).to(self.device)
        else:
            self.model = RLTActionAdapter(
                state_dim=int(args.get("state_dim", 8)),
                action_dim=self.action_dim,
                horizon=self.horizon,
            ).to(self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.no_grad()
    def correct_observation_chunk(self, obs: dict, pi_actions: np.ndarray) -> tuple[np.ndarray, dict]:
        pi_actions = np.asarray(pi_actions, dtype=np.float32)
        original_len = pi_actions.shape[0]

        padded = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        n = min(self.horizon, original_len)
        padded[:n] = pi_actions[:n, : self.action_dim]

        top = _image_to_tensor(obs["observation/image"], self.image_size).to(self.device)
        wrist = _image_to_tensor(obs["observation/wrist_image"], self.image_size).to(self.device)
        state = torch.from_numpy(np.asarray(obs["observation/state"], dtype=np.float32)).unsqueeze(0).to(self.device)
        pi = torch.from_numpy(padded).unsqueeze(0).to(self.device)

        pred = self.model(top, wrist, state, pi)

        logit_key = "risk_logit" if "risk_logit" in pred else "gate_logit"
        risk = float(torch.sigmoid(pred[logit_key])[0].detach().cpu())
        value = float(pred["q1"][0].detach().cpu()) if "q1" in pred else 0.0

        info = {
            "value": value,
            "risk": risk,
            "stopped_by_risk": False,
            "kind": self.kind,
        }
        if self.risk_stop_threshold is not None and risk >= self.risk_stop_threshold:
            info["stopped_by_risk"] = True
            return pi_actions.astype(np.float64), info

        if self.kind == "rlt_actor_critic_semantic":
            ee_offset = pred["ee_offset"][0].detach().cpu().numpy().astype(np.float64)
            ee_offset = np.clip(ee_offset, -self.max_ee_offset, self.max_ee_offset)
            gripper_probs = pred["gripper_probs"][0].detach().cpu().numpy().astype(np.float64)
            gripper_class = int(np.argmax(gripper_probs))
            gripper_name = {0: "none", 1: "close", 2: "open"}.get(gripper_class, "none")
            info.update(
                {
                    "source": "rlt_checkpoint_semantic",
                    "ee_offset": ee_offset,
                    "gripper_class": gripper_class,
                    "gripper_override": gripper_name,
                    "gripper_probs": gripper_probs,
                }
            )
            return pi_actions.astype(np.float64), info

        # Legacy joint residual checkpoints.
        delta = pred["delta_actions"][0].detach().cpu().numpy()
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        corrected = padded + delta
        corrected = corrected[:original_len].astype(np.float64)
        info.update({"source": "rlt_checkpoint_joint", "delta": delta})
        return corrected, info
