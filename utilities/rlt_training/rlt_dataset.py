from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclasses.dataclass(frozen=True)
class RLTBatchSpec:
    horizon: int = 8
    action_dim: int = 8
    image_size: int = 224
    gamma: float = 0.97
    bc_weight_no_intervention: float = 0.05


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _image_to_chw_float(image: np.ndarray, image_size: int) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if image.shape[0] != image_size or image.shape[1] != image_size:
        raise ValueError(f"Expected pre-resized image ({image_size}, {image_size}, 3), got {image.shape}")
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def _pad_or_trim(array: np.ndarray, horizon: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"Expected array shape (T, {dim}), got {array.shape}")
    valid = np.zeros((horizon,), dtype=np.float32)
    out = np.zeros((horizon, dim), dtype=np.float32)
    n = min(horizon, array.shape[0])
    out[:n] = array[:n]
    valid[:n] = 1.0
    return out, valid


class RLTChunkDataset(Dataset):
    def __init__(self, data_dir: str | Path, spec: RLTBatchSpec | None = None):
        self.data_dir = Path(data_dir)
        self.spec = spec or RLTBatchSpec()
        self.files = sorted(self.data_dir.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz RLT chunks found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as item:
            top = _image_to_chw_float(item["observation_image"], self.spec.image_size)
            wrist = _image_to_chw_float(item["observation_wrist_image"], self.spec.image_size)
            state = _as_float_tensor(item["observation_state"])
            pi_actions, valid = _pad_or_trim(item["pi_actions"], self.spec.horizon, self.spec.action_dim)
            human_actions, _ = _pad_or_trim(item["human_actions"], self.spec.horizon, self.spec.action_dim)
            executed_actions, _ = _pad_or_trim(item["executed_actions"], self.spec.horizon, self.spec.action_dim)
            raw_mask = np.asarray(item["intervention_mask"], dtype=np.float32).reshape(-1)
            mask = np.zeros((self.spec.horizon,), dtype=np.float32)
            n = min(self.spec.horizon, raw_mask.shape[0])
            mask[:n] = raw_mask[:n]
            reward = np.float32(item["reward"])
            done = np.float32(item["done"])
            needs_intervention = np.float32(item["needs_intervention"])

        target_actions = np.where(mask[:, None] > 0.5, human_actions, executed_actions)
        target_delta = target_actions - pi_actions
        bc_weight = np.where(mask > 0.5, 1.0, float(self.spec.bc_weight_no_intervention)).astype(np.float32)
        bc_weight *= valid
        return {
            "top_image": top,
            "wrist_image": wrist,
            "state": state,
            "pi_actions": torch.from_numpy(pi_actions),
            "target_delta": torch.from_numpy(target_delta.astype(np.float32)),
            "bc_weight": torch.from_numpy(bc_weight),
            "valid": torch.from_numpy(valid),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32),
            "needs_intervention": torch.tensor(needs_intervention, dtype=torch.float32),
            "path": str(path),
        }
