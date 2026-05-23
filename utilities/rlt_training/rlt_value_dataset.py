from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rlt_training.rlt_dataset import RLTBatchSpec, _image_to_chw_float, _pad_or_trim


class RLTValueReplayDataset(Dataset):
    def __init__(self, replay_dir: str | Path, spec: RLTBatchSpec | None = None):
        self.replay_dir = Path(replay_dir)
        self.spec = spec or RLTBatchSpec()
        self.files = sorted((self.replay_dir / "chunks").rglob("*.npz"))
        if not self.files:
            self.files = sorted(self.replay_dir.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No replay chunks found in {self.replay_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as item:
            top = _image_to_chw_float(item["observation_image"], self.spec.image_size)
            wrist = _image_to_chw_float(item["observation_wrist_image"], self.spec.image_size)
            state = torch.from_numpy(np.asarray(item["observation_state"], dtype=np.float32))
            pi_actions, valid = _pad_or_trim(item["pi_actions"], self.spec.horizon, self.spec.action_dim)
            executed_actions, _ = _pad_or_trim(item["executed_actions"], self.spec.horizon, self.spec.action_dim)
            if "delta_actions" in item:
                delta_actions, _ = _pad_or_trim(item["delta_actions"], self.spec.horizon, self.spec.action_dim)
            else:
                delta_actions = executed_actions - pi_actions
            reward_key = "discounted_return" if "discounted_return" in item else "reward"
            reward = np.float32(item[reward_key])
            success = np.float32(item["success"])
            failure = np.float32(item["failure"])
            collision = np.float32(item["collision"])
            exec_ok = np.float32(item["exec_ok"])
            human_env_reset = np.float32(item["human_env_reset"])
            rlt_enabled = np.float32(item["rlt_enabled"]) if "rlt_enabled" in item else np.float32(0.0)
            rlt_toggle_on = np.float32(item["rlt_toggle_on"]) if "rlt_toggle_on" in item else np.float32(0.0)
            phase_id = np.int64(item["phase_id"]) if "phase_id" in item else np.int64(0)
        risk = np.float32((failure > 0.5) or (collision > 0.5) or (exec_ok < 0.5))
        should_enable_rlt = np.float32((rlt_enabled > 0.5) or (rlt_toggle_on > 0.5))
        useful_for_bc = np.float32((rlt_enabled > 0.5) and (risk < 0.5))
        target_delta = delta_actions
        return {
            "top_image": top,
            "wrist_image": wrist,
            "state": state,
            "pi_actions": torch.from_numpy(pi_actions),
            "target_delta": torch.from_numpy(target_delta.astype(np.float32)),
            "valid": torch.from_numpy(valid),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "risk": torch.tensor(risk, dtype=torch.float32),
            "should_enable_rlt": torch.tensor(should_enable_rlt, dtype=torch.float32),
            "phase_id": torch.tensor(phase_id, dtype=torch.long),
            "bc_weight": torch.from_numpy(valid.astype(np.float32) * useful_for_bc * 0.2),
            "path": str(path),
        }
