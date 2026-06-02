from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rlt_training.rlt_dataset import RLTBatchSpec, _image_to_chw_float, _pad_or_trim


_GRIPPER_TO_ID = {"none": 0, "close": 1, "open": 2}


def _as_scalar_str(x) -> str:
    try:
        if isinstance(x, np.ndarray):
            return str(x.item())
        return str(x)
    except Exception:
        return ""


def _parse_ee_offset_from_note(note: str) -> np.ndarray:
    """Parse old note strings like 'ee_offset=[0.   , 0.006, 0.   ];...'."""
    if not note:
        return np.zeros((3,), dtype=np.float32)
    m = re.search(r"ee_offset=\[([^\]]+)\]", note)
    if not m:
        return np.zeros((3,), dtype=np.float32)
    raw = m.group(1).replace(",", " ").split()
    vals = []
    for token in raw[:3]:
        try:
            vals.append(float(token))
        except ValueError:
            vals.append(0.0)
    while len(vals) < 3:
        vals.append(0.0)
    return np.asarray(vals[:3], dtype=np.float32)


def _parse_gripper_from_note(note: str) -> int:
    """Parse old note strings like 'gripper_override=close'."""
    if not note:
        return 0
    m = re.search(r"gripper_override=([A-Za-z_]+)", note)
    if not m:
        return 0
    return int(_GRIPPER_TO_ID.get(m.group(1).lower(), 0))


def _infer_gripper_from_delta(delta_actions: np.ndarray) -> int:
    """Fallback: infer close/open from gripper residual sign if note field is missing."""
    if delta_actions.size == 0 or delta_actions.shape[-1] < 8:
        return 0
    g = np.asarray(delta_actions[:, 7], dtype=np.float32)
    if np.max(np.abs(g)) < 1e-4:
        return 0
    return 1 if float(np.mean(g)) > 0 else 2


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

            pi_actions, valid = _pad_or_trim(
                item["pi_actions"], self.spec.horizon, self.spec.action_dim
            )
            executed_actions, _ = _pad_or_trim(
                item["executed_actions"], self.spec.horizon, self.spec.action_dim
            )
            if "delta_actions" in item:
                delta_actions, _ = _pad_or_trim(
                    item["delta_actions"], self.spec.horizon, self.spec.action_dim
                )
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

            note = _as_scalar_str(item["note"]) if "note" in item else ""

            if "manual_ee_offset_xyz" in item:
                target_ee_offset = np.asarray(item["manual_ee_offset_xyz"], dtype=np.float32).reshape(3)
            else:
                target_ee_offset = _parse_ee_offset_from_note(note)

            if "gripper_override_id" in item:
                target_gripper = int(np.asarray(item["gripper_override_id"]).item())
            else:
                target_gripper = _parse_gripper_from_note(note)
                if target_gripper == 0:
                    target_gripper = _infer_gripper_from_delta(delta_actions)

        risk = np.float32((failure > 0.5) or (collision > 0.5) or (exec_ok < 0.5))
        should_enable_rlt = np.float32((rlt_enabled > 0.5) or (rlt_toggle_on > 0.5))

        # Semantic BC should focus on human/RLT-enabled chunks that did not immediately fail.
        semantic_nonzero = np.float32(
            (np.max(np.abs(target_ee_offset)) > 1e-6) or (target_gripper != 0)
        )
        bc_weight = np.float32((should_enable_rlt > 0.5) and (risk < 0.5) and (semantic_nonzero > 0.5))

        return {
            "top_image": top,
            "wrist_image": wrist,
            "state": state,
            "pi_actions": torch.from_numpy(pi_actions),
            "executed_actions": torch.from_numpy(executed_actions),
            "target_delta": torch.from_numpy(delta_actions.astype(np.float32)),
            "target_ee_offset": torch.from_numpy(target_ee_offset.astype(np.float32)),
            "target_gripper": torch.tensor(target_gripper, dtype=torch.long),
            "valid": torch.from_numpy(valid),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "risk": torch.tensor(risk, dtype=torch.float32),
            "should_enable_rlt": torch.tensor(should_enable_rlt, dtype=torch.float32),
            "phase_id": torch.tensor(phase_id, dtype=torch.long),
            "bc_weight": torch.tensor(bc_weight, dtype=torch.float32),
            "path": str(path),
        }
