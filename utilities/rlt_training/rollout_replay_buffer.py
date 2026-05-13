from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass
class ChunkLabel:
    success: bool = False
    failure: bool = False
    collision: bool = False
    exec_ok: bool = True
    rlt_enabled: bool = False
    rlt_toggle_on: bool = False
    rlt_toggle_off: bool = False
    human_env_reset: bool = False
    cut_before_reset: bool = False
    reset_type: str = ""
    phase: str = "unknown"
    note: str = ""


PHASE_TO_ID = {
    "unknown": 0,
    "approach_spoon": 1,
    "grasp_spoon": 2,
    "lift_food": 3,
    "approach_mouth": 4,
    "feed": 5,
    "retreat": 6,
}


class RolloutReplayBuffer:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.chunk_dir = self.root / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _reward(self, label: ChunkLabel) -> float:
        reward = 0.0
        if label.success:
            reward += 1.0
        if label.failure:
            reward -= 1.0
        if label.collision:
            reward -= 0.7
        if not label.exec_ok:
            reward -= 0.3
        if label.human_env_reset:
            reward -= 0.1
        return float(reward)

    def add_chunk(
        self,
        obs: dict[str, Any],
        pi_actions: np.ndarray,
        executed_actions: np.ndarray,
        label: ChunkLabel,
        episode_id: str,
        chunk_id: int,
        delta_actions: np.ndarray | None = None,
    ) -> Path:
        timestamp = time.time()
        reward = self._reward(label)
        phase_id = PHASE_TO_ID.get(label.phase, PHASE_TO_ID["unknown"])
        if delta_actions is None:
            delta_actions = np.asarray(executed_actions, dtype=np.float32) - np.asarray(pi_actions, dtype=np.float32)
        path = self.chunk_dir / f"{episode_id}_{chunk_id:06d}_{int(timestamp * 1000)}.npz"
        np.savez_compressed(
            path,
            observation_image=np.asarray(obs["observation/image"], dtype=np.uint8),
            observation_wrist_image=np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            observation_state=np.asarray(obs["observation/state"], dtype=np.float32),
            pi_actions=np.asarray(pi_actions, dtype=np.float32),
            executed_actions=np.asarray(executed_actions, dtype=np.float32),
            delta_actions=np.asarray(delta_actions, dtype=np.float32),
            reward=np.float32(reward),
            success=np.bool_(label.success),
            failure=np.bool_(label.failure),
            collision=np.bool_(label.collision),
            exec_ok=np.bool_(label.exec_ok),
            rlt_enabled=np.bool_(label.rlt_enabled),
            rlt_toggle_on=np.bool_(label.rlt_toggle_on),
            rlt_toggle_off=np.bool_(label.rlt_toggle_off),
            human_env_reset=np.bool_(label.human_env_reset),
            cut_before_reset=np.bool_(label.cut_before_reset),
            reset_type=np.asarray(label.reset_type, dtype=object),
            phase=np.asarray(label.phase, dtype=object),
            phase_id=np.int64(phase_id),
            prompt=np.asarray(obs.get("prompt", ""), dtype=object),
            episode_id=np.asarray(episode_id, dtype=object),
            chunk_id=np.int64(chunk_id),
            timestamp=np.float64(timestamp),
        )
        row = {
            "file": str(path),
            "episode_id": episode_id,
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "reward": reward,
            "success": bool(label.success),
            "failure": bool(label.failure),
            "collision": bool(label.collision),
            "exec_ok": bool(label.exec_ok),
            "rlt_enabled": bool(label.rlt_enabled),
            "rlt_toggle_on": bool(label.rlt_toggle_on),
            "rlt_toggle_off": bool(label.rlt_toggle_off),
            "human_env_reset": bool(label.human_env_reset),
            "cut_before_reset": bool(label.cut_before_reset),
            "reset_type": label.reset_type,
            "phase": label.phase,
            "note": label.note,
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path


class RolloutKeyboard:
    def __init__(self):
        self.rlt_enabled = False
        self.rlt_toggle_on = False
        self.rlt_toggle_off = False
        self.human_env_reset = False
        self.success = False
        self.failure = False
        self.collision = False
        self.stop = False
        self.reset_type = ""
        self.phase = "unknown"
        self.note = ""

    def poll_blocking(self) -> None:
        cmd = input(
            "[rlt] enter=none r=toggle-rlt e=env-reset s=success f=failure c=collision "
            "1=approach 2=grasp 3=lift 4=mouth 5=feed 6=retreat q=quit > "
        ).strip().lower()
        if cmd == "":
            self.note = ""
            return
        if cmd == "r":
            self.rlt_enabled = not self.rlt_enabled
            self.rlt_toggle_on = self.rlt_enabled
            self.rlt_toggle_off = not self.rlt_enabled
            self.note = "rlt_on" if self.rlt_enabled else "rlt_off"
        elif cmd == "e":
            self.human_env_reset = True
            self.reset_type = input("[rlt] reset type, e.g. spoon_reposition > ").strip()
            self.note = "human_env_reset"
        elif cmd == "s":
            self.success = True
            self.failure = False
            self.note = "success"
        elif cmd == "f":
            self.failure = True
            self.success = False
            self.note = "failure"
        elif cmd == "c":
            self.collision = True
            self.note = "collision"
        elif cmd == "q":
            self.stop = True
            self.note = "quit"
        elif cmd in {"1", "2", "3", "4", "5", "6"}:
            phases = {
                "1": "approach_spoon",
                "2": "grasp_spoon",
                "3": "lift_food",
                "4": "approach_mouth",
                "5": "feed",
                "6": "retreat",
            }
            self.phase = phases[cmd]
            self.note = self.phase

    def make_label(self, exec_ok: bool, collision_from_checker: bool) -> ChunkLabel:
        label = ChunkLabel(
            success=self.success,
            failure=self.failure,
            collision=self.collision or collision_from_checker,
            exec_ok=exec_ok,
            rlt_enabled=self.rlt_enabled,
            rlt_toggle_on=self.rlt_toggle_on,
            rlt_toggle_off=self.rlt_toggle_off,
            human_env_reset=self.human_env_reset,
            cut_before_reset=self.human_env_reset,
            reset_type=self.reset_type,
            phase=self.phase,
            note=self.note,
        )
        self.success = False
        self.failure = False
        self.collision = False
        self.rlt_toggle_on = False
        self.rlt_toggle_off = False
        self.human_env_reset = False
        self.reset_type = ""
        self.note = ""
        return label
