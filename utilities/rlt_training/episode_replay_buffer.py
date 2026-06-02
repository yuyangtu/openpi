from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


OUTCOME_REWARD = {
    "success": 1.0,
    "failure": -1.0,
    "unsafe": -0.8,
    "env_reset": -0.2,
    "aborted": -0.1,
    "neutral": 0.0,
}

GRIPPER_OVERRIDE_TO_ID = {
    "none": 0,
    "close": 1,
    "open": 2,
}


@dataclasses.dataclass
class EpisodeChunk:
    obs: dict[str, Any]
    pi_actions: np.ndarray
    executed_actions: np.ndarray
    delta_actions: np.ndarray
    task_mode: str
    prompt: str
    rlt_enabled: bool
    rlt_toggle_on: bool
    rlt_toggle_off: bool
    exec_ok: bool
    collision: bool
    human_env_reset: bool
    infer_dt: float
    exec_dt: float
    timestamp: float
    note: str = ""
    manual_ee_offset_xyz: np.ndarray | None = None
    gripper_override: str = "none"


class EpisodeReplayRecorder:
    def __init__(self, root: str | Path, gamma: float = 0.97):
        self.root = Path(root)
        self.episode_root = self.root / "episodes"
        self.episode_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "episode_index.jsonl"
        self.gamma = float(gamma)
        self.episode_counter = 0
        self.current_id = self._new_episode_id()
        self.chunks: list[EpisodeChunk] = []
        self.started_at = time.time()

    def _new_episode_id(self) -> str:
        self.episode_counter += 1
        return f"ep_{int(time.time())}_{self.episode_counter:04d}"

    def add_chunk(self, chunk: EpisodeChunk) -> None:
        self.chunks.append(chunk)

    def start_new_episode(self) -> str:
        self.current_id = self._new_episode_id()
        self.chunks = []
        self.started_at = time.time()
        return self.current_id

    def finish_episode(self, outcome: str, note: str = "") -> Path | None:
        outcome = outcome if outcome in OUTCOME_REWARD else "neutral"
        if not self.chunks:
            self.start_new_episode()
            return None

        ep_dir = self.episode_root / self.current_id
        chunk_dir = ep_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        final_reward = float(OUTCOME_REWARD[outcome])
        n = len(self.chunks)
        chunk_files = []

        for i, chunk in enumerate(self.chunks):
            discounted_return = final_reward * (self.gamma ** (n - 1 - i))
            is_last = i == n - 1
            reward = final_reward if is_last else 0.0
            success = outcome == "success"
            failure = outcome in {"failure", "unsafe"}
            path = chunk_dir / f"chunk_{i:06d}.npz"

            manual_ee = chunk.manual_ee_offset_xyz
            if manual_ee is None:
                manual_ee = np.zeros((3,), dtype=np.float32)
            manual_ee = np.asarray(manual_ee, dtype=np.float32).reshape(3)

            gripper_name = str(chunk.gripper_override or "none").lower()
            gripper_id = int(GRIPPER_OVERRIDE_TO_ID.get(gripper_name, 0))

            np.savez_compressed(
                path,
                observation_image=np.asarray(chunk.obs["observation/image"], dtype=np.uint8),
                observation_wrist_image=np.asarray(chunk.obs["observation/wrist_image"], dtype=np.uint8),
                observation_state=np.asarray(chunk.obs["observation/state"], dtype=np.float32),
                pi_actions=np.asarray(chunk.pi_actions, dtype=np.float32),
                executed_actions=np.asarray(chunk.executed_actions, dtype=np.float32),
                delta_actions=np.asarray(chunk.delta_actions, dtype=np.float32),

                manual_ee_offset_xyz=manual_ee,
                gripper_override=np.asarray(gripper_name, dtype=object),
                gripper_override_id=np.int64(gripper_id),

                reward=np.float32(reward),
                discounted_return=np.float32(discounted_return),
                episode_return=np.float32(final_reward),
                success=np.bool_(success),
                failure=np.bool_(failure),
                collision=np.bool_(chunk.collision or outcome == "unsafe"),
                exec_ok=np.bool_(chunk.exec_ok),
                rlt_enabled=np.bool_(chunk.rlt_enabled),
                rlt_toggle_on=np.bool_(chunk.rlt_toggle_on),
                rlt_toggle_off=np.bool_(chunk.rlt_toggle_off),
                human_env_reset=np.bool_(chunk.human_env_reset or outcome == "env_reset"),
                cut_before_reset=np.bool_(outcome == "env_reset"),
                task_mode=np.asarray(chunk.task_mode, dtype=object),
                prompt=np.asarray(chunk.prompt, dtype=object),
                episode_id=np.asarray(self.current_id, dtype=object),
                episode_outcome=np.asarray(outcome, dtype=object),
                chunk_id=np.int64(i),
                is_last=np.bool_(is_last),
                infer_dt=np.float32(chunk.infer_dt),
                exec_dt=np.float32(chunk.exec_dt),
                timestamp=np.float64(chunk.timestamp),
                note=np.asarray(chunk.note, dtype=object),
                phase_id=np.int64(0),
            )
            chunk_files.append(str(path))

        metadata = {
            "episode_id": self.current_id,
            "outcome": outcome,
            "final_reward": final_reward,
            "num_chunks": n,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "note": note,
            "chunk_files": chunk_files,
            "task_modes": [c.task_mode for c in self.chunks],
            "rlt_chunks": int(sum(1 for c in self.chunks if c.rlt_enabled)),
        }
        (ep_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        self.start_new_episode()
        return ep_dir
