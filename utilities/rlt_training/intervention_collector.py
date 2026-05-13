from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class InterventionState:
    active: bool = False
    success: bool = False
    failure: bool = False
    stop: bool = False
    last_note: str = ""


class ConsoleInterventionController(threading.Thread):
    def __init__(self, state: InterventionState):
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        while not self.state.stop:
            cmd = input("[rlt] i=intervene s=success f=failure q=quit > ").strip().lower()
            if cmd == "i":
                self.state.active = not self.state.active
                self.state.last_note = "intervention_on" if self.state.active else "intervention_off"
            elif cmd == "s":
                self.state.success = True
                self.state.failure = False
                self.state.last_note = "success"
            elif cmd == "f":
                self.state.failure = True
                self.state.success = False
                self.state.last_note = "failure"
            elif cmd == "q":
                self.state.stop = True
                self.state.last_note = "quit"


class RLTChunkLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.jsonl"

    def _reward_from_flags(self, success: bool, failure: bool, collision: bool, needs_intervention: bool, executed_ok: bool) -> float:
        reward = 0.0
        if success:
            reward += 1.0
        if failure:
            reward -= 1.0
        if collision:
            reward -= 0.6
        if needs_intervention:
            reward -= 0.25
        if not executed_ok:
            reward -= 0.3
        return float(reward)

    def log_chunk(
        self,
        obs: dict,
        pi_actions: np.ndarray,
        executed_actions: np.ndarray,
        human_actions: np.ndarray | None,
        intervention_mask: np.ndarray | None,
        needs_intervention: bool,
        executed_ok: bool,
        collision: bool = False,
        success: bool = False,
        failure: bool = False,
        done: bool = False,
        note: str = "",
    ) -> Path:
        pi_actions = np.asarray(pi_actions, dtype=np.float32)
        executed_actions = np.asarray(executed_actions, dtype=np.float32)
        if human_actions is None:
            human_actions = executed_actions
        human_actions = np.asarray(human_actions, dtype=np.float32)
        if intervention_mask is None:
            intervention_mask = np.full((pi_actions.shape[0],), bool(needs_intervention), dtype=bool)
        intervention_mask = np.asarray(intervention_mask, dtype=bool)
        timestamp = time.time()
        reward = self._reward_from_flags(success, failure, collision, needs_intervention, executed_ok)
        path = self.output_dir / f"chunk_{int(timestamp * 1000)}.npz"
        np.savez_compressed(
            path,
            observation_image=np.asarray(obs["observation/image"], dtype=np.uint8),
            observation_wrist_image=np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            observation_state=np.asarray(obs["observation/state"], dtype=np.float32),
            pi_actions=pi_actions,
            executed_actions=executed_actions,
            human_actions=human_actions,
            intervention_mask=intervention_mask,
            reward=np.float32(reward),
            done=np.bool_(done),
            needs_intervention=np.bool_(needs_intervention),
            prompt=np.asarray(obs.get("prompt", ""), dtype=object),
            timestamp=np.float64(timestamp),
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"file": str(path), "timestamp": timestamp, "reward": reward, "needs_intervention": bool(needs_intervention), "executed_ok": bool(executed_ok), "collision": bool(collision), "success": bool(success), "failure": bool(failure), "done": bool(done), "note": note}) + "\n")
        return path
