# EPR integration notes for `utilities/test_the_server_basic_v3_as.py`

The recommended path for the current two-task feeding experiment is to run the
EPR episode collector directly instead of patching the original inference script:

```bash
PYTHONPATH=utilities python utilities/rlt_training/test_the_server_basic_v3_return_rlt_episode_semantic.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./epr_episode_replay_round1
```

The older `collect_rlt_toggle_v3.py` script is kept for comparison/debugging.
It still uses old RLT-style names internally.

If you do integrate manually, use the existing chunk boundary:

```python
obs = sensor.get_observation(prompt)
result = policy.infer(obs)
acts = result["actions"][:H]
seg = acts[d:d+s]
```

For EPR experiments, log these fields for every chunk:

```text
observation/image
observation/wrist_image
observation/state
pi_actions
executed_actions
delta_actions
epr_enabled / legacy rlt_enabled
epr_toggle_on / legacy rlt_toggle_on
epr_toggle_off / legacy rlt_toggle_off
manual_ee_offset_xyz
gripper_override
human_env_reset
success/failure/collision/exec_ok
```

Key meaning:

- `t`: toggle EPR/manual correction on/off.
- `e`: mark human environment reset, such as moving the spoon.
- `s/f/x`: success/failure/unsafe labels.

Run scripts from the openpi repository root with `PYTHONPATH=utilities` so the
legacy `rlt_training` package path resolves correctly:

```bash
PYTHONPATH=utilities python utilities/rlt_training/test_the_server_basic_v3_return_rlt_episode_semantic.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./epr_episode_replay_round1
```
