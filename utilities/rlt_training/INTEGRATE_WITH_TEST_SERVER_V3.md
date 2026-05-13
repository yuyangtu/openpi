# Integration notes for `utilities/test_the_server_basic_v3_as.py`

The recommended path is now to run `utilities/rlt_training/collect_rlt_toggle_v3.py` directly instead of patching the original inference script.

If you do integrate manually, use the existing chunk boundary:

```python
obs = sensor.get_observation(prompt)
result = policy.infer(obs)
acts = result["actions"][:H]
seg = acts[d:d+s]
```

For RLT-toggle experiments, log these fields for every chunk:

```text
observation/image
observation/wrist_image
observation/state
pi_actions
executed_actions
delta_actions
rlt_enabled
rlt_toggle_on
rlt_toggle_off
human_env_reset
success/failure/collision/exec_ok
```

Key meaning:

- `r`: toggle RLT intervention on/off.
- `e`: mark human environment reset, such as moving the spoon.
- `s/f/c`: success/failure/collision labels.

Run scripts from the openpi repository root with `PYTHONPATH=utilities` so the `rlt_training` package resolves correctly:

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round0
```
