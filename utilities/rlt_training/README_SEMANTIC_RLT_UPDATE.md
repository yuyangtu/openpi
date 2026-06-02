# Semantic RLT update

Copy these files into your existing codebase.

## Replace

```bash
cp rlt_training/rlt_actor_critic.py <YOUR_CODE>/rlt_training/rlt_actor_critic.py
cp rlt_training/rlt_value_dataset.py <YOUR_CODE>/rlt_training/rlt_value_dataset.py
cp rlt_training/train_rlt_actor_critic.py <YOUR_CODE>/rlt_training/train_rlt_actor_critic.py
cp rlt_training/runtime_adapter.py <YOUR_CODE>/rlt_training/runtime_adapter.py
cp rlt_training/episode_replay_buffer.py <YOUR_CODE>/rlt_training/episode_replay_buffer.py
```

If your GitHub layout is `utilities/rlt_training`, copy into `utilities/rlt_training/`.

## What changed

Old actor output:
```text
delta_actions: [horizon, action_dim]
```

New actor output:
```text
ee_offset: [3]
gripper_logits: [none, close, open]
```

Runtime returns semantic prediction in `info`:
```python
info["ee_offset"]
info["gripper_override"]
```

The robot collector/deployment script should then convert `ee_offset` to joint residual using the right-arm Jacobian and apply optional gripper override.

## Train

```bash
PYTHONPATH=$PWD python3 rlt_training/train_rlt_actor_critic.py \
  --replay-dir ./rlt_episode_replay_mix_h12 \
  --output-dir ./rlt_checkpoints_semantic_h12 \
  --epochs 50 \
  --batch-size 16 \
  --horizon 12 \
  --action-dim 8 \
  --state-dim 8 \
  --max-ee-offset 0.03 \
  --ee-weight 100.0 \
  --gripper-weight 1.0 \
  --anchor-weight 10.0 \
  --actor-q-weight 0.0 \
  --device cuda
```

The dataset can use new fields:
```text
manual_ee_offset_xyz
gripper_override_id
```

For older data, it tries to parse:
```text
note="...ee_offset=[...];...gripper_override=close..."
```
