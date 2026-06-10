# EPR adapter for pi0.5 feeding

This folder contains the EPR workflow used for the pi0.5 feeding inference loop.
EPR is an external episode-level policy residual system: it keeps the base pi0.5
VLA policy unchanged, records whole-episode replay data, and trains a small
client-side correction model from the collected outcomes and semantic corrections.

Some code paths, argument names, and checkpoint keys still use `rlt_*` for
backward compatibility with earlier experiments. Treat those names as legacy
implementation names; the current method here is EPR, not a strict reproduction
of the RLT paper.

The design is intentionally client-side:

1. Keep the official pi0.5 policy server unchanged.
2. Log observations, pi0.5 action chunks, executed chunks, EPR intervention labels, environment-reset labels, semantic corrections, and episode outcomes.
3. Train an external correction token from camera observations, robot state, and the VLA action chunk.
4. Use that token with an actor-critic style model: the actor predicts semantic corrections such as end-effector offsets and gripper choices, while the critic/value heads learn from episode-level returns.

Run commands from the openpi repository root. Because this package lives under `utilities/`, set `PYTHONPATH=utilities`.

## Recommended two-task EPR episode collector

Use this for the real FEED/RETURN experiment. It records whole episodes and writes chunk `.npz` files only when the episode is labeled.

```bash
PYTHONPATH=utilities python utilities/rlt_training/test_the_server_basic_v3_return_rlt_episode_semantic.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./epr_episode_replay_round0
```

Keys:

- Enter: continue current mode.
- `r`: switch to RETURN task.
- `c`: switch to FEED task.
- `t`: toggle EPR/manual correction on/off.
- `i/k`, `j/l`, `u/o`: adjust the manual end-effector offset.
- `g`: force gripper close for the next chunk.
- `b`: force gripper open for the next chunk.
- `a`: clear the gripper override.
- `s`: finish current episode as success.
- `f`: finish current episode as failure.
- `x`: finish current episode as unsafe/collision.
- `e`: finish current episode as human environment reset, then continue after moving the object.
- `n`: finish current episode as neutral and start a new one.
- `q`: quit.

For Round 1 manual semantic correction collection:

```bash
PYTHONPATH=utilities python utilities/rlt_training/test_the_server_basic_v3_return_rlt_episode_semantic.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./epr_episode_replay_round1 \
  --record-from-start false
```

After training, deploy with a checkpoint:

```bash
PYTHONPATH=utilities python utilities/rlt_training/test_the_server_basic_v3_return_rlt_episode_semantic.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./epr_episode_replay_round2 \
  --rlt-checkpoint ./epr_checkpoints/best.pt \
  --rlt-max-delta 0.03
```

## Train EPR actor-critic

```bash
PYTHONPATH=utilities python utilities/rlt_training/train_rlt_actor_critic.py \
  --replay-dir ./epr_episode_replay_round1 \
  --output-dir ./epr_checkpoints \
  --epochs 30 \
  --max-ee-offset 0.03
```

## Older single-task toggle collector

The earlier collector still exists for simpler one-task experiments. It keeps
the old RLT-style naming and is mainly useful for comparison/debugging:

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round0
```

## Notes

For the two-task episode collector, you do not need to label every chunk as success/failure. Each chunk is recorded automatically. You only label episode outcomes with `s`, `f`, `x`, `e`, or `n`. The recorder propagates the final outcome backward as `discounted_return` so the actor-critic trainer can still learn from chunk files.

EPR is not a strict reproduction of the original RLT internal-token pipeline.
It uses a client-side external token/correction model instead of extracting
hidden VLA embeddings. The base VLA remains the reference policy, and EPR
learns when and how to apply small semantic corrections from replayed episodes.
