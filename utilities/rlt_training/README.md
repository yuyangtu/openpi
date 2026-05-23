# RLT adapter for pi0.5 feeding

This folder contains a simplified RLT-style residual actor-critic and replay-buffer workflow for the pi0.5 feeding inference loop.

The design is intentionally client-side:

1. Keep the official pi0.5 policy server unchanged.
2. Log observations, pi0.5 action chunks, executed chunks, RLT toggle labels, environment-reset labels, and rewards.
3. Train a simplified external RL token from camera observations, robot state, and the VLA action chunk.
4. Use that token with an actor-critic: actor predicts residual action edits; critic scores the VLA action plus residual.

Run commands from the openpi repository root. Because this package lives under `utilities/`, set `PYTHONPATH=utilities`.

## Recommended two-task episode collector

Use this for the real FEED/RETURN experiment. It records whole episodes and writes chunk `.npz` files only when the episode is labeled.

```bash
PYTHONPATH=utilities python utilities/test_the_server_basic_v3_return_rlt_episode.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_episode_replay_round0
```

Keys:

- Enter: continue current mode.
- `r`: switch to RETURN task.
- `c`: switch to FEED task.
- `t`: toggle RLT residual on/off.
- `s`: finish current episode as success.
- `f`: finish current episode as failure.
- `x`: finish current episode as unsafe/collision.
- `e`: finish current episode as human environment reset, then continue after moving the object.
- `n`: finish current episode as neutral and start a new one.
- `q`: quit.

For Round 1 small residual exploration:

```bash
PYTHONPATH=utilities python utilities/test_the_server_basic_v3_return_rlt_episode.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_episode_replay_round1 \
  --explore-delta-std 0.005 \
  --explore-delta-max 0.02
```

After training, deploy with a checkpoint:

```bash
PYTHONPATH=utilities python utilities/test_the_server_basic_v3_return_rlt_episode.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_episode_replay_round2 \
  --rlt-checkpoint ./rlt_checkpoints_toggle/best.pt \
  --rlt-max-delta 0.03
```

## Train actor-critic

```bash
PYTHONPATH=utilities python utilities/rlt_training/train_rlt_actor_critic.py \
  --replay-dir ./rlt_episode_replay_round1 \
  --output-dir ./rlt_checkpoints_toggle \
  --epochs 30 \
  --max-delta 0.03
```

## Older single-task toggle collector

The earlier collector still exists for simpler one-task experiments:

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round0
```

## Notes

For the two-task episode collector, you do not need to label every chunk as success/failure. Each chunk is recorded automatically. You only label episode outcomes with `s`, `f`, `x`, `e`, or `n`. The recorder propagates the final outcome backward as `discounted_return` so the actor-critic trainer can still learn from chunk files.

This is not a strict reproduction of the original RLT internal-token pipeline. It uses a simplified external token instead of extracting hidden VLA embeddings. The actor-critic deployment logic is RLT-style: VLA remains the reference, RLT predicts a small residual, and an anchor penalty keeps the residual close to the base VLA behavior.
