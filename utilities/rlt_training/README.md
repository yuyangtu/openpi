# RLT adapter for pi0.5 feeding

This folder contains a simplified RLT-style residual actor-critic and replay-buffer workflow for the pi0.5 feeding inference loop.

The design is intentionally client-side:

1. Keep the official pi0.5 policy server unchanged.
2. Log observations, pi0.5 action chunks, executed chunks, RLT toggle labels, environment-reset labels, and rewards.
3. Train a simplified external RL token from camera observations, robot state, and the VLA action chunk.
4. Use that token with an actor-critic: actor predicts residual action edits; critic scores the VLA action plus residual.

Run commands from the openpi repository root. Because this package lives under `utilities/`, set `PYTHONPATH=utilities`.

## Recommended flow

### Round 0: label RLT intervention timing

Run without a checkpoint. Press `r` when you think RLT should intervene; press `r` again when raw VLA can continue alone.

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round0
```

### Round 1: small residual exploration

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round1 \
  --explore-delta-std 0.005 \
  --explore-delta-max 0.02
```

### Train actor-critic

```bash
PYTHONPATH=utilities python utilities/rlt_training/train_rlt_actor_critic.py \
  --replay-dir ./rlt_replay_toggle_round1 \
  --output-dir ./rlt_checkpoints_toggle \
  --epochs 30 \
  --max-delta 0.03
```

### Round 2+: deploy trained RLT when toggled on

```bash
PYTHONPATH=utilities python utilities/rlt_training/collect_rlt_toggle_v3.py \
  --host 134.100.39.19 \
  --port 8000 \
  --replay-dir ./rlt_replay_toggle_round2 \
  --rlt-checkpoint ./rlt_checkpoints_toggle/best.pt \
  --rlt-max-delta 0.03
```

## Keys during collection

- Enter: continue with current mode.
- `r`: toggle RLT intervention on/off.
- `e`: mark a human environment reset, such as moving the spoon.
- `s`: mark success.
- `f`: mark failure.
- `c`: mark collision or unsafe behavior.
- `1`-`6`: phase labels: approach, grasp, lift, mouth, feed, retreat.
- `q`: stop.

## Notes

This is not a strict reproduction of the original RLT internal-token pipeline. It uses a simplified external token instead of extracting hidden VLA embeddings. The actor-critic deployment logic is RLT-style: VLA remains the reference, RLT predicts a small residual, and an anchor penalty keeps the residual close to the base VLA behavior.
