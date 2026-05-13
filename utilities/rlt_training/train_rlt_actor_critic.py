from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from rlt_training.rlt_actor_critic import RLTActorCritic
from rlt_training.rlt_dataset import RLTBatchSpec
from rlt_training.rlt_value_dataset import RLTValueReplayDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-delta", type=float, default=0.03)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--reference-dropout", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def critic_step(model, batch, optimizer, bce, args):
    pred = model(batch["top_image"], batch["wrist_image"], batch["state"], batch["pi_actions"])
    token = pred["rlt_token"].detach()
    q1_data, q2_data = model.q_from_token(token, batch["state"], batch["pi_actions"], batch["target_delta"])
    target_q = batch["reward"]
    critic_loss = (q1_data - target_q).pow(2).mean() + (q2_data - target_q).pow(2).mean()
    gate_loss = bce(pred["gate_logit"], batch["should_enable_rlt"])
    loss = critic_loss + args.gate_weight * gate_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "q_mean": float(((q1_data + q2_data) * 0.5).detach().mean().cpu()),
    }


def actor_step(model, batch, optimizer, args):
    pred = model(
        batch["top_image"],
        batch["wrist_image"],
        batch["state"],
        batch["pi_actions"],
        reference_dropout=args.reference_dropout,
    )
    q1, q2 = model.q_from_token(pred["rlt_token"], batch["state"], batch["pi_actions"], pred["delta_actions"])
    q = torch.minimum(q1, q2)
    anchor_loss = pred["delta_actions"].pow(2).mean()
    delta_error = (pred["delta_actions"] - batch["target_delta"]).pow(2).sum(dim=-1)
    bc_denom = batch["bc_weight"].sum().clamp_min(1.0)
    bc_loss = (delta_error * batch["bc_weight"]).sum() / bc_denom
    actor_loss = -q.mean() + args.anchor_weight * anchor_loss + args.bc_weight * bc_loss
    optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_q": float(q.detach().mean().cpu()),
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "delta_abs": float(pred["delta_actions"].detach().abs().mean().cpu()),
    }


def eval_epoch(model, loader, device, bce):
    model.eval()
    totals = {"critic_loss": 0.0, "gate_loss": 0.0, "actor_q": 0.0, "delta_abs": 0.0, "n": 0.0}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch["top_image"], batch["wrist_image"], batch["state"], batch["pi_actions"])
            q1_data, q2_data = model.q_from_token(pred["rlt_token"], batch["state"], batch["pi_actions"], batch["target_delta"])
            critic_loss = (q1_data - batch["reward"]).pow(2).mean() + (q2_data - batch["reward"]).pow(2).mean()
            gate_loss = bce(pred["gate_logit"], batch["should_enable_rlt"])
            actor_q = torch.minimum(pred["q1"], pred["q2"]).mean()
            n = float(batch["state"].shape[0])
            totals["critic_loss"] += float(critic_loss.cpu()) * n
            totals["gate_loss"] += float(gate_loss.cpu()) * n
            totals["actor_q"] += float(actor_q.cpu()) * n
            totals["delta_abs"] += float(pred["delta_actions"].abs().mean().cpu()) * n
            totals["n"] += n
    return {k: v / max(totals["n"], 1.0) for k, v in totals.items() if k != "n"}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spec = RLTBatchSpec(horizon=args.horizon, action_dim=args.action_dim, image_size=args.image_size)
    dataset = RLTValueReplayDataset(args.replay_dir, spec=spec)
    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 4 else 0
    train_len = len(dataset) - val_len
    if val_len:
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(17))
    else:
        train_ds, val_ds = dataset, None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if val_ds is not None else None
    model = RLTActorCritic(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.horizon,
        max_delta=args.max_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"critic_loss": 0.0, "gate_loss": 0.0, "q_mean": 0.0, "actor_loss": 0.0, "actor_q": 0.0, "anchor_loss": 0.0, "bc_loss": 0.0, "delta_abs": 0.0, "n": 0.0}
        for batch in train_loader:
            batch = move_batch(batch, device)
            c = critic_step(model, batch, optimizer, bce, args)
            a = actor_step(model, batch, optimizer, args)
            n = float(batch["state"].shape[0])
            for key, value in {**c, **a}.items():
                totals[key] += value * n
            totals["n"] += n
        train = {k: v / max(totals["n"], 1.0) for k, v in totals.items() if k != "n"}
        val = eval_epoch(model, val_loader, device, bce) if val_loader is not None else {}
        score = val.get("critic_loss", train["critic_loss"]) + val.get("gate_loss", train["gate_loss"])
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        ckpt = {"kind": "rlt_actor_critic", "model": model.state_dict(), "args": vars(args), "spec": spec.__dict__, "epoch": epoch, "metrics": row}
        torch.save(ckpt, output_dir / "last.pt")
        if score < best:
            best = score
            torch.save(ckpt, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
