from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from rlt_training.rlt_dataset import RLTBatchSpec
from rlt_training.rlt_model import RLTActionAdapter
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
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--bc-loss-weight", type=float, default=0.05)
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "value": 0.0, "risk": 0.0, "gate": 0.0, "bc": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["top_image"], batch["wrist_image"], batch["state"], batch["pi_actions"])
        value_loss = (pred["value"] - batch["reward"]).pow(2).mean()
        risk_loss = bce(pred["risk_logit"], batch["risk"])
        gate_loss = bce(pred["risk_logit"], batch["should_enable_rlt"])
        delta_error = (pred["delta_actions"] - batch["target_delta"]).pow(2).sum(dim=-1)
        bc_denom = batch["bc_weight"].sum().clamp_min(1.0)
        bc_loss = (delta_error * batch["bc_weight"]).sum() / bc_denom
        loss = args.value_loss_weight * value_loss + args.risk_loss_weight * risk_loss + args.gate_loss_weight * gate_loss + args.bc_loss_weight * bc_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        n = float(batch["state"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["value"] += float(value_loss.detach().cpu()) * n
        totals["risk"] += float(risk_loss.detach().cpu()) * n
        totals["gate"] += float(gate_loss.detach().cpu()) * n
        totals["bc"] += float(bc_loss.detach().cpu()) * n
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
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(11))
    else:
        train_ds, val_ds = dataset, None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if val_ds is not None else None
    model = RLTActionAdapter(state_dim=args.state_dim, action_dim=args.action_dim, horizon=args.horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, optimizer, device, args)
        if val_loader is not None:
            with torch.no_grad():
                val = run_epoch(model, val_loader, None, device, args)
            score = val["loss"]
        else:
            val = {}
            score = train["loss"]
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        ckpt = {"model": model.state_dict(), "args": vars(args), "spec": spec.__dict__, "epoch": epoch, "metrics": row}
        torch.save(ckpt, output_dir / "last.pt")
        if score < best:
            best = score
            torch.save(ckpt, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
