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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-delta", type=float, default=0.03)
    parser.add_argument("--max-ee-offset", type=float, default=0.03)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--ee-weight", type=float, default=100.0)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--critic-weight", type=float, default=0.1)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--actor-q-weight", type=float, default=0.0)
    parser.add_argument("--reference-dropout", type=float, default=0.15)
    parser.add_argument("--image-encoder", choices=["small", "dinov2_s"], default="small")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.set_defaults(freeze_image_encoder=True)
    parser.add_argument("--freeze-image-encoder", action="store_true", dest="freeze_image_encoder")
    parser.add_argument("--no-freeze-image-encoder", action="store_false", dest="freeze_image_encoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _target_gripper_onehot(target_gripper: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(target_gripper.clamp(0, 2), num_classes=3).float()


def critic_step(model, batch, optimizer, bce, args):
    pred = model(batch["top_image"], batch["wrist_image"], batch["state"], batch["pi_actions"])
    token = pred["rlt_token"].detach()
    target_onehot = _target_gripper_onehot(batch["target_gripper"])
    q1_data, q2_data = model.q_from_token(
        token,
        batch["state"],
        batch["pi_actions"],
        batch["target_ee_offset"],
        target_onehot,
    )
    target_q = batch["reward"]
    critic_loss = (q1_data - target_q).pow(2).mean() + (q2_data - target_q).pow(2).mean()
    gate_loss = bce(pred["gate_logit"], batch["should_enable_rlt"])
    loss = args.critic_weight * critic_loss + args.gate_weight * gate_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "q_mean": float(((q1_data + q2_data) * 0.5).detach().mean().cpu()),
    }


def actor_step(model, batch, optimizer, ce, args):
    pred = model(
        batch["top_image"],
        batch["wrist_image"],
        batch["state"],
        batch["pi_actions"],
        reference_dropout=args.reference_dropout,
    )

    q1, q2 = model.q_from_token(
        pred["rlt_token"],
        batch["state"],
        batch["pi_actions"],
        pred["ee_offset"],
        pred["gripper_onehot"],
    )
    q = torch.minimum(q1, q2)

    w = batch["bc_weight"].float()
    denom = w.sum().clamp_min(1.0)

    ee_err = (pred["ee_offset"] - batch["target_ee_offset"]).pow(2).sum(dim=-1)
    ee_loss = (ee_err * w).sum() / denom

    gripper_loss_per = ce(pred["gripper_logits"], batch["target_gripper"])
    gripper_loss = (gripper_loss_per * w).sum() / denom

    anchor_loss = pred["ee_offset"].pow(2).mean()

    actor_loss = (
        args.ee_weight * ee_loss
        + args.gripper_weight * gripper_loss
        + args.anchor_weight * anchor_loss
        - args.actor_q_weight * q.mean()
    )

    optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    pred_cls = torch.argmax(pred["gripper_logits"], dim=-1)
    bc_mask = w > 0.5
    if bc_mask.any():
        grip_acc = (pred_cls[bc_mask] == batch["target_gripper"][bc_mask]).float().mean()
    else:
        grip_acc = torch.tensor(0.0, device=pred_cls.device)

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_q": float(q.detach().mean().cpu()),
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "ee_loss": float(ee_loss.detach().cpu()),
        "gripper_loss": float(gripper_loss.detach().cpu()),
        "gripper_acc_bc": float(grip_acc.detach().cpu()),
        "ee_abs": float(pred["ee_offset"].detach().abs().mean().cpu()),
        "bc_frac": float((w > 0.5).float().mean().detach().cpu()),
    }


def eval_epoch(model, loader, device, bce, ce, args):
    model.eval()
    totals = {
        "critic_loss": 0.0,
        "gate_loss": 0.0,
        "actor_q": 0.0,
        "ee_loss": 0.0,
        "gripper_loss": 0.0,
        "gripper_acc_bc": 0.0,
        "ee_abs": 0.0,
        "bc_frac": 0.0,
        "n": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch["top_image"], batch["wrist_image"], batch["state"], batch["pi_actions"])
            token = pred["rlt_token"]
            target_onehot = _target_gripper_onehot(batch["target_gripper"])
            q1_data, q2_data = model.q_from_token(
                token,
                batch["state"],
                batch["pi_actions"],
                batch["target_ee_offset"],
                target_onehot,
            )
            critic_loss = (q1_data - batch["reward"]).pow(2).mean() + (q2_data - batch["reward"]).pow(2).mean()
            gate_loss = bce(pred["gate_logit"], batch["should_enable_rlt"])
            ee_loss = (pred["ee_offset"] - batch["target_ee_offset"]).pow(2).sum(dim=-1).mean()
            gripper_loss = ce(pred["gripper_logits"], batch["target_gripper"]).mean()
            pred_cls = torch.argmax(pred["gripper_logits"], dim=-1)
            w = batch["bc_weight"].float()
            bc_mask = w > 0.5
            grip_acc = (pred_cls[bc_mask] == batch["target_gripper"][bc_mask]).float().mean() if bc_mask.any() else torch.tensor(0.0, device=device)
            actor_q = torch.minimum(pred["q1"], pred["q2"]).mean()
            n = float(batch["state"].shape[0])
            for key, value in {
                "critic_loss": critic_loss,
                "gate_loss": gate_loss,
                "actor_q": actor_q,
                "ee_loss": ee_loss,
                "gripper_loss": gripper_loss,
                "gripper_acc_bc": grip_acc,
                "ee_abs": pred["ee_offset"].abs().mean(),
                "bc_frac": (w > 0.5).float().mean(),
            }.items():
                totals[key] += float(value.detach().cpu()) * n
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(train_ds if val_ds is None else val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds is not None else None

    model = RLTActorCritic(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.horizon,
        max_delta=args.max_delta,
        max_ee_offset=args.max_ee_offset,
        image_encoder=args.image_encoder,
        freeze_image_encoder=args.freeze_image_encoder,
        dinov2_model=args.dinov2_model,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(reduction="none")

    best = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "critic_loss": 0.0,
            "gate_loss": 0.0,
            "q_mean": 0.0,
            "actor_loss": 0.0,
            "actor_q": 0.0,
            "anchor_loss": 0.0,
            "ee_loss": 0.0,
            "gripper_loss": 0.0,
            "gripper_acc_bc": 0.0,
            "ee_abs": 0.0,
            "bc_frac": 0.0,
            "n": 0.0,
        }
        for batch in train_loader:
            batch = move_batch(batch, device)
            c = critic_step(model, batch, optimizer, bce, args)
            a = actor_step(model, batch, optimizer, ce, args)
            n = float(batch["state"].shape[0])
            for key, value in {**c, **a}.items():
                totals[key] += value * n
            totals["n"] += n

        train = {k: v / max(totals["n"], 1.0) for k, v in totals.items() if k != "n"}
        val = eval_epoch(model, val_loader, device, bce, ce, args) if val_loader is not None else {}
        score = val.get("ee_loss", train["ee_loss"]) + val.get("gripper_loss", train["gripper_loss"])
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        ckpt = {
            "kind": "rlt_actor_critic_semantic",
            "model": model.state_dict(),
            "args": vars(args),
            "spec": spec.__dict__,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if score < best:
            best = score
            torch.save(ckpt, output_dir / "best.pt")

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
