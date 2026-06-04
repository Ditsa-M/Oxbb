#!/usr/bin/env python3
"""Train an oxDNA conditional flow-matching model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oxbb.oxdna_dataset import OxDNAZarrDataset, oxdna_collate_fn, split_dataset
from oxbb.oxdna_flow_model import OxdnaConditionalFlow, model_config_from_dict


def load_config(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oxdna.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_structures", type=int, default=None)
    parser.add_argument("--max_nucleotides", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    for key in ("data_dir", "max_structures", "max_nucleotides"):
        value = getattr(args, key)
        if value is not None:
            data_cfg[key] = value
    for key in ("epochs", "batch_size", "lr", "checkpoint_dir", "limit_batches"):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = OxDNAZarrDataset(**data_cfg)
    train_set, val_set = split_dataset(dataset, train_cfg.get("val_fraction", 0.1), train_cfg.get("seed", 7))
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=oxdna_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
            collate_fn=oxdna_collate_fn,
        )

    model = OxdnaConditionalFlow(model_config_from_dict(model_cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = {}
        steps = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(iterator, start=1):
            if train_cfg.get("limit_batches") and batch_idx > train_cfg["limit_batches"]:
                break
            batch = move_batch(batch, device)
            losses = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            optimizer.step()
            steps += 1
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())
            iterator.set_postfix(loss=running["loss"] / steps)

        record = {f"train_{key}": value / max(steps, 1) for key, value in running.items()}
        if val_loader is not None:
            model.eval()
            val_running = {}
            val_steps = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader, start=1):
                    if train_cfg.get("limit_batches") and batch_idx > train_cfg["limit_batches"]:
                        break
                    losses = model.loss(move_batch(batch, device))
                    val_steps += 1
                    for key, value in losses.items():
                        val_running[key] = val_running.get(key, 0.0) + float(value.detach().cpu())
            record.update({f"val_{key}": value / max(val_steps, 1) for key, value in val_running.items()})

        history.append(record)
        print(json.dumps({"epoch": epoch, **record}, sort_keys=True))
        if epoch % train_cfg.get("save_every", 5) == 0 or epoch == train_cfg["epochs"]:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "history": history,
            }
            torch.save(ckpt, checkpoint_dir / f"oxdna_flow_epoch_{epoch}.pt")
            torch.save(ckpt, checkpoint_dir / "latest.pt")


if __name__ == "__main__":
    main()
