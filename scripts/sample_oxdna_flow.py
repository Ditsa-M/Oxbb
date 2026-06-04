#!/usr/bin/env python3
"""Generate oxDNA conformations from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oxbb.oxdna_dataset import OxDNAZarrDataset, oxdna_collate_fn
from oxbb.oxdna_flow_model import OxdnaConditionalFlow, model_config_from_dict


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="/scratch/sroy85/fastDNA/oxdna_zarr_dataset")
    parser.add_argument("--output", default="outputs/oxdna_samples.npz")
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--max_nucleotides", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    model = OxdnaConditionalFlow(model_config_from_dict(cfg.get("model", {}))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = OxDNAZarrDataset(args.data_dir, max_nucleotides=args.max_nucleotides, random_rotate=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=oxdna_collate_fn)
    output = {"positions": [], "a1": [], "a3": [], "target_positions": [], "target_a1": [], "target_a3": [], "mask": []}
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_batches:
                break
            batch = move_batch(batch, device)
            sample = model.sample(batch, num_steps=args.num_steps)
            for key in ("positions", "a1", "a3"):
                output[key].append(sample[key].detach().cpu().numpy())
            output["target_positions"].append(batch["target_positions"].detach().cpu().numpy())
            output["target_a1"].append(batch["target_a1"].detach().cpu().numpy())
            output["target_a3"].append(batch["target_a3"].detach().cpu().numpy())
            output["mask"].append(batch["mask"].detach().cpu().numpy())

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.concatenate(value, axis=0) for key, value in output.items()})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
