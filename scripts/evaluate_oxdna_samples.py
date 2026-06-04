#!/usr/bin/env python3
"""Evaluate generated oxDNA samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oxbb.oxdna_eval import evaluate_generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="NPZ from sample_oxdna_flow.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--target_bond_length", type=float, default=0.6)
    parser.add_argument("--bond_tolerance", type=float, default=0.15)
    parser.add_argument("--run_energy", action="store_true")
    args = parser.parse_args()

    data = np.load(args.samples)
    samples = {
        "positions": torch.as_tensor(data["positions"], dtype=torch.float32),
        "a1": torch.as_tensor(data["a1"], dtype=torch.float32),
        "a3": torch.as_tensor(data["a3"], dtype=torch.float32),
    }
    reference = None
    if "target_positions" in data and "mask" in data:
        reference = {
            "target_positions": torch.as_tensor(data["target_positions"], dtype=torch.float32),
            "target_a1": torch.as_tensor(data["target_a1"] if "target_a1" in data else data["a1"], dtype=torch.float32),
            "target_a3": torch.as_tensor(data["target_a3"] if "target_a3" in data else data["a3"], dtype=torch.float32),
            "mask": torch.as_tensor(data["mask"], dtype=torch.bool),
        }
    metrics = evaluate_generated(
        samples,
        reference,
        target_bond_length=args.target_bond_length,
        bond_tolerance=args.bond_tolerance,
        run_energy=args.run_energy,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
