"""Evaluation checks for generated oxDNA conformations."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .oxdna_geometry import backbone_bond_lengths, orientation_quality, rmsd


def shape_checks(samples: Dict[str, torch.Tensor], reference: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
    positions = samples["positions"]
    a1 = samples["a1"]
    a3 = samples["a3"]
    out: Dict[str, object] = {
        "positions_shape": list(positions.shape),
        "a1_shape": list(a1.shape),
        "a3_shape": list(a3.shape),
        "finite": bool(torch.isfinite(positions).all() and torch.isfinite(a1).all() and torch.isfinite(a3).all()),
    }
    if reference is not None:
        out["matches_reference_position_shape"] = list(positions.shape) == list(reference["target_positions"].shape)
        out["matches_reference_orientation_shape"] = list(a1.shape) == list(reference["target_a1"].shape)
    return out


def ensemble_centroid_metrics(
    generated_positions: torch.Tensor,
    ensemble_positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute RMSD/distance to a target ensemble centroid."""
    if ensemble_positions.ndim == 3:
        centroid = ensemble_positions.mean(dim=0)
    elif ensemble_positions.ndim == 4:
        centroid = ensemble_positions.mean(dim=1)
    else:
        raise ValueError(f"Expected ensemble positions rank 3 or 4, got {ensemble_positions.shape}")
    if generated_positions.ndim == 3 and centroid.ndim == 2:
        centroid = centroid.unsqueeze(0).expand_as(generated_positions)
    value = rmsd(generated_positions, centroid, mask=mask, align=True)
    raw = rmsd(generated_positions, centroid, mask=mask, align=False)
    return {
        "rmsd_to_centroid_aligned": float(value.detach().cpu()),
        "rmsd_to_centroid_raw": float(raw.detach().cpu()),
    }


def bond_length_metrics(
    positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_length: float = 0.6,
    tolerance: float = 0.15,
) -> Dict[str, float]:
    lengths = backbone_bond_lengths(positions, mask)
    if lengths.numel() == 0:
        return {
            "backbone_bond_mean": 0.0,
            "backbone_bond_std": 0.0,
            "backbone_bond_min": 0.0,
            "backbone_bond_max": 0.0,
            "backbone_bond_violation_fraction": 0.0,
        }
    violations = (lengths < target_length - tolerance) | (lengths > target_length + tolerance)
    return {
        "backbone_bond_mean": float(lengths.mean().detach().cpu()),
        "backbone_bond_std": float(lengths.std(unbiased=False).detach().cpu()),
        "backbone_bond_min": float(lengths.min().detach().cpu()),
        "backbone_bond_max": float(lengths.max().detach().cpu()),
        "backbone_bond_violation_fraction": float(violations.float().mean().detach().cpu()),
    }


def optional_energy_metrics(samples: Dict[str, torch.Tensor], batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
    """Try optional oxDNA energy evaluation if an installed implementation exists."""
    try:
        from flowDNA.models.energy_wrapper import compute_oxdna_energy  # type: ignore
    except Exception:
        return {"energy_available": False}

    try:
        energy = compute_oxdna_energy(samples, batch)
    except Exception as exc:
        return {"energy_available": True, "energy_error": str(exc)}
    if isinstance(energy, torch.Tensor):
        energy = energy.detach().cpu()
        return {"energy_available": True, "energy_mean": float(energy.float().mean())}
    return {"energy_available": True, "energy": energy}


def evaluate_generated(
    samples: Dict[str, torch.Tensor],
    reference_batch: Optional[Dict[str, torch.Tensor]] = None,
    target_bond_length: float = 0.6,
    bond_tolerance: float = 0.15,
    run_energy: bool = False,
) -> Dict[str, object]:
    """Run oxDNA-specific generated sample checks."""
    mask = None if reference_batch is None else reference_batch.get("mask")
    metrics: Dict[str, object] = {}
    metrics.update(shape_checks(samples, reference_batch))
    metrics.update(bond_length_metrics(samples["positions"], mask, target_bond_length, bond_tolerance))
    metrics.update(orientation_quality(samples["a1"], samples["a3"], mask))
    if reference_batch is not None:
        metrics.update(ensemble_centroid_metrics(samples["positions"], reference_batch["target_positions"], mask))
    if run_energy:
        metrics.update(optional_energy_metrics(samples, reference_batch))
    return metrics
