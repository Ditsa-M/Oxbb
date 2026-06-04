"""Geometry helpers for oxDNA position and orientation tensors.

oxDNA stores each nucleotide orientation as two body-frame unit vectors, a1 and
a3. Generated vectors must be projected back to a valid orthonormal frame before
being written out or passed to analysis tools.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors with a finite fallback for near-zero norms."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _fallback_axis(reference: torch.Tensor) -> torch.Tensor:
    """Choose an axis that is unlikely to be collinear with reference."""
    basis = torch.zeros_like(reference)
    abs_ref = reference.abs()
    axis_idx = abs_ref.argmin(dim=-1, keepdim=True)
    basis.scatter_(-1, axis_idx, 1.0)
    return basis


def project_a1_a3(
    a1: torch.Tensor,
    a3: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project raw a1/a3 vectors to unit, mutually orthogonal oxDNA axes.

    The projection uses Gram-Schmidt with a deterministic fallback when generated
    a3 is too close to a1. This is differentiable for normal cases and keeps
    sampled conformations valid enough for downstream oxDNA analysis.
    """
    a1_unit = safe_normalize(a1, eps=eps)
    a3_orth = a3 - (a3 * a1_unit).sum(dim=-1, keepdim=True) * a1_unit

    bad = a3_orth.norm(dim=-1, keepdim=True) < eps
    fallback = _fallback_axis(a1_unit)
    fallback = fallback - (fallback * a1_unit).sum(dim=-1, keepdim=True) * a1_unit
    a3_orth = torch.where(bad, fallback, a3_orth)
    a3_unit = safe_normalize(a3_orth, eps=eps)
    return a1_unit, a3_unit


def frame_from_a1_a3(a1: torch.Tensor, a3: torch.Tensor) -> torch.Tensor:
    """Build a right-handed 3x3 frame with columns [a1, a2, a3]."""
    a1_unit, a3_unit = project_a1_a3(a1, a3)
    a2_unit = safe_normalize(torch.cross(a3_unit, a1_unit, dim=-1))
    a3_unit = safe_normalize(torch.cross(a1_unit, a2_unit, dim=-1))
    return torch.stack((a1_unit, a2_unit, a3_unit), dim=-1)


def a1_a3_from_frame(orientation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract oxDNA a1 and a3 axes from a frame whose columns are body axes."""
    if orientation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected orientation shape (..., 3, 3), got {orientation.shape}")
    return project_a1_a3(orientation[..., :, 0], orientation[..., :, 2])


def center_positions(
    positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center positions by masked centroid and return centered positions, centroid."""
    if mask is None:
        centroid = positions.mean(dim=-2, keepdim=True)
    else:
        weights = mask.to(positions.dtype).unsqueeze(-1)
        centroid = (positions * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return positions - centroid, centroid


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    weights = mask.to(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    return (x * weights).sum() / weights.sum().clamp_min(1.0)


def kabsch_align(
    mobile: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rigidly align mobile positions onto target positions."""
    mobile_c, mobile_centroid = center_positions(mobile, mask)
    target_c, target_centroid = center_positions(target, mask)
    if mask is not None:
        weights = mask.to(mobile.dtype).unsqueeze(-1)
        mobile_c = mobile_c * weights
        target_c = target_c * weights
    cov = mobile_c.transpose(-1, -2) @ target_c
    u, _, vh = torch.linalg.svd(cov)
    det = torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    fix = torch.ones((*det.shape, 3), device=mobile.device, dtype=mobile.dtype)
    fix[..., -1] = det.sign()
    rot = vh.transpose(-1, -2) @ torch.diag_embed(fix) @ u.transpose(-1, -2)
    return (mobile - mobile_centroid) @ rot + target_centroid


def rmsd(
    mobile: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    align: bool = True,
) -> torch.Tensor:
    """Compute masked RMSD, optionally after Kabsch alignment."""
    if align:
        mobile = kabsch_align(mobile, target, mask)
    sq = (mobile - target).pow(2).sum(dim=-1)
    return torch.sqrt(masked_mean(sq, mask).clamp_min(0.0))


def backbone_bond_lengths(
    positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bonds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return backbone bond lengths from explicit bonds or adjacent indices."""
    if bonds is not None and bonds.numel() > 0:
        return (positions[..., bonds[:, 1], :] - positions[..., bonds[:, 0], :]).norm(dim=-1)
    lengths = (positions[..., 1:, :] - positions[..., :-1, :]).norm(dim=-1)
    if mask is None:
        return lengths.reshape(-1)
    valid = mask[..., 1:] & mask[..., :-1]
    return lengths[valid]


def orientation_quality(
    a1: torch.Tensor,
    a3: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Summarize a1/a3 unit norm and orthogonality quality."""
    a1_norm = a1.norm(dim=-1)
    a3_norm = a3.norm(dim=-1)
    dot = (a1 * a3).sum(dim=-1).abs()
    if mask is not None:
        a1_norm = a1_norm[mask]
        a3_norm = a3_norm[mask]
        dot = dot[mask]
    if a1_norm.numel() == 0:
        return {
            "a1_norm_mean": 0.0,
            "a1_norm_max_error": 0.0,
            "a3_norm_mean": 0.0,
            "a3_norm_max_error": 0.0,
            "a1_a3_abs_dot_mean": 0.0,
            "a1_a3_abs_dot_max": 0.0,
        }
    return {
        "a1_norm_mean": float(a1_norm.mean().detach().cpu()),
        "a1_norm_max_error": float((a1_norm - 1.0).abs().max().detach().cpu()),
        "a3_norm_mean": float(a3_norm.mean().detach().cpu()),
        "a3_norm_max_error": float((a3_norm - 1.0).abs().max().detach().cpu()),
        "a1_a3_abs_dot_mean": float(dot.mean().detach().cpu()),
        "a1_a3_abs_dot_max": float(dot.max().detach().cpu()),
    }


def project_sample(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Project generated orientation fields in a sample dictionary in-place."""
    if "a1" in sample and "a3" in sample:
        sample["a1"], sample["a3"] = project_a1_a3(sample["a1"], sample["a3"])
        sample["orientations"] = frame_from_a1_a3(sample["a1"], sample["a3"])
    return sample
