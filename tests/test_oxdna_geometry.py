import torch

from oxbb.oxdna_eval import bond_length_metrics
from oxbb.oxdna_geometry import orientation_quality, project_a1_a3


def test_project_a1_a3_returns_unit_orthogonal_vectors():
    a1 = torch.randn(4, 12, 3)
    a3 = torch.randn(4, 12, 3)
    projected_a1, projected_a3 = project_a1_a3(a1, a3)

    assert torch.allclose(projected_a1.norm(dim=-1), torch.ones(4, 12), atol=1e-5)
    assert torch.allclose(projected_a3.norm(dim=-1), torch.ones(4, 12), atol=1e-5)
    assert torch.allclose((projected_a1 * projected_a3).sum(dim=-1), torch.zeros(4, 12), atol=1e-5)


def test_orientation_quality_reports_small_errors_after_projection():
    a1, a3 = project_a1_a3(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
    metrics = orientation_quality(a1, a3)

    assert metrics["a1_norm_max_error"] < 1e-5
    assert metrics["a3_norm_max_error"] < 1e-5
    assert metrics["a1_a3_abs_dot_max"] < 1e-5


def test_bond_length_metrics_uses_adjacent_backbone_fallback():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    metrics = bond_length_metrics(positions, mask, target_length=0.6, tolerance=0.05)

    assert abs(metrics["backbone_bond_mean"] - 0.6) < 1e-6
    assert metrics["backbone_bond_violation_fraction"] == 0.0
