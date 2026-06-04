"""BBFlow-style conditional flow matching model for oxDNA conformations."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .oxdna_geometry import frame_from_a1_a3, project_a1_a3


@dataclass
class OxdnaFlowConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    base_embed_dim: int = 32
    dropout: float = 0.0
    conditional_prior_gamma: float = 0.25
    position_noise_scale: float = 1.0
    self_condition: bool = False
    position_loss_weight: float = 1.0
    orientation_loss_weight: float = 0.25
    bond_loss_weight: float = 0.05
    target_bond_length: float = 0.6


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class OxdnaEndpointDenoiser(nn.Module):
    """Predict relaxed endpoint positions and orientations from a flow state."""

    def __init__(self, cfg: OxdnaFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_embedding = nn.Embedding(4, cfg.base_embed_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        feature_dim = 3 + 6 + 3 + 6 + cfg.base_embed_dim
        if cfg.self_condition:
            feature_dim += 3 + 6

        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)
        self.position_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 3),
        )
        self.orientation_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 6),
        )

    def forward(
        self,
        positions_t: torch.Tensor,
        a1_t: torch.Tensor,
        a3_t: torch.Tensor,
        initial_positions: torch.Tensor,
        initial_a1: torch.Tensor,
        initial_a3: torch.Tensor,
        bases: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        self_condition: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        base_features = self.base_embedding(bases.clamp(min=0, max=3))
        features = [
            positions_t,
            a1_t,
            a3_t,
            initial_positions,
            initial_a1,
            initial_a3,
            base_features,
        ]
        if self.cfg.self_condition:
            if self_condition is None:
                features.extend([torch.zeros_like(positions_t), torch.zeros_like(a1_t), torch.zeros_like(a3_t)])
            else:
                features.extend([self_condition["positions"], self_condition["a1"], self_condition["a3"]])

        x = self.input_projection(torch.cat(features, dim=-1))
        t_embed = self.time_embedding(t).unsqueeze(1)
        x = x + t_embed
        padding_mask = None if mask is None else ~mask.bool()
        x = self.trunk(x, src_key_padding_mask=padding_mask)
        pred_positions = self.position_head(x)
        raw_orientation = self.orientation_head(x)
        pred_a1, pred_a3 = project_a1_a3(raw_orientation[..., :3], raw_orientation[..., 3:])
        if mask is not None:
            pred_positions = pred_positions * mask.unsqueeze(-1)
            pred_a1 = torch.where(mask.unsqueeze(-1), pred_a1, initial_a1)
            pred_a3 = torch.where(mask.unsqueeze(-1), pred_a3, initial_a3)
        return {
            "positions": pred_positions,
            "a1": pred_a1,
            "a3": pred_a3,
            "orientations": frame_from_a1_a3(pred_a1, pred_a3),
        }


class OxdnaConditionalFlow(nn.Module):
    """Conditional endpoint-prediction flow matching for oxDNA.

    The prior is a BBFlow-style conditional prior: Gaussian position noise and
    random orientation vectors are interpolated partway toward the unrelaxed
    conditioning structure. The model predicts the clean relaxed endpoint from
    the intermediate state. Sampling follows the rectified-flow Euler update.
    """

    def __init__(self, cfg: Optional[OxdnaFlowConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or OxdnaFlowConfig()
        self.denoiser = OxdnaEndpointDenoiser(self.cfg)

    def sample_prior(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        initial_pos = batch["initial_positions"]
        mask = batch["mask"]
        noise = torch.randn_like(initial_pos) * self.cfg.position_noise_scale
        noise = noise - (noise * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        raw_a1 = torch.randn_like(batch["initial_a1"])
        raw_a3 = torch.randn_like(batch["initial_a3"])
        prior_a1, prior_a3 = project_a1_a3(raw_a1, raw_a3)
        gamma = self.cfg.conditional_prior_gamma
        prior_pos = (1.0 - gamma) * noise + gamma * initial_pos
        prior_a1, prior_a3 = project_a1_a3(
            (1.0 - gamma) * prior_a1 + gamma * batch["initial_a1"],
            (1.0 - gamma) * prior_a3 + gamma * batch["initial_a3"],
        )
        return {"positions": prior_pos * mask.unsqueeze(-1), "a1": prior_a1, "a3": prior_a3}

    def corrupt_batch(
        self,
        batch: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prior = self.sample_prior(batch)
        if t is None:
            t = torch.rand(batch["target_positions"].shape[0], 1, device=batch["target_positions"].device)
        t_exp = t.unsqueeze(-1)
        positions_t = (1.0 - t_exp) * prior["positions"] + t_exp * batch["target_positions"]
        a1_t, a3_t = project_a1_a3(
            (1.0 - t_exp) * prior["a1"] + t_exp * batch["target_a1"],
            (1.0 - t_exp) * prior["a3"] + t_exp * batch["target_a3"],
        )
        return {
            "t": t,
            "positions_t": positions_t * batch["mask"].unsqueeze(-1),
            "a1_t": a1_t,
            "a3_t": a3_t,
            "prior_positions": prior["positions"],
            "prior_a1": prior["a1"],
            "prior_a3": prior["a3"],
        }

    def forward(self, batch: Dict[str, torch.Tensor], noisy: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        noisy = noisy or self.corrupt_batch(batch)
        return self.denoiser(
            noisy["positions_t"],
            noisy["a1_t"],
            noisy["a3_t"],
            batch["initial_positions"],
            batch["initial_a1"],
            batch["initial_a3"],
            batch["bases"],
            noisy["t"],
            batch["mask"],
        )

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noisy = self.corrupt_batch(batch)
        pred = self.forward(batch, noisy)
        mask = batch["mask"].to(pred["positions"].dtype)
        pos_sq = (pred["positions"] - batch["target_positions"]).pow(2).sum(dim=-1)
        pos_loss = (pos_sq * mask).sum() / mask.sum().clamp_min(1.0)
        ori_sq = (
            (pred["a1"] - batch["target_a1"]).pow(2).sum(dim=-1)
            + (pred["a3"] - batch["target_a3"]).pow(2).sum(dim=-1)
        )
        ori_loss = (ori_sq * mask).sum() / mask.sum().clamp_min(1.0)
        bond_loss = self._bond_length_loss(pred["positions"], batch["mask"])
        total = (
            self.cfg.position_loss_weight * pos_loss
            + self.cfg.orientation_loss_weight * ori_loss
            + self.cfg.bond_loss_weight * bond_loss
        )
        return {"loss": total, "position_loss": pos_loss, "orientation_loss": ori_loss, "bond_loss": bond_loss}

    def _bond_length_loss(self, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if positions.shape[1] < 2:
            return positions.new_tensor(0.0)
        lengths = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)
        valid = mask[:, 1:] & mask[:, :-1]
        if valid.sum() == 0:
            return positions.new_tensor(0.0)
        return (lengths[valid] - self.cfg.target_bond_length).abs().mean()

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        state = self.sample_prior(batch)
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t_value = torch.full((batch["mask"].shape[0], 1), step / float(num_steps), device=batch["mask"].device)
            noisy = {"positions_t": state["positions"], "a1_t": state["a1"], "a3_t": state["a3"], "t": t_value}
            pred = self.forward(batch, noisy)
            denom = max(1.0 - step / float(num_steps), 1e-3)
            state["positions"] = state["positions"] + dt * (pred["positions"] - state["positions"]) / denom
            state["a1"], state["a3"] = project_a1_a3(
                state["a1"] + dt * (pred["a1"] - state["a1"]) / denom,
                state["a3"] + dt * (pred["a3"] - state["a3"]) / denom,
            )
            state["positions"] = state["positions"] * batch["mask"].unsqueeze(-1)
        state["orientations"] = frame_from_a1_a3(state["a1"], state["a3"])
        return state


def model_config_from_dict(values: Dict) -> OxdnaFlowConfig:
    valid = {field.name for field in fields(OxdnaFlowConfig)}
    return OxdnaFlowConfig(**{key: value for key, value in values.items() if key in valid})
