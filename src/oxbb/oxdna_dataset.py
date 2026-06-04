"""Datasets for oxDNA conformation and trajectory data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .oxdna_geometry import a1_a3_from_frame, center_positions, frame_from_a1_a3, project_a1_a3


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


@dataclass(frozen=True)
class OxdnaDataConfig:
    data_dir: str
    initial_group: str = "initial"
    target_group: str = "target"
    max_structures: Optional[int] = None
    max_nucleotides: Optional[int] = 256
    min_nucleotides: int = 1
    frame_strategy: str = "random"
    center: bool = True
    random_rotate: bool = True


def _import_zarr():
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise ImportError("OxDNAZarrDataset requires zarr. Install with `pip install zarr`.") from exc
    return zarr


def _group_candidates(name: str) -> List[str]:
    aliases = {
        "initial": ["initial", "stage_0", "0"],
        "target": ["target", "stage_2", "2", "stage_3", "3"],
    }
    return aliases.get(name, [name, f"stage_{name}"])


def _has_key(group, key: str) -> bool:
    try:
        return key in group
    except Exception:
        return False


def _resolve_group(root, requested: str):
    for key in _group_candidates(requested):
        if _has_key(root, key):
            return root[key]
    available = list(root.keys()) if hasattr(root, "keys") else []
    raise KeyError(f"Could not find group {requested!r}; available groups: {available}")


def _array_to_numpy(array, frame_idx: Optional[int] = None, nucleotide_slice: Optional[slice] = None) -> np.ndarray:
    ndim = len(array.shape)
    if frame_idx is not None and ndim >= 2:
        if nucleotide_slice is None:
            return np.asarray(array[frame_idx])
        return np.asarray(array[frame_idx, nucleotide_slice])
    if nucleotide_slice is not None and ndim >= 1:
        return np.asarray(array[nucleotide_slice])
    return np.asarray(array[:])


def _num_frames(group) -> int:
    if _has_key(group, "positions"):
        shape = tuple(group["positions"].shape)
        return int(shape[0]) if len(shape) == 3 else 1
    return 1


def _choose_frame(num_frames: int, strategy: str) -> int:
    if num_frames <= 1:
        return 0
    if strategy == "first":
        return 0
    if strategy == "last":
        return num_frames - 1
    if strategy == "random":
        return random.randrange(num_frames)
    raise ValueError(f"Unknown frame_strategy {strategy!r}")


def _read_stage(group, frame_idx: int, nucleotide_slice: Optional[slice]) -> Dict[str, torch.Tensor]:
    positions = torch.as_tensor(_array_to_numpy(group["positions"], frame_idx, nucleotide_slice), dtype=torch.float32)

    if _has_key(group, "a1") and _has_key(group, "a3"):
        a1 = torch.as_tensor(_array_to_numpy(group["a1"], frame_idx, nucleotide_slice), dtype=torch.float32)
        a3 = torch.as_tensor(_array_to_numpy(group["a3"], frame_idx, nucleotide_slice), dtype=torch.float32)
        a1, a3 = project_a1_a3(a1, a3)
        orientations = frame_from_a1_a3(a1, a3)
    else:
        orientations = torch.as_tensor(_array_to_numpy(group["orientations"], frame_idx, nucleotide_slice), dtype=torch.float32)
        a1, a3 = a1_a3_from_frame(orientations)
        orientations = frame_from_a1_a3(a1, a3)

    out = {"positions": positions, "a1": a1, "a3": a3, "orientations": orientations}
    if _has_key(group, "bases"):
        out["bases"] = torch.as_tensor(_array_to_numpy(group["bases"], frame_idx, nucleotide_slice), dtype=torch.long)
    return out


def _random_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device, dtype=dtype))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _apply_global_rotation(stage: Dict[str, torch.Tensor], rot: torch.Tensor) -> None:
    stage["positions"] = stage["positions"] @ rot.T
    stage["a1"] = stage["a1"] @ rot.T
    stage["a3"] = stage["a3"] @ rot.T
    stage["orientations"] = torch.einsum("ij,njk->nik", rot, stage["orientations"])


def _topology_bases(topology_path: Path) -> torch.Tensor:
    lines = [line.strip() for line in topology_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty topology file: {topology_path}")
    bases: List[int] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        bases.append(BASE_TO_ID.get(fields[1].upper(), 0))
    return torch.tensor(bases, dtype=torch.long)


def read_oxdna_conf(conf_path: str, topology_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Read an oxDNA text configuration file into positions, a1, and a3 tensors."""
    path = Path(conf_path)
    positions: List[List[float]] = []
    a1_values: List[List[float]] = []
    a3_values: List[List[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("t =") or line.startswith("b =") or line.startswith("E ="):
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        vals = [float(x) for x in fields[:9]]
        positions.append(vals[0:3])
        a1_values.append(vals[3:6])
        a3_values.append(vals[6:9])
    if not positions:
        raise ValueError(f"No nucleotide coordinate rows found in {conf_path}")

    pos = torch.tensor(positions, dtype=torch.float32)
    a1 = torch.tensor(a1_values, dtype=torch.float32)
    a3 = torch.tensor(a3_values, dtype=torch.float32)
    a1, a3 = project_a1_a3(a1, a3)
    bases = torch.zeros(pos.shape[0], dtype=torch.long)
    if topology_path is not None:
        topo_bases = _topology_bases(Path(topology_path))
        bases[: min(len(bases), len(topo_bases))] = topo_bases[: min(len(bases), len(topo_bases))]
    return {
        "positions": pos,
        "a1": a1,
        "a3": a3,
        "orientations": frame_from_a1_a3(a1, a3),
        "bases": bases,
    }


class OxDNAZarrDataset(Dataset):
    """Load paired unrelaxed/relaxed oxDNA structures from zarr stores.

    The expected layout is either `initial/{positions,orientations,bases}` and
    `target/{positions,orientations,bases}` or `stage_0`/`stage_2`. Arrays can be
    a single conformation `(N, ...)` or a trajectory `(F, N, ...)`.
    """

    def __init__(
        self,
        data_dir: str,
        initial_group: str = "initial",
        target_group: str = "target",
        max_structures: Optional[int] = None,
        max_nucleotides: Optional[int] = 256,
        min_nucleotides: int = 1,
        frame_strategy: str = "random",
        center: bool = True,
        random_rotate: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.initial_group = initial_group
        self.target_group = target_group
        self.max_nucleotides = max_nucleotides
        self.min_nucleotides = min_nucleotides
        self.frame_strategy = frame_strategy
        self.center = center
        self.random_rotate = random_rotate
        self.zarr = _import_zarr()

        paths = sorted(self.data_dir.glob("*.zarr"))
        if max_structures is not None:
            paths = paths[:max_structures]
        self.structure_paths = [path for path in paths if self._passes_size_filter(path)]
        if not self.structure_paths:
            raise ValueError(f"No usable .zarr stores found in {self.data_dir}")

    def _passes_size_filter(self, path: Path) -> bool:
        root = self.zarr.open(str(path), mode="r")
        group = _resolve_group(root, self.initial_group)
        n = int(group["positions"].shape[-2])
        if n < self.min_nucleotides:
            return False
        if self.max_nucleotides is not None and n > self.max_nucleotides:
            return False
        return True

    def __len__(self) -> int:
        return len(self.structure_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.structure_paths[idx]
        root = self.zarr.open(str(path), mode="r")
        initial_group = _resolve_group(root, self.initial_group)
        target_group = _resolve_group(root, self.target_group)

        initial_frame = _choose_frame(_num_frames(initial_group), self.frame_strategy)
        target_frame = _choose_frame(_num_frames(target_group), self.frame_strategy)
        n = int(initial_group["positions"].shape[-2])
        nucleotide_slice = slice(0, n)

        initial = _read_stage(initial_group, initial_frame, nucleotide_slice)
        target = _read_stage(target_group, target_frame, nucleotide_slice)

        if "bases" in target:
            bases = target["bases"]
        elif "bases" in initial:
            bases = initial["bases"]
        elif _has_key(root, "bases"):
            bases = torch.as_tensor(_array_to_numpy(root["bases"], nucleotide_slice=nucleotide_slice), dtype=torch.long)
        else:
            bases = torch.zeros(n, dtype=torch.long)

        if self.center:
            initial["positions"], initial_centroid = center_positions(initial["positions"])
            target["positions"] = target["positions"] - initial_centroid.squeeze(0)

        if self.random_rotate:
            rot = _random_rotation(initial["positions"].device, initial["positions"].dtype)
            _apply_global_rotation(initial, rot)
            _apply_global_rotation(target, rot)

        sample = {
            "initial_positions": initial["positions"],
            "initial_a1": initial["a1"],
            "initial_a3": initial["a3"],
            "initial_orientations": initial["orientations"],
            "target_positions": target["positions"],
            "target_a1": target["a1"],
            "target_a3": target["a3"],
            "target_orientations": target["orientations"],
            "bases": bases.long(),
            "mask": torch.ones(n, dtype=torch.bool),
            "n_nucleotides": torch.tensor(n, dtype=torch.long),
            "structure_name": path.stem,
        }
        return sample


class OxDNAConfPairDataset(Dataset):
    """Load paired text `.conf` files, optionally with topology files."""

    def __init__(self, pairs: Sequence[Dict[str, str]], center: bool = True) -> None:
        self.pairs = list(pairs)
        self.center = center

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        initial = read_oxdna_conf(pair["initial"], pair.get("topology"))
        target = read_oxdna_conf(pair["target"], pair.get("topology"))
        n = min(initial["positions"].shape[0], target["positions"].shape[0])
        for stage in (initial, target):
            for key in list(stage.keys()):
                stage[key] = stage[key][:n]
        if self.center:
            initial["positions"], initial_centroid = center_positions(initial["positions"])
            target["positions"] = target["positions"] - initial_centroid.squeeze(0)
        return {
            "initial_positions": initial["positions"],
            "initial_a1": initial["a1"],
            "initial_a3": initial["a3"],
            "initial_orientations": initial["orientations"],
            "target_positions": target["positions"],
            "target_a1": target["a1"],
            "target_a3": target["a3"],
            "target_orientations": target["orientations"],
            "bases": initial["bases"].long(),
            "mask": torch.ones(n, dtype=torch.bool),
            "n_nucleotides": torch.tensor(n, dtype=torch.long),
            "structure_name": pair.get("name", Path(pair["target"]).stem),
        }


def oxdna_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length oxDNA samples into a batch."""
    max_n = max(int(item["n_nucleotides"]) for item in batch)
    batch_size = len(batch)
    out: Dict[str, torch.Tensor] = {
        "initial_positions": torch.zeros(batch_size, max_n, 3),
        "initial_a1": torch.zeros(batch_size, max_n, 3),
        "initial_a3": torch.zeros(batch_size, max_n, 3),
        "initial_orientations": torch.zeros(batch_size, max_n, 3, 3),
        "target_positions": torch.zeros(batch_size, max_n, 3),
        "target_a1": torch.zeros(batch_size, max_n, 3),
        "target_a3": torch.zeros(batch_size, max_n, 3),
        "target_orientations": torch.zeros(batch_size, max_n, 3, 3),
        "bases": torch.zeros(batch_size, max_n, dtype=torch.long),
        "mask": torch.zeros(batch_size, max_n, dtype=torch.bool),
        "n_nucleotides": torch.zeros(batch_size, dtype=torch.long),
    }
    names: List[str] = []
    for i, item in enumerate(batch):
        n = int(item["n_nucleotides"])
        for key, value in out.items():
            if key == "n_nucleotides":
                value[i] = n
            elif key in item:
                value[i, :n] = item[key]
        names.append(str(item["structure_name"]))
    out["structure_names"] = names  # type: ignore[assignment]
    return out


def create_oxdna_dataloader(
    config: OxdnaDataConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = OxDNAZarrDataset(**config.__dict__)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=oxdna_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def split_dataset(dataset: Dataset, val_fraction: float = 0.1, seed: int = 7):
    """Return train/validation subsets with a deterministic split."""
    if len(dataset) < 2 or val_fraction <= 0:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
