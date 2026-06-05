# oxBB

BBFlow-style conditional flow matching for oxDNA conformation generation.

The first target is short duplexes and hairpins from oxDNA simulation data. The
longer-term target is conditional generation from an unrelaxed oxDNA structure
to a plausible relaxed conformation or ensemble.

## What Is Implemented

- `src/oxbb/oxdna_dataset.py`: oxDNA zarr and `.conf` loaders with paired
  initial/target structures, trajectory frame sampling, padding, masks, and
  explicit `a1`/`a3` orientation fields.
- `src/oxbb/oxdna_flow_model.py`: conditional endpoint-prediction flow matching
  model inspired by BBFlow. It uses a conditional prior, time conditioning, a
  Transformer trunk, clean endpoint prediction, and Euler sampling.
- `src/oxbb/oxdna_geometry.py`: projection/normalization utilities that keep
  generated oxDNA `a1` and `a3` vectors unit length and mutually orthogonal.
- `src/oxbb/oxdna_eval.py`: generated sample checks for shapes, RMSD/distance to
  ensemble centroid, backbone bond lengths, orientation norms/orthogonality, and
  optional energy integration.
- `configs/oxdna.yaml`: default training config aimed at short systems.

## Slides

Open `docs/bbflow-update-slides.html` in a browser for a slide report on how the
existing flow baseline was updated into the current BBFlow-style oxDNA model.

## Setup

Use a Python environment with PyTorch and zarr:

```bash
pip install -e ".[dev]"
```

The current cluster login environment may set a broken `PYTHONPATH`; unset it if
Python fails before importing project code:

```bash
unset PYTHONPATH
```

## Train

```bash
python scripts/train_oxdna_flow.py \
  --config configs/oxdna.yaml \
  --data_dir /scratch/sroy85/fastDNA/oxdna_zarr_dataset_full \
  --max_nucleotides 256
```

For a smoke run:

```bash
python scripts/train_oxdna_flow.py --limit_batches 2 --epochs 1
```

The corrected fastDNA dataset is
`/scratch/sroy85/fastDNA/oxdna_zarr_dataset_full`, generated from
`/scratch/sroy85/oxDNN2/output`. The older
`/scratch/sroy85/fastDNA/oxdna_zarr_dataset` directory exists as a fallback but
is not the preferred training source. Many examples are full-origami-scale, so
for the first duplex/hairpin target keep `max_nucleotides` low or point
`data_dir` to a filtered short-system zarr set.

## Sample

```bash
python scripts/sample_oxdna_flow.py \
  --checkpoint checkpoints/oxdna/latest.pt \
  --data_dir /scratch/sroy85/fastDNA/oxdna_zarr_dataset_full \
  --output outputs/oxdna_samples.npz
```

## Evaluate

```bash
python scripts/evaluate_oxdna_samples.py \
  --samples outputs/oxdna_samples.npz \
  --output outputs/oxdna_eval.json
```

Use `--run_energy` only when an oxDNA energy implementation is importable in the
active environment.
