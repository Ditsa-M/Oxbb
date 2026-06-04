# AGENTS.md

This repository builds oxBB: a BBFlow-style conditional flow-matching project
for oxDNA trajectory and conformation generation.

The group goal is to learn plausible relaxed oxDNA conformations and relaxed
conformational ensembles from oxDNA simulation data. The first modeling target
is short duplexes and hairpins. Do not optimize this repository around
full-origami systems first. Full origami support is a later scaling target after
the data path, model interface, and oxDNA-specific validation metrics are solid
on small systems.

Longer term, the project should support conditional generation from an
unrelaxed oxDNA structure to a relaxed conformation or ensemble. Treat
unrelaxed-to-relaxed conditioning as the primary product direction, not an
incidental demo.

## Current Repo Layout

- `src/oxbb/oxdna_dataset.py`: oxDNA data loading. Keep zarr, trajectory, `.conf`,
  topology, padding, masking, and frame-sampling logic here.
- `src/oxbb/oxdna_flow_model.py`: oxDNA conditional flow-matching model. Keep
  model architecture, conditional prior, corruption/noising, training loss, and
  sampler logic here.
- `src/oxbb/oxdna_geometry.py`: oxDNA geometry utilities. Keep orientation
  normalization, `a1`/`a3` projection, rotation-frame conversion, RMSD, centroid,
  and bond-length helpers here unless a helper clearly belongs to evaluation.
- `src/oxbb/oxdna_eval.py`: evaluation metrics for generated oxDNA samples.
- `scripts/train_oxdna_flow.py`: training entry point.
- `scripts/sample_oxdna_flow.py`: checkpoint sampling entry point.
- `scripts/evaluate_oxdna_samples.py`: generated sample evaluation entry point.
- `configs/oxdna.yaml`: default oxDNA experiment config. Add new experiment
  knobs here before hard-coding them in scripts.
- `tests/`: focused tests for geometry, data contracts, and evaluation.

Add new oxDNA-specific code in clearly named modules. Prefer names such as
`oxdna_dataset.py`, `oxdna_flow_model.py`, `oxdna_geometry.py`,
`oxdna_eval.py`, or `configs/oxdna_*.yaml`. Avoid vague names such as
`utils2.py`, `new_model.py`, or `train_final.py`.

## Data Assumptions

The main external dataset currently lives at:

```text
/scratch/sroy85/fastDNA
```

The observed paired zarr dataset layout includes stores such as:

```text
/scratch/sroy85/fastDNA/oxdna_zarr_dataset/<name>.zarr/
  initial/
    positions/
    orientations/
    bases/
  target/
    positions/
    orientations/
    bases/
```

Some older or merged data may use:

```text
stage_0/
stage_2/
stage_3/
bases/
strands/
bonding/
```

The loader must support both naming conventions when reasonable. The current
model should train on paired initial/target conformations and should support
trajectory arrays shaped like `(frames, nucleotides, 3)` for positions and
`(frames, nucleotides, 3, 3)` for orientations.

The first target is short duplexes and hairpins. Use `max_nucleotides` and
dataset filtering to avoid accidentally turning every smoke test into a
full-origami memory stress test. Do not remove the ability to load longer
systems, but keep the default configs conservative.

## oxDNA Orientation Contract

oxDNA nucleotide orientations are represented by body axes `a1` and `a3`.
Generated orientations must remain valid enough for oxDNA analysis:

- `a1` must be unit norm.
- `a3` must be unit norm.
- `a1` and `a3` must be mutually orthogonal.
- The derived frame should be right-handed.
- Projection should happen after raw model orientation outputs and after any
  sampling update that linearly combines orientation vectors.

Use `project_a1_a3` and `frame_from_a1_a3` from `oxdna_geometry.py`. Do not
duplicate ad hoc orientation normalization in training scripts. If a new model
head predicts rotations, quaternions, or frames, convert back to explicit
`a1`/`a3` and run the same validity checks before returning samples.

## Model Direction

The project borrows the following BBFlow ideas:

- Conditional prior rather than unconditional white noise only.
- Continuous flow time `t`.
- Corruption/interpolation from prior state to clean relaxed endpoint.
- Endpoint prediction of the clean target conformation.
- Euler-style sampling that repeatedly predicts the clean endpoint and advances
  the current state.

The current implementation intentionally does not vendor GAFL, OpenFold, or the
protein-specific BBFlow frame stack. Keep the oxDNA baseline trainable in a
plain PyTorch/zarr environment. If an equivariant or SE(3)-aware dependency is
introduced later, isolate it behind a model class and keep the existing data and
evaluation contracts stable.

The conditioning structure means the model input must preserve:

- Current noised/interpolated positions.
- Current noised/interpolated `a1` and `a3`.
- Initial/unrelaxed conditioning positions.
- Initial/unrelaxed conditioning `a1` and `a3`.
- Base identities.
- Nucleotide mask.
- Flow time `t`.

The target must preserve:

- Relaxed/production positions.
- Relaxed/production `a1` and `a3`.
- Relaxed/production orientation frame when needed by analysis.

Do not train only on positions unless the task explicitly says to disable
orientation learning. The model and evaluation path must include orientations.

## Evaluation Requirements

Generated sample evaluation should include these checks:

- Generated sample shape checks.
- Finite-value checks.
- RMSD or distance to ensemble centroid.
- Backbone bond length sanity checks.
- Orientation vector norm checks for `a1` and `a3`.
- Orientation orthogonality checks between `a1` and `a3`.
- Optional oxDNA energy evaluation if a compatible energy function is available.

If an energy function is not available, the evaluation script should report that
clearly and continue. Do not make the whole evaluation path fail only because an
optional energy backend is missing.

When adding new metrics, return machine-readable JSON-compatible values. Keep
human-readable printing as a thin CLI concern.

## Training Rules

Training code should:

- Load oxDNA trajectory/conformation data through `OxDNAZarrDataset` or a
  clearly named successor.
- Use configs for defaults.
- Support CPU for smoke tests and CUDA for real training.
- Keep checkpoints under `checkpoints/` by default.
- Save the model state, optimizer state, config, epoch, and history.
- Support small smoke runs with `--limit_batches`.
- Respect masks for variable-length batches.
- Avoid assuming every adjacent nucleotide pair is a real backbone bond when
  explicit topology bonds are available. The current fallback uses adjacency
  because the target short-system path may not always have topology; improve this
  when topology is present.

Do not commit generated checkpoints, `.npz` samples, W&B runs, or large dataset
artifacts. `.gitignore` should keep these out of git.

## Coding Rules

Prefer small, readable modules with explicit data contracts. Make tensor shapes
obvious in names, comments, or docstrings when ambiguity would slow down future
work.

Use PyTorch tensor operations for differentiable model code. Avoid converting to
NumPy inside training/model functions unless the code is explicitly outside the
gradient path.

Keep scripts thin. Put reusable logic in `src/oxbb`.

Use structured parsers/APIs when available. Use zarr to read zarr stores; do not
parse chunk files manually in production code.

Do not hard-code `/scratch/sroy85/fastDNA` inside reusable modules. It can be a
config default or README example, but library code should accept paths from the
caller.

Do not silently drop orientations. If a dataset has only rotation matrices,
derive `a1`/`a3`. If a dataset has explicit `a1`/`a3`, project them and derive
the frame. If neither exists, raise a clear error.

Keep defaults aimed at short systems. If changing defaults for large systems,
explain why in the config or README.

Write tests for geometry projection and metrics whenever changing orientation
logic. Orientation bugs can look numerically small but break oxDNA analysis.

Use ASCII in source files unless a file already requires Unicode.

## Git Rules

The user asked to keep pushing everything to the GitHub repo. Commit cohesive
changes and push to `origin main` when a working unit is complete.

Before committing, inspect `git status --short`. Do not revert unrelated user
changes. If the worktree is dirty for reasons unrelated to the task, leave those
changes alone and commit only the files relevant to the task.

Do not commit dataset files, generated checkpoints, generated sample archives,
or local run outputs.

## Useful Commands

Install:

```bash
pip install -e ".[dev]"
```

Train smoke run:

```bash
python scripts/train_oxdna_flow.py --config configs/oxdna.yaml --epochs 1 --limit_batches 2
```

Train on short systems:

```bash
python scripts/train_oxdna_flow.py --config configs/oxdna.yaml --max_nucleotides 256
```

Sample:

```bash
python scripts/sample_oxdna_flow.py --checkpoint checkpoints/oxdna/latest.pt --output outputs/oxdna_samples.npz
```

Evaluate:

```bash
python scripts/evaluate_oxdna_samples.py --samples outputs/oxdna_samples.npz --output outputs/oxdna_eval.json
```

Run tests:

```bash
pytest
```

If the cluster environment sets a bad Python path and Python fails before
project imports, run:

```bash
unset PYTHONPATH
```
