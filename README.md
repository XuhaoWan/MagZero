# MagZero

MagZero is a physics-informed graph-learning model for predicting pairwise magnetic-spin orientations and reconstructing magnetic ground states from crystal structures and DMFT-derived magnetic descriptors.

This repository is a cleaned, publishable scaffold built from the original research training code. It keeps the current model logic while adding a more standard Python package layout, CLI entry points, configuration files, and a minimal test skeleton.

## Repository layout

```text
magzero/
├── configs/                # Default experiment configuration
├── data/                   # Data documentation only; store real data locally
├── examples/               # Usage notes / lightweight examples
├── scripts/                # Thin wrappers around package CLIs
├── src/
│   └── magzero/
│       ├── cli/            # Training and inference command-line entry points
│       ├── data_utils.py   # YAML loading, dataset loading, scaling helpers
│       ├── forest.py       # Random-forest helper for physical tabular priors
│       ├── graph_ops.py    # Graph conversion and pair-generation utilities
│       └── model.py        # PyTorch Lightning implementation of MagZero
├── tests/                  # Minimal sanity tests / placeholders
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Status

This version focuses on **repository organization and packaging**. The core model logic is preserved as closely as possible to the current training code. It is not presented as a fully production-hardened reimplementation.

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
pip install -e .
```

### External dependency: `nearedge`

The model imports `nearedge` for radial / spherical basis layers and triplet construction. This dependency is **not bundled here**. Install it separately in the same environment before training or inference.

## Data expectations

Training and inference expect a folder of PyTorch Geometric `HeteroData` objects saved as `.pt` files. Each sample should contain at least:

- `data["atom"]`
- `data["magion"]`
- `data[("magion", "magion_edge", "magion")]`
- the geometric `near` edge types used by the current model

Samples with one or fewer magnetic ions are filtered out automatically.

## Configuration

The default config is stored in `configs/default.yaml` using standard flat YAML.

Example:

```yaml
atom_feat_dim: 164
magion_edge_feat_dim: 7
hidden_dim: 128
batch_size: 2
epochs: 100
```

The loader also supports the original legacy format where each key is nested under `value:`.

## Random-forest side input

`ForestWrapper` fits a `RandomForestRegressor` from a CSV file at startup. By default it expects a file such as `graph_data.csv`, where the first `magion_edge_feat_dim` columns are input features and the next column is the regression target.

Provide it with:

- `--rf-data path/to/graph_data.csv`

## Training

Package CLI:

```bash
magzero-train       --config configs/default.yaml       --data-folder /path/to/heterodata       --rf-data /path/to/graph_data.csv
```

Or the wrapper script:

```bash
python scripts/train.py       --config configs/default.yaml       --data-folder /path/to/heterodata       --rf-data /path/to/graph_data.csv
```

Optional arguments:

- `--gpu 0`
- `--seed 8`
- `--fold 0`
- `--project MagZero`
- `--run-name my-run`
- `--no-wandb`

## Inference

```bash
magzero-infer       --checkpoint checkpoints/best.ckpt       --config configs/default.yaml       --data-folder /path/to/heterodata       --rf-data /path/to/graph_data.csv       --output-csv outputs/inference_results.csv
```

Current inference writes **batch-level summary statistics** because the preserved model returns pair-level predictions for batched graphs without explicit per-crystal demultiplexing.

## Testing

Install dev tools:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

## Recommended next cleanup steps

If you intend to open-source or publish this code, the next high-value changes would be:

1. add a `LightningDataModule`;
2. separate experiment logging from core training;
3. add deterministic dataset splits saved to disk;
4. make per-sample inference explicit instead of batch summaries;
5. add unit tests for graph conversion and edge-centric graph construction.
