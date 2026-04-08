# MagZero

MagZero is a physics-informed graph-learning model for predicting pairwise magnetic-spin orientations and reconstructing magnetic ground states from crystal structures and DMFT-derived magnetic descriptors.
See Paper: X. Wan, Y. Guo, K. Haule. Graph Learning of Magnetic Ground States in Crystalline Solids from DMFT-Derived Weiss Fields (2026)


## Repository layout

```text
magzero/
├── configs/                # Default experiment configuration
├── data/                   # Dataset and data documentation
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

This version focuses on academic open source purpose.

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

## Data expectations

Training and inference expect a folder of PyTorch Geometric `HeteroData` objects saved as `.pt` files. Each sample should contain at least:

- `data["atom"]`
- `data["magion"]`
- `data[("magion", "magion_edge", "magion")]`
- the geometric `near` edge types used by the current model

Samples with one or fewer magnetic ions are filtered out automatically.

## Data preparation

The data used in this project are generated from embedded Dynamical Mean Field Theory (eDMFT) calculations.
All eDMFT calculations were performed using the open-source package developed by the Rutgers University CCMT group:

eDMFT codebase: https://github.com/ru-ccmt/eDMFT
Official tutorials: http://hauleweb.rutgers.edu/tutorials/

Users are strongly encouraged to consult the official repository and tutorials for installation, environment setup, input preparation, and execution details of the eDMFT workflow.

In this project, the Weiss-field-derived interaction parameters used as model inputs are obtained from eDMFT calculations on magnetic crystalline materials. The scripts/ directory provides utilities for main use cases:

High-throughput data generation
Scripts for automatically processing a large set of crystal structures and extracting Weiss-field parameters in a batch workflow.
Single-material Weiss-field extraction
Scripts for running the workflow on an individual crystal structure and obtaining the corresponding Weiss-field parameters for detailed inspection or case studies.

These scripts are intended to facilitate both dataset construction at scale and targeted calculations for specific compounds. Before running them, users should ensure that the eDMFT package is properly installed and configured, and that the required crystal structure and calculation input files are prepared according to the eDMFT documentation.

## Configuration

The default config is stored in `configs/default.yaml` using standard flat YAML.

Example:

```yaml
atom_feat_dim: 164
magion_edge_feat_dim: 7
hidden_dim: 128
batch_size: 4
epochs: 200
```

The loader also supports the original legacy format where each key is nested under `value:`.


## Training

Package CLI:

```bash
magzero-train       --config configs/default.yaml       --data-folder /path/to/heterodata      
```

Or the wrapper script:

```bash
python scripts/train.py       --config configs/default.yaml       --data-folder /path/to/heterodata      
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
magzero-infer       --checkpoint checkpoints/best.ckpt       --config configs/default.yaml       --data-folder /path/to/heterodata      --output-csv outputs/inference_results.csv
```

## Testing

Install dev tools:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```
