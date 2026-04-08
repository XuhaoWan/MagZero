from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from time import time
from typing import Any, List

import torch
import yaml

try:
    from torch_geometric.data import HeteroData
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    HeteroData = Any


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def timer(func):
    """Print the runtime of a function."""

    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        duration = time() - start
        print(f"run {func.__name__} in {duration:.1f} seconds")
        return res

    return wrapper


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def yaml2dict(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config into a flat dictionary.

    The loader supports both of the following formats:

    1. Standard flat YAML:

       batch_size: 4
       epochs: 200

    2. Legacy nested YAML used by the original training script:

       batch_size:
         value: 4
       epochs:
         value: 200
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    result = {}
    for key, value in raw.items():
        result[key] = value["value"] if isinstance(value, dict) and "value" in value else value
    return result



def dict2namedtuple(config_dict: dict[str, Any]):
    """Convert a dictionary to a lightweight config object."""
    return namedtuple("Config", config_dict.keys())(**config_dict)



def load_yaml(path: str | os.PathLike[str]):
    """Load a YAML config and expose it as a namedtuple."""
    config = dict2namedtuple(yaml2dict(path))
    print(config)
    return config


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def load_and_filter_heterodata(folder_path: str | os.PathLike[str]) -> List[HeteroData]:
    """Load valid ``.pt`` files from a directory.

    A sample is kept only when:
    1. the file contains a ``HeteroData`` object;
    2. the ``magion`` node store is present; and
    3. the number of magnetic-ion nodes is larger than one.
    """
    dataset: List[HeteroData] = []
    folder = Path(folder_path)

    for file_path in sorted(folder.glob("*.pt")):
        data = torch.load(file_path, map_location="cpu")
        if not isinstance(data, HeteroData):
            continue

        if "magion" not in data.node_types:
            continue

        magion_x = data["magion"].x
        if magion_x.dim() == 1 or magion_x.size(0) <= 1:
            continue

        dataset.append(data)

    return dataset



def load_and_check_heterodata(folder_path: str | os.PathLike[str]) -> None:
    """Inspect the dataset and print suspicious samples."""
    folder = Path(folder_path)

    for file_path in sorted(folder.glob("*.pt")):
        data = torch.load(file_path, map_location="cpu")
        if not isinstance(data, HeteroData):
            continue

        magion_x = data["magion"].x
        if magion_x.dim() == 1 or magion_x.size(0) <= 1:
            print(file_path.name)
            continue

        magion_edge_attr = data["magion", "magion_edge", "magion"].edge_attr
        if magion_edge_attr.size(1) <= 1:
            print(file_path.name, magion_edge_attr.size())



def load_heterodata_and_comparetime(folder_path: str | os.PathLike[str]) -> None:
    """Estimate the cost difference between exhaustive search and MagZero-style inference."""
    convtime = 0.0
    magzerotime = 0.0
    folder = Path(folder_path)

    for file_path in sorted(folder.glob("*.pt")):
        data = torch.load(file_path, map_location="cpu")
        if not isinstance(data, HeteroData):
            continue

        atom_x = data["atom"].x.size(0)
        magion_x = data["magion"].x.size(0)
        convtime += (magion_x * 30 * 0.02 + atom_x * 30 * 0.005) * (5 ** magion_x)
        magzerotime += magion_x * 30 * 0.02 + atom_x * 30 * 0.005 + 30 * 0.01

    print(convtime, magzerotime)


# -----------------------------------------------------------------------------
# Feature scaling helpers
# -----------------------------------------------------------------------------

def _make_batch_indices(batch: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary batch IDs to a contiguous range starting from zero."""
    _, batch_indices = torch.unique(batch, return_inverse=True)
    return batch_indices



def scale_to_negative_one(tensor: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Min-max scale each batch independently to ``[-1, 1]``."""
    batch_indices = _make_batch_indices(batch)

    min_vals = torch.zeros(
        len(torch.unique(batch_indices)),
        tensor.shape[1],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    max_vals = torch.zeros_like(min_vals)

    for i in torch.unique(batch_indices):
        mask = batch_indices == i
        batch_data = tensor[mask]
        min_vals[i] = torch.min(batch_data, dim=0).values
        max_vals[i] = torch.max(batch_data, dim=0).values

    min_expanded = min_vals[batch_indices]
    max_expanded = max_vals[batch_indices]
    epsilon = 1e-8
    return 2 * (tensor - min_expanded) / (max_expanded - min_expanded + epsilon) - 1



def zscore_standardize(tensor: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Apply per-batch z-score standardization."""
    batch_indices = _make_batch_indices(batch)

    means = torch.zeros(
        len(torch.unique(batch_indices)),
        tensor.shape[1],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    stds = torch.zeros_like(means)

    for i in torch.unique(batch_indices):
        mask = batch_indices == i
        batch_data = tensor[mask]
        means[i] = torch.mean(batch_data, dim=0)
        centered = batch_data - means[i]
        stds[i] = torch.sqrt(torch.mean(centered**2, dim=0) + 1e-8)

    mean_expanded = means[batch_indices]
    std_expanded = stds[batch_indices]
    return (tensor - mean_expanded) / std_expanded



def min_max_normalize(tensor: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each batch independently to ``[0, 1]``."""
    batch_indices = _make_batch_indices(batch)

    min_vals = torch.zeros(
        len(torch.unique(batch_indices)),
        tensor.shape[1],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    max_vals = torch.zeros_like(min_vals)

    for i in torch.unique(batch_indices):
        mask = batch_indices == i
        batch_data = tensor[mask]
        min_vals[i] = torch.min(batch_data, dim=0).values
        max_vals[i] = torch.max(batch_data, dim=0).values

    min_expanded = min_vals[batch_indices]
    max_expanded = max_vals[batch_indices]
    epsilon = 1e-8
    return (tensor - min_expanded) / (max_expanded - min_expanded + epsilon)
