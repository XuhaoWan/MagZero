from __future__ import annotations

from itertools import combinations
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData


# -----------------------------------------------------------------------------
# Pair utilities
# -----------------------------------------------------------------------------

def generate_node_pairs(features: Tensor, batch: Tensor) -> Tensor:
    """Generate concatenated node-pair features within each batch.

    Pairs follow the strict combination order produced by ``torch.combinations``.
    """
    unique_batches = torch.unique(batch, sorted=True)
    pair_list = []

    for batch_id in unique_batches:
        mask = batch == batch_id
        indices = torch.where(mask)[0]
        n_nodes = len(indices)

        if n_nodes < 2:
            continue

        pairs = torch.combinations(indices, r=2, with_replacement=False)
        i = pairs[:, 0]
        j = pairs[:, 1]
        pair_features = torch.cat([features[i], features[j]], dim=1)
        pair_list.append(pair_features)

    if pair_list:
        return torch.cat(pair_list, dim=0)
    return torch.empty((0, features.shape[1] * 2), device=features.device)



def cos_to_class(cos_theta: float, num_classes: int = 6) -> int:
    """Convert ``cos(theta)`` to a discrete angle-bin label."""
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    bin_edges = np.linspace(0, 180, num_classes + 1)
    class_label = np.digitize(theta_deg, bin_edges) - 1
    return int(np.clip(class_label, 0, num_classes - 1))


# -----------------------------------------------------------------------------
# Edge attribute selection helpers
# -----------------------------------------------------------------------------

def select_larger_sum_tensor(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Return the tensor with the larger element-wise sum."""
    return tensor1 if torch.sum(tensor1) > torch.sum(tensor2) else tensor2



def select_smaller_sum_tensor(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Return the tensor with the smaller element-wise sum."""
    return tensor1 if torch.sum(tensor1) < torch.sum(tensor2) else tensor2



def select_closer_to_zero(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Return the tensor whose element-wise sum is closer to zero."""
    sum1 = torch.sum(tensor1)
    sum2 = torch.sum(tensor2)
    return tensor1 if abs(sum1) < abs(sum2) else tensor2


# -----------------------------------------------------------------------------
# Graph conversion helpers
# -----------------------------------------------------------------------------

def get_edge_batch(atom_batch: Tensor, edge_index: Tensor) -> Tensor:
    """Convert atom-level batch IDs to edge-level batch IDs."""
    edge_batch = []
    for i in range(edge_index.size(1)):
        src, _ = edge_index[:, i]
        edge_batch.append(atom_batch[src])
    return torch.tensor(edge_batch, device=edge_index.device)



def build_edge_centric_graph(data: Data) -> Data:
    """Convert the magnetic-ion subgraph into an edge-centric graph.

    - Edge-node features are inherited from the original magnetic-ion edge attributes.
    - Two edge-nodes are connected when the corresponding original edges share an atom.
    - Targets are the relative spin orientations encoded as ``[cos(theta), sin(theta)]``.
    """
    handled = set()
    edge_dict = {}
    edge_targets = []

    for idx in range(data.edge_index.size(1)):
        i, j = data.edge_index[:, idx].tolist()
        if (j, i) in handled:
            continue

        mask_ij = (data.edge_index[0] == i) & (data.edge_index[1] == j)
        mask_ji = (data.edge_index[0] == j) & (data.edge_index[1] == i)
        attr_ij = data.edge_attr[mask_ij].squeeze(0)
        attr_ji = data.edge_attr[mask_ji].squeeze(0)

        mi, mj = data.magmom[i], data.magmom[j]
        mi_norm = mi / (torch.norm(mi) + 1e-6)
        mj_norm = mj / (torch.norm(mj) + 1e-6)
        cos_theta = mi_norm.dot(mj_norm)
        sin_theta = torch.sqrt(1 - cos_theta**2)
        edge_targets.append(torch.stack([cos_theta, sin_theta]))

        if mask_ji.any():
            if torch.allclose(attr_ij, attr_ji, atol=1e-4):
                merged_attr = attr_ij
            elif cos_theta > 0.2:
                merged_attr = select_larger_sum_tensor(attr_ij, attr_ji)
            elif cos_theta < -0.2:
                merged_attr = select_smaller_sum_tensor(attr_ij, attr_ji)
            else:
                merged_attr = select_closer_to_zero(attr_ij, attr_ji)
            handled.add((i, j))
            handled.add((j, i))
        else:
            merged_attr = attr_ij
            handled.add((i, j))

        edge_key = (i, j) if i <= j else (j, i)
        edge_dict[edge_key] = merged_attr

    sorted_edges = sorted(edge_dict.keys())
    edge_index = torch.tensor(sorted_edges, device=data.edge_index.device).t().contiguous()
    edge_attr = torch.stack([edge_dict[k] for k in sorted_edges])

    new_edge_index = []
    new_edge_attr = []
    for e1, e2 in combinations(range(edge_index.size(1)), 2):
        atoms_e1 = set(edge_index[:, e1].tolist())
        atoms_e2 = set(edge_index[:, e2].tolist())
        shared_atoms = list(atoms_e1 & atoms_e2)

        for atom in shared_atoms:
            new_edge_index.append([e1, e2])
            new_edge_index.append([e2, e1])
            new_edge_attr.append(data.x[atom].clone())
            new_edge_attr.append(data.x[atom].clone())

    if edge_index.size(1) > 0 and len(new_edge_index) == 0:
        for e in range(edge_index.size(1)):
            new_edge_index.append([e, e])
            if data.x.size(0) > 0:
                virtual_attr = data.x.mean(dim=0)
            else:
                virtual_attr = torch.zeros(data.x.size(1), device=data.x.device)
            new_edge_attr.append(virtual_attr)

    edge_node_features = edge_attr.contiguous()
    edge_graph_index = torch.tensor(new_edge_index, dtype=torch.long, device=edge_attr.device).t().contiguous()
    edge_labels = torch.stack(edge_targets)

    return Data(
        x=edge_node_features,
        edge_index=edge_graph_index,
        edge_attr=torch.stack(new_edge_attr).float() if new_edge_attr else torch.empty(0, dtype=torch.float32),
        y=edge_labels,
        original_edge_index=edge_index,
        batch=get_edge_batch(data.batch, edge_index),
    )



def convert_hetero_to_global(data: HeteroData) -> Dict[str, Data]:
    """Flatten a ``HeteroData`` object into the global graph and magnetic-ion subgraph."""
    has_batch = "batch" in data["atom"] and "batch" in data["magion"]

    if has_batch:
        atom_batch = data["atom"].batch
        magion_batch = data["magion"].batch
    else:
        atom_batch = torch.zeros(data["atom"].x.size(0), dtype=torch.long)
        magion_batch = torch.zeros(data["magion"].x.size(0), dtype=torch.long)

    global_batch = torch.cat([atom_batch, magion_batch])
    num_atoms = data["atom"].x.size(0)
    num_magions = data["magion"].x.size(0)

    global_edge_indices = []
    global_edge_attrs = []
    near_edge_types = [
        ("atom", "near", "atom"),
        ("atom", "near", "magion"),
        ("magion", "near", "atom"),
        ("magion", "near", "magion"),
    ]

    for edge_type in near_edge_types:
        if edge_type not in data.edge_types:
            continue

        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr

        src_type = edge_type[0]
        dst_type = edge_type[2]
        adj_src = edge_index[0] + num_atoms if src_type == "magion" else edge_index[0]
        adj_dst = edge_index[1] + num_atoms if dst_type == "magion" else edge_index[1]

        adjusted_edge = torch.stack([adj_src, adj_dst], dim=0)
        global_edge_indices.append(adjusted_edge)
        global_edge_attrs.append(edge_attr)

    if global_edge_indices:
        final_edge_index = torch.cat(global_edge_indices, dim=1)
        final_edge_attr = torch.cat(global_edge_attrs, dim=0)
    else:
        final_edge_index = torch.empty((2, 0), dtype=torch.long)
        final_edge_attr = torch.empty((0, 3), dtype=torch.float)

    global_data = Data(
        x=torch.cat([data["atom"].x, data["magion"].x], dim=0).contiguous(),
        edge_index=final_edge_index.contiguous(),
        edge_attr=final_edge_attr.contiguous(),
        batch=global_batch.contiguous(),
    )

    global_data.atom_mask = torch.cat(
        [
            torch.ones(num_atoms, dtype=torch.bool),
            torch.zeros(num_magions, dtype=torch.bool),
        ]
    ).contiguous()

    magion_edge_index = data["magion", "magion_edge", "magion"].edge_index
    magion_edge_attr = data["magion", "magion_edge", "magion"].edge_attr

    magion_data = Data(
        x=data["magion"].x.clone(),
        edge_index=magion_edge_index.clone(),
        edge_attr=magion_edge_attr.clone(),
        magmom=data["magion"].magmom.clone(),
        batch=magion_batch.clone(),
    )

    return {"global": global_data, "magion": magion_data}
