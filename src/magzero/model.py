from __future__ import annotations

from typing import Callable, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.resolver import activation_resolver

from nearedge import (
    BesselBasisLayer,
    EmbeddingLayer,
    InteractionBlock,
    OutputBlock,
    SphericalBasisLayer,
    triplets,
)

from .data_utils import scale_to_negative_one
from .forest import ForestWrapper
from .graph_ops import build_edge_centric_graph, convert_hetero_to_global, generate_node_pairs


torch.set_printoptions(threshold=float("inf"))


class Magzero(pl.LightningModule):
    """PyTorch Lightning implementation of MagZero.

    The model couples a crystal-geometry branch with a magnetic-edge branch and
    predicts pairwise cosine similarities between magnetic moments.
    """

    def __init__(
        self,
        atom_feat_dim: int = 164,
        magion_edge_feat_dim: int = 7,
        hidden_dim: int = 128,
        cutoff: float = 4.0,
        num_blocks: int = 4,
        gat_heads: int = 4,
        num_bilinear: int = 64,
        num_outlayer: int = 4,
        envelope_exponent: int = 5,
        num_spherical: int = 7,
        num_radial: int = 8,
        out_dim: int = 128,
        num_before_skip: int = 2,
        num_after_skip: int = 2,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "glorot_orthogonal",
        lr: float = 1e-4,
        rf_csv_path: str = "graph_data.csv",
        log_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.lr = lr
        self.log_memory = log_memory

        act = activation_resolver(act)
        if num_spherical < 2:
            raise ValueError("'num_spherical' must > 1")

        # Shared atom / magnetic-ion encoder.
        self.atom_encoder = EmbeddingLayer(num_radial, hidden_dim, act, atom_feat_dim)

        # Encoder for magnetic-edge features.
        self.magion_encoder = nn.Sequential(
            nn.Linear(magion_edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DimeNet++ style basis layers.
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff,
            envelope_exponent,
        )

        self.output_blocks = nn.ModuleList(
            [
                OutputBlock(
                    num_radial,
                    hidden_dim,
                    out_dim,
                    num_outlayer,
                    act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        # Fusion block: structural pair features.
        self.fusion_nainfo_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )

        # Reserved fusion block for future magnetic-neighbor feature fusion.
        self.fusion_mn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )

        self.forest = ForestWrapper(
            input_dim=magion_edge_feat_dim,
            output_dim=1,
            csv_path=rf_csv_path,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.lin_out = nn.Linear(2, 1)

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.rbf.reset_parameters()
        self.sbf.reset_parameters()

        for block in self.interaction_blocks:
            block.reset_parameters()
        for block in self.output_blocks:
            block.reset_parameters()
        for layer in self.magion_encoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for fusion in list(self.fusion_nainfo_blocks) + list(self.fusion_mn_blocks):
            for layer in fusion:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def forward(self, data):
        """Run a forward pass and return predictions together with regression targets."""
        data_dict = convert_hetero_to_global(data)
        gdata = data_dict["global"]
        magiondata = data_dict["magion"]
        edge_magiondata = build_edge_centric_graph(magiondata)

        gdata.x = gdata.x.float()
        num_natoms = gdata.atom_mask.sum().item()

        magion_edge_x = scale_to_negative_one(edge_magiondata.x, edge_magiondata.batch)
        magion_edge_physinfo = self.forest(magion_edge_x).unsqueeze(1)
        targets = edge_magiondata.y

        edge_index = gdata.edge_index.to(device=magion_edge_x.device)
        edge_attr = gdata.edge_attr.to(device=magion_edge_x.device)

        i_tri, j_tri, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index,
            num_nodes=gdata.x.size(0),
        )

        dist = torch.norm(gdata.edge_attr, p=2, dim=-1).to(device=magion_edge_x.device)
        pos_ji = -edge_attr[idx_ji]
        pos_kj = -edge_attr[idx_kj]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a).to(device=magion_edge_x.device)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        del pos_ji, pos_kj, a, b, angle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Geometry branch.
        x = self.atom_encoder(gdata.x, rbf, i_tri, j_tri)
        p = self.output_blocks[0](x, rbf, i_tri, num_nodes=gdata.x.size(0))

        # Magnetic-edge branch.
        magion_edge_x = self.magion_encoder(magion_edge_x)

        for block_idx in range(self.num_blocks):
            if self.log_memory and torch.cuda.is_available():
                print(f"Block {block_idx} Mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            x = self.interaction_blocks[block_idx](x, rbf, sbf, idx_kj, idx_ji)
            p = self.output_blocks[block_idx + 1](x, rbf, i_tri, num_nodes=gdata.x.size(0))

            natom_info = generate_node_pairs(p[num_natoms:], magiondata.batch)
            natom_info_u = self.fusion_nainfo_blocks[block_idx](natom_info)

        output = self.lin_out(torch.cat([self.decoder(natom_info_u), magion_edge_physinfo], dim=1))
        constrained = self._constrain_output(output)
        return constrained, targets[:, 0:1]

    def predict_cosine(self, data) -> torch.Tensor:
        """Convenience wrapper that returns only the cosine prediction."""
        pred, _ = self(data)
        return pred

    def _constrain_output(self, output: torch.Tensor) -> torch.Tensor:
        """Optional output constraint hook.

        The current training setup uses a direct scalar regression target, so the raw
        output is returned unchanged.
        """
        return output

    def training_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = F.mse_loss(pred, target)
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = F.mse_loss(pred, target)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, target = self(batch)
        loss = F.mse_loss(pred, target)
        self.log("test_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
