from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from magzero import Magzero
from magzero.data_utils import load_and_filter_heterodata, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained MagZero checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Lightning checkpoint (.ckpt).")
    parser.add_argument("--data-folder", required=True, help="Folder containing heterograph .pt files.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--rf-data", default="graph_data.csv", help="CSV file used by the RandomForest helper.")
    parser.add_argument("--gpu", default="0", help="CUDA device ID exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--batch-size", default=None, type=int, help="Optional batch-size override.")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of DataLoader workers.")
    parser.add_argument("--output-csv", default="inference_results.csv", help="Path to the output CSV.")
    return parser.parse_args()


def summarize_prediction(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu().view(-1)
    target = target.detach().cpu().view(-1)
    return {
        "num_pairs": int(pred.numel()),
        "pred_mean": float(pred.mean().item()) if pred.numel() > 0 else 0.0,
        "target_mean": float(target.mean().item()) if target.numel() > 0 else 0.0,
        "mse": float(torch.mean((pred - target) ** 2).item()) if pred.numel() > 0 else 0.0,
        "mae": float(torch.mean(torch.abs(pred - target)).item()) if pred.numel() > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pl.seed_everything(8, workers=True)
    config = load_yaml(args.config)

    model = Magzero.load_from_checkpoint(
        args.checkpoint,
        atom_feat_dim=config.atom_feat_dim,
        magion_edge_feat_dim=config.magion_edge_feat_dim,
        hidden_dim=config.hidden_dim,
        cutoff=config.cutoff,
        num_blocks=config.num_blocks,
        gat_heads=config.gat_heads,
        num_bilinear=config.num_bilinear,
        num_outlayer=config.num_outlayer,
        envelope_exponent=config.envelope_exponent,
        num_spherical=config.num_spherical,
        num_radial=config.num_radial,
        out_dim=config.out_dim,
        num_before_skip=config.num_before_skip,
        num_after_skip=config.num_after_skip,
        act=config.act,
        output_initializer=config.output_initializer,
        lr=config.lr,
        rf_csv_path=args.rf_data,
        log_memory=getattr(config, "log_memory", False),
        map_location="cpu",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = load_and_filter_heterodata(args.data_folder)
    file_names = sorted([p.name for p in Path(args.data_folder).glob("*.pt")])
    batch_size = args.batch_size or config.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        follow_batch=["atom", "magion"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, target = model(batch)
            summary = summarize_prediction(pred, target)
            start_idx = cursor
            end_idx = min(cursor + batch.num_graphs, len(file_names))
            summary["sample_files"] = ";".join(file_names[start_idx:end_idx])
            rows.append(summary)
            cursor = end_idx

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_files", "num_pairs", "pred_mean", "target_mean", "mse", "mae"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved inference summary to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
