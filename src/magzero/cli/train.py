from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from magzero import Magzero
from magzero.data_utils import load_and_filter_heterodata, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MagZero.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--data-folder", required=True, help="Folder containing heterograph .pt files.")
    parser.add_argument("--rf-data", default="graph_data.csv", help="CSV file used by the RandomForest helper.")
    parser.add_argument("--gpu", default="0", help="CUDA device ID exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--seed", default=8, type=int, help="Random seed.")
    parser.add_argument("--fold", default=0, type=int, help="Fold index used only for logging.")
    parser.add_argument("--project", default="MagZero", help="Weights & Biases project name.")
    parser.add_argument("--run-name", default=None, help="Optional run name for Weights & Biases.")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of DataLoader workers.")
    parser.add_argument("--train-ratio", default=0.8, type=float, help="Fraction of data used for training.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser.parse_args()


def make_dataloaders(data_folder: str, batch_size: int, train_ratio: float, num_workers: int):
    dataset = load_and_filter_heterodata(data_folder)
    if not dataset:
        raise RuntimeError(f"No valid HeteroData samples found in: {data_folder}")

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        follow_batch=["atom", "magion"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        follow_batch=["atom", "magion"],
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_model(config, rf_data: str) -> Magzero:
    return Magzero(
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
        rf_csv_path=rf_data,
        log_memory=getattr(config, "log_memory", False),
    )


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pl.seed_everything(args.seed, workers=True)
    config = load_yaml(args.config)

    train_loader, val_loader = make_dataloaders(
        data_folder=args.data_folder,
        batch_size=config.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
    )

    model = build_model(config, args.rf_data)

    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.project,
            name=args.run_name or f"fold{args.fold}-seed{args.seed}",
            log_model=True,
        )
        logger.experiment.config.update({
            "batch_size": config.batch_size,
            "config_path": str(Path(args.config).resolve()),
            "data_folder": str(Path(args.data_folder).resolve()),
        })

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        enable_checkpointing=True,
        precision=16 if getattr(config, "use_fp16", False) else 32,
        gradient_clip_val=getattr(config, "gradient_clip_val", 0.0),
        deterministic=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metrics = trainer.validate(model, val_loader, ckpt_path="best")

    print(f"Best validation loss: {metrics[0]['val_loss']:.6f}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
