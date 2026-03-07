#!/usr/bin/env python3
"""Train the motion encoder on extracted animation clips.

Usage:
    python -m scripts.train_motion_encoder \\
        --data-dir /path/to/clips \\
        --output-dir checkpoints/motion_encoder \\
        --epochs 200 \\
        --batch-size 64

The script:
  1. Builds a training dataset from JSON animation clips
  2. Trains the motion encoder with triplet loss
  3. Exports the model to ONNX for C++ inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypermotion.data.motion_dataset_builder import build_dataset
from hypermotion.models.motion_encoder import MotionEncoder
from hypermotion.training.motion_encoder_trainer import (
    MotionEmbeddingDataset,
    MotionEncoderTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("train_motion_encoder")


def main():
    parser = argparse.ArgumentParser(description="Train motion encoder")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing JSON animation clips")
    parser.add_argument("--output-dir", type=str, default="checkpoints/motion_encoder",
                        help="Directory for checkpoints and ONNX export")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet loss margin")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length for training samples")
    parser.add_argument("--stride", type=int, default=32,
                        help="Sliding window stride for dataset building")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build dataset
    dataset_path = output_dir / "training_data.npz"
    log.info("Building training dataset...")
    stats = build_dataset(
        data_dir=args.data_dir,
        output_path=dataset_path,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    log.info(f"Dataset: {stats}")

    if stats["num_sequences"] == 0:
        log.error("No training sequences found. Check --data-dir.")
        sys.exit(1)

    # Step 2: Create data loader
    dataset = MotionEmbeddingDataset(dataset_path, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    log.info(f"DataLoader: {len(dataset)} samples, {len(dataloader)} batches")

    # Step 3: Train
    device = torch.device(args.device) if args.device else None
    trainer = MotionEncoderTrainer(
        lr=args.lr,
        margin=args.margin,
        device=device,
    )
    log.info(f"Training on {trainer.device} for {args.epochs} epochs...")

    losses = trainer.train(
        dataloader,
        epochs=args.epochs,
        checkpoint_dir=str(output_dir),
        checkpoint_every=50,
    )

    # Step 4: Export ONNX
    onnx_path = output_dir / "motion_encoder.onnx"
    trainer.export_onnx(onnx_path, seq_len=args.seq_len)

    log.info("Training complete!")
    log.info(f"  Final loss: {losses[-1]:.4f}")
    log.info(f"  ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
