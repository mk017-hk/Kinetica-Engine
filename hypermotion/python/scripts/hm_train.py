#!/usr/bin/env python3
"""HyperMotion training CLI.

Usage:
  # Train motion diffusion model
  python hm_train.py --mode diffusion --data clips/ --epochs 500 --batch 64 --lr 1e-4

  # Train motion classifier (TCN)
  python hm_train.py --mode classifier --data segments/ --epochs 100 --batch 32

  # Export trained models to ONNX
  python hm_train.py --mode diffusion --checkpoint checkpoints/diffusion_final.pt --export models/
  python hm_train.py --mode classifier --checkpoint checkpoints/classifier_final.pt --export models/
"""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_diffusion(args):
    from hypermotion.models.motion_diffusion import MotionDiffusionModel
    from hypermotion.data.motion_dataset import MotionClipDataset
    from hypermotion.training.diffusion_trainer import DiffusionTrainer
    from hypermotion.export.onnx_export import export_diffusion_denoiser

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = MotionDiffusionModel()

    # Export mode: load checkpoint and export to ONNX
    if args.checkpoint and args.export:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        export_diffusion_denoiser(model, f"{args.export}/diffusion_denoiser.onnx")
        log.info("Export complete.")
        return

    # Training mode
    if not args.data:
        log.error("--data is required for training")
        sys.exit(1)

    dataset = MotionClipDataset(args.data, seq_len=64, augment=True)
    if len(dataset) == 0:
        log.error(f"No valid clips found in {args.data}")
        sys.exit(1)
    log.info(f"Loaded {len(dataset)} motion clips")

    dataloader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Resume from checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Resumed from {args.checkpoint}")

    trainer = DiffusionTrainer(model, lr=args.lr, device=device)
    losses = trainer.train(
        dataloader, epochs=args.epochs,
        checkpoint_dir=args.output, checkpoint_every=50,
    )
    log.info(f"Training complete. Final loss: {losses[-1]:.5f}")


def train_classifier(args):
    from hypermotion.models.temporal_conv_net import TemporalConvNet
    from hypermotion.data.motion_dataset import ClassifierDataset
    from hypermotion.training.classifier_trainer import ClassifierTrainer
    from hypermotion.export.onnx_export import export_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = TemporalConvNet()

    # Export mode
    if args.checkpoint and args.export:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        export_classifier(model, f"{args.export}/motion_classifier.onnx")
        log.info("Export complete.")
        return

    # Training mode
    if not args.data:
        log.error("--data is required for training")
        sys.exit(1)

    dataset = ClassifierDataset(args.data, max_seq_len=256)
    if len(dataset) == 0:
        log.error(f"No valid segments found in {args.data}")
        sys.exit(1)
    log.info(f"Loaded {len(dataset)} motion segments")

    dataloader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        collate_fn=ClassifierTrainer.collate_fn,
        num_workers=4, pin_memory=True,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Resumed from {args.checkpoint}")

    trainer = ClassifierTrainer(model, lr=args.lr, device=device)
    losses = trainer.train(
        dataloader, epochs=args.epochs,
        checkpoint_dir=args.output, checkpoint_every=25,
    )
    log.info(f"Training complete. Final loss: {losses[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="HyperMotion ML Training")
    parser.add_argument("--mode", required=True, choices=["diffusion", "classifier"],
                        help="Training mode: 'diffusion' or 'classifier'")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (or export from)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export trained model to ONNX in this directory")
    args = parser.parse_args()

    if args.mode == "diffusion":
        train_diffusion(args)
    elif args.mode == "classifier":
        train_classifier(args)


if __name__ == "__main__":
    main()
