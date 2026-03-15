"""CLI entry point for motion embedding operations.

Commands:
    train   — Train the motion encoder on a directory of .npy motion tracks.
    export  — Export a trained checkpoint to ONNX for C++ inference.
    embed   — Compute embeddings for a directory of motion files and save them.

Usage::

    python -m ml.motion_embeddings.cli train --data-dir ./data/tracks --epochs 200
    python -m ml.motion_embeddings.cli export --checkpoint checkpoints/motion_encoder_final.pt --output model.onnx
    python -m ml.motion_embeddings.cli embed --checkpoint checkpoints/motion_encoder_final.pt --data-dir ./data/tracks --output embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..constants import DEFAULT_SEQ_LEN, MOTION_INPUT_DIM, ONNX_OPSET
from .dataset import MotionSequenceDataset
from .encoder import MotionEncoder
from .export_onnx import export_encoder_onnx
from .trainer import EmbeddingTrainer, TrainerConfig

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Train the motion encoder on .npy motion tracks."""
    _setup_logging(args.verbose)

    dataset = MotionSequenceDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    config = TrainerConfig(
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        margin=args.margin,
    )

    device = torch.device(args.device)
    trainer = EmbeddingTrainer(config=config, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train(dataloader)

    log.info(
        "Training complete. Final loss: %.4f",
        history[-1]["loss"] if history else float("nan"),
    )


# ---------------------------------------------------------------------------
# Export command
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained checkpoint to ONNX."""
    _setup_logging(args.verbose)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model = MotionEncoder()
    model.load_state_dict(ckpt["model_state_dict"])

    output_path = export_encoder_onnx(
        model,
        args.output,
        seq_len=args.seq_len,
        opset=args.opset,
    )
    log.info("ONNX model exported to %s", output_path)


# ---------------------------------------------------------------------------
# Embed command
# ---------------------------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> None:
    """Compute embeddings for motion files and save to .npz."""
    _setup_logging(args.verbose)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model = MotionEncoder()
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Load dataset without augmentation for deterministic embeddings
    dataset = MotionSequenceDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        augment=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_embeddings: list[np.ndarray] = []
    all_track_ids: list[np.ndarray] = []

    log.info("Computing embeddings for %d windows...", len(dataset))

    with torch.no_grad():
        for sequences, track_ids in dataloader:
            sequences = sequences.to(device)
            embeddings = model(sequences)  # [B, 128]
            all_embeddings.append(embeddings.cpu().numpy())
            all_track_ids.append(track_ids.numpy())

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    track_ids_array = np.concatenate(all_track_ids, axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        embeddings=embeddings_array,
        track_ids=track_ids_array,
    )
    log.info(
        "Saved %d embeddings (dim=%d) to %s",
        embeddings_array.shape[0],
        embeddings_array.shape[1],
        output_path,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="motion_embeddings",
        description="Train, export, and compute motion embeddings.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train --
    p_train = subparsers.add_parser("train", help="Train the motion encoder.")
    p_train.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing .npy motion track files.",
    )
    p_train.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    p_train.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    p_train.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin.")
    p_train.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence window length.")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    p_train.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint interval (epochs).")
    p_train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p_train.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p_train.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.set_defaults(func=cmd_train)

    # -- export --
    p_export = subparsers.add_parser("export", help="Export trained encoder to ONNX.")
    p_export.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    p_export.add_argument(
        "--output",
        type=str,
        default="motion_encoder.onnx",
        help="Output ONNX file path.",
    )
    p_export.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Dummy sequence length.")
    p_export.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset version.")
    p_export.set_defaults(func=cmd_export)

    # -- embed --
    p_embed = subparsers.add_parser("embed", help="Compute embeddings for motion files.")
    p_embed.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    p_embed.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing .npy motion track files.",
    )
    p_embed.add_argument(
        "--output",
        type=str,
        default="embeddings.npz",
        help="Output .npz file path.",
    )
    p_embed.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    p_embed.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence window length.")
    p_embed.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p_embed.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_embed.set_defaults(func=cmd_embed)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
