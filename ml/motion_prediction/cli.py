"""CLI entry point for motion prediction training, export, and inference.

Usage:
    python -m ml.motion_prediction.cli train   --data-dir sequences/ --epochs 100 --output checkpoints/
    python -m ml.motion_prediction.cli export  --checkpoint best.pt --output predictor.onnx
    python -m ml.motion_prediction.cli predict --checkpoint best.pt --context input.npy --n-future 30 --output prediction.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from ..constants import FRAME_DIM, DEFAULT_SEQ_LEN
from .model import MotionPredictor
from .dataset import MotionPredictionDataset
from .trainer import MotionPredictionTrainer
from .export_onnx import export_predictor

logger = logging.getLogger(__name__)


# ======================================================================
# Sub-commands
# ======================================================================

def cmd_train(args: argparse.Namespace) -> None:
    """Train a MotionPredictor model on motion sequence data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading dataset from %s", args.data_dir)
    train_dataset = MotionPredictionDataset(
        data_dir=args.data_dir,
        context_len=args.context_len,
        rotation_noise_std=args.noise_std,
    )
    logger.info(train_dataset.summary())

    val_dataset = None
    if args.val_dir is not None:
        val_dataset = MotionPredictionDataset(
            data_dir=args.val_dir,
            context_len=args.context_len,
            rotation_noise_std=0.0,
        )
        logger.info("Validation: %s", val_dataset.summary())

    model = MotionPredictor(
        frame_dim=FRAME_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s (%.2f M)", f"{n_params:,}", n_params / 1e6)

    trainer = MotionPredictionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
    )

    history = trainer.train()

    # Save training history
    history_path = Path(args.output) / "history.npy"
    np.save(str(history_path), history)
    logger.info("Training history saved to %s", history_path)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model checkpoint to ONNX."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, ckpt = MotionPredictionTrainer.load_checkpoint(
        args.checkpoint, device="cpu"
    )

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    logger.info("Checkpoint epoch=%s, val_loss=%.6f", epoch, val_loss)

    output_path = export_predictor(
        model=model,
        output_path=args.output,
        seq_len=args.seq_len,
        validate=not args.skip_validation,
        verbose=True,
    )
    logger.info("ONNX model exported to %s", output_path)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run autoregressive prediction from a context sequence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, _ = MotionPredictionTrainer.load_checkpoint(
        args.checkpoint, device=args.device
    )

    logger.info("Loading context: %s", args.context)
    context = np.load(args.context).astype(np.float32)

    if context.ndim != 2 or context.shape[1] != FRAME_DIM:
        raise ValueError(
            f"Context must have shape [T, {FRAME_DIM}], got {context.shape}"
        )

    context_tensor = torch.from_numpy(context).unsqueeze(0).to(args.device)
    logger.info(
        "Context: %d frames, predicting %d future frames",
        context.shape[0],
        args.n_future,
    )

    prediction = model.predict_future(context_tensor, args.n_future)
    prediction = prediction.squeeze(0).cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), prediction)
    logger.info(
        "Prediction saved: %s (shape %s)", output_path, prediction.shape
    )


# ======================================================================
# Argument parser
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ml.motion_prediction",
        description="Train, export, and run the MotionPredictor temporal transformer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train -----------------------------------------------------------
    p_train = subparsers.add_parser("train", help="Train a motion prediction model.")
    p_train.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing .npy motion sequence files.",
    )
    p_train.add_argument(
        "--val-dir", type=str, default=None,
        help="Optional separate validation data directory.",
    )
    p_train.add_argument(
        "--output", type=str, default="checkpoints",
        help="Output directory for checkpoints (default: checkpoints/).",
    )
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-5)
    p_train.add_argument("--grad-clip", type=float, default=1.0)
    p_train.add_argument("--context-len", type=int, default=DEFAULT_SEQ_LEN)
    p_train.add_argument("--noise-std", type=float, default=0.002,
                         help="Rotation noise std for augmentation.")
    p_train.add_argument("--d-model", type=int, default=512)
    p_train.add_argument("--n-heads", type=int, default=8)
    p_train.add_argument("--n-layers", type=int, default=6)
    p_train.add_argument("--d-ff", type=int, default=2048)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--checkpoint-every", type=int, default=10)
    p_train.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p_train.set_defaults(func=cmd_train)

    # ---- export ----------------------------------------------------------
    p_export = subparsers.add_parser("export", help="Export model to ONNX.")
    p_export.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (.pt).",
    )
    p_export.add_argument(
        "--output", type=str, default="predictor.onnx",
        help="Output ONNX file path.",
    )
    p_export.add_argument("--seq-len", type=int, default=32,
                          help="Sequence length for dummy input during tracing.")
    p_export.add_argument("--skip-validation", action="store_true",
                          help="Skip onnxruntime validation after export.")
    p_export.set_defaults(func=cmd_export)

    # ---- predict ---------------------------------------------------------
    p_predict = subparsers.add_parser(
        "predict", help="Autoregressively predict future frames.",
    )
    p_predict.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (.pt).",
    )
    p_predict.add_argument(
        "--context", type=str, required=True,
        help="Path to context .npy file with shape [T, 132].",
    )
    p_predict.add_argument("--n-future", type=int, default=30,
                           help="Number of future frames to predict.")
    p_predict.add_argument(
        "--output", type=str, default="prediction.npy",
        help="Output .npy file for predicted frames.",
    )
    p_predict.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
