"""Training loop for the MotionPredictor temporal transformer.

Provides :class:`MotionPredictionTrainer` which handles:
    - MSE loss on predicted vs ground-truth next frames
    - Adam optimiser with cosine annealing LR schedule
    - Gradient clipping (max_norm=1.0)
    - Checkpoint saving (best validation loss + periodic)
    - Validation with Final Displacement Error (FDE) metric
    - Automatic train/validation split when no explicit val set is given
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from ..constants import FRAME_DIM, JOINT_COUNT, ROTATION_DIM
from .model import MotionPredictor
from .dataset import MotionPredictionDataset

logger = logging.getLogger(__name__)


class MotionPredictionTrainer:
    """End-to-end trainer for :class:`MotionPredictor`.

    Args:
        model: The :class:`MotionPredictor` model to train.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.  When *None* a fraction
            of ``train_dataset`` is held out automatically.
        output_dir: Directory for checkpoints and logs.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        grad_clip: Maximum gradient norm for clipping.
        val_split: Fraction of training data to use for validation when
            ``val_dataset`` is not provided.
        checkpoint_every: Save a periodic checkpoint every N epochs.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        model: MotionPredictor,
        train_dataset: MotionPredictionDataset,
        val_dataset: Optional[MotionPredictionDataset] = None,
        output_dir: str | Path = "checkpoints",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 100,
        grad_clip: float = 1.0,
        val_split: float = 0.1,
        checkpoint_every: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.checkpoint_every = checkpoint_every

        # ---- Data loaders ------------------------------------------------
        if val_dataset is not None:
            self.train_dataset: Dataset = train_dataset
            self.val_dataset: Dataset = val_dataset
        else:
            n_val = max(1, int(len(train_dataset) * val_split))
            n_train = len(train_dataset) - n_val
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [n_train, n_val]
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
        )

        # ---- Optimiser and scheduler -------------------------------------
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # ---- Tracking ----------------------------------------------------
        self.best_val_loss = float("inf")
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Dictionary with ``"train_loss"``, ``"val_loss"``, and
            ``"val_fde"`` lists (one entry per epoch).
        """
        logger.info(
            "Starting training: %d epochs, %d train / %d val samples",
            self.epochs,
            len(self.train_dataset),
            len(self.val_dataset),
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch()
            val_loss, val_fde = self._validate()

            self.scheduler.step()

            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_fde": val_fde,
                    "lr": lr_now,
                }
            )

            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  "
                "val_fde=%.4f  lr=%.2e  (%.1fs)",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_fde,
                lr_now,
                elapsed,
            )

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch, val_loss)
                logger.info("  -> New best model (val_loss=%.6f)", val_loss)

            # Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_loss)

        # Save final checkpoint
        self._save_checkpoint("final.pt", self.epochs, val_loss)
        logger.info("Training complete. Best val_loss=%.6f", self.best_val_loss)

        return {
            "train_loss": [h["train_loss"] for h in self.history],
            "val_loss": [h["val_loss"] for h in self.history],
            "val_fde": [h["val_fde"] for h in self.history],
        }

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Run validation and return ``(avg_loss, avg_fde)``.

        FDE (Final Displacement Error) measures the L2 distance between
        the predicted and ground-truth **last** frame in each sequence,
        averaged across joints.
        """
        self.model.eval()
        total_loss = 0.0
        total_fde = 0.0
        n_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()

            # FDE: L2 on the final frame, averaged per joint
            pred_last = predictions[:, -1, :]  # [B, FRAME_DIM]
            tgt_last = targets[:, -1, :]       # [B, FRAME_DIM]

            # Reshape to [B, JOINT_COUNT, ROTATION_DIM] for per-joint error
            pred_joints = pred_last.view(-1, JOINT_COUNT, ROTATION_DIM)
            tgt_joints = tgt_last.view(-1, JOINT_COUNT, ROTATION_DIM)
            fde = torch.norm(pred_joints - tgt_joints, dim=-1).mean().item()

            total_fde += fde
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_fde = total_fde / max(n_batches, 1)
        return avg_loss, avg_fde

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, filename: str, epoch: int, val_loss: float
    ) -> None:
        """Save a training checkpoint to ``output_dir / filename``."""
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
                "model_config": {
                    "frame_dim": self.model.frame_dim,
                    "d_model": self.model.d_model,
                    "max_positions": self.model.max_positions,
                },
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> tuple[MotionPredictor, dict]:
        """Load a model from a training checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            device: Device to map the model to.

        Returns:
            Tuple of ``(model, checkpoint_dict)``.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("model_config", {})
        model = MotionPredictor(
            frame_dim=config.get("frame_dim", FRAME_DIM),
            d_model=config.get("d_model", 512),
            max_positions=config.get("max_positions", 512),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model, ckpt
