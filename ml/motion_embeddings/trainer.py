"""Training loop for the motion embedding encoder with triplet loss and hard mining.

Trains the MotionEncoder to produce embeddings where windows from the same
motion track are close together and windows from different tracks are far
apart. Uses triplet margin loss with batch-hard mining for efficient
negative selection.

Supports:
    - Triplet margin loss with batch-hard negative mining
    - Adam optimizer with cosine annealing LR schedule
    - Gradient clipping
    - Periodic checkpointing
    - Training metric logging
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..constants import MOTION_EMBEDDING_DIM, MOTION_INPUT_DIM
from .encoder import MotionEncoder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triplet loss with batch-hard mining
# ---------------------------------------------------------------------------


class BatchHardTripletLoss(nn.Module):
    """Triplet margin loss with batch-hard negative mining.

    For each anchor in the batch, selects:
        - Hardest positive: the farthest sample with the same track ID.
        - Hardest negative: the closest sample with a different track ID.

    This is more efficient than semi-hard mining and produces stronger
    gradients for learning discriminative embeddings.

    Args:
        margin: Minimum desired gap between positive and negative distances.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, track_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute batch-hard triplet loss.

        Args:
            embeddings: L2-normalized embeddings of shape [B, D].
            track_ids: Integer track IDs of shape [B].

        Returns:
            Tuple of (scalar loss, metrics dict with diagnostic values).
        """
        # Pairwise squared Euclidean distance (equivalent for L2-normalized vecs)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        B = embeddings.size(0)
        track_ids = track_ids.view(-1)

        # Masks for positive and negative pairs
        label_eq = track_ids.unsqueeze(0) == track_ids.unsqueeze(1)  # [B, B]
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = label_eq & ~eye
        neg_mask = ~label_eq

        # For each anchor, find hardest positive (max distance) and
        # hardest negative (min distance).
        # Set masked-out entries to extreme values so they are ignored by min/max.
        INF = 1e9

        # Hardest positive: max over positive distances
        pos_dists = dist_matrix.clone()
        pos_dists[~pos_mask] = -INF
        hardest_pos, _ = pos_dists.max(dim=1)  # [B]

        # Hardest negative: min over negative distances
        neg_dists = dist_matrix.clone()
        neg_dists[~neg_mask] = INF
        hardest_neg, _ = neg_dists.min(dim=1)  # [B]

        # Only compute loss for anchors that have at least one positive
        has_positive = pos_mask.any(dim=1)
        has_negative = neg_mask.any(dim=1)
        valid = has_positive & has_negative

        if not valid.any():
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero, {"loss": 0.0, "active_triplets": 0, "mean_pos_dist": 0.0, "mean_neg_dist": 0.0}

        triplet_loss = F.relu(
            hardest_pos[valid] - hardest_neg[valid] + self.margin
        )
        active = (triplet_loss > 0).sum().item()
        loss = triplet_loss.mean()

        metrics = {
            "loss": loss.item(),
            "active_triplets": int(active),
            "num_valid_anchors": int(valid.sum().item()),
            "mean_pos_dist": hardest_pos[valid].mean().item(),
            "mean_neg_dist": hardest_neg[valid].mean().item(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuration for the embedding trainer.

    Args:
        lr: Initial learning rate.
        min_lr: Minimum learning rate for cosine annealing.
        weight_decay: L2 regularization coefficient.
        margin: Triplet loss margin.
        grad_clip: Maximum gradient norm for clipping.
        epochs: Total number of training epochs.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_every: Save a checkpoint every N epochs.
        log_every: Log metrics every N batches.
    """

    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    margin: float = 0.3
    grad_clip: float = 1.0
    epochs: int = 200
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 50
    log_every: int = 10


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EmbeddingTrainer:
    """Trains a MotionEncoder with batch-hard triplet loss.

    Positive pairs are different windows from the same motion track.
    Negative pairs are windows from different tracks.

    Args:
        model: MotionEncoder instance (created with defaults if None).
        config: Training configuration.
        device: Torch device (auto-detected if None).
    """

    def __init__(
        self,
        model: MotionEncoder | None = None,
        config: TrainerConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = (model or MotionEncoder()).to(self.device)
        self.loss_fn = BatchHardTripletLoss(margin=self.config.margin)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_lr,
        )

        self.epoch_history: list[dict[str, float]] = []

    def _clip_gradients(self) -> float:
        """Clip gradients and return the total gradient norm before clipping."""
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        ).item()

    def train(self, dataloader: DataLoader) -> list[dict[str, float]]:
        """Run the full training loop.

        Args:
            dataloader: DataLoader yielding (sequence, track_id) batches.
                The batch should contain multiple samples per track for
                effective triplet mining.

        Returns:
            List of per-epoch metric dictionaries.
        """
        ckpt_path = Path(self.config.checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        log.info(
            "Starting training: %d epochs, lr=%.2e, device=%s, %d batches/epoch",
            self.config.epochs,
            self.config.lr,
            self.device,
            len(dataloader),
        )

        for epoch in range(1, self.config.epochs + 1):
            epoch_metrics = self._train_epoch(dataloader, epoch)
            self.scheduler.step()
            self.epoch_history.append(epoch_metrics)

            current_lr = self.optimizer.param_groups[0]["lr"]
            log.info(
                "Epoch %d/%d  loss=%.4f  active_triplets=%.0f  "
                "pos_dist=%.4f  neg_dist=%.4f  lr=%.2e",
                epoch,
                self.config.epochs,
                epoch_metrics["loss"],
                epoch_metrics["active_triplets"],
                epoch_metrics["mean_pos_dist"],
                epoch_metrics["mean_neg_dist"],
                current_lr,
            )

            # Periodic checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(ckpt_path, epoch, epoch_metrics["loss"])

        # Save final model
        final_path = ckpt_path / "motion_encoder_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        log.info("Saved final model: %s", final_path)

        return self.epoch_history

    def _train_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """Train for a single epoch.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number (1-indexed).

        Returns:
            Aggregated metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_active = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        num_batches = 0
        t0 = time.monotonic()

        for batch_idx, (sequences, track_ids) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            track_ids = track_ids.to(self.device)

            embeddings = self.model(sequences)  # [B, 128]
            loss, metrics = self.loss_fn(embeddings, track_ids)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = self._clip_gradients()
            self.optimizer.step()

            total_loss += metrics["loss"]
            total_active += metrics["active_triplets"]
            total_pos_dist += metrics["mean_pos_dist"]
            total_neg_dist += metrics["mean_neg_dist"]
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                log.debug(
                    "  [%d/%d] batch %d  loss=%.4f  grad_norm=%.3f",
                    epoch,
                    self.config.epochs,
                    batch_idx + 1,
                    metrics["loss"],
                    grad_norm,
                )

        n = max(num_batches, 1)
        elapsed = time.monotonic() - t0
        return {
            "loss": total_loss / n,
            "active_triplets": total_active / n,
            "mean_pos_dist": total_pos_dist / n,
            "mean_neg_dist": total_neg_dist / n,
            "epoch_time_s": elapsed,
        }

    def _save_checkpoint(
        self, ckpt_dir: Path, epoch: int, loss: float
    ) -> Path:
        """Save a training checkpoint.

        Args:
            ckpt_dir: Directory to save the checkpoint.
            epoch: Current epoch number.
            loss: Current epoch loss.

        Returns:
            Path to the saved checkpoint.
        """
        path = ckpt_dir / f"motion_encoder_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
            },
            path,
        )
        log.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load a training checkpoint to resume training.

        Args:
            checkpoint_path: Path to a .pt checkpoint file.

        Returns:
            The epoch number from the checkpoint.
        """
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch = ckpt.get("epoch", 0)
        log.info("Loaded checkpoint from epoch %d: %s", epoch, checkpoint_path)
        return epoch
