"""Trainer for the motion encoder using triplet loss.

Minimises distance between embeddings of similar motions (same label/cluster)
and separates different motions in the 128D embedding space.

Supports:
  - Online triplet mining (semi-hard negatives)
  - Cosine annealing LR schedule
  - Gradient clipping
  - Periodic checkpointing
  - ONNX export of trained model
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..models.motion_encoder import MotionEncoder, MOTION_INPUT_DIM, MOTION_EMBEDDING_DIM

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triplet loss with online mining
# ---------------------------------------------------------------------------

class OnlineTripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining.

    For each anchor-positive pair, selects the hardest negative that is
    farther than the positive but within the margin (semi-hard).
    Falls back to hardest negative if no semi-hard exists.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] integer class labels

        Returns:
            scalar triplet loss
        """
        # Pairwise distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        B = embeddings.size(0)
        labels = labels.view(-1)

        # Mask for positive and negative pairs
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        label_neq = ~label_eq

        # Exclude self-pairs from positives
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = label_eq & ~eye
        neg_mask = label_neq

        loss = torch.tensor(0.0, device=embeddings.device)
        num_triplets = 0

        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            for p_idx in pos_indices:
                d_ap = dist_matrix[i, p_idx]
                d_an = dist_matrix[i, neg_indices]

                # Semi-hard: d_ap < d_an < d_ap + margin
                semi_hard = (d_an > d_ap) & (d_an < d_ap + self.margin)
                if semi_hard.any():
                    d_neg = d_an[semi_hard].max()
                else:
                    d_neg = d_an.min()  # hardest negative fallback

                triplet_loss = F.relu(d_ap - d_neg + self.margin)
                loss = loss + triplet_loss
                num_triplets += 1

        if num_triplets > 0:
            loss = loss / num_triplets
        return loss


# ---------------------------------------------------------------------------
# Dataset wrapper for .npz training data
# ---------------------------------------------------------------------------

class MotionEmbeddingDataset(Dataset):
    """Loads sequences and labels from a .npz file built by motion_dataset_builder."""

    def __init__(self, npz_path: str | Path, augment: bool = True):
        data = np.load(str(npz_path))
        self.sequences = data["sequences"]  # [N, seq_len, 66]
        self.labels = data["labels"]        # [N]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()  # [seq_len, 66]
        label = self.labels[idx]

        if self.augment:
            # Random Gaussian noise
            if np.random.random() < 0.3:
                seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)

            # Random temporal shift (roll)
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 6)
                seq = np.roll(seq, shift, axis=0)

            # Random speed perturbation
            if np.random.random() < 0.2:
                speed = np.random.uniform(0.85, 1.15)
                T = seq.shape[0]
                new_T = int(T * speed)
                if new_T >= T:
                    indices = np.linspace(0, T - 1, new_T).astype(int)
                    seq = seq[indices[:T]]
                else:
                    indices = np.linspace(0, T - 1, new_T).astype(int)
                    stretched = seq[indices]
                    pad = np.tile(stretched[-1:], (T - new_T, 1))
                    seq = np.concatenate([stretched, pad], axis=0)

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MotionEncoderTrainer:
    """Trains MotionEncoder with online triplet loss."""

    def __init__(
        self,
        model: MotionEncoder | None = None,
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 1e-5,
        margin: float = 0.3,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = (model or MotionEncoder()).to(self.device)
        self.loss_fn = OnlineTripletLoss(margin=margin)
        self.grad_clip = grad_clip
        self.lr = lr
        self.min_lr = min_lr

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _cosine_lr(self, epoch: int, total_epochs: int) -> float:
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * epoch / total_epochs)
        )

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 200,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_every: int = 50,
    ) -> list[float]:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            lr = self._cosine_lr(epoch, epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(sequences)  # [B, 128]
                loss = self.loss_fn(embeddings, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            log.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={lr:.2e}")

            if epoch % checkpoint_every == 0:
                path = ckpt_path / f"motion_encoder_epoch{epoch:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    path,
                )
                log.info(f"Checkpoint: {path}")

        final_path = ckpt_path / "motion_encoder_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        log.info(f"Saved final model: {final_path}")

        return epoch_losses

    def export_onnx(self, output_path: str | Path, seq_len: int = 64,
                    opset: int = 17) -> Path:
        """Export trained model to ONNX for C++ inference."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy = torch.randn(1, seq_len, MOTION_INPUT_DIM, device=self.device)

        torch.onnx.export(
            self.model,
            (dummy,),
            str(output_path),
            opset_version=opset,
            input_names=["joint_positions"],
            output_names=["motion_embedding"],
            dynamic_axes={
                "joint_positions": {0: "batch", 1: "seq_len"},
                "motion_embedding": {0: "batch"},
            },
        )
        log.info(f"Exported motion encoder ONNX: {output_path}")
        return output_path
