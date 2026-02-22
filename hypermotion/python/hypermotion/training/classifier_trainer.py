"""Trainer for the temporal convolutional network motion classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.temporal_conv_net import TemporalConvNet

log = logging.getLogger(__name__)


def _collate_variable_length(batch):
    """Pad variable-length sequences to max length in batch."""
    features, labels = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    dim = features[0].shape[1]

    padded_f = torch.zeros(len(features), max_len, dim)
    padded_l = torch.full((len(labels), max_len), -100, dtype=torch.long)  # -100 = ignore

    for i, (f, l) in enumerate(zip(features, labels)):
        T = f.shape[0]
        padded_f[i, :T] = f
        padded_l[i, :T] = l

    return padded_f, padded_l


class ClassifierTrainer:
    """Trains TemporalConvNet with Adam + cross-entropy."""

    def __init__(self, model: TemporalConvNet, lr: float = 1e-3,
                 weight_decay: float = 1e-4, grad_clip: float = 1.0,
                 device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, dataloader: DataLoader, epochs: int = 100,
              checkpoint_dir: str | Path = "checkpoints",
              checkpoint_every: int = 25) -> list[float]:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_frames = 0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)  # [B, T, C]

                # Reshape for cross-entropy: [B*T, C] vs [B*T]
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(-1, C), labels.reshape(-1), ignore_index=-100
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Accuracy on valid frames
                mask = labels != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds[mask] == labels[mask]).sum().item()
                    total_frames += mask.sum().item()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            accuracy = total_correct / max(total_frames, 1)
            epoch_losses.append(avg_loss)

            log.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.3f}")

            if epoch % checkpoint_every == 0:
                path = ckpt_path / f"classifier_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "loss": avg_loss,
                    "accuracy": accuracy,
                }, path)

        final_path = ckpt_path / "classifier_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        log.info(f"Saved final model: {final_path}")

        return epoch_losses

    @staticmethod
    def collate_fn(batch):
        return _collate_variable_length(batch)
