"""Trainer for the contrastive style encoder."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.style_encoder import StyleEncoder
from ..models.contrastive_loss import PairwiseNTXentLoss

log = logging.getLogger(__name__)


class StyleTrainer:
    """Trains StyleEncoder via NT-Xent contrastive loss.

    Adam lr=1e-4, cosine annealing, 200 epochs, gradient clipping.
    """

    def __init__(self, model: StyleEncoder, lr: float = 1e-4, min_lr: float = 1e-6,
                 weight_decay: float = 1e-5, temperature: float = 0.07,
                 grad_clip: float = 1.0, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = PairwiseNTXentLoss(temperature)
        self.grad_clip = grad_clip
        self.lr = lr
        self.min_lr = min_lr

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _cosine_lr(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealing schedule."""
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * epoch / total_epochs)
        )

    def train(self, dataloader: DataLoader, epochs: int = 200,
              checkpoint_dir: str | Path = "checkpoints",
              checkpoint_every: int = 50) -> list[float]:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            # Update learning rate
            lr = self._cosine_lr(epoch, epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for clip_a, clip_b in pbar:
                clip_a = clip_a.to(self.device)
                clip_b = clip_b.to(self.device)

                emb_a = self.model(clip_a)  # [B, 64]
                emb_b = self.model(clip_b)  # [B, 64]

                loss = self.loss_fn(emb_a, emb_b)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            log.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={lr:.2e}")

            if epoch % checkpoint_every == 0:
                path = ckpt_path / f"style_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "loss": avg_loss,
                }, path)

        final_path = ckpt_path / "style_encoder_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        log.info(f"Saved final model: {final_path}")

        return epoch_losses
