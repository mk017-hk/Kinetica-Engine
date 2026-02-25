"""Trainer for the motion diffusion model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.motion_diffusion import MotionDiffusionModel

log = logging.getLogger(__name__)


class DiffusionTrainer:
    """Trains MotionDiffusionModel with Adam + cosine annealing."""

    def __init__(self, model: MotionDiffusionModel, lr: float = 1e-4,
                 weight_decay: float = 1e-5, grad_clip: float = 1.0,
                 device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, dataloader: DataLoader, epochs: int = 500,
              checkpoint_dir: str | Path = "checkpoints",
              checkpoint_every: int = 50) -> list[float]:
        """Run full training loop. Returns per-epoch average losses."""
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        epoch_losses: list[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for x0, condition in pbar:
                x0 = x0.to(self.device)
                condition = condition.to(self.device)

                loss = self.model.training_loss(x0, condition)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            lr = scheduler.get_last_lr()[0]
            log.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.5f}  lr={lr:.2e}")

            if epoch % checkpoint_every == 0:
                path = ckpt_path / f"diffusion_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_loss,
                }, path)
                log.info(f"Saved checkpoint: {path}")

        # Final save
        final_path = ckpt_path / "diffusion_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        log.info(f"Saved final model: {final_path}")

        return epoch_losses
