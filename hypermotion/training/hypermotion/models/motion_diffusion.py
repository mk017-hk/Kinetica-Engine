"""Motion diffusion model: wraps transformer + condition encoder + noise scheduler.

Training:  x0 -> sample t -> add noise -> predict noise -> MSE loss.
Inference: noise -> 50 DDIM steps -> 64 frames of clean motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion_transformer import MotionTransformer
from .condition_encoder import ConditionEncoder
from .noise_scheduler import NoiseScheduler
from .constants import FRAME_DIM, CONDITION_DIM


class MotionDiffusionModel(nn.Module):
    """High-level diffusion model for conditional motion generation."""

    def __init__(self, frame_dim: int = FRAME_DIM, cond_dim: int = CONDITION_DIM,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 cond_latent_dim: int = 256, num_train_steps: int = 1000,
                 num_inference_steps: int = 50):
        super().__init__()

        self.condition_encoder = ConditionEncoder(cond_dim, 512, cond_latent_dim)
        self.transformer = MotionTransformer(
            frame_dim, d_model, nhead, num_layers, dim_feedforward, dropout, cond_latent_dim
        )
        self.scheduler = NoiseScheduler(num_train_steps, num_inference_steps=num_inference_steps)

        self.frame_dim = frame_dim
        self.num_train_steps = num_train_steps

    def to(self, device: torch.device, *args, **kwargs) -> "MotionDiffusionModel":
        result = super().to(device, *args, **kwargs)
        result.scheduler = result.scheduler.to(device)
        return result

    def training_loss(self, x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute training loss (MSE on noise prediction).

        x0:        [batch, seq_len, frame_dim]  clean motion
        condition: [batch, CONDITION_DIM]        motion condition
        """
        batch = x0.shape[0]
        device = x0.device

        # Random timestep per sample
        t = torch.randint(0, self.num_train_steps, (batch,), device=device)

        # Sample noise and create noisy input
        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, t, noise)

        # Encode condition and predict noise
        cond_emb = self.condition_encoder(condition)
        predicted_noise = self.transformer(x_t, t, cond_emb)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, seq_len: int = 64) -> torch.Tensor:
        """Generate motion via DDIM sampling.

        condition: [batch, CONDITION_DIM]
        Returns:   [batch, seq_len, frame_dim]
        """
        device = condition.device
        batch = condition.shape[0]

        # Encode condition once
        cond_emb = self.condition_encoder(condition)

        # Start from pure noise
        x = torch.randn(batch, seq_len, self.frame_dim, device=device)

        # DDIM denoising loop
        timesteps = self.scheduler.ddim_timesteps
        for i in range(len(timesteps)):
            t_cur = timesteps[i].item()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

            t_batch = torch.full((batch,), t_cur, device=device, dtype=torch.long)
            predicted_noise = self.transformer(x, t_batch, cond_emb)
            x = self.scheduler.ddim_step(x, predicted_noise, t_cur, t_prev)

        return x
