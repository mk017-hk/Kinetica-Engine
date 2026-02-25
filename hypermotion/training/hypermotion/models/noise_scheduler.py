"""Diffusion noise scheduler with linear beta schedule and DDIM sampling."""

import torch
import numpy as np


class NoiseScheduler:
    """Linear beta schedule for diffusion training + DDIM deterministic inference.

    Training: T=1000 steps, linear beta 0.0001 -> 0.02.
    Inference: 50-step DDIM sub-schedule.
    """

    def __init__(self, num_train_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, num_inference_steps: int = 50,
                 device: torch.device | None = None):
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        dev = device or torch.device("cpu")

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_train_steps, device=dev)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Precompute useful quantities
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

        # DDIM sub-schedule: evenly spaced timesteps
        self.ddim_timesteps = torch.linspace(
            num_train_steps - 1, 0, num_inference_steps, device=dev
        ).long()

    def to(self, device: torch.device) -> "NoiseScheduler":
        """Move all precomputed tensors to a device."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise.

        x0:    [batch, ...]
        t:     [batch] integer timesteps
        noise: [batch, ...] same shape as x0
        """
        sqrt_ab = self.sqrt_alphas_cumprod[t]
        sqrt_1m_ab = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast to match x0 dims
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1m_ab = sqrt_1m_ab.unsqueeze(-1)

        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def ddim_step(self, x_t: torch.Tensor, predicted_noise: torch.Tensor,
                  t: int, t_prev: int) -> torch.Tensor:
        """Single DDIM deterministic sampling step.

        x_t:             [batch, ...] current noisy sample
        predicted_noise: [batch, ...] model's noise prediction
        t:               current timestep (int)
        t_prev:          previous timestep (int), or -1 for final step
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x0
        x0_pred = (x_t - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

        # Direction pointing to x_t
        dir_xt = (1.0 - alpha_prev).sqrt() * predicted_noise

        # Deterministic DDIM (eta=0)
        return alpha_prev.sqrt() * x0_pred + dir_xt
