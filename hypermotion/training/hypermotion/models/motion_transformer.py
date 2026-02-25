"""Denoising transformer for diffusion-based motion generation.

Architecture:
  - Input projection: FRAME_DIM (132) -> 512
  - Timestep embedding: sinusoidal PE -> MLP -> 512
  - Condition projection: 256 -> 512
  - 8 pre-norm transformer encoder layers (8 heads, FFN 2048, GELU, dropout 0.1)
  - Output projection: 512 -> FRAME_DIM (132)
  ~17.6M parameters
"""

import math

import torch
import torch.nn as nn

from .constants import FRAME_DIM


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal timestep embedding a la 'Attention Is All You Need'."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [batch] integer timesteps -> [batch, dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal PE -> Linear -> GELU -> Linear."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pe(t))


class PreNormTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer with GELU FFN."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class MotionTransformer(nn.Module):
    """Denoising transformer for the motion diffusion model.

    Input:  noisy motion [batch, seq_len, FRAME_DIM]
            timestep     [batch]
            condition    [batch, 256]
    Output: predicted noise [batch, seq_len, FRAME_DIM]
    """

    def __init__(self, frame_dim: int = FRAME_DIM, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
                 cond_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(frame_dim, d_model)
        self.timestep_embed = TimestepEmbedder(d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, frame_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [batch, seq_len, frame_dim]  noisy motion
        t:    [batch]                      diffusion timestep
        cond: [batch, cond_dim]            encoded condition
        Returns: [batch, seq_len, frame_dim] predicted noise
        """
        batch, seq_len, _ = x.shape

        # Project inputs to d_model
        h = self.input_proj(x)                        # [B, S, D]
        t_emb = self.timestep_embed(t).unsqueeze(1)   # [B, 1, D]
        c_emb = self.cond_proj(cond).unsqueeze(1)     # [B, 1, D]

        # Add timestep and condition as bias to every token
        h = h + t_emb + c_emb

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        return self.output_proj(h)
