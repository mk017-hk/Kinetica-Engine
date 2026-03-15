"""Temporal transformer for motion frame prediction.

Architecture (pre-norm transformer encoder with causal masking):
    Input projection:  Linear(FRAME_DIM -> 512)
    Positional encoding: learned embeddings, max 512 positions
    Transformer encoder: 6 layers, 8 heads, d_model=512, d_ff=2048, GELU, dropout=0.1
    Output projection: Linear(512 -> FRAME_DIM)

Input:  [batch, context_len, FRAME_DIM]   (22 joints x 6D rotation = 132)
Output: [batch, context_len, FRAME_DIM]   (next-frame predictions for each position)

During inference the ``predict_future`` method autoregressively generates
*n_future* frames by feeding each predicted frame back as context.

The model is compatible with the C++ OnnxInference pipeline after ONNX export.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import FRAME_DIM


class MotionPredictor(nn.Module):
    """Causal temporal transformer for next-frame motion prediction.

    Args:
        frame_dim: Per-frame feature dimension (default: 132).
        d_model: Internal transformer dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_positions: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        frame_dim: int = FRAME_DIM,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.d_model = d_model
        self.max_positions = max_positions

        # Input / output projections
        self.input_proj = nn.Linear(frame_dim, d_model)
        self.output_proj = nn.Linear(d_model, frame_dim)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_positions, d_model)

        # Pre-norm transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Causal mask
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create an upper-triangular causal attention mask.

        Returns:
            Boolean mask of shape [seq_len, seq_len] where ``True`` means
            the position is **blocked** from attending (following the
            PyTorch ``nn.TransformerEncoder`` convention).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the transformer on a sequence of motion frames.

        Args:
            frames: Input tensor of shape ``[batch, seq_len, frame_dim]``.
            mask: Optional pre-computed causal mask.  If *None* a causal
                mask is generated automatically.

        Returns:
            Predicted next-frame tensor of shape ``[batch, seq_len, frame_dim]``.
        """
        batch_size, seq_len, _ = frames.shape

        if seq_len > self.max_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positions "
                f"{self.max_positions}."
            )

        # Positional indices [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=frames.device)

        # Project and add positional encoding
        h = self.input_proj(frames) + self.pos_embedding(positions)
        h = self.dropout(h)

        # Causal mask
        if mask is None:
            mask = self._generate_causal_mask(seq_len, frames.device)

        h = self.transformer(h, mask=mask)

        return self.output_proj(h)

    # ------------------------------------------------------------------
    # Autoregressive inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_future(
        self,
        context: torch.Tensor,
        n_future: int,
    ) -> torch.Tensor:
        """Autoregressively predict *n_future* frames given context.

        At each step the last predicted frame is appended to the context
        and a new forward pass produces the next frame.  Only the final
        ``n_future`` frames are returned.

        Args:
            context: Context frames of shape ``[batch, T, frame_dim]``.
            n_future: Number of future frames to generate.

        Returns:
            Predicted frames of shape ``[batch, n_future, frame_dim]``.
        """
        self.eval()
        generated: list[torch.Tensor] = []
        current = context

        for _ in range(n_future):
            # Truncate to max_positions if needed (sliding window)
            if current.shape[1] > self.max_positions:
                current = current[:, -self.max_positions :, :]

            pred = self.forward(current)
            # Take the prediction at the last position
            next_frame = pred[:, -1:, :]
            generated.append(next_frame)
            current = torch.cat([current, next_frame], dim=1)

        return torch.cat(generated, dim=1)
