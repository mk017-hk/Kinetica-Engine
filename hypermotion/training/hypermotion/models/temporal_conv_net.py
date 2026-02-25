"""Temporal Convolutional Network for per-frame motion classification.

Architecture:
  - Input projection: Conv1D(70->128, k=1)
  - 6 dilated causal conv blocks [1,2,4,8,16,32], receptive field ~190 frames
  - Each block: Conv1D(k=3) -> BN -> ReLU -> Dropout -> Conv1D -> BN -> ReLU + residual
  - Output projection: Conv1D(128->16, k=1)
  ~600K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import FEATURE_DIM_SEGMENTER, MOTION_TYPE_COUNT


class TCNBlock(nn.Module):
    """Single dilated causal convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left side
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]."""
        residual = self.residual(x)

        # First conv with causal padding
        out = F.pad(x, (self.causal_pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        # Second conv with causal padding
        out = F.pad(out, (self.causal_pad, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + residual, inplace=True)

        return out


class TemporalConvNet(nn.Module):
    """Dilated temporal CNN for motion classification.

    Input:  [batch, time, 70]  (features per frame)
    Output: [batch, time, 16]  (logits per frame per motion type)
    """

    def __init__(self, input_dim: int = FEATURE_DIM_SEGMENTER, hidden_dim: int = 128,
                 num_classes: int = MOTION_TYPE_COUNT, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        dilations = [1, 2, 4, 8, 16, 32]

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, hidden_dim, kernel_size, d, dropout)
            for d in dilations
        ])

        self.output_proj = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, time, input_dim] -> [batch, time, num_classes]."""
        # Conv1d expects [B, C, T]
        h = x.transpose(1, 2)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        logits = self.output_proj(h)
        return logits.transpose(1, 2)  # back to [B, T, C]
