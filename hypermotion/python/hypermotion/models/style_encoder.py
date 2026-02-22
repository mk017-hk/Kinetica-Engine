"""Style encoder: variable-length motion -> 64D L2-normalized embedding.

Architecture:
  - Conv1D(201->128, k=3, p=1) -> BN -> ReLU
  - 4 ResBlocks: 128->128->256->256->512
  - Global Average Pooling over time
  - Linear(512->256) -> ReLU -> Linear(256->64) -> L2 norm
  ~1.9M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import STYLE_INPUT_DIM, STYLE_DIM


class StyleResBlock(nn.Module):
    """Residual block: Conv1D -> BN -> ReLU -> Conv1D -> BN + skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class StyleEncoder(nn.Module):
    """Encodes variable-length motion sequences to a fixed 64D style vector.

    Input:  [batch, time, 201]
    Output: [batch, 64] (L2 normalized)
    """

    def __init__(self, input_dim: int = STYLE_INPUT_DIM, style_dim: int = STYLE_DIM):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            StyleResBlock(128, 128),
            StyleResBlock(128, 256),
            StyleResBlock(256, 256),
            StyleResBlock(256, 512),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, time, input_dim] -> [batch, style_dim]."""
        h = x.transpose(1, 2)       # [B, C, T]
        h = self.input_conv(h)
        h = self.res_blocks(h)
        h = self.pool(h).squeeze(2)  # [B, 512]
        h = self.head(h)             # [B, style_dim]
        return F.normalize(h, p=2, dim=-1)
