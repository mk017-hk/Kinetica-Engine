"""Motion encoder: variable-length joint position sequence -> 128D embedding.

Architecture (Temporal CNN):
  - Conv1D(JOINT_COUNT*3 -> 128, k=3) -> BN -> ReLU
  - 4 ResBlocks: 128->128->256->256->512  (dilated causal convolutions)
  - Global Average Pooling over time
  - Linear(512->256) -> ReLU -> Linear(256->128) -> L2 norm
  ~1.6M parameters

Input:  [batch, time, JOINT_COUNT * 3]  (22 joints * 3 = 66D world positions)
Output: [batch, 128] (L2 normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import JOINT_COUNT

MOTION_EMBEDDING_DIM = 128
MOTION_INPUT_DIM = JOINT_COUNT * 3  # 22 joints * 3D position = 66


class TemporalResBlock(nn.Module):
    """Dilated temporal residual block: Conv1D -> BN -> ReLU -> Conv1D -> BN + skip."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class MotionEncoder(nn.Module):
    """Encodes variable-length joint position sequences to a 128D embedding.

    Input:  [batch, time, 66]  (22 joints * 3D world position)
    Output: [batch, 128] (L2 normalized)
    """

    def __init__(
        self,
        input_dim: int = MOTION_INPUT_DIM,
        embedding_dim: int = MOTION_EMBEDDING_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Dilated residual blocks for multi-scale temporal receptive field
        self.res_blocks = nn.Sequential(
            TemporalResBlock(128, 128, dilation=1),
            TemporalResBlock(128, 256, dilation=2),
            TemporalResBlock(256, 256, dilation=4),
            TemporalResBlock(256, 512, dilation=8),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, time, input_dim] -> [batch, embedding_dim]."""
        h = x.transpose(1, 2)          # [B, C, T]
        h = self.input_conv(h)
        h = self.res_blocks(h)
        h = self.pool(h).squeeze(2)    # [B, 512]
        h = self.head(h)               # [B, embedding_dim]
        return F.normalize(h, p=2, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — encode a motion sequence to an embedding."""
        return self.forward(x)
