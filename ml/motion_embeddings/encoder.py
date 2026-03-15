"""Motion encoder: variable-length joint position sequence -> 128D embedding.

Architecture (Temporal CNN with dilated residual blocks):
    Input(66) -> Conv1D(66->128, k=3) -> BN -> ReLU
    -> TemporalResBlock(128, 128, dilation=1)
    -> TemporalResBlock(128, 256, dilation=2)
    -> TemporalResBlock(256, 256, dilation=4)
    -> TemporalResBlock(256, 512, dilation=8)
    -> Global Avg Pool
    -> Linear(512->256) -> ReLU -> Dropout -> Linear(256->128) -> L2 norm

This mirrors the architecture in hypermotion/training/hypermotion/models/motion_encoder.py
and produces outputs compatible with the C++ OnnxInference pipeline.

Input:  [batch, time, MOTION_INPUT_DIM]  (22 joints x 3 world positions = 66)
Output: [batch, MOTION_EMBEDDING_DIM]     (128, L2-normalized)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import MOTION_INPUT_DIM, MOTION_EMBEDDING_DIM


class TemporalResBlock(nn.Module):
    """Dilated temporal residual block: Conv1D -> BN -> ReLU -> Conv1D -> BN + skip.

    Uses dilated convolutions to capture multi-scale temporal patterns without
    increasing parameter count. A 1x1 convolution is used for the skip
    connection when in_channels != out_channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dilation: Dilation factor for the convolution kernels.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: two dilated convolutions with a residual connection.

        Args:
            x: Input tensor of shape [batch, channels, time].

        Returns:
            Output tensor of shape [batch, out_channels, time].
        """
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class MotionEncoder(nn.Module):
    """Encodes variable-length joint position sequences to a 128D L2-normalized embedding.

    The encoder uses a stack of dilated temporal residual blocks to build
    a multi-scale receptive field, followed by global average pooling and
    a projection head. The output is L2-normalized for use with distance-based
    similarity metrics.

    Args:
        input_dim: Dimensionality of per-frame input (default: 66 = 22 joints x 3).
        embedding_dim: Dimensionality of the output embedding (default: 128).
        dropout: Dropout probability in the projection head.
    """

    def __init__(
        self,
        input_dim: int = MOTION_INPUT_DIM,
        embedding_dim: int = MOTION_EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Initial projection from input dimension to 128 channels
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

        # Global average pooling over the time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection head: 512 -> 256 -> embedding_dim
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of motion sequences to L2-normalized embeddings.

        Args:
            x: Input tensor of shape [batch, time, input_dim].

        Returns:
            L2-normalized embedding tensor of shape [batch, embedding_dim].
        """
        h = x.transpose(1, 2)              # [B, input_dim, T]
        h = self.input_conv(h)              # [B, 128, T]
        h = self.res_blocks(h)              # [B, 512, T]
        h = self.pool(h).squeeze(2)         # [B, 512]
        h = self.head(h)                    # [B, embedding_dim]
        return F.normalize(h, p=2, dim=-1)  # L2 normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward -- encode a motion sequence to an embedding."""
        return self.forward(x)
