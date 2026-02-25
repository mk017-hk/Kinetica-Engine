"""Condition encoder: maps 78D motion condition to 256D latent."""

import torch
import torch.nn as nn

from .constants import CONDITION_DIM


class ConditionEncoder(nn.Module):
    """Linear(78->512) -> ReLU -> Linear(512->512) -> ReLU -> Linear(512->256)."""

    def __init__(self, input_dim: int = CONDITION_DIM, hidden_dim: int = 512,
                 output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, CONDITION_DIM] -> [batch, 256]."""
        return self.net(x)
