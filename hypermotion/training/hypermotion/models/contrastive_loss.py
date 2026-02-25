"""NT-Xent contrastive loss for style encoder training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss.

    Expects embeddings arranged as pairs: [anchor_0, positive_0, anchor_1, positive_1, ...].
    So a batch of shape [2N, dim] has N positive pairs at indices (0,1), (2,3), etc.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: [2N, dim] L2-normalized."""
        N2 = embeddings.shape[0]
        assert N2 % 2 == 0, "Batch size must be even (pairs of anchor/positive)"

        # Cosine similarity matrix
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Mask out self-similarity
        mask_self = torch.eye(N2, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask_self, float("-inf"))

        # Labels: positive pair for index i is i^1 (XOR flips 0<->1, 2<->3, etc.)
        labels = torch.arange(N2, device=sim.device)
        labels = labels ^ 1

        return F.cross_entropy(sim, labels)


class PairwiseNTXentLoss(nn.Module):
    """Explicit anchor/positive form of NT-Xent.

    Takes two separate tensors of anchors and positives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.loss_fn = NTXentLoss(temperature)

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        """anchors: [N, dim], positives: [N, dim]. Both L2-normalized."""
        # Interleave: [a0, p0, a1, p1, ...]
        N = anchors.shape[0]
        combined = torch.stack([anchors, positives], dim=1).reshape(2 * N, -1)
        return self.loss_fn(combined)
