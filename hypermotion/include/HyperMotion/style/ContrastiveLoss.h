#pragma once

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

namespace hm::style {

/// NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
///
/// Positive pairs: same player, Negative pairs: different players.
/// tau = 0.07 (temperature).
class ContrastiveLoss {
public:
    explicit ContrastiveLoss(float temperature = 0.07f);

    /// Compute loss given L2-normalized embeddings and integer player labels.
    /// @param embeddings [B, D] — must be L2-normalized
    /// @param labels     [B]    — integer player IDs (same ID = positive pair)
    torch::Tensor forward(torch::Tensor embeddings, torch::Tensor labels);

private:
    float temperature_;
};

} // namespace hm::style

#else

namespace hm::style {
// ContrastiveLoss requires LibTorch for training.
} // namespace hm::style

#endif
