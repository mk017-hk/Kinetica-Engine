#pragma once

#include <torch/torch.h>

namespace hm::style {

// NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss
// Used for contrastive learning of style embeddings
class ContrastiveLoss {
public:
    explicit ContrastiveLoss(float temperature = 0.07f);
    ~ContrastiveLoss();

    // Compute NT-Xent loss
    // embeddings: [2*N, dim] where pairs (2i, 2i+1) are positive pairs
    // Returns scalar loss
    torch::Tensor compute(const torch::Tensor& embeddings);

    // Alternative: explicit positive/negative
    // anchor: [N, dim], positive: [N, dim]
    torch::Tensor computePairwise(const torch::Tensor& anchor,
                                   const torch::Tensor& positive);

    float temperature() const { return temperature_; }

private:
    float temperature_;
};

} // namespace hm::style
