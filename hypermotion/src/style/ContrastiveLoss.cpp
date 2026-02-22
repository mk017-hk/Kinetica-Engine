#include "HyperMotion/style/ContrastiveLoss.h"

#ifdef HM_HAS_TORCH

namespace hm::style {

ContrastiveLoss::ContrastiveLoss(float temperature)
    : temperature_(temperature) {}

torch::Tensor ContrastiveLoss::forward(torch::Tensor embeddings,
                                        torch::Tensor labels) {
    // embeddings: [B, D] (L2-normalized)
    // labels:     [B] (integer player IDs)
    int B = embeddings.size(0);
    if (B < 2) {
        return torch::zeros({1}, embeddings.options());
    }

    // Cosine similarity matrix: [B, B] (already L2-normalized)
    auto sim = torch::mm(embeddings, embeddings.t()) / temperature_;

    // Positive mask: (i,j) where labels[i] == labels[j] and i != j
    auto labelCol = labels.unsqueeze(1);  // [B, 1]
    auto labelRow = labels.unsqueeze(0);  // [1, B]
    auto positiveMask = (labelCol == labelRow);
    auto selfMask = torch::eye(B, embeddings.options().dtype(torch::kBool));
    positiveMask = positiveMask.logical_and(selfMask.logical_not());

    // Mask self-similarity with large negative
    sim = sim - selfMask.to(embeddings.dtype()) * 1e9;

    // For each anchor i, compute log_sum_exp over all j != i
    auto logSumExp = torch::logsumexp(sim, /*dim=*/1);  // [B]

    // For each anchor, average over positive pairs
    // Numerator: sum of sim[i,j] for positive j
    auto positiveSim = (sim * positiveMask.to(embeddings.dtype()));

    // Count positives per anchor
    auto numPositives = positiveMask.sum(1).to(embeddings.dtype());  // [B]

    // Avoid division by zero for anchors with no positives
    auto validMask = (numPositives > 0);
    if (!validMask.any().item<bool>()) {
        return torch::zeros({1}, embeddings.options());
    }

    auto posSum = positiveSim.sum(1);  // [B]

    // loss_i = -mean_j(sim[i,j]) + log_sum_exp, averaged over valid anchors
    auto loss = (-posSum / numPositives.clamp_min(1.0) + logSumExp);
    loss = (loss * validMask.to(embeddings.dtype())).sum() /
           validMask.to(embeddings.dtype()).sum();

    return loss;
}

} // namespace hm::style

#endif
