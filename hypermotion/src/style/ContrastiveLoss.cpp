#include "HyperMotion/style/ContrastiveLoss.h"

#ifdef HM_HAS_TORCH

#include <cmath>
#include <limits>

namespace hm::style {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ContrastiveLoss::ContrastiveLoss(float temperature) {
    config_.temperature = temperature;
}

ContrastiveLoss::ContrastiveLoss(const ContrastiveLossConfig& config)
    : config_(config) {}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

torch::Tensor ContrastiveLoss::forward(torch::Tensor embeddings,
                                        torch::Tensor labels) {
    if (config_.trackStatistics) {
        return computeLoss(embeddings, labels, &lastStats_);
    }
    return computeLoss(embeddings, labels, nullptr);
}

torch::Tensor ContrastiveLoss::forwardWithStats(torch::Tensor embeddings,
                                                 torch::Tensor labels,
                                                 ContrastiveBatchStats& stats) {
    return computeLoss(embeddings, labels, &stats);
}

float ContrastiveLoss::runningAvgLoss() const {
    return totalBatchesTracked_ > 0
               ? static_cast<float>(runningLossSum_ / totalBatchesTracked_)
               : 0.0f;
}

float ContrastiveLoss::runningAvgPosSim() const {
    return totalBatchesTracked_ > 0
               ? static_cast<float>(runningPosSimSum_ / totalBatchesTracked_)
               : 0.0f;
}

float ContrastiveLoss::runningAvgNegSim() const {
    return totalBatchesTracked_ > 0
               ? static_cast<float>(runningNegSimSum_ / totalBatchesTracked_)
               : 0.0f;
}

void ContrastiveLoss::resetRunningStats() {
    runningLossSum_ = 0.0;
    runningPosSimSum_ = 0.0;
    runningNegSimSum_ = 0.0;
    totalBatchesTracked_ = 0;
}

// ---------------------------------------------------------------------------
// Hard negative mining
// ---------------------------------------------------------------------------

torch::Tensor ContrastiveLoss::applyHardNegativeMining(
    torch::Tensor sim,
    torch::Tensor positiveMask,
    torch::Tensor selfMask,
    int B) {

    // For each anchor i, find the maximum positive similarity
    // sim values are already scaled by 1/tau
    auto posSimValues = sim.clone();
    // Set non-positive entries to very large negative so max picks only positives
    posSimValues.masked_fill_(positiveMask.logical_not(), -1e9f);
    auto maxPosSim = std::get<0>(posSimValues.max(/*dim=*/1, /*keepdim=*/true));  // [B, 1]

    // Negative mask: not positive, not self
    auto negativeMask = positiveMask.logical_not().logical_and(selfMask.logical_not());

    // Semi-hard negatives: negatives where sim_neg > max_pos_sim - margin/tau
    // These are "challenging but not impossibly hard" negatives
    float scaledMargin = config_.hardNegativeMargin / config_.temperature;
    auto semiHardMask = (sim > (maxPosSim - scaledMargin)).logical_and(negativeMask);

    // If no semi-hard negatives exist for an anchor, fall back to all negatives
    auto hasSemiHard = semiHardMask.any(/*dim=*/1, /*keepdim=*/true);  // [B, 1]
    auto effectiveMask = torch::where(hasSemiHard, semiHardMask, negativeMask);

    // Mask out non-selected negatives with large negative value
    auto maskedSim = sim.clone();
    auto excludeMask = effectiveMask.logical_not().logical_and(negativeMask);
    maskedSim.masked_fill_(excludeMask, -1e9f);

    return maskedSim;
}

// ---------------------------------------------------------------------------
// Core loss computation
// ---------------------------------------------------------------------------

torch::Tensor ContrastiveLoss::computeLoss(torch::Tensor embeddings,
                                            torch::Tensor labels,
                                            ContrastiveBatchStats* stats) {
    // embeddings: [B, D] (expected L2-normalized)
    // labels:     [B] (integer player IDs)
    const int B = static_cast<int>(embeddings.size(0));
    const auto device = embeddings.device();
    const auto dtype = embeddings.dtype();

    if (B < 2) {
        if (stats) {
            *stats = ContrastiveBatchStats{};
        }
        return torch::zeros({1}, embeddings.options());
    }

    // ---------------------------------------------------------------
    // Compute raw cosine similarity (unscaled) for statistics
    // ---------------------------------------------------------------
    auto rawSim = torch::mm(embeddings, embeddings.t());  // [B, B]

    // ---------------------------------------------------------------
    // Build masks
    // ---------------------------------------------------------------
    auto labelCol = labels.unsqueeze(1);  // [B, 1]
    auto labelRow = labels.unsqueeze(0);  // [1, B]
    auto positiveMask = (labelCol == labelRow);  // [B, B]
    auto selfMask = torch::eye(B, torch::TensorOptions().dtype(torch::kBool).device(device));
    positiveMask = positiveMask.logical_and(selfMask.logical_not());

    auto negativeMask = positiveMask.logical_not().logical_and(selfMask.logical_not());

    // ---------------------------------------------------------------
    // Gather statistics before scaling (on raw cosine similarities)
    // ---------------------------------------------------------------
    if (stats) {
        torch::NoGradGuard noGrad;

        // Embedding norm statistics
        auto norms = embeddings.norm(2, /*dim=*/1);  // [B]
        stats->embeddingNormMean = norms.mean().item<float>();
        stats->embeddingNormStd = norms.std().item<float>();

        // Positive pair statistics
        auto posMaskFloat = positiveMask.to(dtype);
        auto numPosTotal = posMaskFloat.sum().item<int>();
        stats->numPositivePairs = numPosTotal;
        stats->numAnchorsWithPositives =
            static_cast<int>((posMaskFloat.sum(1) > 0).sum().item<int>());

        if (numPosTotal > 0) {
            auto posValues = rawSim.masked_select(positiveMask);
            stats->meanPositiveSim = posValues.mean().item<float>();
            stats->hardestPositiveSim = posValues.min().item<float>();
        } else {
            stats->meanPositiveSim = 0.0f;
            stats->hardestPositiveSim = 0.0f;
        }

        // Negative pair statistics
        auto negMaskFloat = negativeMask.to(dtype);
        auto numNegTotal = negMaskFloat.sum().item<int>();
        stats->numNegativePairs = numNegTotal;

        if (numNegTotal > 0) {
            auto negValues = rawSim.masked_select(negativeMask);
            stats->meanNegativeSim = negValues.mean().item<float>();
            stats->hardestNegativeSim = negValues.max().item<float>();
        } else {
            stats->meanNegativeSim = 0.0f;
            stats->hardestNegativeSim = 0.0f;
        }
    }

    // ---------------------------------------------------------------
    // Scale by temperature
    // ---------------------------------------------------------------
    auto sim = rawSim / config_.temperature;  // [B, B]

    // ---------------------------------------------------------------
    // Mask self-similarity with large negative (will vanish in softmax)
    // ---------------------------------------------------------------
    sim = sim - selfMask.to(dtype) * 1e9f;

    // ---------------------------------------------------------------
    // Optional hard negative mining: mask out easy negatives
    // ---------------------------------------------------------------
    if (config_.useHardNegativeMining) {
        sim = applyHardNegativeMining(sim, positiveMask, selfMask, B);
    }

    // ---------------------------------------------------------------
    // Log-sum-exp trick for numerical stability
    // For each anchor i: logsumexp_j(sim[i,j]) where j != i
    //
    // logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    // This avoids overflow when sim values are large.
    // ---------------------------------------------------------------

    // We want logsumexp over all j != i. The self-similarity is already
    // masked to -1e9 so it contributes ~0 to the sum, but for numerical
    // stability we compute logsumexp properly:
    auto simForLSE = sim.clone();
    // The self entries are already -1e9 which is fine; they contribute nothing.

    // Per-row max for numerical stability (excluding self via the -1e9 mask)
    auto rowMax = std::get<0>(simForLSE.max(/*dim=*/1, /*keepdim=*/true));  // [B, 1]
    auto shiftedSim = simForLSE - rowMax;  // shift for stability
    auto logSumExp = rowMax.squeeze(1) + torch::log(shiftedSim.exp().sum(/*dim=*/1));  // [B]

    // ---------------------------------------------------------------
    // For each anchor, average the scaled similarity over positive pairs
    // ---------------------------------------------------------------
    auto posMaskFloat = positiveMask.to(dtype);
    auto positiveSim = sim * posMaskFloat;  // zero out non-positives
    auto numPositivesPerAnchor = posMaskFloat.sum(1);  // [B]

    // Avoid division by zero for anchors with no positives
    auto validMask = (numPositivesPerAnchor > 0);
    if (!validMask.any().item<bool>()) {
        if (stats) {
            stats->loss = 0.0f;
        }
        return torch::zeros({1}, embeddings.options());
    }

    auto posSum = positiveSim.sum(1);  // [B]

    // ---------------------------------------------------------------
    // NT-Xent loss: loss_i = -mean_pos(sim[i,j]) + logsumexp(sim[i,:])
    // Average over anchors that have at least one positive
    // ---------------------------------------------------------------
    auto perAnchorLoss = -posSum / numPositivesPerAnchor.clamp_min(1.0f) + logSumExp;

    // ---------------------------------------------------------------
    // Optional label smoothing
    // Softly add a small uniform target to all classes
    // ---------------------------------------------------------------
    if (config_.labelSmoothingEps > 0.0f) {
        // Uniform entropy term: -log(1/(B-1)) = log(B-1)
        float uniformEntropy = std::log(static_cast<float>(B - 1));
        perAnchorLoss = perAnchorLoss * (1.0f - config_.labelSmoothingEps)
                        + config_.labelSmoothingEps * uniformEntropy;
    }

    // ---------------------------------------------------------------
    // Loss clamping for stability
    // ---------------------------------------------------------------
    perAnchorLoss = perAnchorLoss.clamp(config_.lossClampMin, config_.lossClampMax);

    // Average over valid anchors only
    auto validMaskFloat = validMask.to(dtype);
    auto loss = (perAnchorLoss * validMaskFloat).sum() / validMaskFloat.sum();

    // ---------------------------------------------------------------
    // Record statistics
    // ---------------------------------------------------------------
    if (stats) {
        stats->loss = loss.item<float>();

        // Update running accumulators
        runningLossSum_ += static_cast<double>(stats->loss);
        runningPosSimSum_ += static_cast<double>(stats->meanPositiveSim);
        runningNegSimSum_ += static_cast<double>(stats->meanNegativeSim);
        totalBatchesTracked_++;
    } else if (config_.trackStatistics) {
        // Even without explicit stats output, update running accumulators
        lastStats_.loss = loss.item<float>();
        runningLossSum_ += static_cast<double>(lastStats_.loss);
        totalBatchesTracked_++;
    }

    return loss;
}

} // namespace hm::style

#endif
