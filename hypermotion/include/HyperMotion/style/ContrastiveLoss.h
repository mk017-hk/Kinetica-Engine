#pragma once

#ifdef HM_HAS_TORCH
#include <torch/torch.h>
#include <vector>
#include <string>

namespace hm::style {

/// Batch-level statistics tracked during contrastive training.
struct ContrastiveBatchStats {
    float loss = 0.0f;
    float meanPositiveSim = 0.0f;       ///< Average cosine similarity of positive pairs.
    float meanNegativeSim = 0.0f;       ///< Average cosine similarity of negative pairs.
    float hardestNegativeSim = 0.0f;    ///< Highest cosine similarity among negative pairs.
    float hardestPositiveSim = 0.0f;    ///< Lowest cosine similarity among positive pairs.
    int numPositivePairs = 0;
    int numNegativePairs = 0;
    int numAnchorsWithPositives = 0;    ///< Anchors that have at least one positive in the batch.
    float embeddingNormMean = 0.0f;     ///< Mean L2 norm of embeddings (should be ~1).
    float embeddingNormStd = 0.0f;      ///< Std of L2 norms (should be ~0 for normalized).
};

/// Configuration for NT-Xent contrastive loss.
struct ContrastiveLossConfig {
    float temperature = 0.07f;          ///< Temperature scaling (tau).
    bool useHardNegativeMining = false;  ///< Enable semi-hard / hard negative mining.
    float hardNegativeMargin = 0.1f;    ///< Margin for semi-hard negatives: sim_neg > sim_pos - margin.
    float lossClampMin = 0.0f;          ///< Minimum loss value (clamped).
    float lossClampMax = 30.0f;         ///< Maximum loss value (clamped).
    bool trackStatistics = true;        ///< Whether to compute detailed batch statistics.
    float labelSmoothingEps = 0.0f;     ///< Label smoothing epsilon (0 = disabled).
};

/// NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
///
/// Positive pairs: same player, Negative pairs: different players.
/// Implements the log-sum-exp trick for numerical stability, optional hard
/// negative mining, loss clamping, and detailed batch-level statistics.
class ContrastiveLoss {
public:
    explicit ContrastiveLoss(float temperature = 0.07f);
    explicit ContrastiveLoss(const ContrastiveLossConfig& config);
    ~ContrastiveLoss() = default;

    /// Compute loss given L2-normalized embeddings and integer player labels.
    /// @param embeddings [B, D] -- must be L2-normalized
    /// @param labels     [B]    -- integer player IDs (same ID = positive pair)
    /// @return scalar loss tensor
    torch::Tensor forward(torch::Tensor embeddings, torch::Tensor labels);

    /// Compute loss and return detailed batch statistics.
    /// @param embeddings [B, D] -- must be L2-normalized
    /// @param labels     [B]    -- integer player IDs
    /// @param[out] stats  populated with batch-level metrics
    /// @return scalar loss tensor
    torch::Tensor forwardWithStats(torch::Tensor embeddings,
                                   torch::Tensor labels,
                                   ContrastiveBatchStats& stats);

    /// Get the most recent batch statistics (only valid if trackStatistics is enabled).
    const ContrastiveBatchStats& lastBatchStats() const { return lastStats_; }

    /// Update configuration.
    void setConfig(const ContrastiveLossConfig& config) { config_ = config; }
    const ContrastiveLossConfig& config() const { return config_; }

    /// Get running averages accumulated over multiple forward calls.
    /// Call resetRunningStats() to clear.
    float runningAvgLoss() const;
    float runningAvgPosSim() const;
    float runningAvgNegSim() const;
    int totalBatchesTracked() const { return totalBatchesTracked_; }
    void resetRunningStats();

private:
    ContrastiveLossConfig config_;
    ContrastiveBatchStats lastStats_;

    // Running statistics accumulators
    double runningLossSum_ = 0.0;
    double runningPosSimSum_ = 0.0;
    double runningNegSimSum_ = 0.0;
    int totalBatchesTracked_ = 0;

    /// Core loss computation with optional statistics gathering.
    torch::Tensor computeLoss(torch::Tensor embeddings,
                              torch::Tensor labels,
                              ContrastiveBatchStats* stats);

    /// Apply hard negative mining mask to the similarity matrix.
    torch::Tensor applyHardNegativeMining(torch::Tensor sim,
                                          torch::Tensor positiveMask,
                                          torch::Tensor selfMask,
                                          int B);
};

} // namespace hm::style

#else

namespace hm::style {

struct ContrastiveBatchStats {
    float loss = 0.0f;
    float meanPositiveSim = 0.0f;
    float meanNegativeSim = 0.0f;
    float hardestNegativeSim = 0.0f;
    float hardestPositiveSim = 0.0f;
    int numPositivePairs = 0;
    int numNegativePairs = 0;
    int numAnchorsWithPositives = 0;
    float embeddingNormMean = 0.0f;
    float embeddingNormStd = 0.0f;
};

struct ContrastiveLossConfig {
    float temperature = 0.07f;
    bool useHardNegativeMining = false;
    float hardNegativeMargin = 0.1f;
    float lossClampMin = 0.0f;
    float lossClampMax = 30.0f;
    bool trackStatistics = true;
    float labelSmoothingEps = 0.0f;
};

// ContrastiveLoss requires LibTorch for training.
} // namespace hm::style

#endif
