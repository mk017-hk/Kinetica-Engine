#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/style/ContrastiveLoss.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hm::style {

/// A clip from a single player, used for contrastive training.
struct TrainingPair {
    std::string playerID;
    std::vector<SkeletonFrame> clip;
};

/// Data augmentation configuration for style training.
struct AugmentationConfig {
    bool enabled = true;

    // Temporal crop: randomly crop a sub-sequence
    bool temporalCrop = true;
    int minCropFrames = 30;         ///< Minimum frames after crop.
    int maxCropFrames = 0;          ///< 0 = use full length.

    // Speed perturbation: resample at slightly different rate
    bool speedPerturbation = true;
    float speedMin = 0.85f;         ///< Minimum speed factor (15% slower).
    float speedMax = 1.15f;         ///< Maximum speed factor (15% faster).

    // Gaussian noise injection
    bool gaussianNoise = true;
    float noiseStddev = 0.01f;      ///< Standard deviation of additive Gaussian noise (sigma).

    // Random horizontal mirroring (swap left/right joints)
    bool horizontalMirror = false;
    float mirrorProbability = 0.5f;

    // Random rotation around vertical axis
    bool randomYRotation = false;
    float maxYRotationDeg = 15.0f;
};

/// Early stopping configuration.
struct EarlyStoppingConfig {
    bool enabled = true;
    int patience = 20;              ///< Number of epochs without improvement before stopping.
    float minDelta = 1e-4f;         ///< Minimum improvement to be considered progress.
    bool restoreBestWeights = true; ///< Restore weights from best epoch on stop.
};

/// Checkpoint configuration.
struct CheckpointConfig {
    bool enabled = true;
    std::string directory = "checkpoints/";  ///< Directory for checkpoint files.
    int saveEveryNEpochs = 10;               ///< Save checkpoint every N epochs.
    bool saveBestOnly = false;               ///< Only save when validation loss improves.
    int maxCheckpointsToKeep = 5;            ///< Maximum number of checkpoint files to retain.
};

/// Training metrics for a single epoch.
struct EpochMetrics {
    int epoch = 0;
    float trainLoss = 0.0f;
    float valLoss = 0.0f;
    float learningRate = 0.0f;
    float meanPositiveSim = 0.0f;
    float meanNegativeSim = 0.0f;
    float hardestNegativeSim = 0.0f;
    float trainTimeSeconds = 0.0f;
    int numTrainBatches = 0;
    int numValBatches = 0;
};

/// Complete training history containing all epoch metrics.
struct TrainingHistory {
    std::vector<EpochMetrics> epochs;
    int bestEpoch = -1;
    float bestValLoss = std::numeric_limits<float>::max();
    float totalTrainingTimeSeconds = 0.0f;

    /// Save training history to a JSON file.
    bool saveJSON(const std::string& path) const;

    /// Save loss curves as a simple CSV for external plotting.
    bool saveCSV(const std::string& path) const;
};

/// Full trainer configuration.
struct StyleTrainerConfig {
    int numEpochs = 200;
    int batchSize = 32;
    float learningRate = 1e-4f;
    float weightDecay = 1e-5f;          ///< L2 regularization weight decay.
    float gradientClipNorm = 1.0f;      ///< Maximum gradient norm for clipping.
    std::string savePath = "style_encoder.pt";

    // Validation split
    float validationSplit = 0.15f;      ///< Fraction of data used for validation.
    int validationSeed = 42;            ///< Seed for deterministic val split.

    // Learning rate schedule
    bool useCosineAnnealing = true;
    float cosineEtaMin = 1e-6f;         ///< Minimum LR for cosine annealing.
    int warmupEpochs = 5;               ///< Linear warmup epochs before cosine decay.

    // Sub-configs
    AugmentationConfig augmentation;
    EarlyStoppingConfig earlyStopping;
    CheckpointConfig checkpoint;
    ContrastiveLossConfig lossConfig;

    // Device selection
    bool useGPU = true;
    int gpuDeviceIndex = 0;

    // Logging
    int logEveryNBatches = 10;          ///< Log batch loss every N batches.
    bool verboseLogging = false;
};

#ifdef HM_HAS_TORCH

/// Trains a StyleEncoder with NT-Xent contrastive loss.
///
/// Features:
/// - Adam optimizer with weight decay and cosine annealing + warmup
/// - Data augmentation (temporal crop, speed perturbation, Gaussian noise)
/// - Training / validation split with configurable ratio
/// - Early stopping with patience and best-weight restoration
/// - Periodic checkpointing with configurable retention
/// - Detailed per-epoch metrics and training history
/// - Batch-level contrastive loss statistics
class StyleTrainer {
public:
    explicit StyleTrainer(const StyleTrainerConfig& config);
    ~StyleTrainer();

    StyleTrainer(const StyleTrainer&) = delete;
    StyleTrainer& operator=(const StyleTrainer&) = delete;

    using ProgressCallback = std::function<void(int epoch, float loss, float lr)>;

    /// Callback with full epoch metrics.
    using DetailedCallback = std::function<void(const EpochMetrics& metrics)>;

    /// Run the full training loop.
    /// @param data  vector of training pairs (player clips)
    /// @param callback  simple progress callback (epoch, loss, lr)
    void train(const std::vector<TrainingPair>& data,
               ProgressCallback callback = nullptr);

    /// Run the full training loop with detailed per-epoch metrics callback.
    /// @param data  vector of training pairs
    /// @param callback  detailed callback with full epoch metrics
    void trainDetailed(const std::vector<TrainingPair>& data,
                       DetailedCallback callback = nullptr);

    /// Get the complete training history after training completes.
    const TrainingHistory& history() const;

    /// Resume training from a checkpoint.
    /// @param checkpointPath  path to saved checkpoint (.pt file)
    /// @param data  training data
    /// @param callback  optional progress callback
    void resumeTraining(const std::string& checkpointPath,
                        const std::vector<TrainingPair>& data,
                        ProgressCallback callback = nullptr);

    /// Export the trained encoder to ONNX format for deployment.
    /// @param onnxPath  output ONNX file path
    /// @param exampleSequenceLength  example temporal length for tracing
    /// @return true on success
    bool exportONNX(const std::string& onnxPath, int exampleSequenceLength = 120) const;

    /// Evaluate the encoder on a held-out dataset.
    /// Returns average loss over all batches.
    float evaluate(const std::vector<TrainingPair>& data) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#else

class StyleTrainer {
public:
    explicit StyleTrainer(const StyleTrainerConfig&) {}
    using ProgressCallback = std::function<void(int epoch, float loss, float lr)>;
    using DetailedCallback = std::function<void(const EpochMetrics& metrics)>;
    void train(const std::vector<TrainingPair>&, ProgressCallback = nullptr) {}
    void trainDetailed(const std::vector<TrainingPair>&, DetailedCallback = nullptr) {}
    const TrainingHistory& history() const { static TrainingHistory h; return h; }
    void resumeTraining(const std::string&, const std::vector<TrainingPair>&, ProgressCallback = nullptr) {}
    bool exportONNX(const std::string&, int = 120) const { return false; }
    float evaluate(const std::vector<TrainingPair>&) const { return 0.0f; }
};

#endif

} // namespace hm::style
