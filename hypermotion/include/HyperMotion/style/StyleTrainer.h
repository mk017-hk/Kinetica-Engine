#pragma once

#include "HyperMotion/core/Types.h"

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

struct StyleTrainerConfig {
    int numEpochs = 200;
    int batchSize = 32;
    float learningRate = 1e-4f;
    std::string savePath = "style_encoder.pt";
};

#ifdef HM_HAS_TORCH

/// Trains a StyleEncoder with NT-Xent contrastive loss.
///
/// Data is organized as per-player clips.  Each training batch samples
/// pairs of clips from the same and different players.
class StyleTrainer {
public:
    explicit StyleTrainer(const StyleTrainerConfig& config);
    ~StyleTrainer();

    StyleTrainer(const StyleTrainer&) = delete;
    StyleTrainer& operator=(const StyleTrainer&) = delete;

    using ProgressCallback = std::function<void(int epoch, float loss, float lr)>;

    /// Run the full training loop.
    void train(const std::vector<TrainingPair>& data,
               ProgressCallback callback = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#else

class StyleTrainer {
public:
    explicit StyleTrainer(const StyleTrainerConfig&) {}
    using ProgressCallback = std::function<void(int epoch, float loss, float lr)>;
    void train(const std::vector<TrainingPair>&, ProgressCallback = nullptr) {}
};

#endif

} // namespace hm::style
