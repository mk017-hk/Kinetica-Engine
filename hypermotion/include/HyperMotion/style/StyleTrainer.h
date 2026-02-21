#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/style/ContrastiveLoss.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace hm::style {

struct StyleTrainerConfig {
    float learningRate = 1e-4f;
    int numEpochs = 200;
    int batchSize = 32;
    float temperature = 0.07f;

    // Data augmentation
    bool augmentTemporalCrop = true;
    bool augmentSpeedPerturbation = true;
    bool augmentNoise = true;
    float noiseStd = 0.01f;
    float speedPerturbRange = 0.2f;  // +/- 20%

    // Cosine annealing
    float minLearningRate = 1e-6f;

    std::string savePath;
};

struct TrainingPair {
    std::string playerID;
    std::vector<SkeletonFrame> clip;
};

using TrainProgressCallback = std::function<void(int epoch, float loss, float lr)>;

class StyleTrainer {
public:
    explicit StyleTrainer(const StyleTrainerConfig& config = {});
    ~StyleTrainer();

    StyleTrainer(const StyleTrainer&) = delete;
    StyleTrainer& operator=(const StyleTrainer&) = delete;
    StyleTrainer(StyleTrainer&&) noexcept;
    StyleTrainer& operator=(StyleTrainer&&) noexcept;

    // Train the style encoder
    void train(const std::vector<TrainingPair>& data,
               TrainProgressCallback callback = nullptr);

    // Get the trained encoder
    StyleEncoder getEncoder() const;

    // Save/load encoder
    void saveEncoder(const std::string& path);
    void loadEncoder(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::style
