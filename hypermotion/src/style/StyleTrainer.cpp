#include "HyperMotion/style/StyleTrainer.h"
#include "HyperMotion/core/Logger.h"

#include <random>
#include <algorithm>
#include <cmath>

namespace hm::style {

static constexpr const char* TAG = "StyleTrainer";
static constexpr float PI = 3.14159265358979323846f;

struct StyleTrainer::Impl {
    StyleTrainerConfig config;
    StyleEncoder encoder;
    ContrastiveLoss loss;
    bool trained = false;

    Impl(const StyleTrainerConfig& cfg)
        : config(cfg), loss(cfg.temperature) {}

    // Data augmentation: temporal crop
    std::vector<SkeletonFrame> temporalCrop(const std::vector<SkeletonFrame>& clip,
                                             std::mt19937& rng) {
        if (clip.size() <= 30) return clip;
        int minLen = 30;
        int maxLen = static_cast<int>(clip.size());
        std::uniform_int_distribution<int> lenDist(minLen, maxLen);
        int cropLen = lenDist(rng);
        std::uniform_int_distribution<int> startDist(0, static_cast<int>(clip.size()) - cropLen);
        int start = startDist(rng);
        return std::vector<SkeletonFrame>(clip.begin() + start, clip.begin() + start + cropLen);
    }

    // Data augmentation: speed perturbation
    std::vector<SkeletonFrame> speedPerturbation(const std::vector<SkeletonFrame>& clip,
                                                   std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(
            1.0f - config.speedPerturbRange, 1.0f + config.speedPerturbRange);
        float speedFactor = dist(rng);

        int newLen = static_cast<int>(clip.size() * speedFactor);
        if (newLen < 10) newLen = 10;

        std::vector<SkeletonFrame> result(newLen);
        for (int i = 0; i < newLen; ++i) {
            float srcIdx = i / speedFactor;
            int idx0 = std::min(static_cast<int>(srcIdx), static_cast<int>(clip.size()) - 1);
            int idx1 = std::min(idx0 + 1, static_cast<int>(clip.size()) - 1);
            float frac = srcIdx - idx0;

            // Simple linear interpolation of positions
            result[i] = clip[idx0];
            result[i].frameIndex = i;
            for (int j = 0; j < JOINT_COUNT; ++j) {
                auto& p0 = clip[idx0].joints[j].worldPosition;
                auto& p1 = clip[idx1].joints[j].worldPosition;
                result[i].joints[j].worldPosition = {
                    p0.x + (p1.x - p0.x) * frac,
                    p0.y + (p1.y - p0.y) * frac,
                    p0.z + (p1.z - p0.z) * frac
                };
            }
        }
        return result;
    }

    // Data augmentation: add noise
    torch::Tensor addNoise(const torch::Tensor& input, std::mt19937& rng) {
        if (!config.augmentNoise) return input;
        auto noise = torch::randn_like(input) * config.noiseStd;
        return input + noise;
    }
};

StyleTrainer::StyleTrainer(const StyleTrainerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

StyleTrainer::~StyleTrainer() = default;
StyleTrainer::StyleTrainer(StyleTrainer&&) noexcept = default;
StyleTrainer& StyleTrainer::operator=(StyleTrainer&&) noexcept = default;

void StyleTrainer::train(const std::vector<TrainingPair>& data,
                          TrainProgressCallback callback) {
    if (data.size() < 2) {
        HM_LOG_ERROR(TAG, "Need at least 2 training pairs");
        return;
    }

    impl_->encoder = StyleEncoder();

    // Group clips by player
    std::map<std::string, std::vector<int>> playerClips;
    for (size_t i = 0; i < data.size(); ++i) {
        playerClips[data[i].playerID].push_back(static_cast<int>(i));
    }

    // Need at least 2 players for contrastive learning
    std::vector<std::string> playerIDs;
    for (const auto& [id, clips] : playerClips) {
        if (clips.size() >= 2) {
            playerIDs.push_back(id);
        }
    }

    if (playerIDs.size() < 2) {
        HM_LOG_ERROR(TAG, "Need at least 2 players with 2+ clips each for contrastive training");
        return;
    }

    // Setup optimizer with cosine annealing
    auto optimizer = torch::optim::Adam(impl_->encoder->parameters(),
        torch::optim::AdamOptions(impl_->config.learningRate));

    std::mt19937 rng(42);

    HM_LOG_INFO(TAG, "Starting training: " + std::to_string(impl_->config.numEpochs) +
                " epochs, " + std::to_string(playerIDs.size()) + " players");

    for (int epoch = 0; epoch < impl_->config.numEpochs; ++epoch) {
        impl_->encoder->train();
        float epochLoss = 0.0f;
        int numBatches = 0;

        // Cosine annealing learning rate
        float lr = impl_->config.minLearningRate +
            0.5f * (impl_->config.learningRate - impl_->config.minLearningRate) *
            (1.0f + std::cos(PI * epoch / impl_->config.numEpochs));

        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
        }

        // Sample batch of pairs
        int batchPairs = impl_->config.batchSize;
        std::vector<torch::Tensor> anchorInputs, positiveInputs;

        for (int p = 0; p < batchPairs; ++p) {
            // Pick random player
            std::uniform_int_distribution<int> playerDist(0, static_cast<int>(playerIDs.size()) - 1);
            const std::string& pid = playerIDs[playerDist(rng)];
            const auto& clips = playerClips[pid];

            // Pick two different clips from same player
            std::uniform_int_distribution<int> clipDist(0, static_cast<int>(clips.size()) - 1);
            int clipIdx1 = clipDist(rng);
            int clipIdx2 = clipDist(rng);
            while (clipIdx2 == clipIdx1 && clips.size() > 1) {
                clipIdx2 = clipDist(rng);
            }

            auto clip1 = data[clips[clipIdx1]].clip;
            auto clip2 = data[clips[clipIdx2]].clip;

            // Augmentation
            if (impl_->config.augmentTemporalCrop) {
                clip1 = impl_->temporalCrop(clip1, rng);
                clip2 = impl_->temporalCrop(clip2, rng);
            }
            if (impl_->config.augmentSpeedPerturbation) {
                clip1 = impl_->speedPerturbation(clip1, rng);
                clip2 = impl_->speedPerturbation(clip2, rng);
            }

            auto input1 = StyleEncoderImpl::prepareInput(clip1);
            auto input2 = StyleEncoderImpl::prepareInput(clip2);

            if (impl_->config.augmentNoise) {
                input1 = impl_->addNoise(input1, rng);
                input2 = impl_->addNoise(input2, rng);
            }

            anchorInputs.push_back(input1.squeeze(0));
            positiveInputs.push_back(input2.squeeze(0));
        }

        // Pad to same length and stack
        int maxLen = 0;
        for (const auto& t : anchorInputs) maxLen = std::max(maxLen, static_cast<int>(t.size(0)));
        for (const auto& t : positiveInputs) maxLen = std::max(maxLen, static_cast<int>(t.size(0)));

        auto padAndStack = [&](const std::vector<torch::Tensor>& inputs) {
            std::vector<torch::Tensor> padded;
            for (const auto& t : inputs) {
                if (t.size(0) < maxLen) {
                    auto pad = torch::zeros({maxLen - t.size(0), STYLE_INPUT_DIM});
                    padded.push_back(torch::cat({t, pad}, 0));
                } else {
                    padded.push_back(t.slice(0, 0, maxLen));
                }
            }
            return torch::stack(padded);
        };

        auto anchorBatch = padAndStack(anchorInputs);
        auto positiveBatch = padAndStack(positiveInputs);

        // Forward
        auto anchorEmb = impl_->encoder->forward(anchorBatch);
        auto positiveEmb = impl_->encoder->forward(positiveBatch);

        // Contrastive loss
        auto loss = impl_->loss.computePairwise(anchorEmb, positiveEmb);

        // Backward
        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(impl_->encoder->parameters(), 1.0);
        optimizer.step();

        epochLoss += loss.item<float>();
        numBatches++;

        float avgLoss = epochLoss / numBatches;

        if (callback) {
            callback(epoch, avgLoss, lr);
        }

        if ((epoch + 1) % 10 == 0) {
            HM_LOG_INFO(TAG, "Epoch " + std::to_string(epoch + 1) + "/" +
                        std::to_string(impl_->config.numEpochs) +
                        " loss=" + std::to_string(avgLoss) +
                        " lr=" + std::to_string(lr));
        }
    }

    impl_->trained = true;

    if (!impl_->config.savePath.empty()) {
        saveEncoder(impl_->config.savePath);
    }

    HM_LOG_INFO(TAG, "Training complete");
}

StyleEncoder StyleTrainer::getEncoder() const {
    return impl_->encoder;
}

void StyleTrainer::saveEncoder(const std::string& path) {
    torch::serialize::OutputArchive archive;
    impl_->encoder->save(archive);
    archive.save_to(path);
    HM_LOG_INFO(TAG, "Encoder saved to: " + path);
}

void StyleTrainer::loadEncoder(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        impl_->encoder->load(archive);
        HM_LOG_INFO(TAG, "Encoder loaded from: " + path);
    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Failed to load encoder: ") + e.what());
    }
}

} // namespace hm::style
