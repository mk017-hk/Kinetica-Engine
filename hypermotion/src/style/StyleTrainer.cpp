#include "HyperMotion/style/StyleTrainer.h"

#ifdef HM_HAS_TORCH

#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/style/ContrastiveLoss.h"
#include "HyperMotion/core/Logger.h"

#include <torch/torch.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <map>

namespace hm::style {

static constexpr const char* TAG = "StyleTrainer";

struct StyleTrainer::Impl {
    StyleTrainerConfig config;
    StyleEncoder encoder_{nullptr};
    ContrastiveLoss loss_;
    std::mt19937 rng_{std::random_device{}()};

    Impl(const StyleTrainerConfig& cfg)
        : config(cfg)
        , encoder_(STYLE_INPUT_DIM, STYLE_DIM)
        , loss_(0.07f) {}
};

StyleTrainer::StyleTrainer(const StyleTrainerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

StyleTrainer::~StyleTrainer() = default;

void StyleTrainer::train(const std::vector<TrainingPair>& data,
                          ProgressCallback callback) {
    auto& encoder = impl_->encoder_;
    auto& loss = impl_->loss_;
    auto& config = impl_->config;

    if (data.size() < 2) {
        HM_LOG_ERROR(TAG, "Need at least 2 training pairs");
        return;
    }

    // Assign integer IDs to player names
    std::map<std::string, int> playerIDMap;
    int nextID = 0;
    for (const auto& pair : data) {
        if (playerIDMap.find(pair.playerID) == playerIDMap.end()) {
            playerIDMap[pair.playerID] = nextID++;
        }
    }

    HM_LOG_INFO(TAG, "Training style encoder: " +
                std::to_string(data.size()) + " clips, " +
                std::to_string(playerIDMap.size()) + " players");

    // Optimizer with cosine annealing
    auto optimizer = torch::optim::Adam(
        encoder->parameters(),
        torch::optim::AdamOptions(config.learningRate));

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < config.numEpochs; ++epoch) {
        encoder->train();
        float epochLoss = 0.0f;
        int numBatches = 0;

        std::shuffle(indices.begin(), indices.end(), impl_->rng_);

        for (size_t i = 0; i + config.batchSize <= indices.size();
             i += config.batchSize) {
            // Collect batch
            std::vector<torch::Tensor> batchInputs;
            std::vector<int64_t> batchLabels;

            for (int b = 0; b < config.batchSize; ++b) {
                const auto& pair = data[indices[i + b]];
                auto input = StyleEncoderImpl::prepareInput(pair.clip);  // [1, 201, T]
                batchInputs.push_back(input.squeeze(0));  // [201, T]
                batchLabels.push_back(playerIDMap[pair.playerID]);
            }

            // Pad to same length and stack
            int maxLen = 0;
            for (const auto& t : batchInputs) {
                maxLen = std::max(maxLen, static_cast<int>(t.size(1)));
            }

            auto batchTensor = torch::zeros({config.batchSize, STYLE_INPUT_DIM, maxLen});
            for (int b = 0; b < config.batchSize; ++b) {
                int len = batchInputs[b].size(1);
                batchTensor.index_put_(
                    {b, torch::indexing::Slice(), torch::indexing::Slice(0, len)},
                    batchInputs[b]);
            }

            auto labels = torch::tensor(batchLabels, torch::kLong);

            // Forward
            optimizer.zero_grad();
            auto embeddings = encoder->forward(batchTensor);  // [B, 64]
            auto batchLoss = loss.forward(embeddings, labels);

            // Backward
            batchLoss.backward();
            torch::nn::utils::clip_grad_norm_(encoder->parameters(), 1.0);
            optimizer.step();

            epochLoss += batchLoss.item<float>();
            numBatches++;
        }

        float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0.0f;

        // Cosine annealing LR
        float progress = static_cast<float>(epoch) / config.numEpochs;
        float lr = config.learningRate * 0.5f *
                   (1.0f + std::cos(progress * 3.14159265f));
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
        }

        if (callback) {
            callback(epoch, avgLoss, lr);
        }
    }

    // Save trained encoder
    if (!config.savePath.empty()) {
        torch::serialize::OutputArchive archive;
        encoder->save(archive);
        archive.save_to(config.savePath);
        HM_LOG_INFO(TAG, "Encoder saved to: " + config.savePath);
    }
}

} // namespace hm::style

#endif
