#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/MathUtils.h"

#include <iostream>
#include <string>

#ifndef HM_HAS_TORCH

int main(int, char*[]) {
    std::cerr << "hm_train requires LibTorch. Rebuild with -DTorch_DIR=<path-to-libtorch>\n";
    return 1;
}

#else  // HM_HAS_TORCH

#include "HyperMotion/ml/MotionDiffusionModel.h"
#include "HyperMotion/segmenter/TemporalConvNet.h"
#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include "HyperMotion/export/JSONExporter.h"

#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

struct TrainArgs {
    std::string mode;       // "diffusion" or "classifier"
    std::string dataDir;
    int epochs = 500;
    int batchSize = 64;
    float learningRate = 1e-4f;
    std::string outputPath = "model.pt";
    int saveEvery = 50;
};

static void printUsage() {
    std::cout << "Usage: hm_train [options]\n"
              << "Options:\n"
              << "  --mode <mode>     Training mode: diffusion, classifier (required)\n"
              << "  --data <dir>      Data directory with JSON clips (required)\n"
              << "  --epochs <n>      Number of epochs (default: 500)\n"
              << "  --batch <n>       Batch size (default: 64)\n"
              << "  --lr <value>      Learning rate (default: 1e-4)\n"
              << "  --output <path>   Output model path (default: model.pt)\n"
              << "  --save-every <n>  Save checkpoint every N epochs (default: 50)\n"
              << "  --help            Show this help\n";
}

static TrainArgs parseArgs(int argc, char* argv[]) {
    TrainArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) args.mode = argv[++i];
        else if (arg == "--data" && i + 1 < argc) args.dataDir = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) args.epochs = std::stoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) args.batchSize = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) args.learningRate = std::stof(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) args.outputPath = argv[++i];
        else if (arg == "--save-every" && i + 1 < argc) args.saveEvery = std::stoi(argv[++i]);
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

static int trainDiffusion(const TrainArgs& args) {
    std::cout << "Training diffusion model...\n";
    std::cout << "Data: " << args.dataDir << "\n";
    std::cout << "Epochs: " << args.epochs << ", Batch: " << args.batchSize
              << ", LR: " << args.learningRate << "\n";

    hm::ml::MotionDiffusionConfig config;
    config.learningRate = args.learningRate;
    config.batchSize = args.batchSize;

    hm::ml::MotionDiffusionModel model(config);
    if (!model.initialize()) {
        std::cerr << "Failed to initialize diffusion model\n";
        return 1;
    }

    // Load training data from JSON files
    std::vector<torch::Tensor> motionData;
    std::vector<torch::Tensor> conditionData;

    for (const auto& entry : std::filesystem::directory_iterator(args.dataDir)) {
        if (entry.path().extension() != ".json") continue;

        try {
            std::ifstream ifs(entry.path());
            nlohmann::json j;
            ifs >> j;

            if (!j.contains("frames")) continue;

            int numFrames = static_cast<int>(j["frames"].size());
            if (numFrames < 64) continue;

            // Extract 64-frame windows
            for (int start = 0; start + 64 <= numFrames; start += 32) {
                auto motionTensor = torch::zeros({64, hm::FRAME_DIM});
                auto accessor = motionTensor.accessor<float, 2>();

                for (int f = 0; f < 64; ++f) {
                    const auto& frameJson = j["frames"][start + f];
                    if (frameJson.contains("joints")) {
                        const auto& joints = frameJson["joints"];
                        for (int ji = 0; ji < hm::JOINT_COUNT && ji < static_cast<int>(joints.size()); ++ji) {
                            if (joints[ji].contains("euler")) {
                                auto euler = joints[ji]["euler"];
                                auto r6d = hm::MathUtils::quatToRot6D(
                                    hm::MathUtils::eulerDegToQuat({
                                        euler[0].get<float>(),
                                        euler[1].get<float>(),
                                        euler[2].get<float>()
                                    }));
                                for (int d = 0; d < hm::ROTATION_DIM; ++d) {
                                    accessor[f][ji * hm::ROTATION_DIM + d] = r6d[d];
                                }
                            }
                        }
                    }
                }

                motionData.push_back(motionTensor);
                conditionData.push_back(torch::zeros({hm::MotionCondition::DIM}));
            }
        } catch (...) {
            continue;
        }
    }

    if (motionData.empty()) {
        std::cerr << "No training data found in: " << args.dataDir << "\n";
        return 1;
    }

    std::cout << "Loaded " << motionData.size() << " training samples\n";

    // Training loop
    auto optimizer = torch::optim::Adam(
        model.transformer()->parameters(), torch::optim::AdamOptions(args.learningRate));

    // Also add condition encoder params
    for (auto& p : model.condEncoder()->parameters()) {
        optimizer.param_groups()[0].params().push_back(p);
    }

    for (int epoch = 0; epoch < args.epochs; ++epoch) {
        float epochLoss = 0.0f;
        int numBatches = 0;

        // Simple random batching
        std::vector<int> indices(motionData.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

        for (size_t i = 0; i + args.batchSize <= indices.size(); i += args.batchSize) {
            std::vector<torch::Tensor> batchMotion, batchCond;
            for (int b = 0; b < args.batchSize; ++b) {
                batchMotion.push_back(motionData[indices[i + b]]);
                batchCond.push_back(conditionData[indices[i + b]]);
            }

            auto x0 = torch::stack(batchMotion);
            auto cond = torch::stack(batchCond);

            optimizer.zero_grad();
            auto loss = model.trainStep(x0, cond);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model.transformer()->parameters(), 1.0);
            optimizer.step();

            epochLoss += loss.item<float>();
            numBatches++;
        }

        float avgLoss = numBatches > 0 ? epochLoss / numBatches : 0.0f;

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << args.epochs
                      << " loss=" << avgLoss << "\n";
        }

        if ((epoch + 1) % args.saveEvery == 0) {
            std::string checkpointPath = args.outputPath + ".epoch" + std::to_string(epoch + 1);
            model.save(checkpointPath);
            std::cout << "Checkpoint saved: " << checkpointPath << "\n";
        }
    }

    model.save(args.outputPath);
    std::cout << "Training complete. Model saved to: " << args.outputPath << "\n";
    return 0;
}

static int trainClassifier(const TrainArgs& args) {
    std::cout << "Training motion classifier (TCN)...\n";

    auto model = hm::segmenter::TemporalConvNet(
        hm::segmenter::MotionFeatureExtractor::FEATURE_DIM, 128, hm::MOTION_TYPE_COUNT);

    auto optimizer = torch::optim::Adam(model->parameters(),
        torch::optim::AdamOptions(args.learningRate));

    std::cout << "Loading data from: " << args.dataDir << "\n";

    // Load labeled segment data
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> labels;

    for (const auto& entry : std::filesystem::directory_iterator(args.dataDir)) {
        if (entry.path().extension() != ".json") continue;

        try {
            std::ifstream ifs(entry.path());
            nlohmann::json j;
            ifs >> j;

            if (!j.contains("frames") || !j.contains("segments")) continue;

            int numFrames = static_cast<int>(j["frames"].size());
            auto featureTensor = torch::zeros({hm::segmenter::MotionFeatureExtractor::FEATURE_DIM, numFrames});
            auto labelTensor = torch::full({numFrames}, static_cast<long>(hm::MotionType::Unknown));

            // Assign labels from segments
            for (const auto& seg : j["segments"]) {
                int typeIdx = seg.value("typeIndex", static_cast<int>(hm::MotionType::Unknown));
                int start = seg["startFrame"].get<int>();
                int end = seg["endFrame"].get<int>();
                for (int f = start; f <= end && f < numFrames; ++f) {
                    labelTensor[f] = typeIdx;
                }
            }

            features.push_back(featureTensor);
            labels.push_back(labelTensor);
        } catch (...) {
            continue;
        }
    }

    if (features.empty()) {
        std::cerr << "No training data found\n";
        return 1;
    }

    std::cout << "Loaded " << features.size() << " sequences\n";

    for (int epoch = 0; epoch < args.epochs; ++epoch) {
        model->train();
        float epochLoss = 0.0f;

        for (size_t i = 0; i < features.size(); ++i) {
            auto input = features[i].unsqueeze(0);  // [1, feat, time]
            auto target = labels[i];                 // [time]

            auto output = model->forward(input);     // [1, classes, time]
            output = output.squeeze(0).transpose(0, 1); // [time, classes]

            auto loss = torch::nn::functional::cross_entropy(output, target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epochLoss += loss.item<float>();
        }

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << args.epochs
                      << " loss=" << (epochLoss / features.size()) << "\n";
        }
    }

    torch::serialize::OutputArchive archive;
    model->save(archive);
    archive.save_to(args.outputPath);
    std::cout << "Classifier saved to: " << args.outputPath << "\n";

    return 0;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.mode.empty() || args.dataDir.empty()) {
        std::cerr << "Error: --mode and --data are required\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(hm::LogLevel::Info);

    if (args.mode == "diffusion") return trainDiffusion(args);
    if (args.mode == "classifier") return trainClassifier(args);

    std::cerr << "Unknown mode: " << args.mode << "\n";
    return 1;
}

#endif  // HM_HAS_TORCH
