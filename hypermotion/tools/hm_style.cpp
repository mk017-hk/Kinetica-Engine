#include "HyperMotion/style/StyleTrainer.h"
#include "HyperMotion/style/StyleLibrary.h"
#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/MathUtils.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

struct StyleArgs {
    bool trainMode = false;
    bool encodeMode = false;
    std::string dataDir;
    std::string outputDir = "styles";
    std::string modelPath;
    int epochs = 200;
    int batchSize = 32;
    float learningRate = 1e-4f;
};

static void printUsage() {
    std::cout << "Usage: hm_style [options]\n"
              << "Options:\n"
              << "  --train             Train style encoder\n"
              << "  --encode            Encode player clips to style embeddings\n"
              << "  --data <dir>        Data directory with player clip subdirectories (required)\n"
              << "  --output <dir>      Output directory (default: styles)\n"
              << "  --model <path>      Pre-trained encoder path (for --encode)\n"
              << "  --epochs <n>        Training epochs (default: 200)\n"
              << "  --batch <n>         Batch size (default: 32)\n"
              << "  --lr <value>        Learning rate (default: 1e-4)\n"
              << "  --help              Show this help\n";
}

static StyleArgs parseArgs(int argc, char* argv[]) {
    StyleArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") args.trainMode = true;
        else if (arg == "--encode") args.encodeMode = true;
        else if (arg == "--data" && i + 1 < argc) args.dataDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.outputDir = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.modelPath = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) args.epochs = std::stoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) args.batchSize = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) args.learningRate = std::stof(argv[++i]);
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

// Load skeleton frames from JSON clip file
static std::vector<hm::SkeletonFrame> loadClipFrames(const std::string& path) {
    std::vector<hm::SkeletonFrame> frames;
    try {
        std::ifstream ifs(path);
        nlohmann::json j;
        ifs >> j;

        if (!j.contains("frames")) return frames;

        for (const auto& fj : j["frames"]) {
            hm::SkeletonFrame frame;
            frame.frameIndex = fj.value("frameIndex", 0);
            frame.timestamp = fj.value("timestamp", 0.0);

            if (fj.contains("rootPosition")) {
                auto rp = fj["rootPosition"];
                frame.rootPosition = {rp[0].get<float>(), rp[1].get<float>(), rp[2].get<float>()};
            }
            if (fj.contains("rootVelocity")) {
                auto rv = fj["rootVelocity"];
                frame.rootVelocity = {rv[0].get<float>(), rv[1].get<float>(), rv[2].get<float>()};
            }

            if (fj.contains("joints")) {
                for (int ji = 0; ji < hm::JOINT_COUNT && ji < static_cast<int>(fj["joints"].size()); ++ji) {
                    const auto& jj = fj["joints"][ji];
                    if (jj.contains("euler")) {
                        auto e = jj["euler"];
                        frame.joints[ji].localEulerDeg = {
                            e[0].get<float>(), e[1].get<float>(), e[2].get<float>()};
                        frame.joints[ji].localRotation =
                            hm::MathUtils::eulerDegToQuat(frame.joints[ji].localEulerDeg);
                        frame.joints[ji].rotation6D =
                            hm::MathUtils::quatToRot6D(frame.joints[ji].localRotation);
                    }
                    if (jj.contains("position")) {
                        auto p = jj["position"];
                        frame.joints[ji].worldPosition = {
                            p[0].get<float>(), p[1].get<float>(), p[2].get<float>()};
                    }
                }
            }

            frames.push_back(frame);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading " << path << ": " << e.what() << "\n";
    }
    return frames;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.dataDir.empty() || (!args.trainMode && !args.encodeMode)) {
        std::cerr << "Error: --data and either --train or --encode required\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(hm::LogLevel::Info);
    std::filesystem::create_directories(args.outputDir);

    // Load training pairs from player subdirectories
    std::vector<hm::style::TrainingPair> trainingData;

    for (const auto& playerDir : std::filesystem::directory_iterator(args.dataDir)) {
        if (!playerDir.is_directory()) continue;

        std::string playerID = playerDir.path().filename().string();
        std::cout << "Loading clips for player: " << playerID << "\n";

        for (const auto& clipFile : std::filesystem::directory_iterator(playerDir.path())) {
            if (clipFile.path().extension() != ".json") continue;

            auto frames = loadClipFrames(clipFile.path().string());
            if (frames.size() >= 30) {
                hm::style::TrainingPair pair;
                pair.playerID = playerID;
                pair.clip = std::move(frames);
                trainingData.push_back(std::move(pair));
            }
        }
    }

    std::cout << "Total training pairs: " << trainingData.size() << "\n";

    if (args.trainMode) {
        hm::style::StyleTrainerConfig trainConfig;
        trainConfig.numEpochs = args.epochs;
        trainConfig.batchSize = args.batchSize;
        trainConfig.learningRate = args.learningRate;
        trainConfig.savePath = args.outputDir + "/style_encoder.pt";

        hm::style::StyleTrainer trainer(trainConfig);
        trainer.train(trainingData, [](int epoch, float loss, float lr) {
            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << (epoch + 1) << " loss=" << loss << " lr=" << lr << "\n";
            }
        });

        std::cout << "Training complete. Encoder saved to: " << trainConfig.savePath << "\n";
    }

    if (args.encodeMode) {
#ifdef HM_HAS_TORCH
        hm::style::StyleEncoder encoder;
        if (!args.modelPath.empty()) {
            torch::serialize::InputArchive archive;
            archive.load_from(args.modelPath);
            encoder->load(archive);
        }
        encoder->eval();

        hm::style::StyleLibrary library;
        torch::NoGradGuard noGrad;

        // Group clips by player and compute average embedding
        std::map<std::string, std::vector<std::array<float, hm::STYLE_DIM>>> playerEmbeddings;

        for (const auto& pair : trainingData) {
            auto input = hm::style::StyleEncoderImpl::prepareInput(pair.clip);
            auto embedding = encoder->forward(input);
            auto acc = embedding.accessor<float, 2>();

            std::array<float, hm::STYLE_DIM> emb{};
            for (int d = 0; d < hm::STYLE_DIM; ++d) {
                emb[d] = acc[0][d];
            }
            playerEmbeddings[pair.playerID].push_back(emb);
        }

        for (const auto& [pid, embeddings] : playerEmbeddings) {
            hm::PlayerStyle style;
            style.playerID = pid;
            style.playerName = pid;
            style.embedding.fill(0.0f);

            for (const auto& emb : embeddings) {
                for (int d = 0; d < hm::STYLE_DIM; ++d) {
                    style.embedding[d] += emb[d];
                }
            }
            float n = static_cast<float>(embeddings.size());
            for (int d = 0; d < hm::STYLE_DIM; ++d) {
                style.embedding[d] /= n;
            }

            // L2 normalize
            float norm = 0.0f;
            for (int d = 0; d < hm::STYLE_DIM; ++d) norm += style.embedding[d] * style.embedding[d];
            norm = std::sqrt(norm);
            if (norm > 1e-8f) {
                for (int d = 0; d < hm::STYLE_DIM; ++d) style.embedding[d] /= norm;
            }

            library.addStyle(style);
            std::cout << "Encoded style for: " << pid
                      << " (" << embeddings.size() << " clips)\n";
        }

        std::string libPath = args.outputDir + "/style_library.json";
        library.saveJSON(libPath);
        std::cout << "Style library saved to: " << libPath << "\n";
#else
        std::cerr << "Encode mode requires LibTorch. Rebuild with -DTorch_DIR=<path-to-libtorch>\n";
        return 1;
#endif
    }

    return 0;
}
