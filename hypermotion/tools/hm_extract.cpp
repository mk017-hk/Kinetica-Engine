#include "HyperMotion/Pipeline.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/PipelineConfigIO.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct ExtractArgs {
    std::string configFile;
    std::string detectorModel;
    std::string poseModel;
    std::string inputVideo;
    std::string outputDir;
    float fps = -1.0f;     // -1 = not set on CLI
    bool visualize = false;
    bool visualizeSet = false;
    bool split = false;
    bool splitSet = false;
    std::string format;
    std::string depthModel;
    std::string segmenterModel;
    std::string statsOutput;
};

static void printUsage() {
    std::cout << "Usage: hm_extract [options]\n"
              << "Options:\n"
              << "  --config <path>      Pipeline config JSON (use hm_config to generate)\n"
              << "  --detector <path>    YOLOv8 ONNX model path\n"
              << "  --pose <path>        HRNet ONNX model path\n"
              << "  --input <path>       Input video file (required)\n"
              << "  --output <dir>       Output directory (default: output)\n"
              << "  --fps <value>        Target FPS (default: 30)\n"
              << "  --visualize          Enable debug visualization\n"
              << "  --split              Split clips by motion segment\n"
              << "  --format <fmt>       Output format: json, bvh, both (default: json)\n"
              << "  --depth <path>       Depth lifting model path (optional)\n"
              << "  --segmenter <path>   TCN segmenter model path (optional)\n"
              << "  --stats <path>       Write pipeline timing stats to JSON file\n"
              << "  --help               Show this help\n"
              << "\n"
              << "CLI arguments override values from the config file.\n";
}

static ExtractArgs parseArgs(int argc, char* argv[]) {
    ExtractArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) args.configFile = argv[++i];
        else if (arg == "--detector" && i + 1 < argc) args.detectorModel = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) args.poseModel = argv[++i];
        else if (arg == "--input" && i + 1 < argc) args.inputVideo = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.outputDir = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) args.fps = std::stof(argv[++i]);
        else if (arg == "--visualize") { args.visualize = true; args.visualizeSet = true; }
        else if (arg == "--split") { args.split = true; args.splitSet = true; }
        else if (arg == "--format" && i + 1 < argc) args.format = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) args.depthModel = argv[++i];
        else if (arg == "--segmenter" && i + 1 < argc) args.segmenterModel = argv[++i];
        else if (arg == "--stats" && i + 1 < argc) args.statsOutput = argv[++i];
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.inputVideo.empty()) {
        std::cerr << "Error: --input is required\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(hm::LogLevel::Info);

    // Start from config file defaults, then overlay CLI args
    hm::PipelineConfig config;
    if (!args.configFile.empty()) {
        if (!hm::loadPipelineConfig(args.configFile, config)) {
            std::cerr << "Failed to load config: " << args.configFile << "\n";
            return 1;
        }
    }

    // CLI overrides
    if (!args.detectorModel.empty())
        config.poseConfig.detector.modelPath = args.detectorModel;
    if (!args.poseModel.empty())
        config.poseConfig.poseEstimator.modelPath = args.poseModel;
    if (!args.depthModel.empty())
        config.poseConfig.depthLifter.modelPath = args.depthModel;
    if (!args.segmenterModel.empty())
        config.segmenterConfig.modelPath = args.segmenterModel;
    if (args.fps > 0.0f) {
        config.targetFPS = args.fps;
        config.poseConfig.targetFPS = args.fps;
    }
    if (args.visualizeSet) {
        config.enableVisualization = args.visualize;
        config.poseConfig.enableVisualization = args.visualize;
    }
    if (args.splitSet)
        config.splitBySegment = args.split;
    if (!args.outputDir.empty())
        config.outputDirectory = args.outputDir;
    else if (config.outputDirectory.empty())
        config.outputDirectory = "output";
    if (!args.format.empty())
        config.outputFormat = args.format;

    hm::Pipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize pipeline\n";
        return 1;
    }

    auto progressCB = [](float pct, const std::string& stage) {
        std::cout << "\r[" << static_cast<int>(pct) << "%] " << stage << std::flush;
    };

    auto clips = pipeline.processVideo(args.inputVideo, progressCB);
    std::cout << "\n";

    std::cout << "Extracted " << clips.size() << " animation clips\n";
    for (const auto& clip : clips) {
        std::cout << "  - " << clip.name << ": " << clip.frames.size()
                  << " frames, " << clip.segments.size() << " segments\n";
    }

    // Optionally save timing stats
    if (!args.statsOutput.empty()) {
        hm::savePipelineStats(args.statsOutput, pipeline.getLastStats());
        std::cout << "Stats written to: " << args.statsOutput << "\n";
    }

    return 0;
}
