#include "HyperMotion/Pipeline.h"
#include "HyperMotion/core/Logger.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct ExtractArgs {
    std::string detectorModel;
    std::string poseModel;
    std::string inputVideo;
    std::string outputDir = "output";
    float fps = 30.0f;
    bool visualize = false;
    bool split = false;
    std::string format = "json";
    std::string depthModel;
    std::string segmenterModel;
};

static void printUsage() {
    std::cout << "Usage: hm_extract [options]\n"
              << "Options:\n"
              << "  --detector <path>    YOLOv8 ONNX model path (required)\n"
              << "  --pose <path>        HRNet ONNX model path (required)\n"
              << "  --input <path>       Input video file (required)\n"
              << "  --output <dir>       Output directory (default: output)\n"
              << "  --fps <value>        Target FPS (default: 30)\n"
              << "  --visualize          Enable debug visualization\n"
              << "  --split              Split clips by motion segment\n"
              << "  --format <fmt>       Output format: json, bvh, both (default: json)\n"
              << "  --depth <path>       Depth lifting model path (optional)\n"
              << "  --segmenter <path>   TCN segmenter model path (optional)\n"
              << "  --help               Show this help\n";
}

static ExtractArgs parseArgs(int argc, char* argv[]) {
    ExtractArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--detector" && i + 1 < argc) args.detectorModel = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) args.poseModel = argv[++i];
        else if (arg == "--input" && i + 1 < argc) args.inputVideo = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.outputDir = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) args.fps = std::stof(argv[++i]);
        else if (arg == "--visualize") args.visualize = true;
        else if (arg == "--split") args.split = true;
        else if (arg == "--format" && i + 1 < argc) args.format = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) args.depthModel = argv[++i];
        else if (arg == "--segmenter" && i + 1 < argc) args.segmenterModel = argv[++i];
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.detectorModel.empty() || args.poseModel.empty() || args.inputVideo.empty()) {
        std::cerr << "Error: --detector, --pose, and --input are required\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(hm::LogLevel::Info);

    hm::PipelineConfig config;
    config.poseConfig.detector.modelPath = args.detectorModel;
    config.poseConfig.poseEstimator.modelPath = args.poseModel;
    config.poseConfig.depthLifter.modelPath = args.depthModel;
    config.poseConfig.targetFPS = args.fps;
    config.poseConfig.enableVisualization = args.visualize;
    config.segmenterConfig.modelPath = args.segmenterModel;
    config.targetFPS = args.fps;
    config.splitBySegment = args.split;
    config.outputDirectory = args.outputDir;
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

    return 0;
}
