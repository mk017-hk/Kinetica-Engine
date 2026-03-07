#include "HyperMotion/dataset/MatchAnalyser.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/PipelineConfigIO.h"

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

struct AnalyseArgs {
    std::string inputVideo;
    std::string outputDir = "animations";
    std::string configFile;
    std::string detectorModel;
    std::string poseModel;
    std::string depthModel;
    std::string segmenterModel;
    std::string classifierModel;
    float fps = 30.0f;
    bool exportBVH = true;
    bool exportJSON = true;
    bool quiet = false;
    std::string statsOutput;
};

static void printUsage() {
    std::cout <<
        "kinetica_analyse_match — Extract animation clips from match footage\n"
        "\n"
        "Usage:\n"
        "  kinetica_analyse_match <input_video> <output_dir> [options]\n"
        "\n"
        "Arguments:\n"
        "  <input_video>           Path to match video (.mp4, .avi, .mov)\n"
        "  <output_dir>            Output directory for animation database\n"
        "\n"
        "Options:\n"
        "  --config <path>         Pipeline config JSON file\n"
        "  --detector <path>       YOLOv8 detection model (.onnx)\n"
        "  --pose <path>           HRNet pose estimation model (.onnx)\n"
        "  --depth <path>          Depth lifting model (.onnx)\n"
        "  --segmenter <path>      Motion segmenter TCN model (.onnx)\n"
        "  --classifier <path>     Motion classifier model (.onnx)\n"
        "  --fps <value>           Target FPS (default: 30)\n"
        "  --no-bvh               Skip BVH export\n"
        "  --no-json              Skip JSON export\n"
        "  --stats <path>          Write timing stats to JSON file\n"
        "  --quiet                 Suppress progress output\n"
        "  --help                  Show this help\n"
        "\n"
        "Output structure:\n"
        "  <output_dir>/\n"
        "    walk/clip_0001.bvh + .json + .meta.json\n"
        "    jog/clip_0002.bvh + .json + .meta.json\n"
        "    sprint/...\n"
        "    database_summary.json\n"
        "\n"
        "Example:\n"
        "  kinetica_analyse_match match.mp4 animations/ \\\n"
        "    --detector models/yolov8.onnx \\\n"
        "    --pose models/hrnet.onnx\n";
}

static AnalyseArgs parseArgs(int argc, char* argv[]) {
    AnalyseArgs args;

    // First two positional args
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') break;
        if (positional == 0) args.inputVideo = arg;
        else if (positional == 1) args.outputDir = arg;
        positional++;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) args.configFile = argv[++i];
        else if (arg == "--detector" && i + 1 < argc) args.detectorModel = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) args.poseModel = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) args.depthModel = argv[++i];
        else if (arg == "--segmenter" && i + 1 < argc) args.segmenterModel = argv[++i];
        else if (arg == "--classifier" && i + 1 < argc) args.classifierModel = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) args.fps = std::stof(argv[++i]);
        else if (arg == "--no-bvh") args.exportBVH = false;
        else if (arg == "--no-json") args.exportJSON = false;
        else if (arg == "--stats" && i + 1 < argc) args.statsOutput = argv[++i];
        else if (arg == "--quiet") args.quiet = true;
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.inputVideo.empty()) {
        std::cerr << "Error: input video path required\n\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(
        args.quiet ? hm::LogLevel::Warn : hm::LogLevel::Info);

    // Build config: start from file, overlay CLI args
    hm::PipelineConfig pipelineCfg;
    if (!args.configFile.empty()) {
        if (!hm::loadPipelineConfig(args.configFile, pipelineCfg)) {
            std::cerr << "Failed to load config: " << args.configFile << "\n";
            return 1;
        }
    }

    // CLI overrides
    if (!args.detectorModel.empty())
        pipelineCfg.poseConfig.detector.modelPath = args.detectorModel;
    if (!args.poseModel.empty())
        pipelineCfg.poseConfig.poseEstimator.modelPath = args.poseModel;
    if (!args.depthModel.empty())
        pipelineCfg.poseConfig.depthLifter.modelPath = args.depthModel;
    if (!args.segmenterModel.empty())
        pipelineCfg.segmenterConfig.modelPath = args.segmenterModel;
    pipelineCfg.targetFPS = args.fps;
    pipelineCfg.poseConfig.targetFPS = args.fps;

    // Build MatchAnalyserConfig
    hm::dataset::MatchAnalyserConfig matchCfg;
    matchCfg.pipelineConfig = pipelineCfg;
    matchCfg.classifierModelPath = args.classifierModel;
    matchCfg.outputDirectory = args.outputDir;
    matchCfg.exportBVH = args.exportBVH;
    matchCfg.exportJSON = args.exportJSON;

    // Initialise
    hm::dataset::MatchAnalyser analyser(matchCfg);
    if (!analyser.initialize()) {
        std::cerr << "Failed to initialise match analyser\n";
        return 1;
    }

    // Run
    auto progressCB = [&args](float pct, const std::string& stage) {
        if (!args.quiet) {
            std::cout << "\r[" << std::fixed << std::setprecision(1)
                      << pct << "%] " << stage << "          " << std::flush;
        }
    };

    auto result = analyser.processMatch(args.inputVideo, progressCB);

    if (!args.quiet) std::cout << "\n\n";

    // Print summary
    std::cout << "=== Match Analysis Complete ===\n"
              << "  Video frames:    " << result.totalFramesDecoded << "\n"
              << "  Players tracked: " << result.totalPlayersTracked << "\n"
              << "  Clips extracted: " << result.clipsExtracted << "\n"
              << "  Clips accepted:  " << result.clipsAccepted << "\n"
              << "  Clips rejected:  " << result.clipsRejected << "\n"
              << "  Processing time: " << std::fixed << std::setprecision(1)
              << result.totalProcessingMs / 1000.0 << "s\n"
              << "\n"
              << "  Database:\n"
              << "    Total clips:    " << result.dbStats.totalClips << "\n"
              << "    Total frames:   " << result.dbStats.totalFrames << "\n"
              << "    Duration:       " << std::setprecision(1)
              << result.dbStats.totalDurationSec << "s\n"
              << "    Unique players: " << result.dbStats.uniquePlayers << "\n"
              << "\n"
              << "  Clips by type:\n";

    for (int i = 0; i < hm::MOTION_TYPE_COUNT; ++i) {
        if (result.dbStats.clipsByType[i] > 0) {
            std::cout << "    " << std::setw(12) << std::left
                      << hm::MOTION_TYPE_NAMES[i] << ": "
                      << result.dbStats.clipsByType[i] << "\n";
        }
    }

    std::cout << "\n  Output: " << args.outputDir << "/\n";

    // Save stats
    if (!args.statsOutput.empty()) {
        hm::savePipelineStats(args.statsOutput, result.pipelineStats);
        std::cout << "  Stats: " << args.statsOutput << "\n";
    }

    return 0;
}
