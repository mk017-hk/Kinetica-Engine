#include "HyperMotion/dataset/MatchAnalyser.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/PipelineConfigIO.h"
#include "HyperMotion/streaming/StreamingPipeline.h"
#include "HyperMotion/analysis/MotionFingerprint.h"

#include <filesystem>
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
    bool streaming = false;
    bool dryRun = false;
    bool showVersion = false;
    std::string statsOutput;
};

static void printVersion() {
    std::cout << "kinetica_analyse_match v" << hm::HM_VERSION_STRING << "\n"
              << "Schema version: " << hm::HM_SCHEMA_VERSION << "\n"
              << "Backend support:\n"
#ifdef HM_HAS_ONNXRUNTIME
              << "  ONNX Runtime:  enabled\n"
#else
              << "  ONNX Runtime:  disabled (inference uses stubs)\n"
#endif
#ifdef HM_HAS_TORCH
              << "  LibTorch:      enabled\n"
#else
              << "  LibTorch:      disabled (training modules unavailable)\n"
#endif
#ifdef HM_USE_CUDA
              << "  CUDA:          enabled\n"
#else
              << "  CUDA:          disabled (CPU-only)\n"
#endif
#ifdef HM_USE_TENSORRT
              << "  TensorRT:      enabled\n"
#else
              << "  TensorRT:      disabled\n"
#endif
              ;
}

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
        "  --streaming            Use async streaming pipeline\n"
        "  --dry-run              Validate inputs and config, then exit\n"
        "  --stats <path>          Write timing stats to JSON file\n"
        "  --quiet                 Suppress progress output\n"
        "  --version               Show version and backend info\n"
        "  --help                  Show this help\n"
        "\n"
        "Pipeline:\n"
        "  1. Decode video frames\n"
        "  2. Detect players (YOLOv8)\n"
        "  3. Estimate 2D poses (HRNet) and lift to 3D\n"
        "  4. Track players across frames (Hungarian + ReID)\n"
        "  5. Map to 22-joint skeleton\n"
        "  6. Signal processing (outlier, Savitzky-Golay, Butterworth, quat smooth)\n"
        "  7. Canonical motion (stabilise limbs, solve root orientation)\n"
        "  8. Segment motion types (velocity, direction, foot contact)\n"
        "  9. Extract clips (0.5-5s), quality filter, classify\n"
        " 10. Compute motion fingerprints and cluster\n"
        " 11. Export to BVH + JSON animation database\n"
        "\n"
        "Output structure:\n"
        "  <output_dir>/\n"
        "    walk/clip_0001.bvh + .json + .meta.json\n"
        "    jog/clip_0002.bvh + .json + .meta.json\n"
        "    sprint/...\n"
        "    database_summary.json\n"
        "\n"
        "Examples:\n"
        "  kinetica_analyse_match match.mp4 animations/ \\\n"
        "    --detector models/yolov8.onnx \\\n"
        "    --pose models/hrnet.onnx\n"
        "\n"
        "  kinetica_analyse_match match.mp4 animations/ --streaming\n";
}

static AnalyseArgs parseArgs(int argc, char* argv[]) {
    AnalyseArgs args;

    // Collect positional args — skip flags and their values
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("-", 0) == 0) {
            // Skip the value of flags that take a parameter
            if ((arg == "--config" || arg == "--detector" || arg == "--pose" ||
                 arg == "--depth" || arg == "--segmenter" || arg == "--classifier" ||
                 arg == "--fps" || arg == "--stats") && i + 1 < argc) {
                ++i;
            }
            continue;
        }
        if (positional == 0) args.inputVideo = arg;
        else if (positional == 1) args.outputDir = arg;
        positional++;
    }

    // Parse named options
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) args.configFile = argv[++i];
        else if (arg == "--detector" && i + 1 < argc) args.detectorModel = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) args.poseModel = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) args.depthModel = argv[++i];
        else if (arg == "--segmenter" && i + 1 < argc) args.segmenterModel = argv[++i];
        else if (arg == "--classifier" && i + 1 < argc) args.classifierModel = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) {
            try {
                args.fps = std::stof(argv[++i]);
                if (args.fps <= 0.0f || args.fps > 240.0f) {
                    std::cerr << "Error: --fps must be between 0 and 240\n";
                    std::exit(1);
                }
            } catch (...) {
                std::cerr << "Error: --fps requires a numeric value\n";
                std::exit(1);
            }
        }
        else if (arg == "--no-bvh") args.exportBVH = false;
        else if (arg == "--no-json") args.exportJSON = false;
        else if (arg == "--streaming") args.streaming = true;
        else if (arg == "--dry-run") args.dryRun = true;
        else if (arg == "--stats" && i + 1 < argc) args.statsOutput = argv[++i];
        else if (arg == "--quiet") args.quiet = true;
        else if (arg == "--version") args.showVersion = true;
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

static void reportBackendStatus(bool quiet) {
    if (quiet) return;

    std::cout << "  Backends:\n";
#ifdef HM_HAS_ONNXRUNTIME
    std::cout << "    ONNX Runtime:  available\n";
#else
    std::cout << "    ONNX Runtime:  unavailable (ML inference uses stubs/heuristics)\n";
#endif
#ifdef HM_HAS_TORCH
    std::cout << "    LibTorch:      available\n";
#else
    std::cout << "    LibTorch:      unavailable (training modules disabled)\n";
#endif
#ifdef HM_USE_CUDA
    std::cout << "    CUDA:          available\n";
#else
    std::cout << "    CUDA:          unavailable (CPU-only execution)\n";
#endif
#ifdef HM_USE_TENSORRT
    std::cout << "    TensorRT:      available\n";
#else
    std::cout << "    TensorRT:      unavailable\n";
#endif
}

static int runStreaming(const AnalyseArgs& args, hm::PipelineConfig& pipelineCfg) {
    if (!args.quiet) {
        std::cout << "  Mode:     streaming (async decode/inference/analysis)\n\n";
    }

    hm::streaming::StreamingPipelineConfig streamCfg;
    streamCfg.pipelineConfig = pipelineCfg;
    streamCfg.decodeThreads = 1;
    streamCfg.inferenceThreads = 1;
    streamCfg.analysisThreads = 1;

    hm::streaming::StreamingPipeline pipeline(streamCfg);
    if (!pipeline.initialize()) {
        std::cerr << "Error: failed to initialise streaming pipeline\n";
        return 1;
    }

    if (!args.quiet) {
        std::cout << "[Streaming] Pipeline initialised. Processing...\n";
    }

    int clipsDelivered = 0;

    auto clipCB = [&clipsDelivered, &args](hm::AnimClip clip, int playerID) {
        clipsDelivered++;
        if (!args.quiet) {
            std::cout << "  Clip: " << clip.name << " (player " << playerID
                      << ", " << clip.frames.size() << " frames)\n";
        }
    };

    auto progressCB = [&args](const hm::streaming::StreamingStats& stats) {
        if (!args.quiet && stats.framesDecoded % 100 == 0) {
            std::cout << "\r[Streaming] decoded=" << stats.framesDecoded
                      << " inferred=" << stats.framesInferred
                      << " analysed=" << stats.framesAnalysed
                      << " clips=" << stats.clipsProduced
                      << " dropped=" << stats.framesDropped
                      << "          " << std::flush;
        }
    };

    if (!pipeline.startProcessing(args.inputVideo, clipCB, progressCB)) {
        std::cerr << "Error: failed to start streaming processing\n";
        return 1;
    }

    auto clips = pipeline.waitForCompletion();
    auto stats = pipeline.getStats();

    if (!args.quiet) std::cout << "\n\n";

    // Count total frames across clips
    int totalFramesInClips = 0;
    for (const auto& c : clips) {
        totalFramesInClips += static_cast<int>(c.frames.size());
    }

    std::cout << "=== Streaming Analysis Complete ===\n"
              << "  Frames decoded:    " << stats.framesDecoded << "\n"
              << "  Frames inferred:   " << stats.framesInferred << "\n"
              << "  Frames analysed:   " << stats.framesAnalysed << "\n"
              << "  Frames dropped:    " << stats.framesDropped << "\n"
              << "  Clips produced:    " << stats.clipsProduced << "\n"
              << "  Clip frames total: " << totalFramesInClips << "\n"
              << "\n"
              << "  NOTE: Streaming mode produces per-player clips with skeleton\n"
              << "  mapping, signal processing, and segmentation. It does not\n"
              << "  currently run clip extraction, quality filtering, motion\n"
              << "  classification, or database export. Use standard (non-streaming)\n"
              << "  mode for the full pipeline with structured database output.\n"
              << "\n"
              << "  Output: " << args.outputDir << "/\n";

    return 0;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.showVersion) {
        printVersion();
        return 0;
    }

    if (args.inputVideo.empty()) {
        std::cerr << "Error: input video path required\n\n";
        printUsage();
        return 1;
    }

    // Validate input file exists
    if (!std::filesystem::exists(args.inputVideo)) {
        std::cerr << "Error: input video not found: " << args.inputVideo << "\n";
        return 1;
    }

    // Validate input file is a regular file
    if (!std::filesystem::is_regular_file(args.inputVideo)) {
        std::cerr << "Error: input path is not a file: " << args.inputVideo << "\n";
        return 1;
    }

    // Validate output directory is writable (create if needed)
    try {
        std::filesystem::create_directories(args.outputDir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: cannot create output directory: " << args.outputDir
                  << " (" << e.what() << ")\n";
        return 1;
    }

    // Validate model file paths if specified
    auto validateModelPath = [](const std::string& path, const std::string& label) -> bool {
        if (path.empty()) return true;
        if (!std::filesystem::exists(path)) {
            std::cerr << "Error: " << label << " model not found: " << path << "\n";
            return false;
        }
        return true;
    };

    if (!validateModelPath(args.detectorModel, "detector") ||
        !validateModelPath(args.poseModel, "pose") ||
        !validateModelPath(args.depthModel, "depth") ||
        !validateModelPath(args.segmenterModel, "segmenter") ||
        !validateModelPath(args.classifierModel, "classifier")) {
        return 1;
    }

    hm::Logger::instance().setLevel(
        args.quiet ? hm::LogLevel::Warn : hm::LogLevel::Info);

    if (!args.quiet) {
        std::cout << "=== Kinetica Match Analyser v" << hm::HM_VERSION_STRING << " ===\n"
                  << "  Input:  " << args.inputVideo << "\n"
                  << "  Output: " << args.outputDir << "\n"
                  << "  FPS:    " << args.fps << "\n"
                  << "  Export:";
        if (args.exportBVH) std::cout << " BVH";
        if (args.exportJSON) std::cout << " JSON";
        if (!args.exportBVH && !args.exportJSON) std::cout << " (none)";
        std::cout << "\n";

        // Report which ML models are configured
        bool hasModels = false;
        if (!args.detectorModel.empty()) {
            std::cout << "  Detector:   " << args.detectorModel << "\n";
            hasModels = true;
        }
        if (!args.poseModel.empty()) {
            std::cout << "  Pose model: " << args.poseModel << "\n";
            hasModels = true;
        }
        if (!args.depthModel.empty()) {
            std::cout << "  Depth model:" << args.depthModel << "\n";
            hasModels = true;
        }
        if (!args.segmenterModel.empty()) {
            std::cout << "  Segmenter:  " << args.segmenterModel << "\n";
            hasModels = true;
        }
        if (!args.classifierModel.empty()) {
            std::cout << "  Classifier: " << args.classifierModel << "\n";
            hasModels = true;
        }
        if (!hasModels) {
            std::cout << "  Models:     none (heuristic fallbacks will be used)\n";
        }

        reportBackendStatus(false);
        std::cout << "\n";
    }

    // Build config: start from file, overlay CLI args
    hm::PipelineConfig pipelineCfg;
    if (!args.configFile.empty()) {
        if (!std::filesystem::exists(args.configFile)) {
            std::cerr << "Error: config file not found: " << args.configFile << "\n";
            return 1;
        }
        if (!hm::loadPipelineConfig(args.configFile, pipelineCfg)) {
            std::cerr << "Error: failed to parse config: " << args.configFile << "\n";
            return 1;
        }
        if (!args.quiet) {
            std::cout << "  Config loaded from: " << args.configFile << "\n\n";
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

    // Dry run: validate everything then exit
    if (args.dryRun) {
        std::cout << "Dry run: inputs validated, config parsed. Pipeline would run with:\n"
                  << "  Mode:         " << (args.streaming ? "streaming" : "standard") << "\n"
                  << "  Target FPS:   " << args.fps << "\n"
                  << "  Export BVH:   " << (args.exportBVH ? "yes" : "no") << "\n"
                  << "  Export JSON:  " << (args.exportJSON ? "yes" : "no") << "\n"
                  << "  Output dir:   " << args.outputDir << "\n";
        return 0;
    }

    // Streaming mode
    if (args.streaming) {
        return runStreaming(args, pipelineCfg);
    }

    // Standard synchronous mode via MatchAnalyser
    if (!args.quiet) {
        std::cout << "  Mode:     standard (synchronous)\n\n";
    }

    hm::dataset::MatchAnalyserConfig matchCfg;
    matchCfg.pipelineConfig = pipelineCfg;
    matchCfg.classifierModelPath = args.classifierModel;
    matchCfg.outputDirectory = args.outputDir;
    matchCfg.exportBVH = args.exportBVH;
    matchCfg.exportJSON = args.exportJSON;

    // Initialise
    hm::dataset::MatchAnalyser analyser(matchCfg);
    if (!analyser.initialize()) {
        std::cerr << "Error: failed to initialise match analyser.\n"
                  << "  This typically means required dependencies could not be loaded.\n"
                  << "  Check model paths and library availability.\n";
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
              << "  Frames processed:  " << result.totalFramesDecoded << "\n"
              << "  Players tracked:   " << result.totalPlayersTracked << "\n"
              << "  Segments found:    " << result.pipelineStats.segmentsFound << "\n"
              << "  Clips extracted:   " << result.clipsExtracted << "\n"
              << "  Clips accepted:    " << result.clipsAccepted << "\n"
              << "  Clips rejected:    " << result.clipsRejected << "\n"
              << "  Processing time:   " << std::fixed << std::setprecision(1)
              << result.totalProcessingMs / 1000.0 << "s\n"
              << "\n"
              << "  Database:\n"
              << "    Total clips:     " << result.dbStats.totalClips << "\n"
              << "    Total frames:    " << result.dbStats.totalFrames << "\n"
              << "    Duration:        " << std::setprecision(1)
              << result.dbStats.totalDurationSec << "s\n"
              << "    Unique players:  " << result.dbStats.uniquePlayers << "\n"
              << "\n";

    // Clips by type
    bool hasAnyType = false;
    for (int i = 0; i < hm::MOTION_TYPE_COUNT; ++i) {
        if (result.dbStats.clipsByType[i] > 0) hasAnyType = true;
    }
    if (hasAnyType) {
        std::cout << "  Clips by type:\n";
        for (int i = 0; i < hm::MOTION_TYPE_COUNT; ++i) {
            if (result.dbStats.clipsByType[i] > 0) {
                std::cout << "    " << std::setw(12) << std::left
                          << hm::MOTION_TYPE_NAMES[i] << ": "
                          << result.dbStats.clipsByType[i] << "\n";
            }
        }
        std::cout << "\n";
    }

    // Report pipeline stage timings
    std::cout << "  Stage timings:\n";
    auto& ps = result.pipelineStats;
    if (ps.poseExtractionMs > 0)
        std::cout << "    Pose extraction:   " << static_cast<int>(ps.poseExtractionMs) << "ms\n";
    if (ps.skeletonMappingMs > 0)
        std::cout << "    Skeleton mapping:  " << static_cast<int>(ps.skeletonMappingMs) << "ms\n";
    if (ps.signalProcessingMs > 0)
        std::cout << "    Signal processing: " << static_cast<int>(ps.signalProcessingMs) << "ms\n";
    if (ps.segmentationMs > 0)
        std::cout << "    Segmentation:      " << static_cast<int>(ps.segmentationMs) << "ms\n";
    if (ps.canonicalMotionMs > 0)
        std::cout << "    Canonical motion:  " << static_cast<int>(ps.canonicalMotionMs) << "ms\n";
    if (ps.footContactMs > 0)
        std::cout << "    Foot contact:      " << static_cast<int>(ps.footContactMs) << "ms\n";
    if (ps.trajectoryMs > 0)
        std::cout << "    Trajectory:        " << static_cast<int>(ps.trajectoryMs) << "ms\n";
    if (ps.fingerprintMs > 0)
        std::cout << "    Fingerprinting:    " << static_cast<int>(ps.fingerprintMs) << "ms\n";
    if (ps.clusteringMs > 0)
        std::cout << "    Clustering:        " << static_cast<int>(ps.clusteringMs) << "ms\n";
    if (ps.exportMs > 0)
        std::cout << "    Export:            " << static_cast<int>(ps.exportMs) << "ms\n";

    std::cout << "\n  Output: " << args.outputDir << "/\n";

    // Count exported files
    int exportedFiles = 0;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(args.outputDir)) {
            if (entry.is_regular_file()) ++exportedFiles;
        }
        std::cout << "  Files exported:    " << exportedFiles << "\n";
    } catch (...) {}

    // Save stats
    if (!args.statsOutput.empty()) {
        hm::savePipelineStats(args.statsOutput, result.pipelineStats);
        std::cout << "  Stats written to:  " << args.statsOutput << "\n";
    }

    return 0;
}
