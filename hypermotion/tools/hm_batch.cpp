#include "HyperMotion/Pipeline.h"
#include "HyperMotion/core/Logger.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>

struct BatchArgs {
    std::string detectorModel;
    std::string poseModel;
    std::string inputDir;
    std::string outputDir = "batch_output";
    float fps = 30.0f;
    std::string format = "json";
    std::string depthModel;
    std::string segmenterModel;
    bool split = false;
    int maxVideos = -1;
};

static void printUsage() {
    std::cout << "Usage: hm_batch [options]\n"
              << "Options:\n"
              << "  --detector <path>    YOLOv8 ONNX model path (required)\n"
              << "  --pose <path>        HRNet ONNX model path (required)\n"
              << "  --input <dir>        Input directory with video files (required)\n"
              << "  --output <dir>       Output directory (default: batch_output)\n"
              << "  --fps <value>        Target FPS (default: 30)\n"
              << "  --format <fmt>       Output format: json, bvh, both (default: json)\n"
              << "  --depth <path>       Depth lifting model path (optional)\n"
              << "  --segmenter <path>   TCN segmenter model path (optional)\n"
              << "  --split              Split clips by motion segment\n"
              << "  --max <n>            Max number of videos to process\n"
              << "  --help               Show this help\n";
}

static BatchArgs parseArgs(int argc, char* argv[]) {
    BatchArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--detector" && i + 1 < argc) args.detectorModel = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) args.poseModel = argv[++i];
        else if (arg == "--input" && i + 1 < argc) args.inputDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.outputDir = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) args.fps = std::stof(argv[++i]);
        else if (arg == "--format" && i + 1 < argc) args.format = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) args.depthModel = argv[++i];
        else if (arg == "--segmenter" && i + 1 < argc) args.segmenterModel = argv[++i];
        else if (arg == "--split") args.split = true;
        else if (arg == "--max" && i + 1 < argc) args.maxVideos = std::stoi(argv[++i]);
        else if (arg == "--help") { printUsage(); std::exit(0); }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    if (args.detectorModel.empty() || args.poseModel.empty() || args.inputDir.empty()) {
        std::cerr << "Error: --detector, --pose, and --input are required\n";
        printUsage();
        return 1;
    }

    hm::Logger::instance().setLevel(hm::LogLevel::Info);

    // Collect video files
    std::vector<std::string> videoFiles;
    const std::vector<std::string> videoExts = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

    for (const auto& entry : std::filesystem::directory_iterator(args.inputDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(videoExts.begin(), videoExts.end(), ext) != videoExts.end()) {
            videoFiles.push_back(entry.path().string());
        }
    }

    std::sort(videoFiles.begin(), videoFiles.end());

    if (args.maxVideos > 0 && static_cast<int>(videoFiles.size()) > args.maxVideos) {
        videoFiles.resize(args.maxVideos);
    }

    std::cout << "Found " << videoFiles.size() << " video files\n";

    if (videoFiles.empty()) {
        std::cerr << "No video files found in: " << args.inputDir << "\n";
        return 1;
    }

    // Setup pipeline
    hm::PipelineConfig config;
    config.poseConfig.detector.modelPath = args.detectorModel;
    config.poseConfig.poseEstimator.modelPath = args.poseModel;
    config.poseConfig.depthLifter.modelPath = args.depthModel;
    config.poseConfig.targetFPS = args.fps;
    config.segmenterConfig.modelPath = args.segmenterModel;
    config.targetFPS = args.fps;
    config.splitBySegment = args.split;
    config.outputFormat = args.format;

    hm::Pipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize pipeline\n";
        return 1;
    }

    std::filesystem::create_directories(args.outputDir);

    int totalClips = 0;
    int processedVideos = 0;

    for (const auto& videoPath : videoFiles) {
        std::string videoName = std::filesystem::path(videoPath).stem().string();
        std::string videoOutputDir = args.outputDir + "/" + videoName;

        std::cout << "\n[" << (processedVideos + 1) << "/" << videoFiles.size()
                  << "] Processing: " << videoPath << "\n";

        config.outputDirectory = videoOutputDir;

        auto clips = pipeline.processVideo(videoPath, [&](float pct, const std::string& stage) {
            std::cout << "\r  [" << static_cast<int>(pct) << "%] " << stage << std::flush;
        });
        std::cout << "\n";

        // Export
        pipeline.exportClips(clips, videoOutputDir);

        totalClips += static_cast<int>(clips.size());
        processedVideos++;

        std::cout << "  -> " << clips.size() << " clips extracted\n";
    }

    std::cout << "\nBatch processing complete.\n"
              << "  Videos processed: " << processedVideos << "\n"
              << "  Total clips: " << totalClips << "\n"
              << "  Output directory: " << args.outputDir << "\n";

    return 0;
}
