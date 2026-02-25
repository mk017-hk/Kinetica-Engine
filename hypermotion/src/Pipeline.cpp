#include "HyperMotion/Pipeline.h"
#include "HyperMotion/core/Logger.h"

#include <filesystem>
#include <set>
#include <chrono>
#include <sstream>

namespace hm {

static constexpr const char* TAG = "Pipeline";

using Clock = std::chrono::high_resolution_clock;

static double elapsedMs(Clock::time_point start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Pipeline::Impl {
    PipelineConfig config;
    pose::MultiPersonPoseEstimator poseEstimator;
    skeleton::SkeletonMapper skeletonMapper;
    signal::SignalPipeline signalPipeline;
    segmenter::MotionSegmenter motionSegmenter;
    xport::BVHExporter bvhExporter;
    xport::JSONExporter jsonExporter;
    bool initialized = false;
    PipelineStats lastStats;

    Impl(const PipelineConfig& cfg)
        : config(cfg)
        , poseEstimator(cfg.poseConfig)
        , skeletonMapper(cfg.mapperConfig)
        , signalPipeline(cfg.signalConfig)
        , motionSegmenter(cfg.segmenterConfig)
        , bvhExporter(cfg.bvhConfig)
        , jsonExporter(cfg.jsonConfig) {}
};

Pipeline::Pipeline(const PipelineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

bool Pipeline::initialize() {
    HM_LOG_INFO(TAG, "Initializing HyperMotion pipeline...");

    if (!impl_->poseEstimator.initialize()) {
        HM_LOG_ERROR(TAG, "Failed to initialize pose estimator");
        return false;
    }

    if (!impl_->motionSegmenter.initialize()) {
        HM_LOG_WARN(TAG, "Motion segmenter initialization failed (segmentation will be unavailable)");
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Pipeline initialized successfully");
    return true;
}

bool Pipeline::isInitialized() const {
    return impl_->initialized;
}

const PipelineStats& Pipeline::getLastStats() const {
    return impl_->lastStats;
}

std::vector<PoseFrameResult> Pipeline::extractPoses(
    const std::string& videoPath, PipelineProgressCallback callback) {

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Pipeline not initialized");
        return {};
    }

    HM_LOG_INFO(TAG, "Extracting poses from: " + videoPath);

    auto poseCB = [&callback](float pct, const std::string& msg) {
        if (callback) callback(pct * 0.5f, "Pose Extraction: " + msg);
    };

    auto start = Clock::now();
    auto result = impl_->poseEstimator.processVideo(videoPath, poseCB);
    impl_->lastStats.poseExtractionMs = elapsedMs(start);
    impl_->lastStats.totalFramesProcessed = static_cast<int>(result.size());

    return result;
}

std::vector<AnimClip> Pipeline::buildClips(
    const std::vector<PoseFrameResult>& poseResults) {

    if (poseResults.empty()) return {};

    // Collect all unique tracking IDs
    std::set<int> trackingIDs;
    for (const auto& frame : poseResults) {
        for (const auto& person : frame.persons) {
            trackingIDs.insert(person.id);
        }
    }

    impl_->lastStats.trackedPersons = static_cast<int>(trackingIDs.size());
    HM_LOG_INFO(TAG, "Building clips for " + std::to_string(trackingIDs.size()) + " tracked persons");

    std::vector<AnimClip> clips;

    for (int id : trackingIDs) {
        // Step 1: Map COCO keypoints to skeleton
        auto mapStart = Clock::now();
        auto skeletonFrames = impl_->skeletonMapper.mapSequence(poseResults, id);
        impl_->lastStats.skeletonMappingMs += elapsedMs(mapStart);

        if (static_cast<int>(skeletonFrames.size()) < impl_->config.minTrackFrames) {
            HM_LOG_DEBUG(TAG, "Skipping track " + std::to_string(id) +
                         " (only " + std::to_string(skeletonFrames.size()) + " frames)");
            continue;
        }

        // Step 2: Signal processing (smoothing, filtering)
        auto signalStart = Clock::now();
        impl_->signalPipeline.process(skeletonFrames);
        impl_->lastStats.signalProcessingMs += elapsedMs(signalStart);

        // Step 3: Motion segmentation
        auto segStart = Clock::now();
        auto segments = impl_->motionSegmenter.segment(skeletonFrames, id);
        impl_->lastStats.segmentationMs += elapsedMs(segStart);
        impl_->lastStats.segmentsFound += static_cast<int>(segments.size());

        // Build clip
        AnimClip clip;
        clip.name = "track_" + std::to_string(id);
        clip.fps = impl_->config.targetFPS;
        clip.trackingID = id;
        clip.frames = std::move(skeletonFrames);
        clip.segments = std::move(segments);

        clips.push_back(std::move(clip));
    }

    // Optionally split by segment
    if (impl_->config.splitBySegment) {
        std::vector<AnimClip> splitClips;
        for (const auto& clip : clips) {
            auto split = xport::AnimClipUtils::splitBySegments(clip);
            for (auto& s : split) {
                splitClips.push_back(std::move(s));
            }
        }
        if (!splitClips.empty()) {
            clips.insert(clips.end(), splitClips.begin(), splitClips.end());
        }
    }

    impl_->lastStats.clipsProduced = static_cast<int>(clips.size());
    HM_LOG_INFO(TAG, "Built " + std::to_string(clips.size()) + " animation clips");
    return clips;
}

void Pipeline::exportClips(const std::vector<AnimClip>& clips, const std::string& outputDir) {
    if (clips.empty()) return;

    auto exportStart = Clock::now();

    std::filesystem::create_directories(outputDir);

    for (const auto& clip : clips) {
        std::string basePath = outputDir + "/" + clip.name;

        if (impl_->config.outputFormat == "json" || impl_->config.outputFormat == "both") {
            impl_->jsonExporter.exportToFile(clip, basePath + ".json");
        }

        if (impl_->config.outputFormat == "bvh" || impl_->config.outputFormat == "both") {
            impl_->bvhExporter.exportToFile(clip, basePath + ".bvh");
        }
    }

    impl_->lastStats.exportMs = elapsedMs(exportStart);
    HM_LOG_INFO(TAG, "Exported " + std::to_string(clips.size()) + " clips to: " + outputDir);
}

std::vector<AnimClip> Pipeline::processVideo(
    const std::string& videoPath, PipelineProgressCallback callback) {

    // Reset stats
    impl_->lastStats = PipelineStats{};
    auto totalStart = Clock::now();

    // Step 1: Extract poses (0-50%)
    auto poseResults = extractPoses(videoPath, callback);

    if (callback) callback(50.0f, "Building animation clips");

    // Step 2: Build clips (50-80%)
    auto clips = buildClips(poseResults);

    if (callback) callback(80.0f, "Exporting");

    // Step 3: Export (80-100%)
    if (!impl_->config.outputDirectory.empty()) {
        exportClips(clips, impl_->config.outputDirectory);
    }

    impl_->lastStats.totalMs = elapsedMs(totalStart);

    // Log summary stats
    std::ostringstream ss;
    ss << "Pipeline complete: "
       << impl_->lastStats.totalFramesProcessed << " frames, "
       << impl_->lastStats.trackedPersons << " persons, "
       << impl_->lastStats.clipsProduced << " clips, "
       << impl_->lastStats.segmentsFound << " segments | "
       << "Timings: pose=" << static_cast<int>(impl_->lastStats.poseExtractionMs) << "ms, "
       << "skeleton=" << static_cast<int>(impl_->lastStats.skeletonMappingMs) << "ms, "
       << "signal=" << static_cast<int>(impl_->lastStats.signalProcessingMs) << "ms, "
       << "segment=" << static_cast<int>(impl_->lastStats.segmentationMs) << "ms, "
       << "export=" << static_cast<int>(impl_->lastStats.exportMs) << "ms, "
       << "total=" << static_cast<int>(impl_->lastStats.totalMs) << "ms";
    HM_LOG_INFO(TAG, ss.str());

    if (callback) callback(100.0f, "Complete");

    return clips;
}

} // namespace hm
