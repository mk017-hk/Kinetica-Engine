#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/pose/MultiPersonPoseEstimator.h"
#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/export/AnimClipUtils.h"
#include "HyperMotion/motion/FootContactDetector.h"
#include "HyperMotion/motion/TrajectoryExtractor.h"
#include "HyperMotion/dataset/MotionClusterer.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace hm {

struct PipelineConfig {
    pose::MultiPersonPoseConfig poseConfig;
    skeleton::SkeletonMapperConfig mapperConfig;
    signal::SignalPipelineConfig signalConfig;
    segmenter::MotionSegmenterConfig segmenterConfig;
    xport::BVHExportConfig bvhConfig;
    xport::JSONExportConfig jsonConfig;
    motion::FootContactDetectorConfig footContactConfig;
    motion::TrajectoryExtractorConfig trajectoryConfig;
    dataset::MotionClustererConfig clusterConfig;

    bool enableFootContactDetection = true;
    bool enableTrajectoryExtraction = true;
    bool enableMotionClustering = true;

    float targetFPS = 30.0f;
    bool enableVisualization = false;
    bool splitBySegment = true;
    std::string outputDirectory;
    std::string outputFormat = "json";  // "json", "bvh", "both"
    int minTrackFrames = 10;            // Minimum frames to keep a track
};

/// Timing and statistics from a pipeline run.
struct PipelineStats {
    double poseExtractionMs = 0.0;
    double skeletonMappingMs = 0.0;
    double signalProcessingMs = 0.0;
    double segmentationMs = 0.0;
    double footContactMs = 0.0;
    double trajectoryMs = 0.0;
    double clusteringMs = 0.0;
    double exportMs = 0.0;
    double totalMs = 0.0;
    int totalFramesProcessed = 0;
    int trackedPersons = 0;
    int clipsProduced = 0;
    int segmentsFound = 0;
};

using PipelineProgressCallback = std::function<void(float percent, const std::string& stage)>;

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    // Full pipeline: video -> animation clips
    std::vector<AnimClip> processVideo(const std::string& videoPath,
                                        PipelineProgressCallback callback = nullptr);

    // Step-by-step API
    std::vector<PoseFrameResult> extractPoses(const std::string& videoPath,
                                               PipelineProgressCallback callback = nullptr);
    std::vector<AnimClip> buildClips(const std::vector<PoseFrameResult>& poseResults);
    void exportClips(const std::vector<AnimClip>& clips, const std::string& outputDir);

    /// Get timing statistics from the last processVideo() call.
    const PipelineStats& getLastStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm
