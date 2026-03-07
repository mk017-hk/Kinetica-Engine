#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/Pipeline.h"
#include "HyperMotion/tracking/MultiPlayerTracker.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/dataset/MotionClassifier.h"
#include "HyperMotion/dataset/AnimationDatabase.h"
#include "HyperMotion/analysis/MotionEmbedder.h"
#include <functional>
#include <memory>
#include <string>

namespace hm::dataset {

struct MatchAnalyserConfig {
    PipelineConfig pipelineConfig;
    tracking::MultiPlayerTrackerConfig trackerConfig;
    ClipExtractorConfig clipConfig;
    ClipQualityConfig qualityConfig;

    std::string classifierModelPath;   // optional TCN model
    std::string motionEncoderModelPath; // optional motion encoder ONNX model
    std::string outputDirectory;

    bool exportBVH = true;
    bool exportJSON = true;

    // Async pipeline: number of threads for video decoding.
    // 0 = synchronous (no threading).
    int decodingThreads = 0;
};

struct MatchAnalysisResult {
    DatabaseStats dbStats;
    PipelineStats pipelineStats;
    int totalFramesDecoded = 0;
    int totalPlayersTracked = 0;
    int clipsExtracted = 0;
    int clipsAccepted = 0;
    int clipsRejected = 0;
    double totalProcessingMs = 0.0;
};

using MatchProgressCallback =
    std::function<void(float percent, const std::string& stage)>;

/// Top-level orchestrator that processes a full match video and produces
/// a structured animation database.
///
/// Pipeline:
///   video_ingest → player_detection → multi_player_tracking →
///   pose_estimation → skeleton_mapping → signal_processing →
///   canonical_motion → foot_contact → trajectory_extraction →
///   motion_segmentation → clip_extraction → clip_quality_filter →
///   motion_classification → motion_fingerprint → motion_embedding →
///   animation_database_export
class MatchAnalyser {
public:
    explicit MatchAnalyser(const MatchAnalyserConfig& config);
    ~MatchAnalyser();

    MatchAnalyser(const MatchAnalyser&) = delete;
    MatchAnalyser& operator=(const MatchAnalyser&) = delete;
    MatchAnalyser(MatchAnalyser&&) noexcept;
    MatchAnalyser& operator=(MatchAnalyser&&) noexcept;

    bool initialize();

    /// Process a single match video end-to-end.
    MatchAnalysisResult processMatch(const std::string& videoPath,
                                      MatchProgressCallback callback = nullptr);

    /// Access the built animation database.
    const AnimationDatabase& database() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::dataset
