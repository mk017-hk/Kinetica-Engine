#include "HyperMotion/dataset/MatchAnalyser.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/ScopedTimer.h"
#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/signal/FootContactFilter.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"

#include <set>
#include <sstream>

namespace hm::dataset {

static constexpr const char* TAG = "MatchAnalyser";

struct MatchAnalyser::Impl {
    MatchAnalyserConfig config;

    // Pipeline components
    pose::MultiPersonPoseEstimator poseEstimator;
    tracking::MultiPlayerTracker tracker;
    skeleton::SkeletonMapper skeletonMapper;
    signal::SignalPipeline signalPipeline;
    segmenter::MotionSegmenter segmenter;
    signal::FootContactFilter footContact;

    ClipExtractor clipExtractor;
    ClipQualityFilter qualityFilter;
    MotionClassifier classifier;
    AnimationDatabase database;

    bool initialized = false;

    Impl(const MatchAnalyserConfig& cfg)
        : config(cfg)
        , poseEstimator(cfg.pipelineConfig.poseConfig)
        , tracker(cfg.trackerConfig)
        , skeletonMapper(cfg.pipelineConfig.mapperConfig)
        , signalPipeline(cfg.pipelineConfig.signalConfig)
        , segmenter(cfg.pipelineConfig.segmenterConfig)
        , footContact(cfg.pipelineConfig.signalConfig.footConfig)
        , clipExtractor(cfg.clipConfig)
        , qualityFilter(cfg.qualityConfig)
        , classifier(cfg.classifierModelPath) {}
};

MatchAnalyser::MatchAnalyser(const MatchAnalyserConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MatchAnalyser::~MatchAnalyser() = default;
MatchAnalyser::MatchAnalyser(MatchAnalyser&&) noexcept = default;
MatchAnalyser& MatchAnalyser::operator=(MatchAnalyser&&) noexcept = default;

bool MatchAnalyser::initialize() {
    HM_LOG_INFO(TAG, "Initializing match analyser...");

    if (!impl_->poseEstimator.initialize()) {
        HM_LOG_ERROR(TAG, "Pose estimator init failed");
        return false;
    }

    impl_->segmenter.initialize();  // optional — heuristic fallback
    impl_->classifier.initialize(); // optional — heuristic fallback

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Match analyser ready");
    return true;
}

MatchAnalysisResult MatchAnalyser::processMatch(
    const std::string& videoPath, MatchProgressCallback callback) {

    MatchAnalysisResult result;
    double totalMs = 0;
    ScopedTimer totalTimer(totalMs);

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Not initialized");
        return result;
    }

    HM_LOG_INFO(TAG, "Processing match: " + videoPath);

    // ----------------------------------------------------------------
    // Phase 1: Pose estimation (0% - 40%)
    // ----------------------------------------------------------------
    if (callback) callback(0.0f, "Extracting poses from video...");

    double poseMs = 0;
    std::vector<PoseFrameResult> poseResults;
    {
        ScopedTimer t(poseMs);
        auto poseCB = [&callback](float pct, const std::string& msg) {
            if (callback) callback(pct * 0.4f, "Pose: " + msg);
        };
        poseResults = impl_->poseEstimator.processVideo(videoPath, poseCB);
    }
    result.totalFramesDecoded = static_cast<int>(poseResults.size());
    result.pipelineStats.poseExtractionMs = poseMs;

    HM_LOG_INFO(TAG, "Extracted poses: " + std::to_string(poseResults.size()) + " frames");

    // ----------------------------------------------------------------
    // Phase 2: Multi-player tracking (40% - 50%)
    // ----------------------------------------------------------------
    if (callback) callback(40.0f, "Tracking players...");

    double trackMs = 0;
    std::vector<tracking::TrackedFrame> trackedFrames;
    {
        ScopedTimer t(trackMs);
        auto trackCB = [&callback](float pct, const std::string& msg) {
            if (callback) callback(40.0f + pct * 0.1f, "Tracking: " + msg);
        };
        trackedFrames = impl_->tracker.processAll(poseResults, trackCB);
    }
    result.totalPlayersTracked = impl_->tracker.totalPlayersTracked();

    HM_LOG_INFO(TAG, "Tracked " + std::to_string(result.totalPlayersTracked) + " unique players");

    // ----------------------------------------------------------------
    // Phase 3: Per-player skeleton mapping + signal processing (50% - 70%)
    // ----------------------------------------------------------------
    if (callback) callback(50.0f, "Building skeleton sequences...");

    // Collect per-player frame sequences from pose results
    std::set<int> playerIDs;
    std::unordered_map<int, std::vector<int>> playerFrameIndices;
    for (const auto& tf : trackedFrames) {
        for (const auto& tp : tf.players) {
            playerIDs.insert(tp.persistentID);
            playerFrameIndices[tp.persistentID].push_back(tf.frameIndex);
        }
    }

    struct PlayerData {
        int playerID;
        std::vector<SkeletonFrame> skeletonFrames;
        std::vector<MotionSegment> segments;
        std::vector<signal::FootContactFilter::ContactState> contacts;
    };
    std::vector<PlayerData> playerDatas;

    int playerIdx = 0;
    int totalPlayers = static_cast<int>(playerIDs.size());

    double skelMs = 0, sigMs = 0, segMs = 0;

    for (int pid : playerIDs) {
        PlayerData pd;
        pd.playerID = pid;

        // Map skeleton
        {
            ScopedTimer t(skelMs);
            pd.skeletonFrames = impl_->skeletonMapper.mapSequence(poseResults, pid);
        }

        if (pd.skeletonFrames.size() < 15) {
            playerIdx++;
            continue;  // skip very short tracks
        }

        // Signal processing
        {
            ScopedTimer t(sigMs);
            impl_->signalPipeline.process(pd.skeletonFrames);
        }

        // Foot contact detection
        pd.contacts = impl_->footContact.detectContacts(pd.skeletonFrames);

        // Motion segmentation
        {
            ScopedTimer t(segMs);
            pd.segments = impl_->segmenter.segment(pd.skeletonFrames, pid);
        }

        playerDatas.push_back(std::move(pd));
        playerIdx++;

        if (callback && totalPlayers > 0) {
            float pct = 50.0f + (static_cast<float>(playerIdx) / totalPlayers) * 20.0f;
            callback(pct, "Processing player " + std::to_string(playerIdx) +
                    "/" + std::to_string(totalPlayers));
        }
    }

    result.pipelineStats.skeletonMappingMs = skelMs;
    result.pipelineStats.signalProcessingMs = sigMs;
    result.pipelineStats.segmentationMs = segMs;

    // ----------------------------------------------------------------
    // Phase 4: Clip extraction + quality filter + classification (70% - 90%)
    // ----------------------------------------------------------------
    if (callback) callback(70.0f, "Extracting and classifying clips...");

    int totalExtracted = 0;
    int totalAccepted = 0;
    int totalRejected = 0;

    for (size_t i = 0; i < playerDatas.size(); ++i) {
        const auto& pd = playerDatas[i];

        // Extract clips using motion segments
        auto extraction = impl_->clipExtractor.extractFromSegments(
            pd.skeletonFrames, pd.segments, pd.playerID);

        totalExtracted += static_cast<int>(extraction.clips.size());

        // Quality filter
        auto filtered = impl_->qualityFilter.filter(extraction.clips);
        totalRejected += static_cast<int>(filtered.rejected.size());

        // Classify accepted clips
        auto classifications = impl_->classifier.classifyBatch(filtered.accepted);

        // Build database entries
        for (size_t j = 0; j < filtered.accepted.size(); ++j) {
            AnimationEntry entry;
            entry.clip = std::move(filtered.accepted[j]);
            entry.quality = filtered.acceptedResults[j];
            entry.classification = classifications[j];

            // Find matching metadata
            // Search original extraction metadata by clip name
            for (size_t k = 0; k < extraction.clips.size(); ++k) {
                if (k < extraction.metadata.size()) {
                    auto& meta = extraction.metadata[k];
                    if (meta.playerID == pd.playerID) {
                        entry.clipMeta = meta;
                        entry.clipMeta.motionType = classifications[j].label;
                        entry.clipMeta.confidence = classifications[j].confidence;
                        break;
                    }
                }
            }

            // Attach foot contacts for the clip's frame range
            int startF = entry.clipMeta.startFrame;
            int endF = std::min(entry.clipMeta.endFrame,
                               static_cast<int>(pd.contacts.size()) - 1);
            for (int f = startF; f <= endF && f < static_cast<int>(pd.contacts.size()); ++f) {
                entry.footContacts.push_back(pd.contacts[f]);
            }
            entry.clipMeta.hasFootContacts = !entry.footContacts.empty();

            impl_->database.addEntry(std::move(entry));
            totalAccepted++;
        }

        if (callback) {
            float pct = 70.0f + (static_cast<float>(i + 1) / playerDatas.size()) * 20.0f;
            callback(pct, "Clips: player " + std::to_string(i + 1) +
                    "/" + std::to_string(playerDatas.size()));
        }
    }

    result.clipsExtracted = totalExtracted;
    result.clipsAccepted = totalAccepted;
    result.clipsRejected = totalRejected;

    // ----------------------------------------------------------------
    // Phase 5: Export (90% - 100%)
    // ----------------------------------------------------------------
    if (callback) callback(90.0f, "Exporting animation database...");

    if (!impl_->config.outputDirectory.empty()) {
        double exportMs = 0;
        {
            ScopedTimer t(exportMs);
            impl_->database.exportToDirectory(
                impl_->config.outputDirectory,
                impl_->config.exportBVH,
                impl_->config.exportJSON);
            impl_->database.saveSummary(
                impl_->config.outputDirectory + "/database_summary.json");
        }
        result.pipelineStats.exportMs = exportMs;
    }

    result.dbStats = impl_->database.stats();
    result.totalProcessingMs = totalTimer.elapsedMs();
    result.pipelineStats.totalMs = result.totalProcessingMs;

    // Log summary
    std::ostringstream ss;
    ss << "Match analysis complete: "
       << result.totalFramesDecoded << " frames, "
       << result.totalPlayersTracked << " players, "
       << result.clipsExtracted << " clips extracted, "
       << result.clipsAccepted << " accepted, "
       << result.clipsRejected << " rejected | "
       << "Time: " << static_cast<int>(result.totalProcessingMs) << "ms";
    HM_LOG_INFO(TAG, ss.str());

    if (callback) callback(100.0f, "Complete");
    return result;
}

const AnimationDatabase& MatchAnalyser::database() const {
    return impl_->database;
}

} // namespace hm::dataset
