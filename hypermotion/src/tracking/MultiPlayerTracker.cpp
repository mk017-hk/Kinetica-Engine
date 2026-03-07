#include "HyperMotion/tracking/MultiPlayerTracker.h"
#include "HyperMotion/core/Logger.h"
#include <algorithm>

namespace hm::tracking {

static constexpr const char* TAG = "MultiPlayerTracker";

struct MultiPlayerTracker::Impl {
    MultiPlayerTrackerConfig config;
    pose::PoseTracker poseTracker;
    PlayerIDManager idManager;

    Impl(const MultiPlayerTrackerConfig& cfg)
        : config(cfg)
        , poseTracker(cfg.trackerConfig)
        , idManager(cfg.idConfig) {}
};

MultiPlayerTracker::MultiPlayerTracker(const MultiPlayerTrackerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MultiPlayerTracker::~MultiPlayerTracker() = default;
MultiPlayerTracker::MultiPlayerTracker(MultiPlayerTracker&&) noexcept = default;
MultiPlayerTracker& MultiPlayerTracker::operator=(MultiPlayerTracker&&) noexcept = default;

TrackedFrame MultiPlayerTracker::processFrame(
    const std::vector<Detection>& detections,
    const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& poses,
    int frameIndex, double timestamp) {

    // Step 1: Update frame-level tracker (Hungarian assignment)
    impl_->poseTracker.update(detections, poses);

    // Step 2: Get confirmed tracklets
    auto tracklets = impl_->poseTracker.getConfirmedTracklets();

    // Build DetectedPerson list from confirmed tracklets
    std::vector<DetectedPerson> persons;
    persons.reserve(tracklets.size());
    for (const auto& t : tracklets) {
        DetectedPerson p;
        p.id = t.id;
        p.bbox = t.lastDetection.bbox;
        p.classLabel = t.classLabel;
        p.keypoints2D = t.lastPose;
        p.reidFeature = t.reidFeature;
        persons.push_back(p);
    }

    // Step 3: Map tracklet IDs to persistent player IDs
    auto mapping = impl_->idManager.update(persons, frameIndex);

    // Step 4: Build result
    TrackedFrame result;
    result.frameIndex = frameIndex;
    result.timestamp = timestamp;

    for (const auto& person : persons) {
        auto mapIt = mapping.find(person.id);
        if (mapIt == mapping.end()) continue;

        TrackedFrame::TrackedPlayer tp;
        tp.persistentID = mapIt->second;
        tp.trackletID = person.id;
        tp.detection = person;
        result.players.push_back(tp);
    }

    return result;
}

std::vector<TrackedFrame> MultiPlayerTracker::processAll(
    const std::vector<PoseFrameResult>& poseResults,
    TrackingProgressCallback callback) {

    std::vector<TrackedFrame> results;
    results.reserve(poseResults.size());

    int totalFrames = static_cast<int>(poseResults.size());

    for (int i = 0; i < totalFrames; ++i) {
        const auto& frame = poseResults[i];

        // Extract detections and poses from PoseFrameResult
        std::vector<Detection> detections;
        std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>> poses;
        detections.reserve(frame.persons.size());
        poses.reserve(frame.persons.size());

        for (const auto& person : frame.persons) {
            Detection det;
            det.bbox = person.bbox;
            det.classLabel = person.classLabel;
            detections.push_back(det);
            poses.push_back(person.keypoints2D);
        }

        results.push_back(processFrame(
            detections, poses, frame.frameIndex, frame.timestamp));

        if (callback && (i % 100 == 0 || i == totalFrames - 1)) {
            float pct = static_cast<float>(i + 1) / totalFrames * 100.0f;
            callback(pct, "Tracking: frame " + std::to_string(i + 1) +
                    "/" + std::to_string(totalFrames));
        }
    }

    HM_LOG_INFO(TAG, "Tracked " + std::to_string(impl_->idManager.totalPlayersTracked()) +
                " unique players across " + std::to_string(totalFrames) + " frames");
    return results;
}

std::unordered_map<int, std::vector<const TrackedFrame::TrackedPlayer*>>
MultiPlayerTracker::getPlayerSequences(const std::vector<TrackedFrame>& frames) const {
    std::unordered_map<int, std::vector<const TrackedFrame::TrackedPlayer*>> sequences;

    for (const auto& frame : frames) {
        for (const auto& player : frame.players) {
            sequences[player.persistentID].push_back(&player);
        }
    }
    return sequences;
}

const PlayerIDManager& MultiPlayerTracker::idManager() const {
    return impl_->idManager;
}

int MultiPlayerTracker::totalPlayersTracked() const {
    return impl_->idManager.totalPlayersTracked();
}

void MultiPlayerTracker::reset() {
    impl_->poseTracker.reset();
    impl_->idManager.reset();
}

} // namespace hm::tracking
