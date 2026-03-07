#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/tracking/PlayerIDManager.h"
#include "HyperMotion/pose/PoseTracker.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hm::tracking {

struct MultiPlayerTrackerConfig {
    pose::PoseTrackerConfig trackerConfig;
    PlayerIDManagerConfig idConfig;
    int maxPlayers = 22;
};

/// Per-frame tracking result with persistent player IDs.
struct TrackedFrame {
    int frameIndex = 0;
    double timestamp = 0.0;
    struct TrackedPlayer {
        int persistentID = -1;
        int trackletID = -1;
        DetectedPerson detection;
    };
    std::vector<TrackedPlayer> players;
};

using TrackingProgressCallback =
    std::function<void(float percent, const std::string& message)>;

/// High-level tracker that wraps PoseTracker + PlayerIDManager to produce
/// persistent player identities across an entire match.
class MultiPlayerTracker {
public:
    explicit MultiPlayerTracker(const MultiPlayerTrackerConfig& config = {});
    ~MultiPlayerTracker();

    MultiPlayerTracker(const MultiPlayerTracker&) = delete;
    MultiPlayerTracker& operator=(const MultiPlayerTracker&) = delete;
    MultiPlayerTracker(MultiPlayerTracker&&) noexcept;
    MultiPlayerTracker& operator=(MultiPlayerTracker&&) noexcept;

    /// Process a single frame's detections + poses and return tracked players.
    TrackedFrame processFrame(
        const std::vector<Detection>& detections,
        const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& poses,
        int frameIndex, double timestamp);

    /// Process all PoseFrameResults from a video in one pass.
    std::vector<TrackedFrame> processAll(
        const std::vector<PoseFrameResult>& poseResults,
        TrackingProgressCallback callback = nullptr);

    /// Get per-player frame sequences (persistent ID -> ordered frames).
    std::unordered_map<int, std::vector<const TrackedFrame::TrackedPlayer*>>
    getPlayerSequences(const std::vector<TrackedFrame>& frames) const;

    const PlayerIDManager& idManager() const;
    int totalPlayersTracked() const;

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::tracking
