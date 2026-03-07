#pragma once

#include "HyperMotion/core/Types.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hm::tracking {

/// Manages persistent player IDs across a match, including re-identification
/// after temporary occlusion.  Works on top of the frame-level PoseTracker
/// by maintaining a long-term identity pool.
struct PlayerIdentity {
    int persistentID = -1;
    std::string label;               // "player", "referee", "goalkeeper"
    int teamID = -1;                 // -1 = unknown
    int totalFramesSeen = 0;
    int lastSeenFrame = -1;
    int firstSeenFrame = -1;
    std::array<float, REID_DIM> reidTemplate{};  // running average
    float avgConfidence = 0.0f;
};

struct PlayerIDManagerConfig {
    int maxPlayers = 22;
    float reidMatchThreshold = 0.6f;     // cosine similarity threshold
    int maxLostFrames = 150;             // 5 seconds at 30fps before dropping
    float reidTemplateAlpha = 0.05f;     // EMA update rate for ReID template
};

class PlayerIDManager {
public:
    explicit PlayerIDManager(const PlayerIDManagerConfig& config = {});
    ~PlayerIDManager();

    PlayerIDManager(const PlayerIDManager&) = delete;
    PlayerIDManager& operator=(const PlayerIDManager&) = delete;
    PlayerIDManager(PlayerIDManager&&) noexcept;
    PlayerIDManager& operator=(PlayerIDManager&&) noexcept;

    /// Update with confirmed tracklets from PoseTracker.
    /// Returns mapping: tracklet ID -> persistent player ID.
    std::unordered_map<int, int> update(
        const std::vector<DetectedPerson>& persons, int frameIndex);

    /// Get all known player identities.
    std::vector<PlayerIdentity> getActiveIdentities() const;

    /// Get identity for a specific persistent ID.
    const PlayerIdentity* getIdentity(int persistentID) const;

    /// Total unique players seen so far.
    int totalPlayersTracked() const;

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::tracking
