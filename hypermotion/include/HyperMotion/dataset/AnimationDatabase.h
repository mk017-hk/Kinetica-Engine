#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/dataset/MotionClassifier.h"
#include "HyperMotion/signal/FootContactFilter.h"
#include <string>
#include <vector>
#include <memory>

namespace hm::dataset {

/// Entry in the animation database: a classified, quality-checked clip
/// with all associated metadata.
struct AnimationEntry {
    AnimClip clip;
    ClipMetadata clipMeta;
    ClassificationResult classification;
    QualityResult quality;
    std::vector<signal::FootContactFilter::ContactState> footContacts;

    // Trajectory data (if extracted). One trajectory per frame.
    std::vector<TrajectoryPoint> trajectory;

    // Motion embedding (128D, L2-normalized). Empty if not computed.
    std::array<float, MOTION_EMBEDDING_DIM> motionEmbedding{};
    bool hasMotionEmbedding = false;
};

/// Summary statistics for the database.
struct DatabaseStats {
    int totalClips = 0;
    int totalFrames = 0;
    float totalDurationSec = 0.0f;
    int uniquePlayers = 0;
    std::array<int, MOTION_TYPE_COUNT> clipsByType{};
};

/// Structured animation database that stores accepted clips organised
/// by motion type.  Supports on-disk export to a standard folder
/// structure with BVH + JSON for each clip.
class AnimationDatabase {
public:
    AnimationDatabase();
    ~AnimationDatabase();

    AnimationDatabase(const AnimationDatabase&) = delete;
    AnimationDatabase& operator=(const AnimationDatabase&) = delete;
    AnimationDatabase(AnimationDatabase&&) noexcept;
    AnimationDatabase& operator=(AnimationDatabase&&) noexcept;

    /// Add a single entry.
    void addEntry(AnimationEntry entry);

    /// Add multiple entries.
    void addEntries(std::vector<AnimationEntry> entries);

    /// Get all entries.
    const std::vector<AnimationEntry>& entries() const;

    /// Get entries filtered by motion type.
    std::vector<const AnimationEntry*> entriesByType(MotionType type) const;

    /// Get entries filtered by player ID.
    std::vector<const AnimationEntry*> entriesByPlayer(int playerID) const;

    /// Get database stats.
    DatabaseStats stats() const;

    /// Export the entire database to a folder structure:
    ///   rootDir/walk/clip_0001.bvh + .json
    ///   rootDir/jog/clip_0002.bvh + .json
    ///   ...
    /// Returns number of clips exported.
    int exportToDirectory(const std::string& rootDir,
                          bool exportBVH = true,
                          bool exportJSON = true) const;

    /// Export database summary as JSON.
    std::string exportSummaryJSON() const;

    /// Save summary JSON to file.
    bool saveSummary(const std::string& path) const;

    /// Clear all entries.
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::dataset
