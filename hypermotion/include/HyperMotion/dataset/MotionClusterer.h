#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <memory>
#include <string>

namespace hm::dataset {

/// Configuration for motion clustering.
struct MotionClustererConfig {
    int numClusters = 8;              // k for k-means
    int maxIterations = 100;          // max k-means iterations
    float convergenceThreshold = 1e-4f; // centroid movement threshold
    int randomSeed = 42;             // for reproducible initialization
    float fps = 30.0f;               // for feature extraction
};

/// Feature vector extracted from an animation clip for clustering.
struct ClipFeatures {
    float avgVelocity = 0.0f;        // cm/s
    float maxVelocity = 0.0f;        // cm/s
    float avgTurnRate = 0.0f;        // deg/s
    float strideFrequency = 0.0f;    // Hz (estimated from foot oscillation)
    float avgKneeBend = 0.0f;        // degrees
    float avgHipRotation = 0.0f;     // degrees
    float avgArmSwing = 0.0f;        // degrees
    float verticalRange = 0.0f;      // cm (root Y range)
};

/// Result for a single cluster.
struct ClusterInfo {
    int clusterID = -1;
    int memberCount = 0;
    ClipFeatures centroid;
    std::string label;                // auto-generated: "cluster_01", etc.
};

/// Result of clustering a set of clips.
struct ClusteringResult {
    std::vector<int> assignments;     // cluster ID per clip (parallel to input)
    std::vector<ClusterInfo> clusters;
    int numIterations = 0;
    float totalInertia = 0.0f;        // sum of squared distances to centroids
};

/// Automatically discovers motion types by clustering animation clips
/// based on motion features (velocity, turn rate, stride frequency,
/// joint angles).
///
/// Uses k-means clustering. Each clip is assigned a cluster label
/// (e.g. cluster_01, cluster_02, ...) that represents a discovered
/// motion category.
class MotionClusterer {
public:
    explicit MotionClusterer(const MotionClustererConfig& config = {});
    ~MotionClusterer();

    MotionClusterer(const MotionClusterer&) = delete;
    MotionClusterer& operator=(const MotionClusterer&) = delete;
    MotionClusterer(MotionClusterer&&) noexcept;
    MotionClusterer& operator=(MotionClusterer&&) noexcept;

    /// Extract motion features from a single clip.
    ClipFeatures extractFeatures(const AnimClip& clip) const;

    /// Cluster a set of clips. Returns assignments and cluster info.
    ClusteringResult cluster(const std::vector<AnimClip>& clips) const;

    /// Cluster clips and write clusterID into each clip.
    void process(std::vector<AnimClip>& clips) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::dataset
