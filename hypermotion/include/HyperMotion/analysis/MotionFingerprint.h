#pragma once

#include "HyperMotion/core/Types.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::analysis {

/// Feature vector summarising a single animation clip.
/// Used for similarity comparison, clustering, and dataset organisation.
struct FingerprintFeatures {
    // Locomotion
    float avgVelocity = 0.0f;         // cm/s
    float peakVelocity = 0.0f;        // cm/s
    float avgAcceleration = 0.0f;     // cm/s^2

    // Turning
    float avgTurnRate = 0.0f;         // deg/s
    float peakTurnRate = 0.0f;        // deg/s

    // Stride analysis
    float strideLength = 0.0f;        // cm (estimated from foot separation peaks)
    float strideFrequency = 0.0f;     // Hz

    // Joint statistics
    float avgKneeBend = 0.0f;         // degrees
    float avgHipRotation = 0.0f;      // degrees
    float avgArmSwing = 0.0f;         // degrees
    float avgSpineFlexion = 0.0f;     // degrees
    float verticalRange = 0.0f;       // cm (root Y range — jumps vs ground)
    float avgHeadStability = 0.0f;    // degrees (head vs torso relative motion)

    // Temporal
    float clipDurationSec = 0.0f;     // seconds
    int frameCount = 0;

    // Foot contact ratios
    float leftFootContactRatio = 0.0f;  // fraction of frames in contact
    float rightFootContactRatio = 0.0f;

    /// Convert to a flat vector for distance computations.
    static constexpr int DIM = 18;
    std::array<float, DIM> toVector() const;

    /// Euclidean distance to another fingerprint.
    float distanceTo(const FingerprintFeatures& other) const;
};

/// Configuration for the fingerprinting module.
struct MotionFingerprintConfig {
    float fps = 30.0f;
    float groundHeight = 0.0f;        // Y coordinate of ground plane

    // Velocity filtering
    int velocitySmoothWindow = 5;     // frames for velocity averaging

    // Stride estimation
    float minStridePeakHeight = 5.0f; // cm — minimum foot separation for a stride peak
};

/// Computes feature vectors (fingerprints) for animation clips.
///
/// Each fingerprint encodes locomotion dynamics, joint kinematics, and
/// temporal characteristics as a compact feature vector.  These fingerprints
/// enable:
///   - Similarity-based clip retrieval
///   - Automatic motion type labelling
///   - Dataset balancing and curation
class MotionFingerprint {
public:
    explicit MotionFingerprint(const MotionFingerprintConfig& config = {});
    ~MotionFingerprint();

    MotionFingerprint(const MotionFingerprint&) = delete;
    MotionFingerprint& operator=(const MotionFingerprint&) = delete;
    MotionFingerprint(MotionFingerprint&&) noexcept;
    MotionFingerprint& operator=(MotionFingerprint&&) noexcept;

    /// Compute fingerprint for a single animation clip.
    FingerprintFeatures compute(const AnimClip& clip) const;

    /// Compute fingerprints for a batch of clips.
    std::vector<FingerprintFeatures> computeBatch(
        const std::vector<AnimClip>& clips) const;

    /// Compute fingerprint directly from skeleton frames.
    FingerprintFeatures computeFromFrames(
        const std::vector<SkeletonFrame>& frames, float fps = 30.0f) const;

    /// Find the N most similar clips to a query fingerprint.
    struct SimilarityResult {
        int clipIndex = -1;
        float distance = 0.0f;
    };
    std::vector<SimilarityResult> findSimilar(
        const FingerprintFeatures& query,
        const std::vector<FingerprintFeatures>& database,
        int maxResults = 10) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::analysis
