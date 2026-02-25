#include "HyperMotion/signal/QuaternionSmoother.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <numeric>

namespace hm::signal {

static constexpr const char* TAG = "QuaternionSmoother";
static constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
static constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
static constexpr float kEpsilon = 1e-7f;

QuaternionSmoother::QuaternionSmoother(const QuaternionSmootherConfig& config)
    : config_(config) {
    // Clamp smoothing factors to valid range [0, 1]
    config_.smoothingFactor = std::clamp(config_.smoothingFactor, 0.0f, 1.0f);
    config_.rootSmoothingFactor = std::clamp(config_.rootSmoothingFactor, 0.0f, 1.0f);

    for (int j = 0; j < JOINT_COUNT; ++j) {
        if (config_.perJointSmoothing[j] != 0.0f) {
            config_.perJointSmoothing[j] = std::clamp(config_.perJointSmoothing[j], 0.0f, 1.0f);
        }
    }

    if (config_.temporalWindow < 1) {
        config_.temporalWindow = 1;
    }
}

QuaternionSmoother::~QuaternionSmoother() = default;

float QuaternionSmoother::getJointSmoothing(int jointIndex) const {
    if (jointIndex >= 0 && jointIndex < JOINT_COUNT) {
        float perJoint = config_.perJointSmoothing[jointIndex];
        if (perJoint > 0.0f) {
            return perJoint;
        }
    }
    return config_.smoothingFactor;
}

Quat QuaternionSmoother::safeNormalize(const Quat& q) {
    float n = q.norm();
    if (n < kEpsilon) {
        return Quat::identity();
    }
    return Quat{q.w / n, q.x / n, q.y / n, q.z / n};
}

float QuaternionSmoother::angularDistanceDeg(const Quat& a, const Quat& b) {
    // The angular distance between two unit quaternions:
    // angle = 2 * acos(|dot(a, b)|)
    float d = std::abs(a.dot(b));
    d = std::clamp(d, 0.0f, 1.0f);
    return 2.0f * std::acos(d) * kRadToDeg;
}

void QuaternionSmoother::fixDoubleCover(std::vector<Quat>& quaternions) {
    if (quaternions.size() < 2) return;

    // Quaternions q and -q represent the same rotation.
    // To ensure smooth interpolation, we pick the hemisphere that minimises
    // the dot product flip (ensuring shortest path between consecutive frames).
    for (size_t i = 1; i < quaternions.size(); ++i) {
        float dot = quaternions[i].dot(quaternions[i - 1]);
        if (dot < 0.0f) {
            // Negate to stay on the same hemisphere
            quaternions[i] = Quat{
                -quaternions[i].w,
                -quaternions[i].x,
                -quaternions[i].y,
                -quaternions[i].z
            };
        }
    }
}

void QuaternionSmoother::clampAngularVelocity(std::vector<Quat>& quaternions, float maxDegPerFrame) {
    if (quaternions.size() < 2 || maxDegPerFrame <= 0.0f) return;

    for (size_t i = 1; i < quaternions.size(); ++i) {
        float angleDeg = angularDistanceDeg(quaternions[i - 1], quaternions[i]);

        if (angleDeg > maxDegPerFrame) {
            // Clamp by interpolating from previous towards current, but only up to the limit
            float t = maxDegPerFrame / angleDeg;
            t = std::clamp(t, 0.0f, 1.0f);
            quaternions[i] = MathUtils::safeSlerp(quaternions[i - 1], quaternions[i], 1.0f - t);
            quaternions[i] = safeNormalize(quaternions[i]);
        }
    }
}

void QuaternionSmoother::windowedSmoothing(std::vector<Quat>& quaternions, int windowSize, float alpha) {
    if (quaternions.size() < 2 || windowSize < 2) return;

    int n = static_cast<int>(quaternions.size());
    int halfWin = windowSize / 2;
    std::vector<Quat> smoothed(n);

    for (int i = 0; i < n; ++i) {
        // Collect quaternions within the window
        int wStart = std::max(0, i - halfWin);
        int wEnd = std::min(n - 1, i + halfWin);
        int count = wEnd - wStart + 1;

        // Use iterative SLERP to compute a weighted average quaternion.
        // The centre sample gets the most weight (Gaussian-like weighting).
        // We use incremental SLERP: start from the centre, blend in neighbours.
        Quat avg = quaternions[i];

        if (count > 1) {
            // Compute weights: Gaussian-ish triangle weighting centred on i
            float totalWeight = 1.0f;

            for (int k = wStart; k <= wEnd; ++k) {
                if (k == i) continue;

                float dist = static_cast<float>(std::abs(k - i));
                // Triangle weight: weight decreases linearly with distance
                float w = 1.0f - (dist / static_cast<float>(halfWin + 1));
                w *= alpha; // Scale by overall smoothing factor
                w = std::clamp(w, 0.0f, 1.0f);

                if (w < kEpsilon) continue;

                // Ensure shortest path
                Quat neighbour = quaternions[k];
                if (neighbour.dot(avg) < 0.0f) {
                    neighbour = Quat{-neighbour.w, -neighbour.x, -neighbour.y, -neighbour.z};
                }

                // Incremental SLERP: blend neighbour into running average
                float blendFactor = w / (totalWeight + w);
                avg = MathUtils::safeSlerp(avg, neighbour, blendFactor);
                totalWeight += w;
            }
        }

        smoothed[i] = safeNormalize(avg);
    }

    quaternions = smoothed;
}

void QuaternionSmoother::smoothQuatSequence(std::vector<Quat>& quaternions, float alpha) {
    if (quaternions.size() < 2) return;

    // Step 1: Fix double-cover to ensure consistent hemisphere
    fixDoubleCover(quaternions);

    // Step 2: Forward pass - exponential moving average with SLERP
    std::vector<Quat> smoothed = quaternions;
    for (size_t i = 1; i < smoothed.size(); ++i) {
        // SLERP from current towards previous by alpha
        // alpha = 0 means no smoothing (keep current), alpha = 1 means full lag
        smoothed[i] = MathUtils::safeSlerp(smoothed[i], smoothed[i - 1], alpha);
        smoothed[i] = safeNormalize(smoothed[i]);
    }

    // Step 3: Backward pass for symmetric (zero-lag) smoothing
    for (int i = static_cast<int>(smoothed.size()) - 2; i >= 0; --i) {
        smoothed[i] = MathUtils::safeSlerp(smoothed[i], smoothed[i + 1], alpha);
        smoothed[i] = safeNormalize(smoothed[i]);
    }

    quaternions = smoothed;
}

void QuaternionSmoother::process(std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    if (numFrames < 2) {
        HM_LOG_DEBUG(TAG, "Skipping quaternion smoother: only " + std::to_string(numFrames) + " frames");
        return;
    }

    HM_LOG_INFO(TAG, "Smoothing quaternions for " + std::to_string(numFrames) + " frames "
                "(factor=" + std::to_string(config_.smoothingFactor) +
                ", temporal_window=" + std::to_string(config_.temporalWindow) + ")");

    int totalJointsSmoothed = 0;
    float totalAngularCorrection = 0.0f;

    // -------------------------------------------------------------------
    // Smooth each joint's rotation sequence with per-joint smoothing factors
    // -------------------------------------------------------------------
    for (int j = 0; j < JOINT_COUNT; ++j) {
        float jointAlpha = getJointSmoothing(j);

        // Extract rotation sequence for this joint
        std::vector<Quat> rotSequence(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rotSequence[f] = frames[f].joints[j].localRotation;
        }

        // Measure initial roughness (average angular velocity)
        float initialRoughness = 0.0f;
        for (int f = 1; f < numFrames; ++f) {
            initialRoughness += angularDistanceDeg(rotSequence[f - 1], rotSequence[f]);
        }
        initialRoughness /= static_cast<float>(numFrames - 1);

        // Step 1: Fix double-cover across the sequence
        fixDoubleCover(rotSequence);

        // Step 2: Apply angular velocity clamping if configured
        if (config_.maxAngularVelocityDegPerFrame > 0.0f) {
            clampAngularVelocity(rotSequence, config_.maxAngularVelocityDegPerFrame);
        }

        // Step 3: Apply smoothing
        if (config_.temporalWindow > 1) {
            // Windowed smoothing (uses iterative SLERP over a local window)
            windowedSmoothing(rotSequence, config_.temporalWindow, jointAlpha);
        } else {
            // Standard forward-backward EMA smoothing
            smoothQuatSequence(rotSequence, jointAlpha);
        }

        // Measure final roughness
        float finalRoughness = 0.0f;
        for (int f = 1; f < numFrames; ++f) {
            finalRoughness += angularDistanceDeg(rotSequence[f - 1], rotSequence[f]);
        }
        finalRoughness /= static_cast<float>(numFrames - 1);

        totalAngularCorrection += (initialRoughness - finalRoughness);

        // Write back
        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[j].localRotation = rotSequence[f];
        }

        totalJointsSmoothed++;
    }

    // -------------------------------------------------------------------
    // Smooth root rotation with its own factor
    // -------------------------------------------------------------------
    {
        std::vector<Quat> rootRots(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rootRots[f] = frames[f].rootRotation;
        }

        fixDoubleCover(rootRots);

        if (config_.maxAngularVelocityDegPerFrame > 0.0f) {
            // Root rotation typically changes slower, use a stricter limit
            clampAngularVelocity(rootRots, config_.maxAngularVelocityDegPerFrame * 0.5f);
        }

        if (config_.temporalWindow > 1) {
            windowedSmoothing(rootRots, config_.temporalWindow, config_.rootSmoothingFactor);
        } else {
            smoothQuatSequence(rootRots, config_.rootSmoothingFactor);
        }

        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootRotation = rootRots[f];
        }
    }

    // -------------------------------------------------------------------
    // Update derived rotation representations (6D and Euler) from smoothed quaternions
    // -------------------------------------------------------------------
    if (config_.updateDerivedRotations) {
        for (int f = 0; f < numFrames; ++f) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                auto& jt = frames[f].joints[j];
                jt.rotation6D = MathUtils::quatToRot6D(jt.localRotation);
                jt.localEulerDeg = MathUtils::quatToEulerDeg(jt.localRotation);
            }
        }
    }

    // -------------------------------------------------------------------
    // Final continuity check: verify no large jumps remain
    // -------------------------------------------------------------------
    int discontinuities = 0;
    float maxJump = 0.0f;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        for (int f = 1; f < numFrames; ++f) {
            float angleDeg = angularDistanceDeg(
                frames[f - 1].joints[j].localRotation,
                frames[f].joints[j].localRotation);
            if (angleDeg > maxJump) maxJump = angleDeg;
            if (angleDeg > 45.0f) { // Flag anything > 45 degrees per frame as a discontinuity
                discontinuities++;
            }
        }
    }

    if (discontinuities > 0) {
        HM_LOG_WARN(TAG, "Post-smoothing: " + std::to_string(discontinuities) +
                    " rotation discontinuities (>45 deg/frame) remain, max jump: " +
                    std::to_string(maxJump) + " deg");
    }

    float avgCorrection = totalAngularCorrection / static_cast<float>(totalJointsSmoothed);
    HM_LOG_INFO(TAG, "Quaternion smoothing complete: " + std::to_string(totalJointsSmoothed) +
                " joints processed, avg roughness reduction: " +
                std::to_string(avgCorrection) + " deg/frame");
}

} // namespace hm::signal
