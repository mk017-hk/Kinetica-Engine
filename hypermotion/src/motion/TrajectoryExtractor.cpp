#include "HyperMotion/motion/TrajectoryExtractor.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>

namespace hm::motion {

static constexpr const char* TAG = "TrajectoryExtractor";

struct TrajectoryExtractor::Impl {
    TrajectoryExtractorConfig config;

    // Compute smoothed root velocity at a frame by averaging over a window.
    Vec3 smoothedVelocity(const std::vector<SkeletonFrame>& frames,
                          int frameIdx) const {
        int halfWin = config.velocitySmoothingWindow / 2;
        int start = std::max(1, frameIdx - halfWin);
        int end = std::min(static_cast<int>(frames.size()) - 1,
                           frameIdx + halfWin);

        if (start > end) {
            return frames[frameIdx].rootVelocity;
        }

        Vec3 sum{};
        int count = 0;
        float dt = 1.0f / config.fps;

        for (int i = start; i <= end; ++i) {
            Vec3 delta = frames[i].rootPosition - frames[i - 1].rootPosition;
            sum += delta * (1.0f / dt);
            count++;
        }

        if (count == 0) return frames[frameIdx].rootVelocity;
        return sum * (1.0f / static_cast<float>(count));
    }

    // Compute facing angle from root rotation (Y-axis rotation, radians).
    float facingAngle(const SkeletonFrame& frame) const {
        // Extract forward direction from root rotation (Z-forward convention)
        Vec3 forward = frame.rootRotation.rotate({0.0f, 0.0f, 1.0f});
        return std::atan2(forward.x, forward.z);
    }

    // Compute angular velocity (turn rate) in rad/s.
    float turnRate(const std::vector<SkeletonFrame>& frames, int frameIdx) const {
        if (frameIdx <= 0) return 0.0f;
        float angle0 = facingAngle(frames[frameIdx - 1]);
        float angle1 = facingAngle(frames[frameIdx]);
        float diff = angle1 - angle0;
        // Wrap to [-pi, pi]
        while (diff > static_cast<float>(M_PI)) diff -= 2.0f * static_cast<float>(M_PI);
        while (diff < -static_cast<float>(M_PI)) diff += 2.0f * static_cast<float>(M_PI);
        return diff * config.fps;
    }

    // Generate trajectory points for a single frame using constant-velocity
    // extrapolation with turn rate.
    std::vector<TrajectoryPoint> predictTrajectory(
        const std::vector<SkeletonFrame>& frames, int frameIdx) const {

        Vec3 vel = smoothedVelocity(frames, frameIdx);
        float facing = facingAngle(frames[frameIdx]);
        float omega = turnRate(frames, frameIdx);
        Vec3 pos = frames[frameIdx].rootPosition;

        std::vector<TrajectoryPoint> points;
        float dt = config.predictionStep;

        for (float t = dt; t <= config.predictionHorizon + 1e-4f; t += dt) {
            TrajectoryPoint pt;
            pt.deltaTime = t;

            // Predict facing angle
            float predictedFacing = facing + omega * t;

            // Rotate velocity by the accumulated turn
            float dAngle = omega * t;
            float cosA = std::cos(dAngle);
            float sinA = std::sin(dAngle);
            Vec3 rotatedVel = {
                vel.x * cosA - vel.z * sinA,
                vel.y,
                vel.x * sinA + vel.z * cosA
            };

            // Constant-velocity position prediction
            pt.position = pos + rotatedVel * t;
            pt.velocity = rotatedVel;
            pt.facing = predictedFacing;

            points.push_back(pt);
        }

        return points;
    }
};

TrajectoryExtractor::TrajectoryExtractor(const TrajectoryExtractorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

TrajectoryExtractor::~TrajectoryExtractor() = default;
TrajectoryExtractor::TrajectoryExtractor(TrajectoryExtractor&&) noexcept = default;
TrajectoryExtractor& TrajectoryExtractor::operator=(TrajectoryExtractor&&) noexcept = default;

std::vector<std::vector<TrajectoryPoint>> TrajectoryExtractor::extract(
    const std::vector<SkeletonFrame>& frames) const {

    int numFrames = static_cast<int>(frames.size());
    std::vector<std::vector<TrajectoryPoint>> trajectories(numFrames);

    for (int f = 0; f < numFrames; ++f) {
        trajectories[f] = impl_->predictTrajectory(frames, f);
    }

    HM_LOG_DEBUG(TAG, "Extracted trajectories for " +
                 std::to_string(numFrames) + " frames (" +
                 std::to_string(trajectories.empty() ? 0 : trajectories[0].size()) +
                 " points per frame)");
    return trajectories;
}

void TrajectoryExtractor::process(AnimClip& clip) const {
    if (clip.frames.empty()) return;
    clip.trajectories = extract(clip.frames);
    HM_LOG_INFO(TAG, "Processed trajectory for clip '" + clip.name +
                "' (" + std::to_string(clip.frames.size()) + " frames)");
}

} // namespace hm::motion
