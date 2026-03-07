#include "HyperMotion/motion/FootContactDetector.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>

namespace hm::motion {

static constexpr const char* TAG = "FootContactDetector";

struct FootContactDetector::Impl {
    FootContactDetectorConfig config;

    // Compute foot velocity (cm/s) via finite difference.
    float footVelocity(const std::vector<SkeletonFrame>& frames,
                       int frameIdx, int jointIdx) const {
        if (frameIdx <= 0) return 0.0f;
        Vec3 delta = frames[frameIdx].joints[jointIdx].worldPosition -
                     frames[frameIdx - 1].joints[jointIdx].worldPosition;
        float dt = 1.0f / config.fps;
        return delta.length() / dt;
    }

    // Compute foot height above ground.
    float footHeight(const SkeletonFrame& frame, int jointIdx) const {
        return frame.joints[jointIdx].worldPosition.y - config.groundHeight;
    }

    // Check raw contact condition for a single foot at a single frame.
    bool rawContact(const std::vector<SkeletonFrame>& frames,
                    int frameIdx, int jointIdx) const {
        float vel = footVelocity(frames, frameIdx, jointIdx);
        float height = footHeight(frames[frameIdx], jointIdx);
        return (vel < config.velocityThreshold) &&
               (height < config.heightThreshold);
    }

    // Apply temporal stability: require stabilityWindow consecutive
    // raw-contact frames for the contact to be confirmed.
    bool stableContact(const std::vector<bool>& rawContacts,
                       int frameIdx) const {
        int halfWin = config.stabilityWindow / 2;
        int start = std::max(0, frameIdx - halfWin);
        int end = std::min(static_cast<int>(rawContacts.size()) - 1,
                           frameIdx + halfWin);

        for (int i = start; i <= end; ++i) {
            if (!rawContacts[i]) return false;
        }
        return true;
    }
};

FootContactDetector::FootContactDetector(const FootContactDetectorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

FootContactDetector::~FootContactDetector() = default;
FootContactDetector::FootContactDetector(FootContactDetector&&) noexcept = default;
FootContactDetector& FootContactDetector::operator=(FootContactDetector&&) noexcept = default;

std::vector<FootContact> FootContactDetector::detect(
    const std::vector<SkeletonFrame>& frames) const {

    int numFrames = static_cast<int>(frames.size());
    std::vector<FootContact> contacts(numFrames);

    if (numFrames < 2) return contacts;

    int leftFootIdx = static_cast<int>(Joint::LeftFoot);
    int rightFootIdx = static_cast<int>(Joint::RightFoot);

    // Pass 1: raw contact detection per foot
    std::vector<bool> rawLeft(numFrames, false);
    std::vector<bool> rawRight(numFrames, false);

    for (int f = 0; f < numFrames; ++f) {
        rawLeft[f] = impl_->rawContact(frames, f, leftFootIdx);
        rawRight[f] = impl_->rawContact(frames, f, rightFootIdx);
    }

    // Pass 2: temporal stability filter
    for (int f = 0; f < numFrames; ++f) {
        contacts[f].leftFootContact = impl_->stableContact(rawLeft, f);
        contacts[f].rightFootContact = impl_->stableContact(rawRight, f);
    }

    // Pass 3: smooth blend values with EMA
    float alpha = impl_->config.transitionSmoothing;
    for (int f = 0; f < numFrames; ++f) {
        float targetLeft = contacts[f].leftFootContact ? 1.0f : 0.0f;
        float targetRight = contacts[f].rightFootContact ? 1.0f : 0.0f;

        if (f == 0) {
            contacts[f].leftBlend = targetLeft;
            contacts[f].rightBlend = targetRight;
        } else {
            contacts[f].leftBlend = contacts[f - 1].leftBlend +
                alpha * (targetLeft - contacts[f - 1].leftBlend);
            contacts[f].rightBlend = contacts[f - 1].rightBlend +
                alpha * (targetRight - contacts[f - 1].rightBlend);
        }
    }

    HM_LOG_DEBUG(TAG, "Detected foot contacts for " +
                 std::to_string(numFrames) + " frames");
    return contacts;
}

void FootContactDetector::process(AnimClip& clip) const {
    if (clip.frames.empty()) return;
    clip.footContacts = detect(clip.frames);
    HM_LOG_INFO(TAG, "Processed foot contacts for clip '" + clip.name +
                "' (" + std::to_string(clip.frames.size()) + " frames)");
}

} // namespace hm::motion
