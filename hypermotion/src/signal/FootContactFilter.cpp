#include "HyperMotion/signal/FootContactFilter.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>

namespace hm::signal {

static constexpr const char* TAG = "FootContactFilter";

FootContactFilter::FootContactFilter(const FootContactConfig& config)
    : config_(config) {}

FootContactFilter::~FootContactFilter() = default;

std::vector<FootContactFilter::ContactState> FootContactFilter::detectContacts(
    const std::vector<SkeletonFrame>& frames) {

    int numFrames = static_cast<int>(frames.size());
    std::vector<ContactState> contacts(numFrames);

    if (numFrames < 2) return contacts;

    int leftFootIdx = static_cast<int>(Joint::LeftFoot);
    int rightFootIdx = static_cast<int>(Joint::RightFoot);

    for (int f = 0; f < numFrames; ++f) {
        const auto& frame = frames[f];

        // Foot height check
        float leftHeight = frame.joints[leftFootIdx].worldPosition.y - config_.groundHeight;
        float rightHeight = frame.joints[rightFootIdx].worldPosition.y - config_.groundHeight;

        // Foot velocity check (using finite differences)
        float leftVel = 0.0f, rightVel = 0.0f;
        if (f > 0) {
            Vec3 leftDelta = frame.joints[leftFootIdx].worldPosition -
                             frames[f - 1].joints[leftFootIdx].worldPosition;
            Vec3 rightDelta = frame.joints[rightFootIdx].worldPosition -
                              frames[f - 1].joints[rightFootIdx].worldPosition;

            float dt = 1.0f / 30.0f;
            leftVel = leftDelta.length() / dt;
            rightVel = rightDelta.length() / dt;
        }

        // Contact detection: velocity < threshold AND height < threshold
        contacts[f].leftFootContact =
            (leftVel < config_.velocityThreshold) && (leftHeight < config_.heightThreshold);
        contacts[f].rightFootContact =
            (rightVel < config_.velocityThreshold) && (rightHeight < config_.heightThreshold);

        // Smooth blend for transitions
        if (f > 0) {
            float targetLeftBlend = contacts[f].leftFootContact ? 1.0f : 0.0f;
            float targetRightBlend = contacts[f].rightFootContact ? 1.0f : 0.0f;

            contacts[f].leftBlend = contacts[f - 1].leftBlend +
                (targetLeftBlend - contacts[f - 1].leftBlend) * config_.transitionSmoothing;
            contacts[f].rightBlend = contacts[f - 1].rightBlend +
                (targetRightBlend - contacts[f - 1].rightBlend) * config_.transitionSmoothing;
        } else {
            contacts[f].leftBlend = contacts[f].leftFootContact ? 1.0f : 0.0f;
            contacts[f].rightBlend = contacts[f].rightFootContact ? 1.0f : 0.0f;
        }
    }

    return contacts;
}

void FootContactFilter::process(std::vector<SkeletonFrame>& frames) {
    if (frames.size() < 2) return;

    int numFrames = static_cast<int>(frames.size());
    HM_LOG_DEBUG(TAG, "Processing foot contact for " + std::to_string(numFrames) + " frames");

    auto contacts = detectContacts(frames);

    int leftFootIdx = static_cast<int>(Joint::LeftFoot);
    int leftToeIdx = static_cast<int>(Joint::LeftToeBase);
    int rightFootIdx = static_cast<int>(Joint::RightFoot);
    int rightToeIdx = static_cast<int>(Joint::RightToeBase);

    // Store last known grounded XZ positions
    Vec3 leftLockedPos = frames[0].joints[leftFootIdx].worldPosition;
    Vec3 rightLockedPos = frames[0].joints[rightFootIdx].worldPosition;

    for (int f = 0; f < numFrames; ++f) {
        auto& frame = frames[f];
        float leftBlend = contacts[f].leftBlend;
        float rightBlend = contacts[f].rightBlend;

        // Left foot
        if (contacts[f].leftFootContact) {
            if (f > 0 && !contacts[f - 1].leftFootContact) {
                // Just entered contact: lock position
                leftLockedPos = frame.joints[leftFootIdx].worldPosition;
                leftLockedPos.y = config_.groundHeight;
            }

            // During contact: blend towards locked position
            Vec3& footPos = frame.joints[leftFootIdx].worldPosition;
            footPos.x = footPos.x * (1.0f - leftBlend) + leftLockedPos.x * leftBlend;
            footPos.y = footPos.y * (1.0f - leftBlend) + config_.groundHeight * leftBlend;
            footPos.z = footPos.z * (1.0f - leftBlend) + leftLockedPos.z * leftBlend;

            // Also adjust toe
            Vec3& toePos = frame.joints[leftToeIdx].worldPosition;
            float toeDelta = toePos.y - frame.joints[leftFootIdx].worldPosition.y;
            toePos.y = config_.groundHeight + std::max(0.0f, toeDelta) * (1.0f - leftBlend);
        }

        // Right foot
        if (contacts[f].rightFootContact) {
            if (f > 0 && !contacts[f - 1].rightFootContact) {
                rightLockedPos = frame.joints[rightFootIdx].worldPosition;
                rightLockedPos.y = config_.groundHeight;
            }

            Vec3& footPos = frame.joints[rightFootIdx].worldPosition;
            footPos.x = footPos.x * (1.0f - rightBlend) + rightLockedPos.x * rightBlend;
            footPos.y = footPos.y * (1.0f - rightBlend) + config_.groundHeight * rightBlend;
            footPos.z = footPos.z * (1.0f - rightBlend) + rightLockedPos.z * rightBlend;

            Vec3& toePos = frame.joints[rightToeIdx].worldPosition;
            float toeDelta = toePos.y - frame.joints[rightFootIdx].worldPosition.y;
            toePos.y = config_.groundHeight + std::max(0.0f, toeDelta) * (1.0f - rightBlend);
        }
    }
}

} // namespace hm::signal
