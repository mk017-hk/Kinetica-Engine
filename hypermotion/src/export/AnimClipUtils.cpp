#include "HyperMotion/export/AnimClipUtils.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>

namespace hm::xport {

static constexpr const char* TAG = "AnimClipUtils";

AnimClip AnimClipUtils::subClip(const AnimClip& clip, int startFrame, int endFrame) {
    AnimClip result;
    result.name = clip.name + "_sub";
    result.fps = clip.fps;
    result.trackingID = clip.trackingID;

    startFrame = std::max(0, startFrame);
    endFrame = std::min(static_cast<int>(clip.frames.size()) - 1, endFrame);

    result.frames.reserve(endFrame - startFrame + 1);
    for (int f = startFrame; f <= endFrame; ++f) {
        result.frames.push_back(clip.frames[f]);
        result.frames.back().frameIndex = f - startFrame;
    }

    // Include overlapping segments
    for (const auto& seg : clip.segments) {
        if (seg.endFrame >= startFrame && seg.startFrame <= endFrame) {
            MotionSegment newSeg = seg;
            newSeg.startFrame = std::max(seg.startFrame - startFrame, 0);
            newSeg.endFrame = std::min(seg.endFrame - startFrame, endFrame - startFrame);
            result.segments.push_back(newSeg);
        }
    }

    return result;
}

std::vector<AnimClip> AnimClipUtils::splitBySegments(const AnimClip& clip) {
    std::vector<AnimClip> result;

    for (const auto& seg : clip.segments) {
        auto sub = subClip(clip, seg.startFrame, seg.endFrame);
        sub.name = clip.name + "_" + MOTION_TYPE_NAMES[static_cast<int>(seg.type)] +
                   "_" + std::to_string(seg.startFrame);
        result.push_back(sub);
    }

    return result;
}

AnimClip AnimClipUtils::concatenate(const std::vector<AnimClip>& clips) {
    AnimClip result;
    if (clips.empty()) return result;

    result.name = clips[0].name + "_concat";
    result.fps = clips[0].fps;
    result.trackingID = clips[0].trackingID;

    size_t totalFrames = 0;
    for (const auto& clip : clips) totalFrames += clip.frames.size();
    result.frames.reserve(totalFrames);

    int frameOffset = 0;
    for (const auto& clip : clips) {
        for (const auto& frame : clip.frames) {
            result.frames.push_back(frame);
            result.frames.back().frameIndex = frameOffset + frame.frameIndex;
        }

        for (const auto& seg : clip.segments) {
            auto newSeg = seg;
            newSeg.startFrame += frameOffset;
            newSeg.endFrame += frameOffset;
            result.segments.push_back(newSeg);
        }

        frameOffset += static_cast<int>(clip.frames.size());
    }

    return result;
}

AnimClip AnimClipUtils::resample(const AnimClip& clip, float targetFPS) {
    if (clip.frames.empty() || clip.fps <= 0 || targetFPS <= 0) return clip;

    AnimClip result;
    result.name = clip.name;
    result.fps = targetFPS;
    result.trackingID = clip.trackingID;

    float duration = static_cast<float>(clip.frames.size() - 1) / clip.fps;
    int newNumFrames = static_cast<int>(duration * targetFPS) + 1;

    result.frames.reserve(newNumFrames);

    for (int f = 0; f < newNumFrames; ++f) {
        float targetTime = f / targetFPS;
        float srcFrame = targetTime * clip.fps;

        int idx0 = std::min(static_cast<int>(srcFrame), static_cast<int>(clip.frames.size()) - 1);
        int idx1 = std::min(idx0 + 1, static_cast<int>(clip.frames.size()) - 1);
        float frac = srcFrame - idx0;

        SkeletonFrame newFrame = clip.frames[idx0];
        newFrame.frameIndex = f;
        newFrame.timestamp = targetTime;

        if (idx0 != idx1 && frac > 0.0f) {
            // Interpolate root position
            const auto& f0 = clip.frames[idx0];
            const auto& f1 = clip.frames[idx1];

            newFrame.rootPosition = f0.rootPosition + (f1.rootPosition - f0.rootPosition) * frac;
            newFrame.rootRotation = MathUtils::safeSlerp(f0.rootRotation, f1.rootRotation, frac);
            newFrame.rootVelocity = f0.rootVelocity + (f1.rootVelocity - f0.rootVelocity) * frac;

            // Interpolate joints
            for (int j = 0; j < JOINT_COUNT; ++j) {
                newFrame.joints[j].localRotation =
                    MathUtils::safeSlerp(f0.joints[j].localRotation,
                                          f1.joints[j].localRotation, frac);
                newFrame.joints[j].rotation6D =
                    MathUtils::quatToRot6D(newFrame.joints[j].localRotation);
                newFrame.joints[j].localEulerDeg =
                    MathUtils::quatToEulerDeg(newFrame.joints[j].localRotation);

                auto& p0 = f0.joints[j].worldPosition;
                auto& p1 = f1.joints[j].worldPosition;
                newFrame.joints[j].worldPosition = {
                    p0.x + (p1.x - p0.x) * frac,
                    p0.y + (p1.y - p0.y) * frac,
                    p0.z + (p1.z - p0.z) * frac
                };
            }
        }

        result.frames.push_back(newFrame);
    }

    // Remap segments
    for (const auto& seg : clip.segments) {
        MotionSegment newSeg = seg;
        newSeg.startFrame = static_cast<int>(seg.startFrame * targetFPS / clip.fps);
        newSeg.endFrame = static_cast<int>(seg.endFrame * targetFPS / clip.fps);
        newSeg.endFrame = std::min(newSeg.endFrame, newNumFrames - 1);
        result.segments.push_back(newSeg);
    }

    return result;
}

AnimClip AnimClipUtils::mirror(const AnimClip& clip) {
    AnimClip result = clip;
    result.name = clip.name + "_mirrored";

    // Left-right joint swap pairs
    constexpr std::pair<int, int> swapPairs[] = {
        {static_cast<int>(Joint::LeftShoulder), static_cast<int>(Joint::RightShoulder)},
        {static_cast<int>(Joint::LeftArm), static_cast<int>(Joint::RightArm)},
        {static_cast<int>(Joint::LeftForeArm), static_cast<int>(Joint::RightForeArm)},
        {static_cast<int>(Joint::LeftHand), static_cast<int>(Joint::RightHand)},
        {static_cast<int>(Joint::LeftUpLeg), static_cast<int>(Joint::RightUpLeg)},
        {static_cast<int>(Joint::LeftLeg), static_cast<int>(Joint::RightLeg)},
        {static_cast<int>(Joint::LeftFoot), static_cast<int>(Joint::RightFoot)},
        {static_cast<int>(Joint::LeftToeBase), static_cast<int>(Joint::RightToeBase)}
    };

    for (auto& frame : result.frames) {
        // Mirror root position (negate X)
        frame.rootPosition.x = -frame.rootPosition.x;
        frame.rootVelocity.x = -frame.rootVelocity.x;

        // Mirror root rotation (negate Y and Z components of quaternion)
        frame.rootRotation.y = -frame.rootRotation.y;
        frame.rootRotation.z = -frame.rootRotation.z;

        // Swap left-right joint pairs
        for (const auto& [left, right] : swapPairs) {
            std::swap(frame.joints[left], frame.joints[right]);
        }

        // Mirror X positions and rotations for all joints
        for (int j = 0; j < JOINT_COUNT; ++j) {
            frame.joints[j].worldPosition.x = -frame.joints[j].worldPosition.x;
            frame.joints[j].localEulerDeg.y = -frame.joints[j].localEulerDeg.y;
            frame.joints[j].localEulerDeg.z = -frame.joints[j].localEulerDeg.z;
            frame.joints[j].localRotation =
                MathUtils::eulerDegToQuat(frame.joints[j].localEulerDeg);
            frame.joints[j].rotation6D =
                MathUtils::quatToRot6D(frame.joints[j].localRotation);
        }
    }

    // Mirror segment directions
    for (auto& seg : result.segments) {
        seg.avgDirection.x = -seg.avgDirection.x;
        if (seg.type == MotionType::TurnLeft) seg.type = MotionType::TurnRight;
        else if (seg.type == MotionType::TurnRight) seg.type = MotionType::TurnLeft;
    }

    return result;
}

float AnimClipUtils::getDuration(const AnimClip& clip) {
    if (clip.frames.empty() || clip.fps <= 0) return 0.0f;
    return static_cast<float>(clip.frames.size() - 1) / clip.fps;
}

int AnimClipUtils::getFrameCount(const AnimClip& clip) {
    return static_cast<int>(clip.frames.size());
}

AnimClip AnimClipUtils::trimSilence(const AnimClip& clip, float velocityThreshold) {
    if (clip.frames.size() < 3) return clip;

    int startTrim = 0;
    int endTrim = static_cast<int>(clip.frames.size()) - 1;

    // Trim from start
    for (int f = 0; f < static_cast<int>(clip.frames.size()); ++f) {
        if (clip.frames[f].rootVelocity.length() > velocityThreshold) {
            startTrim = std::max(0, f - 5); // Keep 5 frames of lead-in
            break;
        }
    }

    // Trim from end
    for (int f = static_cast<int>(clip.frames.size()) - 1; f >= 0; --f) {
        if (clip.frames[f].rootVelocity.length() > velocityThreshold) {
            endTrim = std::min(static_cast<int>(clip.frames.size()) - 1, f + 5);
            break;
        }
    }

    return subClip(clip, startTrim, endTrim);
}

} // namespace hm::xport
