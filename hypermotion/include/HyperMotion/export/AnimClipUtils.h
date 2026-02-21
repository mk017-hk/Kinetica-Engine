#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <string>

namespace hm::xport {

class AnimClipUtils {
public:
    // Extract a sub-clip by frame range
    static AnimClip subClip(const AnimClip& clip, int startFrame, int endFrame);

    // Split clip into segments based on motion segments
    static std::vector<AnimClip> splitBySegments(const AnimClip& clip);

    // Concatenate multiple clips
    static AnimClip concatenate(const std::vector<AnimClip>& clips);

    // Resample to a different frame rate
    static AnimClip resample(const AnimClip& clip, float targetFPS);

    // Mirror clip (left-right swap)
    static AnimClip mirror(const AnimClip& clip);

    // Get clip duration in seconds
    static float getDuration(const AnimClip& clip);

    // Get number of frames
    static int getFrameCount(const AnimClip& clip);

    // Trim silence (low-motion) from start and end
    static AnimClip trimSilence(const AnimClip& clip, float velocityThreshold = 5.0f);
};

} // namespace hm::xport
