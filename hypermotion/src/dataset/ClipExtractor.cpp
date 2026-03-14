#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/core/Logger.h"
#include <cmath>
#include <algorithm>

namespace hm::dataset {

static constexpr const char* TAG = "ClipExtractor";

struct ClipExtractor::Impl {
    ClipExtractorConfig config;

    // Compute average root velocity over a window
    float avgVelocity(const std::vector<SkeletonFrame>& frames, int center, int halfWin) {
        int start = std::max(0, center - halfWin);
        int end = std::min(static_cast<int>(frames.size()) - 1, center + halfWin);
        float sum = 0;
        int count = 0;
        for (int i = start; i <= end; ++i) {
            const auto& v = frames[i].rootVelocity;
            sum += std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            count++;
        }
        return count > 0 ? sum / count : 0;
    }

    // Compute average horizontal direction (radians) over a window
    float avgDirection(const std::vector<SkeletonFrame>& frames, int center, int halfWin) {
        int start = std::max(0, center - halfWin);
        int end = std::min(static_cast<int>(frames.size()) - 1, center + halfWin);
        float sumX = 0, sumZ = 0;
        for (int i = start; i <= end; ++i) {
            sumX += frames[i].rootVelocity.x;
            sumZ += frames[i].rootVelocity.z;
        }
        return std::atan2(sumZ, sumX);
    }

    // Detect if a frame is a boundary (motion change)
    bool isBoundary(const std::vector<SkeletonFrame>& frames, int idx) {
        int hw = config.analysisWindowFrames / 2;
        if (idx < hw + 1 || idx >= static_cast<int>(frames.size()) - hw - 1) return false;

        float velBefore = avgVelocity(frames, idx - hw, hw);
        float velAfter = avgVelocity(frames, idx + hw, hw);

        // Velocity change
        float velChange = std::abs(velAfter - velBefore);
        if (velChange > config.velocityChangeThreshold) return true;

        // Direction change
        float dirBefore = avgDirection(frames, idx - hw, hw);
        float dirAfter = avgDirection(frames, idx + hw, hw);
        float dirChange = std::abs(dirAfter - dirBefore);
        if (dirChange > M_PI) dirChange = 2.0f * M_PI - dirChange;
        float dirChangeDeg = dirChange * 180.0f / M_PI;
        if (dirChangeDeg > config.directionChangeThreshold && velBefore > 30.0f) return true;

        // Jump detection (vertical velocity spike)
        float vy = frames[idx].rootVelocity.y;
        if (std::abs(vy) > config.jumpVelocityThreshold) return true;

        // Sudden stop
        if (velBefore > 50.0f && velAfter < config.stopVelocityThreshold) return true;

        return false;
    }

    // Build a clip from a frame range
    AnimClip buildClip(const std::vector<SkeletonFrame>& frames,
                       int start, int end, int playerID, int clipIdx) {
        AnimClip clip;
        clip.name = "player_" + std::to_string(playerID) +
                    "_clip_" + std::to_string(clipIdx);
        clip.fps = config.fps;
        clip.trackingID = playerID;

        int actualEnd = std::min(end, static_cast<int>(frames.size()) - 1);
        clip.frames.reserve(actualEnd - start + 1);
        for (int i = start; i <= actualEnd; ++i) {
            clip.frames.push_back(frames[i]);
            clip.frames.back().frameIndex = i - start;
        }
        return clip;
    }

    // Compute metadata for a clip
    ClipMetadata computeMetadata(const AnimClip& clip, int playerID,
                                  int startFrame, int endFrame) {
        ClipMetadata meta;
        meta.playerID = playerID;
        meta.startFrame = startFrame;
        meta.endFrame = endFrame;
        meta.durationSec = clip.frames.empty() ? 0.0f :
            static_cast<float>(clip.frames.size() - 1) / config.fps;

        float sumVel = 0, maxVel = 0;
        Vec3 sumDir{};
        for (const auto& f : clip.frames) {
            float v = std::sqrt(f.rootVelocity.x * f.rootVelocity.x +
                               f.rootVelocity.y * f.rootVelocity.y +
                               f.rootVelocity.z * f.rootVelocity.z);
            sumVel += v;
            maxVel = std::max(maxVel, v);
            sumDir.x += f.rootVelocity.x;
            sumDir.y += f.rootVelocity.y;
            sumDir.z += f.rootVelocity.z;
        }
        int n = static_cast<int>(clip.frames.size());
        if (n > 0) {
            meta.avgVelocity = sumVel / n;
            meta.maxVelocity = maxVel;
            meta.avgDirection = {sumDir.x / n, sumDir.y / n, sumDir.z / n};
        }
        return meta;
    }
};

ClipExtractor::ClipExtractor(const ClipExtractorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

ClipExtractor::~ClipExtractor() = default;
ClipExtractor::ClipExtractor(ClipExtractor&&) noexcept = default;
ClipExtractor& ClipExtractor::operator=(ClipExtractor&&) noexcept = default;

ClipExtractor::ExtractionResult ClipExtractor::extract(
    const std::vector<SkeletonFrame>& frames, int playerID) {

    ExtractionResult result;
    if (frames.empty()) return result;

    int minFrames = std::max(1, static_cast<int>(impl_->config.minClipDurationSec * impl_->config.fps));
    int maxFrames = static_cast<int>(impl_->config.maxClipDurationSec * impl_->config.fps);

    // Find boundary frames
    std::vector<int> boundaries;
    boundaries.push_back(0);
    for (int i = 1; i < static_cast<int>(frames.size()) - 1; ++i) {
        if (impl_->isBoundary(frames, i)) {
            boundaries.push_back(i);
        }
    }
    boundaries.push_back(static_cast<int>(frames.size()) - 1);

    // Merge boundaries that are too close
    std::vector<int> merged;
    merged.push_back(boundaries[0]);
    for (size_t i = 1; i < boundaries.size(); ++i) {
        if (boundaries[i] - merged.back() >= minFrames) {
            merged.push_back(boundaries[i]);
        }
    }

    // Split segments that exceed maxFrames
    std::vector<std::pair<int, int>> ranges;
    for (size_t i = 0; i + 1 < merged.size(); ++i) {
        int start = merged[i];
        int end = merged[i + 1];
        while (end - start > maxFrames) {
            ranges.push_back({start, start + maxFrames});
            start += maxFrames;
        }
        if (end - start >= minFrames) {
            ranges.push_back({start, end});
        }
    }

    // Build clips
    int clipIdx = 0;
    for (const auto& [start, end] : ranges) {
        auto clip = impl_->buildClip(frames, start, end, playerID, clipIdx);
        auto meta = impl_->computeMetadata(clip, playerID, start, end);
        result.clips.push_back(std::move(clip));
        result.metadata.push_back(meta);
        clipIdx++;
    }

    HM_LOG_INFO(TAG, "Extracted " + std::to_string(result.clips.size()) +
                " clips from " + std::to_string(frames.size()) +
                " frames (player " + std::to_string(playerID) + ")");
    return result;
}

ClipExtractor::ExtractionResult ClipExtractor::extractFromSegments(
    const std::vector<SkeletonFrame>& frames,
    const std::vector<MotionSegment>& segments,
    int playerID) {

    ExtractionResult result;
    if (frames.empty() || segments.empty()) return result;

    int minFrames = std::max(1, static_cast<int>(impl_->config.minClipDurationSec * impl_->config.fps));
    int maxFrames = static_cast<int>(impl_->config.maxClipDurationSec * impl_->config.fps);
    int clipIdx = 0;

    for (const auto& seg : segments) {
        int start = seg.startFrame;
        int end = std::min(seg.endFrame, static_cast<int>(frames.size()) - 1);
        int segLen = end - start;

        if (segLen < minFrames) continue;

        // Split long segments
        while (end - start > maxFrames) {
            auto clip = impl_->buildClip(frames, start, start + maxFrames, playerID, clipIdx);
            auto meta = impl_->computeMetadata(clip, playerID, start, start + maxFrames);
            meta.motionType = MOTION_TYPE_NAMES[static_cast<int>(seg.type)];
            meta.confidence = seg.confidence;
            result.clips.push_back(std::move(clip));
            result.metadata.push_back(meta);
            start += maxFrames;
            clipIdx++;
        }

        if (end - start >= minFrames) {
            auto clip = impl_->buildClip(frames, start, end, playerID, clipIdx);
            auto meta = impl_->computeMetadata(clip, playerID, start, end);
            meta.motionType = MOTION_TYPE_NAMES[static_cast<int>(seg.type)];
            meta.confidence = seg.confidence;
            result.clips.push_back(std::move(clip));
            result.metadata.push_back(meta);
            clipIdx++;
        }
    }

    HM_LOG_INFO(TAG, "Extracted " + std::to_string(result.clips.size()) +
                " clips from " + std::to_string(segments.size()) +
                " segments (player " + std::to_string(playerID) + ")");
    return result;
}

} // namespace hm::dataset
