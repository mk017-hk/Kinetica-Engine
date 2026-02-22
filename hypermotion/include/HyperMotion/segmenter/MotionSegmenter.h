#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include "HyperMotion/segmenter/TemporalConvNet.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::segmenter {

struct MotionSegmenterConfig {
    /// Path to the ONNX classifier model (exported from Python).
    std::string modelPath;
    int slidingWindowSize = 256;
    int slidingWindowStride = 128;
    int minSegmentLength = 10;
    float confidenceThreshold = 0.5f;
    bool useGPU = true;
};

class MotionSegmenter {
public:
    explicit MotionSegmenter(const MotionSegmenterConfig& config = {});
    ~MotionSegmenter();

    MotionSegmenter(const MotionSegmenter&) = delete;
    MotionSegmenter& operator=(const MotionSegmenter&) = delete;
    MotionSegmenter(MotionSegmenter&&) noexcept;
    MotionSegmenter& operator=(MotionSegmenter&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// Segment a sequence of skeleton frames into motion segments.
    std::vector<MotionSegment> segment(const std::vector<SkeletonFrame>& frames,
                                        int trackingID = -1);

    /// Get per-frame classification probabilities.
    std::vector<std::array<float, MOTION_TYPE_COUNT>> classifyFrames(
        const std::vector<SkeletonFrame>& frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::segmenter
