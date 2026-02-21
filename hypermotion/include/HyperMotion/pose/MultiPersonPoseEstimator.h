#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/pose/PlayerDetector.h"
#include "HyperMotion/pose/SinglePoseEstimator.h"
#include "HyperMotion/pose/PoseTracker.h"
#include "HyperMotion/pose/DepthLifter.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

namespace hm::pose {

struct MultiPersonPoseConfig {
    PlayerDetectorConfig detector;
    SinglePoseEstimatorConfig poseEstimator;
    PoseTrackerConfig tracker;
    DepthLifterConfig depthLifter;
    float targetFPS = 30.0f;
    bool enableVisualization = false;
};

using ProgressCallback = std::function<void(float percent, const std::string& message)>;

class MultiPersonPoseEstimator {
public:
    explicit MultiPersonPoseEstimator(const MultiPersonPoseConfig& config);
    ~MultiPersonPoseEstimator();

    MultiPersonPoseEstimator(const MultiPersonPoseEstimator&) = delete;
    MultiPersonPoseEstimator& operator=(const MultiPersonPoseEstimator&) = delete;
    MultiPersonPoseEstimator(MultiPersonPoseEstimator&&) noexcept;
    MultiPersonPoseEstimator& operator=(MultiPersonPoseEstimator&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    // Process full video file
    std::vector<PoseFrameResult> processVideo(const std::string& videoPath,
                                               ProgressCallback callback = nullptr);

    // Process single frame (for streaming / real-time)
    PoseFrameResult processFrame(const cv::Mat& frame, double timestamp, int frameIndex);

    // Debug visualization
    cv::Mat drawDebug(const cv::Mat& frame, const PoseFrameResult& result) const;

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
