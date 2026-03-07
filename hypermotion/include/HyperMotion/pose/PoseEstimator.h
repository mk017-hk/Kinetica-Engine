#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/pose/PlayerDetector.h"
#include "HyperMotion/pose/SinglePoseEstimator.h"
#include "HyperMotion/pose/DepthLifter.h"
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace hm::pose {

/// Unified pose estimator that wraps detection, 2D pose estimation, and 3D
/// lifting into a single call.  Accepts raw BGR frames and returns per-person
/// COCO-17 keypoints in both 2D and 3D.
///
/// Unlike MultiPersonPoseEstimator (which also tracks across frames), this
/// class is stateless per-frame and suitable for use inside batch or streaming
/// pipelines where tracking is handled externally.
struct PoseEstimatorConfig {
    PlayerDetectorConfig detector;
    SinglePoseEstimatorConfig poseModel;
    DepthLifterConfig depthLifter;
    bool useGPU = true;
    float confidenceThreshold = 0.3f;
};

class PoseEstimator {
public:
    explicit PoseEstimator(const PoseEstimatorConfig& config = {});
    ~PoseEstimator();

    PoseEstimator(const PoseEstimator&) = delete;
    PoseEstimator& operator=(const PoseEstimator&) = delete;
    PoseEstimator(PoseEstimator&&) noexcept;
    PoseEstimator& operator=(PoseEstimator&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// Run pose inference on a single frame.
    /// Returns all detected persons with 2D and 3D keypoints.
    PoseFrameResult estimateFrame(const cv::Mat& frame, double timestamp = 0.0,
                                  int frameIndex = 0);

    /// Run inference on a list of pre-cropped person images.
    /// Each crop should be a tight bounding-box crop of a single person.
    std::vector<DetectedPerson> estimateCrops(
        const std::vector<cv::Mat>& crops,
        const std::vector<BBox>& bboxes);

    /// Get the internal detector (e.g. for reuse).
    const PlayerDetector& detector() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
