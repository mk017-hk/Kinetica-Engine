#pragma once

#include "HyperMotion/core/Types.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace hm::pose {

struct SinglePoseEstimatorConfig {
    std::string modelPath;
    int inputWidth = 192;
    int inputHeight = 256;
    float confidenceThreshold = 0.3f;
    float bboxPadding = 0.2f;
    bool useFlipTest = true;
    bool useTensorRT = false;
    std::string tensorrtCachePath;
};

class SinglePoseEstimator {
public:
    explicit SinglePoseEstimator(const SinglePoseEstimatorConfig& config);
    ~SinglePoseEstimator();

    SinglePoseEstimator(const SinglePoseEstimator&) = delete;
    SinglePoseEstimator& operator=(const SinglePoseEstimator&) = delete;
    SinglePoseEstimator(SinglePoseEstimator&&) noexcept;
    SinglePoseEstimator& operator=(SinglePoseEstimator&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    std::array<Keypoint2D, COCO_KEYPOINTS> estimate(const cv::Mat& frame, const BBox& bbox);

    std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>> estimateBatch(
        const cv::Mat& frame, const std::vector<BBox>& bboxes);

    // COCO keypoint flip pairs for flip-test augmentation
    static constexpr std::array<std::pair<int,int>, 8> kFlipPairs = {{
        {1, 2}, {3, 4}, {5, 6}, {7, 8},
        {9, 10}, {11, 12}, {13, 14}, {15, 16}
    }};

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
