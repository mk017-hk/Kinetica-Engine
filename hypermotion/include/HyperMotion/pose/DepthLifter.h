#pragma once

#include "HyperMotion/core/Types.h"
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace hm::pose {

struct DepthLifterConfig {
    std::string modelPath;
    bool useGeometricFallback = true;
    float defaultSubjectHeight = 175.0f; // cm
};

class DepthLifter {
public:
    explicit DepthLifter(const DepthLifterConfig& config = {});
    ~DepthLifter();

    DepthLifter(const DepthLifter&) = delete;
    DepthLifter& operator=(const DepthLifter&) = delete;
    DepthLifter(DepthLifter&&) noexcept;
    DepthLifter& operator=(DepthLifter&&) noexcept;

    bool initialize();
    bool isInitialized() const;
    bool hasModel() const;

    std::array<Keypoint3D, COCO_KEYPOINTS> lift(
        const std::array<Keypoint2D, COCO_KEYPOINTS>& keypoints2D,
        const BBox& bbox);

    std::vector<std::array<Keypoint3D, COCO_KEYPOINTS>> liftBatch(
        const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& keypoints2DList,
        const std::vector<BBox>& bboxes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
