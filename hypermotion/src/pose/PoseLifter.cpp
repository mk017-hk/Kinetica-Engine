#include "HyperMotion/pose/PoseLifter.h"
#include "HyperMotion/core/Logger.h"

namespace hm::pose {

struct PoseLifter::Impl {
    PoseLifterConfig config;
    DepthLifter lifter;
    bool initialized = false;

    explicit Impl(const PoseLifterConfig& cfg)
        : config(cfg), lifter(cfg.depthConfig) {}
};

PoseLifter::PoseLifter(const PoseLifterConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PoseLifter::~PoseLifter() = default;
PoseLifter::PoseLifter(PoseLifter&&) noexcept = default;
PoseLifter& PoseLifter::operator=(PoseLifter&&) noexcept = default;

bool PoseLifter::initialize() {
    if (impl_->initialized) return true;
    impl_->initialized = impl_->lifter.initialize();
    if (!impl_->initialized) {
        HM_LOG_WARN("PoseLifter", "DepthLifter init failed; geometric fallback will be used");
        impl_->initialized = true;
    }
    return impl_->initialized;
}

bool PoseLifter::isInitialized() const { return impl_->initialized; }

Pose3D PoseLifter::lift(const Pose2D& pose2D, const BBox& bbox) {
    // Convert Pose2D to Keypoint2D array for DepthLifter
    std::array<Keypoint2D, COCO_KEYPOINTS> kp2D{};
    for (int i = 0; i < COCO_KEYPOINTS; ++i) {
        kp2D[i].position = pose2D.joints[i];
        kp2D[i].confidence = pose2D.confidence[i];
    }

    auto kp3D = impl_->lifter.lift(kp2D, bbox);

    Pose3D result;
    for (int i = 0; i < COCO_KEYPOINTS; ++i) {
        result.joints[i] = kp3D[i].position;
        result.confidence[i] = kp3D[i].confidence;
    }
    return result;
}

std::vector<Pose3D> PoseLifter::liftSequence(
    const std::vector<Pose2D>& poses,
    const std::vector<BBox>& bboxes) {

    std::vector<Pose3D> results;
    results.reserve(poses.size());

    size_t n = std::min(poses.size(), bboxes.size());
    for (size_t i = 0; i < n; ++i) {
        results.push_back(lift(poses[i], bboxes[i]));
    }
    return results;
}

DepthLifter& PoseLifter::depthLifter() { return impl_->lifter; }
const DepthLifter& PoseLifter::depthLifter() const { return impl_->lifter; }

} // namespace hm::pose
