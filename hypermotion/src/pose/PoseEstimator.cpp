#include "HyperMotion/pose/PoseEstimator.h"
#include "HyperMotion/core/Logger.h"

namespace hm::pose {

static constexpr const char* TAG = "PoseEstimator";

struct PoseEstimator::Impl {
    PoseEstimatorConfig config;
    PlayerDetector detector;
    SinglePoseEstimator poseModel;
    DepthLifter depthLifter;
    bool initialized = false;

    Impl(const PoseEstimatorConfig& cfg)
        : config(cfg)
        , detector(cfg.detector)
        , poseModel(cfg.poseModel)
        , depthLifter(cfg.depthLifter) {}
};

PoseEstimator::PoseEstimator(const PoseEstimatorConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PoseEstimator::~PoseEstimator() = default;
PoseEstimator::PoseEstimator(PoseEstimator&&) noexcept = default;
PoseEstimator& PoseEstimator::operator=(PoseEstimator&&) noexcept = default;

bool PoseEstimator::initialize() {
    HM_LOG_INFO(TAG, "Initializing pose estimator...");

    if (!impl_->detector.initialize()) {
        HM_LOG_WARN(TAG, "Player detector not available (no model)");
    }
    if (!impl_->poseModel.initialize()) {
        HM_LOG_WARN(TAG, "Pose model not available (no model)");
    }
    if (!impl_->depthLifter.initialize()) {
        HM_LOG_WARN(TAG, "Depth lifter not available (no model)");
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Pose estimator initialized");
    return true;
}

bool PoseEstimator::isInitialized() const {
    return impl_->initialized;
}

PoseFrameResult PoseEstimator::estimateFrame(const cv::Mat& frame,
                                              double timestamp,
                                              int frameIndex) {
    PoseFrameResult result;
    result.timestamp = timestamp;
    result.frameIndex = frameIndex;
    result.videoWidth = frame.cols;
    result.videoHeight = frame.rows;

    // Step 1: Detect players
    auto detections = impl_->detector.detect(frame);

    // Step 2: For each detection, run pose estimation
    for (const auto& det : detections) {
        if (det.confidence < impl_->config.confidenceThreshold) continue;

        // Run 2D pose estimation using the detection bbox
        auto keypoints2D = impl_->poseModel.estimate(frame, det.bbox);

        // Step 3: Lift to 3D
        auto keypoints3D = impl_->depthLifter.lift(keypoints2D, det.bbox);

        DetectedPerson person;
        person.id = static_cast<int>(result.persons.size());
        person.keypoints2D = keypoints2D;
        person.keypoints3D = keypoints3D;
        person.bbox = det.bbox;
        person.classLabel = det.classLabel;
        result.persons.push_back(std::move(person));
    }

    return result;
}

std::vector<DetectedPerson> PoseEstimator::estimateCrops(
    const std::vector<cv::Mat>& crops,
    const std::vector<BBox>& bboxes) {

    std::vector<DetectedPerson> results;
    results.reserve(crops.size());

    for (size_t i = 0; i < crops.size(); ++i) {
        BBox bbox;
        if (i < bboxes.size()) bbox = bboxes[i];

        auto keypoints2D = impl_->poseModel.estimate(crops[i], bbox);
        auto keypoints3D = impl_->depthLifter.lift(keypoints2D, bbox);

        DetectedPerson person;
        person.id = static_cast<int>(i);
        person.keypoints2D = keypoints2D;
        person.keypoints3D = keypoints3D;
        if (i < bboxes.size()) person.bbox = bboxes[i];
        results.push_back(std::move(person));
    }

    return results;
}

const PlayerDetector& PoseEstimator::detector() const {
    return impl_->detector;
}

} // namespace hm::pose
