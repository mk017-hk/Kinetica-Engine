#include "HyperMotion/pose/MultiPersonPoseEstimator.h"
#include "HyperMotion/core/Logger.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <sstream>

namespace hm::pose {

static constexpr const char* TAG = "MultiPersonPoseEstimator";

// COCO skeleton connections for visualization
static constexpr std::pair<int, int> COCO_SKELETON[] = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},       // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // Arms
    {5, 11}, {6, 12}, {11, 12},            // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16} // Legs
};
static constexpr int COCO_SKELETON_COUNT = 16;

struct MultiPersonPoseEstimator::Impl {
    MultiPersonPoseConfig config;
    PlayerDetector detector;
    SinglePoseEstimator poseEstimator;
    PoseTracker tracker;
    DepthLifter depthLifter;
    bool initialized = false;

    Impl(const MultiPersonPoseConfig& cfg)
        : config(cfg)
        , detector(cfg.detector)
        , poseEstimator(cfg.poseEstimator)
        , tracker(cfg.tracker)
        , depthLifter(cfg.depthLifter) {}
};

MultiPersonPoseEstimator::MultiPersonPoseEstimator(const MultiPersonPoseConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MultiPersonPoseEstimator::~MultiPersonPoseEstimator() = default;
MultiPersonPoseEstimator::MultiPersonPoseEstimator(MultiPersonPoseEstimator&&) noexcept = default;
MultiPersonPoseEstimator& MultiPersonPoseEstimator::operator=(MultiPersonPoseEstimator&&) noexcept = default;

bool MultiPersonPoseEstimator::initialize() {
    HM_LOG_INFO(TAG, "Initializing multi-person pose estimator...");

    if (!impl_->detector.initialize()) {
        HM_LOG_ERROR(TAG, "Failed to initialize player detector");
        return false;
    }

    if (!impl_->poseEstimator.initialize()) {
        HM_LOG_ERROR(TAG, "Failed to initialize pose estimator");
        return false;
    }

    if (!impl_->depthLifter.initialize()) {
        HM_LOG_WARN(TAG, "Depth lifter initialization failed, 3D poses will use fallback");
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Multi-person pose estimator initialized");
    return true;
}

bool MultiPersonPoseEstimator::isInitialized() const {
    return impl_->initialized;
}

std::vector<PoseFrameResult> MultiPersonPoseEstimator::processVideo(
    const std::string& videoPath, ProgressCallback callback) {

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Not initialized");
        return {};
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        HM_LOG_ERROR(TAG, "Cannot open video: " + videoPath);
        return {};
    }

    double videoFPS = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    HM_LOG_INFO(TAG, "Video: " + videoPath +
                " (" + std::to_string(width) + "x" + std::to_string(height) +
                " @ " + std::to_string(videoFPS) + "fps, " +
                std::to_string(totalFrames) + " frames)");

    // Frame skipping for target FPS
    int frameSkip = 1;
    if (videoFPS > 0 && impl_->config.targetFPS > 0 && videoFPS > impl_->config.targetFPS) {
        frameSkip = static_cast<int>(std::round(videoFPS / impl_->config.targetFPS));
        frameSkip = std::max(1, frameSkip);
    }

    std::vector<PoseFrameResult> results;
    cv::Mat frame;
    int frameIndex = 0;
    int processedCount = 0;

    impl_->tracker.reset();

    while (cap.read(frame)) {
        if (frameIndex % frameSkip != 0) {
            frameIndex++;
            continue;
        }

        double timestamp = frameIndex / std::max(1.0, videoFPS);
        auto result = processFrame(frame, timestamp, processedCount);
        result.videoWidth = width;
        result.videoHeight = height;
        results.push_back(std::move(result));

        processedCount++;
        frameIndex++;

        if (callback && totalFrames > 0) {
            float progress = static_cast<float>(frameIndex) / totalFrames * 100.0f;
            callback(progress, "Processing frame " + std::to_string(frameIndex) +
                     "/" + std::to_string(totalFrames));
        }
    }

    HM_LOG_INFO(TAG, "Processed " + std::to_string(processedCount) + " frames");
    return results;
}

PoseFrameResult MultiPersonPoseEstimator::processFrame(
    const cv::Mat& frame, double timestamp, int frameIndex) {

    PoseFrameResult result;
    result.timestamp = timestamp;
    result.frameIndex = frameIndex;
    result.videoWidth = frame.cols;
    result.videoHeight = frame.rows;

    if (!impl_->initialized || frame.empty()) {
        return result;
    }

    // Step 1: Detect players
    auto detections = impl_->detector.detect(frame);

    // Step 2: Estimate pose for each detection
    std::vector<BBox> bboxes;
    bboxes.reserve(detections.size());
    for (const auto& det : detections) {
        bboxes.push_back(det.bbox);
    }

    auto poses2D = impl_->poseEstimator.estimateBatch(frame, bboxes);

    // Step 3: Track across frames
    impl_->tracker.update(detections, poses2D);
    auto tracklets = impl_->tracker.getConfirmedTracklets();

    // Step 4: Lift to 3D for confirmed tracklets
    std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>> trackletPoses;
    std::vector<BBox> trackletBboxes;
    for (const auto& t : tracklets) {
        trackletPoses.push_back(t.lastPose);
        trackletBboxes.push_back(t.lastDetection.bbox);
    }

    auto poses3D = impl_->depthLifter.liftBatch(trackletPoses, trackletBboxes);

    // Build output
    for (size_t i = 0; i < tracklets.size(); ++i) {
        DetectedPerson person;
        person.id = tracklets[i].id;
        person.bbox = tracklets[i].lastDetection.bbox;
        person.classLabel = tracklets[i].classLabel;
        person.keypoints2D = tracklets[i].lastPose;
        person.reidFeature = tracklets[i].reidFeature;

        if (i < poses3D.size()) {
            person.keypoints3D = poses3D[i];
        }

        result.persons.push_back(std::move(person));
    }

    return result;
}

cv::Mat MultiPersonPoseEstimator::drawDebug(const cv::Mat& frame,
                                             const PoseFrameResult& result) const {
    cv::Mat vis = frame.clone();

    // Color palette for different classes
    const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0),    // player: green
        cv::Scalar(0, 255, 255),  // referee: yellow
        cv::Scalar(255, 0, 0)     // goalkeeper: blue
    };

    for (const auto& person : result.persons) {
        int colorIdx = 0;
        if (person.classLabel == "referee") colorIdx = 1;
        else if (person.classLabel == "goalkeeper") colorIdx = 2;
        cv::Scalar color = colors[colorIdx];

        // Draw bbox
        cv::rectangle(vis,
                      cv::Point(static_cast<int>(person.bbox.x),
                                static_cast<int>(person.bbox.y)),
                      cv::Point(static_cast<int>(person.bbox.x + person.bbox.width),
                                static_cast<int>(person.bbox.y + person.bbox.height)),
                      color, 2);

        // Draw ID
        std::string label = person.classLabel + " #" + std::to_string(person.id);
        cv::putText(vis, label,
                    cv::Point(static_cast<int>(person.bbox.x),
                              static_cast<int>(person.bbox.y) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

        // Draw skeleton
        int w = result.videoWidth;
        int h = result.videoHeight;

        for (int s = 0; s < COCO_SKELETON_COUNT; ++s) {
            auto [k1, k2] = COCO_SKELETON[s];
            const auto& kp1 = person.keypoints2D[k1];
            const auto& kp2 = person.keypoints2D[k2];

            if (kp1.confidence > 0.3f && kp2.confidence > 0.3f) {
                cv::Point p1(static_cast<int>(kp1.position.x * w),
                             static_cast<int>(kp1.position.y * h));
                cv::Point p2(static_cast<int>(kp2.position.x * w),
                             static_cast<int>(kp2.position.y * h));
                cv::line(vis, p1, p2, color, 2);
            }
        }

        // Draw keypoints
        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            const auto& kp = person.keypoints2D[k];
            if (kp.confidence > 0.3f) {
                cv::Point pt(static_cast<int>(kp.position.x * w),
                             static_cast<int>(kp.position.y * h));
                cv::circle(vis, pt, 3, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    return vis;
}

void MultiPersonPoseEstimator::reset() {
    impl_->tracker.reset();
}

} // namespace hm::pose
