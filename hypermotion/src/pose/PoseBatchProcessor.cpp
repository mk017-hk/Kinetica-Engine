#include "HyperMotion/pose/PoseBatchProcessor.h"
#include "HyperMotion/core/Logger.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>

namespace hm::pose {

static constexpr const char* TAG = "PoseBatchProcessor";

struct BufferedFrame {
    cv::Mat image;
    double timestamp = 0.0;
    int frameIndex = 0;
};

struct PoseBatchProcessor::Impl {
    PoseBatchProcessorConfig config;
    PoseEstimator estimator;
    std::vector<BufferedFrame> buffer;
    bool initialized = false;

    Impl(const PoseBatchProcessorConfig& cfg)
        : config(cfg)
        , estimator(cfg.estimatorConfig) {
        buffer.reserve(cfg.batchSize);
    }
};

PoseBatchProcessor::PoseBatchProcessor(const PoseBatchProcessorConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PoseBatchProcessor::~PoseBatchProcessor() = default;
PoseBatchProcessor::PoseBatchProcessor(PoseBatchProcessor&&) noexcept = default;
PoseBatchProcessor& PoseBatchProcessor::operator=(PoseBatchProcessor&&) noexcept = default;

bool PoseBatchProcessor::initialize() {
    HM_LOG_INFO(TAG, "Initializing batch processor (batch_size=" +
                std::to_string(impl_->config.batchSize) + ")");

    if (!impl_->estimator.initialize()) {
        HM_LOG_WARN(TAG, "Estimator initialization had warnings (may use stubs)");
    }

    impl_->initialized = true;
    return true;
}

bool PoseBatchProcessor::isInitialized() const {
    return impl_->initialized;
}

bool PoseBatchProcessor::addFrame(const cv::Mat& frame, double timestamp,
                                   int frameIndex) {
    BufferedFrame bf;
    bf.image = frame.clone();
    bf.timestamp = timestamp;
    bf.frameIndex = frameIndex;
    impl_->buffer.push_back(std::move(bf));

    return static_cast<int>(impl_->buffer.size()) >= impl_->config.batchSize;
}

std::vector<PoseFrameResult> PoseBatchProcessor::processBatch() {
    if (impl_->buffer.empty()) return {};

    int count = std::min(static_cast<int>(impl_->buffer.size()),
                         impl_->config.batchSize);

    std::vector<PoseFrameResult> results;
    results.reserve(count);

    // Process each frame in the batch through the estimator.
    // In a production build with ONNX Runtime batching, the detector and
    // pose model would accept a tensor batch.  Here we iterate per-frame
    // but the crop-level inference inside estimateFrame is already batched
    // by the underlying SinglePoseEstimator when multiple persons are found.
    for (int i = 0; i < count; ++i) {
        auto& bf = impl_->buffer[i];
        auto result = impl_->estimator.estimateFrame(
            bf.image, bf.timestamp, bf.frameIndex);
        results.push_back(std::move(result));
    }

    // Remove processed frames from buffer
    impl_->buffer.erase(impl_->buffer.begin(),
                        impl_->buffer.begin() + count);

    return results;
}

std::vector<PoseFrameResult> PoseBatchProcessor::flush() {
    std::vector<PoseFrameResult> results;

    while (!impl_->buffer.empty()) {
        auto batch = processBatch();
        results.insert(results.end(),
                       std::make_move_iterator(batch.begin()),
                       std::make_move_iterator(batch.end()));
    }

    return results;
}

std::vector<PoseFrameResult> PoseBatchProcessor::processVideo(
    const std::string& videoPath, BatchProgressCallback callback) {

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        HM_LOG_ERROR(TAG, "Failed to open video: " + videoPath);
        return {};
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    HM_LOG_INFO(TAG, "Processing video: " + videoPath +
                " (" + std::to_string(totalFrames) + " frames, " +
                std::to_string(fps) + " fps)");

    std::vector<PoseFrameResult> allResults;
    allResults.reserve(totalFrames);

    cv::Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        double timestamp = frameIdx / fps;

        bool batchReady = addFrame(frame, timestamp, frameIdx);

        if (batchReady) {
            auto batch = processBatch();
            allResults.insert(allResults.end(),
                              std::make_move_iterator(batch.begin()),
                              std::make_move_iterator(batch.end()));
        }

        frameIdx++;

        if (callback && (frameIdx % 50 == 0 || frameIdx == totalFrames)) {
            float pct = totalFrames > 0
                ? static_cast<float>(frameIdx) / totalFrames * 100.0f
                : 0.0f;
            callback(pct, "Batch processing: frame " +
                     std::to_string(frameIdx) + "/" +
                     std::to_string(totalFrames));
        }
    }

    // Flush remaining frames
    auto remaining = flush();
    allResults.insert(allResults.end(),
                      std::make_move_iterator(remaining.begin()),
                      std::make_move_iterator(remaining.end()));

    HM_LOG_INFO(TAG, "Batch processing complete: " +
                std::to_string(allResults.size()) + " frames processed");
    return allResults;
}

std::vector<DetectedPerson> PoseBatchProcessor::toUnifiedSkeleton(
    const std::vector<DetectedPerson>& raw2D,
    DepthLifter& lifter) {

    std::vector<DetectedPerson> result;
    result.reserve(raw2D.size());

    for (const auto& person : raw2D) {
        DetectedPerson unified = person;
        unified.keypoints3D = lifter.lift(person.keypoints2D, person.bbox);
        result.push_back(std::move(unified));
    }

    return result;
}

} // namespace hm::pose
