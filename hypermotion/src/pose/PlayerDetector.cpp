#include "HyperMotion/pose/PlayerDetector.h"
#include "HyperMotion/core/Logger.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>

namespace hm::pose {

static constexpr const char* TAG = "PlayerDetector";

struct PlayerDetector::Impl {
    PlayerDetectorConfig config;
    cv::dnn::Net net;
    bool initialized = false;

    // Class labels: 0=player, 1=referee, 2=goalkeeper
    static constexpr int NUM_CLASSES = 3;
    static constexpr const char* CLASS_NAMES[] = {"player", "referee", "goalkeeper"};

    cv::Mat preprocess(const cv::Mat& frame) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(config.inputWidth, config.inputHeight),
                               cv::Scalar(0, 0, 0), true, false, CV_32F);
        return blob;
    }

    std::vector<Detection> postprocess(const cv::Mat& output, int origWidth, int origHeight) {
        std::vector<Detection> detections;

        // YOLOv8 output shape: [1, 4+nclass, ndet] — need to transpose to [1, ndet, 4+nclass]
        int rows = output.size[1];
        int cols = output.size[2];

        // Determine if we need to transpose
        cv::Mat data;
        if (rows == (4 + NUM_CLASSES) && cols > rows) {
            // Shape is [4+nclass, ndet], transpose
            cv::Mat reshaped = output.reshape(1, rows);
            cv::transpose(reshaped, data);
        } else {
            data = output.reshape(1, rows);
        }

        int numDetections = data.rows;
        int dataCols = data.cols;
        if (dataCols < 4 + NUM_CLASSES) return detections;

        float scaleX = static_cast<float>(origWidth) / config.inputWidth;
        float scaleY = static_cast<float>(origHeight) / config.inputHeight;

        struct RawDetection {
            BBox bbox;
            float confidence;
            int classID;
        };

        // Per-class storage for NMS
        std::array<std::vector<RawDetection>, NUM_CLASSES> perClassDetections;

        for (int i = 0; i < numDetections; ++i) {
            const float* row = data.ptr<float>(i);

            // YOLOv8: cx, cy, w, h, class_scores...
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            // Find best class
            int bestClass = 0;
            float bestScore = row[4];
            for (int c = 1; c < NUM_CLASSES; ++c) {
                if (row[4 + c] > bestScore) {
                    bestScore = row[4 + c];
                    bestClass = c;
                }
            }

            if (bestScore < config.confidenceThreshold) continue;

            RawDetection det;
            det.bbox.x = (cx - w * 0.5f) * scaleX;
            det.bbox.y = (cy - h * 0.5f) * scaleY;
            det.bbox.width = w * scaleX;
            det.bbox.height = h * scaleY;
            det.bbox.confidence = bestScore;
            det.confidence = bestScore;
            det.classID = bestClass;

            perClassDetections[bestClass].push_back(det);
        }

        // Per-class NMS
        for (int c = 0; c < NUM_CLASSES; ++c) {
            auto& dets = perClassDetections[c];
            if (dets.empty()) continue;

            // Sort by confidence descending
            std::sort(dets.begin(), dets.end(),
                      [](const RawDetection& a, const RawDetection& b) {
                          return a.confidence > b.confidence;
                      });

            std::vector<bool> suppressed(dets.size(), false);
            for (size_t i = 0; i < dets.size(); ++i) {
                if (suppressed[i]) continue;

                Detection det;
                det.bbox = dets[i].bbox;
                det.confidence = dets[i].confidence;
                det.classID = dets[i].classID;
                det.classLabel = CLASS_NAMES[c];
                detections.push_back(det);

                if (static_cast<int>(detections.size()) >= config.maxDetections)
                    break;

                for (size_t j = i + 1; j < dets.size(); ++j) {
                    if (suppressed[j]) continue;
                    if (dets[i].bbox.iou(dets[j].bbox) > config.nmsIouThreshold) {
                        suppressed[j] = true;
                    }
                }
            }

            if (static_cast<int>(detections.size()) >= config.maxDetections)
                break;
        }

        // Limit total
        if (static_cast<int>(detections.size()) > config.maxDetections) {
            std::sort(detections.begin(), detections.end(),
                      [](const Detection& a, const Detection& b) {
                          return a.confidence > b.confidence;
                      });
            detections.resize(config.maxDetections);
        }

        return detections;
    }
};

PlayerDetector::PlayerDetector(const PlayerDetectorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

PlayerDetector::~PlayerDetector() = default;
PlayerDetector::PlayerDetector(PlayerDetector&&) noexcept = default;
PlayerDetector& PlayerDetector::operator=(PlayerDetector&&) noexcept = default;

bool PlayerDetector::initialize() {
    try {
        if (impl_->config.modelPath.empty()) {
            HM_LOG_ERROR(TAG, "Model path is empty");
            return false;
        }

        impl_->net = cv::dnn::readNetFromONNX(impl_->config.modelPath);

        if (impl_->net.empty()) {
            HM_LOG_ERROR(TAG, "Failed to load model from: " + impl_->config.modelPath);
            return false;
        }

        // Prefer CUDA backend if available
        impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#ifdef HM_USE_CUDA
        try {
            impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            HM_LOG_INFO(TAG, "Using CUDA backend for detection");
        } catch (...) {
            impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            HM_LOG_WARN(TAG, "CUDA not available, falling back to CPU");
        }
#endif

        impl_->initialized = true;
        HM_LOG_INFO(TAG, "Initialized with model: " + impl_->config.modelPath);
        return true;

    } catch (const cv::Exception& e) {
        HM_LOG_ERROR(TAG, std::string("OpenCV error: ") + e.what());
        return false;
    }
}

bool PlayerDetector::isInitialized() const {
    return impl_->initialized;
}

std::vector<Detection> PlayerDetector::detect(const cv::Mat& frame) {
    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Not initialized");
        return {};
    }

    if (frame.empty()) {
        HM_LOG_WARN(TAG, "Empty frame");
        return {};
    }

    cv::Mat blob = impl_->preprocess(frame);
    impl_->net.setInput(blob);

    std::vector<cv::Mat> outputs;
    impl_->net.forward(outputs, impl_->net.getUnconnectedOutLayersNames());

    if (outputs.empty()) return {};

    return impl_->postprocess(outputs[0], frame.cols, frame.rows);
}

std::vector<std::vector<Detection>> PlayerDetector::detectBatch(
    const std::vector<cv::Mat>& frames) {
    std::vector<std::vector<Detection>> results;
    results.reserve(frames.size());
    for (const auto& frame : frames) {
        results.push_back(detect(frame));
    }
    return results;
}

void PlayerDetector::setConfidenceThreshold(float threshold) {
    impl_->config.confidenceThreshold = threshold;
}

void PlayerDetector::setNmsIouThreshold(float threshold) {
    impl_->config.nmsIouThreshold = threshold;
}

} // namespace hm::pose
