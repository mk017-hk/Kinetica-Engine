#include "HyperMotion/pose/SinglePoseEstimator.h"
#include "HyperMotion/core/Logger.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace hm::pose {

static constexpr const char* TAG = "SinglePoseEstimator";

// ImageNet normalization constants
static constexpr float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
static constexpr float IMAGENET_STD[] = {0.229f, 0.224f, 0.225f};

struct SinglePoseEstimator::Impl {
    SinglePoseEstimatorConfig config;
    cv::dnn::Net net;
    bool initialized = false;

    cv::Mat cropAndResize(const cv::Mat& frame, const BBox& bbox,
                          float& scaleX, float& scaleY,
                          float& offsetX, float& offsetY) {
        // Apply padding
        float pad = config.bboxPadding;
        float bx = bbox.x - bbox.width * pad;
        float by = bbox.y - bbox.height * pad;
        float bw = bbox.width * (1.0f + 2.0f * pad);
        float bh = bbox.height * (1.0f + 2.0f * pad);

        // Adjust aspect ratio to match input (width:height = 192:256 = 3:4)
        float targetAspect = static_cast<float>(config.inputWidth) / config.inputHeight;
        float currentAspect = bw / std::max(bh, 1.0f);

        if (currentAspect > targetAspect) {
            // Too wide, increase height
            float newH = bw / targetAspect;
            by -= (newH - bh) * 0.5f;
            bh = newH;
        } else {
            // Too tall, increase width
            float newW = bh * targetAspect;
            bx -= (newW - bw) * 0.5f;
            bw = newW;
        }

        // Clamp to frame bounds
        int x1 = std::max(0, static_cast<int>(bx));
        int y1 = std::max(0, static_cast<int>(by));
        int x2 = std::min(frame.cols, static_cast<int>(bx + bw));
        int y2 = std::min(frame.rows, static_cast<int>(by + bh));

        if (x2 <= x1 || y2 <= y1) {
            return cv::Mat();
        }

        cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        scaleX = static_cast<float>(x2 - x1) / config.inputWidth;
        scaleY = static_cast<float>(y2 - y1) / config.inputHeight;
        offsetX = static_cast<float>(x1);
        offsetY = static_cast<float>(y1);

        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(config.inputWidth, config.inputHeight));
        return resized;
    }

    cv::Mat normalizeImage(const cv::Mat& img) {
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(floatImg, channels);

        // BGR to RGB order for ImageNet normalization
        // OpenCV loads as BGR, so channels[0]=B, [1]=G, [2]=R
        channels[0] = (channels[0] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]; // B channel
        channels[1] = (channels[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]; // G channel
        channels[2] = (channels[2] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]; // R channel

        cv::Mat normalized;
        cv::merge(channels, normalized);
        return normalized;
    }

    // Sub-pixel refinement via quadratic peak fitting on heatmap
    Vec2 refinePeak(const cv::Mat& heatmap, int px, int py) {
        float dx = 0.0f, dy = 0.0f;

        if (px > 0 && px < heatmap.cols - 1) {
            float left = heatmap.at<float>(py, px - 1);
            float right = heatmap.at<float>(py, px + 1);
            float center = heatmap.at<float>(py, px);
            float denom = 2.0f * center - left - right;
            if (std::abs(denom) > 1e-6f) {
                dx = (left - right) / (2.0f * denom);
                dx = std::clamp(dx, -0.5f, 0.5f);
            }
        }

        if (py > 0 && py < heatmap.rows - 1) {
            float top = heatmap.at<float>(py - 1, px);
            float bottom = heatmap.at<float>(py + 1, px);
            float center = heatmap.at<float>(py, px);
            float denom = 2.0f * center - top - bottom;
            if (std::abs(denom) > 1e-6f) {
                dy = (top - bottom) / (2.0f * denom);
                dy = std::clamp(dy, -0.5f, 0.5f);
            }
        }

        return {static_cast<float>(px) + dx, static_cast<float>(py) + dy};
    }

    std::array<Keypoint2D, COCO_KEYPOINTS> extractKeypoints(
        const cv::Mat& heatmaps, float scaleX, float scaleY,
        float offsetX, float offsetY, int frameWidth, int frameHeight) {

        std::array<Keypoint2D, COCO_KEYPOINTS> keypoints{};
        int hmWidth = heatmaps.size[3];
        int hmHeight = heatmaps.size[2];

        float hmScaleX = static_cast<float>(config.inputWidth) / hmWidth;
        float hmScaleY = static_cast<float>(config.inputHeight) / hmHeight;

        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            // Extract single heatmap for keypoint k
            cv::Mat hm(hmHeight, hmWidth, CV_32F);
            for (int y = 0; y < hmHeight; ++y) {
                for (int x = 0; x < hmWidth; ++x) {
                    hm.at<float>(y, x) = heatmaps.ptr<float>(0, k, y)[x];
                }
            }

            // Find peak
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(hm, &minVal, &maxVal, &minLoc, &maxLoc);

            // Sub-pixel refinement
            Vec2 refined = refinePeak(hm, maxLoc.x, maxLoc.y);

            // Map back to original frame coordinates
            float px = (refined.x * hmScaleX) * scaleX + offsetX;
            float py = (refined.y * hmScaleY) * scaleY + offsetY;

            keypoints[k].position.x = px / frameWidth;
            keypoints[k].position.y = py / frameHeight;
            keypoints[k].confidence = static_cast<float>(maxVal);
        }

        return keypoints;
    }

    std::array<Keypoint2D, COCO_KEYPOINTS> flipKeypoints(
        const std::array<Keypoint2D, COCO_KEYPOINTS>& kps) {
        auto flipped = kps;
        for (auto& kp : flipped) {
            kp.position.x = 1.0f - kp.position.x;
        }
        // Swap left-right pairs
        for (const auto& [l, r] : SinglePoseEstimator::kFlipPairs) {
            std::swap(flipped[l], flipped[r]);
        }
        return flipped;
    }

    std::array<Keypoint2D, COCO_KEYPOINTS> averageKeypoints(
        const std::array<Keypoint2D, COCO_KEYPOINTS>& a,
        const std::array<Keypoint2D, COCO_KEYPOINTS>& b) {
        std::array<Keypoint2D, COCO_KEYPOINTS> result;
        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            result[k].position.x = (a[k].position.x + b[k].position.x) * 0.5f;
            result[k].position.y = (a[k].position.y + b[k].position.y) * 0.5f;
            result[k].confidence = (a[k].confidence + b[k].confidence) * 0.5f;
        }
        return result;
    }
};

SinglePoseEstimator::SinglePoseEstimator(const SinglePoseEstimatorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

SinglePoseEstimator::~SinglePoseEstimator() = default;
SinglePoseEstimator::SinglePoseEstimator(SinglePoseEstimator&&) noexcept = default;
SinglePoseEstimator& SinglePoseEstimator::operator=(SinglePoseEstimator&&) noexcept = default;

bool SinglePoseEstimator::initialize() {
    try {
        if (impl_->config.modelPath.empty()) {
            HM_LOG_ERROR(TAG, "Model path is empty");
            return false;
        }

        impl_->net = cv::dnn::readNetFromONNX(impl_->config.modelPath);
        if (impl_->net.empty()) {
            HM_LOG_ERROR(TAG, "Failed to load model: " + impl_->config.modelPath);
            return false;
        }

        impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#ifdef HM_USE_CUDA
        try {
            impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            HM_LOG_INFO(TAG, "Using CUDA backend for pose estimation");
        } catch (...) {
            impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            HM_LOG_WARN(TAG, "CUDA not available, falling back to CPU");
        }
#endif

        impl_->initialized = true;
        HM_LOG_INFO(TAG, "Initialized HRNet pose estimator");
        return true;

    } catch (const cv::Exception& e) {
        HM_LOG_ERROR(TAG, std::string("OpenCV error: ") + e.what());
        return false;
    }
}

bool SinglePoseEstimator::isInitialized() const {
    return impl_->initialized;
}

std::array<Keypoint2D, COCO_KEYPOINTS> SinglePoseEstimator::estimate(
    const cv::Mat& frame, const BBox& bbox) {

    if (!impl_->initialized || frame.empty()) {
        return {};
    }

    float scaleX, scaleY, offsetX, offsetY;
    cv::Mat cropped = impl_->cropAndResize(frame, bbox, scaleX, scaleY, offsetX, offsetY);

    if (cropped.empty()) return {};

    cv::Mat normalized = impl_->normalizeImage(cropped);
    cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false);

    impl_->net.setInput(blob);
    cv::Mat heatmaps = impl_->net.forward();

    auto keypoints = impl_->extractKeypoints(heatmaps, scaleX, scaleY,
                                              offsetX, offsetY,
                                              frame.cols, frame.rows);

    // Flip test augmentation
    if (impl_->config.useFlipTest) {
        cv::Mat flippedCrop;
        cv::flip(cropped, flippedCrop, 1); // horizontal flip

        cv::Mat flippedNorm = impl_->normalizeImage(flippedCrop);
        cv::Mat flippedBlob = cv::dnn::blobFromImage(flippedNorm, 1.0, cv::Size(), cv::Scalar(), false, false);

        impl_->net.setInput(flippedBlob);
        cv::Mat flippedHeatmaps = impl_->net.forward();

        auto flippedKps = impl_->extractKeypoints(flippedHeatmaps, scaleX, scaleY,
                                                    offsetX, offsetY,
                                                    frame.cols, frame.rows);
        flippedKps = impl_->flipKeypoints(flippedKps);
        keypoints = impl_->averageKeypoints(keypoints, flippedKps);
    }

    return keypoints;
}

std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>> SinglePoseEstimator::estimateBatch(
    const cv::Mat& frame, const std::vector<BBox>& bboxes) {

    std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>> results;
    results.reserve(bboxes.size());
    for (const auto& bbox : bboxes) {
        results.push_back(estimate(frame, bbox));
    }
    return results;
}

} // namespace hm::pose
