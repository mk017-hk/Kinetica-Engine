#pragma once

#include "HyperMotion/core/Types.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

namespace hm::pose {

struct PlayerDetectorConfig {
    std::string modelPath;
    float confidenceThreshold = 0.5f;
    float nmsIouThreshold = 0.45f;
    int inputWidth = 640;
    int inputHeight = 640;
    int maxDetections = 30;
    bool useTensorRT = false;
    std::string tensorrtCachePath;
};

class PlayerDetector {
public:
    explicit PlayerDetector(const PlayerDetectorConfig& config);
    ~PlayerDetector();

    PlayerDetector(const PlayerDetector&) = delete;
    PlayerDetector& operator=(const PlayerDetector&) = delete;
    PlayerDetector(PlayerDetector&&) noexcept;
    PlayerDetector& operator=(PlayerDetector&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    std::vector<Detection> detect(const cv::Mat& frame);
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

    void setConfidenceThreshold(float threshold);
    void setNmsIouThreshold(float threshold);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
