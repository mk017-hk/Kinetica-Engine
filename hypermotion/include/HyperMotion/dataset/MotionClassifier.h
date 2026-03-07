#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include <memory>
#include <string>
#include <vector>

namespace hm::dataset {

/// Classification result for a single animation clip.
struct ClassificationResult {
    MotionType type = MotionType::Unknown;
    float confidence = 0.0f;
    std::string label;
    std::array<float, MOTION_TYPE_COUNT> probabilities{};
};

/// Classifies extracted animation clips into motion categories using
/// the existing MotionFeatureExtractor heuristics and, when available,
/// the TemporalConvNet ONNX model.
class MotionClassifier {
public:
    explicit MotionClassifier(const std::string& modelPath = "");
    ~MotionClassifier();

    MotionClassifier(const MotionClassifier&) = delete;
    MotionClassifier& operator=(const MotionClassifier&) = delete;
    MotionClassifier(MotionClassifier&&) noexcept;
    MotionClassifier& operator=(MotionClassifier&&) noexcept;

    bool initialize();

    /// Classify a single animation clip.
    ClassificationResult classify(const AnimClip& clip);

    /// Classify a batch of clips.
    std::vector<ClassificationResult> classifyBatch(const std::vector<AnimClip>& clips);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::dataset
