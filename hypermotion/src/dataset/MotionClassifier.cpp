#include "HyperMotion/dataset/MotionClassifier.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/core/Logger.h"

namespace hm::dataset {

static constexpr const char* TAG = "MotionClassifier";

struct MotionClassifier::Impl {
    std::string modelPath;
    segmenter::MotionSegmenter segmenter;
    segmenter::MotionFeatureExtractor featureExtractor;
    bool initialized = false;

    Impl(const std::string& path)
        : modelPath(path) {
        segmenter::MotionSegmenterConfig cfg;
        cfg.modelPath = path;
        cfg.minSegmentLength = 5;
        segmenter = segmenter::MotionSegmenter(cfg);
    }
};

MotionClassifier::MotionClassifier(const std::string& modelPath)
    : impl_(std::make_unique<Impl>(modelPath)) {}

MotionClassifier::~MotionClassifier() = default;
MotionClassifier::MotionClassifier(MotionClassifier&&) noexcept = default;
MotionClassifier& MotionClassifier::operator=(MotionClassifier&&) noexcept = default;

bool MotionClassifier::initialize() {
    impl_->initialized = impl_->segmenter.initialize();
    if (!impl_->initialized) {
        HM_LOG_WARN(TAG, "TCN model not loaded, using heuristic classification");
        impl_->initialized = true;  // heuristic fallback is always available
    }
    return impl_->initialized;
}

ClassificationResult MotionClassifier::classify(const AnimClip& clip) {
    ClassificationResult result;

    if (clip.frames.empty()) {
        result.type = MotionType::Unknown;
        result.label = "Unknown";
        return result;
    }

    // Try TCN-based classification via per-frame probabilities
    auto frameProbs = impl_->segmenter.classifyFrames(clip.frames);

    if (!frameProbs.empty()) {
        // Average probabilities across all frames
        std::array<float, MOTION_TYPE_COUNT> avgProbs{};
        for (const auto& fp : frameProbs) {
            for (int i = 0; i < MOTION_TYPE_COUNT; ++i) {
                avgProbs[i] += fp[i];
            }
        }
        int n = static_cast<int>(frameProbs.size());
        for (int i = 0; i < MOTION_TYPE_COUNT; ++i) {
            avgProbs[i] /= n;
        }

        // Find best class
        int bestIdx = 0;
        float bestProb = avgProbs[0];
        for (int i = 1; i < MOTION_TYPE_COUNT; ++i) {
            if (avgProbs[i] > bestProb) {
                bestProb = avgProbs[i];
                bestIdx = i;
            }
        }

        result.type = static_cast<MotionType>(bestIdx);
        result.confidence = bestProb;
        result.label = MOTION_TYPE_NAMES[bestIdx];
        result.probabilities = avgProbs;
    } else {
        // Heuristic fallback
        int startFrame = 0;
        int endFrame = static_cast<int>(clip.frames.size()) - 1;
        result.type = impl_->featureExtractor.classifyHeuristic(
            clip.frames, startFrame, endFrame);
        result.label = MOTION_TYPE_NAMES[static_cast<int>(result.type)];
        result.confidence = 0.6f;  // lower confidence for heuristic
    }

    return result;
}

std::vector<ClassificationResult> MotionClassifier::classifyBatch(
    const std::vector<AnimClip>& clips) {
    std::vector<ClassificationResult> results;
    results.reserve(clips.size());
    for (const auto& clip : clips) {
        results.push_back(classify(clip));
    }
    HM_LOG_INFO(TAG, "Classified " + std::to_string(clips.size()) + " clips");
    return results;
}

} // namespace hm::dataset
