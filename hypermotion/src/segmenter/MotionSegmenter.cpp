#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>

namespace hm::segmenter {

static constexpr const char* TAG = "MotionSegmenter";

struct MotionSegmenter::Impl {
    MotionSegmenterConfig config;
    TemporalConvNetOnnx tcn;
    MotionFeatureExtractor featureExtractor;
    bool initialized = false;
};

MotionSegmenter::MotionSegmenter(const MotionSegmenterConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionSegmenter::~MotionSegmenter() = default;
MotionSegmenter::MotionSegmenter(MotionSegmenter&&) noexcept = default;
MotionSegmenter& MotionSegmenter::operator=(MotionSegmenter&&) noexcept = default;

bool MotionSegmenter::initialize() {
    if (!impl_->config.modelPath.empty()) {
        if (!impl_->tcn.load(impl_->config.modelPath, impl_->config.useGPU)) {
            HM_LOG_WARN(TAG, "Could not load TCN model; classification will default to Unknown");
        }
    } else {
        HM_LOG_WARN(TAG, "No model path specified; classification will default to Unknown");
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Motion segmenter initialized (ONNX inference)");
    return true;
}

bool MotionSegmenter::isInitialized() const {
    return impl_->initialized;
}

std::vector<std::array<float, MOTION_TYPE_COUNT>> MotionSegmenter::classifyFrames(
    const std::vector<SkeletonFrame>& frames) {

    int numFrames = static_cast<int>(frames.size());
    std::vector<std::array<float, MOTION_TYPE_COUNT>> probabilities(numFrames);

    if (!impl_->initialized || numFrames == 0) return probabilities;

    // Extract 70D features per frame
    auto features = impl_->featureExtractor.extractSequence(frames);

    // Convert to array format for TCN
    std::vector<std::array<float, 70>> featureArrays(numFrames);
    for (int f = 0; f < numFrames; ++f) {
        std::copy(features[f].begin(), features[f].end(), featureArrays[f].begin());
    }

    if (!impl_->tcn.isLoaded()) {
        // No model: everything is Unknown
        for (auto& p : probabilities) {
            p.fill(0.0f);
            p[static_cast<int>(MotionType::Unknown)] = 1.0f;
        }
        return probabilities;
    }

    // Sliding window inference with vote accumulation
    std::vector<std::vector<float>> allLogits(numFrames,
        std::vector<float>(MOTION_TYPE_COUNT, 0.0f));
    std::vector<int> voteCounts(numFrames, 0);

    int windowSize = std::min(impl_->config.slidingWindowSize, numFrames);
    int stride = impl_->config.slidingWindowStride;

    for (int start = 0; start < numFrames; start += stride) {
        int end = std::min(start + windowSize, numFrames);

        // Extract window
        std::vector<std::array<float, 70>> window(
            featureArrays.begin() + start, featureArrays.begin() + end);

        auto logits = impl_->tcn.classify(window);

        // Accumulate
        for (int t = 0; t < static_cast<int>(logits.size()); ++t) {
            int frameIdx = start + t;
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                allLogits[frameIdx][c] += logits[t][c];
            }
            voteCounts[frameIdx]++;
        }

        if (end >= numFrames) break;
    }

    // Average logits -> softmax
    for (int f = 0; f < numFrames; ++f) {
        if (voteCounts[f] > 0) {
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                allLogits[f][c] /= voteCounts[f];
            }

            float maxLogit = *std::max_element(allLogits[f].begin(), allLogits[f].end());
            float sumExp = 0.0f;
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                probabilities[f][c] = std::exp(allLogits[f][c] - maxLogit);
                sumExp += probabilities[f][c];
            }
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                probabilities[f][c] /= sumExp;
            }
        } else {
            probabilities[f].fill(0.0f);
            probabilities[f][static_cast<int>(MotionType::Unknown)] = 1.0f;
        }
    }

    return probabilities;
}

std::vector<MotionSegment> MotionSegmenter::segment(
    const std::vector<SkeletonFrame>& frames, int trackingID) {

    if (!impl_->initialized || frames.empty()) return {};

    int numFrames = static_cast<int>(frames.size());
    HM_LOG_INFO(TAG, "Segmenting " + std::to_string(numFrames) + " frames");

    auto probabilities = classifyFrames(frames);

    // Argmax per-frame
    std::vector<int> labels(numFrames);
    std::vector<float> confidences(numFrames);
    for (int f = 0; f < numFrames; ++f) {
        int bestClass = 0;
        float bestProb = probabilities[f][0];
        for (int c = 1; c < MOTION_TYPE_COUNT; ++c) {
            if (probabilities[f][c] > bestProb) {
                bestProb = probabilities[f][c];
                bestClass = c;
            }
        }
        labels[f] = bestClass;
        confidences[f] = bestProb;
    }

    // Merge consecutive same-label frames
    std::vector<MotionSegment> segments;
    int segStart = 0;

    for (int f = 1; f <= numFrames; ++f) {
        if (f == numFrames || labels[f] != labels[segStart]) {
            MotionSegment seg;
            seg.type = static_cast<MotionType>(labels[segStart]);
            seg.startFrame = segStart;
            seg.endFrame = f - 1;
            seg.trackingID = trackingID;

            float avgConf = 0.0f, avgVel = 0.0f;
            Vec3 avgDir{0, 0, 0};
            int count = f - segStart;

            for (int i = segStart; i < f; ++i) {
                avgConf += confidences[i];
                avgVel += frames[i].rootVelocity.length();
                avgDir = avgDir + frames[i].rootVelocity;
            }

            seg.confidence = avgConf / count;
            seg.avgVelocity = avgVel / count;
            seg.avgDirection = avgDir.normalized();

            segments.push_back(seg);
            segStart = f;
        }
    }

    // Enforce minimum segment length
    std::vector<MotionSegment> merged;
    for (auto& seg : segments) {
        int segLen = seg.endFrame - seg.startFrame + 1;
        if (segLen < impl_->config.minSegmentLength && !merged.empty()) {
            merged.back().endFrame = seg.endFrame;
            int totalLen = merged.back().endFrame - merged.back().startFrame + 1;
            int prevLen = totalLen - segLen;
            merged.back().avgVelocity =
                (merged.back().avgVelocity * prevLen + seg.avgVelocity * segLen) / totalLen;
            merged.back().confidence =
                (merged.back().confidence * prevLen + seg.confidence * segLen) / totalLen;
        } else {
            merged.push_back(seg);
        }
    }

    HM_LOG_INFO(TAG, "Found " + std::to_string(merged.size()) + " segments");
    return merged;
}

} // namespace hm::segmenter
