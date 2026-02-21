#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::segmenter {

static constexpr const char* TAG = "MotionSegmenter";

struct MotionSegmenter::Impl {
    MotionSegmenterConfig config;
    TemporalConvNet model;
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
    try {
        impl_->model = TemporalConvNet(
            MotionFeatureExtractor::FEATURE_DIM, 128, MOTION_TYPE_COUNT, 3, 0.1f);

        if (!impl_->config.modelPath.empty()) {
            try {
                torch::serialize::InputArchive archive;
                archive.load_from(impl_->config.modelPath);
                impl_->model->load(archive);
                HM_LOG_INFO(TAG, "Loaded TCN model from: " + impl_->config.modelPath);
            } catch (const std::exception& e) {
                HM_LOG_WARN(TAG, std::string("Could not load model: ") + e.what());
                HM_LOG_INFO(TAG, "Using untrained TCN — classification results will be random");
            }
        }

        impl_->model->eval();
        impl_->initialized = true;
        HM_LOG_INFO(TAG, "Motion segmenter initialized");
        return true;

    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Initialization failed: ") + e.what());
        return false;
    }
}

bool MotionSegmenter::isInitialized() const {
    return impl_->initialized;
}

std::vector<std::array<float, MOTION_TYPE_COUNT>> MotionSegmenter::classifyFrames(
    const std::vector<SkeletonFrame>& frames) {

    int numFrames = static_cast<int>(frames.size());
    std::vector<std::array<float, MOTION_TYPE_COUNT>> probabilities(numFrames);

    if (!impl_->initialized || numFrames == 0) return probabilities;

    // Extract features
    auto features = impl_->featureExtractor.extractSequence(frames);

    // Process using sliding windows
    std::vector<std::vector<float>> allLogits(numFrames,
        std::vector<float>(MOTION_TYPE_COUNT, 0.0f));
    std::vector<int> voteCounts(numFrames, 0);

    int windowSize = std::min(impl_->config.slidingWindowSize, numFrames);
    int stride = impl_->config.slidingWindowStride;

    torch::NoGradGuard noGrad;

    for (int start = 0; start < numFrames; start += stride) {
        int end = std::min(start + windowSize, numFrames);
        int chunkLen = end - start;

        // Build input tensor: [1, features, time]
        auto inputTensor = torch::zeros({1, MotionFeatureExtractor::FEATURE_DIM, chunkLen});
        auto accessor = inputTensor.accessor<float, 3>();

        for (int t = 0; t < chunkLen; ++t) {
            for (int f = 0; f < MotionFeatureExtractor::FEATURE_DIM; ++f) {
                accessor[0][f][t] = features[start + t][f];
            }
        }

        // Forward pass
        auto output = impl_->model->forward(inputTensor);  // [1, classes, time]
        auto outputAcc = output.accessor<float, 3>();

        // Accumulate logits
        for (int t = 0; t < chunkLen; ++t) {
            int frameIdx = start + t;
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                allLogits[frameIdx][c] += outputAcc[0][c][t];
            }
            voteCounts[frameIdx]++;
        }

        if (end >= numFrames) break;
    }

    // Average logits and compute softmax
    for (int f = 0; f < numFrames; ++f) {
        if (voteCounts[f] > 0) {
            float maxLogit = *std::max_element(allLogits[f].begin(), allLogits[f].end());
            float sumExp = 0.0f;

            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                allLogits[f][c] /= voteCounts[f];
            }

            maxLogit = *std::max_element(allLogits[f].begin(), allLogits[f].end());

            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                probabilities[f][c] = std::exp(allLogits[f][c] - maxLogit);
                sumExp += probabilities[f][c];
            }
            for (int c = 0; c < MOTION_TYPE_COUNT; ++c) {
                probabilities[f][c] /= sumExp;
            }
        } else {
            // Default to Unknown
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

    // Argmax per-frame classification
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

    // Merge consecutive same-label frames into segments
    std::vector<MotionSegment> segments;
    int segStart = 0;

    for (int f = 1; f <= numFrames; ++f) {
        if (f == numFrames || labels[f] != labels[segStart]) {
            MotionSegment seg;
            seg.type = static_cast<MotionType>(labels[segStart]);
            seg.startFrame = segStart;
            seg.endFrame = f - 1;
            seg.trackingID = trackingID;

            // Average confidence
            float avgConf = 0.0f;
            float avgVel = 0.0f;
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

    // Enforce minimum segment length: merge short segments with neighbors
    std::vector<MotionSegment> mergedSegments;
    for (auto& seg : segments) {
        int segLen = seg.endFrame - seg.startFrame + 1;

        if (segLen < impl_->config.minSegmentLength && !mergedSegments.empty()) {
            // Merge with previous segment
            mergedSegments.back().endFrame = seg.endFrame;
            // Recompute averages
            int totalLen = mergedSegments.back().endFrame - mergedSegments.back().startFrame + 1;
            int prevLen = totalLen - segLen;
            mergedSegments.back().avgVelocity =
                (mergedSegments.back().avgVelocity * prevLen + seg.avgVelocity * segLen) / totalLen;
            mergedSegments.back().confidence =
                (mergedSegments.back().confidence * prevLen + seg.confidence * segLen) / totalLen;
        } else {
            mergedSegments.push_back(seg);
        }
    }

    HM_LOG_INFO(TAG, "Found " + std::to_string(mergedSegments.size()) + " segments");
    return mergedSegments;
}

} // namespace hm::segmenter
