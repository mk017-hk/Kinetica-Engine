#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/core/Logger.h"

namespace hm::signal {

static constexpr const char* TAG = "SignalPipeline";

SignalPipeline::SignalPipeline(const SignalPipelineConfig& config)
    : config_(config)
    , outlierFilter_(config.outlierConfig)
    , savitzkyGolay_(config.sgConfig)
    , butterworth_(config.butterworthConfig)
    , quatSmoother_(config.quatConfig)
    , footContact_(config.footConfig) {}

SignalPipeline::~SignalPipeline() = default;

void SignalPipeline::process(std::vector<SkeletonFrame>& frames) {
    if (frames.empty()) return;

    HM_LOG_INFO(TAG, "Processing " + std::to_string(frames.size()) + " frames through signal pipeline");

    // Stage 1: Outlier removal
    if (config_.enableOutlierFilter) {
        HM_LOG_DEBUG(TAG, "Stage 1: Outlier filter");
        outlierFilter_.process(frames);
    }

    // Stage 2: Savitzky-Golay polynomial smoothing
    if (config_.enableSavitzkyGolay) {
        HM_LOG_DEBUG(TAG, "Stage 2: Savitzky-Golay filter");
        savitzkyGolay_.process(frames);
    }

    // Stage 3: Butterworth low-pass filter
    if (config_.enableButterworth) {
        HM_LOG_DEBUG(TAG, "Stage 3: Butterworth filter");
        butterworth_.process(frames);
    }

    // Stage 4: Quaternion smoothing
    if (config_.enableQuaternionSmoothing) {
        HM_LOG_DEBUG(TAG, "Stage 4: Quaternion smoother");
        quatSmoother_.process(frames);
    }

    // Stage 5: Foot contact cleanup
    if (config_.enableFootContact) {
        HM_LOG_DEBUG(TAG, "Stage 5: Foot contact filter");
        footContact_.process(frames);
    }

    HM_LOG_INFO(TAG, "Signal pipeline complete");
}

void SignalPipeline::setStageEnabled(const std::string& stageName, bool enabled) {
    if (stageName == "outlier") config_.enableOutlierFilter = enabled;
    else if (stageName == "savitzky_golay" || stageName == "sg") config_.enableSavitzkyGolay = enabled;
    else if (stageName == "butterworth") config_.enableButterworth = enabled;
    else if (stageName == "quaternion") config_.enableQuaternionSmoothing = enabled;
    else if (stageName == "foot_contact") config_.enableFootContact = enabled;
    else HM_LOG_WARN(TAG, "Unknown stage: " + stageName);
}

} // namespace hm::signal
