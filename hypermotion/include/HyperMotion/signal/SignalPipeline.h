#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/signal/OutlierFilter.h"
#include "HyperMotion/signal/SavitzkyGolay.h"
#include "HyperMotion/signal/ButterworthFilter.h"
#include "HyperMotion/signal/QuaternionSmoother.h"
#include "HyperMotion/signal/FootContactFilter.h"
#include <vector>

namespace hm::signal {

struct SignalPipelineConfig {
    bool enableOutlierFilter = true;
    bool enableSavitzkyGolay = true;
    bool enableButterworth = true;
    bool enableQuaternionSmoothing = true;
    bool enableFootContact = true;

    OutlierFilterConfig outlierConfig;
    SavitzkyGolayConfig sgConfig;
    ButterworthConfig butterworthConfig;
    QuaternionSmootherConfig quatConfig;
    FootContactConfig footConfig;
};

class SignalPipeline {
public:
    explicit SignalPipeline(const SignalPipelineConfig& config = {});
    ~SignalPipeline();

    // Process entire sequence in-place, chaining all enabled filters
    void process(std::vector<SkeletonFrame>& frames);

    // Enable/disable individual stages
    void setStageEnabled(const std::string& stageName, bool enabled);

private:
    SignalPipelineConfig config_;
    OutlierFilter outlierFilter_;
    SavitzkyGolay savitzkyGolay_;
    ButterworthFilter butterworth_;
    QuaternionSmoother quatSmoother_;
    FootContactFilter footContact_;
};

} // namespace hm::signal
