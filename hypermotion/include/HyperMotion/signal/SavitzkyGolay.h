#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::signal {

struct SavitzkyGolayConfig {
    int windowSize = 7;     // Must be odd
    int polyOrder = 3;      // Polynomial order
};

class SavitzkyGolay {
public:
    explicit SavitzkyGolay(const SavitzkyGolayConfig& config = {});
    ~SavitzkyGolay();

    // Process skeleton frames in-place
    void process(std::vector<SkeletonFrame>& frames);

    // Process a single 1D signal
    static void filterChannel(std::vector<float>& signal, int windowSize, int polyOrder);

    // Compute Savitzky-Golay coefficients via Vandermonde matrix + least squares
    static std::vector<float> computeCoefficients(int windowSize, int polyOrder);

private:
    SavitzkyGolayConfig config_;
};

} // namespace hm::signal
