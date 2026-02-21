#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::signal {

struct QuaternionSmootherConfig {
    float smoothingFactor = 0.3f;
};

class QuaternionSmoother {
public:
    explicit QuaternionSmoother(const QuaternionSmootherConfig& config = {});
    ~QuaternionSmoother();

    // Process skeleton frames in-place with SLERP-based smoothing
    void process(std::vector<SkeletonFrame>& frames);

    // Smooth a single quaternion sequence
    static void smoothQuatSequence(std::vector<Quat>& quaternions, float alpha);

private:
    QuaternionSmootherConfig config_;
};

} // namespace hm::signal
