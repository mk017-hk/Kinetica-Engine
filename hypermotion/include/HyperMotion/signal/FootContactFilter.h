#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::signal {

struct FootContactConfig {
    float velocityThreshold = 2.0f;  // cm/s
    float heightThreshold = 5.0f;    // cm above ground
    float transitionSmoothing = 0.1f; // Blend factor for in/out of contact
    float groundHeight = 0.0f;       // Ground plane Y coordinate
};

class FootContactFilter {
public:
    explicit FootContactFilter(const FootContactConfig& config = {});
    ~FootContactFilter();

    // Process skeleton frames in-place
    void process(std::vector<SkeletonFrame>& frames);

    // Detect foot contact for a sequence
    struct ContactState {
        bool leftFootContact = false;
        bool rightFootContact = false;
        float leftBlend = 0.0f;  // 0 = no contact, 1 = full contact
        float rightBlend = 0.0f;
    };

    std::vector<ContactState> detectContacts(const std::vector<SkeletonFrame>& frames);

private:
    FootContactConfig config_;
};

} // namespace hm::signal
