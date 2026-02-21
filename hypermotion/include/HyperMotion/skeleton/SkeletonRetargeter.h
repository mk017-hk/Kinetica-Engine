#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace hm::skeleton {

struct RetargetMapping {
    std::string sourceJointName;
    std::string targetJointName;
    float scaleCompensation = 1.0f;
};

struct SkeletonRetargeterConfig {
    std::vector<RetargetMapping> mappings;
    float globalScale = 1.0f;
    bool preserveRootMotion = true;
};

class SkeletonRetargeter {
public:
    explicit SkeletonRetargeter(const SkeletonRetargeterConfig& config = {});
    ~SkeletonRetargeter();

    SkeletonRetargeter(const SkeletonRetargeter&) = delete;
    SkeletonRetargeter& operator=(const SkeletonRetargeter&) = delete;
    SkeletonRetargeter(SkeletonRetargeter&&) noexcept;
    SkeletonRetargeter& operator=(SkeletonRetargeter&&) noexcept;

    // Set up automatic mapping by joint name similarity
    void autoMap(const std::vector<std::string>& sourceJointNames,
                 const std::vector<std::string>& targetJointNames);

    // Retarget a single frame
    SkeletonFrame retarget(const SkeletonFrame& sourceFrame) const;

    // Retarget an entire clip
    AnimClip retargetClip(const AnimClip& sourceClip) const;

    // Set scale compensation for a specific mapping
    void setScaleCompensation(const std::string& jointName, float scale);

    const std::vector<RetargetMapping>& getMappings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::skeleton
