#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/MotionDiffusionModel.h"
#include <memory>
#include <string>
#include <vector>

namespace hm::ml {

struct MotionGeneratorConfig {
    MotionDiffusionConfig diffusionConfig;
    bool enableJointLimits = true;
    bool enableFootContactCleanup = true;
    bool enablePlausibilityCheck = true;
};

class MotionGenerator {
public:
    explicit MotionGenerator(const MotionGeneratorConfig& config = {});
    ~MotionGenerator();

    MotionGenerator(const MotionGenerator&) = delete;
    MotionGenerator& operator=(const MotionGenerator&) = delete;
    MotionGenerator(MotionGenerator&&) noexcept;
    MotionGenerator& operator=(MotionGenerator&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// High-level: MotionCondition -> GeneratedMotion (64 frames)
    GeneratedMotion generate(const MotionCondition& condition);

    /// Batch generation
    std::vector<GeneratedMotion> generateBatch(const std::vector<MotionCondition>& conditions);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
