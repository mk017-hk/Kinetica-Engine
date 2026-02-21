#include "HyperMotion/skeleton/SkeletonRetargeter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace hm::skeleton {

static constexpr const char* TAG = "SkeletonRetargeter";

struct SkeletonRetargeter::Impl {
    SkeletonRetargeterConfig config;

    // Source joint name -> target joint index mapping
    std::unordered_map<std::string, int> sourceToTargetIdx;
    std::unordered_map<std::string, float> scaleCompensations;

    static std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    static std::string stripPrefixSuffix(const std::string& name) {
        // Remove common prefixes like "mixamorig:", "Bip01_", etc.
        std::string s = name;
        auto colonPos = s.find(':');
        if (colonPos != std::string::npos) {
            s = s.substr(colonPos + 1);
        }
        // Remove leading underscores or "Bip01_" etc.
        if (s.find("Bip01_") == 0) s = s.substr(6);
        if (s.find("Bip01 ") == 0) s = s.substr(6);
        return s;
    }

    static float nameSimilarity(const std::string& a, const std::string& b) {
        std::string la = toLower(stripPrefixSuffix(a));
        std::string lb = toLower(stripPrefixSuffix(b));

        if (la == lb) return 1.0f;

        // Check if one contains the other
        if (la.find(lb) != std::string::npos || lb.find(la) != std::string::npos) {
            float longer = static_cast<float>(std::max(la.size(), lb.size()));
            float shorter = static_cast<float>(std::min(la.size(), lb.size()));
            return shorter / longer;
        }

        // Levenshtein distance based similarity
        int n = static_cast<int>(la.size());
        int m = static_cast<int>(lb.size());
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

        for (int i = 0; i <= n; ++i) dp[i][0] = i;
        for (int j = 0; j <= m; ++j) dp[0][j] = j;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                int cost = (la[i - 1] == lb[j - 1]) ? 0 : 1;
                dp[i][j] = std::min({dp[i - 1][j] + 1,
                                     dp[i][j - 1] + 1,
                                     dp[i - 1][j - 1] + cost});
            }
        }

        float maxLen = static_cast<float>(std::max(n, m));
        return maxLen > 0 ? 1.0f - static_cast<float>(dp[n][m]) / maxLen : 0.0f;
    }

    int findTargetJointIndex(const std::string& name) {
        // First check exact match
        std::string lower = toLower(name);
        for (int i = 0; i < JOINT_COUNT; ++i) {
            if (toLower(JOINT_NAMES[i]) == lower) return i;
        }

        // Try stripped match
        std::string stripped = toLower(stripPrefixSuffix(name));
        for (int i = 0; i < JOINT_COUNT; ++i) {
            if (toLower(JOINT_NAMES[i]) == stripped) return i;
        }

        return -1;
    }
};

SkeletonRetargeter::SkeletonRetargeter(const SkeletonRetargeterConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;

    // Build mapping from config
    for (const auto& mapping : config.mappings) {
        int targetIdx = impl_->findTargetJointIndex(mapping.targetJointName);
        if (targetIdx >= 0) {
            impl_->sourceToTargetIdx[mapping.sourceJointName] = targetIdx;
            if (std::abs(mapping.scaleCompensation - 1.0f) > 1e-5f) {
                impl_->scaleCompensations[mapping.targetJointName] = mapping.scaleCompensation;
            }
        }
    }
}

SkeletonRetargeter::~SkeletonRetargeter() = default;
SkeletonRetargeter::SkeletonRetargeter(SkeletonRetargeter&&) noexcept = default;
SkeletonRetargeter& SkeletonRetargeter::operator=(SkeletonRetargeter&&) noexcept = default;

void SkeletonRetargeter::autoMap(
    const std::vector<std::string>& sourceJointNames,
    const std::vector<std::string>& targetJointNames) {

    impl_->config.mappings.clear();
    impl_->sourceToTargetIdx.clear();

    for (const auto& srcName : sourceJointNames) {
        float bestScore = 0.0f;
        int bestTargetIdx = -1;
        std::string bestTargetName;

        for (size_t t = 0; t < targetJointNames.size(); ++t) {
            float score = Impl::nameSimilarity(srcName, targetJointNames[t]);
            if (score > bestScore) {
                bestScore = score;
                bestTargetIdx = static_cast<int>(t);
                bestTargetName = targetJointNames[t];
            }
        }

        if (bestScore > 0.5f && bestTargetIdx >= 0) {
            RetargetMapping mapping;
            mapping.sourceJointName = srcName;
            mapping.targetJointName = bestTargetName;
            mapping.scaleCompensation = 1.0f;
            impl_->config.mappings.push_back(mapping);

            int gameIdx = impl_->findTargetJointIndex(bestTargetName);
            if (gameIdx >= 0) {
                impl_->sourceToTargetIdx[srcName] = gameIdx;
            }

            HM_LOG_DEBUG(TAG, "Mapped: " + srcName + " -> " + bestTargetName +
                         " (score=" + std::to_string(bestScore) + ")");
        }
    }

    HM_LOG_INFO(TAG, "Auto-mapped " + std::to_string(impl_->config.mappings.size()) +
                " joints from " + std::to_string(sourceJointNames.size()) + " source joints");
}

SkeletonFrame SkeletonRetargeter::retarget(const SkeletonFrame& sourceFrame) const {
    SkeletonFrame targetFrame = sourceFrame;

    // Apply global scale to root position
    if (std::abs(impl_->config.globalScale - 1.0f) > 1e-5f) {
        targetFrame.rootPosition = targetFrame.rootPosition * impl_->config.globalScale;
        targetFrame.rootVelocity = targetFrame.rootVelocity * impl_->config.globalScale;
    }

    if (!impl_->config.preserveRootMotion) {
        targetFrame.rootPosition = {0, 0, 0};
        targetFrame.rootVelocity = {0, 0, 0};
    }

    // Apply per-joint scale compensations
    for (const auto& [jointName, scale] : impl_->scaleCompensations) {
        int idx = impl_->findTargetJointIndex(jointName);
        if (idx >= 0) {
            targetFrame.joints[idx].worldPosition =
                targetFrame.joints[idx].worldPosition * scale;
        }
    }

    return targetFrame;
}

AnimClip SkeletonRetargeter::retargetClip(const AnimClip& sourceClip) const {
    AnimClip targetClip;
    targetClip.name = sourceClip.name + "_retargeted";
    targetClip.fps = sourceClip.fps;
    targetClip.trackingID = sourceClip.trackingID;
    targetClip.segments = sourceClip.segments;

    targetClip.frames.reserve(sourceClip.frames.size());
    for (const auto& frame : sourceClip.frames) {
        targetClip.frames.push_back(retarget(frame));
    }

    return targetClip;
}

void SkeletonRetargeter::setScaleCompensation(const std::string& jointName, float scale) {
    impl_->scaleCompensations[jointName] = scale;
}

const std::vector<RetargetMapping>& SkeletonRetargeter::getMappings() const {
    return impl_->config.mappings;
}

} // namespace hm::skeleton
