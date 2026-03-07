#include "HyperMotion/dataset/MotionClusterer.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <iomanip>

namespace hm::dataset {

static constexpr const char* TAG = "MotionClusterer";
static constexpr int FEATURE_DIM = 8; // number of fields in ClipFeatures

struct MotionClusterer::Impl {
    MotionClustererConfig config;

    // Convert ClipFeatures to a flat float array for distance computation.
    static std::array<float, FEATURE_DIM> toArray(const ClipFeatures& f) {
        return {f.avgVelocity, f.maxVelocity, f.avgTurnRate,
                f.strideFrequency, f.avgKneeBend, f.avgHipRotation,
                f.avgArmSwing, f.verticalRange};
    }

    static ClipFeatures fromArray(const std::array<float, FEATURE_DIM>& a) {
        ClipFeatures f;
        f.avgVelocity = a[0];
        f.maxVelocity = a[1];
        f.avgTurnRate = a[2];
        f.strideFrequency = a[3];
        f.avgKneeBend = a[4];
        f.avgHipRotation = a[5];
        f.avgArmSwing = a[6];
        f.verticalRange = a[7];
        return f;
    }

    // Squared Euclidean distance between two feature vectors.
    static float distanceSq(const std::array<float, FEATURE_DIM>& a,
                            const std::array<float, FEATURE_DIM>& b) {
        float d = 0.0f;
        for (int i = 0; i < FEATURE_DIM; ++i) {
            float diff = a[i] - b[i];
            d += diff * diff;
        }
        return d;
    }

    // Normalise features to zero-mean unit-variance for better clustering.
    struct NormParams {
        std::array<float, FEATURE_DIM> mean{};
        std::array<float, FEATURE_DIM> stddev{};
    };

    static NormParams computeNorm(
        const std::vector<std::array<float, FEATURE_DIM>>& data) {
        NormParams p;
        int n = static_cast<int>(data.size());
        if (n == 0) return p;

        for (int d = 0; d < FEATURE_DIM; ++d) {
            float sum = 0;
            for (const auto& v : data) sum += v[d];
            p.mean[d] = sum / n;

            float sumSq = 0;
            for (const auto& v : data) {
                float diff = v[d] - p.mean[d];
                sumSq += diff * diff;
            }
            p.stddev[d] = std::sqrt(sumSq / n);
            if (p.stddev[d] < 1e-8f) p.stddev[d] = 1.0f; // avoid div-by-zero
        }
        return p;
    }

    static std::array<float, FEATURE_DIM> normalise(
        const std::array<float, FEATURE_DIM>& v, const NormParams& p) {
        std::array<float, FEATURE_DIM> out;
        for (int d = 0; d < FEATURE_DIM; ++d) {
            out[d] = (v[d] - p.mean[d]) / p.stddev[d];
        }
        return out;
    }

    // Estimate stride frequency from vertical oscillation of the feet.
    float estimateStrideFrequency(const AnimClip& clip) const {
        if (clip.frames.size() < 10) return 0.0f;

        int leftFootIdx = static_cast<int>(Joint::LeftFoot);
        int crossings = 0;
        float prevHeight = clip.frames[0].joints[leftFootIdx].worldPosition.y;
        float meanHeight = 0.0f;

        for (const auto& f : clip.frames) {
            meanHeight += f.joints[leftFootIdx].worldPosition.y;
        }
        meanHeight /= static_cast<float>(clip.frames.size());

        bool wasAbove = prevHeight > meanHeight;
        for (size_t i = 1; i < clip.frames.size(); ++i) {
            float h = clip.frames[i].joints[leftFootIdx].worldPosition.y;
            bool isAbove = h > meanHeight;
            if (isAbove != wasAbove) {
                crossings++;
                wasAbove = isAbove;
            }
        }

        // Each full stride is 2 zero-crossings
        float duration = static_cast<float>(clip.frames.size() - 1) / config.fps;
        if (duration < 0.1f) return 0.0f;
        return (crossings / 2.0f) / duration;
    }

    // Average angular difference between consecutive frames for a joint.
    float avgJointAngle(const AnimClip& clip, Joint joint) const {
        if (clip.frames.size() < 2) return 0.0f;
        int idx = static_cast<int>(joint);
        float sum = 0.0f;
        for (const auto& f : clip.frames) {
            const auto& e = f.joints[idx].localEulerDeg;
            sum += std::sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
        }
        return sum / static_cast<float>(clip.frames.size());
    }
};

MotionClusterer::MotionClusterer(const MotionClustererConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionClusterer::~MotionClusterer() = default;
MotionClusterer::MotionClusterer(MotionClusterer&&) noexcept = default;
MotionClusterer& MotionClusterer::operator=(MotionClusterer&&) noexcept = default;

ClipFeatures MotionClusterer::extractFeatures(const AnimClip& clip) const {
    ClipFeatures feat;
    if (clip.frames.empty()) return feat;

    int n = static_cast<int>(clip.frames.size());

    // Velocity statistics
    float sumVel = 0.0f;
    float maxVel = 0.0f;
    for (const auto& f : clip.frames) {
        float v = f.rootVelocity.length();
        sumVel += v;
        maxVel = std::max(maxVel, v);
    }
    feat.avgVelocity = sumVel / n;
    feat.maxVelocity = maxVel;

    // Turn rate (angular velocity around Y axis)
    float sumTurn = 0.0f;
    for (int i = 1; i < n; ++i) {
        Vec3 fwd0 = clip.frames[i - 1].rootRotation.rotate({0, 0, 1});
        Vec3 fwd1 = clip.frames[i].rootRotation.rotate({0, 0, 1});
        float dot = fwd0.x * fwd1.x + fwd0.z * fwd1.z;
        dot = std::clamp(dot, -1.0f, 1.0f);
        float angle = std::acos(dot) * 180.0f / static_cast<float>(M_PI);
        sumTurn += angle * impl_->config.fps; // deg/s
    }
    feat.avgTurnRate = (n > 1) ? sumTurn / (n - 1) : 0.0f;

    // Stride frequency from foot oscillation
    feat.strideFrequency = impl_->estimateStrideFrequency(clip);

    // Joint angle averages
    feat.avgKneeBend = (impl_->avgJointAngle(clip, Joint::LeftLeg) +
                        impl_->avgJointAngle(clip, Joint::RightLeg)) * 0.5f;
    feat.avgHipRotation = (impl_->avgJointAngle(clip, Joint::LeftUpLeg) +
                           impl_->avgJointAngle(clip, Joint::RightUpLeg)) * 0.5f;
    feat.avgArmSwing = (impl_->avgJointAngle(clip, Joint::LeftArm) +
                        impl_->avgJointAngle(clip, Joint::RightArm)) * 0.5f;

    // Vertical range (root Y)
    float minY = clip.frames[0].rootPosition.y;
    float maxY = minY;
    for (const auto& f : clip.frames) {
        minY = std::min(minY, f.rootPosition.y);
        maxY = std::max(maxY, f.rootPosition.y);
    }
    feat.verticalRange = maxY - minY;

    return feat;
}

ClusteringResult MotionClusterer::cluster(
    const std::vector<AnimClip>& clips) const {

    ClusteringResult result;
    int numClips = static_cast<int>(clips.size());

    if (numClips == 0) return result;

    int k = std::min(impl_->config.numClusters, numClips);

    // Extract and normalise features
    std::vector<std::array<float, FEATURE_DIM>> features(numClips);
    for (int i = 0; i < numClips; ++i) {
        features[i] = Impl::toArray(extractFeatures(clips[i]));
    }

    auto normParams = Impl::computeNorm(features);
    std::vector<std::array<float, FEATURE_DIM>> normFeatures(numClips);
    for (int i = 0; i < numClips; ++i) {
        normFeatures[i] = Impl::normalise(features[i], normParams);
    }

    // k-means++ initialization
    std::mt19937 rng(impl_->config.randomSeed);
    std::vector<std::array<float, FEATURE_DIM>> centroids(k);
    std::vector<int> assignments(numClips, 0);

    // Pick first centroid randomly
    std::uniform_int_distribution<int> uniformDist(0, numClips - 1);
    centroids[0] = normFeatures[uniformDist(rng)];

    // Pick remaining centroids proportional to D^2
    for (int c = 1; c < k; ++c) {
        std::vector<float> minDists(numClips, std::numeric_limits<float>::max());
        for (int i = 0; i < numClips; ++i) {
            for (int j = 0; j < c; ++j) {
                float d = Impl::distanceSq(normFeatures[i], centroids[j]);
                minDists[i] = std::min(minDists[i], d);
            }
        }
        std::discrete_distribution<int> dist(minDists.begin(), minDists.end());
        centroids[c] = normFeatures[dist(rng)];
    }

    // k-means iterations
    int iter = 0;
    for (; iter < impl_->config.maxIterations; ++iter) {
        // Assignment step
        for (int i = 0; i < numClips; ++i) {
            float bestDist = std::numeric_limits<float>::max();
            for (int c = 0; c < k; ++c) {
                float d = Impl::distanceSq(normFeatures[i], centroids[c]);
                if (d < bestDist) {
                    bestDist = d;
                    assignments[i] = c;
                }
            }
        }

        // Update step
        std::vector<std::array<float, FEATURE_DIM>> newCentroids(
            k, std::array<float, FEATURE_DIM>{});
        std::vector<int> counts(k, 0);

        for (int i = 0; i < numClips; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (int d = 0; d < FEATURE_DIM; ++d) {
                newCentroids[c][d] += normFeatures[i][d];
            }
        }

        float maxShift = 0.0f;
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < FEATURE_DIM; ++d) {
                    newCentroids[c][d] /= counts[c];
                }
            }
            maxShift = std::max(maxShift,
                Impl::distanceSq(centroids[c], newCentroids[c]));
        }

        centroids = std::move(newCentroids);

        if (maxShift < impl_->config.convergenceThreshold) {
            iter++;
            break;
        }
    }

    // Compute inertia and build result
    float totalInertia = 0.0f;
    for (int i = 0; i < numClips; ++i) {
        totalInertia += Impl::distanceSq(normFeatures[i], centroids[assignments[i]]);
    }

    result.assignments = assignments;
    result.numIterations = iter;
    result.totalInertia = totalInertia;

    // Build cluster info (de-normalise centroids for readability)
    for (int c = 0; c < k; ++c) {
        ClusterInfo info;
        info.clusterID = c;
        info.memberCount = static_cast<int>(
            std::count(assignments.begin(), assignments.end(), c));

        // De-normalise centroid
        std::array<float, FEATURE_DIM> raw;
        for (int d = 0; d < FEATURE_DIM; ++d) {
            raw[d] = centroids[c][d] * normParams.stddev[d] + normParams.mean[d];
        }
        info.centroid = Impl::fromArray(raw);

        // Generate label: "cluster_01", "cluster_02", ...
        std::ostringstream oss;
        oss << "cluster_" << std::setw(2) << std::setfill('0') << (c + 1);
        info.label = oss.str();

        result.clusters.push_back(info);
    }

    HM_LOG_INFO(TAG, "Clustered " + std::to_string(numClips) + " clips into " +
                std::to_string(k) + " clusters (" +
                std::to_string(iter) + " iterations, inertia=" +
                std::to_string(totalInertia) + ")");

    return result;
}

void MotionClusterer::process(std::vector<AnimClip>& clips) const {
    if (clips.empty()) return;

    auto result = cluster(clips);

    for (size_t i = 0; i < clips.size(); ++i) {
        clips[i].clusterID = result.assignments[i];
    }

    HM_LOG_INFO(TAG, "Assigned cluster labels to " +
                std::to_string(clips.size()) + " clips");
}

} // namespace hm::dataset
