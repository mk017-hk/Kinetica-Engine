#include "HyperMotion/analysis/MotionFingerprint.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::analysis {

static constexpr const char* TAG = "MotionFingerprint";

// --- FingerprintFeatures ---

std::array<float, FingerprintFeatures::DIM> FingerprintFeatures::toVector() const {
    return {{
        avgVelocity, peakVelocity, avgAcceleration,
        avgTurnRate, peakTurnRate,
        strideLength, strideFrequency,
        avgKneeBend, avgHipRotation, avgArmSwing,
        avgSpineFlexion, verticalRange, avgHeadStability,
        clipDurationSec, static_cast<float>(frameCount),
        leftFootContactRatio, rightFootContactRatio,
        0.0f  // padding to DIM=18
    }};
}

float FingerprintFeatures::distanceTo(const FingerprintFeatures& other) const {
    auto a = toVector();
    auto b = other.toVector();
    float sum = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// --- MotionFingerprint ---

struct MotionFingerprint::Impl {
    MotionFingerprintConfig config;
};

MotionFingerprint::MotionFingerprint(const MotionFingerprintConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionFingerprint::~MotionFingerprint() = default;
MotionFingerprint::MotionFingerprint(MotionFingerprint&&) noexcept = default;
MotionFingerprint& MotionFingerprint::operator=(MotionFingerprint&&) noexcept = default;

// Helper: compute angle between two vectors (degrees)
static float angleBetween(const Vec3& a, const Vec3& b) {
    float la = a.length(), lb = b.length();
    if (la < 1e-8f || lb < 1e-8f) return 0.0f;
    float cosAngle = a.dot(b) / (la * lb);
    cosAngle = std::clamp(cosAngle, -1.0f, 1.0f);
    return std::acos(cosAngle) * 180.0f / 3.14159265f;
}

// Helper: compute joint angle from three joint positions (parent-joint-child)
static float jointAngle(const SkeletonFrame& frame, int parent, int joint, int child) {
    Vec3 a = frame.joints[parent].worldPosition;
    Vec3 b = frame.joints[joint].worldPosition;
    Vec3 c = frame.joints[child].worldPosition;
    Vec3 ba = a - b;
    Vec3 bc = c - b;
    return angleBetween(ba, bc);
}

FingerprintFeatures MotionFingerprint::computeFromFrames(
    const std::vector<SkeletonFrame>& frames, float fps) const {

    FingerprintFeatures fp;
    int n = static_cast<int>(frames.size());
    fp.frameCount = n;
    fp.clipDurationSec = (fps > 0.0f && n > 0) ? static_cast<float>(n) / fps : 0.0f;

    if (n < 2) return fp;

    float dt = 1.0f / fps;

    // Compute per-frame velocities
    std::vector<float> velocities(n, 0.0f);
    std::vector<float> turnRates(n, 0.0f);

    for (int i = 1; i < n; ++i) {
        Vec3 delta = frames[i].rootPosition - frames[i - 1].rootPosition;
        velocities[i] = delta.length() / dt;

        // Turn rate from root rotation change (Y-axis heading)
        Vec3 forward_prev = frames[i - 1].rootRotation.rotate(Vec3(0, 0, 1));
        Vec3 forward_curr = frames[i].rootRotation.rotate(Vec3(0, 0, 1));
        float turnAngle = angleBetween(
            Vec3(forward_prev.x, 0, forward_prev.z),
            Vec3(forward_curr.x, 0, forward_curr.z));
        turnRates[i] = turnAngle / dt;
    }

    // Smooth velocities
    int window = impl_->config.velocitySmoothWindow;
    std::vector<float> smoothVel(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        int lo = std::max(0, i - window / 2);
        int hi = std::min(n - 1, i + window / 2);
        float sum = 0;
        for (int j = lo; j <= hi; ++j) sum += velocities[j];
        smoothVel[i] = sum / (hi - lo + 1);
    }

    // Average and peak velocity
    float velSum = 0, maxVel = 0;
    for (float v : smoothVel) {
        velSum += v;
        maxVel = std::max(maxVel, v);
    }
    fp.avgVelocity = velSum / n;
    fp.peakVelocity = maxVel;

    // Average acceleration
    float accelSum = 0;
    for (int i = 1; i < n; ++i) {
        accelSum += std::abs(smoothVel[i] - smoothVel[i - 1]) / dt;
    }
    fp.avgAcceleration = accelSum / (n - 1);

    // Turn rates
    float turnSum = 0, maxTurn = 0;
    for (float t : turnRates) {
        turnSum += t;
        maxTurn = std::max(maxTurn, t);
    }
    fp.avgTurnRate = turnSum / n;
    fp.peakTurnRate = maxTurn;

    // Vertical range
    float minY = frames[0].rootPosition.y, maxY = minY;
    for (const auto& f : frames) {
        minY = std::min(minY, f.rootPosition.y);
        maxY = std::max(maxY, f.rootPosition.y);
    }
    fp.verticalRange = maxY - minY;

    // Joint statistics (averaged across all frames)
    float kneeBendSum = 0, hipRotSum = 0, armSwingSum = 0;
    float spineFlexSum = 0, headStabSum = 0;
    int validFrames = 0;

    for (const auto& f : frames) {
        // Knee bend: angle at LeftLeg between LeftUpLeg and LeftFoot
        float leftKnee = jointAngle(f,
            static_cast<int>(Joint::LeftUpLeg),
            static_cast<int>(Joint::LeftLeg),
            static_cast<int>(Joint::LeftFoot));
        float rightKnee = jointAngle(f,
            static_cast<int>(Joint::RightUpLeg),
            static_cast<int>(Joint::RightLeg),
            static_cast<int>(Joint::RightFoot));
        kneeBendSum += (180.0f - (leftKnee + rightKnee) * 0.5f);

        // Hip rotation: angle between left-hip and right-hip vectors
        Vec3 leftHipVec = f.joints[static_cast<int>(Joint::LeftUpLeg)].worldPosition -
                          f.joints[static_cast<int>(Joint::Hips)].worldPosition;
        Vec3 rightHipVec = f.joints[static_cast<int>(Joint::RightUpLeg)].worldPosition -
                           f.joints[static_cast<int>(Joint::Hips)].worldPosition;
        hipRotSum += angleBetween(leftHipVec, rightHipVec);

        // Arm swing: average shoulder-to-hand angle deviation
        float leftArmAngle = jointAngle(f,
            static_cast<int>(Joint::LeftShoulder),
            static_cast<int>(Joint::LeftArm),
            static_cast<int>(Joint::LeftForeArm));
        float rightArmAngle = jointAngle(f,
            static_cast<int>(Joint::RightShoulder),
            static_cast<int>(Joint::RightArm),
            static_cast<int>(Joint::RightForeArm));
        armSwingSum += (leftArmAngle + rightArmAngle) * 0.5f;

        // Spine flexion: hips-spine-spine2 angle
        spineFlexSum += jointAngle(f,
            static_cast<int>(Joint::Hips),
            static_cast<int>(Joint::Spine1),
            static_cast<int>(Joint::Spine2));

        // Head stability: angle between head and spine2 forward directions
        Vec3 headDir = f.joints[static_cast<int>(Joint::Head)].worldPosition -
                       f.joints[static_cast<int>(Joint::Neck)].worldPosition;
        Vec3 torsoDir = f.joints[static_cast<int>(Joint::Spine2)].worldPosition -
                        f.joints[static_cast<int>(Joint::Spine)].worldPosition;
        headStabSum += angleBetween(headDir, torsoDir);

        validFrames++;
    }

    if (validFrames > 0) {
        fp.avgKneeBend = kneeBendSum / validFrames;
        fp.avgHipRotation = hipRotSum / validFrames;
        fp.avgArmSwing = armSwingSum / validFrames;
        fp.avgSpineFlexion = spineFlexSum / validFrames;
        fp.avgHeadStability = headStabSum / validFrames;
    }

    // Stride length estimation from foot position oscillation peaks
    std::vector<float> footSeparation(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        Vec3 lf = frames[i].joints[static_cast<int>(Joint::LeftFoot)].worldPosition;
        Vec3 rf = frames[i].joints[static_cast<int>(Joint::RightFoot)].worldPosition;
        footSeparation[i] = (lf - rf).length();
    }

    // Find peaks in foot separation to estimate stride
    int stridePeaks = 0;
    float totalStrideDist = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (footSeparation[i] > footSeparation[i - 1] &&
            footSeparation[i] > footSeparation[i + 1] &&
            footSeparation[i] > impl_->config.minStridePeakHeight) {
            stridePeaks++;
            totalStrideDist += footSeparation[i];
        }
    }
    if (stridePeaks > 0) {
        fp.strideLength = totalStrideDist / stridePeaks;
        fp.strideFrequency = stridePeaks / fp.clipDurationSec;
    }

    return fp;
}

FingerprintFeatures MotionFingerprint::compute(const AnimClip& clip) const {
    FingerprintFeatures fp = computeFromFrames(clip.frames, clip.fps);

    // Add foot contact ratios if available
    if (!clip.footContacts.empty()) {
        int leftCount = 0, rightCount = 0;
        for (const auto& fc : clip.footContacts) {
            if (fc.leftFootContact) leftCount++;
            if (fc.rightFootContact) rightCount++;
        }
        int total = static_cast<int>(clip.footContacts.size());
        fp.leftFootContactRatio = static_cast<float>(leftCount) / total;
        fp.rightFootContactRatio = static_cast<float>(rightCount) / total;
    }

    return fp;
}

std::vector<FingerprintFeatures> MotionFingerprint::computeBatch(
    const std::vector<AnimClip>& clips) const {

    std::vector<FingerprintFeatures> results;
    results.reserve(clips.size());
    for (const auto& clip : clips) {
        results.push_back(compute(clip));
    }
    return results;
}

std::vector<MotionFingerprint::SimilarityResult> MotionFingerprint::findSimilar(
    const FingerprintFeatures& query,
    const std::vector<FingerprintFeatures>& database,
    int maxResults) const {

    std::vector<SimilarityResult> results;
    results.reserve(database.size());

    for (size_t i = 0; i < database.size(); ++i) {
        SimilarityResult sr;
        sr.clipIndex = static_cast<int>(i);
        sr.distance = query.distanceTo(database[i]);
        results.push_back(sr);
    }

    std::sort(results.begin(), results.end(),
              [](const SimilarityResult& a, const SimilarityResult& b) {
                  return a.distance < b.distance;
              });

    if (static_cast<int>(results.size()) > maxResults) {
        results.resize(maxResults);
    }

    return results;
}

} // namespace hm::analysis
