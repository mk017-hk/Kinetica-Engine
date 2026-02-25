#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <cassert>

namespace hm::xport {

static constexpr const char* TAG = "BVHExporter";

// =======================================================================
// Construction / Destruction
// =======================================================================

BVHExporter::BVHExporter(const BVHExportConfig& config)
    : config_(config) {}

BVHExporter::~BVHExporter() = default;

// =======================================================================
// Single Clip Export
// =======================================================================

bool BVHExporter::exportToFile(const AnimClip& clip, const std::string& path) {
    // Validate if configured
    if (config_.validateBeforeExport) {
        auto result = validate(clip);
        if (!result.valid) {
            for (const auto& err : result.errors) {
                HM_LOG_ERROR(TAG, "Validation error: " + err);
            }
            return false;
        }
        for (const auto& warn : result.warnings) {
            HM_LOG_WARN(TAG, "Validation warning: " + warn);
        }
    }

    std::string content = exportToString(clip);
    if (content.empty()) {
        HM_LOG_ERROR(TAG, "Failed to generate BVH content for clip: " + clip.name);
        return false;
    }

    // Ensure parent directory exists
    std::filesystem::path filePath(path);
    if (filePath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);
        if (ec) {
            HM_LOG_ERROR(TAG, "Cannot create directory: " + filePath.parent_path().string()
                         + " (" + ec.message() + ")");
            return false;
        }
    }

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot open file for writing: " + path);
        return false;
    }

    ofs << content;
    ofs.flush();

    if (ofs.fail()) {
        HM_LOG_ERROR(TAG, "Write failed for file: " + path);
        return false;
    }

    ofs.close();

    HM_LOG_INFO(TAG, "Exported BVH to: " + path + " ("
                + std::to_string(clip.frames.size()) + " frames, "
                + std::to_string(getChannelCount()) + " channels, "
                + std::to_string(content.size()) + " bytes)");
    return true;
}

std::string BVHExporter::exportToString(const AnimClip& clip) {
    if (clip.frames.empty()) {
        HM_LOG_WARN(TAG, "Clip '" + clip.name + "' has no frames, generating empty BVH");
    }

    // Invalidate traversal order cache so it rebuilds for current config
    traversalOrderDirty_ = true;

    std::string result;
    // Pre-allocate a reasonable estimate: ~200 bytes per joint for hierarchy + ~50 bytes per channel per frame
    size_t estimatedSize = JOINT_COUNT * 200 + clip.frames.size() * getChannelCount() * 12;
    result.reserve(estimatedSize);

    result += buildHierarchy();
    result += buildMotion(clip);

    return result;
}

// =======================================================================
// Batch Export
// =======================================================================

BVHExporter::BatchExportResult BVHExporter::exportBatch(
    const std::vector<AnimClip>& clips,
    const std::string& outputDirectory,
    const std::string& prefix,
    ProgressCallback progress) {

    BatchExportResult result;
    result.totalClips = static_cast<int>(clips.size());

    if (clips.empty()) {
        HM_LOG_WARN(TAG, "No clips to export in batch");
        return result;
    }

    // Ensure output directory exists
    std::error_code ec;
    std::filesystem::create_directories(outputDirectory, ec);
    if (ec) {
        HM_LOG_ERROR(TAG, "Cannot create output directory: " + outputDirectory
                     + " (" + ec.message() + ")");
        result.failCount = result.totalClips;
        return result;
    }

    for (int i = 0; i < static_cast<int>(clips.size()); ++i) {
        const auto& clip = clips[i];

        // Build filename: prefix_clipname_index.bvh
        std::string clipNameSafe = sanitiseFilename(clip.name.empty() ? "unnamed" : clip.name);
        std::string filename = prefix + "_" + clipNameSafe + "_"
                             + std::to_string(i) + ".bvh";

        std::filesystem::path outPath = std::filesystem::path(outputDirectory) / filename;
        std::string outPathStr = outPath.string();

        if (progress) {
            progress(i + 1, result.totalClips, clip.name);
        }

        if (exportToFile(clip, outPathStr)) {
            result.successCount++;
            result.exportedPaths.push_back(outPathStr);
        } else {
            result.failCount++;
            result.failedClips.push_back(clip.name + " (index " + std::to_string(i) + ")");
            HM_LOG_ERROR(TAG, "Failed to export clip '" + clip.name + "' to: " + outPathStr);
        }
    }

    HM_LOG_INFO(TAG, "Batch export complete: " + std::to_string(result.successCount)
                + "/" + std::to_string(result.totalClips) + " succeeded");

    return result;
}

// =======================================================================
// Validation
// =======================================================================

BVHValidationResult BVHExporter::validate(const AnimClip& clip) const {
    BVHValidationResult result;

    result.frameCount = static_cast<int>(clip.frames.size());
    result.jointCount = JOINT_COUNT;
    result.channelCount = getChannelCount();

    // Check for empty clip
    if (clip.frames.empty()) {
        result.warnings.push_back("Clip has no frames");
    }

    // Check FPS
    if (clip.fps <= 0.0f) {
        result.errors.push_back("Invalid FPS: " + std::to_string(clip.fps));
        result.valid = false;
    } else if (clip.fps < 1.0f || clip.fps > 240.0f) {
        result.warnings.push_back("Unusual FPS value: " + std::to_string(clip.fps)
                                  + " (expected 1-240)");
    }

    result.durationSeconds = clip.fps > 0.0f
        ? static_cast<float>(clip.frames.size()) / clip.fps
        : 0.0f;

    // Check for NaN or inf in frame data
    for (size_t f = 0; f < clip.frames.size(); ++f) {
        const auto& frame = clip.frames[f];

        // Check root position
        if (std::isnan(frame.rootPosition.x) || std::isnan(frame.rootPosition.y) ||
            std::isnan(frame.rootPosition.z) ||
            std::isinf(frame.rootPosition.x) || std::isinf(frame.rootPosition.y) ||
            std::isinf(frame.rootPosition.z)) {
            result.errors.push_back("Frame " + std::to_string(f)
                                    + " has NaN/Inf root position");
            result.valid = false;
            break; // Don't flood with errors
        }

        // Check root rotation quaternion norm
        float qNorm = frame.rootRotation.norm();
        if (std::abs(qNorm - 1.0f) > 0.1f) {
            result.warnings.push_back("Frame " + std::to_string(f)
                                      + " root quaternion not normalised (norm="
                                      + std::to_string(qNorm) + ")");
        }

        // Check joint data
        for (int j = 0; j < JOINT_COUNT; ++j) {
            const auto& jt = frame.joints[j];
            if (std::isnan(jt.localEulerDeg.x) || std::isnan(jt.localEulerDeg.y) ||
                std::isnan(jt.localEulerDeg.z)) {
                result.errors.push_back("Frame " + std::to_string(f)
                                        + " joint " + JOINT_NAMES[j]
                                        + " has NaN Euler angles");
                result.valid = false;
                break;
            }

            // Check for extreme rotation values (likely data corruption)
            if (std::abs(jt.localEulerDeg.x) > 360.0f ||
                std::abs(jt.localEulerDeg.y) > 360.0f ||
                std::abs(jt.localEulerDeg.z) > 360.0f) {
                result.warnings.push_back("Frame " + std::to_string(f)
                                          + " joint " + JOINT_NAMES[j]
                                          + " has extreme Euler values (>"
                                          + std::to_string(360) + " deg)");
            }
        }
        if (!result.valid) break; // Stop early on errors
    }

    // Check for reasonable scale
    if (config_.scale <= 0.0f) {
        result.errors.push_back("Scale must be positive, got: " + std::to_string(config_.scale));
        result.valid = false;
    }

    return result;
}

// =======================================================================
// Configuration helpers
// =======================================================================

void BVHExporter::setBoneNameOverride(const std::string& originalName, const std::string& newName) {
    config_.boneNameOverrides[originalName] = newName;
}

int BVHExporter::getChannelCount() const {
    // Root: 6 channels (3 position + 3 rotation) if root motion enabled, else 3 rotation
    // All other joints: 3 rotation channels each
    int rootChannels = config_.exportRootMotion ? 6 : 3;
    return rootChannels + (JOINT_COUNT - 1) * 3;
}

std::vector<int> BVHExporter::getJointTraversalOrder() const {
    if (traversalOrderDirty_) {
        buildTraversalOrder();
    }
    return cachedTraversalOrder_;
}

// =======================================================================
// Hierarchy Building
// =======================================================================

std::string BVHExporter::buildHierarchy() const {
    std::string hierarchy;
    hierarchy.reserve(JOINT_COUNT * 200);

    hierarchy += "HIERARCHY\n";
    writeJoint(hierarchy, 0, 0); // Start from Hips (root, index 0)

    return hierarchy;
}

void BVHExporter::writeJoint(std::string& output, int jointIndex, int depth) const {
    std::string indent(depth * 2, ' ');

    // Determine children of this joint
    std::vector<int> children = getChildren(jointIndex);

    // Joint type keyword
    bool isRoot = (jointIndex == 0);
    const char* jointType = isRoot ? "ROOT" : "JOINT";

    // Bone name (with optional override)
    std::string boneName = getBoneName(jointIndex);

    // Joint header line
    output += indent + jointType + " " + boneName + "\n";
    output += indent + "{\n";

    // OFFSET: relative to parent in rest pose
    Vec3 offset = getJointOffset(jointIndex);
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(config_.floatPrecision);
        oss << offset.x << " " << offset.y << " " << offset.z;
        output += indent + "  OFFSET " + oss.str() + "\n";
    }

    // CHANNELS
    if (isRoot) {
        if (config_.exportRootMotion) {
            output += indent + "  CHANNELS 6 Xposition Yposition Zposition "
                    + getRotationChannelString() + "\n";
        } else {
            output += indent + "  CHANNELS 3 " + getRotationChannelString() + "\n";
        }
    } else {
        output += indent + "  CHANNELS 3 " + getRotationChannelString() + "\n";
    }

    // Children or End Site
    if (children.empty()) {
        if (config_.useEndSites) {
            writeEndSite(output, jointIndex, depth);
        }
    } else {
        for (int child : children) {
            writeJoint(output, child, depth + 1);
        }
    }

    output += indent + "}\n";
}

void BVHExporter::writeEndSite(std::string& output, int jointIndex, int depth) const {
    std::string indent(depth * 2 + 2, ' ');

    output += indent + "End Site\n";
    output += indent + "{\n";

    // Compute end site offset along the bone's direction
    Vec3 endOffset = getEndSiteOffset(jointIndex);
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(config_.floatPrecision);
        oss << endOffset.x << " " << endOffset.y << " " << endOffset.z;
        output += indent + "  OFFSET " + oss.str() + "\n";
    }

    output += indent + "}\n";
}

// =======================================================================
// Motion Data Building
// =======================================================================

std::string BVHExporter::buildMotion(const AnimClip& clip) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(config_.floatPrecision);

    oss << "MOTION\n";
    oss << "Frames: " << clip.frames.size() << "\n";

    // Frame Time: inverse of FPS
    float frameTime = (clip.fps > 0.0f) ? (1.0 / clip.fps) : (1.0 / 30.0);
    oss << "Frame Time: " << frameTime << "\n";

    // Build joint traversal order matching the hierarchy write order
    if (traversalOrderDirty_) {
        buildTraversalOrder();
    }
    const auto& jointOrder = cachedTraversalOrder_;

    // Separator character between values
    const char sep = config_.compactMotionData ? ' ' : ' ';

    // Write each frame
    for (size_t frameIdx = 0; frameIdx < clip.frames.size(); ++frameIdx) {
        const auto& frame = clip.frames[frameIdx];
        bool firstValue = true;

        for (size_t orderIdx = 0; orderIdx < jointOrder.size(); ++orderIdx) {
            int jIdx = jointOrder[orderIdx];
            bool isRoot = (jIdx == 0);

            if (!firstValue) {
                oss << sep;
            }
            firstValue = false;

            // Root joint: position channels first (if enabled)
            if (isRoot && config_.exportRootMotion) {
                Vec3 pos = frame.rootPosition * config_.scale;
                oss << pos.x << sep << pos.y << sep << pos.z << sep;
            }

            // Extract Euler angles in the configured rotation order
            Vec3 euler = extractEulerForBVH(frame.joints[jIdx]);

            // Write rotation values in the order matching the CHANNELS specification
            switch (config_.rotationOrder) {
                case BVHExportConfig::RotationOrder::ZXY:
                    oss << euler.z << sep << euler.x << sep << euler.y;
                    break;
                case BVHExportConfig::RotationOrder::XYZ:
                    oss << euler.x << sep << euler.y << sep << euler.z;
                    break;
                case BVHExportConfig::RotationOrder::YZX:
                    oss << euler.y << sep << euler.z << sep << euler.x;
                    break;
                case BVHExportConfig::RotationOrder::ZYX:
                    oss << euler.z << sep << euler.y << sep << euler.x;
                    break;
                case BVHExportConfig::RotationOrder::XZY:
                    oss << euler.x << sep << euler.z << sep << euler.y;
                    break;
                case BVHExportConfig::RotationOrder::YXZ:
                    oss << euler.y << sep << euler.x << sep << euler.z;
                    break;
            }
        }
        oss << "\n";
    }

    return oss.str();
}

void BVHExporter::writeJointFrameData(
    std::ostringstream& oss,
    const SkeletonFrame& frame,
    int jointIndex,
    bool isRoot) const {

    // Root: position channels
    if (isRoot && config_.exportRootMotion) {
        Vec3 pos = frame.rootPosition * config_.scale;
        oss << pos.x << " " << pos.y << " " << pos.z << " ";
    }

    // Rotation
    Vec3 euler = extractEulerForBVH(frame.joints[jointIndex]);

    switch (config_.rotationOrder) {
        case BVHExportConfig::RotationOrder::ZXY:
            oss << euler.z << " " << euler.x << " " << euler.y;
            break;
        case BVHExportConfig::RotationOrder::XYZ:
            oss << euler.x << " " << euler.y << " " << euler.z;
            break;
        case BVHExportConfig::RotationOrder::YZX:
            oss << euler.y << " " << euler.z << " " << euler.x;
            break;
        case BVHExportConfig::RotationOrder::ZYX:
            oss << euler.z << " " << euler.y << " " << euler.x;
            break;
        case BVHExportConfig::RotationOrder::XZY:
            oss << euler.x << " " << euler.z << " " << euler.y;
            break;
        case BVHExportConfig::RotationOrder::YXZ:
            oss << euler.y << " " << euler.x << " " << euler.z;
            break;
    }
}

// =======================================================================
// Helper Methods
// =======================================================================

std::string BVHExporter::getBoneName(int jointIndex) const {
    if (jointIndex < 0 || jointIndex >= JOINT_COUNT) {
        return "Unknown";
    }

    std::string originalName = JOINT_NAMES[jointIndex];

    // Check for user-specified overrides
    auto it = config_.boneNameOverrides.find(originalName);
    if (it != config_.boneNameOverrides.end()) {
        return it->second;
    }

    return originalName;
}

Vec3 BVHExporter::getJointOffset(int jointIndex) const {
    if (jointIndex < 0 || jointIndex >= JOINT_COUNT) {
        return Vec3{0.0f, 0.0f, 0.0f};
    }

    const auto& restOffsets = getRestPoseBoneOffsets();
    Vec3 offset = restOffsets[jointIndex];

    // Apply scale
    return offset * config_.scale;
}

Vec3 BVHExporter::getEndSiteOffset(int jointIndex) const {
    if (jointIndex < 0 || jointIndex >= JOINT_COUNT) {
        return Vec3{0.0f, config_.endSiteBoneLength * config_.scale, 0.0f};
    }

    // Compute end site offset in the direction of the bone from parent to this joint
    const auto& restOffsets = getRestPoseBoneOffsets();
    Vec3 boneDir = restOffsets[jointIndex];
    float boneLen = boneDir.length();

    if (boneLen > 1e-6f) {
        // Extend along the same direction as the bone
        Vec3 dir = boneDir.normalized();
        return dir * (config_.endSiteBoneLength * config_.scale);
    }

    // Fallback: use the default bone length along the parent bone's direction
    // For joints at the extremities, try to infer direction from rest pose
    // Head end-site: upward
    if (jointIndex == static_cast<int>(Joint::Head)) {
        return Vec3{0.0f, config_.endSiteBoneLength * config_.scale, 0.0f};
    }
    // Hand end-sites: along forearm direction
    if (jointIndex == static_cast<int>(Joint::LeftHand)) {
        return Vec3{-config_.endSiteBoneLength * config_.scale, 0.0f, 0.0f};
    }
    if (jointIndex == static_cast<int>(Joint::RightHand)) {
        return Vec3{config_.endSiteBoneLength * config_.scale, 0.0f, 0.0f};
    }
    // Toe end-sites: forward
    if (jointIndex == static_cast<int>(Joint::LeftToeBase) ||
        jointIndex == static_cast<int>(Joint::RightToeBase)) {
        return Vec3{0.0f, 0.0f, config_.endSiteBoneLength * config_.scale};
    }

    // Generic fallback: short upward offset
    return Vec3{0.0f, config_.endSiteBoneLength * config_.scale, 0.0f};
}

void BVHExporter::buildTraversalOrder() const {
    cachedTraversalOrder_.clear();
    cachedTraversalOrder_.reserve(JOINT_COUNT);

    // Depth-first traversal matching hierarchy write order
    std::function<void(int)> traverse = [&](int idx) {
        cachedTraversalOrder_.push_back(idx);
        // Find all children of this joint, in order
        for (int i = 0; i < JOINT_COUNT; ++i) {
            if (JOINT_PARENT[i] == idx) {
                traverse(i);
            }
        }
    };

    traverse(0); // Start from root (Hips)
    traversalOrderDirty_ = false;
}

std::vector<int> BVHExporter::getChildren(int jointIndex) const {
    std::vector<int> children;
    for (int i = 0; i < JOINT_COUNT; ++i) {
        if (JOINT_PARENT[i] == jointIndex) {
            children.push_back(i);
        }
    }
    return children;
}

Vec3 BVHExporter::extractEulerForBVH(const JointTransform& joint) const {
    // The joint already has localEulerDeg computed (XYZ intrinsic order from MathUtils).
    // For BVH we may need to re-decompose from the quaternion if the rotation order differs.
    // For the default ZXY order, we use the stored Euler values directly since the
    // MathUtils stores them as XYZ intrinsic which is equivalent to ZYX extrinsic.
    // BVH uses extrinsic (fixed-axis) rotations, so we need to be careful.

    // For maximum accuracy, always decompose from the quaternion in the target BVH order.
    const Quat& q = joint.localRotation;

    // Convert quaternion to rotation matrix
    Mat3 m = MathUtils::quatToMat3(q);

    Vec3 euler;

    // Extract Euler angles for the configured rotation order
    // All angles returned in degrees
    // BVH applies rotations in the order listed (extrinsic/fixed-axis), which is
    // equivalent to intrinsic rotations in the reverse order.
    // For ZXY extrinsic: R = Ry * Rx * Rz

    switch (config_.rotationOrder) {
        case BVHExportConfig::RotationOrder::ZXY: {
            // Extrinsic ZXY = Intrinsic YXZ
            // R = Ry * Rx * Rz
            // Extract: asin(m[1][2]) for X, etc.
            float sinX = m.m[1][2];
            sinX = std::clamp(sinX, -1.0f, 1.0f);
            euler.x = std::asin(sinX) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinX) < 0.9999f) {
                euler.y = std::atan2(-m.m[0][2], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
                euler.z = std::atan2(-m.m[1][0], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
            } else {
                // Gimbal lock
                euler.y = std::atan2(m.m[2][0], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
                euler.z = 0.0f;
            }
            break;
        }
        case BVHExportConfig::RotationOrder::XYZ: {
            // Extrinsic XYZ = Intrinsic ZYX
            float sinY = -m.m[2][0];
            sinY = std::clamp(sinY, -1.0f, 1.0f);
            euler.y = std::asin(sinY) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinY) < 0.9999f) {
                euler.x = std::atan2(m.m[2][1], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
                euler.z = std::atan2(m.m[1][0], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
            } else {
                euler.x = std::atan2(-m.m[1][2], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
                euler.z = 0.0f;
            }
            break;
        }
        case BVHExportConfig::RotationOrder::YZX: {
            // Extrinsic YZX = Intrinsic XZY
            float sinZ = m.m[0][1];
            sinZ = std::clamp(sinZ, -1.0f, 1.0f);
            euler.z = std::asin(sinZ) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinZ) < 0.9999f) {
                euler.x = std::atan2(-m.m[2][1], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
                euler.y = std::atan2(-m.m[0][2], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
            } else {
                euler.x = 0.0f;
                euler.y = std::atan2(m.m[2][0], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
            }
            break;
        }
        case BVHExportConfig::RotationOrder::ZYX: {
            // Extrinsic ZYX = Intrinsic XYZ
            float sinY = m.m[0][2];
            sinY = std::clamp(sinY, -1.0f, 1.0f);
            euler.y = std::asin(sinY) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinY) < 0.9999f) {
                euler.x = std::atan2(-m.m[1][2], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
                euler.z = std::atan2(-m.m[0][1], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
            } else {
                euler.x = 0.0f;
                euler.z = std::atan2(m.m[1][0], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
            }
            break;
        }
        case BVHExportConfig::RotationOrder::XZY: {
            // Extrinsic XZY = Intrinsic YZX
            float sinZ = -m.m[0][1];
            sinZ = std::clamp(sinZ, -1.0f, 1.0f);
            euler.z = std::asin(sinZ) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinZ) < 0.9999f) {
                euler.x = std::atan2(m.m[2][1], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
                euler.y = std::atan2(m.m[0][2], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
            } else {
                euler.x = 0.0f;
                euler.y = std::atan2(-m.m[2][0], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
            }
            break;
        }
        case BVHExportConfig::RotationOrder::YXZ: {
            // Extrinsic YXZ = Intrinsic ZXY
            float sinX = -m.m[1][2];
            sinX = std::clamp(sinX, -1.0f, 1.0f);
            euler.x = std::asin(sinX) * (180.0f / 3.14159265358979323846f);

            if (std::abs(sinX) < 0.9999f) {
                euler.y = std::atan2(m.m[0][2], m.m[2][2]) * (180.0f / 3.14159265358979323846f);
                euler.z = std::atan2(m.m[1][0], m.m[1][1]) * (180.0f / 3.14159265358979323846f);
            } else {
                euler.y = std::atan2(-m.m[2][0], m.m[0][0]) * (180.0f / 3.14159265358979323846f);
                euler.z = 0.0f;
            }
            break;
        }
    }

    return euler;
}

std::string BVHExporter::getRotationChannelString() const {
    switch (config_.rotationOrder) {
        case BVHExportConfig::RotationOrder::ZXY:
            return "Zrotation Xrotation Yrotation";
        case BVHExportConfig::RotationOrder::XYZ:
            return "Xrotation Yrotation Zrotation";
        case BVHExportConfig::RotationOrder::YZX:
            return "Yrotation Zrotation Xrotation";
        case BVHExportConfig::RotationOrder::ZYX:
            return "Zrotation Yrotation Xrotation";
        case BVHExportConfig::RotationOrder::XZY:
            return "Xrotation Zrotation Yrotation";
        case BVHExportConfig::RotationOrder::YXZ:
            return "Yrotation Xrotation Zrotation";
    }
    return "Zrotation Xrotation Yrotation"; // Default fallback
}

std::string BVHExporter::sanitiseFilename(const std::string& name) {
    std::string result;
    result.reserve(name.size());

    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            result += c;
        } else if (c == ' ' || c == '.' || c == '/' || c == '\\') {
            result += '_';
        }
        // Skip other special characters
    }

    if (result.empty()) {
        result = "unnamed";
    }

    // Truncate to reasonable length
    if (result.size() > 128) {
        result.resize(128);
    }

    return result;
}

} // namespace hm::xport
