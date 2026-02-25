#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <filesystem>

namespace hm::xport {

// -----------------------------------------------------------------------
// BVH Export Configuration
// -----------------------------------------------------------------------

struct BVHExportConfig {
    float scale = 1.0f;                  // Scale factor for positions (e.g., cm->m = 0.01)
    bool exportRootMotion = true;        // Include root position channels
    int floatPrecision = 6;              // Decimal places for floating point values
    bool useEndSites = true;             // Include End Site nodes for leaf joints
    float endSiteBoneLength = 5.0f;      // Length of end-site pseudo bones (cm)
    bool validateBeforeExport = true;    // Run validation checks before writing
    bool compactMotionData = false;      // Omit extra whitespace in MOTION section

    // Bone name remapping: BVH name -> custom name (e.g., for MotionBuilder compatibility)
    std::unordered_map<std::string, std::string> boneNameOverrides;

    // Rotation order for CHANNELS specification
    // Default: ZXY (standard BVH for Y-up coordinate systems)
    enum class RotationOrder {
        ZXY,   // BVH standard for Y-up
        XYZ,
        YZX,
        ZYX,
        XZY,
        YXZ
    };
    RotationOrder rotationOrder = RotationOrder::ZXY;
};

// -----------------------------------------------------------------------
// BVH Validation Result
// -----------------------------------------------------------------------

struct BVHValidationResult {
    bool valid = true;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    int frameCount = 0;
    int jointCount = 0;
    int channelCount = 0;
    float durationSeconds = 0.0f;
};

// -----------------------------------------------------------------------
// BVH Exporter
// -----------------------------------------------------------------------

class BVHExporter {
public:
    explicit BVHExporter(const BVHExportConfig& config = {});
    ~BVHExporter();

    // Non-copyable, movable
    BVHExporter(const BVHExporter&) = delete;
    BVHExporter& operator=(const BVHExporter&) = delete;
    BVHExporter(BVHExporter&&) noexcept = default;
    BVHExporter& operator=(BVHExporter&&) noexcept = default;

    // -------------------------------------------------------------------
    // Single clip export
    // -------------------------------------------------------------------

    // Export AnimClip to BVH file
    bool exportToFile(const AnimClip& clip, const std::string& path);

    // Export to in-memory string
    std::string exportToString(const AnimClip& clip);

    // -------------------------------------------------------------------
    // Batch export
    // -------------------------------------------------------------------

    // Export multiple clips to individual BVH files in a directory
    // Files are named: {prefix}_{clipName}_{index}.bvh
    struct BatchExportResult {
        int totalClips = 0;
        int successCount = 0;
        int failCount = 0;
        std::vector<std::string> exportedPaths;
        std::vector<std::string> failedClips;
    };

    using ProgressCallback = std::function<void(int current, int total, const std::string& clipName)>;

    BatchExportResult exportBatch(
        const std::vector<AnimClip>& clips,
        const std::string& outputDirectory,
        const std::string& prefix = "hm",
        ProgressCallback progress = nullptr);

    // -------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------

    BVHValidationResult validate(const AnimClip& clip) const;

    // -------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------

    void setConfig(const BVHExportConfig& config) { config_ = config; }
    const BVHExportConfig& getConfig() const { return config_; }

    // Convenience: set bone name override
    void setBoneNameOverride(const std::string& originalName, const std::string& newName);

    // Get the BVH channel count for a clip
    int getChannelCount() const;

    // Get the joint traversal order (matches hierarchy writing order)
    std::vector<int> getJointTraversalOrder() const;

private:
    BVHExportConfig config_;

    // Joint traversal order cache (built once, same order as hierarchy)
    mutable std::vector<int> cachedTraversalOrder_;
    mutable bool traversalOrderDirty_ = true;

    // -------------------------------------------------------------------
    // Hierarchy building
    // -------------------------------------------------------------------

    // Build complete HIERARCHY section string
    std::string buildHierarchy() const;

    // Recursively write a joint and its children
    void writeJoint(std::string& output, int jointIndex, int depth) const;

    // Write End Site for leaf joints
    void writeEndSite(std::string& output, int jointIndex, int depth) const;

    // -------------------------------------------------------------------
    // Motion data building
    // -------------------------------------------------------------------

    // Build complete MOTION section string
    std::string buildMotion(const AnimClip& clip) const;

    // Write a single frame's data for one joint
    void writeJointFrameData(
        std::ostringstream& oss,
        const SkeletonFrame& frame,
        int jointIndex,
        bool isRoot) const;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    // Get the BVH-compatible bone name (applying overrides)
    std::string getBoneName(int jointIndex) const;

    // Get OFFSET for a joint from rest pose
    Vec3 getJointOffset(int jointIndex) const;

    // Compute end-site offset direction from parent bone direction
    Vec3 getEndSiteOffset(int jointIndex) const;

    // Build joint traversal order (depth-first matching hierarchy)
    void buildTraversalOrder() const;

    // Get children of a joint
    std::vector<int> getChildren(int jointIndex) const;

    // Extract Euler angles from a joint in the configured rotation order
    Vec3 extractEulerForBVH(const JointTransform& joint) const;

    // Get CHANNELS string for the configured rotation order
    std::string getRotationChannelString() const;

    // Sanitise clip name for filesystem use
    static std::string sanitiseFilename(const std::string& name);
};

} // namespace hm::xport
