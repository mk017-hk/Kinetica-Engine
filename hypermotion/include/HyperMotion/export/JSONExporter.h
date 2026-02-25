#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <vector>
#include <functional>
#include <ostream>
#include <filesystem>

namespace hm::xport {

// -----------------------------------------------------------------------
// JSON Export Configuration
// -----------------------------------------------------------------------

struct JSONExportConfig {
    // Data inclusion flags
    bool includePositions = true;        // Joint world positions
    bool includeQuaternions = true;      // Joint quaternion rotations
    bool includeEuler = true;            // Joint Euler angle rotations (degrees)
    bool includeRotation6D = false;      // 6D rotation representation
    bool includeSegments = true;         // Motion segments
    bool includeMetadata = true;         // Clip metadata (name, fps, etc.)
    bool includeHierarchy = true;        // Bone hierarchy (parent indices)
    bool includeRestPose = true;         // Rest pose offsets for each joint
    bool includeRootMotion = true;       // Root position/velocity per frame
    bool includeConfidence = true;       // Per-joint confidence values
    bool includeStyleData = false;       // Player style data (if available)
    bool includeFrameTimestamps = true;  // Per-frame timestamps

    // Formatting
    int floatPrecision = 4;              // Decimal places for float values
    bool prettyPrint = true;             // Indented output (human readable)
    int indentSpaces = 2;               // Number of spaces for indentation (if pretty)

    // Streaming options (for large clips)
    bool enableStreaming = false;        // Use streaming export for large clips
    int streamingFrameChunkSize = 100;   // Number of frames per chunk in streaming mode

    // Compression-friendly options
    bool useArraysNotObjects = false;    // Use arrays instead of named objects for joints
                                          // (smaller output, needs schema to interpret)
};

// -----------------------------------------------------------------------
// JSON Export Statistics
// -----------------------------------------------------------------------

struct JSONExportStats {
    size_t totalBytes = 0;
    int framesExported = 0;
    int jointsPerFrame = 0;
    int segmentsExported = 0;
    float durationSeconds = 0.0f;
    double exportTimeMs = 0.0;
};

// -----------------------------------------------------------------------
// JSON Exporter
// -----------------------------------------------------------------------

class JSONExporter {
public:
    explicit JSONExporter(const JSONExportConfig& config = {});
    ~JSONExporter();

    // Non-copyable, movable
    JSONExporter(const JSONExporter&) = delete;
    JSONExporter& operator=(const JSONExporter&) = delete;
    JSONExporter(JSONExporter&&) noexcept = default;
    JSONExporter& operator=(JSONExporter&&) noexcept = default;

    // -------------------------------------------------------------------
    // Single clip export
    // -------------------------------------------------------------------

    // Export single clip to JSON file
    bool exportToFile(const AnimClip& clip, const std::string& path);

    // Export to JSON string
    std::string exportToString(const AnimClip& clip);

    // -------------------------------------------------------------------
    // Batch export
    // -------------------------------------------------------------------

    // Export multiple clips to a single JSON file (array of clips)
    bool exportBatchToFile(const std::vector<AnimClip>& clips, const std::string& path);

    // Export multiple clips to individual JSON files in a directory
    struct BatchResult {
        int totalClips = 0;
        int successCount = 0;
        int failCount = 0;
        std::vector<std::string> exportedPaths;
        std::vector<std::string> failedClips;
    };

    using ProgressCallback = std::function<void(int current, int total, const std::string& clipName)>;

    BatchResult exportBatchToDirectory(
        const std::vector<AnimClip>& clips,
        const std::string& outputDirectory,
        const std::string& prefix = "hm",
        ProgressCallback progress = nullptr);

    // -------------------------------------------------------------------
    // Streaming export (for large clips that may not fit in memory as a string)
    // -------------------------------------------------------------------

    // Export clip directly to an output stream in chunks
    bool exportToStream(const AnimClip& clip, std::ostream& out);

    // -------------------------------------------------------------------
    // Style-enriched export
    // -------------------------------------------------------------------

    // Export clip with associated player style data
    bool exportWithStyle(const AnimClip& clip, const PlayerStyle& style, const std::string& path);
    std::string exportWithStyleToString(const AnimClip& clip, const PlayerStyle& style);

    // -------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------

    // Get export statistics from the last export operation
    const JSONExportStats& getLastExportStats() const { return lastStats_; }

    // -------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------

    void setConfig(const JSONExportConfig& config) { config_ = config; }
    const JSONExportConfig& getConfig() const { return config_; }

private:
    JSONExportConfig config_;
    mutable JSONExportStats lastStats_;

    // -------------------------------------------------------------------
    // Internal JSON building
    // -------------------------------------------------------------------

    // Build metadata section
    void buildMetadata(void* jsonObj, const AnimClip& clip) const;

    // Build hierarchy section
    void buildHierarchy(void* jsonObj) const;

    // Build rest pose section
    void buildRestPose(void* jsonObj) const;

    // Build a single frame's JSON
    void buildFrame(void* jsonObj, const SkeletonFrame& frame) const;

    // Build a single joint's JSON
    void buildJoint(void* jsonObj, const JointTransform& joint, int jointIndex) const;

    // Build a single joint's JSON in compact array format
    void buildJointCompact(void* jsonArr, const JointTransform& joint, int jointIndex) const;

    // Build segments section
    void buildSegments(void* jsonArr, const std::vector<MotionSegment>& segments) const;

    // Build style section
    void buildStyle(void* jsonObj, const PlayerStyle& style) const;

    // -------------------------------------------------------------------
    // Streaming helpers
    // -------------------------------------------------------------------

    // Write JSON opening structure to stream
    void writeStreamHeader(std::ostream& out, const AnimClip& clip) const;

    // Write a chunk of frames to stream
    void writeStreamFrameChunk(std::ostream& out,
                                const AnimClip& clip,
                                int startFrame, int endFrame,
                                bool isFirstChunk) const;

    // Write JSON closing structure to stream
    void writeStreamFooter(std::ostream& out, const AnimClip& clip) const;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    // Ensure output directory exists
    static bool ensureDirectoryExists(const std::string& path);

    // Sanitise filename
    static std::string sanitiseFilename(const std::string& name);

    // Round float to configured precision
    float roundToPrecision(float value) const;
};

} // namespace hm::xport
