#pragma once

#include "HyperMotion/core/Types.h"
#include <string>

namespace hm::xport {

struct BVHExportConfig {
    float scale = 1.0f;           // Scale factor for positions
    bool exportRootMotion = true;
    int floatPrecision = 6;
};

class BVHExporter {
public:
    explicit BVHExporter(const BVHExportConfig& config = {});
    ~BVHExporter();

    // Export AnimClip to BVH file
    bool exportToFile(const AnimClip& clip, const std::string& path);

    // Export to string
    std::string exportToString(const AnimClip& clip);

private:
    BVHExportConfig config_;

    // Build HIERARCHY section
    std::string buildHierarchy() const;

    // Write recursive joint hierarchy
    void writeJoint(std::string& output, int jointIndex, int depth) const;

    // Build MOTION section
    std::string buildMotion(const AnimClip& clip) const;
};

} // namespace hm::xport
