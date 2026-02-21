#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <vector>

namespace hm::xport {

struct JSONExportConfig {
    bool includePositions = true;
    bool includeQuaternions = true;
    bool includeEuler = true;
    bool includeRotation6D = false;
    bool includeSegments = true;
    bool includeMetadata = true;
    int floatPrecision = 4;
    bool prettyPrint = true;
};

class JSONExporter {
public:
    explicit JSONExporter(const JSONExportConfig& config = {});
    ~JSONExporter();

    // Export single clip to JSON file
    bool exportToFile(const AnimClip& clip, const std::string& path);

    // Export to JSON string
    std::string exportToString(const AnimClip& clip);

    // Export multiple clips
    bool exportBatchToFile(const std::vector<AnimClip>& clips, const std::string& path);

private:
    JSONExportConfig config_;
};

} // namespace hm::xport
