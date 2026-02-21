#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

namespace hm::xport {

static constexpr const char* TAG = "BVHExporter";

BVHExporter::BVHExporter(const BVHExportConfig& config)
    : config_(config) {}

BVHExporter::~BVHExporter() = default;

void BVHExporter::writeJoint(std::string& output, int jointIndex, int depth) const {
    std::string indent(depth * 2, ' ');
    const auto& offsets = getRestPoseBoneOffsets();

    // Determine children
    std::vector<int> children;
    for (int i = 0; i < JOINT_COUNT; ++i) {
        if (JOINT_PARENT[i] == jointIndex) {
            children.push_back(i);
        }
    }

    bool isRoot = (jointIndex == 0);
    std::string jointType = isRoot ? "ROOT" : "JOINT";

    output += indent + jointType + " " + JOINT_NAMES[jointIndex] + "\n";
    output += indent + "{\n";

    // OFFSET
    Vec3 offset = offsets[jointIndex] * config_.scale;
    output += indent + "  OFFSET ";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(config_.floatPrecision);
    oss << offset.x << " " << offset.y << " " << offset.z;
    output += oss.str() + "\n";

    // CHANNELS
    if (isRoot) {
        output += indent + "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n";
    } else {
        output += indent + "  CHANNELS 3 Zrotation Xrotation Yrotation\n";
    }

    if (children.empty()) {
        // End site
        output += indent + "  End Site\n";
        output += indent + "  {\n";
        output += indent + "    OFFSET 0.000000 0.000000 0.000000\n";
        output += indent + "  }\n";
    } else {
        for (int child : children) {
            writeJoint(output, child, depth + 1);
        }
    }

    output += indent + "}\n";
}

std::string BVHExporter::buildHierarchy() const {
    std::string hierarchy = "HIERARCHY\n";
    writeJoint(hierarchy, 0, 0);
    return hierarchy;
}

std::string BVHExporter::buildMotion(const AnimClip& clip) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(config_.floatPrecision);

    oss << "MOTION\n";
    oss << "Frames: " << clip.frames.size() << "\n";
    oss << "Frame Time: " << (1.0f / clip.fps) << "\n";

    // Build joint traversal order (same as hierarchy)
    std::vector<int> jointOrder;
    std::function<void(int)> traverse = [&](int idx) {
        jointOrder.push_back(idx);
        for (int i = 0; i < JOINT_COUNT; ++i) {
            if (JOINT_PARENT[i] == idx) {
                traverse(i);
            }
        }
    };
    traverse(0);

    for (const auto& frame : clip.frames) {
        for (int idx : jointOrder) {
            if (idx == 0 && config_.exportRootMotion) {
                // Root: 6 channels (position + rotation)
                Vec3 pos = frame.rootPosition * config_.scale;
                oss << pos.x << " " << pos.y << " " << pos.z << " ";
            }

            // Rotation in ZXY Euler order (BVH convention)
            const auto& euler = frame.joints[idx].localEulerDeg;
            oss << euler.z << " " << euler.x << " " << euler.y;

            if (idx != jointOrder.back()) {
                oss << " ";
            }
        }
        oss << "\n";
    }

    return oss.str();
}

bool BVHExporter::exportToFile(const AnimClip& clip, const std::string& path) {
    std::string content = exportToString(clip);

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot open file: " + path);
        return false;
    }

    ofs << content;
    HM_LOG_INFO(TAG, "Exported BVH to: " + path + " (" +
                std::to_string(clip.frames.size()) + " frames)");
    return true;
}

std::string BVHExporter::exportToString(const AnimClip& clip) {
    return buildHierarchy() + buildMotion(clip);
}

} // namespace hm::xport
