#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/core/Logger.h"

#include <nlohmann/json.hpp>
#include <fstream>

namespace hm::xport {

static constexpr const char* TAG = "JSONExporter";

JSONExporter::JSONExporter(const JSONExportConfig& config)
    : config_(config) {}

JSONExporter::~JSONExporter() = default;

std::string JSONExporter::exportToString(const AnimClip& clip) {
    nlohmann::json j;

    // Schema version (always included)
    j["schemaVersion"] = HM_SCHEMA_VERSION;

    // Metadata
    if (config_.includeMetadata) {
        j["metadata"]["name"] = clip.name;
        j["metadata"]["fps"] = clip.fps;
        j["metadata"]["trackingID"] = clip.trackingID;
        j["metadata"]["numFrames"] = clip.frames.size();
        j["metadata"]["numJoints"] = JOINT_COUNT;
        j["metadata"]["jointNames"] = nlohmann::json::array();
        for (int i = 0; i < JOINT_COUNT; ++i) {
            j["metadata"]["jointNames"].push_back(JOINT_NAMES[i]);
        }
        j["metadata"]["hierarchy"] = nlohmann::json::array();
        for (int i = 0; i < JOINT_COUNT; ++i) {
            j["metadata"]["hierarchy"].push_back(JOINT_PARENT[i]);
        }
    }

    // Frames
    j["frames"] = nlohmann::json::array();
    for (const auto& frame : clip.frames) {
        nlohmann::json fj;
        fj["timestamp"] = frame.timestamp;
        fj["frameIndex"] = frame.frameIndex;
        fj["rootPosition"] = {frame.rootPosition.x, frame.rootPosition.y, frame.rootPosition.z};
        fj["rootVelocity"] = {frame.rootVelocity.x, frame.rootVelocity.y, frame.rootVelocity.z};

        if (config_.includeQuaternions) {
            fj["rootRotation"] = {frame.rootRotation.w, frame.rootRotation.x,
                                   frame.rootRotation.y, frame.rootRotation.z};
        }

        fj["joints"] = nlohmann::json::array();
        for (int i = 0; i < JOINT_COUNT; ++i) {
            nlohmann::json jj;
            jj["name"] = JOINT_NAMES[i];

            if (config_.includePositions) {
                jj["position"] = {
                    frame.joints[i].worldPosition.x,
                    frame.joints[i].worldPosition.y,
                    frame.joints[i].worldPosition.z
                };
            }

            if (config_.includeQuaternions) {
                jj["quaternion"] = {
                    frame.joints[i].localRotation.w,
                    frame.joints[i].localRotation.x,
                    frame.joints[i].localRotation.y,
                    frame.joints[i].localRotation.z
                };
            }

            if (config_.includeEuler) {
                jj["euler"] = {
                    frame.joints[i].localEulerDeg.x,
                    frame.joints[i].localEulerDeg.y,
                    frame.joints[i].localEulerDeg.z
                };
            }

            if (config_.includeRotation6D) {
                jj["rotation6D"] = {
                    frame.joints[i].rotation6D[0],
                    frame.joints[i].rotation6D[1],
                    frame.joints[i].rotation6D[2],
                    frame.joints[i].rotation6D[3],
                    frame.joints[i].rotation6D[4],
                    frame.joints[i].rotation6D[5]
                };
            }

            jj["confidence"] = frame.joints[i].confidence;
            fj["joints"].push_back(jj);
        }

        j["frames"].push_back(fj);
    }

    // Segments
    if (config_.includeSegments && !clip.segments.empty()) {
        j["segments"] = nlohmann::json::array();
        for (const auto& seg : clip.segments) {
            nlohmann::json sj;
            sj["type"] = MOTION_TYPE_NAMES[static_cast<int>(seg.type)];
            sj["typeIndex"] = static_cast<int>(seg.type);
            sj["startFrame"] = seg.startFrame;
            sj["endFrame"] = seg.endFrame;
            sj["avgVelocity"] = seg.avgVelocity;
            sj["avgDirection"] = {seg.avgDirection.x, seg.avgDirection.y, seg.avgDirection.z};
            sj["confidence"] = seg.confidence;
            j["segments"].push_back(sj);
        }
    }

    if (config_.prettyPrint) {
        return j.dump(2);
    }
    return j.dump();
}

bool JSONExporter::exportToFile(const AnimClip& clip, const std::string& path) {
    std::string content = exportToString(clip);

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot open file: " + path);
        return false;
    }

    ofs << content;
    HM_LOG_INFO(TAG, "Exported JSON to: " + path);
    return true;
}

bool JSONExporter::exportBatchToFile(const std::vector<AnimClip>& clips, const std::string& path) {
    nlohmann::json j;
    j["clips"] = nlohmann::json::array();

    for (const auto& clip : clips) {
        auto clipStr = exportToString(clip);
        j["clips"].push_back(nlohmann::json::parse(clipStr));
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot open file: " + path);
        return false;
    }

    if (config_.prettyPrint) {
        ofs << j.dump(2);
    } else {
        ofs << j.dump();
    }

    HM_LOG_INFO(TAG, "Exported " + std::to_string(clips.size()) + " clips to: " + path);
    return true;
}

// -------------------------------------------------------------------
// Batch directory export
// -------------------------------------------------------------------

JSONExporter::BatchResult JSONExporter::exportBatchToDirectory(
    const std::vector<AnimClip>& clips,
    const std::string& outputDirectory,
    const std::string& prefix,
    ProgressCallback progress) {

    BatchResult result;
    result.totalClips = static_cast<int>(clips.size());

    std::filesystem::create_directories(outputDirectory);

    for (int i = 0; i < static_cast<int>(clips.size()); ++i) {
        const auto& clip = clips[i];
        std::string name = clip.name.empty() ? "clip_" + std::to_string(i) : clip.name;
        std::string path = outputDirectory + "/" + prefix + "_" + name + ".json";

        if (progress) progress(i + 1, result.totalClips, name);

        if (exportToFile(clip, path)) {
            result.successCount++;
            result.exportedPaths.push_back(path);
        } else {
            result.failCount++;
            result.failedClips.push_back(name);
        }
    }
    return result;
}

// -------------------------------------------------------------------
// Streaming export
// -------------------------------------------------------------------

bool JSONExporter::exportToStream(const AnimClip& clip, std::ostream& out) {
    std::string content = exportToString(clip);
    out << content;
    return !out.fail();
}

// -------------------------------------------------------------------
// Style-enriched export
// -------------------------------------------------------------------

bool JSONExporter::exportWithStyle(const AnimClip& clip, const PlayerStyle& style, const std::string& path) {
    std::string content = exportWithStyleToString(clip, style);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << content;
    return true;
}

std::string JSONExporter::exportWithStyleToString(const AnimClip& clip, const PlayerStyle& style) {
    // Get base export, parse, add style, re-serialize
    std::string base = exportToString(clip);
    auto j = nlohmann::json::parse(base);
    nlohmann::json sj;
    sj["playerID"] = style.playerID;
    sj["playerName"] = style.playerName;
    sj["embedding"] = style.embedding;
    j["style"] = sj;
    return config_.prettyPrint ? j.dump(2) : j.dump();
}

// -------------------------------------------------------------------
// Internal helpers (minimal implementations for declared interface)
// -------------------------------------------------------------------

void JSONExporter::buildMetadata(void*, const AnimClip&) const {}
void JSONExporter::buildHierarchy(void*) const {}
void JSONExporter::buildRestPose(void*) const {}
void JSONExporter::buildFrame(void*, const SkeletonFrame&) const {}
void JSONExporter::buildJoint(void*, const JointTransform&, int) const {}
void JSONExporter::buildJointCompact(void*, const JointTransform&, int) const {}
void JSONExporter::buildSegments(void*, const std::vector<MotionSegment>&) const {}
void JSONExporter::buildStyle(void*, const PlayerStyle&) const {}
void JSONExporter::writeStreamHeader(std::ostream&, const AnimClip&) const {}
void JSONExporter::writeStreamFrameChunk(std::ostream&, const AnimClip&, int, int, bool) const {}
void JSONExporter::writeStreamFooter(std::ostream&, const AnimClip&) const {}

bool JSONExporter::ensureDirectoryExists(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
}

std::string JSONExporter::sanitiseFilename(const std::string& name) {
    std::string result;
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') result += c;
        else if (c == ' ') result += '_';
    }
    return result.empty() ? "unnamed" : result;
}

float JSONExporter::roundToPrecision(float value) const {
    float factor = std::pow(10.0f, static_cast<float>(config_.floatPrecision));
    return std::round(value * factor) / factor;
}

} // namespace hm::xport
