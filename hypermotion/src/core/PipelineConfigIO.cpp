#include "HyperMotion/core/PipelineConfigIO.h"
#include "HyperMotion/core/Logger.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace hm {

using json = nlohmann::json;

static constexpr const char* TAG = "PipelineConfigIO";

// ------------------------------------------------------------------
// Helpers: read optional JSON fields into config values
// ------------------------------------------------------------------

template <typename T>
static void readOpt(const json& j, const char* key, T& dest) {
    if (j.contains(key) && !j[key].is_null())
        dest = j[key].get<T>();
}

// ------------------------------------------------------------------
// Deserialise
// ------------------------------------------------------------------

static void parseFromJson(const json& root, PipelineConfig& cfg) {
    // Top-level pipeline settings
    readOpt(root, "targetFPS", cfg.targetFPS);
    readOpt(root, "splitBySegment", cfg.splitBySegment);
    readOpt(root, "outputDirectory", cfg.outputDirectory);
    readOpt(root, "outputFormat", cfg.outputFormat);
    readOpt(root, "minTrackFrames", cfg.minTrackFrames);
    readOpt(root, "enableVisualization", cfg.enableVisualization);

    // Pose estimation
    if (root.contains("pose")) {
        const auto& p = root["pose"];
        readOpt(p, "targetFPS", cfg.poseConfig.targetFPS);
        readOpt(p, "enableVisualization", cfg.poseConfig.enableVisualization);

        if (p.contains("detector")) {
            const auto& d = p["detector"];
            readOpt(d, "modelPath", cfg.poseConfig.detector.modelPath);
            readOpt(d, "confidenceThreshold", cfg.poseConfig.detector.confidenceThreshold);
            readOpt(d, "nmsIouThreshold", cfg.poseConfig.detector.nmsIouThreshold);
            readOpt(d, "inputWidth", cfg.poseConfig.detector.inputWidth);
            readOpt(d, "inputHeight", cfg.poseConfig.detector.inputHeight);
            readOpt(d, "maxDetections", cfg.poseConfig.detector.maxDetections);
        }
        if (p.contains("poseEstimator")) {
            const auto& pe = p["poseEstimator"];
            readOpt(pe, "modelPath", cfg.poseConfig.poseEstimator.modelPath);
            readOpt(pe, "inputWidth", cfg.poseConfig.poseEstimator.inputWidth);
            readOpt(pe, "inputHeight", cfg.poseConfig.poseEstimator.inputHeight);
            readOpt(pe, "confidenceThreshold", cfg.poseConfig.poseEstimator.confidenceThreshold);
        }
        if (p.contains("depthLifter")) {
            const auto& dl = p["depthLifter"];
            readOpt(dl, "modelPath", cfg.poseConfig.depthLifter.modelPath);
            readOpt(dl, "useGeometricFallback", cfg.poseConfig.depthLifter.useGeometricFallback);
            readOpt(dl, "defaultSubjectHeight", cfg.poseConfig.depthLifter.defaultSubjectHeight);
        }
        if (p.contains("tracker")) {
            const auto& t = p["tracker"];
            readOpt(t, "iouWeight", cfg.poseConfig.tracker.iouWeight);
            readOpt(t, "oksWeight", cfg.poseConfig.tracker.oksWeight);
            readOpt(t, "reidWeight", cfg.poseConfig.tracker.reidWeight);
            readOpt(t, "minHitsToConfirm", cfg.poseConfig.tracker.minHitsToConfirm);
            readOpt(t, "lostTimeout", cfg.poseConfig.tracker.lostTimeout);
        }
    }

    // Skeleton mapping
    if (root.contains("skeleton")) {
        const auto& s = root["skeleton"];
        readOpt(s, "minConfidenceThreshold", cfg.mapperConfig.minConfidenceThreshold);
        readOpt(s, "useVelocitySmoothing", cfg.mapperConfig.useVelocitySmoothing);
        readOpt(s, "velocitySmoothingAlpha", cfg.mapperConfig.velocitySmoothingAlpha);
    }

    // Signal processing
    if (root.contains("signal")) {
        const auto& sig = root["signal"];
        readOpt(sig, "enableOutlierFilter", cfg.signalConfig.enableOutlierFilter);
        readOpt(sig, "enableSavitzkyGolay", cfg.signalConfig.enableSavitzkyGolay);
        readOpt(sig, "enableButterworth", cfg.signalConfig.enableButterworth);
        readOpt(sig, "enableQuaternionSmoothing", cfg.signalConfig.enableQuaternionSmoothing);
        readOpt(sig, "enableFootContact", cfg.signalConfig.enableFootContact);

        if (sig.contains("butterworth")) {
            const auto& b = sig["butterworth"];
            readOpt(b, "order", cfg.signalConfig.butterworthConfig.order);
            readOpt(b, "cutoffFreqBody", cfg.signalConfig.butterworthConfig.cutoffFreqBody);
            readOpt(b, "cutoffFreqExtrem", cfg.signalConfig.butterworthConfig.cutoffFreqExtrem);
            readOpt(b, "sampleRate", cfg.signalConfig.butterworthConfig.sampleRate);
        }
        if (sig.contains("savitzkyGolay")) {
            const auto& sg = sig["savitzkyGolay"];
            readOpt(sg, "windowSize", cfg.signalConfig.sgConfig.windowSize);
            readOpt(sg, "polyOrder", cfg.signalConfig.sgConfig.polyOrder);
        }
        if (sig.contains("footContact")) {
            const auto& fc = sig["footContact"];
            readOpt(fc, "velocityThreshold", cfg.signalConfig.footConfig.velocityThreshold);
            readOpt(fc, "heightThreshold", cfg.signalConfig.footConfig.heightThreshold);
            readOpt(fc, "groundHeight", cfg.signalConfig.footConfig.groundHeight);
        }
    }

    // Motion segmenter
    if (root.contains("segmenter")) {
        const auto& seg = root["segmenter"];
        readOpt(seg, "modelPath", cfg.segmenterConfig.modelPath);
        readOpt(seg, "slidingWindowSize", cfg.segmenterConfig.slidingWindowSize);
        readOpt(seg, "slidingWindowStride", cfg.segmenterConfig.slidingWindowStride);
        readOpt(seg, "minSegmentLength", cfg.segmenterConfig.minSegmentLength);
        readOpt(seg, "confidenceThreshold", cfg.segmenterConfig.confidenceThreshold);
    }

    // BVH export
    if (root.contains("bvhExport")) {
        const auto& b = root["bvhExport"];
        readOpt(b, "scale", cfg.bvhConfig.scale);
        readOpt(b, "exportRootMotion", cfg.bvhConfig.exportRootMotion);
        readOpt(b, "floatPrecision", cfg.bvhConfig.floatPrecision);
    }

    // JSON export
    if (root.contains("jsonExport")) {
        const auto& je = root["jsonExport"];
        readOpt(je, "includePositions", cfg.jsonConfig.includePositions);
        readOpt(je, "includeQuaternions", cfg.jsonConfig.includeQuaternions);
        readOpt(je, "includeEuler", cfg.jsonConfig.includeEuler);
        readOpt(je, "includeRotation6D", cfg.jsonConfig.includeRotation6D);
        readOpt(je, "includeSegments", cfg.jsonConfig.includeSegments);
        readOpt(je, "includeMetadata", cfg.jsonConfig.includeMetadata);
        readOpt(je, "prettyPrint", cfg.jsonConfig.prettyPrint);
        readOpt(je, "floatPrecision", cfg.jsonConfig.floatPrecision);
    }
}

// ------------------------------------------------------------------
// Serialise
// ------------------------------------------------------------------

static json toJson(const PipelineConfig& cfg) {
    json root;

    root["targetFPS"] = cfg.targetFPS;
    root["splitBySegment"] = cfg.splitBySegment;
    root["outputDirectory"] = cfg.outputDirectory;
    root["outputFormat"] = cfg.outputFormat;
    root["minTrackFrames"] = cfg.minTrackFrames;
    root["enableVisualization"] = cfg.enableVisualization;

    // Pose
    json pose;
    pose["targetFPS"] = cfg.poseConfig.targetFPS;
    pose["enableVisualization"] = cfg.poseConfig.enableVisualization;

    pose["detector"] = {
        {"modelPath", cfg.poseConfig.detector.modelPath},
        {"confidenceThreshold", cfg.poseConfig.detector.confidenceThreshold},
        {"nmsIouThreshold", cfg.poseConfig.detector.nmsIouThreshold},
        {"inputWidth", cfg.poseConfig.detector.inputWidth},
        {"inputHeight", cfg.poseConfig.detector.inputHeight},
        {"maxDetections", cfg.poseConfig.detector.maxDetections}
    };
    pose["poseEstimator"] = {
        {"modelPath", cfg.poseConfig.poseEstimator.modelPath},
        {"inputWidth", cfg.poseConfig.poseEstimator.inputWidth},
        {"inputHeight", cfg.poseConfig.poseEstimator.inputHeight},
        {"confidenceThreshold", cfg.poseConfig.poseEstimator.confidenceThreshold}
    };
    pose["depthLifter"] = {
        {"modelPath", cfg.poseConfig.depthLifter.modelPath},
        {"useGeometricFallback", cfg.poseConfig.depthLifter.useGeometricFallback},
        {"defaultSubjectHeight", cfg.poseConfig.depthLifter.defaultSubjectHeight}
    };
    pose["tracker"] = {
        {"iouWeight", cfg.poseConfig.tracker.iouWeight},
        {"oksWeight", cfg.poseConfig.tracker.oksWeight},
        {"reidWeight", cfg.poseConfig.tracker.reidWeight},
        {"minHitsToConfirm", cfg.poseConfig.tracker.minHitsToConfirm},
        {"lostTimeout", cfg.poseConfig.tracker.lostTimeout}
    };
    root["pose"] = pose;

    // Skeleton
    root["skeleton"] = {
        {"minConfidenceThreshold", cfg.mapperConfig.minConfidenceThreshold},
        {"useVelocitySmoothing", cfg.mapperConfig.useVelocitySmoothing},
        {"velocitySmoothingAlpha", cfg.mapperConfig.velocitySmoothingAlpha}
    };

    // Signal
    json sig;
    sig["enableOutlierFilter"] = cfg.signalConfig.enableOutlierFilter;
    sig["enableSavitzkyGolay"] = cfg.signalConfig.enableSavitzkyGolay;
    sig["enableButterworth"] = cfg.signalConfig.enableButterworth;
    sig["enableQuaternionSmoothing"] = cfg.signalConfig.enableQuaternionSmoothing;
    sig["enableFootContact"] = cfg.signalConfig.enableFootContact;
    sig["butterworth"] = {
        {"order", cfg.signalConfig.butterworthConfig.order},
        {"cutoffFreqBody", cfg.signalConfig.butterworthConfig.cutoffFreqBody},
        {"cutoffFreqExtrem", cfg.signalConfig.butterworthConfig.cutoffFreqExtrem},
        {"sampleRate", cfg.signalConfig.butterworthConfig.sampleRate}
    };
    sig["savitzkyGolay"] = {
        {"windowSize", cfg.signalConfig.sgConfig.windowSize},
        {"polyOrder", cfg.signalConfig.sgConfig.polyOrder}
    };
    sig["footContact"] = {
        {"velocityThreshold", cfg.signalConfig.footConfig.velocityThreshold},
        {"heightThreshold", cfg.signalConfig.footConfig.heightThreshold},
        {"groundHeight", cfg.signalConfig.footConfig.groundHeight}
    };
    root["signal"] = sig;

    // Segmenter
    root["segmenter"] = {
        {"modelPath", cfg.segmenterConfig.modelPath},
        {"slidingWindowSize", cfg.segmenterConfig.slidingWindowSize},
        {"slidingWindowStride", cfg.segmenterConfig.slidingWindowStride},
        {"minSegmentLength", cfg.segmenterConfig.minSegmentLength},
        {"confidenceThreshold", cfg.segmenterConfig.confidenceThreshold}
    };

    // BVH export
    root["bvhExport"] = {
        {"scale", cfg.bvhConfig.scale},
        {"exportRootMotion", cfg.bvhConfig.exportRootMotion},
        {"floatPrecision", cfg.bvhConfig.floatPrecision}
    };

    // JSON export
    root["jsonExport"] = {
        {"includePositions", cfg.jsonConfig.includePositions},
        {"includeQuaternions", cfg.jsonConfig.includeQuaternions},
        {"includeEuler", cfg.jsonConfig.includeEuler},
        {"includeRotation6D", cfg.jsonConfig.includeRotation6D},
        {"includeSegments", cfg.jsonConfig.includeSegments},
        {"includeMetadata", cfg.jsonConfig.includeMetadata},
        {"prettyPrint", cfg.jsonConfig.prettyPrint},
        {"floatPrecision", cfg.jsonConfig.floatPrecision}
    };

    return root;
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

bool parsePipelineConfig(const std::string& jsonStr, PipelineConfig& out) {
    try {
        auto root = json::parse(jsonStr);
        parseFromJson(root, out);
        return true;
    } catch (const json::exception& e) {
        HM_LOG_ERROR(TAG, std::string("JSON parse error: ") + e.what());
        return false;
    }
}

bool loadPipelineConfig(const std::string& path, PipelineConfig& out) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot open config file: " + path);
        return false;
    }
    try {
        json root = json::parse(ifs);
        parseFromJson(root, out);
        HM_LOG_INFO(TAG, "Loaded config from: " + path);
        return true;
    } catch (const json::exception& e) {
        HM_LOG_ERROR(TAG, "JSON parse error in " + path + ": " + e.what());
        return false;
    }
}

std::string serialisePipelineConfig(const PipelineConfig& config) {
    return toJson(config).dump(2);
}

bool savePipelineConfig(const std::string& path, const PipelineConfig& config) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot write config file: " + path);
        return false;
    }
    ofs << toJson(config).dump(2) << "\n";
    HM_LOG_INFO(TAG, "Saved config to: " + path);
    return true;
}

// ------------------------------------------------------------------
// Stats serialisation
// ------------------------------------------------------------------

std::string serialisePipelineStats(const PipelineStats& stats) {
    json j;
    j["poseExtractionMs"] = stats.poseExtractionMs;
    j["skeletonMappingMs"] = stats.skeletonMappingMs;
    j["signalProcessingMs"] = stats.signalProcessingMs;
    j["segmentationMs"] = stats.segmentationMs;
    j["exportMs"] = stats.exportMs;
    j["totalMs"] = stats.totalMs;
    j["totalFramesProcessed"] = stats.totalFramesProcessed;
    j["trackedPersons"] = stats.trackedPersons;
    j["clipsProduced"] = stats.clipsProduced;
    j["segmentsFound"] = stats.segmentsFound;
    return j.dump(2);
}

bool savePipelineStats(const std::string& path, const PipelineStats& stats) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot write stats file: " + path);
        return false;
    }
    ofs << serialisePipelineStats(stats) << "\n";
    HM_LOG_INFO(TAG, "Saved stats to: " + path);
    return true;
}

} // namespace hm
