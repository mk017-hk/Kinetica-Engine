#include "HyperMotion/dataset/AnimationDatabase.h"
#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/core/Logger.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>

namespace hm::dataset {

static constexpr const char* TAG = "AnimationDatabase";

struct AnimationDatabase::Impl {
    std::vector<AnimationEntry> entries;
};

AnimationDatabase::AnimationDatabase()
    : impl_(std::make_unique<Impl>()) {}

AnimationDatabase::~AnimationDatabase() = default;
AnimationDatabase::AnimationDatabase(AnimationDatabase&&) noexcept = default;
AnimationDatabase& AnimationDatabase::operator=(AnimationDatabase&&) noexcept = default;

void AnimationDatabase::addEntry(AnimationEntry entry) {
    impl_->entries.push_back(std::move(entry));
}

void AnimationDatabase::addEntries(std::vector<AnimationEntry> entries) {
    impl_->entries.reserve(impl_->entries.size() + entries.size());
    for (auto& e : entries) {
        impl_->entries.push_back(std::move(e));
    }
}

const std::vector<AnimationEntry>& AnimationDatabase::entries() const {
    return impl_->entries;
}

std::vector<const AnimationEntry*> AnimationDatabase::entriesByType(MotionType type) const {
    std::vector<const AnimationEntry*> result;
    for (const auto& e : impl_->entries) {
        if (e.classification.type == type)
            result.push_back(&e);
    }
    return result;
}

std::vector<const AnimationEntry*> AnimationDatabase::entriesByPlayer(int playerID) const {
    std::vector<const AnimationEntry*> result;
    for (const auto& e : impl_->entries) {
        if (e.clipMeta.playerID == playerID)
            result.push_back(&e);
    }
    return result;
}

DatabaseStats AnimationDatabase::stats() const {
    DatabaseStats s;
    s.totalClips = static_cast<int>(impl_->entries.size());
    std::set<int> players;

    for (const auto& e : impl_->entries) {
        int nf = static_cast<int>(e.clip.frames.size());
        s.totalFrames += nf;
        s.totalDurationSec += e.clipMeta.durationSec;
        s.clipsByType[static_cast<int>(e.classification.type)]++;
        if (e.clipMeta.playerID >= 0) players.insert(e.clipMeta.playerID);
    }
    s.uniquePlayers = static_cast<int>(players.size());
    return s;
}

int AnimationDatabase::exportToDirectory(const std::string& rootDir,
                                          bool exportBVH,
                                          bool exportJSON) const {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(rootDir, ec);
    if (ec) {
        HM_LOG_ERROR(TAG, "Cannot create output directory: " + rootDir + " (" + ec.message() + ")");
        return 0;
    }

    // Create subdirectories for each motion type
    for (int i = 0; i < MOTION_TYPE_COUNT; ++i) {
        std::string typeName = MOTION_TYPE_NAMES[i];
        for (auto& c : typeName) c = static_cast<char>(std::tolower(c));
        fs::create_directories(rootDir + "/" + typeName, ec);
        if (ec) {
            HM_LOG_WARN(TAG, "Cannot create subdirectory: " + typeName + " (" + ec.message() + ")");
        }
    }

    xport::BVHExporter bvhExporter;
    xport::JSONExporter jsonExporter;

    int exported = 0;
    int failed = 0;
    for (size_t i = 0; i < impl_->entries.size(); ++i) {
        const auto& entry = impl_->entries[i];
        int typeIdx = static_cast<int>(entry.classification.type);
        std::string typeName = MOTION_TYPE_NAMES[typeIdx];
        for (auto& c : typeName) c = static_cast<char>(std::tolower(c));

        // Build filename
        std::ostringstream oss;
        oss << "clip_" << std::setw(4) << std::setfill('0') << i;
        std::string basePath = rootDir + "/" + typeName + "/" + oss.str();

        bool clipOk = true;

        if (exportBVH) {
            if (!bvhExporter.exportToFile(entry.clip, basePath + ".bvh")) {
                HM_LOG_WARN(TAG, "BVH export failed for clip " + std::to_string(i));
                clipOk = false;
            }
        }
        if (exportJSON) {
            if (!jsonExporter.exportToFile(entry.clip, basePath + ".json")) {
                HM_LOG_WARN(TAG, "JSON export failed for clip " + std::to_string(i));
                clipOk = false;
            }
        }

        // Write clip metadata
        nlohmann::json meta;
        meta["playerID"] = entry.clipMeta.playerID;
        meta["motionType"] = entry.classification.label;
        meta["confidence"] = entry.classification.confidence;
        meta["duration"] = entry.clipMeta.durationSec;
        meta["frames"] = static_cast<int>(entry.clip.frames.size());
        meta["avgVelocity"] = entry.clipMeta.avgVelocity;
        meta["maxVelocity"] = entry.clipMeta.maxVelocity;
        meta["qualityScore"] = entry.quality.overallScore;
        meta["startFrame"] = entry.clipMeta.startFrame;
        meta["endFrame"] = entry.clipMeta.endFrame;

        // Foot contacts summary
        int contactFrames = 0;
        for (const auto& fc : entry.footContacts) {
            if (fc.leftFootContact || fc.rightFootContact) contactFrames++;
        }
        meta["footContactFrames"] = contactFrames;
        meta["clusterID"] = entry.clip.clusterID;

        // Motion embedding
        if (entry.hasMotionEmbedding) {
            nlohmann::json embArr = nlohmann::json::array();
            for (int d = 0; d < MOTION_EMBEDDING_DIM; ++d) {
                embArr.push_back(entry.motionEmbedding[d]);
            }
            meta["motionEmbedding"] = embArr;
        }

        std::ofstream metaFile(basePath + ".meta.json");
        if (metaFile.is_open()) {
            metaFile << meta.dump(2) << "\n";
            metaFile.flush();
            if (metaFile.fail()) {
                HM_LOG_WARN(TAG, "Metadata write failed for clip " + std::to_string(i));
                clipOk = false;
            }
        } else {
            HM_LOG_WARN(TAG, "Cannot open metadata file: " + basePath + ".meta.json");
            clipOk = false;
        }

        if (clipOk) {
            exported++;
        } else {
            failed++;
        }
    }

    if (failed > 0) {
        HM_LOG_WARN(TAG, "Export completed with " + std::to_string(failed) + " failures");
    }
    HM_LOG_INFO(TAG, "Exported " + std::to_string(exported) + "/" +
                std::to_string(impl_->entries.size()) + " clips to: " + rootDir);
    return exported;
}

std::string AnimationDatabase::exportSummaryJSON() const {
    auto s = stats();
    nlohmann::json j;
    j["schemaVersion"] = HM_SCHEMA_VERSION;
    j["totalClips"] = s.totalClips;
    j["totalFrames"] = s.totalFrames;
    j["totalDurationSec"] = s.totalDurationSec;
    j["uniquePlayers"] = s.uniquePlayers;

    nlohmann::json byType;
    for (int i = 0; i < MOTION_TYPE_COUNT; ++i) {
        if (s.clipsByType[i] > 0) {
            std::string name = MOTION_TYPE_NAMES[i];
            for (auto& c : name) c = static_cast<char>(std::tolower(c));
            byType[name] = s.clipsByType[i];
        }
    }
    j["clipsByType"] = byType;

    return j.dump(2);
}

bool AnimationDatabase::saveSummary(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        HM_LOG_ERROR(TAG, "Cannot write summary: " + path);
        return false;
    }
    ofs << exportSummaryJSON() << "\n";
    ofs.flush();
    if (ofs.fail()) {
        HM_LOG_ERROR(TAG, "Failed to write summary to: " + path);
        return false;
    }
    return true;
}

void AnimationDatabase::clear() {
    impl_->entries.clear();
}

} // namespace hm::dataset
