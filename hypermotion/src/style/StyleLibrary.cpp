#include "HyperMotion/style/StyleLibrary.h"
#include "HyperMotion/core/Logger.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::style {

static constexpr const char* TAG = "StyleLibrary";

StyleLibrary::StyleLibrary() = default;
StyleLibrary::~StyleLibrary() = default;

void StyleLibrary::addStyle(const PlayerStyle& style) {
    styles_[style.playerID] = style;
}

void StyleLibrary::removeStyle(const std::string& playerID) {
    styles_.erase(playerID);
}

std::optional<PlayerStyle> StyleLibrary::getStyle(const std::string& playerID) const {
    auto it = styles_.find(playerID);
    if (it != styles_.end()) return it->second;
    return std::nullopt;
}

bool StyleLibrary::hasStyle(const std::string& playerID) const {
    return styles_.count(playerID) > 0;
}

std::vector<std::string> StyleLibrary::getAllPlayerIDs() const {
    std::vector<std::string> ids;
    ids.reserve(styles_.size());
    for (const auto& [id, _] : styles_) {
        ids.push_back(id);
    }
    return ids;
}

int StyleLibrary::size() const {
    return static_cast<int>(styles_.size());
}

PlayerStyle StyleLibrary::interpolate(const PlayerStyle& a, const PlayerStyle& b, float t) {
    PlayerStyle result;
    result.playerID = a.playerID + "_blend_" + b.playerID;
    result.playerName = a.playerName + "/" + b.playerName;

    for (int i = 0; i < STYLE_DIM; ++i) {
        result.embedding[i] = a.embedding[i] * (1.0f - t) + b.embedding[i] * t;
    }

    // Normalize embedding
    float norm = 0.0f;
    for (int i = 0; i < STYLE_DIM; ++i) norm += result.embedding[i] * result.embedding[i];
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (int i = 0; i < STYLE_DIM; ++i) result.embedding[i] /= norm;
    }

    // Interpolate manual overrides
    result.strideLengthScale = a.strideLengthScale * (1 - t) + b.strideLengthScale * t;
    result.armSwingIntensity = a.armSwingIntensity * (1 - t) + b.armSwingIntensity * t;
    result.sprintLeanAngle = a.sprintLeanAngle * (1 - t) + b.sprintLeanAngle * t;
    result.hipRotationScale = a.hipRotationScale * (1 - t) + b.hipRotationScale * t;
    result.kneeLiftScale = a.kneeLiftScale * (1 - t) + b.kneeLiftScale * t;
    result.cadenceScale = a.cadenceScale * (1 - t) + b.cadenceScale * t;
    result.decelerationSharpness = a.decelerationSharpness * (1 - t) + b.decelerationSharpness * t;
    result.turnLeadBody = a.turnLeadBody * (1 - t) + b.turnLeadBody * t;

    return result;
}

float StyleLibrary::embeddingDistance(const std::array<float, STYLE_DIM>& a,
                                       const std::array<float, STYLE_DIM>& b) {
    float dist = 0.0f;
    for (int i = 0; i < STYLE_DIM; ++i) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return std::sqrt(dist);
}

std::optional<PlayerStyle> StyleLibrary::findNearest(
    const std::array<float, STYLE_DIM>& embedding, float maxDistance) const {

    float bestDist = std::numeric_limits<float>::max();
    const PlayerStyle* bestStyle = nullptr;

    for (const auto& [_, style] : styles_) {
        float dist = embeddingDistance(embedding, style.embedding);
        if (dist < bestDist) {
            bestDist = dist;
            bestStyle = &style;
        }
    }

    if (bestStyle && (maxDistance < 0 || bestDist <= maxDistance)) {
        return *bestStyle;
    }
    return std::nullopt;
}

std::vector<PlayerStyle> StyleLibrary::findKNearest(
    const std::array<float, STYLE_DIM>& embedding, int k) const {

    std::vector<std::pair<float, const PlayerStyle*>> distances;
    for (const auto& [_, style] : styles_) {
        float dist = embeddingDistance(embedding, style.embedding);
        distances.push_back({dist, &style});
    }

    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<PlayerStyle> result;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        result.push_back(*distances[i].second);
    }
    return result;
}

bool StyleLibrary::saveJSON(const std::string& path) const {
    try {
        nlohmann::json j;
        j["version"] = "1.0";
        j["styles"] = nlohmann::json::array();

        for (const auto& [_, style] : styles_) {
            nlohmann::json s;
            s["playerID"] = style.playerID;
            s["playerName"] = style.playerName;
            s["embedding"] = style.embedding;
            s["strideLengthScale"] = style.strideLengthScale;
            s["armSwingIntensity"] = style.armSwingIntensity;
            s["sprintLeanAngle"] = style.sprintLeanAngle;
            s["hipRotationScale"] = style.hipRotationScale;
            s["kneeLiftScale"] = style.kneeLiftScale;
            s["cadenceScale"] = style.cadenceScale;
            s["decelerationSharpness"] = style.decelerationSharpness;
            s["turnLeadBody"] = style.turnLeadBody;
            j["styles"].push_back(s);
        }

        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            HM_LOG_ERROR(TAG, "Cannot open file for writing: " + path);
            return false;
        }
        ofs << j.dump(2);
        HM_LOG_INFO(TAG, "Saved " + std::to_string(styles_.size()) + " styles to: " + path);
        return true;

    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Failed to save JSON: ") + e.what());
        return false;
    }
}

bool StyleLibrary::loadJSON(const std::string& path) {
    try {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            HM_LOG_ERROR(TAG, "Cannot open file: " + path);
            return false;
        }

        nlohmann::json j;
        ifs >> j;

        styles_.clear();

        for (const auto& s : j["styles"]) {
            PlayerStyle style;
            style.playerID = s["playerID"].get<std::string>();
            style.playerName = s["playerName"].get<std::string>();

            auto embArray = s["embedding"].get<std::array<float, STYLE_DIM>>();
            style.embedding = embArray;

            style.strideLengthScale = s.value("strideLengthScale", 1.0f);
            style.armSwingIntensity = s.value("armSwingIntensity", 1.0f);
            style.sprintLeanAngle = s.value("sprintLeanAngle", 0.0f);
            style.hipRotationScale = s.value("hipRotationScale", 1.0f);
            style.kneeLiftScale = s.value("kneeLiftScale", 1.0f);
            style.cadenceScale = s.value("cadenceScale", 1.0f);
            style.decelerationSharpness = s.value("decelerationSharpness", 1.0f);
            style.turnLeadBody = s.value("turnLeadBody", 0.0f);

            styles_[style.playerID] = style;
        }

        HM_LOG_INFO(TAG, "Loaded " + std::to_string(styles_.size()) + " styles from: " + path);
        return true;

    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Failed to load JSON: ") + e.what());
        return false;
    }
}

void StyleLibrary::clear() {
    styles_.clear();
}

} // namespace hm::style
