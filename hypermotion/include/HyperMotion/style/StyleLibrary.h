#pragma once

#include "HyperMotion/core/Types.h"
#include <map>
#include <string>
#include <vector>
#include <optional>

namespace hm::style {

class StyleLibrary {
public:
    StyleLibrary();
    ~StyleLibrary();

    // Add/update a player style
    void addStyle(const PlayerStyle& style);
    void removeStyle(const std::string& playerID);

    // Lookup
    std::optional<PlayerStyle> getStyle(const std::string& playerID) const;
    bool hasStyle(const std::string& playerID) const;
    std::vector<std::string> getAllPlayerIDs() const;
    int size() const;

    // Interpolate between two styles
    static PlayerStyle interpolate(const PlayerStyle& a, const PlayerStyle& b, float t);

    // Find nearest style by embedding distance
    std::optional<PlayerStyle> findNearest(const std::array<float, STYLE_DIM>& embedding,
                                            float maxDistance = -1.0f) const;

    // Find K nearest styles
    std::vector<PlayerStyle> findKNearest(const std::array<float, STYLE_DIM>& embedding,
                                           int k) const;

    // Serialization
    bool saveJSON(const std::string& path) const;
    bool loadJSON(const std::string& path);

    // Clear all styles
    void clear();

private:
    std::map<std::string, PlayerStyle> styles_;

    static float embeddingDistance(const std::array<float, STYLE_DIM>& a,
                                   const std::array<float, STYLE_DIM>& b);
};

} // namespace hm::style
