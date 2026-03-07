#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/dataset/AnimationDatabase.h"
#include <array>
#include <string>
#include <vector>
#include <memory>

namespace hm::analysis {

/// A search result: a database entry with its distance to the query.
struct SearchResult {
    const dataset::AnimationEntry* entry = nullptr;
    float distance = 0.0f;   // lower is more similar
    float similarity = 0.0f; // higher is more similar (cosine similarity)
    int index = -1;           // index into the database
};

/// Configuration for motion search.
struct MotionSearchConfig {
    enum class Metric { Cosine, Euclidean };
    Metric metric = Metric::Cosine;
    int maxResults = 10;
    float maxDistance = -1.0f;  // -1 = no distance threshold
};

/// Searches an AnimationDatabase by motion embedding similarity.
///
/// Given a query embedding vector, returns the nearest animation clips
/// ranked by cosine similarity or Euclidean distance.
class MotionSearch {
public:
    explicit MotionSearch(const MotionSearchConfig& config = {});
    ~MotionSearch();

    MotionSearch(const MotionSearch&) = delete;
    MotionSearch& operator=(const MotionSearch&) = delete;
    MotionSearch(MotionSearch&&) noexcept;
    MotionSearch& operator=(MotionSearch&&) noexcept;

    /// Build the search index from a database. Only entries with
    /// hasMotionEmbedding == true are indexed.
    void buildIndex(const dataset::AnimationDatabase& database);

    /// Search for nearest animations to a query embedding.
    std::vector<SearchResult> search(
        const std::array<float, MOTION_EMBEDDING_DIM>& queryEmbedding,
        int maxResults = -1) const;

    /// Search using the embedding from an existing database entry.
    std::vector<SearchResult> searchSimilar(
        const dataset::AnimationEntry& entry,
        int maxResults = -1) const;

    /// Number of indexed entries.
    int indexSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::analysis
