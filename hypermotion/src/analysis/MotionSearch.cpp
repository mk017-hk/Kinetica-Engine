#include "HyperMotion/analysis/MotionSearch.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::analysis {

static constexpr const char* TAG = "MotionSearch";

struct MotionSearch::Impl {
    MotionSearchConfig config;

    // Indexed entries: pointers into the database (valid while database lives)
    struct IndexEntry {
        const dataset::AnimationEntry* entry = nullptr;
        int originalIndex = -1;
    };
    std::vector<IndexEntry> index;

    // Cosine similarity between two L2-normalized vectors.
    static float cosineSimilarity(
        const std::array<float, MOTION_EMBEDDING_DIM>& a,
        const std::array<float, MOTION_EMBEDDING_DIM>& b) {
        float dot = 0.0f;
        for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    // Euclidean distance between two vectors.
    static float euclideanDistance(
        const std::array<float, MOTION_EMBEDDING_DIM>& a,
        const std::array<float, MOTION_EMBEDDING_DIM>& b) {
        float sum = 0.0f;
        for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }
};

MotionSearch::MotionSearch(const MotionSearchConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionSearch::~MotionSearch() = default;
MotionSearch::MotionSearch(MotionSearch&&) noexcept = default;
MotionSearch& MotionSearch::operator=(MotionSearch&&) noexcept = default;

void MotionSearch::buildIndex(const dataset::AnimationDatabase& database) {
    impl_->index.clear();

    const auto& entries = database.entries();
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].hasMotionEmbedding) {
            impl_->index.push_back({&entries[i], static_cast<int>(i)});
        }
    }

    HM_LOG_INFO(TAG, "Built search index: " + std::to_string(impl_->index.size()) +
                " entries with embeddings (out of " +
                std::to_string(entries.size()) + " total)");
}

std::vector<SearchResult> MotionSearch::search(
    const std::array<float, MOTION_EMBEDDING_DIM>& queryEmbedding,
    int maxResults) const {

    if (maxResults < 0) maxResults = impl_->config.maxResults;

    std::vector<SearchResult> results;
    results.reserve(impl_->index.size());

    bool useCosine = (impl_->config.metric == MotionSearchConfig::Metric::Cosine);

    for (const auto& idx : impl_->index) {
        SearchResult r;
        r.entry = idx.entry;
        r.index = idx.originalIndex;

        if (useCosine) {
            r.similarity = Impl::cosineSimilarity(queryEmbedding, idx.entry->motionEmbedding);
            r.distance = 1.0f - r.similarity;  // cosine distance
        } else {
            r.distance = Impl::euclideanDistance(queryEmbedding, idx.entry->motionEmbedding);
            r.similarity = 1.0f / (1.0f + r.distance);
        }

        // Apply distance threshold if configured
        if (impl_->config.maxDistance >= 0.0f && r.distance > impl_->config.maxDistance) {
            continue;
        }

        results.push_back(r);
    }

    // Sort by distance (ascending) or similarity (descending)
    if (useCosine) {
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.similarity > b.similarity;
                  });
    } else {
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
    }

    if (static_cast<int>(results.size()) > maxResults) {
        results.resize(maxResults);
    }

    return results;
}

std::vector<SearchResult> MotionSearch::searchSimilar(
    const dataset::AnimationEntry& entry, int maxResults) const {
    if (!entry.hasMotionEmbedding) return {};
    return search(entry.motionEmbedding, maxResults);
}

int MotionSearch::indexSize() const {
    return static_cast<int>(impl_->index.size());
}

} // namespace hm::analysis
