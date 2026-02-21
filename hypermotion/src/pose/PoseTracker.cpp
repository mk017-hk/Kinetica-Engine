#include "HyperMotion/pose/PoseTracker.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>

namespace hm::pose {

static constexpr const char* TAG = "PoseTracker";

// COCO OKS sigmas for each of the 17 keypoints
static constexpr float OKS_SIGMAS[COCO_KEYPOINTS] = {
    0.026f, 0.025f, 0.025f, 0.035f, 0.035f, 0.079f, 0.079f, 0.072f, 0.072f,
    0.062f, 0.062f, 0.107f, 0.107f, 0.087f, 0.087f, 0.089f, 0.089f
};

struct PoseTracker::Impl {
    PoseTrackerConfig config;
    std::vector<Tracklet> tracklets;
    int nextID = 0;

    // Hungarian algorithm for optimal assignment
    std::vector<std::pair<int, int>> hungarianMatch(
        const std::vector<std::vector<float>>& costMatrix,
        int numRows, int numCols) {

        if (numRows == 0 || numCols == 0) return {};

        int n = std::max(numRows, numCols);
        std::vector<std::vector<float>> cost(n, std::vector<float>(n, 1e6f));

        for (int i = 0; i < numRows; ++i)
            for (int j = 0; j < numCols; ++j)
                cost[i][j] = costMatrix[i][j];

        // Hungarian algorithm (Kuhn-Munkres)
        std::vector<float> u(n + 1), v(n + 1);
        std::vector<int> p(n + 1), way(n + 1);

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
            std::vector<bool> used(n + 1, false);

            do {
                used[j0] = true;
                int i0 = p[j0];
                float delta = std::numeric_limits<float>::max();
                int j1 = -1;

                for (int j = 1; j <= n; ++j) {
                    if (!used[j]) {
                        float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (int j = 0; j <= n; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }

        std::vector<std::pair<int, int>> assignments;
        for (int j = 1; j <= n; ++j) {
            if (p[j] > 0 && p[j] - 1 < numRows && j - 1 < numCols) {
                if (costMatrix[p[j] - 1][j - 1] < config.maxMatchDistance) {
                    assignments.push_back({p[j] - 1, j - 1});
                }
            }
        }
        return assignments;
    }

    float computeOKS(const std::array<Keypoint2D, COCO_KEYPOINTS>& gt,
                      const std::array<Keypoint2D, COCO_KEYPOINTS>& pred,
                      float bboxArea) {
        if (bboxArea <= 0.0f) bboxArea = 1.0f;
        float s2 = bboxArea;
        float oks = 0.0f;
        int validCount = 0;

        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            if (gt[k].confidence < 0.1f) continue;
            float dx = gt[k].position.x - pred[k].position.x;
            float dy = gt[k].position.y - pred[k].position.y;
            float d2 = dx * dx + dy * dy;
            float sigma2 = OKS_SIGMAS[k] * OKS_SIGMAS[k] * 2.0f;
            oks += std::exp(-d2 / (2.0f * sigma2 * s2));
            ++validCount;
        }

        return validCount > 0 ? oks / validCount : 0.0f;
    }

    float cosineSimilarity(const std::array<float, REID_DIM>& a,
                           const std::array<float, REID_DIM>& b) {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (int i = 0; i < REID_DIM; ++i) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = std::sqrt(normA) * std::sqrt(normB);
        return denom > 1e-8f ? dot / denom : 0.0f;
    }

    void predictTracklets() {
        for (auto& t : tracklets) {
            if (t.velocityHistory.size() >= 2) {
                Vec2 avgVel{0, 0};
                int count = std::min(static_cast<int>(t.velocityHistory.size()), 5);
                for (int i = static_cast<int>(t.velocityHistory.size()) - count;
                     i < static_cast<int>(t.velocityHistory.size()); ++i) {
                    avgVel = avgVel + t.velocityHistory[i];
                }
                avgVel = avgVel / static_cast<float>(count);

                Vec2 lastCenter{t.lastDetection.bbox.centerX(),
                                t.lastDetection.bbox.centerY()};
                t.predictedCenter = lastCenter + avgVel;
            } else {
                t.predictedCenter = {t.lastDetection.bbox.centerX(),
                                     t.lastDetection.bbox.centerY()};
            }

            t.predictedBbox = t.lastDetection.bbox;
            t.predictedBbox.x = t.predictedCenter.x - t.predictedBbox.width * 0.5f;
            t.predictedBbox.y = t.predictedCenter.y - t.predictedBbox.height * 0.5f;
        }
    }
};

PoseTracker::PoseTracker(const PoseTrackerConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

PoseTracker::~PoseTracker() = default;
PoseTracker::PoseTracker(PoseTracker&&) noexcept = default;
PoseTracker& PoseTracker::operator=(PoseTracker&&) noexcept = default;

void PoseTracker::update(
    const std::vector<Detection>& detections,
    const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& poses) {

    // Step 1: Predict existing tracklet positions
    impl_->predictTracklets();

    int numTracks = static_cast<int>(impl_->tracklets.size());
    int numDets = static_cast<int>(detections.size());

    if (numTracks == 0 && numDets == 0) return;

    // Step 2: Build cost matrix
    std::vector<std::vector<float>> costMatrix(numTracks, std::vector<float>(numDets, 1.0f));

    for (int i = 0; i < numTracks; ++i) {
        for (int j = 0; j < numDets; ++j) {
            float iouCost = 1.0f - impl_->tracklets[i].predictedBbox.iou(detections[j].bbox);

            float oksCost = 1.0f;
            if (j < static_cast<int>(poses.size())) {
                float bboxArea = detections[j].bbox.area() /
                                 std::max(1.0f, detections[j].bbox.width * detections[j].bbox.height);
                oksCost = 1.0f - impl_->computeOKS(impl_->tracklets[i].lastPose, poses[j], bboxArea);
            }

            float reidCost = 1.0f;
            // ReID similarity (if features available)
            bool hasReID = false;
            for (int f = 0; f < REID_DIM; ++f) {
                if (std::abs(impl_->tracklets[i].reidFeature[f]) > 1e-8f) {
                    hasReID = true;
                    break;
                }
            }
            if (hasReID) {
                std::array<float, REID_DIM> detReID{};
                reidCost = 1.0f - impl_->cosineSimilarity(impl_->tracklets[i].reidFeature, detReID);
            }

            costMatrix[i][j] = impl_->config.iouWeight * iouCost
                              + impl_->config.oksWeight * oksCost
                              + impl_->config.reidWeight * reidCost;
        }
    }

    // Step 3: Hungarian matching
    auto matches = impl_->hungarianMatch(costMatrix, numTracks, numDets);

    std::vector<bool> trackMatched(numTracks, false);
    std::vector<bool> detMatched(numDets, false);

    // Step 4: Update matched tracklets
    for (const auto& [trackIdx, detIdx] : matches) {
        trackMatched[trackIdx] = true;
        detMatched[detIdx] = true;

        auto& t = impl_->tracklets[trackIdx];
        Vec2 prevCenter{t.lastDetection.bbox.centerX(), t.lastDetection.bbox.centerY()};
        Vec2 newCenter{detections[detIdx].bbox.centerX(), detections[detIdx].bbox.centerY()};

        t.lastDetection = detections[detIdx];
        t.classLabel = detections[detIdx].classLabel;
        t.hitCount++;
        t.framesSinceLast = 0;
        t.age++;

        if (detIdx < static_cast<int>(poses.size())) {
            t.lastPose = poses[detIdx];
        }

        t.positionHistory.push_back(newCenter);
        if (t.positionHistory.size() > 30) {
            t.positionHistory.erase(t.positionHistory.begin());
        }

        Vec2 velocity = newCenter - prevCenter;
        t.velocityHistory.push_back(velocity);
        if (t.velocityHistory.size() > 30) {
            t.velocityHistory.erase(t.velocityHistory.begin());
        }
    }

    // Step 5: Create new tracklets for unmatched detections
    for (int j = 0; j < numDets; ++j) {
        if (detMatched[j]) continue;
        if (static_cast<int>(impl_->tracklets.size()) >= impl_->config.maxTracklets) break;

        Tracklet newTrack;
        newTrack.id = impl_->nextID++;
        newTrack.classLabel = detections[j].classLabel;
        newTrack.age = 1;
        newTrack.hitCount = 1;
        newTrack.framesSinceLast = 0;
        newTrack.lastDetection = detections[j];

        if (j < static_cast<int>(poses.size())) {
            newTrack.lastPose = poses[j];
        }

        Vec2 center{detections[j].bbox.centerX(), detections[j].bbox.centerY()};
        newTrack.positionHistory.push_back(center);
        newTrack.predictedCenter = center;
        newTrack.predictedBbox = detections[j].bbox;

        impl_->tracklets.push_back(std::move(newTrack));
    }

    // Step 6: Age and prune unmatched tracklets
    for (int i = 0; i < numTracks; ++i) {
        if (!trackMatched[i]) {
            impl_->tracklets[i].framesSinceLast++;
            impl_->tracklets[i].age++;
        }
    }

    impl_->tracklets.erase(
        std::remove_if(impl_->tracklets.begin(), impl_->tracklets.end(),
                        [this](const Tracklet& t) {
                            return t.framesSinceLast > impl_->config.lostTimeout;
                        }),
        impl_->tracklets.end());
}

std::vector<Tracklet> PoseTracker::getConfirmedTracklets() const {
    std::vector<Tracklet> confirmed;
    for (const auto& t : impl_->tracklets) {
        if (t.hitCount >= impl_->config.minHitsToConfirm && t.framesSinceLast == 0) {
            confirmed.push_back(t);
        }
    }
    return confirmed;
}

std::vector<Tracklet> PoseTracker::getAllTracklets() const {
    return impl_->tracklets;
}

void PoseTracker::reset() {
    impl_->tracklets.clear();
    impl_->nextID = 0;
}

} // namespace hm::pose
