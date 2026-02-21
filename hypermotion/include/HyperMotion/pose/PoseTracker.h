#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <memory>

namespace hm::pose {

struct PoseTrackerConfig {
    float iouWeight = 0.4f;
    float oksWeight = 0.3f;
    float reidWeight = 0.3f;
    int minHitsToConfirm = 3;
    int lostTimeout = 30;
    float maxMatchDistance = 0.7f;
    int maxTracklets = 50;
};

struct Tracklet {
    int id = -1;
    std::string classLabel;
    int age = 0;
    int hitCount = 0;
    int framesSinceLast = 0;
    Detection lastDetection;
    std::array<Keypoint2D, COCO_KEYPOINTS> lastPose{};
    std::vector<Vec2> positionHistory;
    std::vector<Vec2> velocityHistory;
    std::array<float, REID_DIM> reidFeature{};

    Vec2 predictedCenter;
    BBox predictedBbox;

    bool isConfirmed() const { return hitCount >= 3; }
};

class PoseTracker {
public:
    explicit PoseTracker(const PoseTrackerConfig& config = {});
    ~PoseTracker();

    PoseTracker(const PoseTracker&) = delete;
    PoseTracker& operator=(const PoseTracker&) = delete;
    PoseTracker(PoseTracker&&) noexcept;
    PoseTracker& operator=(PoseTracker&&) noexcept;

    void update(const std::vector<Detection>& detections,
                const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& poses);

    std::vector<Tracklet> getConfirmedTracklets() const;
    std::vector<Tracklet> getAllTracklets() const;

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
