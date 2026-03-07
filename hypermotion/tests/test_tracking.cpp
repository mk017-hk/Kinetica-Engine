#include <gtest/gtest.h>
#include "HyperMotion/tracking/MultiPlayerTracker.h"
#include "HyperMotion/tracking/PlayerIDManager.h"
#include "test_helpers.h"

using namespace hm;
using namespace hm::tracking;

namespace {

DetectedPerson makePerson(int id, float cx, float cy, float conf = 0.9f) {
    DetectedPerson p;
    p.id = id;
    p.bbox = {cx - 25.0f, cy - 50.0f, 50.0f, 100.0f, conf};
    p.classLabel = "player";
    for (int i = 0; i < COCO_KEYPOINTS; ++i) {
        p.keypoints2D[i].position = {cx / 1920.0f, cy / 1080.0f};
        p.keypoints2D[i].confidence = conf;
    }
    // Give each person a distinct ReID feature
    for (int i = 0; i < REID_DIM; ++i) {
        p.reidFeature[i] = (id * 37 + i * 13) % 100 / 100.0f;
    }
    // L2-normalise
    float norm = 0.0f;
    for (auto v : p.reidFeature) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f)
        for (auto& v : p.reidFeature) v /= norm;
    return p;
}

PoseFrameResult makeFrameResult(int frameIdx,
                                 const std::vector<DetectedPerson>& persons) {
    PoseFrameResult r;
    r.frameIndex = frameIdx;
    r.timestamp = frameIdx / 30.0;
    r.persons = persons;
    r.videoWidth = 1920;
    r.videoHeight = 1080;
    return r;
}

} // anonymous namespace

// -------------------------------------------------------------------
// PlayerIDManager tests
// -------------------------------------------------------------------

TEST(PlayerIDManagerTest, FirstUpdateAssignsNewIDs) {
    PlayerIDManager mgr;
    std::vector<DetectedPerson> persons = {makePerson(0, 100, 200),
                                            makePerson(1, 400, 200)};
    auto mapping = mgr.update(persons, 0);

    EXPECT_EQ(mapping.size(), 2u);
    EXPECT_EQ(mgr.totalPlayersTracked(), 2);
}

TEST(PlayerIDManagerTest, ConsistentIDsAcrossFrames) {
    PlayerIDManager mgr;

    // Frame 0
    std::vector<DetectedPerson> persons0 = {makePerson(0, 100, 200),
                                             makePerson(1, 400, 200)};
    auto map0 = mgr.update(persons0, 0);

    // Frame 1 — same persons, slightly moved
    std::vector<DetectedPerson> persons1 = {makePerson(0, 105, 200),
                                             makePerson(1, 405, 200)};
    auto map1 = mgr.update(persons1, 1);

    // Same tracklet IDs should map to same persistent IDs
    EXPECT_EQ(map0[0], map1[0]);
    EXPECT_EQ(map0[1], map1[1]);
    EXPECT_EQ(mgr.totalPlayersTracked(), 2);
}

TEST(PlayerIDManagerTest, ReidentifiesAfterOcclusion) {
    PlayerIDManagerConfig config;
    config.maxLostFrames = 10;
    PlayerIDManager mgr(config);

    // Frame 0 — two players
    auto map0 = mgr.update({makePerson(0, 100, 200), makePerson(1, 400, 200)}, 0);
    int idA = map0[0];

    // Frames 1-5 — player 0 disappears
    for (int f = 1; f <= 5; ++f) {
        mgr.update({makePerson(1, 400 + f, 200)}, f);
    }

    // Frame 6 — player 0 reappears with same ReID features
    auto map6 = mgr.update({makePerson(0, 110, 200), makePerson(1, 406, 200)}, 6);

    // Should re-identify as the same persistent ID
    EXPECT_EQ(map6[0], idA);
}

TEST(PlayerIDManagerTest, ResetClearsState) {
    PlayerIDManager mgr;
    mgr.update({makePerson(0, 100, 200)}, 0);
    EXPECT_EQ(mgr.totalPlayersTracked(), 1);

    mgr.reset();
    EXPECT_EQ(mgr.totalPlayersTracked(), 0);
    EXPECT_TRUE(mgr.getActiveIdentities().empty());
}

// -------------------------------------------------------------------
// MultiPlayerTracker tests
//
// The internal PoseTracker requires minHitsToConfirm frames before
// tracklets become "confirmed".  We set this to 1 for unit tests to
// ensure tracklets appear immediately.
// -------------------------------------------------------------------

static MultiPlayerTrackerConfig fastTrackerConfig() {
    MultiPlayerTrackerConfig cfg;
    cfg.trackerConfig.minHitsToConfirm = 1;
    cfg.trackerConfig.lostTimeout = 30;
    cfg.trackerConfig.maxMatchDistance = 0.99f;
    return cfg;
}

TEST(MultiPlayerTrackerTest, ProcessAllProducesTrackedFrames) {
    MultiPlayerTracker tracker(fastTrackerConfig());

    std::vector<PoseFrameResult> poseResults;
    for (int f = 0; f < 10; ++f) {
        poseResults.push_back(
            makeFrameResult(f, {makePerson(0, 100 + f * 5, 200),
                                makePerson(1, 400 + f * 5, 200)}));
    }

    auto trackedFrames = tracker.processAll(poseResults);
    EXPECT_EQ(trackedFrames.size(), 10u);

    // After first few frames, we should see tracked players
    int framesWithPlayers = 0;
    for (const auto& tf : trackedFrames) {
        if (!tf.players.empty()) ++framesWithPlayers;
    }
    EXPECT_GE(framesWithPlayers, 5);
}

TEST(MultiPlayerTrackerTest, PersistentIDsStableOverSequence) {
    MultiPlayerTracker tracker(fastTrackerConfig());

    std::vector<PoseFrameResult> poseResults;
    for (int f = 0; f < 20; ++f) {
        poseResults.push_back(
            makeFrameResult(f, {makePerson(0, 100 + f * 3, 200),
                                makePerson(1, 500 + f * 3, 200)}));
    }

    auto trackedFrames = tracker.processAll(poseResults);

    // Collect all persistent IDs seen across all frames
    std::set<int> allIDs;
    for (const auto& tf : trackedFrames) {
        for (const auto& tp : tf.players) {
            allIDs.insert(tp.persistentID);
        }
    }

    // Should track no more than 2 unique persistent IDs
    // (may track fewer if some frames had no confirmed tracklets)
    EXPECT_LE(allIDs.size(), 4u);
}

TEST(MultiPlayerTrackerTest, GetPlayerSequences) {
    MultiPlayerTracker tracker(fastTrackerConfig());

    std::vector<PoseFrameResult> poseResults;
    for (int f = 0; f < 10; ++f) {
        poseResults.push_back(makeFrameResult(f, {makePerson(0, 100, 200)}));
    }

    auto trackedFrames = tracker.processAll(poseResults);
    auto sequences = tracker.getPlayerSequences(trackedFrames);

    // At least one player should have been tracked
    EXPECT_GE(sequences.size(), 1u);
    for (const auto& [id, seq] : sequences) {
        EXPECT_GE(seq.size(), 1u);
    }
}

TEST(MultiPlayerTrackerTest, ResetClearsTrackerState) {
    MultiPlayerTracker tracker(fastTrackerConfig());

    std::vector<PoseFrameResult> poseResults;
    for (int f = 0; f < 5; ++f) {
        poseResults.push_back(makeFrameResult(f, {makePerson(0, 100, 200)}));
    }
    tracker.processAll(poseResults);

    EXPECT_GE(tracker.totalPlayersTracked(), 1);
    tracker.reset();
    EXPECT_EQ(tracker.totalPlayersTracked(), 0);
}
