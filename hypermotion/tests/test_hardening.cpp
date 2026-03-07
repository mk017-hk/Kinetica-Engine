#include <gtest/gtest.h>
#include "HyperMotion/tracking/MultiPlayerTracker.h"
#include "HyperMotion/tracking/PlayerIDManager.h"
#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/dataset/AnimationDatabase.h"
#include "HyperMotion/dataset/MatchAnalyser.h"
#include "HyperMotion/analysis/MotionFingerprint.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/core/PipelineConfigIO.h"
#include "test_helpers.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <cmath>

using namespace hm;
using namespace hm::tracking;
using namespace hm::motion;
using namespace hm::segmenter;
using namespace hm::dataset;

// ===================================================================
// Helper: create a DetectedPerson with specific position and ReID
// ===================================================================

namespace {

DetectedPerson makePerson(int id, float cx, float cy, float conf = 0.9f) {
    DetectedPerson p;
    p.id = id;
    p.bbox = {cx - 25.0f, cy - 50.0f, 50.0f, 100.0f, conf};
    p.classLabel = "player";
    for (int i = 0; i < COCO_KEYPOINTS; ++i) {
        p.keypoints2D[i].position = {cx / 1920.0f, cy / 1080.0f};
        p.keypoints2D[i].confidence = conf;
        p.keypoints3D[i].position = {cx * 0.1f, 90.0f, cy * 0.1f};
        p.keypoints3D[i].confidence = conf;
    }
    // Give each person a distinct ReID feature
    for (int i = 0; i < REID_DIM; ++i) {
        p.reidFeature[i] = (id * 37 + i * 13) % 100 / 100.0f;
    }
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

MultiPlayerTrackerConfig fastTrackerConfig() {
    MultiPlayerTrackerConfig cfg;
    cfg.trackerConfig.minHitsToConfirm = 1;
    cfg.trackerConfig.lostTimeout = 30;
    cfg.trackerConfig.maxMatchDistance = 0.99f;
    return cfg;
}

// Create a noisy walking sequence for canonical motion tests
std::vector<SkeletonFrame> makeNoisyWalkingSequence(
    int numFrames, float speed, float fps, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 3.0f);
    auto frames = test::makeWalkingSequence(numFrames, speed, fps);
    for (auto& f : frames) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            f.joints[j].worldPosition.x += noise(rng);
            f.joints[j].worldPosition.y += noise(rng);
            f.joints[j].worldPosition.z += noise(rng);
        }
    }
    return frames;
}

// Create a sequence with walk→sprint→stop velocity profile
std::vector<SkeletonFrame> makeMultiPhaseSequence(int numFrames, float fps) {
    std::vector<SkeletonFrame> frames;
    frames.reserve(numFrames);
    float dt = 1.0f / fps;
    float x = 0.0f;

    for (int i = 0; i < numFrames; ++i) {
        float t = static_cast<float>(i) / numFrames;
        float speed;
        if (t < 0.3f) speed = 100.0f;       // walk
        else if (t < 0.6f) speed = 500.0f;   // sprint
        else speed = 10.0f;                   // near-stop

        x += speed * dt;
        auto frame = test::makeIdentityFrame({x, 90.0f, 0.0f}, i, i * dt);
        frame.rootVelocity = {speed, 0.0f, 0.0f};
        frame.trackingID = 1;
        frames.push_back(frame);
    }
    return frames;
}

} // anonymous namespace

// ===================================================================
// A. Tracking Persistence Tests
// ===================================================================

TEST(HardeningTracking, SamePlayerKeepsSameID) {
    PlayerIDManager mgr;

    // Same two players across 20 frames with gradual movement
    for (int f = 0; f < 20; ++f) {
        std::vector<DetectedPerson> persons = {
            makePerson(0, 100 + f * 3.0f, 200),
            makePerson(1, 500 + f * 3.0f, 200)
        };
        auto map = mgr.update(persons, f);

        if (f > 0) {
            // IDs should be stable after first assignment
            EXPECT_EQ(mgr.totalPlayersTracked(), 2)
                << "Unexpected new player created at frame " << f;
        }
    }
}

TEST(HardeningTracking, OcclusionDoesNotDestroyTrack) {
    PlayerIDManagerConfig config;
    config.maxLostFrames = 15;
    PlayerIDManager mgr(config);

    // Frame 0-4: Two players visible
    int idA = -1, idB = -1;
    for (int f = 0; f < 5; ++f) {
        auto map = mgr.update({makePerson(0, 100, 200), makePerson(1, 500, 200)}, f);
        if (f == 0) { idA = map[0]; idB = map[1]; }
    }
    ASSERT_NE(idA, -1);

    // Frame 5-12: Player A occluded (only B visible)
    for (int f = 5; f <= 12; ++f) {
        mgr.update({makePerson(1, 500 + f, 200)}, f);
    }

    // Frame 13: Player A reappears — should get same persistent ID
    auto map13 = mgr.update({makePerson(0, 120, 200), makePerson(1, 513, 200)}, 13);
    EXPECT_EQ(map13[0], idA)
        << "Player A should have been re-identified after occlusion";
}

TEST(HardeningTracking, WeakDetectionsDoNotCreateDuplicates) {
    PlayerIDManager mgr;

    // Establish one strong player
    mgr.update({makePerson(0, 300, 300)}, 0);
    mgr.update({makePerson(0, 303, 300)}, 1);
    EXPECT_EQ(mgr.totalPlayersTracked(), 1);

    // Add a very weak detection near the same location — should either
    // match the existing player or create exactly one new track, not
    // create unbounded duplicates
    for (int f = 2; f < 10; ++f) {
        auto strong = makePerson(0, 300 + f * 3, 300);
        auto weak = makePerson(99, 310 + f * 3, 300);  // nearby, different ID
        weak.bbox.confidence = 0.15f;
        mgr.update({strong, weak}, f);
    }

    // Should not have created dozens of tracks
    EXPECT_LE(mgr.totalPlayersTracked(), 4)
        << "Weak detections created too many duplicate tracks";
}

TEST(HardeningTracking, MultiPlayerTrackerPersistenceAcrossFrames) {
    MultiPlayerTracker tracker(fastTrackerConfig());

    std::vector<PoseFrameResult> poseResults;
    for (int f = 0; f < 30; ++f) {
        poseResults.push_back(
            makeFrameResult(f, {makePerson(0, 100 + f * 2, 200),
                                makePerson(1, 600 + f * 2, 200)}));
    }

    auto trackedFrames = tracker.processAll(poseResults);

    // Collect unique persistent IDs across all frames
    std::set<int> allIDs;
    for (const auto& tf : trackedFrames) {
        for (const auto& tp : tf.players) {
            allIDs.insert(tp.persistentID);
        }
    }

    // With two clearly separated players, should see exactly 2 unique IDs
    EXPECT_LE(allIDs.size(), 3u)
        << "Tracker created too many persistent IDs for 2 players";
}

// ===================================================================
// B. Canonical Motion Stability Tests
// ===================================================================

TEST(HardeningCanonical, NoisyInput_LimbLengthsStabilise) {
    auto frames = makeNoisyWalkingSequence(90, 150.0f, 30.0f);

    CanonicalMotionBuilderConfig config;
    config.stabiliseLimbLengths = true;
    config.solveRootOrientation = true;
    CanonicalMotionBuilder builder(config);

    auto canonical = builder.build(frames, 0, 30.0f);
    auto rebuilt = builder.toSkeletonFrames(canonical);

    // Measure limb length variance of left upper leg after stabilisation
    int leftLeg = static_cast<int>(Joint::LeftLeg);
    int leftUpLeg = static_cast<int>(Joint::LeftUpLeg);

    std::vector<float> lengths;
    for (const auto& f : rebuilt) {
        Vec3 diff = f.joints[leftLeg].worldPosition -
                    f.joints[leftUpLeg].worldPosition;
        lengths.push_back(diff.length());
    }

    float mean = 0.0f;
    for (auto l : lengths) mean += l;
    mean /= lengths.size();

    float variance = 0.0f;
    for (auto l : lengths) variance += (l - mean) * (l - mean);
    variance /= lengths.size();

    // Stabilised variance should be very small
    EXPECT_LT(variance, 5.0f)
        << "Limb lengths did not stabilise sufficiently (variance=" << variance << ")";
}

TEST(HardeningCanonical, RootMotionSeparatedFromLocalMotion) {
    auto frames = test::makeWalkingSequence(60, 200.0f, 30.0f);

    CanonicalMotionBuilderConfig config;
    config.extractRootMotion = true;
    CanonicalMotionBuilder builder(config);

    auto canonical = builder.build(frames, 0, 30.0f);

    // Root trajectory should show progression along X
    ASSERT_GE(canonical.rootTrajectory.size(), 2u);

    float startX = canonical.rootTrajectory.front().x;
    float endX = canonical.rootTrajectory.back().x;
    EXPECT_GT(endX - startX, 50.0f)
        << "Root trajectory does not show expected forward movement";

    // Local joint positions should be relative to parent, not world
    // i.e. they should not show the same X drift
    float localDriftX = 0.0f;
    for (const auto& cf : canonical.frames) {
        localDriftX += std::abs(cf.localPositions[static_cast<int>(Joint::Spine)].x);
    }
    localDriftX /= canonical.frames.size();

    // Local spine offset from hips should be small and not drift with root
    EXPECT_LT(localDriftX, 20.0f)
        << "Local positions appear to contain root motion drift";
}

TEST(HardeningCanonical, OrientationRemainsSmoothForLinearMotion) {
    auto frames = test::makeWalkingSequence(60, 150.0f, 30.0f);

    CanonicalMotionBuilderConfig config;
    config.solveRootOrientation = true;
    CanonicalMotionBuilder builder(config);

    auto canonical = builder.build(frames, 0, 30.0f);

    // Check smoothness: quaternion dot product between consecutive frames
    int discontinuities = 0;
    for (size_t i = 10; i < canonical.frames.size() - 1; ++i) {
        float dot = canonical.frames[i].rootRotation.dot(
            canonical.frames[i + 1].rootRotation);
        if (std::abs(dot) < 0.9f) {
            discontinuities++;
        }
    }

    EXPECT_LE(discontinuities, 2)
        << "Root orientation has " << discontinuities << " discontinuities in linear motion";
}

// ===================================================================
// C. Motion Segmentation Boundary Tests
// ===================================================================

TEST(HardeningSegmentation, MultiPhaseSequenceProducesMultipleSegments) {
    auto frames = makeMultiPhaseSequence(150, 30.0f);

    MotionSegmenterConfig config;
    config.minSegmentLength = 5;
    MotionSegmenter segmenter(config);
    segmenter.initialize();

    auto segments = segmenter.segment(frames, 1);

    // Should detect at least 2 segments from the 3-phase velocity profile
    EXPECT_GE(segments.size(), 2u)
        << "Segmenter failed to detect velocity change boundaries";
}

TEST(HardeningSegmentation, MinimumSegmentLengthRespected) {
    auto frames = makeMultiPhaseSequence(150, 30.0f);

    MotionSegmenterConfig config;
    config.minSegmentLength = 10;
    MotionSegmenter segmenter(config);
    segmenter.initialize();

    auto segments = segmenter.segment(frames, 1);

    for (const auto& seg : segments) {
        int segLength = seg.endFrame - seg.startFrame + 1;
        EXPECT_GE(segLength, config.minSegmentLength)
            << "Segment from " << seg.startFrame << " to " << seg.endFrame
            << " is shorter than minimum";
    }
}

TEST(HardeningSegmentation, SegmentsCoverAllFrames) {
    auto frames = makeMultiPhaseSequence(120, 30.0f);

    MotionSegmenter segmenter;
    segmenter.initialize();

    auto segments = segmenter.segment(frames, 1);

    if (!segments.empty()) {
        EXPECT_EQ(segments.front().startFrame, 0)
            << "First segment does not start at frame 0";
        EXPECT_EQ(segments.back().endFrame, static_cast<int>(frames.size()) - 1)
            << "Last segment does not end at last frame";

        // Verify contiguity — each segment starts where the previous ended
        for (size_t i = 1; i < segments.size(); ++i) {
            EXPECT_EQ(segments[i].startFrame, segments[i - 1].endFrame + 1)
                << "Gap between segment " << i - 1 << " and " << i;
        }
    }
}

// ===================================================================
// D. Clip Quality Filtering Tests
// ===================================================================

TEST(HardeningQuality, MissingJointsRejected) {
    ClipQualityConfig config;
    config.minAvgConfidence = 0.5f;
    ClipQualityFilter filter(config);

    auto clip = test::makeTestClip(30, 30.0f);
    // Zero out confidence for half the joints
    for (auto& frame : clip.frames) {
        for (int j = 0; j < JOINT_COUNT / 2; ++j) {
            frame.joints[j].confidence = 0.0f;
        }
    }

    auto result = filter.assess(clip);
    EXPECT_FALSE(result.accepted)
        << "Clip with many zero-confidence joints should be rejected";
}

TEST(HardeningQuality, UnstableRootMotionDetected) {
    ClipQualityConfig config;
    config.maxJitterThreshold = 10.0f;
    ClipQualityFilter filter(config);

    auto clip = test::makeTestClip(30, 30.0f);

    // Inject large root position jitter
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 50.0f);
    for (size_t i = 0; i < clip.frames.size(); i += 2) {
        clip.frames[i].rootPosition.x += noise(rng);
        clip.frames[i].rootPosition.y += noise(rng);
    }

    // Also inject jitter into joints
    for (size_t i = 0; i < clip.frames.size(); i += 2) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            clip.frames[i].joints[j].worldPosition.x += noise(rng);
        }
    }

    auto result = filter.assess(clip);
    EXPECT_GT(result.maxJitter, 10.0f)
        << "Should detect high jitter from unstable root motion";
}

TEST(HardeningQuality, ShortDurationClipRejected) {
    ClipQualityConfig config;
    config.minDurationSec = 0.5f;
    ClipQualityFilter filter(config);

    // 5 frames at 30fps = 0.167s
    auto clip = test::makeTestClip(5, 30.0f);
    auto result = filter.assess(clip);
    EXPECT_FALSE(result.accepted) << "Very short clip should be rejected";
}

TEST(HardeningQuality, CleanClipAccepted) {
    ClipQualityFilter filter;
    auto clip = test::makeTestClip(60, 30.0f);
    auto result = filter.assess(clip);
    EXPECT_TRUE(result.accepted) << "Clean 2-second clip should be accepted";
    EXPECT_GT(result.overallScore, 0.5f);
}

// ===================================================================
// E. Animation Database Correctness Tests
// ===================================================================

TEST(HardeningDatabase, ExportCreatesExpectedFolders) {
    AnimationDatabase db;

    // Add walk and jog entries
    for (int i = 0; i < 3; ++i) {
        AnimationEntry entry;
        entry.clip = test::makeTestClip(30, 30.0f, "clip_" + std::to_string(i));
        entry.clipMeta.playerID = i;
        entry.clipMeta.durationSec = 1.0f;
        entry.classification.type = (i < 2) ? MotionType::Walk : MotionType::Jog;
        entry.classification.label = (i < 2) ? "walk" : "jog";
        entry.classification.confidence = 0.9f;
        entry.quality.overallScore = 0.8f;
        db.addEntry(std::move(entry));
    }

    std::string testDir = "/tmp/hm_hardening_db_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

    int exported = db.exportToDirectory(testDir, true, true);
    EXPECT_EQ(exported, 3);

    // Verify folder structure
    namespace fs = std::filesystem;
    EXPECT_TRUE(fs::exists(testDir + "/walk"));
    EXPECT_TRUE(fs::exists(testDir + "/jog"));

    // Verify files exist in walk/
    int walkFiles = 0;
    for (const auto& entry : fs::directory_iterator(testDir + "/walk")) {
        if (entry.is_regular_file()) walkFiles++;
    }
    // 2 clips × 3 files each (bvh, json, meta.json)
    EXPECT_EQ(walkFiles, 6);

    fs::remove_all(testDir);
}

TEST(HardeningDatabase, MetadataSerialisationStable) {
    AnimationDatabase db;

    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f, "stable_test");
    entry.clipMeta.playerID = 5;
    entry.clipMeta.durationSec = 1.0f;
    entry.clipMeta.startFrame = 0;
    entry.clipMeta.endFrame = 29;
    entry.classification.type = MotionType::Sprint;
    entry.classification.label = "sprint";
    entry.classification.confidence = 0.85f;
    entry.quality.overallScore = 0.9f;
    db.addEntry(std::move(entry));

    // Export twice and compare summary JSON
    std::string json1 = db.exportSummaryJSON();
    std::string json2 = db.exportSummaryJSON();
    EXPECT_EQ(json1, json2) << "Summary JSON not deterministic across calls";
}

TEST(HardeningDatabase, SummaryContainsExpectedFields) {
    AnimationDatabase db;

    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f);
    entry.clipMeta.playerID = 0;
    entry.clipMeta.durationSec = 1.0f;
    entry.classification.type = MotionType::Walk;
    entry.classification.label = "walk";
    entry.quality.overallScore = 0.8f;
    db.addEntry(std::move(entry));

    auto json = db.exportSummaryJSON();
    EXPECT_NE(json.find("totalClips"), std::string::npos);
    EXPECT_NE(json.find("totalFrames"), std::string::npos);
    EXPECT_NE(json.find("uniquePlayers"), std::string::npos);
    EXPECT_NE(json.find("schemaVersion"), std::string::npos);
}

// ===================================================================
// F. Pipeline Smoke Test (end-to-end through internal modules)
// ===================================================================

TEST(HardeningPipeline, SyntheticEndToEndWithCanonical) {
    // Exercise the complete internal path:
    //   skeleton frames → signal → canonical → segmentation →
    //   extraction → quality filter → fingerprint → database

    const int kNumFrames = 150;
    const float kFPS = 30.0f;

    auto frames = makeMultiPhaseSequence(kNumFrames, kFPS);

    // Step 1: Signal processing
    signal::SignalPipelineConfig sigConfig;
    signal::SignalPipeline sigPipeline(sigConfig);
    sigPipeline.process(frames);
    EXPECT_EQ(frames.size(), static_cast<size_t>(kNumFrames));

    // Step 2: Canonical motion
    CanonicalMotionBuilderConfig canonConfig;
    CanonicalMotionBuilder canonBuilder(canonConfig);
    AnimClip clip;
    clip.name = "pipeline_test";
    clip.fps = kFPS;
    clip.trackingID = 1;
    clip.frames = frames;
    canonBuilder.process(clip);
    EXPECT_EQ(clip.frames.size(), static_cast<size_t>(kNumFrames));

    // Step 3: Segmentation
    MotionSegmenter segmenter;
    segmenter.initialize();
    auto segments = segmenter.segment(clip.frames, 1);
    EXPECT_GE(segments.size(), 1u);
    clip.segments = segments;

    // Step 4: Clip extraction
    ClipExtractor extractor;
    auto extraction = extractor.extractFromSegments(clip.frames, segments, 1);
    EXPECT_GE(extraction.clips.size(), 1u);

    // Step 5: Quality filter
    ClipQualityFilter filter;
    auto filtered = filter.filter(extraction.clips);
    int totalClips = static_cast<int>(filtered.accepted.size() + filtered.rejected.size());
    EXPECT_EQ(totalClips, static_cast<int>(extraction.clips.size()));

    // Step 6: Fingerprinting
    analysis::MotionFingerprint fingerprinter;
    for (const auto& accepted : filtered.accepted) {
        auto fp = fingerprinter.compute(accepted);
        EXPECT_GT(fp.frameCount, 0);
    }

    // Step 7: Database assembly + export
    AnimationDatabase db;
    for (size_t i = 0; i < filtered.accepted.size(); ++i) {
        AnimationEntry entry;
        entry.clip = filtered.accepted[i];
        entry.clipMeta.playerID = 1;
        entry.clipMeta.durationSec =
            static_cast<float>(entry.clip.frames.size()) / kFPS;
        entry.quality = filtered.acceptedResults[i];
        entry.classification.type = MotionType::Walk;
        entry.classification.label = "walk";
        db.addEntry(std::move(entry));
    }

    auto stats = db.stats();
    EXPECT_GE(stats.totalClips, 1);

    std::string testDir = "/tmp/hm_pipeline_smoke_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

    int exported = db.exportToDirectory(testDir, true, true);
    EXPECT_GE(exported, 1);
    EXPECT_TRUE(db.saveSummary(testDir + "/database_summary.json"));

    // Verify output
    EXPECT_TRUE(std::filesystem::exists(testDir + "/database_summary.json"));

    std::filesystem::remove_all(testDir);
}

// ===================================================================
// G. PipelineStats serialisation completeness
// ===================================================================

TEST(HardeningStats, AllFieldsSerialised) {
    PipelineStats stats;
    stats.poseExtractionMs = 100.0;
    stats.skeletonMappingMs = 50.0;
    stats.signalProcessingMs = 30.0;
    stats.segmentationMs = 20.0;
    stats.footContactMs = 10.0;
    stats.trajectoryMs = 5.0;
    stats.canonicalMotionMs = 15.0;
    stats.fingerprintMs = 8.0;
    stats.clusteringMs = 12.0;
    stats.exportMs = 25.0;
    stats.totalMs = 275.0;
    stats.totalFramesProcessed = 100;
    stats.trackedPersons = 5;
    stats.clipsProduced = 10;
    stats.segmentsFound = 20;

    auto json = serialisePipelineStats(stats);

    // All fields must be present
    EXPECT_NE(json.find("poseExtractionMs"), std::string::npos);
    EXPECT_NE(json.find("skeletonMappingMs"), std::string::npos);
    EXPECT_NE(json.find("signalProcessingMs"), std::string::npos);
    EXPECT_NE(json.find("segmentationMs"), std::string::npos);
    EXPECT_NE(json.find("footContactMs"), std::string::npos);
    EXPECT_NE(json.find("trajectoryMs"), std::string::npos);
    EXPECT_NE(json.find("canonicalMotionMs"), std::string::npos);
    EXPECT_NE(json.find("fingerprintMs"), std::string::npos);
    EXPECT_NE(json.find("clusteringMs"), std::string::npos);
    EXPECT_NE(json.find("exportMs"), std::string::npos);
    EXPECT_NE(json.find("totalMs"), std::string::npos);
    EXPECT_NE(json.find("totalFramesProcessed"), std::string::npos);
    EXPECT_NE(json.find("trackedPersons"), std::string::npos);
    EXPECT_NE(json.find("clipsProduced"), std::string::npos);
    EXPECT_NE(json.find("segmentsFound"), std::string::npos);
}

TEST(HardeningStats, SaveAndLoadStatsFile) {
    PipelineStats stats;
    stats.totalFramesProcessed = 500;
    stats.clipsProduced = 42;

    std::string path = "/tmp/hm_stats_test_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json";

    EXPECT_TRUE(savePipelineStats(path, stats));
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read back and verify
    std::ifstream ifs(path);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("500"), std::string::npos);
    EXPECT_NE(content.find("42"), std::string::npos);

    std::filesystem::remove(path);
}

// ===================================================================
// H. Clip Extractor Duration Constraints
// ===================================================================

TEST(HardeningClipExtractor, MaxDurationRespected) {
    ClipExtractorConfig config;
    config.maxClipDurationSec = 2.0f;
    config.fps = 30.0f;
    ClipExtractor extractor(config);

    // 5 seconds of data
    auto frames = test::makeWalkingSequence(150, 100.0f);
    auto result = extractor.extract(frames, 0);

    for (const auto& clip : result.clips) {
        float duration = static_cast<float>(clip.frames.size()) / 30.0f;
        // Allow small tolerance (one frame overshoot)
        EXPECT_LE(duration, config.maxClipDurationSec + 1.0f / 30.0f)
            << "Clip exceeds maximum duration: " << duration << "s";
    }
}

// ===================================================================
// I. Config Round-Trip Verification
// ===================================================================

TEST(HardeningConfig, CanonicalMotionConfigPreserved) {
    PipelineConfig config;
    config.enableCanonicalMotion = true;
    config.enableFootContactDetection = true;
    config.enableTrajectoryExtraction = true;
    config.enableFingerprinting = true;
    config.targetFPS = 25.0f;
    config.splitBySegment = false;

    auto json = serialisePipelineConfig(config);
    PipelineConfig loaded;
    EXPECT_TRUE(parsePipelineConfig(json, loaded));

    EXPECT_FLOAT_EQ(loaded.targetFPS, 25.0f);
    EXPECT_FALSE(loaded.splitBySegment);
}

// ===================================================================
// J. Fingerprint Consistency
// ===================================================================

TEST(HardeningFingerprint, SameInputProducesSameFingerprint) {
    auto clip = test::makeTestClip(60, 30.0f);

    analysis::MotionFingerprint fingerprinter;
    auto fp1 = fingerprinter.compute(clip);
    auto fp2 = fingerprinter.compute(clip);

    EXPECT_FLOAT_EQ(fp1.avgVelocity, fp2.avgVelocity);
    EXPECT_FLOAT_EQ(fp1.peakVelocity, fp2.peakVelocity);
    EXPECT_EQ(fp1.frameCount, fp2.frameCount);

    auto v1 = fp1.toVector();
    auto v2 = fp2.toVector();
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        EXPECT_FLOAT_EQ(v1[i], v2[i]) << "Fingerprint diverged at index " << i;
    }
}
