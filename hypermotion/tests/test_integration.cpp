#include <gtest/gtest.h>
#include "HyperMotion/dataset/AnimationDatabase.h"
#include "HyperMotion/dataset/MatchAnalyser.h"
#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/analysis/MotionFingerprint.h"
#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "test_helpers.h"

#include <filesystem>
#include <fstream>

using namespace hm;
using namespace hm::dataset;

// -------------------------------------------------------------------
// AnimationDatabase tests
// -------------------------------------------------------------------

TEST(AnimationDatabaseTest, AddAndRetrieveEntries) {
    AnimationDatabase db;

    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f, "walk_clip");
    entry.clipMeta.playerID = 5;
    entry.clipMeta.durationSec = 1.0f;
    entry.classification.type = MotionType::Walk;
    entry.classification.label = "walk";
    entry.classification.confidence = 0.9f;
    entry.quality.overallScore = 0.8f;

    db.addEntry(std::move(entry));

    EXPECT_EQ(db.entries().size(), 1u);
    auto byType = db.entriesByType(MotionType::Walk);
    EXPECT_EQ(byType.size(), 1u);
    auto byPlayer = db.entriesByPlayer(5);
    EXPECT_EQ(byPlayer.size(), 1u);
}

TEST(AnimationDatabaseTest, StatsComputed) {
    AnimationDatabase db;

    for (int i = 0; i < 3; ++i) {
        AnimationEntry entry;
        entry.clip = test::makeTestClip(30, 30.0f);
        entry.clipMeta.playerID = i;
        entry.clipMeta.durationSec = 1.0f;
        entry.classification.type = (i < 2) ? MotionType::Walk : MotionType::Jog;
        entry.classification.label = (i < 2) ? "walk" : "jog";
        entry.quality.overallScore = 0.8f;
        db.addEntry(std::move(entry));
    }

    auto stats = db.stats();
    EXPECT_EQ(stats.totalClips, 3);
    EXPECT_EQ(stats.uniquePlayers, 3);
    EXPECT_EQ(stats.clipsByType[static_cast<int>(MotionType::Walk)], 2);
    EXPECT_EQ(stats.clipsByType[static_cast<int>(MotionType::Jog)], 1);
}

TEST(AnimationDatabaseTest, ExportSummaryJSON) {
    AnimationDatabase db;

    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f);
    entry.clipMeta.playerID = 0;
    entry.clipMeta.durationSec = 1.0f;
    entry.classification.type = MotionType::Walk;
    entry.classification.label = "walk";
    db.addEntry(std::move(entry));

    auto json = db.exportSummaryJSON();
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("totalClips"), std::string::npos);
    EXPECT_NE(json.find("walk"), std::string::npos);
}

TEST(AnimationDatabaseTest, ExportToDirectory) {
    AnimationDatabase db;

    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f, "test_walk");
    entry.clipMeta.playerID = 0;
    entry.clipMeta.durationSec = 1.0f;
    entry.classification.type = MotionType::Walk;
    entry.classification.label = "walk";
    entry.classification.confidence = 0.9f;
    entry.quality.overallScore = 0.8f;
    db.addEntry(std::move(entry));

    std::string testDir = "/tmp/hm_test_db_export_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

    int exported = db.exportToDirectory(testDir, true, true);
    EXPECT_EQ(exported, 1);

    // Check that files were created
    namespace fs = std::filesystem;
    EXPECT_TRUE(fs::exists(testDir + "/walk"));
    EXPECT_TRUE(fs::exists(testDir + "/walk/clip_0000.bvh"));
    EXPECT_TRUE(fs::exists(testDir + "/walk/clip_0000.json"));
    EXPECT_TRUE(fs::exists(testDir + "/walk/clip_0000.meta.json"));

    // Read and validate meta.json
    std::ifstream metaFile(testDir + "/walk/clip_0000.meta.json");
    ASSERT_TRUE(metaFile.is_open());
    std::string metaContent((std::istreambuf_iterator<char>(metaFile)),
                             std::istreambuf_iterator<char>());
    EXPECT_NE(metaContent.find("\"motionType\""), std::string::npos);
    EXPECT_NE(metaContent.find("\"qualityScore\""), std::string::npos);

    // Save summary
    EXPECT_TRUE(db.saveSummary(testDir + "/database_summary.json"));
    EXPECT_TRUE(fs::exists(testDir + "/database_summary.json"));

    // Cleanup
    fs::remove_all(testDir);
}

TEST(AnimationDatabaseTest, ClearRemovesAll) {
    AnimationDatabase db;
    AnimationEntry entry;
    entry.clip = test::makeTestClip(30, 30.0f);
    entry.classification.type = MotionType::Idle;
    db.addEntry(std::move(entry));
    EXPECT_EQ(db.entries().size(), 1u);

    db.clear();
    EXPECT_EQ(db.entries().size(), 0u);
}

// -------------------------------------------------------------------
// Integration test: full pipeline path (synthetic data)
// -------------------------------------------------------------------

TEST(PipelineIntegrationTest, SyntheticEndToEnd) {
    // This test exercises the core pipeline path:
    //   skeleton frames → signal processing → canonical motion →
    //   segmentation → clip extraction → quality filter → database

    // Step 1: Generate synthetic skeleton frames (simulating a walk→jog)
    const int kNumFrames = 120; // 4 seconds at 30fps
    const float kFPS = 30.0f;
    std::vector<SkeletonFrame> frames;
    frames.reserve(kNumFrames);

    float x = 0.0f;
    for (int i = 0; i < kNumFrames; ++i) {
        float speed = (i < 60) ? 100.0f : 300.0f; // walk then jog
        float dt = 1.0f / kFPS;
        x += speed * dt;

        auto frame = test::makeIdentityFrame({x, 90.0f, 0.0f}, i, i * dt);
        frame.rootVelocity = {speed, 0.0f, 0.0f};
        frame.trackingID = 1;
        frames.push_back(frame);
    }

    // Step 2: Signal processing
    signal::SignalPipelineConfig sigConfig;
    signal::SignalPipeline sigPipeline(sigConfig);
    sigPipeline.process(frames);

    EXPECT_EQ(frames.size(), static_cast<size_t>(kNumFrames));

    // Step 3: Canonical motion
    motion::CanonicalMotionBuilderConfig canonConfig;
    motion::CanonicalMotionBuilder canonBuilder(canonConfig);

    AnimClip clip;
    clip.name = "integration_test";
    clip.fps = kFPS;
    clip.trackingID = 1;
    clip.frames = frames;

    canonBuilder.process(clip);
    EXPECT_EQ(clip.frames.size(), static_cast<size_t>(kNumFrames));

    // Verify canonical motion preserved confidence
    for (const auto& f : clip.frames) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            EXPECT_GE(f.joints[j].confidence, 0.0f);
        }
    }

    // Step 4: Motion segmentation
    segmenter::MotionSegmenter segmenter;
    segmenter.initialize();
    auto segments = segmenter.segment(clip.frames, 1);
    EXPECT_GE(segments.size(), 1u);

    clip.segments = segments;

    // Step 5: Clip extraction from segments
    ClipExtractor extractor;
    auto extraction = extractor.extractFromSegments(clip.frames, segments, 1);
    EXPECT_GE(extraction.clips.size(), 1u);

    // Step 6: Quality filtering
    ClipQualityFilter filter;
    auto filtered = filter.filter(extraction.clips);

    int totalClips = static_cast<int>(filtered.accepted.size() + filtered.rejected.size());
    EXPECT_EQ(totalClips, static_cast<int>(extraction.clips.size()));

    // Step 7: Fingerprinting
    analysis::MotionFingerprint fingerprinter;
    for (const auto& accepted : filtered.accepted) {
        auto fp = fingerprinter.compute(accepted);
        EXPECT_GT(fp.avgVelocity, 0.0f);
        EXPECT_GT(fp.frameCount, 0);
    }

    // Step 8: Database assembly
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
        entry.classification.confidence = 0.85f;
        db.addEntry(std::move(entry));
    }

    auto stats = db.stats();
    EXPECT_GE(stats.totalClips, 1);
    EXPECT_GT(stats.totalFrames, 0);

    // Step 9: Export to temporary directory
    std::string testDir = "/tmp/hm_integration_test_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

    int exported = db.exportToDirectory(testDir, true, true);
    EXPECT_GE(exported, 1);
    EXPECT_TRUE(db.saveSummary(testDir + "/database_summary.json"));

    // Verify output structure
    EXPECT_TRUE(std::filesystem::exists(testDir + "/walk"));
    EXPECT_TRUE(std::filesystem::exists(testDir + "/database_summary.json"));

    // Cleanup
    std::filesystem::remove_all(testDir);
}

// -------------------------------------------------------------------
// Skeleton mapper confidence propagation test
// -------------------------------------------------------------------

TEST(SkeletonMapperTest, ConfidencePropagation) {
    skeleton::SkeletonMapper mapper;

    DetectedPerson person;
    person.id = 0;
    person.bbox = {100, 100, 50, 100};
    person.classLabel = "player";

    // Set varied confidence across COCO keypoints
    for (int i = 0; i < COCO_KEYPOINTS; ++i) {
        person.keypoints3D[i].position = {0, 90.0f, 0};
        person.keypoints3D[i].confidence = 0.5f + 0.03f * i;
    }
    // Set specific keypoints with known positions
    person.keypoints3D[11].position = {-10, 80, 0};  // L hip
    person.keypoints3D[12].position = {10, 80, 0};   // R hip
    person.keypoints3D[5].position = {-15, 120, 0};  // L shoulder
    person.keypoints3D[6].position = {15, 120, 0};   // R shoulder
    person.keypoints3D[0].position = {0, 140, 0};    // Nose

    auto frame = mapper.mapToSkeleton(person, 0.0, 0);

    // All joints should have non-zero confidence (propagated from COCO)
    for (int j = 0; j < JOINT_COUNT; ++j) {
        EXPECT_GT(frame.joints[j].confidence, 0.0f)
            << "Joint " << JOINT_NAMES[j] << " has zero confidence";
    }
}

// -------------------------------------------------------------------
// Pipeline split-by-segment correctness
// -------------------------------------------------------------------

TEST(PipelineSplitTest, SplitReplacesRatherThanDuplicates) {
    // Verify the fix: splitting by segment should replace clips, not append
    auto clip = test::makeTestClip(60, 30.0f, "split_test");

    // Ensure there are segments to split on
    ASSERT_GE(clip.segments.size(), 2u);

    auto split = xport::AnimClipUtils::splitBySegments(clip);

    if (!split.empty()) {
        // Split clips should NOT duplicate the entire original clip.
        // Frame counts may slightly exceed original due to inclusive
        // segment boundaries, but should never be double.
        int totalSplitFrames = 0;
        for (const auto& s : split) {
            totalSplitFrames += static_cast<int>(s.frames.size());
        }
        int origFrames = static_cast<int>(clip.frames.size());
        EXPECT_LT(totalSplitFrames, origFrames * 2)
            << "Split produced too many frames — possible duplication bug";
    }
}
