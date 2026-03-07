#include <gtest/gtest.h>
#include "HyperMotion/analysis/MotionFingerprint.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;
using namespace hm::analysis;

// -------------------------------------------------------------------
// MotionFingerprint tests
// -------------------------------------------------------------------

TEST(MotionFingerprintTest, ComputeFromClip_BasicFields) {
    MotionFingerprint fp;
    auto clip = test::makeTestClip(30, 30.0f);

    auto features = fp.compute(clip);

    EXPECT_GT(features.avgVelocity, 0.0f);
    EXPECT_GT(features.peakVelocity, 0.0f);
    EXPECT_GE(features.peakVelocity, features.avgVelocity);
    EXPECT_EQ(features.frameCount, 30);
    EXPECT_NEAR(features.clipDurationSec, 1.0f, 0.1f);
}

TEST(MotionFingerprintTest, ComputeFromFrames) {
    MotionFingerprint fp;
    auto frames = test::makeWalkingSequence(60, 200.0f);

    auto features = fp.computeFromFrames(frames, 30.0f);

    EXPECT_GT(features.avgVelocity, 50.0f);
    EXPECT_EQ(features.frameCount, 60);
}

TEST(MotionFingerprintTest, ToVector_CorrectDimension) {
    MotionFingerprint fp;
    auto clip = test::makeTestClip(30, 30.0f);
    auto features = fp.compute(clip);

    auto vec = features.toVector();
    EXPECT_EQ(vec.size(), static_cast<size_t>(FingerprintFeatures::DIM));
    // Should be 18 dimensions
    EXPECT_EQ(FingerprintFeatures::DIM, 18);
}

TEST(MotionFingerprintTest, DistanceTo_SameClipIsZero) {
    MotionFingerprint fp;
    auto clip = test::makeTestClip(30, 30.0f);
    auto features = fp.compute(clip);

    float dist = features.distanceTo(features);
    EXPECT_NEAR(dist, 0.0f, 1e-5f);
}

TEST(MotionFingerprintTest, DistanceTo_DifferentClips) {
    MotionFingerprint fp;
    auto slowClip = test::makeTestClip(30, 30.0f);
    auto fastFrames = test::makeWalkingSequence(30, 400.0f);
    AnimClip fastClip;
    fastClip.name = "fast";
    fastClip.fps = 30.0f;
    fastClip.frames = fastFrames;

    auto slowFP = fp.compute(slowClip);
    auto fastFP = fp.compute(fastClip);

    float dist = slowFP.distanceTo(fastFP);
    EXPECT_GT(dist, 0.0f);
}

TEST(MotionFingerprintTest, ComputeBatch) {
    MotionFingerprint fp;
    std::vector<AnimClip> clips;
    clips.push_back(test::makeTestClip(30, 30.0f, "a"));
    clips.push_back(test::makeTestClip(60, 30.0f, "b"));
    clips.push_back(test::makeTestClip(45, 30.0f, "c"));

    auto batch = fp.computeBatch(clips);
    EXPECT_EQ(batch.size(), 3u);
}

TEST(MotionFingerprintTest, FindSimilar_ReturnsClosestFirst) {
    MotionFingerprint fp;

    // Create a database of clips with varying speeds
    std::vector<AnimClip> clips;
    std::vector<float> speeds = {50, 100, 150, 200, 300, 400};
    for (float s : speeds) {
        AnimClip c;
        c.fps = 30.0f;
        c.frames = test::makeWalkingSequence(30, s);
        c.name = "speed_" + std::to_string(static_cast<int>(s));
        clips.push_back(c);
    }

    auto database = fp.computeBatch(clips);

    // Query with speed=100 — closest should be index 1
    auto query = database[1]; // speed=100
    auto results = fp.findSimilar(query, database, 3);

    ASSERT_GE(results.size(), 1u);
    EXPECT_EQ(results[0].clipIndex, 1);  // exact match
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-5f);
}

TEST(MotionFingerprintTest, FingerprintFeatures_Symmetry) {
    MotionFingerprint fp;
    auto clip1 = test::makeTestClip(30, 30.0f);
    auto clip2 = test::makeTestClip(60, 30.0f);

    auto f1 = fp.compute(clip1);
    auto f2 = fp.compute(clip2);

    EXPECT_FLOAT_EQ(f1.distanceTo(f2), f2.distanceTo(f1));
}
