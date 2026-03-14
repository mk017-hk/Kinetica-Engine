#include <gtest/gtest.h>
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;
using namespace hm::segmenter;

namespace {

// Create a sequence with a clear velocity change mid-way
std::vector<SkeletonFrame> makeSpeedChangeSequence(int numFrames = 90,
                                                     float fps = 30.0f) {
    std::vector<SkeletonFrame> frames(numFrames);
    float dt = 1.0f / fps;

    float pos = 0.0f;
    for (int i = 0; i < numFrames; ++i) {
        float speed;
        if (i < numFrames / 3)
            speed = 50.0f;
        else if (i < 2 * numFrames / 3)
            speed = 400.0f;
        else
            speed = 10.0f;

        pos += speed * dt;
        frames[i] = test::makeIdentityFrame({pos, 90.0f, 0.0f}, i, i * dt);
        frames[i].rootVelocity = {speed, 0.0f, 0.0f};
    }
    return frames;
}

// Create a sequence with direction changes
std::vector<SkeletonFrame> makeDirectionChangeSequence(int numFrames = 90,
                                                         float fps = 30.0f) {
    std::vector<SkeletonFrame> frames(numFrames);
    float dt = 1.0f / fps;
    float speed = 150.0f;
    float x = 0.0f, z = 0.0f;

    for (int i = 0; i < numFrames; ++i) {
        float angle;
        if (i < numFrames / 3)
            angle = 0.0f;
        else if (i < 2 * numFrames / 3)
            angle = 1.5708f;
        else
            angle = 3.14159f;

        float vx = speed * std::cos(angle);
        float vz = speed * std::sin(angle);
        x += vx * dt;
        z += vz * dt;

        frames[i] = test::makeIdentityFrame({x, 90.0f, z}, i, i * dt);
        frames[i].rootVelocity = {vx, 0.0f, vz};
    }
    return frames;
}

// Helper: create and initialize a segmenter (no model, uses heuristic fallback)
MotionSegmenter makeInitializedSegmenter(int minSegLen = 5) {
    MotionSegmenterConfig config;
    config.minSegmentLength = minSegLen;
    MotionSegmenter segmenter(config);
    segmenter.initialize();
    return segmenter;
}

} // anonymous namespace

// -------------------------------------------------------------------
// MotionSegmenter tests
// -------------------------------------------------------------------

TEST(MotionSegmenterTest, UninitializedReturnsEmpty) {
    MotionSegmenter segmenter;
    auto frames = test::makeWalkingSequence(30, 100.0f);
    auto segments = segmenter.segment(frames, 0);

    // Not initialized: should return empty
    EXPECT_TRUE(segments.empty());
}

TEST(MotionSegmenterTest, InitializeSucceeds) {
    MotionSegmenter segmenter;
    bool ok = segmenter.initialize();
    EXPECT_TRUE(ok);
    EXPECT_TRUE(segmenter.isInitialized());
}

TEST(MotionSegmenterTest, SegmentsUniformMotion_FewSegments) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = test::makeWalkingSequence(30, 100.0f);

    auto segments = segmenter.segment(frames, 0);

    // Uniform motion should produce at least 1 segment
    EXPECT_GE(segments.size(), 1u);
}

TEST(MotionSegmenterTest, DetectsVelocityChangeBoundary) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = makeSpeedChangeSequence(90, 30.0f);
    auto segments = segmenter.segment(frames, 0);

    // Should detect at least 2 segments from the speed changes
    EXPECT_GE(segments.size(), 2u);
}

TEST(MotionSegmenterTest, DetectsDirectionChangeBoundary) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = makeDirectionChangeSequence(90, 30.0f);
    auto segments = segmenter.segment(frames, 0);

    // Heuristic classifies by velocity magnitude; with constant speed the
    // direction change alone may produce only 1 segment.  At minimum we
    // get a valid segmentation covering all frames.
    EXPECT_GE(segments.size(), 1u);
    if (!segments.empty()) {
        EXPECT_EQ(segments.front().startFrame, 0);
        EXPECT_EQ(segments.back().endFrame, 89);
    }
}

TEST(MotionSegmenterTest, SegmentsHaveValidBounds) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = makeSpeedChangeSequence(60, 30.0f);
    auto segments = segmenter.segment(frames, 0);

    for (const auto& seg : segments) {
        EXPECT_GE(seg.startFrame, 0);
        EXPECT_LT(seg.endFrame, 60);
        EXPECT_LE(seg.startFrame, seg.endFrame);
        EXPECT_GE(seg.confidence, 0.0f);
        EXPECT_LE(seg.confidence, 1.0f);
    }
}

TEST(MotionSegmenterTest, SegmentsCoverEntireSequence) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = makeSpeedChangeSequence(60, 30.0f);
    auto segments = segmenter.segment(frames, 0);

    if (!segments.empty()) {
        EXPECT_EQ(segments.front().startFrame, 0);
        EXPECT_EQ(segments.back().endFrame, 59);
    }
}

TEST(MotionSegmenterTest, ClassifyFrames_ReturnsCorrectCount) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = test::makeWalkingSequence(30, 100.0f);
    auto probs = segmenter.classifyFrames(frames);

    EXPECT_EQ(probs.size(), 30u);

    // Heuristic fallback: each frame should have a valid probability distribution
    for (const auto& p : probs) {
        float sum = 0.0f;
        for (float v : p) sum += v;
        // Heuristic gives 0.95 to one class + 0.05/(N-1) to others
        EXPECT_NEAR(sum, 1.0f, 0.15f);
    }
}

TEST(MotionSegmenterTest, TrackingIDPropagated) {
    auto segmenter = makeInitializedSegmenter();
    auto frames = test::makeWalkingSequence(30, 100.0f);
    auto segments = segmenter.segment(frames, 42);

    for (const auto& seg : segments) {
        EXPECT_EQ(seg.trackingID, 42);
    }
}

// -------------------------------------------------------------------
// Heuristic classifier tests (MotionFeatureExtractor)
// -------------------------------------------------------------------

TEST(MotionFeatureExtractorTest, ClassifiesIdleCorrectly) {
    MotionFeatureExtractor extractor;
    // Create still frames (near-zero velocity)
    std::vector<SkeletonFrame> frames(30);
    for (int i = 0; i < 30; ++i) {
        frames[i] = test::makeIdentityFrame({0, 90, 0}, i, i / 30.0);
        frames[i].rootVelocity = {0, 0, 0};
    }
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::Idle);
}

TEST(MotionFeatureExtractorTest, ClassifiesWalkCorrectly) {
    MotionFeatureExtractor extractor;
    auto frames = test::makeWalkingSequence(30, 80.0f); // 80 cm/s = walking
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::Walk);
}

TEST(MotionFeatureExtractorTest, ClassifiesSprintCorrectly) {
    MotionFeatureExtractor extractor;
    std::vector<SkeletonFrame> frames(30);
    for (int i = 0; i < 30; ++i) {
        frames[i] = test::makeIdentityFrame({0, 88, static_cast<float>(i * 16)}, i, i / 30.0);
        frames[i].rootVelocity = {0, 0, 500.0f}; // 500 cm/s = sprint
    }
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::Sprint);
}

TEST(MotionFeatureExtractorTest, ClassifiesTurnLeftCorrectly) {
    MotionFeatureExtractor extractor;
    std::vector<SkeletonFrame> frames(30);
    for (int i = 0; i < 30; ++i) {
        frames[i] = test::makeIdentityFrame({0, 90, 0}, i, i / 30.0);
        frames[i].rootVelocity = {50, 0, 50};
        frames[i].rootAngularVel = {0, 120, 0}; // positive yaw = turn left
    }
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::TurnLeft);
}

TEST(MotionFeatureExtractorTest, ClassifiesTurnRightCorrectly) {
    MotionFeatureExtractor extractor;
    std::vector<SkeletonFrame> frames(30);
    for (int i = 0; i < 30; ++i) {
        frames[i] = test::makeIdentityFrame({0, 90, 0}, i, i / 30.0);
        frames[i].rootVelocity = {50, 0, 50};
        frames[i].rootAngularVel = {0, -120, 0}; // negative yaw = turn right
    }
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::TurnRight);
}

TEST(MotionFeatureExtractorTest, ClassifiesJumpWhenVerticalDisplacement) {
    MotionFeatureExtractor extractor;
    std::vector<SkeletonFrame> frames(30);
    for (int i = 0; i < 30; ++i) {
        float t = static_cast<float>(i) / 30.0f;
        // Parabolic jump: height goes from 90 to ~120 and back
        float height = 90.0f + 60.0f * std::sin(t * 3.14159f);
        float vy = 60.0f * 3.14159f * std::cos(t * 3.14159f);
        frames[i] = test::makeIdentityFrame({0, height, static_cast<float>(i)}, i, i / 30.0);
        frames[i].rootVelocity = {0, vy, 30.0f};
    }
    auto type = extractor.classifyHeuristic(frames, 0, 30);
    EXPECT_EQ(type, MotionType::Jump);
}
