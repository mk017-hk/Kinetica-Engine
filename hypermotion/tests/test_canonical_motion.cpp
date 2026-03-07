#include <gtest/gtest.h>
#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/core/MathUtils.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;
using namespace hm::motion;

// -------------------------------------------------------------------
// Limb length stabilisation
// -------------------------------------------------------------------

TEST(CanonicalMotionTest, MeasureLimbLengths_ReturnsNonZero) {
    auto frames = test::makeWalkingSequence(30, 100.0f);
    auto lengths = CanonicalMotionBuilder::measureLimbLengths(frames);

    // At least the spine chain should have non-zero limb lengths
    // Spine (joint 1) parent is Hips (joint 0)
    EXPECT_GT(lengths[1], 0.0f);
    // LeftUpLeg (joint 14) parent is Hips (joint 0)
    EXPECT_GT(lengths[14], 0.0f);
}

TEST(CanonicalMotionTest, LimbLengthStabilisation_ReducesVariance) {
    // Create frames with jittery limb lengths
    auto frames = test::makeWalkingSequence(60, 100.0f);

    // Inject random bone-length noise into left leg
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 5.0f);
    for (auto& f : frames) {
        auto& knee = f.joints[static_cast<int>(Joint::LeftLeg)].worldPosition;
        knee.y += noise(rng);
    }

    // Measure original limb length variance
    std::vector<float> origLengths;
    int leftLeg = static_cast<int>(Joint::LeftLeg);
    int leftUpLeg = static_cast<int>(Joint::LeftUpLeg);
    for (const auto& f : frames) {
        Vec3 diff = f.joints[leftLeg].worldPosition -
                    f.joints[leftUpLeg].worldPosition;
        origLengths.push_back(diff.length());
    }
    float origMean = 0.0f;
    for (auto l : origLengths) origMean += l;
    origMean /= origLengths.size();
    float origVar = 0.0f;
    for (auto l : origLengths) origVar += (l - origMean) * (l - origMean);
    origVar /= origLengths.size();

    // Build canonical motion with stabilisation
    CanonicalMotionBuilderConfig config;
    config.stabiliseLimbLengths = true;
    CanonicalMotionBuilder builder(config);
    auto canonical = builder.build(frames, 0, 30.0f);

    // Convert back to skeleton frames
    auto rebuilt = builder.toSkeletonFrames(canonical);

    // Measure rebuilt limb length variance
    std::vector<float> newLengths;
    for (const auto& f : rebuilt) {
        Vec3 diff = f.joints[leftLeg].worldPosition -
                    f.joints[leftUpLeg].worldPosition;
        newLengths.push_back(diff.length());
    }
    float newMean = 0.0f;
    for (auto l : newLengths) newMean += l;
    newMean /= newLengths.size();
    float newVar = 0.0f;
    for (auto l : newLengths) newVar += (l - newMean) * (l - newMean);
    newVar /= newLengths.size();

    // Stabilised variance should be significantly lower
    EXPECT_LT(newVar, origVar);
}

// -------------------------------------------------------------------
// Root orientation
// -------------------------------------------------------------------

TEST(CanonicalMotionTest, RootOrientation_ForwardMotion) {
    // Moving along +X: root should face roughly +X
    auto frames = test::makeWalkingSequence(30, 200.0f);

    CanonicalMotionBuilderConfig config;
    config.solveRootOrientation = true;
    CanonicalMotionBuilder builder(config);
    auto canonical = builder.build(frames, 0, 30.0f);

    // Check that the forward direction of the root roughly aligns with +X
    // by rotating the unit Z vector by root rotation and checking X component
    for (size_t i = 5; i < canonical.frames.size(); ++i) {
        Vec3 fwd = canonical.frames[i].rootRotation.rotate({0, 0, 1});
        // The forward direction should have a significant X component
        // (depending on convention, might be Z or X)
        float horizontalMag = std::sqrt(fwd.x * fwd.x + fwd.z * fwd.z);
        EXPECT_GT(horizontalMag, 0.5f)
            << "Frame " << i << " root orientation forward has weak horizontal component";
    }
}

TEST(CanonicalMotionTest, RootOrientation_ConsistentOverTime) {
    auto frames = test::makeWalkingSequence(60, 150.0f);

    CanonicalMotionBuilderConfig config;
    config.solveRootOrientation = true;
    CanonicalMotionBuilder builder(config);
    auto canonical = builder.build(frames, 0, 30.0f);

    // Root orientation should be consistent for uniform linear motion
    for (size_t i = 10; i < canonical.frames.size() - 1; ++i) {
        float dot = canonical.frames[i].rootRotation.dot(
            canonical.frames[i + 1].rootRotation);
        EXPECT_GT(std::abs(dot), 0.95f)
            << "Root orientation jumped between frames " << i << " and " << i + 1;
    }
}

// -------------------------------------------------------------------
// Canonical motion build / round-trip
// -------------------------------------------------------------------

TEST(CanonicalMotionTest, BuildProducesCorrectFrameCount) {
    auto frames = test::makeWalkingSequence(40, 100.0f);

    CanonicalMotionBuilder builder;
    auto canonical = builder.build(frames, 5, 30.0f);

    EXPECT_EQ(canonical.frames.size(), 40u);
    EXPECT_EQ(canonical.trackingID, 5);
    EXPECT_FLOAT_EQ(canonical.fps, 30.0f);
}

TEST(CanonicalMotionTest, RootTrajectoryMatchesFrameCount) {
    auto frames = test::makeWalkingSequence(30, 100.0f);

    CanonicalMotionBuilder builder;
    auto canonical = builder.build(frames, 0, 30.0f);

    EXPECT_EQ(canonical.rootTrajectory.size(), canonical.frames.size());
}

TEST(CanonicalMotionTest, RoundTrip_PreservesRootPosition) {
    auto frames = test::makeWalkingSequence(30, 100.0f);

    CanonicalMotionBuilder builder;
    auto canonical = builder.build(frames, 0, 30.0f);
    auto rebuilt = builder.toSkeletonFrames(canonical);

    ASSERT_EQ(rebuilt.size(), frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        EXPECT_NEAR(rebuilt[i].rootPosition.x, frames[i].rootPosition.x, 5.0f)
            << "Root X diverged at frame " << i;
        EXPECT_NEAR(rebuilt[i].rootPosition.y, frames[i].rootPosition.y, 5.0f)
            << "Root Y diverged at frame " << i;
    }
}

TEST(CanonicalMotionTest, ProcessAnimClip_UpdatesFrames) {
    auto clip = test::makeTestClip(30, 30.0f);
    auto origFrameCount = clip.frames.size();

    CanonicalMotionBuilder builder;
    builder.process(clip);

    EXPECT_EQ(clip.frames.size(), origFrameCount);
}

TEST(CanonicalMotionTest, RootVelocity_NonZeroForMovingSequence) {
    auto frames = test::makeWalkingSequence(30, 200.0f);

    CanonicalMotionBuilder builder;
    auto canonical = builder.build(frames, 0, 30.0f);

    // Most frames should have non-zero root velocity
    int nonZeroCount = 0;
    for (const auto& cf : canonical.frames) {
        if (cf.rootVelocity.length() > 1.0f) ++nonZeroCount;
    }
    EXPECT_GT(nonZeroCount, 20);
}
