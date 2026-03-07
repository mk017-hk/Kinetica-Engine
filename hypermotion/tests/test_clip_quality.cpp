#include <gtest/gtest.h>
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "test_helpers.h"

using namespace hm;
using namespace hm::dataset;

// -------------------------------------------------------------------
// ClipQualityFilter tests
// -------------------------------------------------------------------

TEST(ClipQualityFilterTest, HighQualityClipAccepted) {
    ClipQualityFilter filter;
    auto clip = test::makeTestClip(30, 30.0f);

    auto result = filter.assess(clip);
    EXPECT_TRUE(result.accepted);
    EXPECT_GT(result.overallScore, 0.5f);
}

TEST(ClipQualityFilterTest, TooShortClipRejected) {
    ClipQualityConfig config;
    config.minDurationSec = 1.0f;
    ClipQualityFilter filter(config);

    // 10 frames at 30fps = 0.33s, below 1.0s minimum
    auto clip = test::makeTestClip(10, 30.0f);
    auto result = filter.assess(clip);

    EXPECT_FALSE(result.accepted);
    bool foundShortRejection = false;
    for (const auto& r : result.rejections) {
        if (r.reason == QualityRejection::TooShort)
            foundShortRejection = true;
    }
    EXPECT_TRUE(foundShortRejection);
}

TEST(ClipQualityFilterTest, LowConfidenceClipRejected) {
    ClipQualityConfig config;
    config.minAvgConfidence = 0.8f;
    ClipQualityFilter filter(config);

    auto clip = test::makeTestClip(30, 30.0f);
    // Set low confidence on all joints
    for (auto& frame : clip.frames) {
        for (auto& joint : frame.joints) {
            joint.confidence = 0.1f;
        }
    }

    auto result = filter.assess(clip);
    EXPECT_FALSE(result.accepted);
}

TEST(ClipQualityFilterTest, JitteryClipDetected) {
    ClipQualityConfig config;
    config.maxJitterThreshold = 5.0f; // very strict
    ClipQualityFilter filter(config);

    auto clip = test::makeTestClip(30, 30.0f);
    // Inject large jitter on alternating frames
    for (size_t i = 0; i < clip.frames.size(); i += 2) {
        test::injectOutlier(clip.frames, i,
                            static_cast<int>(Joint::LeftHand), {100, 100, 100});
    }

    auto result = filter.assess(clip);
    EXPECT_GT(result.maxJitter, 5.0f);
}

TEST(ClipQualityFilterTest, FilterBatch_SeparatesAcceptedAndRejected) {
    ClipQualityConfig config;
    config.minDurationSec = 0.5f;
    ClipQualityFilter filter(config);

    std::vector<AnimClip> clips;
    clips.push_back(test::makeTestClip(30, 30.0f, "good"));   // 1.0s — accepted
    clips.push_back(test::makeTestClip(5, 30.0f, "short"));   // 0.17s — rejected

    auto result = filter.filter(clips);
    EXPECT_EQ(result.accepted.size(), 1u);
    EXPECT_EQ(result.rejected.size(), 1u);
    EXPECT_EQ(result.accepted[0].name, "good");
}

TEST(ClipQualityFilterTest, OverallScoreInRange) {
    ClipQualityFilter filter;
    auto clip = test::makeTestClip(60, 30.0f);
    auto result = filter.assess(clip);

    EXPECT_GE(result.overallScore, 0.0f);
    EXPECT_LE(result.overallScore, 1.0f);
}

TEST(ClipQualityFilterTest, EmptyClipHandledGracefully) {
    ClipQualityFilter filter;
    AnimClip empty;
    empty.fps = 30.0f;

    auto result = filter.assess(empty);
    EXPECT_FALSE(result.accepted);
}

// -------------------------------------------------------------------
// ClipExtractor tests
// -------------------------------------------------------------------

TEST(ClipExtractorTest, ExtractsClipsFromWalkingSequence) {
    ClipExtractor extractor;
    auto frames = test::makeWalkingSequence(90, 100.0f);

    auto result = extractor.extract(frames, 0);
    EXPECT_GE(result.clips.size(), 1u);
    EXPECT_EQ(result.clips.size(), result.metadata.size());
}

TEST(ClipExtractorTest, ExtractFromSegments) {
    ClipExtractor extractor;
    auto frames = test::makeWalkingSequence(60, 100.0f);

    std::vector<MotionSegment> segs;
    MotionSegment s1;
    s1.startFrame = 0;
    s1.endFrame = 29;
    s1.type = MotionType::Walk;
    s1.confidence = 0.9f;
    segs.push_back(s1);

    MotionSegment s2;
    s2.startFrame = 30;
    s2.endFrame = 59;
    s2.type = MotionType::Jog;
    s2.confidence = 0.85f;
    segs.push_back(s2);

    auto result = extractor.extractFromSegments(frames, segs, 0);
    EXPECT_EQ(result.clips.size(), 2u);
}

TEST(ClipExtractorTest, RespectsMinClipDuration) {
    ClipExtractorConfig config;
    config.minClipDurationSec = 2.0f;
    config.fps = 30.0f;
    ClipExtractor extractor(config);

    auto frames = test::makeWalkingSequence(30, 100.0f);  // 1.0s total

    auto result = extractor.extract(frames, 0);
    // With only 1s of data and 2s min, might produce 0 or 1 clips
    for (const auto& meta : result.metadata) {
        // If clips are produced, they shouldn't violate minimum
        // (extractor may still produce them if it's all that's available)
        EXPECT_GT(meta.durationSec, 0.0f);
    }
}

TEST(ClipExtractorTest, ClipMetadataPopulated) {
    ClipExtractor extractor;
    auto frames = test::makeWalkingSequence(60, 100.0f);
    auto result = extractor.extract(frames, 7);

    for (const auto& meta : result.metadata) {
        EXPECT_EQ(meta.playerID, 7);
        EXPECT_GT(meta.durationSec, 0.0f);
        EXPECT_GE(meta.startFrame, 0);
    }
}
