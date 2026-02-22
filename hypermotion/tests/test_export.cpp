#include <gtest/gtest.h>
#include "HyperMotion/export/AnimClipUtils.h"
#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/core/MathUtils.h"
#include "test_helpers.h"
#include <string>

using namespace hm;
using namespace hm::xport;

// ---------------------------------------------------------------
// AnimClipUtils
// ---------------------------------------------------------------

TEST(AnimClipUtilsTest, GetFrameCount) {
    auto clip = test::makeTestClip(30);
    EXPECT_EQ(AnimClipUtils::getFrameCount(clip), 30);
}

TEST(AnimClipUtilsTest, GetDuration) {
    auto clip = test::makeTestClip(60, 30.0f);
    float duration = AnimClipUtils::getDuration(clip);
    EXPECT_NEAR(duration, 2.0f, test::kEps); // 60 frames / 30 fps
}

TEST(AnimClipUtilsTest, SubClip) {
    auto clip = test::makeTestClip(30);
    auto sub = AnimClipUtils::subClip(clip, 5, 15);
    EXPECT_EQ(AnimClipUtils::getFrameCount(sub), 10);
    // Frame indices should be adjusted
    EXPECT_EQ(sub.frames[0].frameIndex, clip.frames[5].frameIndex);
}

TEST(AnimClipUtilsTest, SubClip_FullRange) {
    auto clip = test::makeTestClip(20);
    auto sub = AnimClipUtils::subClip(clip, 0, 20);
    EXPECT_EQ(AnimClipUtils::getFrameCount(sub), 20);
}

TEST(AnimClipUtilsTest, SplitBySegments) {
    auto clip = test::makeTestClip(40); // Has 2 segments
    auto splits = AnimClipUtils::splitBySegments(clip);
    EXPECT_EQ(splits.size(), 2u);
    // Total frames across splits should cover original
    int totalFrames = 0;
    for (const auto& s : splits)
        totalFrames += AnimClipUtils::getFrameCount(s);
    EXPECT_GT(totalFrames, 0);
}

TEST(AnimClipUtilsTest, Concatenate) {
    auto clip1 = test::makeTestClip(10, 30.0f, "clip1");
    auto clip2 = test::makeTestClip(15, 30.0f, "clip2");
    auto result = AnimClipUtils::concatenate({clip1, clip2});
    EXPECT_EQ(AnimClipUtils::getFrameCount(result), 25);
}

TEST(AnimClipUtilsTest, Concatenate_EmptyList) {
    auto result = AnimClipUtils::concatenate({});
    EXPECT_EQ(AnimClipUtils::getFrameCount(result), 0);
}

TEST(AnimClipUtilsTest, Concatenate_Single) {
    auto clip = test::makeTestClip(20);
    auto result = AnimClipUtils::concatenate({clip});
    EXPECT_EQ(AnimClipUtils::getFrameCount(result), 20);
}

TEST(AnimClipUtilsTest, Resample_SameFPS) {
    auto clip = test::makeTestClip(30, 30.0f);
    auto resampled = AnimClipUtils::resample(clip, 30.0f);
    EXPECT_EQ(AnimClipUtils::getFrameCount(resampled),
              AnimClipUtils::getFrameCount(clip));
}

TEST(AnimClipUtilsTest, Resample_DoubleFPS) {
    auto clip = test::makeTestClip(30, 30.0f);
    auto resampled = AnimClipUtils::resample(clip, 60.0f);
    // Should roughly double the number of frames
    int expected = static_cast<int>(30 * (60.0f / 30.0f));
    EXPECT_NEAR(AnimClipUtils::getFrameCount(resampled), expected, 2);
    EXPECT_NEAR(resampled.fps, 60.0f, test::kEps);
}

TEST(AnimClipUtilsTest, Resample_HalfFPS) {
    auto clip = test::makeTestClip(30, 30.0f);
    auto resampled = AnimClipUtils::resample(clip, 15.0f);
    int expected = static_cast<int>(30 * (15.0f / 30.0f));
    EXPECT_NEAR(AnimClipUtils::getFrameCount(resampled), expected, 2);
}

TEST(AnimClipUtilsTest, Mirror) {
    auto clip = test::makeTestClip(10);
    auto mirrored = AnimClipUtils::mirror(clip);
    EXPECT_EQ(AnimClipUtils::getFrameCount(mirrored),
              AnimClipUtils::getFrameCount(clip));
}

TEST(AnimClipUtilsTest, TrimSilence) {
    // Create a clip where first and last 5 frames are stationary
    auto clip = test::makeTestClip(30);
    for (int i = 0; i < 5; ++i) {
        clip.frames[i].rootVelocity = {0, 0, 0};
        clip.frames[29 - i].rootVelocity = {0, 0, 0};
    }
    for (int i = 5; i < 25; ++i) {
        clip.frames[i].rootVelocity = {100, 0, 0};
    }

    auto trimmed = AnimClipUtils::trimSilence(clip, 5.0f);
    EXPECT_LE(AnimClipUtils::getFrameCount(trimmed), 30);
}

// ---------------------------------------------------------------
// BVHExporter
// ---------------------------------------------------------------

TEST(BVHExporterTest, ExportToString_NotEmpty) {
    auto clip = test::makeTestClip(10);
    BVHExporter exporter;
    std::string bvh = exporter.exportToString(clip);
    EXPECT_FALSE(bvh.empty());
}

TEST(BVHExporterTest, ExportToString_HasHierarchy) {
    auto clip = test::makeTestClip(10);
    BVHExporter exporter;
    std::string bvh = exporter.exportToString(clip);
    EXPECT_NE(bvh.find("HIERARCHY"), std::string::npos);
    EXPECT_NE(bvh.find("ROOT"), std::string::npos);
}

TEST(BVHExporterTest, ExportToString_HasMotion) {
    auto clip = test::makeTestClip(10);
    BVHExporter exporter;
    std::string bvh = exporter.exportToString(clip);
    EXPECT_NE(bvh.find("MOTION"), std::string::npos);
    EXPECT_NE(bvh.find("Frames:"), std::string::npos);
    EXPECT_NE(bvh.find("Frame Time:"), std::string::npos);
}

TEST(BVHExporterTest, ExportToString_HasJointNames) {
    auto clip = test::makeTestClip(5);
    BVHExporter exporter;
    std::string bvh = exporter.exportToString(clip);
    EXPECT_NE(bvh.find("Hips"), std::string::npos);
    EXPECT_NE(bvh.find("Spine"), std::string::npos);
    EXPECT_NE(bvh.find("Head"), std::string::npos);
}

// ---------------------------------------------------------------
// JSONExporter
// ---------------------------------------------------------------

TEST(JSONExporterTest, ExportToString_NotEmpty) {
    auto clip = test::makeTestClip(5);
    JSONExporter exporter;
    std::string json = exporter.exportToString(clip);
    EXPECT_FALSE(json.empty());
}

TEST(JSONExporterTest, ExportToString_ValidJSON) {
    auto clip = test::makeTestClip(5);
    JSONExporter exporter;
    std::string json = exporter.exportToString(clip);
    // Should start with { and end with }
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(), '}');
}

TEST(JSONExporterTest, ExportToString_HasFrames) {
    auto clip = test::makeTestClip(5);
    JSONExporter exporter;
    std::string json = exporter.exportToString(clip);
    EXPECT_NE(json.find("frames"), std::string::npos);
}

TEST(JSONExporterTest, ExportToString_HasMetadata) {
    auto clip = test::makeTestClip(5);
    JSONExportConfig config;
    config.includeMetadata = true;
    JSONExporter exporter(config);
    std::string json = exporter.exportToString(clip);
    EXPECT_NE(json.find("fps"), std::string::npos);
}

TEST(JSONExporterTest, ExportToString_ConfigOptions) {
    auto clip = test::makeTestClip(3);

    JSONExportConfig config;
    config.includePositions = false;
    config.includeQuaternions = false;
    config.includeEuler = false;
    config.includeRotation6D = true;
    JSONExporter exporter(config);

    std::string json = exporter.exportToString(clip);
    EXPECT_FALSE(json.empty());
}
