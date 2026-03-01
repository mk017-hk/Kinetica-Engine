#include <gtest/gtest.h>
#include "HyperMotion/core/PipelineConfigIO.h"
#include "HyperMotion/core/ScopedTimer.h"
#include "test_helpers.h"
#include <thread>
#include <chrono>
#include <cmath>

using namespace hm;

// ---------------------------------------------------------------
// PipelineConfigIO — round-trip
// ---------------------------------------------------------------

TEST(PipelineConfigIOTest, DefaultConfig_RoundTrip) {
    PipelineConfig original;
    std::string json = serialisePipelineConfig(original);
    ASSERT_FALSE(json.empty());

    PipelineConfig loaded;
    ASSERT_TRUE(parsePipelineConfig(json, loaded));

    EXPECT_FLOAT_EQ(loaded.targetFPS, original.targetFPS);
    EXPECT_EQ(loaded.splitBySegment, original.splitBySegment);
    EXPECT_EQ(loaded.outputFormat, original.outputFormat);
    EXPECT_EQ(loaded.minTrackFrames, original.minTrackFrames);
}

TEST(PipelineConfigIOTest, CustomValues_RoundTrip) {
    PipelineConfig cfg;
    cfg.targetFPS = 60.0f;
    cfg.splitBySegment = false;
    cfg.outputFormat = "both";
    cfg.outputDirectory = "/tmp/test_output";
    cfg.minTrackFrames = 25;
    cfg.poseConfig.detector.modelPath = "models/yolov8.onnx";
    cfg.poseConfig.detector.confidenceThreshold = 0.7f;
    cfg.poseConfig.poseEstimator.modelPath = "models/hrnet.onnx";
    cfg.segmenterConfig.modelPath = "models/tcn.onnx";
    cfg.segmenterConfig.slidingWindowSize = 512;

    std::string json = serialisePipelineConfig(cfg);
    PipelineConfig loaded;
    ASSERT_TRUE(parsePipelineConfig(json, loaded));

    EXPECT_FLOAT_EQ(loaded.targetFPS, 60.0f);
    EXPECT_FALSE(loaded.splitBySegment);
    EXPECT_EQ(loaded.outputFormat, "both");
    EXPECT_EQ(loaded.outputDirectory, "/tmp/test_output");
    EXPECT_EQ(loaded.minTrackFrames, 25);
    EXPECT_EQ(loaded.poseConfig.detector.modelPath, "models/yolov8.onnx");
    EXPECT_NEAR(loaded.poseConfig.detector.confidenceThreshold, 0.7f, test::kEps);
    EXPECT_EQ(loaded.poseConfig.poseEstimator.modelPath, "models/hrnet.onnx");
    EXPECT_EQ(loaded.segmenterConfig.modelPath, "models/tcn.onnx");
    EXPECT_EQ(loaded.segmenterConfig.slidingWindowSize, 512);
}

TEST(PipelineConfigIOTest, PartialJson_MissingFields_KeepDefaults) {
    std::string json = R"({"targetFPS": 120, "outputFormat": "bvh"})";
    PipelineConfig cfg;
    ASSERT_TRUE(parsePipelineConfig(json, cfg));

    EXPECT_FLOAT_EQ(cfg.targetFPS, 120.0f);
    EXPECT_EQ(cfg.outputFormat, "bvh");
    // Defaults should be preserved for fields not in JSON
    EXPECT_TRUE(cfg.splitBySegment);
    EXPECT_EQ(cfg.minTrackFrames, 10);
}

TEST(PipelineConfigIOTest, InvalidJson_ReturnsFalse) {
    PipelineConfig cfg;
    EXPECT_FALSE(parsePipelineConfig("not json at all", cfg));
    EXPECT_FALSE(parsePipelineConfig("{broken", cfg));
}

TEST(PipelineConfigIOTest, EmptyJson_KeepsAllDefaults) {
    PipelineConfig cfg;
    ASSERT_TRUE(parsePipelineConfig("{}", cfg));
    PipelineConfig defaults;
    EXPECT_FLOAT_EQ(cfg.targetFPS, defaults.targetFPS);
    EXPECT_EQ(cfg.outputFormat, defaults.outputFormat);
}

TEST(PipelineConfigIOTest, SignalConfig_RoundTrip) {
    PipelineConfig cfg;
    cfg.signalConfig.enableButterworth = false;
    cfg.signalConfig.butterworthConfig.cutoffFreqBody = 20.0f;
    cfg.signalConfig.sgConfig.windowSize = 11;

    std::string json = serialisePipelineConfig(cfg);
    PipelineConfig loaded;
    ASSERT_TRUE(parsePipelineConfig(json, loaded));

    EXPECT_FALSE(loaded.signalConfig.enableButterworth);
    EXPECT_NEAR(loaded.signalConfig.butterworthConfig.cutoffFreqBody, 20.0f, test::kEps);
    EXPECT_EQ(loaded.signalConfig.sgConfig.windowSize, 11);
}

// ---------------------------------------------------------------
// PipelineStats serialisation
// ---------------------------------------------------------------

TEST(PipelineStatsTest, Serialise_ContainsAllFields) {
    PipelineStats stats;
    stats.poseExtractionMs = 100.5;
    stats.totalFramesProcessed = 300;
    stats.clipsProduced = 5;
    std::string json = serialisePipelineStats(stats);

    EXPECT_NE(json.find("poseExtractionMs"), std::string::npos);
    EXPECT_NE(json.find("totalFramesProcessed"), std::string::npos);
    EXPECT_NE(json.find("clipsProduced"), std::string::npos);
    EXPECT_NE(json.find("300"), std::string::npos);
}

// ---------------------------------------------------------------
// ScopedTimer
// ---------------------------------------------------------------

TEST(ScopedTimerTest, MeasuresNonZeroTime) {
    double elapsed = 0;
    {
        ScopedTimer t(elapsed);
        // Do minimal work
        volatile int sum = 0;
        for (int i = 0; i < 10000; ++i) sum += i;
    }
    // Should have measured some positive time
    EXPECT_GT(elapsed, 0.0);
}

TEST(ScopedTimerTest, ElapsedMsDuringScope) {
    double elapsed = 0;
    ScopedTimer t(elapsed);
    double mid = t.elapsedMs();
    EXPECT_GE(mid, 0.0);
    // elapsed hasn't been written yet (not destroyed)
    EXPECT_EQ(elapsed, 0.0);
}

TEST(ScopedTimerTest, WritesOnDestruction) {
    double elapsed = -1.0;
    {
        ScopedTimer t(elapsed);
    }
    EXPECT_GE(elapsed, 0.0);
}

// ---------------------------------------------------------------
// Deterministic JSON export (synthetic skeleton)
// ---------------------------------------------------------------

TEST(JSONExportDeterministicTest, SyntheticClip_Deterministic) {
    auto clip = test::makeTestClip(5, 30.0f, "deterministic_test");

    xport::JSONExportConfig config;
    config.prettyPrint = false;  // compact for easier comparison
    config.includeMetadata = true;
    config.includePositions = true;
    config.includeQuaternions = true;

    xport::JSONExporter exporter(config);
    std::string json1 = exporter.exportToString(clip);
    std::string json2 = exporter.exportToString(clip);

    // Same input should always produce identical output
    ASSERT_EQ(json1, json2);
    EXPECT_FALSE(json1.empty());

    // Should contain expected structural elements
    EXPECT_NE(json1.find("deterministic_test"), std::string::npos);
    EXPECT_NE(json1.find("frames"), std::string::npos);
    EXPECT_NE(json1.find("fps"), std::string::npos);
}

TEST(JSONExportDeterministicTest, SyntheticClip_HasCorrectFrameCount) {
    auto clip = test::makeTestClip(10, 30.0f, "frame_count_test");

    xport::JSONExporter exporter;
    std::string json = exporter.exportToString(clip);

    // Count occurrences of "frameIndex" to verify frame count
    size_t count = 0;
    size_t pos = 0;
    while ((pos = json.find("frameIndex", pos)) != std::string::npos) {
        count++;
        pos += 10;
    }
    EXPECT_EQ(count, 10u);
}
