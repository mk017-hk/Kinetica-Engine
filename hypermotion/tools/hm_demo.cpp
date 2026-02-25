// hm_demo — HyperMotion vertical slice demo
//
// Generates a synthetic 90-frame walking animation through the full pipeline:
// skeleton construction, signal processing, motion segmentation, and export.
// Outputs demo_clip.json (with schema version) and demo_clip.bvh.
// No ML models or video input required.

#include "HyperMotion/core/Types.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/export/AnimClipUtils.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <string>

using namespace hm;

// Generate a synthetic walking skeleton frame at a given phase
static SkeletonFrame generateWalkFrame(int frameIndex, float fps, float speedCmPerSec) {
    SkeletonFrame frame;
    frame.frameIndex = frameIndex;
    frame.timestamp = static_cast<double>(frameIndex) / fps;
    frame.trackingID = 0;

    float t = frame.timestamp;
    float phase = t * 2.0f * 3.14159265f; // ~1 Hz walk cycle
    float stride = speedCmPerSec / fps;

    // Root: move forward along Z
    frame.rootPosition = Vec3{0.0f, 90.0f, frameIndex * stride};
    frame.rootVelocity = Vec3{0.0f, 0.0f, speedCmPerSec};
    frame.rootRotation = Quat::identity();

    // Initialize all joints to identity
    for (int j = 0; j < JOINT_COUNT; ++j) {
        frame.joints[j].localRotation = Quat::identity();
        frame.joints[j].rotation6D = MathUtils::quatToRot6D(Quat::identity());
        frame.joints[j].localEulerDeg = Vec3{0, 0, 0};
        frame.joints[j].confidence = 0.95f;
    }

    // Spine: slight forward lean
    float spineLean = 5.0f;
    frame.joints[static_cast<int>(Joint::Spine)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, spineLean);

    // Spine counter-rotation (torso twist during walk)
    float torsoTwist = 3.0f * std::sin(phase);
    frame.joints[static_cast<int>(Joint::Spine2)].localRotation =
        MathUtils::fromAxisAngle({0, 1, 0}, torsoTwist);

    // Left leg: hip flexion/extension
    float leftHipAngle = 25.0f * std::sin(phase);
    frame.joints[static_cast<int>(Joint::LeftUpLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, leftHipAngle);

    // Left knee: always bent, more at swing phase
    float leftKneeAngle = 15.0f + 30.0f * std::max(0.0f, std::sin(phase - 0.5f));
    frame.joints[static_cast<int>(Joint::LeftLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, leftKneeAngle);

    // Right leg: opposite phase
    float rightHipAngle = 25.0f * std::sin(phase + 3.14159265f);
    frame.joints[static_cast<int>(Joint::RightUpLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, rightHipAngle);

    float rightKneeAngle = 15.0f + 30.0f * std::max(0.0f, std::sin(phase + 3.14159265f - 0.5f));
    frame.joints[static_cast<int>(Joint::RightLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, rightKneeAngle);

    // Arms: counter-swing to legs
    float leftArmSwing = 15.0f * std::sin(phase + 3.14159265f);
    frame.joints[static_cast<int>(Joint::LeftArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, leftArmSwing);

    float leftElbowBend = 20.0f + 10.0f * std::sin(phase);
    frame.joints[static_cast<int>(Joint::LeftForeArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, leftElbowBend);

    float rightArmSwing = 15.0f * std::sin(phase);
    frame.joints[static_cast<int>(Joint::RightArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, rightArmSwing);

    float rightElbowBend = 20.0f + 10.0f * std::sin(phase + 3.14159265f);
    frame.joints[static_cast<int>(Joint::RightForeArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, rightElbowBend);

    // Head: slight bob
    float headBob = 2.0f * std::sin(2.0f * phase);
    frame.joints[static_cast<int>(Joint::Head)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, headBob);

    // Compute all derived representations
    std::array<Quat, JOINT_COUNT> localRots;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        auto& jt = frame.joints[j];
        jt.rotation6D = MathUtils::quatToRot6D(jt.localRotation);
        jt.localEulerDeg = MathUtils::quatToEulerDeg(jt.localRotation);
        localRots[j] = jt.localRotation;
    }

    // Forward kinematics for world positions
    auto worldPos = MathUtils::forwardKinematics(
        frame.rootPosition, frame.rootRotation, localRots);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        frame.joints[j].worldPosition = worldPos[j];
    }

    return frame;
}

int main(int argc, char* argv[]) {
    // Parse optional output directory
    std::string outputDir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: hm_demo [--output <dir>]\n"
                      << "  Generates demo_clip.json and demo_clip.bvh\n";
            return 0;
        }
    }

    auto totalStart = std::chrono::high_resolution_clock::now();

    Logger::instance().setLevel(LogLevel::Info);
    HM_LOG_INFO("Demo", "HyperMotion demo pipeline v" + std::string(HM_VERSION_STRING));
    HM_LOG_INFO("Demo", "Schema version: " + std::string(HM_SCHEMA_VERSION));

    // ------------------------------------------------------------------
    // Stage 1: Generate synthetic walking skeleton sequence
    // ------------------------------------------------------------------
    constexpr float fps = 30.0f;
    constexpr int numFrames = 90; // 3 seconds
    constexpr float walkSpeed = 120.0f; // cm/s (~4.3 km/h walking pace)

    std::cout << "\n=== HyperMotion Demo Pipeline ===\n\n";
    std::cout << "[1/4] Generating synthetic walking animation...\n";

    std::vector<SkeletonFrame> frames;
    frames.reserve(numFrames);
    for (int f = 0; f < numFrames; ++f) {
        frames.push_back(generateWalkFrame(f, fps, walkSpeed));
    }
    std::cout << "      Generated " << numFrames << " frames at " << fps << " FPS\n";

    // ------------------------------------------------------------------
    // Stage 2: Signal processing
    // ------------------------------------------------------------------
    std::cout << "[2/4] Running signal processing pipeline...\n";

    signal::SignalPipelineConfig signalCfg;
    signalCfg.enableOutlierFilter = true;
    signalCfg.enableSavitzkyGolay = true;
    signalCfg.enableButterworth = true;
    signalCfg.enableQuaternionSmoothing = true;
    signalCfg.enableFootContact = true;
    signal::SignalPipeline signalPipeline(signalCfg);
    signalPipeline.process(frames);
    std::cout << "      5-stage filtering complete\n";

    // ------------------------------------------------------------------
    // Stage 3: Motion segmentation (heuristic)
    // ------------------------------------------------------------------
    std::cout << "[3/4] Segmenting motion...\n";

    segmenter::MotionSegmenterConfig segCfg;
    segmenter::MotionSegmenter segmenter(segCfg);
    segmenter.initialize();
    auto segments = segmenter.segment(frames, 0);
    std::cout << "      Found " << segments.size() << " motion segments\n";

    // ------------------------------------------------------------------
    // Stage 4: Export
    // ------------------------------------------------------------------
    std::cout << "[4/4] Exporting animation data...\n";

    AnimClip clip;
    clip.name = "demo_walk";
    clip.fps = fps;
    clip.trackingID = 0;
    clip.frames = std::move(frames);
    clip.segments = std::move(segments);

    // JSON export
    xport::JSONExportConfig jsonCfg;
    jsonCfg.includeMetadata = true;
    jsonCfg.includePositions = true;
    jsonCfg.includeQuaternions = true;
    jsonCfg.includeEuler = true;
    jsonCfg.includeSegments = true;
    jsonCfg.prettyPrint = true;
    xport::JSONExporter jsonExporter(jsonCfg);

    std::string jsonPath = outputDir + "/demo_clip.json";
    bool jsonOk = jsonExporter.exportToFile(clip, jsonPath);

    // BVH export
    xport::BVHExportConfig bvhCfg;
    bvhCfg.exportRootMotion = true;
    bvhCfg.useEndSites = true;
    xport::BVHExporter bvhExporter(bvhCfg);

    std::string bvhPath = outputDir + "/demo_clip.bvh";
    bool bvhOk = bvhExporter.exportToFile(clip, bvhPath);

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    std::cout << "\n=== Demo Complete ===\n\n";
    std::cout << "  Frames:       " << clip.frames.size() << "\n";
    std::cout << "  Joints:       " << JOINT_COUNT << "\n";
    std::cout << "  Segments:     " << clip.segments.size() << "\n";
    std::cout << "  FPS:          " << clip.fps << "\n";
    std::cout << "  Duration:     " << clip.frames.size() / clip.fps << "s\n";
    std::cout << "  Schema:       " << HM_SCHEMA_VERSION << "\n";
    std::cout << "\n";
    std::cout << "  JSON output:  " << jsonPath << (jsonOk ? " [OK]" : " [FAILED]") << "\n";
    std::cout << "  BVH output:   " << bvhPath << (bvhOk ? " [OK]" : " [FAILED]") << "\n";
    std::cout << "\n";
    std::cout << "  Elapsed:      " << static_cast<int>(elapsedMs) << " ms\n";
    std::cout << "\n";

    return (jsonOk && bvhOk) ? 0 : 1;
}
