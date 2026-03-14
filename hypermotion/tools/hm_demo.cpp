// hm_demo — HyperMotion full pipeline demonstration
//
// Generates a synthetic multi-phase motion sequence (walk → sprint → turn →
// idle → kick) through the complete 10-stage pipeline:
//   1. Skeleton construction      5. Canonical motion
//   2. Signal processing          6. Foot contact detection
//   3. Motion segmentation        7. Trajectory extraction
//   4. Clip extraction            8. Quality filtering
//   9. Motion classification     10. Database export
//
// Outputs BVH + JSON files per clip and a database summary.
// No ML models or video input required.

#include "HyperMotion/core/Types.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/motion/FootContactDetector.h"
#include "HyperMotion/motion/TrajectoryExtractor.h"
#include "HyperMotion/analysis/MotionFingerprint.h"
#include "HyperMotion/dataset/ClipExtractor.h"
#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/dataset/MotionClassifier.h"
#include "HyperMotion/dataset/AnimationDatabase.h"
#include "HyperMotion/export/JSONExporter.h"
#include "HyperMotion/export/BVHExporter.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <filesystem>
#include <iomanip>

using namespace hm;

// ===================================================================
// Synthetic motion generators
// ===================================================================

static constexpr float kPi = 3.14159265f;

static SkeletonFrame buildBaseFrame(int frameIndex, float fps) {
    SkeletonFrame frame{};
    frame.frameIndex = frameIndex;
    frame.timestamp = static_cast<double>(frameIndex) / fps;
    frame.trackingID = 0;
    frame.rootRotation = Quat::identity();

    for (int j = 0; j < JOINT_COUNT; ++j) {
        frame.joints[j].localRotation = Quat::identity();
        frame.joints[j].rotation6D = MathUtils::quatToRot6D(Quat::identity());
        frame.joints[j].localEulerDeg = Vec3{0, 0, 0};
        frame.joints[j].confidence = 0.95f;
    }
    return frame;
}

static void applyWalkPose(SkeletonFrame& frame, float phase) {
    float torsoTwist = 3.0f * std::sin(phase);
    frame.joints[static_cast<int>(Joint::Spine2)].localRotation =
        MathUtils::fromAxisAngle({0, 1, 0}, torsoTwist);
    frame.joints[static_cast<int>(Joint::Spine)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 5.0f);

    frame.joints[static_cast<int>(Joint::LeftUpLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 25.0f * std::sin(phase));
    frame.joints[static_cast<int>(Joint::LeftLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 15.0f + 30.0f * std::max(0.0f, std::sin(phase - 0.5f)));
    frame.joints[static_cast<int>(Joint::RightUpLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 25.0f * std::sin(phase + kPi));
    frame.joints[static_cast<int>(Joint::RightLeg)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 15.0f + 30.0f * std::max(0.0f, std::sin(phase + kPi - 0.5f)));

    frame.joints[static_cast<int>(Joint::LeftArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 15.0f * std::sin(phase + kPi));
    frame.joints[static_cast<int>(Joint::LeftForeArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 20.0f + 10.0f * std::sin(phase));
    frame.joints[static_cast<int>(Joint::RightArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 15.0f * std::sin(phase));
    frame.joints[static_cast<int>(Joint::RightForeArm)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 20.0f + 10.0f * std::sin(phase + kPi));

    frame.joints[static_cast<int>(Joint::Head)].localRotation =
        MathUtils::fromAxisAngle({1, 0, 0}, 2.0f * std::sin(2.0f * phase));
}

static void computeDerivedFields(SkeletonFrame& frame) {
    std::array<Quat, JOINT_COUNT> localRots;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        auto& jt = frame.joints[j];
        jt.rotation6D = MathUtils::quatToRot6D(jt.localRotation);
        jt.localEulerDeg = MathUtils::quatToEulerDeg(jt.localRotation);
        localRots[j] = jt.localRotation;
    }
    auto worldPos = MathUtils::forwardKinematics(
        frame.rootPosition, frame.rootRotation, localRots);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        frame.joints[j].worldPosition = worldPos[j];
    }
}

// Generate a multi-phase synthetic motion sequence: walk → sprint → turn → idle → kick
static std::vector<SkeletonFrame> generateMultiPhaseSequence(float fps) {
    std::vector<SkeletonFrame> frames;

    int idx = 0;
    float zPos = 0.0f;
    float facing = 0.0f; // degrees around Y

    // Phase 1: Walk (60 frames, 2 seconds)
    for (int f = 0; f < 60; ++f) {
        auto frame = buildBaseFrame(idx, fps);
        float speed = 120.0f; // cm/s walking
        float stride = speed / fps;
        zPos += stride;
        float phase = frame.timestamp * 2.0f * kPi;

        frame.rootPosition = Vec3{0.0f, 90.0f, zPos};
        frame.rootVelocity = Vec3{0.0f, 0.0f, speed};
        applyWalkPose(frame, phase);
        computeDerivedFields(frame);
        frames.push_back(frame);
        idx++;
    }

    // Phase 2: Sprint (45 frames, 1.5 seconds)
    for (int f = 0; f < 45; ++f) {
        auto frame = buildBaseFrame(idx, fps);
        float speed = 500.0f; // cm/s sprinting
        float stride = speed / fps;
        zPos += stride;
        float phase = frame.timestamp * 3.5f * kPi; // faster cadence

        frame.rootPosition = Vec3{0.0f, 88.0f, zPos}; // slight crouch
        frame.rootVelocity = Vec3{0.0f, 0.0f, speed};

        // Deeper forward lean for sprint
        frame.joints[static_cast<int>(Joint::Spine)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 12.0f);
        frame.joints[static_cast<int>(Joint::LeftUpLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 40.0f * std::sin(phase));
        frame.joints[static_cast<int>(Joint::RightUpLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 40.0f * std::sin(phase + kPi));
        frame.joints[static_cast<int>(Joint::LeftLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 20.0f + 50.0f * std::max(0.0f, std::sin(phase - 0.5f)));
        frame.joints[static_cast<int>(Joint::RightLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 20.0f + 50.0f * std::max(0.0f, std::sin(phase + kPi - 0.5f)));
        frame.joints[static_cast<int>(Joint::LeftArm)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 30.0f * std::sin(phase + kPi));
        frame.joints[static_cast<int>(Joint::RightArm)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 30.0f * std::sin(phase));

        computeDerivedFields(frame);
        frames.push_back(frame);
        idx++;
    }

    // Phase 3: Turn left (30 frames, 1 second)
    float turnRate = 150.0f; // deg/s
    for (int f = 0; f < 30; ++f) {
        auto frame = buildBaseFrame(idx, fps);
        facing += turnRate / fps;
        float rad = facing * kPi / 180.0f;
        float speed = 80.0f;
        float stride = speed / fps;
        zPos += stride * std::cos(rad);
        float xPos = stride * std::sin(rad);

        frame.rootPosition = Vec3{xPos * f, 90.0f, zPos};
        frame.rootVelocity = Vec3{speed * std::sin(rad), 0.0f, speed * std::cos(rad)};
        frame.rootAngularVel = Vec3{0.0f, turnRate, 0.0f};
        frame.rootRotation = MathUtils::fromAxisAngle({0, 1, 0}, facing);

        applyWalkPose(frame, frame.timestamp * 2.0f * kPi);
        computeDerivedFields(frame);
        frames.push_back(frame);
        idx++;
    }

    // Phase 4: Idle (30 frames, 1 second)
    float lastZ = zPos;
    for (int f = 0; f < 30; ++f) {
        auto frame = buildBaseFrame(idx, fps);
        frame.rootPosition = Vec3{0.0f, 90.0f, lastZ};
        frame.rootVelocity = Vec3{0.0f, 0.0f, 0.0f};

        // Subtle breathing/sway
        float sway = 1.0f * std::sin(frame.timestamp * 1.5f * kPi);
        frame.joints[static_cast<int>(Joint::Spine)].localRotation =
            MathUtils::fromAxisAngle({0, 0, 1}, sway);

        computeDerivedFields(frame);
        frames.push_back(frame);
        idx++;
    }

    // Phase 5: Kick (30 frames, 1 second)
    for (int f = 0; f < 30; ++f) {
        auto frame = buildBaseFrame(idx, fps);
        frame.rootPosition = Vec3{0.0f, 90.0f, lastZ};
        frame.rootVelocity = Vec3{0.0f, 0.0f, 20.0f}; // slight forward lean

        float t = static_cast<float>(f) / 30.0f;
        // Right leg kicks forward: wind up → swing through → follow through
        float kickAngle = 0.0f;
        if (t < 0.3f) kickAngle = -30.0f * (t / 0.3f);       // wind up (back)
        else if (t < 0.6f) kickAngle = -30.0f + 120.0f * ((t - 0.3f) / 0.3f); // swing (forward)
        else kickAngle = 90.0f - 60.0f * ((t - 0.6f) / 0.4f); // follow through

        frame.joints[static_cast<int>(Joint::RightUpLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, kickAngle);
        frame.joints[static_cast<int>(Joint::RightLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, std::max(0.0f, -kickAngle * 0.3f));
        // Left leg (plant foot) stable with slight knee bend
        frame.joints[static_cast<int>(Joint::LeftUpLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 5.0f);
        frame.joints[static_cast<int>(Joint::LeftLeg)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, 15.0f);
        // Counter-rotation with arms
        frame.joints[static_cast<int>(Joint::LeftArm)].localRotation =
            MathUtils::fromAxisAngle({1, 0, 0}, kickAngle * 0.2f);

        computeDerivedFields(frame);
        frames.push_back(frame);
        idx++;
    }

    return frames;
}

// ===================================================================
// Demo entry point
// ===================================================================

int main(int argc, char* argv[]) {
    std::string outputDir = "./hm_demo_output";
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: hm_demo [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  -o, --output <dir>  Output directory (default: ./hm_demo_output)\n"
                      << "  -v, --verbose       Show detailed stage timing\n"
                      << "  -h, --help          Show this help\n\n"
                      << "Runs the full HyperMotion pipeline on synthetic multi-phase motion data.\n"
                      << "No ML models or video files required.\n";
            return 0;
        }
    }

    auto totalStart = std::chrono::high_resolution_clock::now();
    auto stageStart = totalStart;
    auto stageTime = [&]() -> double {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - stageStart).count();
        stageStart = now;
        return ms;
    };

    Logger::instance().setLevel(verbose ? LogLevel::Info : LogLevel::Warn);

    constexpr float fps = 30.0f;

    std::cout << "\n";
    std::cout << "  ╔═══════════════════════════════════════════════╗\n";
    std::cout << "  ║   HyperMotion Pipeline Demo v" << HM_VERSION_STRING << "             ║\n";
    std::cout << "  ║   Schema: " << HM_SCHEMA_VERSION << "   Joints: " << JOINT_COUNT
              << "   MotionTypes: " << MOTION_TYPE_COUNT << "   ║\n";
    std::cout << "  ╚═══════════════════════════════════════════════╝\n\n";

    std::cout << "  Output: " << outputDir << "\n\n";

    // ── Stage 1: Generate synthetic multi-phase motion ──────────────
    std::cout << "  [ 1/10] Generating synthetic motion (walk→sprint→turn→idle→kick)...";
    stageTime();
    auto frames = generateMultiPhaseSequence(fps);
    double genMs = stageTime();
    std::cout << " " << frames.size() << " frames (" << std::fixed << std::setprecision(1)
              << genMs << " ms)\n";

    // ── Stage 2: Signal processing ──────────────────────────────────
    std::cout << "  [ 2/10] Signal processing (outlier→SG→Butterworth→quat→foot)...";
    signal::SignalPipelineConfig signalCfg;
    signalCfg.enableOutlierFilter = true;
    signalCfg.enableSavitzkyGolay = true;
    signalCfg.enableButterworth = true;
    signalCfg.enableQuaternionSmoothing = true;
    signalCfg.enableFootContact = true;
    signal::SignalPipeline signalPipeline(signalCfg);
    signalPipeline.process(frames);
    double sigMs = stageTime();
    std::cout << " done (" << std::setprecision(1) << sigMs << " ms)\n";

    // ── Stage 3: Canonical motion ───────────────────────────────────
    std::cout << "  [ 3/10] Canonical motion (limb stabilisation, root extraction)...";
    AnimClip fullClip;
    fullClip.name = "demo_multi_phase";
    fullClip.fps = fps;
    fullClip.trackingID = 0;
    fullClip.frames = std::move(frames);

    motion::CanonicalMotionBuilderConfig canonCfg;
    canonCfg.stabiliseLimbLengths = true;
    canonCfg.extractRootMotion = true;
    canonCfg.solveRootOrientation = true;
    motion::CanonicalMotionBuilder canonicalBuilder(canonCfg);
    canonicalBuilder.process(fullClip);
    double canonMs = stageTime();
    std::cout << " done (" << std::setprecision(1) << canonMs << " ms)\n";

    // ── Stage 4: Motion segmentation ────────────────────────────────
    std::cout << "  [ 4/10] Motion segmentation (heuristic classifier)...";
    segmenter::MotionSegmenterConfig segCfg;
    segCfg.minSegmentLength = 10;
    segmenter::MotionSegmenter segmenter(segCfg);
    segmenter.initialize();
    fullClip.segments = segmenter.segment(fullClip.frames, 0);
    double segMs = stageTime();
    std::cout << " " << fullClip.segments.size() << " segments (" << std::setprecision(1)
              << segMs << " ms)\n";

    // ── Stage 5: Foot contact detection ─────────────────────────────
    std::cout << "  [ 5/10] Foot contact detection...";
    motion::FootContactDetectorConfig footCfg;
    footCfg.fps = fps;
    motion::FootContactDetector footDetector(footCfg);
    footDetector.process(fullClip);
    double footMs = stageTime();
    int contactFrames = 0;
    for (const auto& fc : fullClip.footContacts) {
        if (fc.leftFootContact || fc.rightFootContact) contactFrames++;
    }
    std::cout << " " << contactFrames << " contact frames (" << std::setprecision(1)
              << footMs << " ms)\n";

    // ── Stage 6: Trajectory extraction ──────────────────────────────
    std::cout << "  [ 6/10] Trajectory extraction...";
    motion::TrajectoryExtractorConfig trajCfg;
    motion::TrajectoryExtractor trajExtractor(trajCfg);
    trajExtractor.process(fullClip);
    double trajMs = stageTime();
    std::cout << " done (" << std::setprecision(1) << trajMs << " ms)\n";

    // ── Stage 7: Clip extraction from segments ──────────────────────
    std::cout << "  [ 7/10] Clip extraction from segments...";
    dataset::ClipExtractorConfig extractorCfg;
    extractorCfg.fps = fps;
    dataset::ClipExtractor extractor(extractorCfg);
    auto extraction = extractor.extractFromSegments(fullClip.frames, fullClip.segments, 0);
    double extractMs = stageTime();
    std::cout << " " << extraction.clips.size() << " clips (" << std::setprecision(1)
              << extractMs << " ms)\n";

    // ── Stage 8: Quality filtering ──────────────────────────────────
    std::cout << "  [ 8/10] Quality filtering...";
    dataset::ClipQualityFilter qualityFilter;
    auto filtered = qualityFilter.filter(extraction.clips);
    double qualMs = stageTime();
    std::cout << " " << filtered.accepted.size() << " accepted, "
              << filtered.rejected.size() << " rejected (" << std::setprecision(1)
              << qualMs << " ms)\n";

    // ── Stage 9: Classification + fingerprinting ────────────────────
    std::cout << "  [ 9/10] Classification + fingerprinting...";
    dataset::MotionClassifier classifier;
    classifier.initialize();
    auto classifications = classifier.classifyBatch(filtered.accepted);

    analysis::MotionFingerprintConfig fpCfg;
    analysis::MotionFingerprint fingerprinter(fpCfg);

    dataset::AnimationDatabase database;
    for (size_t i = 0; i < filtered.accepted.size(); ++i) {
        dataset::AnimationEntry entry;
        entry.clip = std::move(filtered.accepted[i]);
        entry.quality = filtered.acceptedResults[i];
        entry.classification = classifications[i];

        // Metadata
        entry.clipMeta.playerID = 0;
        entry.clipMeta.durationSec = entry.clip.frames.empty() ? 0.0f :
            static_cast<float>(entry.clip.frames.size() - 1) / fps;

        if (i < extraction.metadata.size()) {
            entry.clipMeta.startFrame = extraction.metadata[i].startFrame;
            entry.clipMeta.endFrame = extraction.metadata[i].endFrame;
            entry.clipMeta.avgVelocity = extraction.metadata[i].avgVelocity;
            entry.clipMeta.maxVelocity = extraction.metadata[i].maxVelocity;
        }

        // Fingerprint
        auto fp = fingerprinter.compute(entry.clip);
        (void)fp;

        database.addEntry(std::move(entry));
    }
    double classMs = stageTime();
    std::cout << " " << database.entries().size() << " entries (" << std::setprecision(1)
              << classMs << " ms)\n";

    // ── Stage 10: Export ────────────────────────────────────────────
    std::cout << "  [10/10] Exporting to " << outputDir << "...";
    std::filesystem::create_directories(outputDir);
    int exported = database.exportToDirectory(outputDir, true, true);
    database.saveSummary(outputDir + "/database_summary.json");

    // Also export the full unclipped animation
    xport::JSONExportConfig jsonCfg;
    jsonCfg.includeMetadata = true;
    jsonCfg.includePositions = true;
    jsonCfg.includeQuaternions = true;
    jsonCfg.includeSegments = true;
    jsonCfg.prettyPrint = true;
    xport::JSONExporter jsonExporter(jsonCfg);
    jsonExporter.exportToFile(fullClip, outputDir + "/full_sequence.json");

    xport::BVHExportConfig bvhCfg;
    bvhCfg.exportRootMotion = true;
    bvhCfg.useEndSites = true;
    xport::BVHExporter bvhExporter(bvhCfg);
    bvhExporter.exportToFile(fullClip, outputDir + "/full_sequence.bvh");

    double exportMs = stageTime();
    std::cout << " " << exported << " clips + full sequence (" << std::setprecision(1)
              << exportMs << " ms)\n";

    // ── Summary ─────────────────────────────────────────────────────
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    auto dbStats = database.stats();

    std::cout << "\n  ┌─────────────────────────────────────────────┐\n";
    std::cout << "  │              Pipeline Summary                │\n";
    std::cout << "  ├─────────────────────────────────────────────┤\n";
    std::cout << "  │  Input frames:      " << std::setw(7) << fullClip.frames.size() << "               │\n";
    std::cout << "  │  Joints:            " << std::setw(7) << JOINT_COUNT << "               │\n";
    std::cout << "  │  Segments found:    " << std::setw(7) << fullClip.segments.size() << "               │\n";
    std::cout << "  │  Clips extracted:   " << std::setw(7) << extraction.clips.size() << "               │\n";
    std::cout << "  │  Quality accepted:  " << std::setw(7) << filtered.accepted.size() << "               │\n";
    std::cout << "  │  Clips exported:    " << std::setw(7) << exported << "               │\n";
    std::cout << "  │  Database entries:  " << std::setw(7) << dbStats.totalClips << "               │\n";
    std::cout << "  │  Unique players:    " << std::setw(7) << dbStats.uniquePlayers << "               │\n";
    std::cout << "  │  Foot contact frs:  " << std::setw(7) << contactFrames << "               │\n";
    std::cout << "  ├─────────────────────────────────────────────┤\n";
    std::cout << "  │  Motion type distribution:                  │\n";

    for (int i = 0; i < MOTION_TYPE_COUNT; ++i) {
        if (dbStats.clipsByType[i] > 0) {
            std::string name = MOTION_TYPE_NAMES[i];
            std::cout << "  │    " << std::left << std::setw(15) << name
                      << std::right << std::setw(5) << dbStats.clipsByType[i] << " clips"
                      << std::string(15, ' ') << "│\n";
        }
    }

    std::cout << "  ├─────────────────────────────────────────────┤\n";

    if (verbose) {
        std::cout << "  │  Stage timing (ms):                        │\n";
        std::cout << "  │    Generation:   " << std::setw(8) << std::setprecision(1) << genMs << "                 │\n";
        std::cout << "  │    Signal proc:  " << std::setw(8) << sigMs << "                 │\n";
        std::cout << "  │    Canonical:    " << std::setw(8) << canonMs << "                 │\n";
        std::cout << "  │    Segmentation: " << std::setw(8) << segMs << "                 │\n";
        std::cout << "  │    Foot contact: " << std::setw(8) << footMs << "                 │\n";
        std::cout << "  │    Trajectory:   " << std::setw(8) << trajMs << "                 │\n";
        std::cout << "  │    Clip extract: " << std::setw(8) << extractMs << "                 │\n";
        std::cout << "  │    Quality:      " << std::setw(8) << qualMs << "                 │\n";
        std::cout << "  │    Classify/FP:  " << std::setw(8) << classMs << "                 │\n";
        std::cout << "  │    Export:       " << std::setw(8) << exportMs << "                 │\n";
        std::cout << "  ├─────────────────────────────────────────────┤\n";
    }

    std::cout << "  │  Total elapsed:     " << std::setw(7) << static_cast<int>(totalMs) << " ms"
              << "            │\n";
    std::cout << "  │  Schema version:    " << std::setw(7) << HM_SCHEMA_VERSION << "               │\n";
    std::cout << "  └─────────────────────────────────────────────┘\n\n";

    return 0;
}
