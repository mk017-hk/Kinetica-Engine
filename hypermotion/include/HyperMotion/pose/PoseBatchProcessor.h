#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/pose/PoseEstimator.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace hm::pose {

/// Configuration for GPU-friendly batch processing of pose inference.
struct PoseBatchProcessorConfig {
    PoseEstimatorConfig estimatorConfig;

    /// Number of frames to accumulate before running a batched inference pass.
    int batchSize = 8;

    /// Maximum number of persons to process per frame.
    int maxPersonsPerFrame = 22;

    /// Number of worker threads for frame pre-processing (crop / resize).
    int preprocessThreads = 2;
};

using BatchProgressCallback =
    std::function<void(float percent, const std::string& message)>;

/// Batches video frames for efficient GPU pose inference.
///
/// Frames are accumulated into batches of `batchSize`. Once a batch is full,
/// detection runs on each frame, person crops are batched together, and 2D
/// pose estimation runs once on the full batch of crops. This amortises GPU
/// kernel launch overhead and saturates tensor cores.
class PoseBatchProcessor {
public:
    explicit PoseBatchProcessor(const PoseBatchProcessorConfig& config = {});
    ~PoseBatchProcessor();

    PoseBatchProcessor(const PoseBatchProcessor&) = delete;
    PoseBatchProcessor& operator=(const PoseBatchProcessor&) = delete;
    PoseBatchProcessor(PoseBatchProcessor&&) noexcept;
    PoseBatchProcessor& operator=(PoseBatchProcessor&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// Add a single frame to the internal buffer.
    /// Returns true if a batch is now ready for processing.
    bool addFrame(const cv::Mat& frame, double timestamp, int frameIndex);

    /// Flush the current (possibly partial) batch.
    /// Returns results for all buffered frames.
    std::vector<PoseFrameResult> flush();

    /// Process all buffered frames whose batch is complete.
    /// Returns results for the completed batch(es).  Any leftover frames
    /// remain buffered until the next addFrame() or flush().
    std::vector<PoseFrameResult> processBatch();

    /// Convenience: process an entire video file in batched mode.
    std::vector<PoseFrameResult> processVideo(
        const std::string& videoPath,
        BatchProgressCallback callback = nullptr);

    /// Convert model outputs to unified skeleton format.
    /// Applies depth lifting and maps raw COCO keypoints into 3D.
    static std::vector<DetectedPerson> toUnifiedSkeleton(
        const std::vector<DetectedPerson>& raw2D,
        DepthLifter& lifter);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
