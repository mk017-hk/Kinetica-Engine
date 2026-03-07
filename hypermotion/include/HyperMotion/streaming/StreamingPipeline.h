#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/Pipeline.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hm::streaming {

/// Configuration for the asynchronous streaming pipeline.
struct StreamingPipelineConfig {
    PipelineConfig pipelineConfig;

    /// Ring buffer capacity (number of frames).
    int frameBufferSize = 120;

    /// Number of decode worker threads.
    int decodeThreads = 1;

    /// Number of pose inference threads.
    int inferenceThreads = 1;

    /// Number of motion analysis threads.
    int analysisThreads = 1;

    /// Maximum queue depth before back-pressuring the decoder.
    int maxQueueDepth = 64;

    /// If true, drop frames when queues are full instead of blocking.
    bool allowFrameDrop = false;
};

/// Status of the streaming pipeline.
struct StreamingStats {
    int framesDecoded = 0;
    int framesInferred = 0;
    int framesAnalysed = 0;
    int framesDropped = 0;
    int clipsProduced = 0;
    double decodeLatencyMs = 0.0;
    double inferenceLatencyMs = 0.0;
    double analysisLatencyMs = 0.0;
    bool isRunning = false;
    bool isFinished = false;
};

using StreamingClipCallback =
    std::function<void(AnimClip clip, int playerID)>;

using StreamingProgressCallback =
    std::function<void(const StreamingStats& stats)>;

/// Asynchronous frame processing pipeline that decouples video decoding,
/// pose inference, and motion analysis into concurrent stages.
///
/// Architecture:
///   Decode Thread → [Frame Queue] → Inference Thread → [Pose Queue]
///       → Analysis Thread → [Clip Output]
///
/// Each stage runs in its own thread, communicating via thread-safe queues.
/// This allows GPU inference to overlap with CPU-side video decoding and
/// motion analysis, maximising hardware utilisation.
class StreamingPipeline {
public:
    explicit StreamingPipeline(const StreamingPipelineConfig& config);
    ~StreamingPipeline();

    StreamingPipeline(const StreamingPipeline&) = delete;
    StreamingPipeline& operator=(const StreamingPipeline&) = delete;
    StreamingPipeline(StreamingPipeline&&) noexcept;
    StreamingPipeline& operator=(StreamingPipeline&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// Start processing a video file asynchronously.
    /// Clips are delivered via the callback as they are completed.
    bool startProcessing(const std::string& videoPath,
                         StreamingClipCallback clipCallback = nullptr,
                         StreamingProgressCallback progressCallback = nullptr);

    /// Block until all processing is complete.
    /// Returns all produced clips.
    std::vector<AnimClip> waitForCompletion();

    /// Request graceful shutdown (may be called from any thread).
    void requestStop();

    /// Check whether the pipeline is still running.
    bool isRunning() const;

    /// Get current statistics.
    StreamingStats getStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::streaming
