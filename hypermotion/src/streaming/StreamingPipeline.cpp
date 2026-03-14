#include "HyperMotion/streaming/StreamingPipeline.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/ScopedTimer.h"
#include "HyperMotion/pose/PoseBatchProcessor.h"
#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/segmenter/MotionSegmenter.h"
#include "HyperMotion/motion/FootContactDetector.h"
#include "HyperMotion/motion/TrajectoryExtractor.h"
#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/analysis/MotionFingerprint.h"
#include "HyperMotion/export/BVHExporter.h"
#include "HyperMotion/export/JSONExporter.h"

#include <opencv2/videoio.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <thread>

namespace hm::streaming {

static constexpr const char* TAG = "StreamingPipeline";

// Thread-safe bounded queue
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(int capacity) : capacity_(capacity) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [&] { return static_cast<int>(queue_.size()) < capacity_ || done_; });
        if (done_) return false;
        queue_.push_back(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    bool tryPush(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(queue_.size()) >= capacity_) return false;
        queue_.push_back(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [&] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop_front();
        cv_push_.notify_one();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_pop_.notify_all();
        cv_push_.notify_all();
    }

    bool isDone() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_ && queue_.empty();
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(queue_.size());
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        done_ = false;
    }

private:
    int capacity_;
    std::deque<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    bool done_ = false;
};

struct DecodedFrame {
    cv::Mat image;
    double timestamp = 0.0;
    int frameIndex = 0;
};

struct StreamingPipeline::Impl {
    StreamingPipelineConfig config;
    bool initialized = false;

    // Pipeline components
    std::unique_ptr<pose::PoseBatchProcessor> batchProcessor;
    skeleton::SkeletonMapper skeletonMapper;
    signal::SignalPipeline signalPipeline;
    segmenter::MotionSegmenter motionSegmenter;
    motion::FootContactDetector footContactDetector;
    motion::TrajectoryExtractor trajectoryExtractor;
    motion::CanonicalMotionBuilder canonicalBuilder;
    analysis::MotionFingerprint fingerprinter;

    // Queues
    std::unique_ptr<BoundedQueue<DecodedFrame>> frameQueue;
    std::unique_ptr<BoundedQueue<PoseFrameResult>> poseQueue;

    // Threads
    std::thread decodeThread;
    std::thread inferenceThread;
    std::thread analysisThread;

    // State
    std::atomic<bool> stopRequested{false};
    StreamingStats stats;
    mutable std::mutex statsMutex;

    // Output
    StreamingClipCallback clipCallback;
    StreamingProgressCallback progressCallback;
    std::mutex clipsMutex;
    std::vector<AnimClip> completedClips;

    Impl(const StreamingPipelineConfig& cfg)
        : config(cfg)
        , skeletonMapper(cfg.pipelineConfig.mapperConfig)
        , signalPipeline(cfg.pipelineConfig.signalConfig)
        , motionSegmenter(cfg.pipelineConfig.segmenterConfig)
        , footContactDetector(cfg.pipelineConfig.footContactConfig)
        , trajectoryExtractor(cfg.pipelineConfig.trajectoryConfig)
        , canonicalBuilder(cfg.pipelineConfig.canonicalConfig)
        , fingerprinter(cfg.pipelineConfig.fingerprintConfig) {

        pose::PoseBatchProcessorConfig batchCfg;
        batchCfg.estimatorConfig.detector = cfg.pipelineConfig.poseConfig.detector;
        batchCfg.estimatorConfig.poseModel = cfg.pipelineConfig.poseConfig.poseEstimator;
        batchCfg.estimatorConfig.depthLifter = cfg.pipelineConfig.poseConfig.depthLifter;
        batchCfg.batchSize = 8;
        batchProcessor = std::make_unique<pose::PoseBatchProcessor>(batchCfg);

        int queueDepth = cfg.maxQueueDepth > 0 ? cfg.maxQueueDepth : 64;
        frameQueue = std::make_unique<BoundedQueue<DecodedFrame>>(queueDepth);
        poseQueue = std::make_unique<BoundedQueue<PoseFrameResult>>(queueDepth);
    }

    void updateStats(std::function<void(StreamingStats&)> fn) {
        std::lock_guard<std::mutex> lock(statsMutex);
        fn(stats);
    }
};

StreamingPipeline::StreamingPipeline(const StreamingPipelineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

StreamingPipeline::~StreamingPipeline() {
    requestStop();
    if (impl_->decodeThread.joinable()) impl_->decodeThread.join();
    if (impl_->inferenceThread.joinable()) impl_->inferenceThread.join();
    if (impl_->analysisThread.joinable()) impl_->analysisThread.join();
}

StreamingPipeline::StreamingPipeline(StreamingPipeline&&) noexcept = default;
StreamingPipeline& StreamingPipeline::operator=(StreamingPipeline&&) noexcept = default;

bool StreamingPipeline::initialize() {
    HM_LOG_INFO(TAG, "Initializing streaming pipeline...");

    if (!impl_->batchProcessor->initialize()) {
        HM_LOG_WARN(TAG, "Batch processor init had warnings");
    }

    impl_->motionSegmenter.initialize();
    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Streaming pipeline initialized");
    return true;
}

bool StreamingPipeline::isInitialized() const {
    return impl_->initialized;
}

bool StreamingPipeline::startProcessing(
    const std::string& videoPath,
    StreamingClipCallback clipCallback,
    StreamingProgressCallback progressCallback) {

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Pipeline not initialized");
        return false;
    }

    // Join any previously running threads before starting new ones
    if (impl_->decodeThread.joinable()) impl_->decodeThread.join();
    if (impl_->inferenceThread.joinable()) impl_->inferenceThread.join();
    if (impl_->analysisThread.joinable()) impl_->analysisThread.join();

    // Reset queues for reuse
    impl_->frameQueue->reset();
    impl_->poseQueue->reset();

    // Clear previous results
    {
        std::lock_guard<std::mutex> lock(impl_->clipsMutex);
        impl_->completedClips.clear();
    }

    impl_->clipCallback = std::move(clipCallback);
    impl_->progressCallback = std::move(progressCallback);
    impl_->stopRequested = false;

    impl_->updateStats([](StreamingStats& s) {
        s = StreamingStats{};
        s.isRunning = true;
    });

    // Stage 1: Decode thread
    impl_->decodeThread = std::thread([this, videoPath]() {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            HM_LOG_ERROR(TAG, "Failed to open video: " + videoPath);
            impl_->frameQueue->finish();
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        cv::Mat frame;
        int frameIdx = 0;

        while (cap.read(frame) && !impl_->stopRequested) {
            DecodedFrame df;
            df.image = frame.clone();
            df.timestamp = frameIdx / fps;
            df.frameIndex = frameIdx;

            if (impl_->config.allowFrameDrop) {
                if (!impl_->frameQueue->tryPush(std::move(df))) {
                    impl_->updateStats([](StreamingStats& s) { s.framesDropped++; });
                }
            } else {
                impl_->frameQueue->push(std::move(df));
            }

            impl_->updateStats([](StreamingStats& s) { s.framesDecoded++; });
            frameIdx++;
        }

        impl_->frameQueue->finish();
        HM_LOG_INFO(TAG, "Decode complete: " + std::to_string(frameIdx) + " frames");
    });

    // Stage 2: Inference thread
    impl_->inferenceThread = std::thread([this]() {
        DecodedFrame df;
        while (impl_->frameQueue->pop(df) && !impl_->stopRequested) {
            double latency = 0;
            PoseFrameResult result;
            {
                ScopedTimer t(latency);
                impl_->batchProcessor->addFrame(df.image, df.timestamp, df.frameIndex);
                auto batch = impl_->batchProcessor->processBatch();
                if (!batch.empty()) {
                    for (auto& r : batch) {
                        impl_->poseQueue->push(std::move(r));
                    }
                }
            }
            impl_->updateStats([latency](StreamingStats& s) {
                s.framesInferred++;
                s.inferenceLatencyMs = latency;
            });
        }

        // Flush remaining
        auto remaining = impl_->batchProcessor->flush();
        for (auto& r : remaining) {
            impl_->poseQueue->push(std::move(r));
        }

        impl_->poseQueue->finish();
        HM_LOG_INFO(TAG, "Inference complete");
    });

    // Stage 3: Analysis thread
    impl_->analysisThread = std::thread([this]() {
        // Accumulate pose results per tracked person
        std::map<int, std::vector<PoseFrameResult>> perPersonFrames;
        PoseFrameResult pfr;

        while (impl_->poseQueue->pop(pfr) && !impl_->stopRequested) {
            // Group by person ID
            for (const auto& person : pfr.persons) {
                PoseFrameResult singleResult;
                singleResult.timestamp = pfr.timestamp;
                singleResult.frameIndex = pfr.frameIndex;
                singleResult.videoWidth = pfr.videoWidth;
                singleResult.videoHeight = pfr.videoHeight;
                singleResult.persons = {person};
                perPersonFrames[person.id].push_back(std::move(singleResult));
            }

            impl_->updateStats([](StreamingStats& s) { s.framesAnalysed++; });

            // Report progress periodically
            if (impl_->progressCallback) {
                StreamingStats snapshot;
                {
                    std::lock_guard<std::mutex> lock(impl_->statsMutex);
                    snapshot = impl_->stats;
                }
                impl_->progressCallback(snapshot);
            }
        }

        // Build clips for each tracked person
        int minFrames = impl_->config.pipelineConfig.minTrackFrames;

        for (auto& [personID, poseResults] : perPersonFrames) {
            auto skeletonFrames = impl_->skeletonMapper.mapSequence(poseResults, personID);

            if (static_cast<int>(skeletonFrames.size()) < minFrames) continue;

            impl_->signalPipeline.process(skeletonFrames);

            AnimClip clip;
            clip.name = "player_" + std::to_string(personID);
            clip.fps = impl_->config.pipelineConfig.targetFPS;
            clip.trackingID = personID;
            clip.frames = std::move(skeletonFrames);

            // Canonical motion (stabilise limbs, extract root motion)
            if (impl_->config.pipelineConfig.enableCanonicalMotion) {
                impl_->canonicalBuilder.process(clip);
            }

            auto segments = impl_->motionSegmenter.segment(clip.frames, personID);
            clip.segments = std::move(segments);

            impl_->footContactDetector.process(clip);

            if (impl_->config.pipelineConfig.enableTrajectoryExtraction) {
                impl_->trajectoryExtractor.process(clip);
            }

            // Motion fingerprinting
            if (impl_->config.pipelineConfig.enableFingerprinting && !clip.frames.empty()) {
                impl_->fingerprinter.compute(clip);
            }

            if (impl_->clipCallback) {
                impl_->clipCallback(clip, personID);
            }

            {
                std::lock_guard<std::mutex> lock(impl_->clipsMutex);
                impl_->completedClips.push_back(std::move(clip));
            }

            impl_->updateStats([](StreamingStats& s) { s.clipsProduced++; });
        }

        impl_->updateStats([](StreamingStats& s) {
            s.isRunning = false;
            s.isFinished = true;
        });

        HM_LOG_INFO(TAG, "Analysis complete");
    });

    return true;
}

std::vector<AnimClip> StreamingPipeline::waitForCompletion() {
    if (impl_->decodeThread.joinable()) impl_->decodeThread.join();
    if (impl_->inferenceThread.joinable()) impl_->inferenceThread.join();
    if (impl_->analysisThread.joinable()) impl_->analysisThread.join();

    std::lock_guard<std::mutex> lock(impl_->clipsMutex);
    return std::move(impl_->completedClips);
}

void StreamingPipeline::requestStop() {
    impl_->stopRequested = true;
    impl_->frameQueue->finish();
    impl_->poseQueue->finish();
}

bool StreamingPipeline::isRunning() const {
    std::lock_guard<std::mutex> lock(impl_->statsMutex);
    return impl_->stats.isRunning;
}

StreamingStats StreamingPipeline::getStats() const {
    std::lock_guard<std::mutex> lock(impl_->statsMutex);
    return impl_->stats;
}

} // namespace hm::streaming
