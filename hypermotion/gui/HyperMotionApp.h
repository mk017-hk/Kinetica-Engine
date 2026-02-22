#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/Pipeline.h"

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <filesystem>

// Forward declarations — imgui/GLFW/OpenGL are included only in the .cpp
struct GLFWwindow;
struct ImFont;

namespace hm::gui {

// -------------------------------------------------------------------
// Video file entry in the queue
// -------------------------------------------------------------------
struct VideoEntry {
    std::string path;
    std::string filename;
    std::string extension;
    int64_t fileSizeBytes = 0;

    enum class Status { Queued, Processing, Complete, Error };
    Status status = Status::Queued;

    float progress = 0.0f;
    std::string progressMessage;
    std::string errorMessage;

    // Results
    int numPersonsDetected = 0;
    int numClipsExtracted = 0;
    int totalFramesProcessed = 0;
    float processingTimeSec = 0.0f;
    std::vector<AnimClip> clips;
};

// -------------------------------------------------------------------
// Application log message
// -------------------------------------------------------------------
struct LogMessage {
    enum class Level { Info, Warn, Error, Success };
    Level level;
    std::string text;
    double timestamp;
};

// -------------------------------------------------------------------
// Model configuration panel state
// -------------------------------------------------------------------
struct ModelConfig {
    char detectorModelPath[512] = "";
    char poseModelPath[512] = "";
    char depthModelPath[512] = "";
    char segmenterModelPath[512] = "";
    char outputDirectory[512] = "output";
    float targetFPS = 30.0f;
    bool splitBySegment = true;
    bool enableVisualization = false;
    int outputFormatIdx = 0;  // 0=JSON, 1=BVH, 2=Both
};

// -------------------------------------------------------------------
// Main GUI Application
// -------------------------------------------------------------------
class HyperMotionApp {
public:
    HyperMotionApp();
    ~HyperMotionApp();

    HyperMotionApp(const HyperMotionApp&) = delete;
    HyperMotionApp& operator=(const HyperMotionApp&) = delete;

    // Initialize GLFW window, OpenGL context, Dear ImGui
    bool initialize(int width = 1400, int height = 900);

    // Main loop — returns when window closes
    void run();

    // Shutdown
    void shutdown();

    // Add video files (called from drag-and-drop or file browser)
    void addVideoFiles(const std::vector<std::string>& paths);

private:
    // --- Rendering ---
    void beginFrame();
    void endFrame();
    void renderUI();

    // --- UI Panels ---
    void renderMenuBar();
    void renderToolbar();
    void renderVideoQueuePanel();
    void renderVideoPreviewPanel();
    void renderResultsPanel();
    void renderLogPanel();
    void renderSettingsPanel();
    void renderDropOverlay();
    void renderAboutPopup();

    // --- File Operations ---
    void openFileBrowser();
    void openFolderBrowser();
    bool isVideoFile(const std::string& path) const;
    std::string formatFileSize(int64_t bytes) const;

    // --- Pipeline ---
    void startProcessing();
    void stopProcessing();
    void processNextVideo();
    void onPipelineProgress(float percent, const std::string& message);
    void onPipelineComplete(int videoIdx);
    void onPipelineError(int videoIdx, const std::string& error);

    // --- Video Preview ---
    void loadVideoPreview(const std::string& path);
    void updatePreviewFrame();
    void seekPreview(float normalizedPos);

    // --- Log ---
    void addLog(LogMessage::Level level, const std::string& text);

    // --- Drag-and-drop callbacks (static, forwarded to instance) ---
    static void glfwDropCallback(GLFWwindow* window, int count, const char** paths);
    static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // --- State ---
    GLFWwindow* window_ = nullptr;
    ImFont* fontRegular_ = nullptr;
    ImFont* fontBold_ = nullptr;
    ImFont* fontMono_ = nullptr;

    // Video queue
    std::vector<VideoEntry> videoQueue_;
    int selectedVideoIdx_ = -1;
    std::mutex queueMutex_;

    // Processing
    std::atomic<bool> isProcessing_{false};
    std::atomic<bool> stopRequested_{false};
    std::thread processingThread_;
    int currentProcessingIdx_ = -1;

    // Video preview
    unsigned int previewTextureID_ = 0;
    int previewWidth_ = 0;
    int previewHeight_ = 0;
    int previewTotalFrames_ = 0;
    int previewCurrentFrame_ = 0;
    float previewFPS_ = 30.0f;
    bool previewPlaying_ = false;
    double previewLastFrameTime_ = 0.0;
    bool hasPreview_ = false;
    std::vector<uint8_t> previewPixelBuffer_;

    // Settings / Config
    ModelConfig modelConfig_;
    bool showSettings_ = false;
    bool showAbout_ = false;

    // Log
    std::deque<LogMessage> logMessages_;
    std::mutex logMutex_;
    static constexpr int MAX_LOG_MESSAGES = 500;

    // Drag overlay
    bool showDropOverlay_ = false;
    double dropOverlayTimer_ = 0.0;

    // Supported video extensions
    static constexpr const char* VIDEO_EXTENSIONS[] = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"
    };
};

} // namespace hm::gui
