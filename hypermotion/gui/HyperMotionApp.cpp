#include "HyperMotionApp.h"
#include "HyperMotion/core/Logger.h"

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// GLFW / OpenGL
#include <GLFW/glfw3.h>

// OpenCV for video decoding
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// Native file dialog
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#else
#include <cstdlib>
#include <cstdio>
#include <array>
#endif

#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <set>

namespace hm::gui {

// Global pointer for GLFW callbacks
static HyperMotionApp* g_appInstance = nullptr;

// ===================================================================
// Construction / Destruction
// ===================================================================

HyperMotionApp::HyperMotionApp() {
    g_appInstance = this;
}

HyperMotionApp::~HyperMotionApp() {
    shutdown();
    g_appInstance = nullptr;
}

// ===================================================================
// Initialize
// ===================================================================

bool HyperMotionApp::initialize(int width, int height) {
    // --- GLFW ---
    if (!glfwInit()) {
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(width, height, "HyperMotion Studio", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // VSync

    // Register drag-and-drop callback
    glfwSetDropCallback(window_, glfwDropCallback);
    glfwSetKeyCallback(window_, glfwKeyCallback);
    glfwSetWindowUserPointer(window_, this);

    // --- Dear ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(8, 5);
    style.FramePadding = ImVec2(6, 4);

    // Custom dark theme with accent color
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.36f, 0.56f, 0.60f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.46f, 0.70f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.46f, 0.70f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.36f, 0.56f, 0.65f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.46f, 0.70f, 0.85f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.52f, 0.80f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.46f, 0.70f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.36f, 0.56f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.52f, 0.80f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.46f, 0.70f, 0.70f);

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Create OpenGL texture for video preview
    glGenTextures(1, &previewTextureID_);

    // Hook the Logger so messages appear in the GUI log
    Logger::instance().setCallback([this](LogLevel level, const std::string& tag,
                                          const std::string& message) {
        LogMessage::Level guiLevel = LogMessage::Level::Info;
        if (level >= LogLevel::Error) guiLevel = LogMessage::Level::Error;
        else if (level >= LogLevel::Warn) guiLevel = LogMessage::Level::Warn;
        addLog(guiLevel, "[" + tag + "] " + message);
    });

    addLog(LogMessage::Level::Success, "HyperMotion Studio initialized");
    addLog(LogMessage::Level::Info, "Drag and drop video files onto the window, or use File > Open");

    return true;
}

// ===================================================================
// Main Loop
// ===================================================================

void HyperMotionApp::run() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        beginFrame();
        renderUI();
        endFrame();

        // Update video preview playback
        if (previewPlaying_ && hasPreview_) {
            double now = glfwGetTime();
            if (now - previewLastFrameTime_ >= 1.0 / previewFPS_) {
                updatePreviewFrame();
                previewLastFrameTime_ = now;
            }
        }

        // Fade out drop overlay
        if (showDropOverlay_) {
            double now = glfwGetTime();
            if (now - dropOverlayTimer_ > 1.5) {
                showDropOverlay_ = false;
            }
        }
    }
}

void HyperMotionApp::shutdown() {
    stopProcessing();

    if (processingThread_.joinable()) {
        processingThread_.join();
    }

    if (previewTextureID_ != 0) {
        glDeleteTextures(1, &previewTextureID_);
        previewTextureID_ = 0;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void HyperMotionApp::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void HyperMotionApp::endFrame() {
    ImGui::Render();
    int displayW, displayH;
    glfwGetFramebufferSize(window_, &displayW, &displayH);
    glViewport(0, 0, displayW, displayH);
    glClearColor(0.08f, 0.08f, 0.10f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

// ===================================================================
// Top-Level UI Render
// ===================================================================

void HyperMotionApp::renderUI() {
    // Full-window dockspace
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags dockspaceFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::Begin("MainDockspace", nullptr, dockspaceFlags);
    ImGui::PopStyleVar(2);

    renderMenuBar();

    ImGuiID dockspaceID = ImGui::GetID("HyperMotionDockspace");
    ImGui::DockSpace(dockspaceID, ImVec2(0, 0), ImGuiDockNodeFlags_None);

    ImGui::End();

    // --- Panels ---
    renderToolbar();
    renderVideoQueuePanel();
    renderVideoPreviewPanel();
    renderResultsPanel();
    renderLogPanel();

    if (showSettings_) renderSettingsPanel();
    if (showAbout_) renderAboutPopup();
    if (showDropOverlay_) renderDropOverlay();
}

// ===================================================================
// Menu Bar
// ===================================================================

void HyperMotionApp::renderMenuBar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Video Files...", "Ctrl+O")) {
                openFileBrowser();
            }
            if (ImGui::MenuItem("Open Folder...", "Ctrl+Shift+O")) {
                openFolderBrowser();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Queue")) {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (!isProcessing_) {
                    videoQueue_.clear();
                    selectedVideoIdx_ = -1;
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Process")) {
            bool canStart = !videoQueue_.empty() && !isProcessing_;
            if (ImGui::MenuItem("Start Processing", "F5", false, canStart)) {
                startProcessing();
            }
            if (ImGui::MenuItem("Stop Processing", "Esc", false, isProcessing_.load())) {
                stopProcessing();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Settings", "Ctrl+,", &showSettings_);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About HyperMotion")) {
                showAbout_ = true;
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

// ===================================================================
// Toolbar
// ===================================================================

void HyperMotionApp::renderToolbar() {
    ImGui::Begin("Toolbar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar);

    // Open files button
    if (ImGui::Button("Open Files")) {
        openFileBrowser();
    }
    ImGui::SameLine();

    if (ImGui::Button("Open Folder")) {
        openFolderBrowser();
    }
    ImGui::SameLine();

    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Process buttons
    bool canStart = !videoQueue_.empty() && !isProcessing_;
    ImGui::BeginDisabled(!canStart);
    ImVec4 startColor(0.2f, 0.65f, 0.3f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, startColor);
    if (ImGui::Button("Start Processing")) {
        startProcessing();
    }
    ImGui::PopStyleColor();
    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::BeginDisabled(!isProcessing_.load());
    ImVec4 stopColor(0.75f, 0.2f, 0.2f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, stopColor);
    if (ImGui::Button("Stop")) {
        stopProcessing();
    }
    ImGui::PopStyleColor();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Status display
    if (isProcessing_) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Processing...");
        if (currentProcessingIdx_ >= 0 && currentProcessingIdx_ < static_cast<int>(videoQueue_.size())) {
            ImGui::SameLine();
            std::lock_guard<std::mutex> lock(queueMutex_);
            ImGui::ProgressBar(videoQueue_[currentProcessingIdx_].progress, ImVec2(200, 0));
        }
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Ready");
    }

    ImGui::SameLine();
    float rightEdge = ImGui::GetWindowContentRegionMax().x;
    std::string queueInfo = std::to_string(videoQueue_.size()) + " video(s) in queue";
    float textWidth = ImGui::CalcTextSize(queueInfo.c_str()).x;
    ImGui::SetCursorPosX(rightEdge - textWidth);
    ImGui::Text("%s", queueInfo.c_str());

    ImGui::End();
}

// ===================================================================
// Video Queue Panel — list of loaded videos with status
// ===================================================================

void HyperMotionApp::renderVideoQueuePanel() {
    ImGui::Begin("Video Queue");

    // Drop zone hint when queue is empty
    if (videoQueue_.empty()) {
        ImVec2 availSize = ImGui::GetContentRegionAvail();
        float centerY = availSize.y * 0.4f;

        ImGui::SetCursorPosY(centerY);

        const char* hint = "Drag & drop video files here";
        float textW = ImGui::CalcTextSize(hint).x;
        ImGui::SetCursorPosX((availSize.x - textW) * 0.5f);
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.6f, 1.0f), "%s", hint);

        const char* hint2 = "or use File > Open";
        float textW2 = ImGui::CalcTextSize(hint2).x;
        ImGui::SetCursorPosX((availSize.x - textW2) * 0.5f);
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.5f, 1.0f), "%s", hint2);

        const char* hint3 = "Supported: MP4, AVI, MOV, MKV, WebM";
        float textW3 = ImGui::CalcTextSize(hint3).x;
        ImGui::SetCursorPosX((availSize.x - textW3) * 0.5f);
        ImGui::TextColored(ImVec4(0.35f, 0.35f, 0.42f, 1.0f), "%s", hint3);
    }

    std::lock_guard<std::mutex> lock(queueMutex_);

    for (int i = 0; i < static_cast<int>(videoQueue_.size()); ++i) {
        auto& entry = videoQueue_[i];

        ImGui::PushID(i);

        bool isSelected = (selectedVideoIdx_ == i);

        // Status color
        ImVec4 statusColor;
        const char* statusIcon;
        switch (entry.status) {
            case VideoEntry::Status::Queued:
                statusColor = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
                statusIcon = "[Queued]";
                break;
            case VideoEntry::Status::Processing:
                statusColor = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
                statusIcon = "[Processing]";
                break;
            case VideoEntry::Status::Complete:
                statusColor = ImVec4(0.3f, 0.9f, 0.3f, 1.0f);
                statusIcon = "[Done]";
                break;
            case VideoEntry::Status::Error:
                statusColor = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                statusIcon = "[Error]";
                break;
        }

        // Selectable row
        if (ImGui::Selectable("##entry", isSelected, ImGuiSelectableFlags_SpanAllColumns, ImVec2(0, 40))) {
            selectedVideoIdx_ = i;
            loadVideoPreview(entry.path);
        }

        ImGui::SameLine();
        ImGui::TextColored(statusColor, "%s", statusIcon);
        ImGui::SameLine();
        ImGui::Text("%s", entry.filename.c_str());

        // File size
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(%s)", formatFileSize(entry.fileSizeBytes).c_str());

        // Progress bar for processing entries
        if (entry.status == VideoEntry::Status::Processing) {
            ImGui::ProgressBar(entry.progress, ImVec2(-1, 3));
        }

        // Result summary for complete entries
        if (entry.status == VideoEntry::Status::Complete) {
            ImGui::TextColored(ImVec4(0.5f, 0.7f, 0.5f, 1.0f),
                "  %d clips, %d persons, %.1fs",
                entry.numClipsExtracted, entry.numPersonsDetected, entry.processingTimeSec);
        }

        // Error message
        if (entry.status == VideoEntry::Status::Error) {
            ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "  %s", entry.errorMessage.c_str());
        }

        // Context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Remove from Queue", nullptr, false, !isProcessing_.load())) {
                videoQueue_.erase(videoQueue_.begin() + i);
                if (selectedVideoIdx_ >= static_cast<int>(videoQueue_.size()))
                    selectedVideoIdx_ = static_cast<int>(videoQueue_.size()) - 1;
                ImGui::EndPopup();
                ImGui::PopID();
                break; // Iterator invalidated
            }
            if (ImGui::MenuItem("Open in File Manager")) {
                std::string dir = std::filesystem::path(entry.path).parent_path().string();
#ifdef _WIN32
                std::system(("explorer \"" + dir + "\"").c_str());
#elif defined(__APPLE__)
                std::system(("open \"" + dir + "\"").c_str());
#else
                std::system(("xdg-open \"" + dir + "\" &").c_str());
#endif
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    ImGui::End();
}

// ===================================================================
// Video Preview Panel — plays selected video with scrub bar
// ===================================================================

void HyperMotionApp::renderVideoPreviewPanel() {
    ImGui::Begin("Video Preview");

    if (!hasPreview_) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        const char* msg = "Select a video from the queue to preview";
        float textW = ImGui::CalcTextSize(msg).x;
        ImGui::SetCursorPos(ImVec2((avail.x - textW) * 0.5f, avail.y * 0.45f));
        ImGui::TextColored(ImVec4(0.45f, 0.45f, 0.5f, 1.0f), "%s", msg);
    } else {
        // Display video frame as texture
        ImVec2 avail = ImGui::GetContentRegionAvail();
        avail.y -= 40; // Reserve space for controls

        float aspect = static_cast<float>(previewWidth_) / std::max(1, previewHeight_);
        float displayW = avail.x;
        float displayH = displayW / aspect;
        if (displayH > avail.y) {
            displayH = avail.y;
            displayW = displayH * aspect;
        }

        float offsetX = (avail.x - displayW) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(previewTextureID_)),
                     ImVec2(displayW, displayH));

        // Playback controls
        ImGui::Separator();

        if (ImGui::Button(previewPlaying_ ? "Pause" : "Play")) {
            previewPlaying_ = !previewPlaying_;
            previewLastFrameTime_ = glfwGetTime();
        }
        ImGui::SameLine();

        // Scrub bar
        float normalizedPos = previewTotalFrames_ > 0 ?
            static_cast<float>(previewCurrentFrame_) / previewTotalFrames_ : 0.0f;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 120);
        if (ImGui::SliderFloat("##scrub", &normalizedPos, 0.0f, 1.0f, "")) {
            seekPreview(normalizedPos);
        }

        ImGui::SameLine();
        ImGui::Text("%d / %d", previewCurrentFrame_, previewTotalFrames_);
    }

    ImGui::End();
}

// ===================================================================
// Results Panel — shows extracted clip details
// ===================================================================

void HyperMotionApp::renderResultsPanel() {
    ImGui::Begin("Results");

    if (selectedVideoIdx_ < 0 || selectedVideoIdx_ >= static_cast<int>(videoQueue_.size())) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No video selected");
        ImGui::End();
        return;
    }

    std::lock_guard<std::mutex> lock(queueMutex_);
    auto& entry = videoQueue_[selectedVideoIdx_];

    ImGui::Text("File: %s", entry.filename.c_str());
    ImGui::Text("Size: %s", formatFileSize(entry.fileSizeBytes).c_str());
    ImGui::Separator();

    if (entry.status == VideoEntry::Status::Complete) {
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "Processing Complete");
        ImGui::Text("Time: %.2f seconds", entry.processingTimeSec);
        ImGui::Text("Persons detected: %d", entry.numPersonsDetected);
        ImGui::Text("Clips extracted: %d", entry.numClipsExtracted);
        ImGui::Text("Total frames: %d", entry.totalFramesProcessed);

        ImGui::Separator();
        ImGui::Text("Extracted Clips:");

        if (ImGui::BeginTable("ClipsTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY)) {
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Frames");
            ImGui::TableSetupColumn("Segments");
            ImGui::TableSetupColumn("Track ID");
            ImGui::TableHeadersRow();

            for (const auto& clip : entry.clips) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", clip.name.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%d", static_cast<int>(clip.frames.size()));
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%d", static_cast<int>(clip.segments.size()));
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%d", clip.trackingID);
            }

            ImGui::EndTable();
        }
    } else if (entry.status == VideoEntry::Status::Processing) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Processing...");
        ImGui::ProgressBar(entry.progress, ImVec2(-1, 0), entry.progressMessage.c_str());
    } else if (entry.status == VideoEntry::Status::Error) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", entry.errorMessage.c_str());
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Queued for processing");
    }

    ImGui::End();
}

// ===================================================================
// Log Panel
// ===================================================================

void HyperMotionApp::renderLogPanel() {
    ImGui::Begin("Log");

    if (ImGui::Button("Clear")) {
        std::lock_guard<std::mutex> lock(logMutex_);
        logMessages_.clear();
    }

    ImGui::Separator();
    ImGui::BeginChild("LogScrollRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    {
        std::lock_guard<std::mutex> lock(logMutex_);
        for (const auto& msg : logMessages_) {
            ImVec4 color;
            switch (msg.level) {
                case LogMessage::Level::Info:    color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f); break;
                case LogMessage::Level::Warn:    color = ImVec4(1.0f, 0.85f, 0.3f, 1.0f); break;
                case LogMessage::Level::Error:   color = ImVec4(1.0f, 0.35f, 0.35f, 1.0f); break;
                case LogMessage::Level::Success: color = ImVec4(0.35f, 0.9f, 0.35f, 1.0f); break;
            }
            ImGui::TextColored(color, "%s", msg.text.c_str());
        }
    }

    // Auto-scroll
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }

    ImGui::EndChild();
    ImGui::End();
}

// ===================================================================
// Settings Panel
// ===================================================================

void HyperMotionApp::renderSettingsPanel() {
    ImGui::Begin("Settings", &showSettings_);

    ImGui::SeparatorText("Model Paths");

    ImGui::InputText("Detector (YOLOv8)", modelConfig_.detectorModelPath,
                     sizeof(modelConfig_.detectorModelPath));
    ImGui::SameLine();
    if (ImGui::Button("...##det")) { /* Would open file dialog for model */ }

    ImGui::InputText("Pose (HRNet)", modelConfig_.poseModelPath,
                     sizeof(modelConfig_.poseModelPath));
    ImGui::SameLine();
    if (ImGui::Button("...##pose")) { /* Would open file dialog for model */ }

    ImGui::InputText("Depth Lifter", modelConfig_.depthModelPath,
                     sizeof(modelConfig_.depthModelPath));

    ImGui::InputText("Segmenter (TCN)", modelConfig_.segmenterModelPath,
                     sizeof(modelConfig_.segmenterModelPath));

    ImGui::SeparatorText("Output");

    ImGui::InputText("Output Directory", modelConfig_.outputDirectory,
                     sizeof(modelConfig_.outputDirectory));

    const char* formatOptions[] = { "JSON", "BVH", "Both" };
    ImGui::Combo("Output Format", &modelConfig_.outputFormatIdx, formatOptions, 3);

    ImGui::SeparatorText("Processing");

    ImGui::SliderFloat("Target FPS", &modelConfig_.targetFPS, 1.0f, 60.0f, "%.0f");
    ImGui::Checkbox("Split clips by motion segment", &modelConfig_.splitBySegment);
    ImGui::Checkbox("Enable visualization output", &modelConfig_.enableVisualization);

    ImGui::End();
}

// ===================================================================
// Drag-and-Drop Overlay
// ===================================================================

void HyperMotionApp::renderDropOverlay() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.3f, 0.6f, 0.3f));

    ImGui::Begin("##DropOverlay", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing);

    ImVec2 center = ImVec2(viewport->WorkSize.x * 0.5f, viewport->WorkSize.y * 0.45f);
    const char* text = "Drop video files here";
    float textW = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPos(ImVec2(center.x - textW * 0.5f, center.y));
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 0.9f), "%s", text);

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ===================================================================
// About Popup
// ===================================================================

void HyperMotionApp::renderAboutPopup() {
    ImGui::Begin("About HyperMotion", &showAbout_, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("HyperMotion Studio v1.0.0");
    ImGui::Separator();
    ImGui::Text("Full Motion Capture & Animation System");
    ImGui::Text("for UE5 Football Games");
    ImGui::Spacing();
    ImGui::Text("Modules:");
    ImGui::BulletText("Multi-Person Pose Estimation (YOLOv8 + HRNet)");
    ImGui::BulletText("Skeleton Mapping & Retargeting");
    ImGui::BulletText("Signal Processing Pipeline");
    ImGui::BulletText("Motion Segmentation (TCN)");
    ImGui::BulletText("ML Animation Generation (Diffusion)");
    ImGui::BulletText("Player Style Fingerprinting");
    ImGui::BulletText("UE5 Runtime Plugin");
    ImGui::BulletText("BVH/JSON Export");

    if (ImGui::Button("Close")) {
        showAbout_ = false;
    }

    ImGui::End();
}

// ===================================================================
// File Operations
// ===================================================================

void HyperMotionApp::openFileBrowser() {
#ifdef _WIN32
    // Windows native file dialog
    OPENFILENAMEA ofn;
    char fileName[4096] = "";
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Video Files\0*.mp4;*.avi;*.mov;*.mkv;*.webm;*.flv;*.wmv\0All Files\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = sizeof(fileName);
    ofn.Flags = OFN_ALLOWMULTISELECT | OFN_EXPLORER | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        std::vector<std::string> paths;
        // Handle multi-select: first string is directory, subsequent are filenames
        char* p = fileName;
        std::string dir = p;
        p += dir.size() + 1;
        if (*p == '\0') {
            paths.push_back(dir);
        } else {
            while (*p) {
                paths.push_back(dir + "\\" + p);
                p += strlen(p) + 1;
            }
        }
        addVideoFiles(paths);
    }
#else
    // Linux/macOS: use zenity or kdialog
    std::string cmd;
#if defined(__APPLE__)
    cmd = "osascript -e 'set theFiles to choose file of type {\"public.movie\"} with multiple selections allowed' 2>/dev/null";
#else
    cmd = "zenity --file-selection --multiple --separator='|' "
          "--file-filter='Video files|*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv' "
          "--file-filter='All files|*' 2>/dev/null";
#endif
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        std::array<char, 4096> buffer;
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);

        // Remove trailing newline
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
            result.pop_back();

        if (!result.empty()) {
            std::vector<std::string> paths;
            std::stringstream ss(result);
            std::string path;
            while (std::getline(ss, path, '|')) {
                if (!path.empty()) paths.push_back(path);
            }
            addVideoFiles(paths);
        }
    }
#endif
}

void HyperMotionApp::openFolderBrowser() {
#ifdef _WIN32
    BROWSEINFOA bi = { 0 };
    bi.lpszTitle = "Select folder containing videos";
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderA(&bi);
    if (pidl) {
        char path[MAX_PATH];
        SHGetPathFromIDListA(pidl, path);
        CoTaskMemFree(pidl);
        // Scan directory for videos
        std::vector<std::string> paths;
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && isVideoFile(entry.path().string())) {
                paths.push_back(entry.path().string());
            }
        }
        std::sort(paths.begin(), paths.end());
        addVideoFiles(paths);
    }
#else
    std::string cmd = "zenity --file-selection --directory 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        std::array<char, 4096> buffer;
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);

        while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
            result.pop_back();

        if (!result.empty() && std::filesystem::is_directory(result)) {
            std::vector<std::string> paths;
            for (const auto& entry : std::filesystem::directory_iterator(result)) {
                if (entry.is_regular_file() && isVideoFile(entry.path().string())) {
                    paths.push_back(entry.path().string());
                }
            }
            std::sort(paths.begin(), paths.end());
            addVideoFiles(paths);
        }
    }
#endif
}

bool HyperMotionApp::isVideoFile(const std::string& path) const {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    for (const char* validExt : VIDEO_EXTENSIONS) {
        if (ext == validExt) return true;
    }
    return false;
}

std::string HyperMotionApp::formatFileSize(int64_t bytes) const {
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
    if (bytes < 1024LL * 1024 * 1024) {
        double mb = bytes / (1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << mb << " MB";
        return oss.str();
    }
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gb << " GB";
    return oss.str();
}

// ===================================================================
// Add Video Files
// ===================================================================

void HyperMotionApp::addVideoFiles(const std::vector<std::string>& paths) {
    int added = 0;
    std::lock_guard<std::mutex> lock(queueMutex_);

    for (const auto& path : paths) {
        if (!isVideoFile(path)) continue;
        if (!std::filesystem::exists(path)) continue;

        // Check for duplicates
        bool duplicate = false;
        for (const auto& entry : videoQueue_) {
            if (entry.path == path) { duplicate = true; break; }
        }
        if (duplicate) continue;

        VideoEntry entry;
        entry.path = path;
        entry.filename = std::filesystem::path(path).filename().string();
        entry.extension = std::filesystem::path(path).extension().string();
        entry.fileSizeBytes = std::filesystem::file_size(path);
        entry.status = VideoEntry::Status::Queued;

        videoQueue_.push_back(std::move(entry));
        added++;
    }

    if (added > 0) {
        addLog(LogMessage::Level::Success,
               "Added " + std::to_string(added) + " video(s) to queue");

        if (selectedVideoIdx_ < 0 && !videoQueue_.empty()) {
            selectedVideoIdx_ = 0;
            loadVideoPreview(videoQueue_[0].path);
        }
    }
}

// ===================================================================
// Pipeline Processing
// ===================================================================

void HyperMotionApp::startProcessing() {
    if (isProcessing_ || videoQueue_.empty()) return;

    isProcessing_ = true;
    stopRequested_ = false;

    addLog(LogMessage::Level::Info, "Starting batch processing...");

    processingThread_ = std::thread([this]() {
        for (int i = 0; i < static_cast<int>(videoQueue_.size()); ++i) {
            if (stopRequested_) break;

            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (videoQueue_[i].status == VideoEntry::Status::Complete) continue;
            }

            currentProcessingIdx_ = i;
            processNextVideo();
        }

        isProcessing_ = false;
        currentProcessingIdx_ = -1;
        addLog(LogMessage::Level::Success, "Batch processing complete");
    });

    processingThread_.detach();
}

void HyperMotionApp::stopProcessing() {
    stopRequested_ = true;
    addLog(LogMessage::Level::Warn, "Stop requested...");
}

void HyperMotionApp::processNextVideo() {
    int idx = currentProcessingIdx_;

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        videoQueue_[idx].status = VideoEntry::Status::Processing;
        videoQueue_[idx].progress = 0.0f;
    }

    addLog(LogMessage::Level::Info, "Processing: " + videoQueue_[idx].filename);

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        // Build pipeline config from GUI settings
        PipelineConfig config;
        config.poseConfig.detector.modelPath = modelConfig_.detectorModelPath;
        config.poseConfig.poseEstimator.modelPath = modelConfig_.poseModelPath;
        config.poseConfig.depthLifter.modelPath = modelConfig_.depthModelPath;
        config.poseConfig.targetFPS = modelConfig_.targetFPS;
        config.poseConfig.enableVisualization = modelConfig_.enableVisualization;
        config.segmenterConfig.modelPath = modelConfig_.segmenterModelPath;
        config.targetFPS = modelConfig_.targetFPS;
        config.splitBySegment = modelConfig_.splitBySegment;
        config.outputDirectory = std::string(modelConfig_.outputDirectory) + "/" +
            std::filesystem::path(videoQueue_[idx].path).stem().string();

        const char* fmtOptions[] = { "json", "bvh", "both" };
        config.outputFormat = fmtOptions[modelConfig_.outputFormatIdx];

        Pipeline pipeline(config);
        if (!pipeline.initialize()) {
            onPipelineError(idx, "Failed to initialize pipeline — check model paths in Settings");
            return;
        }

        auto clips = pipeline.processVideo(videoQueue_[idx].path,
            [this, idx](float pct, const std::string& msg) {
                if (stopRequested_) return;
                std::lock_guard<std::mutex> lock(queueMutex_);
                videoQueue_[idx].progress = pct / 100.0f;
                videoQueue_[idx].progressMessage = msg;
            });

        if (stopRequested_) {
            std::lock_guard<std::mutex> lock(queueMutex_);
            videoQueue_[idx].status = VideoEntry::Status::Queued;
            videoQueue_[idx].progress = 0.0f;
            return;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(endTime - startTime).count();

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            videoQueue_[idx].status = VideoEntry::Status::Complete;
            videoQueue_[idx].progress = 1.0f;
            videoQueue_[idx].processingTimeSec = elapsed;
            videoQueue_[idx].numClipsExtracted = static_cast<int>(clips.size());

            int totalFrames = 0;
            std::set<int> personIDs;
            for (const auto& clip : clips) {
                totalFrames += static_cast<int>(clip.frames.size());
                personIDs.insert(clip.trackingID);
            }
            videoQueue_[idx].totalFramesProcessed = totalFrames;
            videoQueue_[idx].numPersonsDetected = static_cast<int>(personIDs.size());
            videoQueue_[idx].clips = std::move(clips);
        }

        onPipelineComplete(idx);

    } catch (const std::exception& e) {
        onPipelineError(idx, e.what());
    }
}

void HyperMotionApp::onPipelineProgress(float percent, const std::string& message) {
    // Called from pipeline callback — already handled inline
}

void HyperMotionApp::onPipelineComplete(int videoIdx) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    auto& entry = videoQueue_[videoIdx];
    addLog(LogMessage::Level::Success,
           entry.filename + ": " + std::to_string(entry.numClipsExtracted) + " clips, " +
           std::to_string(entry.numPersonsDetected) + " persons, " +
           std::to_string(entry.totalFramesProcessed) + " frames in " +
           std::to_string(static_cast<int>(entry.processingTimeSec)) + "s");
}

void HyperMotionApp::onPipelineError(int videoIdx, const std::string& error) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    videoQueue_[videoIdx].status = VideoEntry::Status::Error;
    videoQueue_[videoIdx].errorMessage = error;
    addLog(LogMessage::Level::Error, videoQueue_[videoIdx].filename + ": " + error);
}

// ===================================================================
// Video Preview
// ===================================================================

void HyperMotionApp::loadVideoPreview(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        hasPreview_ = false;
        return;
    }

    previewWidth_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    previewHeight_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    previewTotalFrames_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    previewFPS_ = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    previewCurrentFrame_ = 0;
    previewPlaying_ = false;

    // Read first frame
    cv::Mat frame;
    if (cap.read(frame)) {
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        previewPixelBuffer_.resize(rgb.total() * rgb.elemSize());
        std::memcpy(previewPixelBuffer_.data(), rgb.data, previewPixelBuffer_.size());

        glBindTexture(GL_TEXTURE_2D, previewTextureID_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, previewWidth_, previewHeight_, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, previewPixelBuffer_.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        hasPreview_ = true;
    }
}

void HyperMotionApp::updatePreviewFrame() {
    if (!hasPreview_ || selectedVideoIdx_ < 0) return;

    // Lazy open: re-open on each seek/play cycle for simplicity
    cv::VideoCapture cap(videoQueue_[selectedVideoIdx_].path);
    if (!cap.isOpened()) return;

    previewCurrentFrame_++;
    if (previewCurrentFrame_ >= previewTotalFrames_) {
        previewCurrentFrame_ = 0;
        previewPlaying_ = false;
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, previewCurrentFrame_);
    cv::Mat frame;
    if (cap.read(frame)) {
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        glBindTexture(GL_TEXTURE_2D, previewTextureID_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, previewWidth_, previewHeight_,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void HyperMotionApp::seekPreview(float normalizedPos) {
    previewCurrentFrame_ = static_cast<int>(normalizedPos * previewTotalFrames_);
    previewCurrentFrame_ = std::clamp(previewCurrentFrame_, 0, previewTotalFrames_ - 1);

    if (selectedVideoIdx_ < 0) return;

    cv::VideoCapture cap(videoQueue_[selectedVideoIdx_].path);
    if (!cap.isOpened()) return;

    cap.set(cv::CAP_PROP_POS_FRAMES, previewCurrentFrame_);
    cv::Mat frame;
    if (cap.read(frame)) {
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        glBindTexture(GL_TEXTURE_2D, previewTextureID_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, previewWidth_, previewHeight_,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

// ===================================================================
// Logging
// ===================================================================

void HyperMotionApp::addLog(LogMessage::Level level, const std::string& text) {
    std::lock_guard<std::mutex> lock(logMutex_);
    LogMessage msg;
    msg.level = level;
    msg.text = text;
    msg.timestamp = glfwGetTime();
    logMessages_.push_back(msg);
    if (logMessages_.size() > MAX_LOG_MESSAGES) {
        logMessages_.pop_front();
    }
}

// ===================================================================
// GLFW Callbacks
// ===================================================================

void HyperMotionApp::glfwDropCallback(GLFWwindow* window, int count, const char** paths) {
    if (!g_appInstance) return;

    std::vector<std::string> filePaths;
    for (int i = 0; i < count; ++i) {
        std::string path = paths[i];

        if (std::filesystem::is_directory(path)) {
            // Recursively add videos from dropped folder
            for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                if (entry.is_regular_file() && g_appInstance->isVideoFile(entry.path().string())) {
                    filePaths.push_back(entry.path().string());
                }
            }
        } else {
            filePaths.push_back(path);
        }
    }

    g_appInstance->addVideoFiles(filePaths);

    // Show drop overlay briefly
    g_appInstance->showDropOverlay_ = true;
    g_appInstance->dropOverlayTimer_ = glfwGetTime();
}

void HyperMotionApp::glfwKeyCallback(GLFWwindow* window, int key, int scancode,
                                       int action, int mods) {
    if (!g_appInstance || action != GLFW_PRESS) return;

    // Ctrl+O: Open file browser
    if (key == GLFW_KEY_O && (mods & GLFW_MOD_CONTROL)) {
        if (mods & GLFW_MOD_SHIFT) {
            g_appInstance->openFolderBrowser();
        } else {
            g_appInstance->openFileBrowser();
        }
    }

    // F5: Start processing
    if (key == GLFW_KEY_F5) {
        g_appInstance->startProcessing();
    }

    // Escape: Stop processing
    if (key == GLFW_KEY_ESCAPE) {
        g_appInstance->stopProcessing();
    }

    // Ctrl+,: Settings
    if (key == GLFW_KEY_COMMA && (mods & GLFW_MOD_CONTROL)) {
        g_appInstance->showSettings_ = !g_appInstance->showSettings_;
    }
}

} // namespace hm::gui
