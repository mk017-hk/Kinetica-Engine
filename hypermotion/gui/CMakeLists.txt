# ===================================================================
#  HyperMotion Studio — GUI Application
#  Self-contained build for the desktop .exe / binary
# ===================================================================

include(FetchContent)

# --- GLFW: try system first, fall back to FetchContent ---
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "GLFW not found on system — fetching...")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glfw)
    message(STATUS "GLFW fetched")
else()
    message(STATUS "GLFW found on system")
endif()

find_package(OpenGL REQUIRED)

# --- Dear ImGui: use IMGUI_DIR if set, otherwise fetch ---
if(NOT DEFINED IMGUI_DIR)
    set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")
endif()

if(NOT EXISTS "${IMGUI_DIR}/imgui.cpp")
    message(STATUS "Dear ImGui not found at ${IMGUI_DIR} — fetching...")
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)
    set(IMGUI_DIR "${imgui_SOURCE_DIR}")
    message(STATUS "Dear ImGui fetched to ${IMGUI_DIR}")
endif()

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(HM_GUI_SOURCES
    main.cpp
    HyperMotionApp.cpp
)

# Windows: embed icon, version info, and manifest into the .exe
set(HM_GUI_RESOURCES "")
if(WIN32)
    set(HM_GUI_RESOURCES resources/hm_studio.rc)
endif()

# WIN32 flag: produces a Windows GUI app (no console window on launch)
# On non-Windows platforms this flag is silently ignored
add_executable(hm_studio WIN32
    ${HM_GUI_SOURCES}
    ${IMGUI_SOURCES}
    ${HM_GUI_RESOURCES}
)

# On Windows with WIN32 subsystem, entry point is WinMain.
# We redirect back to standard main() via /ENTRY:mainCRTStartup.
if(WIN32)
    target_link_options(hm_studio PRIVATE "/ENTRY:mainCRTStartup")
endif()

target_include_directories(hm_studio PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/resources
)

target_link_libraries(hm_studio PRIVATE
    hypermotion
    glfw
    OpenGL::GL
)

if(WIN32)
    target_link_libraries(hm_studio PRIVATE dwmapi)
endif()

target_compile_definitions(hm_studio PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_CUSTOM
)

# Output name and Visual Studio working directory
set_target_properties(hm_studio PROPERTIES
    OUTPUT_NAME "HyperMotion Studio"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Copy required DLLs next to the .exe on Windows after build
if(WIN32)
    add_custom_command(TARGET hm_studio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:hm_studio>
            $<TARGET_FILE_DIR:hm_studio>
        COMMAND_EXPAND_LISTS
    )
endif()

install(TARGETS hm_studio RUNTIME DESTINATION bin)

# -------------------------------------------------------------------
# CPack — distributable package (ZIP portable or NSIS installer)
# -------------------------------------------------------------------

set(CPACK_PACKAGE_NAME "HyperMotion-Studio")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "HyperMotion")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Video-to-Animation Pipeline Studio")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "HyperMotion Studio")

if(EXISTS "${CMAKE_SOURCE_DIR}/../LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/../LICENSE")
endif()

if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "HyperMotion Studio")
    set(CPACK_NSIS_PACKAGE_NAME "HyperMotion Studio")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\HyperMotion Studio.lnk' '$INSTDIR\\\\bin\\\\HyperMotion Studio.exe'")
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$START_MENU\\\\HyperMotion Studio.lnk'")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
