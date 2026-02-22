@echo off
REM ================================================================
REM  HyperMotion Studio — Windows Build Script
REM  Run from anywhere; the script locates the hypermotion/ root.
REM  Produces: hypermotion/build/Release/HyperMotion Studio.exe
REM ================================================================

setlocal enabledelayedexpansion

echo.
echo  ========================================
echo   HyperMotion Studio — Build
echo  ========================================
echo.

REM --- Locate the hypermotion project root (one level up from gui/) ---
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\"
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

REM --- Check for CMake ---
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found. Install from https://cmake.org/download/
    pause
    exit /b 1
)

REM --- Check for Visual Studio ---
set "GENERATOR="
if exist "%ProgramFiles%\Microsoft Visual Studio\2022" (
    set "GENERATOR=Visual Studio 17 2022"
) else if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019" (
    set "GENERATOR=Visual Studio 16 2019"
) else (
    echo [WARN] Visual Studio not detected, using default CMake generator
)

REM --- Configuration ---
set BUILD_DIR=%PROJECT_ROOT%\build
set BUILD_TYPE=Release

REM You can override dependency paths by setting environment variables:
REM   set TORCH_DIR=C:\path\to\libtorch\share\cmake\Torch
REM   set ONNXRUNTIME_ROOT=C:\path\to\onnxruntime
REM   set IMGUI_DIR=C:\path\to\imgui
REM   set OpenCV_DIR=C:\path\to\opencv\build

echo [1/3] Configuring CMake...
echo   Source:  %PROJECT_ROOT%
echo   Build:   %BUILD_DIR%
echo.

set CMAKE_ARGS=-DHM_BUILD_GUI=ON -DHM_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if defined IMGUI_DIR (
    set CMAKE_ARGS=!CMAKE_ARGS! -DIMGUI_DIR="%IMGUI_DIR%"
)
if defined TORCH_DIR (
    set CMAKE_ARGS=!CMAKE_ARGS! -DTorch_DIR="%TORCH_DIR%"
)
if defined ONNXRUNTIME_ROOT (
    set CMAKE_ARGS=!CMAKE_ARGS! -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%"
)
if defined OpenCV_DIR (
    set CMAKE_ARGS=!CMAKE_ARGS! -DOpenCV_DIR="%OpenCV_DIR%"
)

if defined GENERATOR (
    cmake -G "%GENERATOR%" -A x64 -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" %CMAKE_ARGS%
) else (
    cmake -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" %CMAKE_ARGS%
)

if errorlevel 1 (
    echo.
    echo [ERROR] CMake configuration failed.
    echo Make sure all dependencies are installed and paths are set.
    pause
    exit /b 1
)

echo.
echo [2/3] Building...
echo.

cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% --target hm_studio -- /m

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Done!
echo.
echo  Executable:  %BUILD_DIR%\%BUILD_TYPE%\HyperMotion Studio.exe
echo.
echo  To create a distributable package (ZIP or installer):
echo    cd "%BUILD_DIR%" ^&^& cpack -C %BUILD_TYPE%
echo.

REM --- Optionally run ---
set /p LAUNCH="Launch HyperMotion Studio now? (y/n): "
if /i "%LAUNCH%"=="y" (
    start "" "%BUILD_DIR%\%BUILD_TYPE%\HyperMotion Studio.exe"
)

endlocal
