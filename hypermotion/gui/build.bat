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
    echo [ERROR] CMake not found.
    echo.
    echo   Install CMake from: https://cmake.org/download/
    echo   Make sure "Add CMake to the system PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

REM ================================================================
REM  Compiler detection — check, in order:
REM    1. Visual Studio 2022/2019 full IDE
REM    2. Visual Studio Build Tools (standalone)
REM    3. cl.exe already on PATH (Developer Command Prompt)
REM    4. MinGW g++ (MSYS2 or standalone MinGW)
REM    5. clang++ on PATH
REM ================================================================
set "GENERATOR="
set "COMPILER_NAME="
set "BUILD_PARALLEL=/m"

REM --- 1. Visual Studio full IDE ---
if exist "%ProgramFiles%\Microsoft Visual Studio\2022" (
    set "GENERATOR=Visual Studio 17 2022"
    set "COMPILER_NAME=Visual Studio 2022"
    goto :compiler_found
)
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019" (
    set "GENERATOR=Visual Studio 16 2019"
    set "COMPILER_NAME=Visual Studio 2019"
    goto :compiler_found
)

REM --- 2. VS Build Tools (standalone, no full IDE) ---
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "GENERATOR=Visual Studio 17 2022"
    set "COMPILER_NAME=VS 2022 Build Tools"
    goto :compiler_found
)
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "GENERATOR=Visual Studio 16 2019"
    set "COMPILER_NAME=VS 2019 Build Tools"
    goto :compiler_found
)

REM --- 3. cl.exe on PATH (e.g. from Developer Command Prompt) ---
where cl >nul 2>&1
if not errorlevel 1 (
    set "GENERATOR=NMake Makefiles"
    set "COMPILER_NAME=MSVC (cl.exe on PATH)"
    set "BUILD_PARALLEL="
    goto :compiler_found
)

REM --- 4. MinGW g++ (MSYS2 or standalone) ---
where g++ >nul 2>&1
if not errorlevel 1 (
    set "GENERATOR=MinGW Makefiles"
    set "COMPILER_NAME=MinGW g++"
    set "BUILD_PARALLEL="
    goto :compiler_found
)
REM Check common MSYS2 install paths
if exist "C:\msys64\mingw64\bin\g++.exe" (
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
    set "GENERATOR=MinGW Makefiles"
    set "COMPILER_NAME=MSYS2 MinGW g++ (C:\msys64)"
    set "BUILD_PARALLEL="
    goto :compiler_found
)

REM --- 5. clang++ on PATH ---
where clang++ >nul 2>&1
if not errorlevel 1 (
    set "GENERATOR=Ninja"
    set "COMPILER_NAME=clang++"
    set "BUILD_PARALLEL="
    REM Check for Ninja, fall back to NMake
    where ninja >nul 2>&1
    if errorlevel 1 (
        set "GENERATOR=NMake Makefiles"
    )
    goto :compiler_found
)

REM --- No compiler found — show clear install instructions ---
echo [ERROR] No C++ compiler found.
echo.
echo  HyperMotion Studio needs a C++ compiler to build. Install ONE of:
echo.
echo  Option A — Visual Studio Community (free, recommended):
echo    1. Download from: https://visualstudio.microsoft.com/downloads/
echo    2. In the installer, check "Desktop development with C++"
echo    3. Click Install, then re-run this script
echo.
echo  Option B — Visual Studio Build Tools (smaller, command-line only):
echo    1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo    2. In the installer, check "Desktop development with C++"
echo    3. Click Install, then re-run this script
echo.
echo  Option C — MSYS2 MinGW (lighter-weight alternative):
echo    1. Download from: https://www.msys2.org/
echo    2. Open MSYS2 terminal and run:
echo         pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake
echo    3. Add C:\msys64\mingw64\bin to your system PATH
echo    4. Re-run this script from a new terminal
echo.
pause
exit /b 1

:compiler_found
echo  Compiler: %COMPILER_NAME%
echo  Generator: %GENERATOR%
echo.

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

REM CUDA and TensorRT are OFF by default — enable with: set HM_ENABLE_CUDA=ON
if not defined HM_ENABLE_CUDA set HM_ENABLE_CUDA=OFF
if not defined HM_ENABLE_TENSORRT set HM_ENABLE_TENSORRT=OFF
set CMAKE_ARGS=!CMAKE_ARGS! -DHM_ENABLE_CUDA=!HM_ENABLE_CUDA! -DHM_ENABLE_TENSORRT=!HM_ENABLE_TENSORRT!

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

REM Visual Studio generators need -A x64 for 64-bit builds
set "ARCH_FLAG="
echo "%GENERATOR%" | findstr /C:"Visual Studio" >nul 2>&1
if not errorlevel 1 (
    set "ARCH_FLAG=-A x64"
)

if defined ARCH_FLAG (
    cmake -G "%GENERATOR%" %ARCH_FLAG% -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" %CMAKE_ARGS%
) else (
    cmake -G "%GENERATOR%" -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" %CMAKE_ARGS%
)

if errorlevel 1 (
    echo.
    echo [ERROR] CMake configuration failed.
    echo.
    echo  Possible fixes:
    echo   - Make sure your compiler is installed correctly
    echo   - If using VS Build Tools, run this from "Developer Command Prompt"
    echo   - If using MinGW, make sure g++ is on PATH: g++ --version
    echo   - Delete the build/ folder and try again: rmdir /s /q "%BUILD_DIR%"
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Building...
echo.

if defined BUILD_PARALLEL (
    cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% --target hm_studio -- %BUILD_PARALLEL%
) else (
    cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% --target hm_studio
)

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Done!
echo.

REM Determine output path based on generator
if "%GENERATOR%"=="MinGW Makefiles" (
    set "EXE_PATH=%BUILD_DIR%\gui\HyperMotion Studio.exe"
) else if "%GENERATOR%"=="NMake Makefiles" (
    set "EXE_PATH=%BUILD_DIR%\gui\HyperMotion Studio.exe"
) else (
    set "EXE_PATH=%BUILD_DIR%\%BUILD_TYPE%\HyperMotion Studio.exe"
)

REM Check both common output locations
if not exist "!EXE_PATH!" (
    if exist "%BUILD_DIR%\gui\%BUILD_TYPE%\HyperMotion Studio.exe" (
        set "EXE_PATH=%BUILD_DIR%\gui\%BUILD_TYPE%\HyperMotion Studio.exe"
    ) else if exist "%BUILD_DIR%\gui\HyperMotion Studio.exe" (
        set "EXE_PATH=%BUILD_DIR%\gui\HyperMotion Studio.exe"
    ) else if exist "%BUILD_DIR%\%BUILD_TYPE%\HyperMotion Studio.exe" (
        set "EXE_PATH=%BUILD_DIR%\%BUILD_TYPE%\HyperMotion Studio.exe"
    )
)

echo  Executable:  !EXE_PATH!
echo.
echo  To create a distributable package (ZIP or installer):
echo    cd "%BUILD_DIR%" ^&^& cpack -C %BUILD_TYPE%
echo.

REM --- Optionally run ---
set /p LAUNCH="Launch HyperMotion Studio now? (y/n): "
if /i "%LAUNCH%"=="y" (
    start "" "!EXE_PATH!"
)

endlocal
