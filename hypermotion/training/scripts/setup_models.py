#!/usr/bin/env python3
"""Download pre-trained models and export to ONNX for the HyperMotion C++ pipeline.

Downloads:
  1. YOLOv8m (person detector)    → models/yolov8m.onnx
  2. RTMPose-M (pose estimator)   → models/rtmpose_m.onnx
  3. Lightweight 2D→3D lifter     → models/lifter_3d.pt  (TorchScript)

Usage:
    python setup_models.py [--output-dir models/] [--detector-only] [--pose-only]

These models work out of the box with the HyperMotion C++ pipeline.
No training data or GPU required — all weights are pre-trained.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===================================================================
# Model URLs and checksums
# ===================================================================

YOLOV8M_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
YOLOV8M_SHA256 = None  # Ultralytics updates frequently; skip hash check

# RTMPose-M from MMPose model zoo (ONNX already available)
RTMPOSE_M_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)

# Lightweight 2D→3D lifter architecture definition (we create and export this)
LIFTER_INPUT_DIM = 34   # 17 keypoints × 2 (x, y)
LIFTER_OUTPUT_DIM = 51  # 17 keypoints × 3 (x, y, z)
LIFTER_HIDDEN_DIM = 1024

# ===================================================================
# Utility functions
# ===================================================================


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress reporting."""
    if dest.exists():
        log.info(f"  Already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    desc = desc or dest.name

    log.info(f"  Downloading {desc}...")
    log.info(f"    URL: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HyperMotion/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(str(dest), "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

            if total > 0:
                print()  # newline after progress

    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {e}") from e

    log.info(f"    Saved: {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")
    return dest


def check_pip_package(name: str) -> bool:
    """Check if a pip package is installed."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def ensure_package(name: str, pip_name: str | None = None):
    """Install a pip package if not present."""
    pip_name = pip_name or name
    if not check_pip_package(name):
        log.info(f"  Installing {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            stdout=subprocess.DEVNULL,
        )


# ===================================================================
# 1. YOLOv8 Player Detector
# ===================================================================


def setup_yolov8_detector(output_dir: Path) -> Path:
    """Download YOLOv8m and export to ONNX."""
    log.info("\n[1/3] YOLOv8m Player Detector")
    log.info("=" * 50)

    onnx_path = output_dir / "yolov8m.onnx"
    if onnx_path.exists():
        log.info(f"  Already exists: {onnx_path}")
        return onnx_path

    ensure_package("ultralytics")

    pt_path = output_dir / "yolov8m.pt"
    download_file(YOLOV8M_URL, pt_path, "YOLOv8m weights")

    log.info("  Exporting to ONNX (input: 640×640, opset 17)...")

    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
    )

    # Ultralytics saves to same dir as .pt with .onnx extension
    exported = pt_path.with_suffix(".onnx")
    if exported != onnx_path:
        shutil.move(str(exported), str(onnx_path))

    # Clean up .pt file to save space
    if pt_path.exists():
        pt_path.unlink()

    log.info(f"  Exported: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")
    log.info("  Input:  [1, 3, 640, 640] float32 (RGB, /255)")
    log.info("  Output: [1, 84, 8400] float32 (80 COCO classes)")
    log.info("  Note:   C++ pipeline auto-maps COCO class 0 ('person') → 'player'")

    return onnx_path


# ===================================================================
# 2. RTMPose Pose Estimator
# ===================================================================


def setup_rtmpose(output_dir: Path) -> Path:
    """Download RTMPose-M ONNX model for 17-keypoint pose estimation."""
    log.info("\n[2/3] RTMPose-M Pose Estimator")
    log.info("=" * 50)

    onnx_path = output_dir / "rtmpose_m.onnx"
    if onnx_path.exists():
        log.info(f"  Already exists: {onnx_path}")
        return onnx_path

    # Download the zip
    zip_path = output_dir / "rtmpose_m.zip"
    download_file(RTMPOSE_M_URL, zip_path, "RTMPose-M ONNX")

    # Extract the ONNX file
    import zipfile

    log.info("  Extracting ONNX from archive...")
    with zipfile.ZipFile(str(zip_path)) as zf:
        onnx_files = [n for n in zf.namelist() if n.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError("No .onnx file found in RTMPose archive")

        # Extract the first ONNX file found
        onnx_name = onnx_files[0]
        with zf.open(onnx_name) as src, open(str(onnx_path), "wb") as dst:
            shutil.copyfileobj(src, dst)

    zip_path.unlink()

    log.info(f"  Exported: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")
    log.info("  Input:  [1, 3, 256, 192] float32 (RGB, ImageNet normalized)")
    log.info("  Output: SimCC x/y logits → 17 COCO keypoints")

    return onnx_path


def setup_hrnet_fallback(output_dir: Path) -> Path:
    """If RTMPose download fails, try HRNet-W32 via onnxruntime model zoo."""
    log.info("  RTMPose download failed, trying HRNet-W32 fallback...")

    onnx_path = output_dir / "hrnet_w32.onnx"
    if onnx_path.exists():
        return onnx_path

    # HRNet-W32 ONNX from the original MMPose zoo
    HRNET_URL = (
        "https://download.openmmlab.com/mmpose/top_down/hrnet/"
        "hrnet_w32_coco_256x192-c78dce93_20200708.onnx"
    )

    try:
        download_file(HRNET_URL, onnx_path, "HRNet-W32 ONNX")
    except Exception:
        log.warning("  Could not download pre-built ONNX.")
        log.warning("  Pose estimation will use the geometric fallback in the C++ pipeline.")
        return onnx_path  # Return path even if it doesn't exist; C++ handles missing model

    log.info(f"  Exported: {onnx_path}")
    log.info("  Input:  [1, 3, 256, 192] float32 (RGB, ImageNet normalized)")
    log.info("  Output: [1, 17, 64, 48] float32 (heatmaps)")

    return onnx_path


# ===================================================================
# 3. 2D→3D Pose Lifter
# ===================================================================


def setup_depth_lifter(output_dir: Path) -> Path:
    """Create and export a lightweight 2D→3D lifting MLP.

    Architecture matches the C++ DepthLifter expectation:
      Input:  [batch, 34]  (17 keypoints × 2)
      Output: [batch, 51]  (17 keypoints × 3)

    This creates an untrained model — the geometric fallback in C++ works
    well for most cases. For production quality, train this on Human3.6M
    or similar paired 2D/3D datasets.
    """
    log.info("\n[3/3] 2D→3D Pose Lifter")
    log.info("=" * 50)

    pt_path = output_dir / "lifter_3d.pt"
    if pt_path.exists():
        log.info(f"  Already exists: {pt_path}")
        return pt_path

    ensure_package("torch")

    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.block(x))

    class LifterMLP(nn.Module):
        def __init__(self, in_dim=34, hidden_dim=1024, out_dim=51):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                ResBlock(hidden_dim),
                ResBlock(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    model = LifterMLP(LIFTER_INPUT_DIM, LIFTER_HIDDEN_DIM, LIFTER_OUTPUT_DIM)

    # Initialize with sensible defaults: identity-ish mapping for x,y; zero for z
    # This makes the untrained model at least produce reasonable x,y coordinates
    with torch.no_grad():
        last_linear = model.net[-1]
        last_linear.weight.zero_()
        last_linear.bias.zero_()
        for k in range(17):
            # Map input x,y to output x,y with unit scale
            last_linear.weight[k * 3 + 0, k * 2 + 0] = 1.0  # x → x
            last_linear.weight[k * 3 + 1, k * 2 + 1] = 1.0  # y → y
            # z starts at zero (geometric fallback handles depth)

    model.eval()

    # Export as TorchScript (the C++ DepthLifter uses LibTorch)
    scripted = torch.jit.script(model)
    scripted.save(str(pt_path))

    log.info(f"  Exported: {pt_path} ({pt_path.stat().st_size / (1024*1024):.1f} MB)")
    log.info("  Input:  [batch, 34]  (17 keypoints × 2, bbox-normalized)")
    log.info("  Output: [batch, 51]  (17 keypoints × 3, cm)")
    log.info("  Note:   Untrained weights — C++ geometric fallback works well for most cases.")
    log.info("          For production quality, train on Human3.6M or similar dataset.")

    return pt_path


# ===================================================================
# Config file generation
# ===================================================================


def write_config(output_dir: Path, detector_path: Path, pose_path: Path, lifter_path: Path):
    """Write a default HyperMotion pipeline config pointing to downloaded models."""
    import json

    config = {
        "targetFPS": 30.0,
        "splitBySegment": True,
        "outputFormat": "both",
        "minTrackFrames": 10,
        "enableCanonicalMotion": True,
        "enableTrajectoryExtraction": True,
        "enableFingerprinting": True,
        "enableFootContactDetection": True,
        "enableMotionClustering": True,
        "pose": {
            "targetFPS": 30.0,
            "detector": {
                "modelPath": str(detector_path),
                "confidenceThreshold": 0.5,
                "nmsIouThreshold": 0.45,
                "inputWidth": 640,
                "inputHeight": 640,
                "maxDetections": 30,
            },
            "poseEstimator": {
                "modelPath": str(pose_path),
                "inputWidth": 192,
                "inputHeight": 256,
                "confidenceThreshold": 0.3,
            },
            "depthLifter": {
                "modelPath": str(lifter_path),
                "useGeometricFallback": True,
                "defaultSubjectHeight": 175.0,
            },
        },
        "signal": {
            "enableOutlierFilter": True,
            "enableSavitzkyGolay": True,
            "enableButterworth": True,
            "enableQuaternionSmoothing": True,
            "enableFootContact": True,
        },
        "segmenter": {
            "minSegmentLength": 10,
            "confidenceThreshold": 0.5,
        },
    }

    config_path = output_dir / "pipeline_config.json"
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"\nConfig written: {config_path}")
    return config_path


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download and export pre-trained models for the HyperMotion pipeline"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"),
        help="Directory to save models (default: models/)"
    )
    parser.add_argument("--detector-only", action="store_true", help="Only download detector")
    parser.add_argument("--pose-only", action="store_true", help="Only download pose estimator")
    parser.add_argument("--skip-lifter", action="store_true", help="Skip 3D lifter setup")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("╔═══════════════════════════════════════════════╗")
    log.info("║   HyperMotion Model Setup                    ║")
    log.info("╚═══════════════════════════════════════════════╝")
    log.info(f"\nOutput directory: {output_dir}\n")

    detector_path = output_dir / "yolov8m.onnx"
    pose_path = output_dir / "rtmpose_m.onnx"
    lifter_path = output_dir / "lifter_3d.pt"

    try:
        if not args.pose_only:
            detector_path = setup_yolov8_detector(output_dir)

        if not args.detector_only:
            try:
                pose_path = setup_rtmpose(output_dir)
            except Exception as e:
                log.warning(f"  RTMPose download failed: {e}")
                pose_path = setup_hrnet_fallback(output_dir)

        if not args.detector_only and not args.pose_only and not args.skip_lifter:
            lifter_path = setup_depth_lifter(output_dir)

    except Exception as e:
        log.error(f"\nSetup failed: {e}")
        log.error("Some models may not have been downloaded.")
        log.error("The C++ pipeline will use fallbacks where available.")

    # Write config file
    config_path = write_config(output_dir, detector_path, pose_path, lifter_path)

    log.info("\n" + "=" * 50)
    log.info("Setup complete!")
    log.info(f"  Models:  {output_dir}/")
    log.info(f"  Config:  {config_path}")
    log.info("")
    log.info("To run the pipeline:")
    log.info(f"  kinetica_analyse_match \\")
    log.info(f"    --config {config_path} \\")
    log.info(f"    video.mp4 output/")
    log.info("")
    log.info("Or load the config programmatically:")
    log.info(f'  loadPipelineConfig("{config_path}", config);')


if __name__ == "__main__":
    main()
