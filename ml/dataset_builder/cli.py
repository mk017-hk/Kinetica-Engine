"""CLI entry point for the dataset builder pipeline.

Usage::

    python -m ml.dataset_builder.cli --video match.mp4 --output-dir dataset/
    python -m ml.dataset_builder.cli --video-dir videos/ --fps 30 --device cuda

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import DatasetPipeline, PipelineConfig, PipelineResult
from .sequence_exporter import FORMAT_WORLD_POSITIONS, FORMAT_6D_ROTATIONS

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".ts", ".flv"}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the dataset builder CLI."""
    parser = argparse.ArgumentParser(
        prog="dataset_builder",
        description="Convert football match footage into skeleton motion datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to a single input video file.",
    )
    input_group.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing video files to process.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_out",
        help="Root directory for exported sequences.",
    )

    # Frame extraction
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target frame extraction rate.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract per video.",
    )

    # Pose detection
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolov8m-pose",
        help="YOLOv8 pose model name or path.",
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Minimum bounding-box confidence.",
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.3,
        help="Minimum per-keypoint confidence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Inference device.",
    )

    # Skeleton normalisation
    parser.add_argument(
        "--normalizer-min-conf",
        type=float,
        default=0.3,
        help="Minimum keypoint confidence for normaliser.",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=["linear", "zero"],
        help="Temporal interpolation method for missing joints.",
    )
    parser.add_argument(
        "--height-ref",
        type=str,
        default="torso",
        choices=["torso", "full"],
        help="Reference for unit-height scaling.",
    )

    # Sequence export
    parser.add_argument(
        "--window-length",
        type=int,
        default=64,
        help="Frames per export window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Step between consecutive windows.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=FORMAT_WORLD_POSITIONS,
        choices=[FORMAT_WORLD_POSITIONS, FORMAT_6D_ROTATIONS],
        help="Output format for exported sequences.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=16,
        help="Minimum frames for a person track to be kept.",
    )

    # General
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return parser


def _collect_videos(args: argparse.Namespace) -> list[Path]:
    """Resolve the list of video files from CLI arguments."""
    if args.video:
        path = Path(args.video)
        if not path.exists():
            logger.error("Video file not found: %s", path)
            return []
        return [path]

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        logger.error("Video directory not found: %s", video_dir)
        return []

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in _VIDEO_EXTENSIONS and p.is_file()
    )
    if not videos:
        logger.error("No video files found in %s", video_dir)

    return videos


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments."""
    return PipelineConfig(
        target_fps=args.fps,
        max_frames=args.max_frames,
        pose_model=args.pose_model,
        detection_conf=args.detection_conf,
        keypoint_conf=args.keypoint_conf,
        device=args.device,
        min_visible_keypoints=6,
        normalizer_min_confidence=args.normalizer_min_conf,
        interpolation_method=args.interpolation,
        unit_height_reference=args.height_ref,
        window_length=args.window_length,
        stride=args.stride,
        output_format=args.format,
        output_dir=args.output_dir,
        min_track_length=args.min_track_length,
    )


def _print_summary(results: list[PipelineResult]) -> None:
    """Print a summary table of all processed videos."""
    total_frames = sum(r.total_frames_extracted for r in results)
    total_detections = sum(r.total_detections for r in results)
    total_tracks = sum(r.total_tracks for r in results)
    total_windows = sum(
        r.export_stats.total_windows for r in results if r.export_stats
    )
    total_time = sum(r.elapsed_sec for r in results)
    total_errors = sum(len(r.errors) for r in results)

    print("\n" + "=" * 60)
    print("Dataset Builder — Summary")
    print("=" * 60)
    print(f"  Videos processed : {len(results)}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Total detections : {total_detections}")
    print(f"  Total tracks     : {total_tracks}")
    print(f"  Total windows    : {total_windows}")
    print(f"  Total time       : {total_time:.1f}s")

    if total_errors > 0:
        print(f"  Errors           : {total_errors}")
        for r in results:
            for err in r.errors:
                print(f"    [{r.video_path}] {err}")

    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  ``None`` defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    videos = _collect_videos(args)
    if not videos:
        return 1

    config = _build_config(args)
    pipeline = DatasetPipeline(config)

    results: list[PipelineResult] = []
    for i, video_path in enumerate(videos, 1):
        logger.info(
            "Processing video %d/%d: %s", i, len(videos), video_path.name
        )
        result = pipeline.run(video_path)
        results.append(result)

    _print_summary(results)

    has_errors = any(r.errors for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
