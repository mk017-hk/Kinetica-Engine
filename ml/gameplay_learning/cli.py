"""CLI entry point for the gameplay learning pipeline.

Usage
-----
Extract clips from a video::

    python -m ml.gameplay_learning.cli extract --video match.mp4 --output-dir clips/ --fps 30

Show statistics for previously extracted clips::

    python -m ml.gameplay_learning.cli stats --clips-dir clips/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..constants import MOTION_TYPES
from .pipeline import GameplayLearningPipeline, PipelineConfig


def _setup_logging(verbose: bool) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# Progress helper
# ------------------------------------------------------------------

def _cli_progress(stage: str, current: int, total: int) -> None:
    """Print a progress line to stderr."""
    if total > 0:
        pct = current * 100 // total
        print(
            f"\r  {stage}: {current}/{total} ({pct}%)",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if current >= total:
            print(file=sys.stderr)


# ------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------

def _cmd_extract(args: argparse.Namespace) -> None:
    """Run the extraction pipeline."""
    config = PipelineConfig(
        target_fps=args.fps,
        pose_model_path=args.pose_model,
        pose_confidence=args.pose_confidence,
        tracker_max_age=args.tracker_max_age,
        tracker_min_hits=args.tracker_min_hits,
        tracker_iou_threshold=args.tracker_iou_threshold,
        min_track_length=args.min_track_length,
        min_segment_length=args.min_segment_length,
        output_dir=args.output_dir,
    )

    pipeline = GameplayLearningPipeline(
        config=config,
        progress_callback=_cli_progress,
    )
    summary = pipeline.run(args.video)

    print(f"\nExtraction complete:")
    print(f"  Video:       {summary['video']}")
    print(f"  Frames:      {summary['total_frames']}")
    print(f"  Tracks:      {summary['tracks']}")
    print(f"  Clips:       {summary['clips']}")
    print(f"  Output:      {summary['output_dir']}")
    print(f"  Elapsed:     {summary['elapsed_sec']:.1f}s")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Print statistics for a directory of previously extracted clips."""
    clips_dir = Path(args.clips_dir)
    manifest_path = clips_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {clips_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    total_clips = manifest.get("total_clips", 0)
    total_tracks = manifest.get("total_tracks", 0)
    type_counts = manifest.get("motion_type_counts", {})
    clips_list = manifest.get("clips", [])

    # Compute total frames across all clips.
    total_frames = sum(c.get("num_frames", 0) for c in clips_list)

    print(f"Clip statistics for: {clips_dir}")
    print(f"  Source video: {manifest.get('video', 'N/A')}")
    print(f"  Total tracks: {total_tracks}")
    print(f"  Total clips:  {total_clips}")
    print(f"  Total frames: {total_frames}")
    print()

    if type_counts:
        print("  Motion type breakdown:")
        # Show all 16 canonical types, even if count is zero.
        for mt in MOTION_TYPES:
            count = type_counts.get(mt, 0)
            bar = "#" * count
            print(f"    {mt:<14s} {count:>5d}  {bar}")
    else:
        print("  No motion type breakdown available.")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gameplay_learning",
        description="Extract animation clips from match footage.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract ---------------------------------------------------------
    p_extract = subparsers.add_parser(
        "extract", help="Extract motion clips from a video."
    )
    p_extract.add_argument(
        "--video", required=True, help="Path to input video."
    )
    p_extract.add_argument(
        "--output-dir",
        default="clips",
        help="Output directory for clips (default: clips/).",
    )
    p_extract.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frame rate for sampling (default: 30).",
    )
    p_extract.add_argument(
        "--pose-model",
        default=None,
        help="Path to YOLOv8-pose model (default: yolov8n-pose.pt).",
    )
    p_extract.add_argument(
        "--pose-confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence (default: 0.5).",
    )
    p_extract.add_argument(
        "--tracker-max-age",
        type=int,
        default=30,
        help="Max frames before a lost track is finished (default: 30).",
    )
    p_extract.add_argument(
        "--tracker-min-hits",
        type=int,
        default=3,
        help="Minimum detections to confirm a track (default: 3).",
    )
    p_extract.add_argument(
        "--tracker-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for track association (default: 0.3).",
    )
    p_extract.add_argument(
        "--min-track-length",
        type=int,
        default=30,
        help="Minimum detections in a track for export (default: 30).",
    )
    p_extract.add_argument(
        "--min-segment-length",
        type=int,
        default=10,
        help="Minimum frames in a motion segment (default: 10).",
    )
    p_extract.set_defaults(func=_cmd_extract)

    # --- stats -----------------------------------------------------------
    p_stats = subparsers.add_parser(
        "stats", help="Show statistics for extracted clips."
    )
    p_stats.add_argument(
        "--clips-dir",
        required=True,
        help="Directory containing clips and manifest.json.",
    )
    p_stats.set_defaults(func=_cmd_stats)

    return parser


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
