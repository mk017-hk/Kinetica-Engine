"""Full gameplay extraction pipeline orchestrating all stages.

Combines frame extraction, pose detection, multi-player tracking, skeleton
normalisation, motion segmentation, and clip export into a single
configurable pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..constants import COCO_KEYPOINTS, JOINT_COUNT, MOTION_TYPES
from .motion_segmenter import MotionClip, MotionSegmenter
from .player_tracker import Detection, PlayerTracker, Track

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the gameplay learning pipeline.

    Attributes
    ----------
    target_fps : float
        Frame rate at which to sample the source video.
    pose_model_path : str or None
        Path to a YOLOv8-pose model.  If ``None``, the default
        ``yolov8n-pose.pt`` is used (downloaded automatically by
        *ultralytics*).
    pose_confidence : float
        Minimum detection confidence for pose estimation.
    tracker_max_age : int
        Frames before an unmatched track is finished.
    tracker_min_hits : int
        Minimum detections to confirm a track.
    tracker_iou_threshold : float
        IoU threshold for Hungarian assignment.
    min_track_length : int
        Minimum number of detections in a completed track for export.
    min_segment_length : int
        Minimum number of frames in a motion segment.
    output_dir : str
        Directory for exported clips and metadata.
    """

    target_fps: float = 30.0
    pose_model_path: Optional[str] = None
    pose_confidence: float = 0.5
    tracker_max_age: int = 30
    tracker_min_hits: int = 3
    tracker_iou_threshold: float = 0.3
    min_track_length: int = 30
    min_segment_length: int = 10
    output_dir: str = "clips"


# ------------------------------------------------------------------
# Progress callback type
# ------------------------------------------------------------------

ProgressCallback = Callable[[str, int, int], None]
"""Signature: ``(stage_name, current_step, total_steps) -> None``."""


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class GameplayLearningPipeline:
    """End-to-end pipeline for extracting motion clips from match footage.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    progress_callback : ProgressCallback or None
        Optional callback invoked during each processing stage to report
        progress.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._progress_cb = progress_callback
        self._pose_model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str | Path) -> dict:
        """Execute the full pipeline on a single video.

        Parameters
        ----------
        video_path : str or Path
            Path to the source video file.

        Returns
        -------
        dict
            Summary with keys: ``video``, ``total_frames``, ``tracks``,
            ``clips``, ``output_dir``, ``elapsed_sec``.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        t0 = time.monotonic()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1 — extract frames.
        frames, source_fps = self._extract_frames(video_path)
        effective_fps = min(self.config.target_fps, source_fps)

        # Stage 2 — detect poses.
        detections_per_frame = self._detect_poses(frames)

        # Stage 3 — track players.
        tracks = self._track_players(detections_per_frame)

        # Stage 4 + 5 — normalise and segment.
        all_clips: list[MotionClip] = []
        segmenter = MotionSegmenter(
            fps=effective_fps,
            min_segment_length=self.config.min_segment_length,
        )

        self._report("segmenting", 0, len(tracks))
        for t_idx, track in enumerate(tracks):
            if track.length < self.config.min_track_length:
                continue
            kp_array = track.get_keypoints_array()
            normalised = self._normalise_skeleton(kp_array)
            clips = segmenter.segment(normalised, track.frame_indices)
            all_clips.extend(clips)
            self._report("segmenting", t_idx + 1, len(tracks))

        # Stage 6 — export.
        self._export_clips(all_clips, tracks, video_path, out_dir)

        elapsed = time.monotonic() - t0
        summary = {
            "video": str(video_path),
            "total_frames": len(frames),
            "tracks": len(tracks),
            "clips": len(all_clips),
            "output_dir": str(out_dir),
            "elapsed_sec": round(elapsed, 2),
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Stage 1 — Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self, video_path: Path
    ) -> tuple[list[np.ndarray], float]:
        """Extract frames from the video at the configured FPS."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0:
            raise RuntimeError(f"Invalid source FPS ({source_fps})")

        effective_fps = min(self.config.target_fps, source_fps)
        frame_interval = source_fps / effective_fps

        frames: list[np.ndarray] = []
        next_sample: float = 0.0
        frame_idx = 0

        self._report("extracting_frames", 0, total_frames)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx >= next_sample:
                    frames.append(frame)
                    next_sample += frame_interval
                frame_idx += 1
                if frame_idx % 100 == 0:
                    self._report("extracting_frames", frame_idx, total_frames)
        finally:
            cap.release()

        self._report("extracting_frames", frame_idx, total_frames)
        logger.info(
            "Extracted %d frames from %s (%.1f fps -> %.1f fps)",
            len(frames),
            video_path.name,
            source_fps,
            effective_fps,
        )
        return frames, source_fps

    # ------------------------------------------------------------------
    # Stage 2 — Pose detection
    # ------------------------------------------------------------------

    def _load_pose_model(self):
        """Lazy-load the YOLOv8-pose model."""
        if self._pose_model is not None:
            return self._pose_model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for pose detection. "
                "Install with: pip install ultralytics"
            ) from exc

        model_path = self.config.pose_model_path or "yolov8n-pose.pt"
        self._pose_model = YOLO(model_path)
        logger.info("Loaded pose model: %s", model_path)
        return self._pose_model

    def _detect_poses(
        self, frames: list[np.ndarray]
    ) -> list[list[Detection]]:
        """Run YOLOv8-pose on each frame and return detections."""
        model = self._load_pose_model()
        detections_per_frame: list[list[Detection]] = []

        self._report("detecting_poses", 0, len(frames))
        for i, frame in enumerate(frames):
            results = model(frame, verbose=False, conf=self.config.pose_confidence)
            frame_dets: list[Detection] = []

            for result in results:
                if result.boxes is None or result.keypoints is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

                for j in range(len(boxes)):
                    frame_dets.append(
                        Detection(
                            bbox=boxes[j],
                            keypoints=kps[j],
                            confidence=float(confs[j]),
                        )
                    )

            detections_per_frame.append(frame_dets)
            if (i + 1) % 50 == 0 or i == len(frames) - 1:
                self._report("detecting_poses", i + 1, len(frames))

        logger.info(
            "Detected poses in %d frames (%d total detections)",
            len(frames),
            sum(len(d) for d in detections_per_frame),
        )
        return detections_per_frame

    # ------------------------------------------------------------------
    # Stage 3 — Player tracking
    # ------------------------------------------------------------------

    def _track_players(
        self, detections_per_frame: list[list[Detection]]
    ) -> list[Track]:
        """Track players across frames and return completed tracks."""
        tracker = PlayerTracker(
            max_age=self.config.tracker_max_age,
            min_hits=self.config.tracker_min_hits,
            iou_threshold=self.config.tracker_iou_threshold,
        )

        self._report("tracking", 0, len(detections_per_frame))
        for frame_idx, dets in enumerate(detections_per_frame):
            tracker.update(dets, frame_idx)
            if (frame_idx + 1) % 100 == 0:
                self._report(
                    "tracking", frame_idx + 1, len(detections_per_frame)
                )

        tracks = tracker.finalize()
        self._report("tracking", len(detections_per_frame), len(detections_per_frame))
        logger.info("Tracking complete: %d tracks", len(tracks))
        return tracks

    # ------------------------------------------------------------------
    # Stage 4 — Skeleton normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_skeleton(keypoints: np.ndarray) -> np.ndarray:
        """Normalise a keypoint sequence to be hip-centred and unit-scaled.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(T, K, 3)`` with ``[x, y, confidence]``.

        Returns
        -------
        np.ndarray
            Normalised array of the same shape.
        """
        normalised = keypoints.copy()
        positions = normalised[:, :, :2]

        # Hip midpoint (COCO indices 11=left_hip, 12=right_hip).
        hip_centre = (positions[:, 11, :] + positions[:, 12, :]) / 2.0
        positions -= hip_centre[:, np.newaxis, :]

        # Scale so that the torso length (hip to shoulder midpoint) ≈ 1.
        shoulder_centre = (positions[:, 5, :] + positions[:, 6, :]) / 2.0
        hip_mid = (positions[:, 11, :] + positions[:, 12, :]) / 2.0
        torso_len = np.linalg.norm(shoulder_centre - hip_mid, axis=1)
        mean_torso = np.mean(torso_len)
        if mean_torso > 1e-6:
            positions /= mean_torso

        normalised[:, :, :2] = positions
        return normalised

    # ------------------------------------------------------------------
    # Stage 6 — Export
    # ------------------------------------------------------------------

    def _export_clips(
        self,
        clips: list[MotionClip],
        tracks: list[Track],
        video_path: Path,
        out_dir: Path,
    ) -> None:
        """Export motion clips as ``.npy`` files with a JSON manifest."""
        self._report("exporting", 0, len(clips))

        manifest_entries: list[dict] = []
        for i, clip in enumerate(clips):
            filename = f"clip_{i:05d}_{clip.motion_type}.npy"
            np.save(str(out_dir / filename), clip.keypoints)

            manifest_entries.append(
                {
                    "index": i,
                    "file": filename,
                    "motion_type": clip.motion_type,
                    "motion_type_index": clip.motion_type_index,
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "num_frames": clip.length,
                }
            )
            if (i + 1) % 50 == 0 or i == len(clips) - 1:
                self._report("exporting", i + 1, len(clips))

        manifest = {
            "video": str(video_path),
            "total_tracks": len(tracks),
            "total_clips": len(clips),
            "motion_type_counts": self._count_types(clips),
            "clips": manifest_entries,
        }

        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Exported %d clips to %s (manifest: %s)",
            len(clips),
            out_dir,
            manifest_path,
        )

    @staticmethod
    def _count_types(clips: list[MotionClip]) -> dict[str, int]:
        """Count clips per motion type."""
        counts: dict[str, int] = {}
        for clip in clips:
            counts[clip.motion_type] = counts.get(clip.motion_type, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _report(self, stage: str, current: int, total: int) -> None:
        """Invoke the progress callback if one is registered."""
        if self._progress_cb is not None:
            self._progress_cb(stage, current, total)
