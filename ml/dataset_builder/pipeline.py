"""Orchestrate the full dataset-building pipeline.

Pipeline stages:
    video -> frames -> poses -> normalised skeletons -> windowed sequences

Each stage can be run independently or chained via :meth:`DatasetPipeline.run`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import COCO_KEYPOINTS, JOINT_COUNT
from .frame_extractor import FrameExtractor, FrameResult
from .pose_detector import PoseDetector, FrameDetections, PersonDetection
from .skeleton_normalizer import SkeletonNormalizer, NormalizedSkeleton
from .sequence_exporter import (
    SequenceExporter,
    TrackSequence,
    ExportStats,
    FORMAT_WORLD_POSITIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full dataset-building pipeline.

    Attributes
    ----------
    target_fps : float
        Sampling rate for frame extraction.
    max_frames : int or None
        Optional cap on total extracted frames.
    pose_model : str
        YOLOv8 pose model name or path.
    detection_conf : float
        Minimum bounding-box confidence.
    keypoint_conf : float
        Minimum per-keypoint confidence.
    device : str
        Inference device (``"cpu"``, ``"cuda"``).
    min_visible_keypoints : int
        Minimum visible keypoints per detection.
    normalizer_min_confidence : float
        Keypoint confidence threshold for the normalizer.
    interpolation_method : str
        Temporal interpolation method for missing joints.
    unit_height_reference : str
        Scale reference for the normalizer.
    window_length : int
        Frames per exported window.
    stride : int
        Step between windows.
    output_format : str
        ``"world_positions"`` or ``"6d_rotations"``.
    output_dir : str
        Root directory for exported data.
    min_track_length : int
        Minimum frames for a person track to be considered.
    """

    # Frame extraction
    target_fps: float = 25.0
    max_frames: int | None = None

    # Pose detection
    pose_model: str = "yolov8m-pose"
    detection_conf: float = 0.5
    keypoint_conf: float = 0.3
    device: str = "cpu"
    min_visible_keypoints: int = 6

    # Skeleton normalisation
    normalizer_min_confidence: float = 0.3
    interpolation_method: str = "linear"
    unit_height_reference: str = "torso"

    # Sequence export
    window_length: int = 64
    stride: int = 32
    output_format: str = FORMAT_WORLD_POSITIONS
    output_dir: str = "dataset_out"
    min_track_length: int = 16


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""

    video_path: str
    total_frames_extracted: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    export_stats: ExportStats | None = None
    elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


class DatasetPipeline:
    """End-to-end pipeline: video -> frames -> poses -> skeletons -> sequences.

    Parameters
    ----------
    config : PipelineConfig
        Full configuration for all pipeline stages.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        self.frame_extractor = FrameExtractor(
            target_fps=self.config.target_fps,
            max_frames=self.config.max_frames,
        )
        self.pose_detector = PoseDetector(
            model_name=self.config.pose_model,
            detection_conf=self.config.detection_conf,
            keypoint_conf=self.config.keypoint_conf,
            device=self.config.device,
            min_visible_keypoints=self.config.min_visible_keypoints,
        )
        self.skeleton_normalizer = SkeletonNormalizer(
            min_confidence=self.config.normalizer_min_confidence,
            interpolation_method=self.config.interpolation_method,
            unit_height_reference=self.config.unit_height_reference,
        )
        self.sequence_exporter = SequenceExporter(
            output_dir=self.config.output_dir,
            window_length=self.config.window_length,
            stride=self.config.stride,
            output_format=self.config.output_format,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, video_path: str | Path) -> PipelineResult:
        """Execute the complete pipeline on a single video.

        Parameters
        ----------
        video_path : str or Path
            Path to the input video file.

        Returns
        -------
        PipelineResult
            Summary including counts and any errors encountered.
        """
        video_path = Path(video_path)
        result = PipelineResult(video_path=str(video_path))
        t0 = time.monotonic()

        try:
            # Stage 1: Extract frames
            logger.info("Stage 1/4: Extracting frames from %s", video_path.name)
            frames = self.extract_frames(video_path)
            result.total_frames_extracted = len(frames)

            if not frames:
                result.errors.append("No frames extracted from video")
                return result

            # Stage 2: Detect poses
            logger.info("Stage 2/4: Detecting poses in %d frames", len(frames))
            all_detections = self.detect_poses(frames)
            result.total_detections = sum(d.num_persons for d in all_detections)

            if result.total_detections == 0:
                result.errors.append("No persons detected in any frame")
                return result

            # Stage 3: Build tracks and normalise
            logger.info("Stage 3/4: Building tracks and normalising skeletons")
            tracks = self.build_tracks(all_detections)
            result.total_tracks = len(tracks)

            if not tracks:
                result.errors.append("No valid tracks built from detections")
                return result

            # Stage 4: Export sequences
            logger.info("Stage 4/4: Exporting %d tracks as sequences", len(tracks))
            result.export_stats = self.sequence_exporter.export_tracks(
                tracks,
                source_video=str(video_path),
            )

        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", video_path, exc)
            result.errors.append(str(exc))

        result.elapsed_sec = time.monotonic() - t0
        logger.info(
            "Pipeline complete for %s in %.1fs: %d frames, %d detections, %d tracks",
            video_path.name,
            result.elapsed_sec,
            result.total_frames_extracted,
            result.total_detections,
            result.total_tracks,
        )
        return result

    # ------------------------------------------------------------------
    # Individual stages (usable independently)
    # ------------------------------------------------------------------

    def extract_frames(self, video_path: str | Path) -> list[FrameResult]:
        """Stage 1: Extract frames from video."""
        return self.frame_extractor.extract_to_list(video_path)

    def detect_poses(self, frames: list[FrameResult]) -> list[FrameDetections]:
        """Stage 2: Run pose detection on extracted frames."""
        detections = []
        for frame in frames:
            det = self.pose_detector.detect_frame(frame.image, frame.frame_index)
            detections.append(det)
        return detections

    def build_tracks(
        self, all_detections: list[FrameDetections]
    ) -> list[TrackSequence]:
        """Stage 3: Group detections into per-person tracks and normalise.

        This uses a simple nearest-neighbour association based on bounding
        box IoU.  For production use, a proper tracker (e.g. ByteTrack)
        should replace this heuristic.

        Parameters
        ----------
        all_detections : list[FrameDetections]
            Per-frame detection results from stage 2.

        Returns
        -------
        list[TrackSequence]
            Normalised skeleton tracks ready for export.
        """
        # Simple greedy tracking by bbox IoU
        tracks: dict[int, _TrackBuilder] = {}
        next_id = 0

        for frame_det in all_detections:
            frame_idx = frame_det.frame_index
            unmatched = list(range(len(frame_det.detections)))
            matched_track_ids: set[int] = set()

            # Try to match each detection to an existing track
            if tracks:
                cost_matrix = self._compute_iou_matrix(
                    [tracks[tid].last_bbox for tid in tracks if tid not in matched_track_ids],
                    [frame_det.detections[i].bbox for i in unmatched],
                )
                track_ids_list = [tid for tid in tracks if tid not in matched_track_ids]

                while cost_matrix.size > 0:
                    best = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                    best_iou = cost_matrix[best]
                    if best_iou < 0.3:
                        break

                    tid = track_ids_list[best[0]]
                    det_local_idx = best[1]
                    det_idx = unmatched[det_local_idx]

                    tracks[tid].add(frame_det.detections[det_idx], frame_idx)
                    matched_track_ids.add(tid)

                    # Remove matched row/col
                    cost_matrix = np.delete(cost_matrix, best[0], axis=0)
                    cost_matrix = np.delete(cost_matrix, best[1], axis=1)
                    track_ids_list.pop(best[0])
                    unmatched.pop(det_local_idx)

            # Create new tracks for unmatched detections
            for det_idx in unmatched:
                tracks[next_id] = _TrackBuilder(next_id)
                tracks[next_id].add(frame_det.detections[det_idx], frame_idx)
                next_id += 1

        # Convert to TrackSequence with normalisation
        result: list[TrackSequence] = []
        for tid, builder in tracks.items():
            if len(builder.keypoints_list) < self.config.min_track_length:
                continue

            normalised = self.skeleton_normalizer.normalize_sequence(builder.keypoints_list)

            # Filter out tracks where most frames are invalid
            valid_count = sum(1 for s in normalised if s.valid)
            if valid_count < len(normalised) * 0.5:
                continue

            positions = np.stack([s.positions for s in normalised], axis=0)
            confidence = np.stack([s.confidence for s in normalised], axis=0)

            result.append(
                TrackSequence(
                    person_id=tid,
                    positions=positions,
                    confidence=confidence,
                    frame_indices=builder.frame_indices,
                )
            )

        logger.info(
            "Built %d valid tracks from %d candidates (min length=%d)",
            len(result),
            len(tracks),
            self.config.min_track_length,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: list[np.ndarray], boxes_b: list[np.ndarray]
    ) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes.

        Returns shape ``(len(boxes_a), len(boxes_b))``.
        """
        if not boxes_a or not boxes_b:
            return np.empty((0, 0))

        a = np.array(boxes_a)  # (N, 4)
        b = np.array(boxes_b)  # (M, 4)

        # Intersection
        x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        union = area_a[:, None] + area_b[None, :] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        return iou


class _TrackBuilder:
    """Accumulate detections for a single tracked person."""

    def __init__(self, person_id: int) -> None:
        self.person_id = person_id
        self.keypoints_list: list[np.ndarray] = []
        self.frame_indices: list[int] = []
        self.last_bbox: np.ndarray = np.zeros(4)

    def add(self, detection: PersonDetection, frame_index: int) -> None:
        self.keypoints_list.append(detection.keypoints.copy())
        self.frame_indices.append(frame_index)
        self.last_bbox = detection.bbox.copy()
