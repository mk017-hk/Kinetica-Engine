"""Extract frames from video at a configurable target FPS using OpenCV.

Handles variable frame-rate sources by computing the correct temporal sampling
interval and emitting only the frames that fall on the target grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Container for a single extracted frame."""

    image: np.ndarray  # BGR, HWC uint8
    frame_index: int  # Original frame index in the source video
    timestamp_sec: float  # Timestamp in seconds


@dataclass
class VideoMetadata:
    """Basic metadata read from the video container."""

    path: str
    width: int
    height: int
    source_fps: float
    total_frames: int
    duration_sec: float


class FrameExtractor:
    """Extract frames from a video file at a target sampling rate.

    Parameters
    ----------
    target_fps : float
        Desired output frame rate.  Must be > 0 and <= source FPS.
        If *target_fps* exceeds the source FPS a warning is logged and
        every source frame is emitted.
    max_frames : int or None
        Optional hard cap on the number of frames to emit.
    """

    def __init__(self, target_fps: float = 25.0, max_frames: int | None = None) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        self.target_fps = target_fps
        self.max_frames = max_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, video_path: str | Path) -> VideoMetadata:
        """Read video metadata without decoding frames.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If OpenCV cannot open the file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total / fps if fps > 0 else 0.0

            return VideoMetadata(
                path=str(video_path),
                width=width,
                height=height,
                source_fps=fps,
                total_frames=total,
                duration_sec=duration,
            )
        finally:
            cap.release()

    def extract(self, video_path: str | Path) -> Iterator[FrameResult]:
        """Yield frames sampled at *target_fps* from *video_path*.

        Frames are yielded lazily so arbitrarily long videos can be
        processed without holding the full decoded sequence in memory.

        Yields
        ------
        FrameResult
            Extracted frame with index and timestamp metadata.
        """
        video_path = Path(video_path)
        meta = self.probe(video_path)

        source_fps = meta.source_fps
        if source_fps <= 0:
            raise RuntimeError(f"Invalid source FPS ({source_fps}) for {video_path}")

        effective_fps = self.target_fps
        if effective_fps > source_fps:
            logger.warning(
                "target_fps (%.1f) exceeds source FPS (%.1f); using source FPS",
                effective_fps,
                source_fps,
            )
            effective_fps = source_fps

        frame_interval = source_fps / effective_fps

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        try:
            emitted = 0
            next_sample: float = 0.0
            frame_idx = 0

            while True:
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx >= next_sample:
                    timestamp = frame_idx / source_fps
                    yield FrameResult(
                        image=frame,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp,
                    )
                    emitted += 1
                    next_sample += frame_interval

                frame_idx += 1

            logger.info(
                "Extracted %d frames from %s (source: %d @ %.1f fps, target: %.1f fps)",
                emitted,
                video_path.name,
                meta.total_frames,
                source_fps,
                effective_fps,
            )
        finally:
            cap.release()

    def extract_to_list(self, video_path: str | Path) -> list[FrameResult]:
        """Convenience wrapper that materialises all frames into a list."""
        return list(self.extract(video_path))
