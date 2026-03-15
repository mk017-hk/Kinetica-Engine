"""Multi-player tracking using IoU-based Hungarian assignment.

Tracks detected players across video frames, maintaining per-track state
(active / lost / finished) and storing bounding boxes, keypoints, and
frame indices for each detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..constants import COCO_KEYPOINTS

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Lifecycle state of a player track."""

    ACTIVE = auto()
    LOST = auto()
    FINISHED = auto()


@dataclass
class Detection:
    """A single player detection in one frame.

    Attributes
    ----------
    bbox : np.ndarray
        Bounding box as ``[x1, y1, x2, y2]``.
    keypoints : np.ndarray
        Keypoint array of shape ``(K, 3)`` where K is the number of
        keypoints and each row is ``[x, y, confidence]``.
    confidence : float
        Overall detection confidence.
    """

    bbox: np.ndarray
    keypoints: np.ndarray
    confidence: float = 1.0


@dataclass
class Track:
    """A tracked player across multiple frames.

    Attributes
    ----------
    track_id : int
        Unique track identifier.
    bboxes : list[np.ndarray]
        Bounding box per frame, each ``[x1, y1, x2, y2]``.
    keypoints : list[np.ndarray]
        Keypoint array per frame, each ``(K, 3)``.
    frame_indices : list[int]
        Source frame index for each stored detection.
    state : TrackState
        Current lifecycle state.
    hits : int
        Number of consecutive frames with a matched detection.
    age : int
        Number of frames since the last matched detection.
    total_hits : int
        Total number of matched detections over the track lifetime.
    """

    track_id: int
    bboxes: list[np.ndarray] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    state: TrackState = TrackState.ACTIVE
    hits: int = 0
    age: int = 0
    total_hits: int = 0

    @property
    def last_bbox(self) -> np.ndarray:
        """Return the most recent bounding box."""
        return self.bboxes[-1]

    @property
    def length(self) -> int:
        """Number of stored detections."""
        return len(self.frame_indices)

    def get_keypoints_array(self) -> np.ndarray:
        """Stack all keypoint arrays into ``(T, K, 3)``."""
        return np.stack(self.keypoints, axis=0)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute intersection-over-union between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _iou_cost_matrix(
    tracks: list[Track], detections: list[Detection]
) -> np.ndarray:
    """Build a cost matrix of shape ``(len(tracks), len(detections))``
    where each entry is ``1 - IoU``."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.ones((n_tracks, n_dets), dtype=np.float64)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            cost[t_idx, d_idx] = 1.0 - _iou(track.last_bbox, det.bbox)
    return cost


class PlayerTracker:
    """IoU-based multi-player tracker with Hungarian assignment.

    Parameters
    ----------
    max_age : int
        Number of consecutive unmatched frames before a track is finished.
    min_hits : int
        Minimum number of matched detections before a track is considered
        confirmed (returned in results).
    iou_threshold : float
        Minimum IoU required to associate a detection with a track.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(
                f"iou_threshold must be in (0, 1], got {iou_threshold}"
            )

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._next_id: int = 0
        self._active_tracks: list[Track] = []
        self._finished_tracks: list[Track] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: list[Detection], frame_index: int
    ) -> list[Track]:
        """Process detections for a single frame.

        Parameters
        ----------
        detections : list[Detection]
            Player detections for the current frame.
        frame_index : int
            Source frame index (used for bookkeeping).

        Returns
        -------
        list[Track]
            Currently active (confirmed) tracks after this update.
        """
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        # Update matched tracks.
        for t_idx, d_idx in matched:
            self._update_track(
                self._active_tracks[t_idx], detections[d_idx], frame_index
            )

        # Mark unmatched tracks as lost / finished.
        for t_idx in sorted(unmatched_trks, reverse=True):
            track = self._active_tracks[t_idx]
            track.age += 1
            if track.age > self.max_age:
                track.state = TrackState.FINISHED
                if track.total_hits >= self.min_hits:
                    self._finished_tracks.append(track)
                self._active_tracks.pop(t_idx)
            else:
                track.state = TrackState.LOST

        # Create new tracks from unmatched detections.
        for d_idx in unmatched_dets:
            self._init_track(detections[d_idx], frame_index)

        return [
            t for t in self._active_tracks if t.total_hits >= self.min_hits
        ]

    def finalize(self) -> list[Track]:
        """Finish all remaining active tracks and return every completed track.

        Call this after the last frame has been processed.

        Returns
        -------
        list[Track]
            All tracks that accumulated at least *min_hits* detections.
        """
        for track in self._active_tracks:
            track.state = TrackState.FINISHED
            if track.total_hits >= self.min_hits:
                self._finished_tracks.append(track)
        self._active_tracks.clear()

        logger.info(
            "Tracker finalized: %d completed tracks", len(self._finished_tracks)
        )
        return list(self._finished_tracks)

    def reset(self) -> None:
        """Reset the tracker state so it can be reused on a new video."""
        self._next_id = 0
        self._active_tracks.clear()
        self._finished_tracks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections with existing tracks using Hungarian matching.

        Returns
        -------
        matched : list[tuple[int, int]]
            ``(track_index, detection_index)`` pairs.
        unmatched_dets : list[int]
            Indices into *detections* with no match.
        unmatched_trks : list[int]
            Indices into *self._active_tracks* with no match.
        """
        if not self._active_tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self._active_tracks)))

        cost = _iou_cost_matrix(self._active_tracks, detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(self._active_tracks)))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > 1.0 - self.iou_threshold:
                # IoU below threshold — treat as unmatched.
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_trks.discard(r)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)

    def _init_track(self, det: Detection, frame_index: int) -> Track:
        """Create a new track from an unmatched detection."""
        track = Track(
            track_id=self._next_id,
            bboxes=[det.bbox.copy()],
            keypoints=[det.keypoints.copy()],
            frame_indices=[frame_index],
            state=TrackState.ACTIVE,
            hits=1,
            age=0,
            total_hits=1,
        )
        self._next_id += 1
        self._active_tracks.append(track)
        return track

    def _update_track(
        self, track: Track, det: Detection, frame_index: int
    ) -> None:
        """Append a matched detection to an existing track."""
        track.bboxes.append(det.bbox.copy())
        track.keypoints.append(det.keypoints.copy())
        track.frame_indices.append(frame_index)
        track.hits += 1
        track.age = 0
        track.total_hits += 1
        track.state = TrackState.ACTIVE
