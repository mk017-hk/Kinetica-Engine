"""Segment continuous player tracks into labeled motion clips.

Computes a 70-dimensional feature vector per frame matching the C++
``MotionFeatureExtractor`` layout, then applies heuristic velocity-based
classification to split the sequence into typed motion segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..constants import (
    COCO_KEYPOINTS,
    FEATURE_DIM_SEGMENTER,
    JOINT_COUNT,
    MOTION_TYPE_COUNT,
    MOTION_TYPES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Velocity thresholds (cm/s) for heuristic classification
# ---------------------------------------------------------------------------
_IDLE_SPEED = 50.0
_WALK_SPEED = 200.0
_JOG_SPEED = 500.0
# >= _JOG_SPEED -> Sprint

_TURN_ANGULAR_VEL = 90.0  # deg/s — triggers TurnLeft / TurnRight
_DECEL_THRESHOLD = -300.0  # cm/s^2 — negative acceleration
_JUMP_VERTICAL_VEL = 200.0  # cm/s upward

# Normalisation constants (matching C++ pipeline)
_ROOT_VEL_NORM = 800.0  # cm/s
_ANGULAR_VEL_NORM = 360.0  # deg/s

_MIN_SEGMENT_LENGTH = 10  # frames


@dataclass
class MotionClip:
    """A labeled segment of motion extracted from a player track.

    Attributes
    ----------
    motion_type : str
        One of the 16 canonical motion type names.
    motion_type_index : int
        Integer index into ``MOTION_TYPES``.
    start_frame : int
        First frame index (inclusive) within the source track.
    end_frame : int
        Last frame index (exclusive) within the source track.
    keypoints : np.ndarray
        Keypoint sequence of shape ``(T, K, 3)``.
    features : np.ndarray
        Feature matrix of shape ``(T, 70)``.
    """

    motion_type: str
    motion_type_index: int
    start_frame: int
    end_frame: int
    keypoints: np.ndarray
    features: np.ndarray

    @property
    def length(self) -> int:
        """Number of frames in this clip."""
        return self.end_frame - self.start_frame


class MotionSegmenter:
    """Segment a keypoint track into typed motion clips.

    Parameters
    ----------
    fps : float
        Frame rate of the source video (needed for velocity computation).
    min_segment_length : int
        Minimum number of frames for a valid segment.
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_segment_length: int = _MIN_SEGMENT_LENGTH,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.min_segment_length = min_segment_length
        self._dt = 1.0 / fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        keypoints: np.ndarray,
        frame_indices: Optional[list[int]] = None,
    ) -> list[MotionClip]:
        """Segment a keypoint sequence into labeled motion clips.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(T, K, 3)`` — COCO keypoints with ``[x, y, conf]``.
        frame_indices : list[int] or None
            Original frame indices corresponding to each row.  If ``None``,
            a zero-based sequence is assumed.

        Returns
        -------
        list[MotionClip]
            Segments that are at least *min_segment_length* frames long.
        """
        if keypoints.ndim != 3 or keypoints.shape[1] < COCO_KEYPOINTS:
            raise ValueError(
                f"Expected keypoints of shape (T, >={COCO_KEYPOINTS}, 3), "
                f"got {keypoints.shape}"
            )

        n_frames = keypoints.shape[0]
        if n_frames < self.min_segment_length:
            logger.debug(
                "Sequence too short (%d < %d), skipping",
                n_frames,
                self.min_segment_length,
            )
            return []

        if frame_indices is None:
            frame_indices = list(range(n_frames))

        features = self._compute_features(keypoints)
        labels = self._classify_frames(keypoints, features)
        clips = self._merge_segments(keypoints, features, labels, frame_indices)

        logger.debug(
            "Segmented %d frames into %d clips", n_frames, len(clips)
        )
        return clips

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_features(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute 70-D feature vectors matching C++ MotionFeatureExtractor.

        Layout (per frame):
          - 22 joints x 3 Euler angles (normalised to [-1, 1]) = 66D
          - Root velocity 3D (normalised by 800 cm/s) = 3D
          - Root angular velocity magnitude (normalised by 360 deg/s) = 1D

        Because we only have COCO 2D keypoints (not full 22-joint Euler
        angles), we approximate the Euler block by repeating and padding
        the available joint angles, keeping the feature dimension aligned
        with the C++ pipeline for downstream compatibility.
        """
        n_frames = keypoints.shape[0]
        features = np.zeros((n_frames, FEATURE_DIM_SEGMENTER), dtype=np.float32)

        # --- Approximate Euler angles from 2D keypoints -----------------
        # Use inter-keypoint angles as a proxy for joint rotations.
        euler_approx = self._approximate_euler_angles(keypoints)
        features[:, :66] = euler_approx

        # --- Root velocity (hip midpoint) --------------------------------
        root_positions = self._compute_root_positions(keypoints)
        root_vel = np.zeros_like(root_positions)
        root_vel[1:] = (root_positions[1:] - root_positions[:-1]) / self._dt
        root_vel[0] = root_vel[1] if n_frames > 1 else 0.0
        features[:, 66:69] = np.clip(root_vel / _ROOT_VEL_NORM, -1.0, 1.0)

        # --- Root angular velocity magnitude ----------------------------
        forward = self._compute_forward_direction(keypoints)
        angular_vel = np.zeros(n_frames, dtype=np.float32)
        for i in range(1, n_frames):
            dot = float(np.clip(np.dot(forward[i], forward[i - 1]), -1.0, 1.0))
            angle_deg = np.degrees(np.arccos(dot))
            angular_vel[i] = angle_deg / self._dt
        angular_vel[0] = angular_vel[1] if n_frames > 1 else 0.0
        features[:, 69] = np.clip(angular_vel / _ANGULAR_VEL_NORM, -1.0, 1.0)

        return features

    def _approximate_euler_angles(self, keypoints: np.ndarray) -> np.ndarray:
        """Approximate 22x3 Euler-angle block from COCO 2D keypoints.

        Computes pairwise angles between connected keypoints along the
        COCO skeleton and maps them into a ``(T, 66)`` array normalised
        to ``[-1, 1]`` (angles / 180).
        """
        n_frames = keypoints.shape[0]
        euler = np.zeros((n_frames, JOINT_COUNT * 3), dtype=np.float32)

        # Map COCO indices to approximate joint angles.
        # We use the 2D position deltas between connected keypoints to
        # derive (angle_x, angle_y, 0) triples.
        _COCO_PAIRS = [
            (11, 13),  # left hip -> left knee
            (13, 15),  # left knee -> left ankle
            (12, 14),  # right hip -> right knee
            (14, 16),  # right knee -> right ankle
            (5, 7),    # left shoulder -> left elbow
            (7, 9),    # left shoulder -> left wrist
            (6, 8),    # right shoulder -> right elbow
            (8, 10),   # right shoulder -> right wrist
            (5, 6),    # shoulder span
            (11, 12),  # hip span
            (5, 11),   # left torso
            (6, 12),   # right torso
            (0, 5),    # nose -> left shoulder (head proxy)
            (0, 6),    # nose -> right shoulder
        ]

        for pair_idx, (a, b) in enumerate(_COCO_PAIRS):
            if pair_idx >= JOINT_COUNT:
                break
            dx = keypoints[:, b, 0] - keypoints[:, a, 0]
            dy = keypoints[:, b, 1] - keypoints[:, a, 1]
            angle = np.arctan2(dy, dx)
            slot = pair_idx * 3
            euler[:, slot] = angle / np.pi      # normalised to [-1, 1]
            euler[:, slot + 1] = angle / np.pi
            euler[:, slot + 2] = 0.0

        return euler

    def _compute_root_positions(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute root (hip midpoint) position per frame.

        Returns shape ``(T, 3)`` — the third coordinate is zero for 2D input.
        """
        left_hip = keypoints[:, 11, :2]
        right_hip = keypoints[:, 12, :2]
        mid = (left_hip + right_hip) / 2.0
        return np.column_stack([mid, np.zeros(len(mid), dtype=np.float32)])

    def _compute_forward_direction(self, keypoints: np.ndarray) -> np.ndarray:
        """Estimate forward-facing direction per frame (2D unit vector)."""
        left_hip = keypoints[:, 11, :2]
        right_hip = keypoints[:, 12, :2]
        hip_vec = right_hip - left_hip
        # Forward is perpendicular to the hip vector (rotated 90 deg).
        forward = np.column_stack([-hip_vec[:, 1], hip_vec[:, 0]])
        norms = np.linalg.norm(forward, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return forward / norms

    # ------------------------------------------------------------------
    # Frame-level classification
    # ------------------------------------------------------------------

    def _classify_frames(
        self, keypoints: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Assign a motion type index to each frame using velocity heuristics.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(T,)`` with indices into ``MOTION_TYPES``.
        """
        n_frames = keypoints.shape[0]
        labels = np.full(n_frames, MOTION_TYPES.index("Unknown"), dtype=np.int32)

        root_vel_raw = features[:, 66:69] * _ROOT_VEL_NORM  # de-normalise
        speed = np.linalg.norm(root_vel_raw[:, :2], axis=1)  # horizontal speed
        vertical_vel = root_vel_raw[:, 2]

        angular_vel_raw = features[:, 69] * _ANGULAR_VEL_NORM

        # Acceleration (central difference on speed).
        accel = np.zeros(n_frames, dtype=np.float32)
        if n_frames > 2:
            accel[1:-1] = (speed[2:] - speed[:-2]) / (2.0 * self._dt)

        # Signed angular velocity (positive = right, negative = left).
        forward = self._compute_forward_direction(keypoints)
        signed_angular_vel = np.zeros(n_frames, dtype=np.float32)
        for i in range(1, n_frames):
            cross = float(
                forward[i - 1, 0] * forward[i, 1]
                - forward[i - 1, 1] * forward[i, 0]
            )
            signed_angular_vel[i] = np.degrees(np.arcsin(
                np.clip(cross, -1.0, 1.0)
            )) / self._dt

        for i in range(n_frames):
            # Priority-ordered checks.
            if vertical_vel[i] > _JUMP_VERTICAL_VEL:
                labels[i] = MOTION_TYPES.index("Jump")
            elif accel[i] < _DECEL_THRESHOLD:
                labels[i] = MOTION_TYPES.index("Decelerate")
            elif abs(signed_angular_vel[i]) > _TURN_ANGULAR_VEL:
                if signed_angular_vel[i] < 0:
                    labels[i] = MOTION_TYPES.index("TurnLeft")
                else:
                    labels[i] = MOTION_TYPES.index("TurnRight")
            elif speed[i] < _IDLE_SPEED:
                labels[i] = MOTION_TYPES.index("Idle")
            elif speed[i] < _WALK_SPEED:
                labels[i] = MOTION_TYPES.index("Walk")
            elif speed[i] < _JOG_SPEED:
                labels[i] = MOTION_TYPES.index("Jog")
            else:
                labels[i] = MOTION_TYPES.index("Sprint")

        return labels

    # ------------------------------------------------------------------
    # Segment merging
    # ------------------------------------------------------------------

    def _merge_segments(
        self,
        keypoints: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        frame_indices: list[int],
    ) -> list[MotionClip]:
        """Merge consecutive frames with the same label into segments.

        Segments shorter than *min_segment_length* are discarded.
        """
        clips: list[MotionClip] = []
        n_frames = len(labels)

        seg_start = 0
        for i in range(1, n_frames + 1):
            if i < n_frames and labels[i] == labels[seg_start]:
                continue

            seg_len = i - seg_start
            if seg_len >= self.min_segment_length:
                label_idx = int(labels[seg_start])
                clips.append(
                    MotionClip(
                        motion_type=MOTION_TYPES[label_idx],
                        motion_type_index=label_idx,
                        start_frame=frame_indices[seg_start],
                        end_frame=frame_indices[i - 1] + 1,
                        keypoints=keypoints[seg_start:i].copy(),
                        features=features[seg_start:i].copy(),
                    )
                )
            seg_start = i

        return clips
