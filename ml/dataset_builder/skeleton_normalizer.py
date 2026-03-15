"""Normalize detected COCO skeletons and map to the 22-joint internal representation.

Steps:
1. Center the skeleton at the hip midpoint (COCO joints 11, 12).
2. Scale to unit height using torso length.
3. Interpolate missing or low-confidence keypoints.
4. Map 17 COCO keypoints to 22 internal joints, synthesising the five
   additional joints (Spine1, Spine2, LeftShoulder, RightShoulder, Head-top)
   from neighbouring COCO landmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..constants import COCO_KEYPOINTS, JOINT_COUNT, JOINT_NAMES, MOTION_INPUT_DIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO index shortcuts
# ---------------------------------------------------------------------------
_COCO_LEFT_HIP = 11
_COCO_RIGHT_HIP = 12
_COCO_LEFT_SHOULDER = 5
_COCO_RIGHT_SHOULDER = 6
_COCO_NOSE = 0
_COCO_LEFT_EYE = 1
_COCO_RIGHT_EYE = 2
_COCO_LEFT_EAR = 3
_COCO_RIGHT_EAR = 4
_COCO_LEFT_ELBOW = 7
_COCO_RIGHT_ELBOW = 8
_COCO_LEFT_WRIST = 9
_COCO_RIGHT_WRIST = 10
_COCO_LEFT_KNEE = 13
_COCO_RIGHT_KNEE = 14
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16

# ---------------------------------------------------------------------------
# COCO -> 22-joint mapping table
# ---------------------------------------------------------------------------
# Each entry is either:
#   int            — direct COCO index
#   tuple[int,...] — average of the listed COCO indices (synthetic joint)
_COCO_TO_INTERNAL: list[int | tuple[int, ...]] = [
    # 0  Hips          — midpoint of left_hip, right_hip
    (_COCO_LEFT_HIP, _COCO_RIGHT_HIP),
    # 1  Spine         — midpoint hips–shoulders
    (_COCO_LEFT_HIP, _COCO_RIGHT_HIP, _COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER),
    # 2  Spine1        — 1/3 from hips toward shoulders
    None,  # computed procedurally
    # 3  Spine2        — 2/3 from hips toward shoulders
    None,  # computed procedurally
    # 4  Neck          — midpoint of shoulders
    (_COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER),
    # 5  Head          — nose (best COCO proxy for head top)
    _COCO_NOSE,
    # 6  LeftShoulder  — synthetic: between neck and left_shoulder
    None,  # computed procedurally
    # 7  LeftArm       — COCO left_shoulder
    _COCO_LEFT_SHOULDER,
    # 8  LeftForeArm   — COCO left_elbow
    _COCO_LEFT_ELBOW,
    # 9  LeftHand      — COCO left_wrist
    _COCO_LEFT_WRIST,
    # 10 RightShoulder — synthetic: between neck and right_shoulder
    None,  # computed procedurally
    # 11 RightArm      — COCO right_shoulder
    _COCO_RIGHT_SHOULDER,
    # 12 RightForeArm  — COCO right_elbow
    _COCO_RIGHT_ELBOW,
    # 13 RightHand     — COCO right_wrist
    _COCO_RIGHT_WRIST,
    # 14 LeftUpLeg     — COCO left_hip
    _COCO_LEFT_HIP,
    # 15 LeftLeg       — COCO left_knee
    _COCO_LEFT_KNEE,
    # 16 LeftFoot      — COCO left_ankle
    _COCO_LEFT_ANKLE,
    # 17 LeftToeBase   — synthetic: extrapolate from knee->ankle
    None,  # computed procedurally
    # 18 RightUpLeg    — COCO right_hip
    _COCO_RIGHT_HIP,
    # 19 RightLeg      — COCO right_knee
    _COCO_RIGHT_KNEE,
    # 20 RightFoot     — COCO right_ankle
    _COCO_RIGHT_ANKLE,
    # 21 RightToeBase  — synthetic: extrapolate from knee->ankle
    None,  # computed procedurally
]

# Minimum torso length (in pixels) to avoid division by near-zero
_MIN_TORSO_LENGTH = 1e-4


@dataclass
class NormalizedSkeleton:
    """A single normalised 22-joint skeleton frame.

    Attributes
    ----------
    positions : np.ndarray
        (22, 3) world positions after centering and scaling.
    confidence : np.ndarray
        (22,) per-joint confidence in [0, 1].
    valid : bool
        False when the source detection was too sparse to normalise.
    """

    positions: np.ndarray  # (JOINT_COUNT, 3)
    confidence: np.ndarray  # (JOINT_COUNT,)
    valid: bool = True


class SkeletonNormalizer:
    """Normalize and re-map COCO keypoints to the 22-joint internal skeleton.

    Parameters
    ----------
    min_confidence : float
        Keypoints below this confidence are treated as missing.
    interpolation_method : str
        ``"linear"`` (default) — linearly interpolate missing joints from
        temporal neighbours.  ``"zero"`` — leave missing joints at zero.
    unit_height_reference : str
        How to compute the scale factor.  ``"torso"`` uses the distance
        from hip-centre to neck.  ``"full"`` uses the distance from
        ankle midpoint to nose (when available).
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        interpolation_method: str = "linear",
        unit_height_reference: str = "torso",
    ) -> None:
        self.min_confidence = min_confidence
        self.interpolation_method = interpolation_method
        self.unit_height_reference = unit_height_reference

    # ------------------------------------------------------------------
    # Single-frame normalisation
    # ------------------------------------------------------------------

    def normalize_frame(self, coco_keypoints: np.ndarray) -> NormalizedSkeleton:
        """Normalise a single set of COCO keypoints.

        Parameters
        ----------
        coco_keypoints : np.ndarray
            Shape ``(17, 3)`` — x, y, confidence per COCO keypoint.
            Coordinates can be pixel positions or arbitrary units.

        Returns
        -------
        NormalizedSkeleton
            The re-mapped 22-joint skeleton with confidence array.
        """
        if coco_keypoints is None or coco_keypoints.shape[0] < COCO_KEYPOINTS:
            return NormalizedSkeleton(
                positions=np.zeros((JOINT_COUNT, 3), dtype=np.float32),
                confidence=np.zeros(JOINT_COUNT, dtype=np.float32),
                valid=False,
            )

        kpts = coco_keypoints.copy().astype(np.float32)
        coords = kpts[:, :2]  # (17, 2)
        conf = kpts[:, 2]  # (17,)

        # Mark low-confidence keypoints as missing
        missing = conf < self.min_confidence
        coords[missing] = 0.0

        # ---- 1. Compute hip centre ----
        hip_visible = not missing[_COCO_LEFT_HIP] and not missing[_COCO_RIGHT_HIP]
        if not hip_visible:
            return NormalizedSkeleton(
                positions=np.zeros((JOINT_COUNT, 3), dtype=np.float32),
                confidence=np.zeros(JOINT_COUNT, dtype=np.float32),
                valid=False,
            )

        hip_centre = (coords[_COCO_LEFT_HIP] + coords[_COCO_RIGHT_HIP]) / 2.0

        # ---- 2. Centre at hips ----
        centred = np.zeros_like(coords)
        for i in range(COCO_KEYPOINTS):
            if not missing[i]:
                centred[i] = coords[i] - hip_centre

        # ---- 3. Scale to unit height ----
        scale = self._compute_scale(centred, conf)
        if scale < _MIN_TORSO_LENGTH:
            return NormalizedSkeleton(
                positions=np.zeros((JOINT_COUNT, 3), dtype=np.float32),
                confidence=np.zeros(JOINT_COUNT, dtype=np.float32),
                valid=False,
            )
        centred /= scale

        # ---- 4. Map to 22 joints ----
        positions_22 = np.zeros((JOINT_COUNT, 3), dtype=np.float32)  # z stays 0 (2D)
        confidence_22 = np.zeros(JOINT_COUNT, dtype=np.float32)

        # Process direct and averaged mappings first
        for j, mapping in enumerate(_COCO_TO_INTERNAL):
            if mapping is None:
                continue
            if isinstance(mapping, int):
                if not missing[mapping]:
                    positions_22[j, :2] = centred[mapping]
                    confidence_22[j] = conf[mapping]
            elif isinstance(mapping, tuple):
                visible_indices = [m for m in mapping if not missing[m]]
                if visible_indices:
                    positions_22[j, :2] = np.mean(centred[visible_indices], axis=0)
                    confidence_22[j] = np.mean([conf[m] for m in visible_indices])

        # Synthesise procedural joints
        self._synthesise_spine(positions_22, confidence_22)
        self._synthesise_shoulder_offsets(positions_22, confidence_22)
        self._synthesise_toes(positions_22, confidence_22)

        return NormalizedSkeleton(
            positions=positions_22,
            confidence=confidence_22,
            valid=True,
        )

    # ------------------------------------------------------------------
    # Sequence normalisation with temporal interpolation
    # ------------------------------------------------------------------

    def normalize_sequence(
        self, keypoints_seq: list[np.ndarray]
    ) -> list[NormalizedSkeleton]:
        """Normalise a temporal sequence of COCO keypoints.

        Missing joints are interpolated across time when
        ``interpolation_method == "linear"``.

        Parameters
        ----------
        keypoints_seq : list of np.ndarray
            Each element has shape ``(17, 3)``.

        Returns
        -------
        list[NormalizedSkeleton]
            One normalised skeleton per input frame.
        """
        normalised = [self.normalize_frame(kp) for kp in keypoints_seq]

        if self.interpolation_method == "linear" and len(normalised) > 1:
            self._temporal_interpolation(normalised)

        return normalised

    def to_position_array(self, skeletons: list[NormalizedSkeleton]) -> np.ndarray:
        """Stack normalised skeletons into a ``(T, 66)`` position array.

        Parameters
        ----------
        skeletons : list[NormalizedSkeleton]
            Output from :meth:`normalize_sequence`.

        Returns
        -------
        np.ndarray
            Shape ``(T, MOTION_INPUT_DIM)`` where ``MOTION_INPUT_DIM = 66``.
        """
        T = len(skeletons)
        out = np.zeros((T, MOTION_INPUT_DIM), dtype=np.float32)
        for t, skel in enumerate(skeletons):
            if skel.valid:
                out[t] = skel.positions.flatten()
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_scale(self, centred: np.ndarray, conf: np.ndarray) -> float:
        """Compute the normalisation scale factor."""
        if self.unit_height_reference == "torso":
            # Torso = hip_centre (origin after centering) to neck midpoint
            neck_visible = (
                conf[_COCO_LEFT_SHOULDER] >= self.min_confidence
                and conf[_COCO_RIGHT_SHOULDER] >= self.min_confidence
            )
            if neck_visible:
                neck = (centred[_COCO_LEFT_SHOULDER] + centred[_COCO_RIGHT_SHOULDER]) / 2.0
                return float(np.linalg.norm(neck) + _MIN_TORSO_LENGTH)
        elif self.unit_height_reference == "full":
            # Full height: ankle midpoint to nose
            ankles_visible = (
                conf[_COCO_LEFT_ANKLE] >= self.min_confidence
                and conf[_COCO_RIGHT_ANKLE] >= self.min_confidence
            )
            nose_visible = conf[_COCO_NOSE] >= self.min_confidence
            if ankles_visible and nose_visible:
                ankle_mid = (centred[_COCO_LEFT_ANKLE] + centred[_COCO_RIGHT_ANKLE]) / 2.0
                return float(np.linalg.norm(centred[_COCO_NOSE] - ankle_mid) + _MIN_TORSO_LENGTH)

        # Fallback: torso length
        for s in [_COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER]:
            if conf[s] >= self.min_confidence:
                return float(np.linalg.norm(centred[s]) + _MIN_TORSO_LENGTH)

        return 0.0  # Cannot compute scale

    def _synthesise_spine(
        self, positions: np.ndarray, confidence: np.ndarray
    ) -> None:
        """Fill Spine1 (idx 2) and Spine2 (idx 3) by interpolating between Hips and Neck."""
        hips = positions[0]  # Hips
        neck = positions[4]  # Neck

        hips_conf = confidence[0]
        neck_conf = confidence[4]

        if hips_conf > 0 and neck_conf > 0:
            positions[2] = hips + (neck - hips) * (1.0 / 3.0)  # Spine1
            positions[3] = hips + (neck - hips) * (2.0 / 3.0)  # Spine2
            avg_conf = (hips_conf + neck_conf) / 2.0
            confidence[2] = avg_conf
            confidence[3] = avg_conf

    def _synthesise_shoulder_offsets(
        self, positions: np.ndarray, confidence: np.ndarray
    ) -> None:
        """Synthesise LeftShoulder (6) and RightShoulder (10) between neck and arm."""
        neck = positions[4]
        neck_conf = confidence[4]

        # LeftShoulder: midpoint of neck and LeftArm (7)
        left_arm = positions[7]
        left_arm_conf = confidence[7]
        if neck_conf > 0 and left_arm_conf > 0:
            positions[6] = (neck + left_arm) / 2.0
            confidence[6] = (neck_conf + left_arm_conf) / 2.0

        # RightShoulder: midpoint of neck and RightArm (11)
        right_arm = positions[11]
        right_arm_conf = confidence[11]
        if neck_conf > 0 and right_arm_conf > 0:
            positions[10] = (neck + right_arm) / 2.0
            confidence[10] = (neck_conf + right_arm_conf) / 2.0

    def _synthesise_toes(
        self, positions: np.ndarray, confidence: np.ndarray
    ) -> None:
        """Synthesise LeftToeBase (17) and RightToeBase (21) by extrapolating from knee to ankle."""
        toe_factor = 0.2  # Extrapolation amount past ankle

        # Left toe
        left_knee = positions[15]  # LeftLeg
        left_ankle = positions[16]  # LeftFoot
        if confidence[15] > 0 and confidence[16] > 0:
            direction = left_ankle - left_knee
            positions[17] = left_ankle + direction * toe_factor
            confidence[17] = min(confidence[15], confidence[16]) * 0.8

        # Right toe
        right_knee = positions[19]  # RightLeg
        right_ankle = positions[20]  # RightFoot
        if confidence[19] > 0 and confidence[20] > 0:
            direction = right_ankle - right_knee
            positions[21] = right_ankle + direction * toe_factor
            confidence[21] = min(confidence[19], confidence[20]) * 0.8

    def _temporal_interpolation(self, skeletons: list[NormalizedSkeleton]) -> None:
        """Linearly interpolate missing joints across time in-place.

        For each joint, find contiguous gaps where the joint is missing
        and fill them using the nearest valid values on either side.
        """
        T = len(skeletons)
        if T < 2:
            return

        for j in range(JOINT_COUNT):
            # Collect validity per frame for this joint
            valid_frames = [
                t for t in range(T)
                if skeletons[t].valid and skeletons[t].confidence[j] > 0
            ]

            if len(valid_frames) < 2:
                continue

            for gap_start_idx in range(len(valid_frames) - 1):
                t0 = valid_frames[gap_start_idx]
                t1 = valid_frames[gap_start_idx + 1]
                if t1 - t0 <= 1:
                    continue  # No gap

                pos0 = skeletons[t0].positions[j]
                pos1 = skeletons[t1].positions[j]
                conf0 = skeletons[t0].confidence[j]
                conf1 = skeletons[t1].confidence[j]

                for t in range(t0 + 1, t1):
                    alpha = (t - t0) / (t1 - t0)
                    skeletons[t].positions[j] = pos0 * (1 - alpha) + pos1 * alpha
                    skeletons[t].confidence[j] = conf0 * (1 - alpha) + conf1 * alpha
