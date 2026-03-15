"""Dataset loader for motion sequence .npy files.

Loads per-track .npy files (shape [T, 66]) from a directory, samples
fixed-length windows, and applies data augmentation (temporal offset,
left/right mirroring, additive noise).

Directory structure expected::

    data_dir/
        track_0001.npy    # shape [T, 66]
        track_0002.npy
        ...

Each .npy file represents a single motion track with T frames of 22 joints
x 3 world-position coordinates (66 dimensions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import (
    DEFAULT_SEQ_LEN,
    JOINT_COUNT,
    MOTION_INPUT_DIM,
)

log = logging.getLogger(__name__)

# Joint index mapping for left/right mirroring.
# Swaps: LeftShoulder(6)<->RightShoulder(10), LeftArm(7)<->RightArm(11),
#         LeftForeArm(8)<->RightForeArm(12), LeftHand(9)<->RightHand(13),
#         LeftUpLeg(14)<->RightUpLeg(18), LeftLeg(15)<->RightLeg(19),
#         LeftFoot(16)<->RightFoot(20), LeftToeBase(17)<->RightToeBase(21)
_MIRROR_PAIRS: list[tuple[int, int]] = [
    (6, 10), (7, 11), (8, 12), (9, 13),
    (14, 18), (15, 19), (16, 20), (17, 21),
]

# Precompute the full permutation index array for mirroring
_MIRROR_INDICES: np.ndarray = np.arange(JOINT_COUNT, dtype=np.int64)
for _l, _r in _MIRROR_PAIRS:
    _MIRROR_INDICES[_l] = _r
    _MIRROR_INDICES[_r] = _l


def _build_mirror_column_indices() -> np.ndarray:
    """Build column-level index array for mirroring a [T, 66] motion array.

    Each joint occupies 3 consecutive columns (x, y, z). Mirroring swaps the
    columns of left joints with their right counterparts and negates the
    lateral (x) axis.

    Returns:
        Array of shape [66] mapping source columns to destination columns.
    """
    col_indices = np.zeros(MOTION_INPUT_DIM, dtype=np.int64)
    for dst_joint in range(JOINT_COUNT):
        src_joint = int(_MIRROR_INDICES[dst_joint])
        for axis in range(3):
            col_indices[dst_joint * 3 + axis] = src_joint * 3 + axis
    return col_indices


_MIRROR_COL_INDICES: np.ndarray = _build_mirror_column_indices()


class MotionSequenceDataset(Dataset):
    """Dataset of fixed-length motion windows sampled from per-track .npy files.

    Each item is a tuple ``(sequence, track_id)`` where ``sequence`` has shape
    ``[seq_len, 66]`` and ``track_id`` is an integer identifying the source
    track. Track IDs can be used for triplet mining (same-track pairs are
    positives, different-track pairs are negatives).

    Args:
        data_dir: Directory containing .npy files.
        seq_len: Number of frames per sampled window.
        augment: Whether to apply data augmentation.
        noise_std: Standard deviation of additive Gaussian noise.
        mirror_prob: Probability of applying left/right mirroring.
        min_track_length: Minimum number of frames for a track to be included.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = DEFAULT_SEQ_LEN,
        augment: bool = True,
        noise_std: float = 0.005,
        mirror_prob: float = 0.5,
        min_track_length: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std
        self.mirror_prob = mirror_prob

        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        min_len = min_track_length if min_track_length is not None else seq_len

        # Load all tracks and record (track_data, track_id) pairs.
        self.tracks: list[np.ndarray] = []
        self.windows: list[tuple[int, int]] = []  # (track_idx, start_frame)

        npy_files = sorted(data_path.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_path}")

        for track_id, npy_file in enumerate(npy_files):
            track = np.load(str(npy_file)).astype(np.float32)
            if track.ndim != 2 or track.shape[1] != MOTION_INPUT_DIM:
                log.warning(
                    "Skipping %s: expected shape [T, %d], got %s",
                    npy_file.name,
                    MOTION_INPUT_DIM,
                    track.shape,
                )
                continue
            if track.shape[0] < min_len:
                log.debug(
                    "Skipping %s: too short (%d < %d frames)",
                    npy_file.name,
                    track.shape[0],
                    min_len,
                )
                continue

            self.tracks.append(track)
            # Enumerate all possible window start positions for this track
            num_windows = max(1, track.shape[0] - seq_len + 1)
            for start in range(num_windows):
                self.windows.append((len(self.tracks) - 1, start))

        log.info(
            "Loaded %d tracks (%d windows of %d frames) from %s",
            len(self.tracks),
            len(self.windows),
            seq_len,
            data_path,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a motion window and its track ID.

        Args:
            idx: Window index.

        Returns:
            Tuple of (sequence [seq_len, 66], track_id).
        """
        track_idx, start = self.windows[idx]
        track = self.tracks[track_idx]

        # Apply random temporal offset during augmentation
        if self.augment:
            max_start = max(0, track.shape[0] - self.seq_len)
            start = np.random.randint(0, max_start + 1)

        end = start + self.seq_len
        if end > track.shape[0]:
            # Pad by repeating the last frame if the track is too short
            seq = np.zeros((self.seq_len, MOTION_INPUT_DIM), dtype=np.float32)
            available = track.shape[0] - start
            seq[:available] = track[start : track.shape[0]]
            seq[available:] = track[-1]
        else:
            seq = track[start:end].copy()

        if self.augment:
            seq = self._apply_augmentation(seq)

        return torch.from_numpy(seq), track_idx

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a motion window.

        Augmentations:
            - Left/right mirroring: swap left and right joints and negate x-axis.
            - Additive Gaussian noise: small perturbation to joint positions.

        Args:
            seq: Motion window of shape [seq_len, 66].

        Returns:
            Augmented motion window.
        """
        # Left/right mirroring
        if np.random.random() < self.mirror_prob:
            seq = seq[:, _MIRROR_COL_INDICES]
            # Negate x-axis (every 3rd column starting at 0) to reflect
            seq[:, 0::3] = -seq[:, 0::3]

        # Additive Gaussian noise
        if self.noise_std > 0.0:
            seq = seq + np.random.normal(0, self.noise_std, seq.shape).astype(
                np.float32
            )

        return seq

    def get_track_count(self) -> int:
        """Return the number of loaded tracks."""
        return len(self.tracks)

    def sample_from_track(self, track_idx: int) -> torch.Tensor:
        """Sample a random window from a specific track.

        Useful for generating positive pairs during triplet mining.

        Args:
            track_idx: Index of the track to sample from.

        Returns:
            Motion window tensor of shape [seq_len, 66].
        """
        track = self.tracks[track_idx]
        max_start = max(0, track.shape[0] - self.seq_len)
        start = np.random.randint(0, max_start + 1)
        end = min(start + self.seq_len, track.shape[0])

        seq = np.zeros((self.seq_len, MOTION_INPUT_DIM), dtype=np.float32)
        length = end - start
        seq[:length] = track[start:end]
        if length < self.seq_len:
            seq[length:] = track[-1]

        if self.augment:
            seq = self._apply_augmentation(seq)

        return torch.from_numpy(seq)
