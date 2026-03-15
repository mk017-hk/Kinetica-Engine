"""Dataset loader for motion prediction training.

Loads ``.npy`` files containing skeletal animation sequences with shape
``[T, FRAME_DIM]`` (22 joints x 6D rotation = 132 features per frame).

For training, random windows of ``context_len + 1`` frames are sampled.
The input is ``frames[:-1]`` and the target is ``frames[1:]`` (teacher
forcing / next-frame prediction).

Augmentations:
    - Random temporal offset: window start is uniformly sampled.
    - Rotation noise: small Gaussian noise added to 6D rotation values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import FRAME_DIM, DEFAULT_SEQ_LEN


class MotionPredictionDataset(Dataset):
    """PyTorch dataset for next-frame motion prediction.

    Each sample is a pair ``(input_frames, target_frames)`` of shape
    ``[context_len, FRAME_DIM]`` extracted from a sliding window over
    the source motion file.

    Args:
        data_dir: Directory containing ``.npy`` motion files.
        context_len: Number of input frames per sample (window size minus 1).
        rotation_noise_std: Standard deviation of Gaussian noise added to
            6D rotation values as augmentation.  Set to ``0.0`` to disable.
        frame_dim: Expected per-frame feature dimension.
    """

    def __init__(
        self,
        data_dir: str | Path,
        context_len: int = DEFAULT_SEQ_LEN,
        rotation_noise_std: float = 0.0,
        frame_dim: int = FRAME_DIM,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.rotation_noise_std = rotation_noise_std
        self.frame_dim = frame_dim

        # Window covers context_len input frames + 1 target frame
        self.window_size = context_len + 1

        self.sequences: list[np.ndarray] = []
        self.sample_index: list[tuple[int, int]] = []  # (seq_idx, start_frame)

        self._load_sequences(Path(data_dir))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_sequences(self, data_dir: Path) -> None:
        """Scan *data_dir* for ``.npy`` files and build a sample index."""
        npy_files = sorted(data_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No .npy files found in {data_dir}. Expected motion "
                f"sequences with shape [T, {self.frame_dim}]."
            )

        for fpath in npy_files:
            seq = np.load(str(fpath)).astype(np.float32)

            if seq.ndim != 2 or seq.shape[1] != self.frame_dim:
                raise ValueError(
                    f"Expected shape [T, {self.frame_dim}], got {seq.shape} "
                    f"in {fpath.name}."
                )

            if seq.shape[0] < self.window_size:
                # Sequence too short for even one window — skip silently.
                continue

            seq_idx = len(self.sequences)
            self.sequences.append(seq)

            # Create all valid starting positions for this sequence
            n_windows = seq.shape[0] - self.window_size + 1
            for start in range(n_windows):
                self.sample_index.append((seq_idx, start))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an ``(input, target)`` pair for next-frame prediction.

        Both tensors have shape ``[context_len, frame_dim]``.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_frames, target_frames).
        """
        seq_idx, start = self.sample_index[idx]
        window = self.sequences[seq_idx][start : start + self.window_size].copy()

        # Augmentation: small rotation noise
        if self.rotation_noise_std > 0.0:
            noise = np.random.randn(*window.shape).astype(np.float32)
            window = window + noise * self.rotation_noise_std

        frames = torch.from_numpy(window)
        input_frames = frames[:-1]   # [context_len, frame_dim]
        target_frames = frames[1:]   # [context_len, frame_dim]
        return input_frames, target_frames

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        """Number of source motion files loaded."""
        return len(self.sequences)

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        total_frames = sum(s.shape[0] for s in self.sequences)
        return (
            f"MotionPredictionDataset: {self.num_sequences} sequences, "
            f"{total_frames} total frames, {len(self)} samples "
            f"(window={self.window_size}, context_len={self.context_len})"
        )
