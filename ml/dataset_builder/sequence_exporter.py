"""Export normalised skeleton sequences as NumPy .npy files with JSON metadata sidecars.

Continuous person tracks are split into fixed-length overlapping windows
and saved in a directory structure suitable for ML training.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import (
    FRAME_DIM,
    JOINT_COUNT,
    MOTION_INPUT_DIM,
    DEFAULT_SEQ_LEN,
)

logger = logging.getLogger(__name__)

# Output format identifiers
FORMAT_WORLD_POSITIONS = "world_positions"  # (T, 66)
FORMAT_6D_ROTATIONS = "6d_rotations"  # (T, 132)


@dataclass
class TrackSequence:
    """A contiguous skeleton track for a single person across frames.

    Attributes
    ----------
    person_id : int
        Identifier for the tracked person.
    positions : np.ndarray
        Shape ``(T, JOINT_COUNT, 3)`` — normalised world positions.
    confidence : np.ndarray
        Shape ``(T, JOINT_COUNT)`` — per-joint confidence scores.
    frame_indices : list[int]
        Source frame indices corresponding to each row.
    """

    person_id: int
    positions: np.ndarray  # (T, 22, 3)
    confidence: np.ndarray  # (T, 22)
    frame_indices: list[int] = field(default_factory=list)


@dataclass
class ExportStats:
    """Summary statistics for an export run."""

    total_tracks: int = 0
    total_windows: int = 0
    skipped_short: int = 0
    output_dir: str = ""


class SequenceExporter:
    """Export normalised skeleton tracks as windowed .npy files.

    Parameters
    ----------
    output_dir : str or Path
        Root directory where exported sequences are written.
    window_length : int
        Number of frames per window (default 64).
    stride : int
        Step size between consecutive windows (default 32).
    output_format : str
        ``"world_positions"`` for (T, 66) or ``"6d_rotations"`` for (T, 132).
    min_confidence : float
        Minimum mean confidence for a window to be exported.
    min_valid_ratio : float
        Minimum fraction of valid frames in a window (0.0–1.0).
    """

    def __init__(
        self,
        output_dir: str | Path = "dataset_out",
        window_length: int = DEFAULT_SEQ_LEN,
        stride: int = 32,
        output_format: str = FORMAT_WORLD_POSITIONS,
        min_confidence: float = 0.2,
        min_valid_ratio: float = 0.8,
    ) -> None:
        if window_length < 1:
            raise ValueError(f"window_length must be >= 1, got {window_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if output_format not in (FORMAT_WORLD_POSITIONS, FORMAT_6D_ROTATIONS):
            raise ValueError(
                f"output_format must be '{FORMAT_WORLD_POSITIONS}' or "
                f"'{FORMAT_6D_ROTATIONS}', got '{output_format}'"
            )

        self.output_dir = Path(output_dir)
        self.window_length = window_length
        self.stride = stride
        self.output_format = output_format
        self.min_confidence = min_confidence
        self.min_valid_ratio = min_valid_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_tracks(
        self,
        tracks: list[TrackSequence],
        source_video: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExportStats:
        """Window and export all tracks to disk.

        Parameters
        ----------
        tracks : list[TrackSequence]
            Skeleton tracks to export.
        source_video : str
            Path or identifier for the source video (stored in metadata).
        extra_metadata : dict, optional
            Additional key-value pairs to include in metadata sidecars.

        Returns
        -------
        ExportStats
            Summary of the export operation.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stats = ExportStats(
            total_tracks=len(tracks),
            output_dir=str(self.output_dir),
        )

        for track in tracks:
            windows = self._split_windows(track)
            if not windows:
                stats.skipped_short += 1
                continue

            for win_idx, (win_data, win_conf, win_frames) in enumerate(windows):
                # Quality gate
                mean_conf = float(np.mean(win_conf))
                valid_ratio = float(np.mean(win_conf > 0))

                if mean_conf < self.min_confidence:
                    continue
                if valid_ratio < self.min_valid_ratio:
                    continue

                # Flatten to target format
                flat = self._flatten(win_data)

                # Generate unique filename
                seq_id = uuid.uuid4().hex[:12]
                basename = f"p{track.person_id:04d}_w{win_idx:04d}_{seq_id}"
                npy_path = self.output_dir / f"{basename}.npy"
                meta_path = self.output_dir / f"{basename}.json"

                # Save array
                np.save(npy_path, flat)

                # Save metadata sidecar
                metadata = {
                    "sequence_id": basename,
                    "person_id": track.person_id,
                    "window_index": win_idx,
                    "num_frames": int(flat.shape[0]),
                    "feature_dim": int(flat.shape[1]),
                    "format": self.output_format,
                    "frame_indices": [int(f) for f in win_frames],
                    "mean_confidence": round(mean_conf, 4),
                    "valid_ratio": round(valid_ratio, 4),
                    "source_video": source_video,
                    "window_length": self.window_length,
                    "stride": self.stride,
                    "schemaVersion": "1.0.0",
                }
                if extra_metadata:
                    metadata.update(extra_metadata)

                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                stats.total_windows += 1

        logger.info(
            "Exported %d windows from %d tracks (%d skipped) to %s",
            stats.total_windows,
            stats.total_tracks,
            stats.skipped_short,
            self.output_dir,
        )
        return stats

    def export_single_sequence(
        self,
        positions: np.ndarray,
        output_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export a single sequence array directly (no windowing).

        Parameters
        ----------
        positions : np.ndarray
            Shape ``(T, 22, 3)`` or already flattened ``(T, 66)``/``(T, 132)``.
        output_path : str or Path
            Destination ``.npy`` file path.
        metadata : dict, optional
            If provided, a JSON sidecar is written alongside the ``.npy`` file.

        Returns
        -------
        Path
            The path to the written ``.npy`` file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if positions.ndim == 3:
            flat = self._flatten(positions)
        else:
            flat = positions.astype(np.float32)

        np.save(output_path, flat)

        if metadata is not None:
            meta_path = output_path.with_suffix(".json")
            metadata.setdefault("schemaVersion", "1.0.0")
            metadata.setdefault("num_frames", int(flat.shape[0]))
            metadata.setdefault("feature_dim", int(flat.shape[1]))
            metadata.setdefault("format", self.output_format)
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_windows(
        self, track: TrackSequence
    ) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
        """Split a track into overlapping fixed-length windows.

        Returns a list of ``(positions, confidence, frame_indices)`` tuples.
        Each ``positions`` array has shape ``(window_length, 22, 3)``
        and ``confidence`` has shape ``(window_length, 22)``.
        """
        T = track.positions.shape[0]
        if T < self.window_length:
            # Track is shorter than window — skip or pad
            if T < self.window_length // 2:
                return []  # Too short even to pad

            # Pad with last frame (repeat)
            pad_len = self.window_length - T
            padded_pos = np.concatenate(
                [track.positions, np.tile(track.positions[-1:], (pad_len, 1, 1))],
                axis=0,
            )
            padded_conf = np.concatenate(
                [track.confidence, np.tile(track.confidence[-1:], (pad_len, 1))],
                axis=0,
            )
            padded_frames = track.frame_indices + [track.frame_indices[-1]] * pad_len
            return [(padded_pos, padded_conf, padded_frames)]

        windows = []
        for start in range(0, T - self.window_length + 1, self.stride):
            end = start + self.window_length
            win_pos = track.positions[start:end]
            win_conf = track.confidence[start:end]
            win_frames = track.frame_indices[start:end]
            windows.append((win_pos, win_conf, win_frames))

        return windows

    def _flatten(self, positions: np.ndarray) -> np.ndarray:
        """Flatten ``(T, 22, 3)`` positions to ``(T, D)`` based on output format.

        For ``world_positions`` the output dimension is 66 (22 * 3).
        For ``6d_rotations`` this pads to 132 (22 * 6) with zeros for
        the missing rotation components — the actual rotation conversion
        is expected to happen in a downstream training step.
        """
        T = positions.shape[0]

        if self.output_format == FORMAT_WORLD_POSITIONS:
            return positions.reshape(T, MOTION_INPUT_DIM).astype(np.float32)

        # 6D rotation format: fill first 3 dims with position, pad remaining 3
        flat = np.zeros((T, FRAME_DIM), dtype=np.float32)
        flat[:, :MOTION_INPUT_DIM] = positions.reshape(T, MOTION_INPUT_DIM)
        return flat
