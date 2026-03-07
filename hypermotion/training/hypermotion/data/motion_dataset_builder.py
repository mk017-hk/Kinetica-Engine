"""Dataset builder: converts exported animation clips into training samples
for the motion encoder.

Responsibilities:
  - Load animation clips from JSON (C++ export format)
  - Normalise joint positions (zero-centre on hips, scale to unit height)
  - Create fixed-length pose sequences with augmentation
  - Save training dataset as a single .npz archive
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from ..models.constants import JOINT_COUNT

log = logging.getLogger(__name__)

POSITIONS_PER_FRAME = JOINT_COUNT * 3  # 22 * 3 = 66


def _load_clip_positions(clip_path: Path) -> np.ndarray | None:
    """Load joint world positions from a JSON clip file.

    Returns array of shape [T, 22, 3] or None if parsing fails.
    """
    try:
        data = json.loads(clip_path.read_text())
        frames_data = data.get("frames", [])
        if len(frames_data) < 4:
            return None

        positions = []
        for frame in frames_data:
            joints = frame.get("joints", [])
            frame_pos = []
            for j in range(JOINT_COUNT):
                if j < len(joints):
                    wp = joints[j].get("worldPosition", [0.0, 0.0, 0.0])
                else:
                    wp = [0.0, 0.0, 0.0]
                frame_pos.append(wp)
            positions.append(frame_pos)

        return np.array(positions, dtype=np.float32)  # [T, 22, 3]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _load_clip_label(clip_path: Path) -> int:
    """Load motion type label from a JSON clip (default: -1 = unknown)."""
    try:
        data = json.loads(clip_path.read_text())
        return int(data.get("label", data.get("motionType", -1)))
    except (json.JSONDecodeError, KeyError, ValueError):
        return -1


def _load_clip_cluster(clip_path: Path) -> int:
    """Load cluster ID from a JSON clip (default: -1 = unassigned)."""
    try:
        data = json.loads(clip_path.read_text())
        return int(data.get("clusterID", -1))
    except (json.JSONDecodeError, KeyError, ValueError):
        return -1


def normalise_positions(positions: np.ndarray) -> np.ndarray:
    """Zero-centre on hips (joint 0) and scale to unit height per frame.

    positions: [T, 22, 3] -> [T, 22, 3]
    """
    # Centre on hips
    hips = positions[:, 0:1, :]  # [T, 1, 3]
    centred = positions - hips

    # Scale by skeleton height (distance from hips to head, joint 5)
    head = centred[:, 5, :]  # [T, 3]
    heights = np.linalg.norm(head, axis=-1, keepdims=True)  # [T, 1]
    mean_height = np.mean(heights)
    if mean_height < 1e-6:
        mean_height = 1.0
    return centred / mean_height


def build_dataset(
    data_dir: str | Path,
    output_path: str | Path,
    seq_len: int = 64,
    stride: int = 32,
    min_clip_frames: int = 16,
) -> dict:
    """Scan data_dir for JSON clips, extract fixed-length sequences, and save.

    Args:
        data_dir: directory with JSON clip files (recursive scan)
        output_path: path to write .npz archive
        seq_len: number of frames per training sample
        stride: sliding window stride for extracting sequences
        min_clip_frames: minimum clip length to include

    Returns:
        dict with dataset statistics
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("**/*.json"))
    log.info(f"Found {len(json_files)} JSON files in {data_dir}")

    sequences = []
    labels = []
    cluster_ids = []
    source_files = []

    for fpath in json_files:
        positions = _load_clip_positions(fpath)
        if positions is None or len(positions) < min_clip_frames:
            continue

        # Normalise
        norm_pos = normalise_positions(positions)
        flat = norm_pos.reshape(len(norm_pos), POSITIONS_PER_FRAME)  # [T, 66]

        label = _load_clip_label(fpath)
        cluster = _load_clip_cluster(fpath)

        # Extract windows with sliding stride
        T = flat.shape[0]
        if T >= seq_len:
            for start in range(0, T - seq_len + 1, stride):
                window = flat[start : start + seq_len]
                sequences.append(window)
                labels.append(label)
                cluster_ids.append(cluster)
                source_files.append(str(fpath.relative_to(data_dir)))
        else:
            # Pad short clips by repeating last frame
            pad_len = seq_len - T
            padded = np.concatenate(
                [flat, np.tile(flat[-1:], (pad_len, 1))], axis=0
            )
            sequences.append(padded)
            labels.append(label)
            cluster_ids.append(cluster)
            source_files.append(str(fpath.relative_to(data_dir)))

    if not sequences:
        log.warning("No valid sequences extracted")
        return {"num_sequences": 0, "num_clips": 0}

    sequences_arr = np.stack(sequences, axis=0)  # [N, seq_len, 66]
    labels_arr = np.array(labels, dtype=np.int32)
    clusters_arr = np.array(cluster_ids, dtype=np.int32)

    np.savez_compressed(
        str(output_path),
        sequences=sequences_arr,
        labels=labels_arr,
        cluster_ids=clusters_arr,
    )

    stats = {
        "num_sequences": len(sequences),
        "num_clips": len(json_files),
        "seq_len": seq_len,
        "feature_dim": POSITIONS_PER_FRAME,
        "output_path": str(output_path),
    }
    log.info(
        f"Built dataset: {stats['num_sequences']} sequences from "
        f"{stats['num_clips']} clips -> {output_path}"
    )
    return stats


def load_dataset(npz_path: str | Path) -> dict:
    """Load a previously built dataset.

    Returns dict with keys: sequences [N, seq_len, 66], labels [N], cluster_ids [N].
    """
    data = np.load(str(npz_path))
    return {
        "sequences": data["sequences"],
        "labels": data["labels"],
        "cluster_ids": data["cluster_ids"],
    }
