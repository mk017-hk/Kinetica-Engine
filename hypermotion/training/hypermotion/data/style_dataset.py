"""Dataset for contrastive style encoder training.

Expects data organized as:
  player_clips/
    player_A/
      clip1.json
      clip2.json
    player_B/
      clip1.json
      ...

Each JSON clip has {"frames": [...]} with joint rotations and velocities.
Positive pairs = two clips from the same player.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.constants import JOINT_COUNT, STYLE_INPUT_DIM


def _extract_style_features(frames_data: list[dict]) -> np.ndarray | None:
    """Extract 201D per-frame features: 132 rot + 3 root vel + 66 angular vel.

    Returns [T, 201] or None if too short.
    """
    if len(frames_data) < 16:
        return None

    features = []
    prev_eulers = None

    for frame in frames_data:
        joints = frame.get("joints", [])

        # 22 joints x 6D rotation = 132D (use Euler*2 channels as proxy when 6D not stored)
        eulers = []
        for j in range(JOINT_COUNT):
            if j < len(joints):
                rot = joints[j].get("localEulerDeg", [0.0, 0.0, 0.0])
            else:
                rot = [0.0, 0.0, 0.0]
            eulers.extend(rot)

        # Root velocity (3D)
        rv = frame.get("rootVelocity", [0.0, 0.0, 0.0])

        # Angular velocity: finite differences on Euler angles (66D)
        euler_arr = np.array(eulers, dtype=np.float32)
        if prev_eulers is not None:
            angular_vel = (euler_arr - prev_eulers).tolist()
        else:
            angular_vel = [0.0] * 66
        prev_eulers = euler_arr

        # 132 (6D rot placeholder from euler*2) + 3 (root vel) + 66 (angular vel) = 201
        # We use the Euler x 2 to fill the 132D slot (66 euler -> duplicate)
        rot_132 = []
        for j in range(JOINT_COUNT):
            e = eulers[j * 3 : j * 3 + 3]
            # Fill 6D from Euler: [ex, ey, ez, ex, ey, ez] normalized
            rot_132.extend([v / 180.0 for v in e] * 2)

        row = rot_132 + [v / 800.0 for v in rv] + [v / 360.0 for v in angular_vel]
        features.append(row)

    return np.array(features, dtype=np.float32)


class StylePairDataset(Dataset):
    """Returns pairs of clips from the same player for contrastive training.

    Each __getitem__ returns (clip_a, clip_b) from the same player.
    """

    def __init__(self, data_dir: str | Path, min_clip_frames: int = 30,
                 max_clip_frames: int = 256, augment: bool = True):
        self.min_clip_frames = min_clip_frames
        self.max_clip_frames = max_clip_frames
        self.augment = augment

        # player_id -> list of feature arrays [T, 201]
        self.player_clips: dict[str, list[np.ndarray]] = {}

        data_path = Path(data_dir)
        for player_dir in sorted(data_path.iterdir()):
            if not player_dir.is_dir():
                continue
            player_id = player_dir.name
            clips = []
            for f in sorted(player_dir.glob("*.json")):
                try:
                    clip_data = json.loads(f.read_text())
                    feat = _extract_style_features(clip_data.get("frames", []))
                    if feat is not None:
                        clips.append(feat)
                except (json.JSONDecodeError, KeyError):
                    continue
            if len(clips) >= 2:
                self.player_clips[player_id] = clips

        # Flatten to (player_id, clip_idx) pairs for indexing
        self.pairs: list[tuple[str, int]] = []
        for pid, clips in self.player_clips.items():
            for i in range(len(clips)):
                self.pairs.append((pid, i))

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop_clip(self, clip: np.ndarray) -> np.ndarray:
        """Random temporal crop and optional augmentation."""
        T = clip.shape[0]
        max_len = min(T, self.max_clip_frames)
        min_len = min(self.min_clip_frames, max_len)

        if self.augment:
            crop_len = np.random.randint(min_len, max_len + 1)
            start = np.random.randint(0, T - crop_len + 1)
        else:
            crop_len = max_len
            start = 0

        window = clip[start : start + crop_len].copy()

        # Speed perturbation
        if self.augment and np.random.random() < 0.3:
            speed = np.random.uniform(0.8, 1.2)
            new_len = max(min_len, int(crop_len / speed))
            indices = np.linspace(0, crop_len - 1, new_len).astype(int)
            window = window[np.clip(indices, 0, crop_len - 1)]

        # Additive noise
        if self.augment and np.random.random() < 0.3:
            window = window + np.random.normal(0, 0.01, window.shape).astype(np.float32)

        return window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        player_id, clip_idx = self.pairs[idx]
        clips = self.player_clips[player_id]

        # Anchor clip
        clip_a = self._crop_clip(clips[clip_idx])

        # Positive: different clip from same player
        other_indices = [i for i in range(len(clips)) if i != clip_idx]
        pos_idx = np.random.choice(other_indices)
        clip_b = self._crop_clip(clips[pos_idx])

        return torch.from_numpy(clip_a), torch.from_numpy(clip_b)


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length clips to the longest in the batch."""
    clips_a, clips_b = zip(*batch)

    max_len = max(max(c.shape[0] for c in clips_a), max(c.shape[0] for c in clips_b))
    dim = clips_a[0].shape[1]

    def pad_list(clips: tuple[torch.Tensor, ...]) -> torch.Tensor:
        padded = torch.zeros(len(clips), max_len, dim)
        for i, c in enumerate(clips):
            padded[i, : c.shape[0]] = c
        return padded

    return pad_list(clips_a), pad_list(clips_b)
