"""Dataset classes for loading motion clips from the JSON format exported by C++."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.constants import JOINT_COUNT, ROTATION_DIM, FRAME_DIM, CONDITION_DIM


def _euler_to_6d(euler_deg: np.ndarray) -> np.ndarray:
    """Convert Euler angles (degrees, XYZ) to 6D rotation representation.

    Uses the first two columns of the rotation matrix (Zhou et al.).
    euler_deg: [..., 3]  -> [..., 6]
    """
    rad = np.radians(euler_deg)
    cx, cy, cz = np.cos(rad[..., 0]), np.cos(rad[..., 1]), np.cos(rad[..., 2])
    sx, sy, sz = np.sin(rad[..., 0]), np.sin(rad[..., 1]), np.sin(rad[..., 2])

    # Rotation matrix columns (XYZ intrinsic = ZYX extrinsic)
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r20 = -sy
    r21 = sx * cy

    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


class MotionClipDataset(Dataset):
    """Loads motion clips for diffusion model training.

    Each JSON clip file contains frames with joint Euler angles.
    Returns (x0, condition) where:
      x0:        [seq_len, FRAME_DIM]  6D rotation vectors
      condition: [CONDITION_DIM]       motion condition (zero placeholder if absent)
    """

    def __init__(self, data_dir: str | Path, seq_len: int = 64, augment: bool = True):
        self.seq_len = seq_len
        self.augment = augment
        self.clips: list[np.ndarray] = []
        self.conditions: list[np.ndarray] = []

        data_path = Path(data_dir)
        json_files = sorted(data_path.glob("**/*.json"))

        for f in json_files:
            try:
                clip_data = json.loads(f.read_text())
                frames_data = clip_data.get("frames", [])
                if len(frames_data) < 16:
                    continue

                # Parse frames -> [num_frames, JOINT_COUNT, 3] Euler degrees
                frames = []
                for frame in frames_data:
                    joints = frame.get("joints", [])
                    joint_eulers = []
                    for j in range(JOINT_COUNT):
                        if j < len(joints):
                            rot = joints[j].get("localEulerDeg", [0.0, 0.0, 0.0])
                        else:
                            rot = [0.0, 0.0, 0.0]
                        joint_eulers.append(rot)
                    frames.append(joint_eulers)

                euler_array = np.array(frames, dtype=np.float32)     # [T, 22, 3]
                rot6d = _euler_to_6d(euler_array)                     # [T, 22, 6]
                flat = rot6d.reshape(len(frames), FRAME_DIM)         # [T, 132]
                self.clips.append(flat)

                # Condition if present, else zeros
                cond = np.zeros(CONDITION_DIM, dtype=np.float32)
                if "condition" in clip_data:
                    c = clip_data["condition"]
                    cond_list = c if isinstance(c, list) else c.get("values", [])
                    for i, v in enumerate(cond_list[:CONDITION_DIM]):
                        cond[i] = float(v)
                self.conditions.append(cond)

            except (json.JSONDecodeError, KeyError):
                continue

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = self.clips[idx]
        cond = self.conditions[idx]
        num_frames = clip.shape[0]

        # Extract a window of seq_len frames
        if num_frames >= self.seq_len:
            if self.augment:
                start = np.random.randint(0, num_frames - self.seq_len + 1)
            else:
                start = 0
            window = clip[start : start + self.seq_len]
        else:
            # Pad with last frame repeated
            pad_len = self.seq_len - num_frames
            window = np.concatenate([clip, np.tile(clip[-1:], (pad_len, 1))], axis=0)

        # Optional speed perturbation
        if self.augment and np.random.random() < 0.3:
            speed = np.random.uniform(0.8, 1.2)
            new_len = int(self.seq_len * speed)
            indices = np.linspace(0, self.seq_len - 1, new_len).astype(int)
            indices = np.clip(indices, 0, self.seq_len - 1)
            stretched = window[indices]
            if len(stretched) >= self.seq_len:
                window = stretched[: self.seq_len]
            else:
                pad_len = self.seq_len - len(stretched)
                window = np.concatenate(
                    [stretched, np.tile(stretched[-1:], (pad_len, 1))], axis=0
                )

        # Optional noise
        if self.augment and np.random.random() < 0.2:
            window = window + np.random.normal(0, 0.01, window.shape).astype(np.float32)

        return torch.from_numpy(window), torch.from_numpy(cond)


class ClassifierDataset(Dataset):
    """Loads labeled motion segments for TCN classifier training.

    Each JSON file should contain:
      {"frames": [...], "label": <int 0-15>, "label_name": "Sprint", ...}

    Returns (features, labels) where:
      features: [seq_len, 70]  per-frame feature vectors
      labels:   [seq_len]      per-frame class labels
    """

    def __init__(self, data_dir: str | Path, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.features: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

        data_path = Path(data_dir)
        json_files = sorted(data_path.glob("**/*.json"))

        for f in json_files:
            try:
                seg_data = json.loads(f.read_text())
                frames_data = seg_data.get("frames", [])
                label = int(seg_data.get("label", 15))  # default Unknown
                if len(frames_data) < 4:
                    continue

                feats = []
                for frame in frames_data:
                    joints = frame.get("joints", [])
                    # 22 joints x 3 Euler = 66D
                    euler_flat = []
                    for j in range(JOINT_COUNT):
                        if j < len(joints):
                            rot = joints[j].get("localEulerDeg", [0.0, 0.0, 0.0])
                        else:
                            rot = [0.0, 0.0, 0.0]
                        euler_flat.extend([r / 180.0 for r in rot])  # normalize

                    # Root velocity (3D) normalized
                    rv = frame.get("rootVelocity", [0.0, 0.0, 0.0])
                    euler_flat.extend([v / 800.0 for v in rv])

                    # Angular velocity magnitude
                    av = frame.get("rootAngularVelocity", [0.0, 0.0, 0.0])
                    ang_mag = math.sqrt(sum(a * a for a in av)) / 360.0
                    euler_flat.append(ang_mag)

                    feats.append(euler_flat)

                feat_array = np.array(feats, dtype=np.float32)  # [T, 70]
                label_array = np.full(len(feats), label, dtype=np.int64)

                self.features.append(feat_array)
                self.labels.append(label_array)

            except (json.JSONDecodeError, KeyError):
                continue

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features[idx]
        labels = self.labels[idx]
        T = feats.shape[0]

        if T > self.max_seq_len:
            start = np.random.randint(0, T - self.max_seq_len + 1)
            feats = feats[start : start + self.max_seq_len]
            labels = labels[start : start + self.max_seq_len]

        return torch.from_numpy(feats), torch.from_numpy(labels)
