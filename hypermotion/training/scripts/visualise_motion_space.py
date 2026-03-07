#!/usr/bin/env python3
"""Visualise the motion embedding space using PCA or t-SNE.

Usage:
    python -m scripts.visualise_motion_space \\
        --data-dir /path/to/clips \\
        --model checkpoints/motion_encoder/motion_encoder_final.pt \\
        --method tsne \\
        --output motion_space.png

Generates a 2D scatter plot coloured by motion type or cluster label.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    PCA = None
    TSNE = None

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypermotion.models.motion_encoder import MotionEncoder, MOTION_INPUT_DIM
from hypermotion.models.constants import MOTION_TYPE_NAMES, JOINT_COUNT
from hypermotion.data.motion_dataset_builder import (
    _load_clip_positions,
    _load_clip_label,
    _load_clip_cluster,
    normalise_positions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("visualise_motion_space")


def embed_clips(
    model: MotionEncoder,
    data_dir: Path,
    seq_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute embeddings for all clips in data_dir.

    Returns:
        embeddings: [N, 128]
        labels: [N]
        cluster_ids: [N]
        filenames: [N]
    """
    model.eval()
    json_files = sorted(data_dir.glob("**/*.json"))

    embeddings = []
    labels = []
    cluster_ids = []
    filenames = []

    with torch.no_grad():
        for fpath in json_files:
            positions = _load_clip_positions(fpath)
            if positions is None or len(positions) < 4:
                continue

            norm = normalise_positions(positions)
            flat = norm.reshape(len(norm), JOINT_COUNT * 3)  # [T, 66]

            # Pad or truncate to seq_len
            T = flat.shape[0]
            if T >= seq_len:
                window = flat[:seq_len]
            else:
                pad = np.tile(flat[-1:], (seq_len - T, 1))
                window = np.concatenate([flat, pad], axis=0)

            tensor = torch.from_numpy(window).unsqueeze(0).to(device)  # [1, seq_len, 66]
            emb = model(tensor).cpu().numpy()[0]  # [128]

            embeddings.append(emb)
            labels.append(_load_clip_label(fpath))
            cluster_ids.append(_load_clip_cluster(fpath))
            filenames.append(fpath.name)

    return (
        np.array(embeddings),
        np.array(labels),
        np.array(cluster_ids),
        filenames,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualise motion embedding space")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with JSON animation clips")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to motion_encoder .pt checkpoint")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["pca", "tsne"],
                        help="Dimensionality reduction method")
    parser.add_argument("--output", type=str, default="motion_space.png",
                        help="Output image path")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--colour-by", type=str, default="label",
                        choices=["label", "cluster"],
                        help="Colour points by motion label or cluster ID")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    args = parser.parse_args()

    if plt is None:
        log.error("matplotlib is required: pip install matplotlib")
        sys.exit(1)
    if PCA is None:
        log.error("scikit-learn is required: pip install scikit-learn")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MotionEncoder()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    log.info(f"Loaded model from {args.model}")

    # Compute embeddings
    data_dir = Path(args.data_dir)
    embeddings, labels, cluster_ids, filenames = embed_clips(
        model, data_dir, args.seq_len, device
    )
    log.info(f"Embedded {len(embeddings)} clips")

    if len(embeddings) < 3:
        log.error("Need at least 3 clips to visualise")
        sys.exit(1)

    # Reduce to 2D
    if args.method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
        title_suffix = "PCA"
    else:
        perp = min(args.perplexity, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        coords = reducer.fit_transform(embeddings)
        title_suffix = "t-SNE"

    # Colour
    if args.colour_by == "cluster":
        colour_values = cluster_ids
        colour_label = "Cluster ID"
    else:
        colour_values = labels
        colour_label = "Motion Type"

    unique = np.unique(colour_values)
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 2))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, val in enumerate(unique):
        mask = colour_values == val
        name = MOTION_TYPE_NAMES[val] if (0 <= val < len(MOTION_TYPE_NAMES) and args.colour_by == "label") else f"cluster_{val:02d}"
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                   label=name, s=30, alpha=0.7)

    ax.set_title(f"Motion Embedding Space ({title_suffix})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    log.info(f"Saved visualisation: {args.output}")


if __name__ == "__main__":
    main()
