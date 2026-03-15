"""CLI entry point for motion search operations.

Usage::

    python -m ml.motion_search.cli build-index --data-dir sequences/ --model encoder.pt --output motion.index
    python -m ml.motion_search.cli search --index motion.index --query query.npy --k 10
    python -m ml.motion_search.cli stats --index motion.index
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .embedder import BatchEmbedder
from .index import MotionIndex

logger = logging.getLogger(__name__)


def _build_index(args: argparse.Namespace) -> None:
    """Build a FAISS index from a directory of .npy motion files."""
    embedder = BatchEmbedder(
        model_path=Path(args.model),
        batch_size=args.batch_size,
        device=args.device,
    )

    embeddings, metadata = embedder.embed_files(Path(args.data_dir))

    index = MotionIndex(use_ivf=args.use_ivf)
    index.build(embeddings, metadata)
    index.save(Path(args.output))

    print(f"Index built with {index.size} vectors -> {args.output}")


def _search(args: argparse.Namespace) -> None:
    """Search the index for nearest neighbours of a query."""
    index = MotionIndex.load(Path(args.index))

    query = np.load(args.query).astype(np.float32)

    # If the query is a raw sequence (T, D), embed it first.
    from ..constants import MOTION_INPUT_DIM, MOTION_EMBEDDING_DIM

    if query.ndim == 2 and query.shape[1] == MOTION_INPUT_DIM:
        if args.model is None:
            print(
                "Error: query appears to be a raw motion sequence "
                f"(shape {query.shape}). Provide --model to embed it first.",
                file=sys.stderr,
            )
            sys.exit(1)
        embedder = BatchEmbedder(model_path=Path(args.model), device=args.device)
        query = embedder.embed(query)
    elif query.ndim == 1 and query.shape[0] == MOTION_EMBEDDING_DIM:
        pass  # Already an embedding vector.
    elif query.ndim == 2 and query.shape == (1, MOTION_EMBEDDING_DIM):
        query = query.squeeze(0)
    else:
        print(
            f"Error: unexpected query shape {query.shape}. "
            f"Expected ({MOTION_EMBEDDING_DIM},) embedding or (T, {MOTION_INPUT_DIM}) sequence.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = index.search(query, k=args.k)

    print(f"Top-{len(results)} results:")
    for r in results:
        meta_str = json.dumps(r.metadata, indent=None)
        print(f"  [{r.rank}] score={r.score:.4f}  {meta_str}")


def _stats(args: argparse.Namespace) -> None:
    """Print statistics about an existing index."""
    index = MotionIndex.load(Path(args.index))

    print(f"Index file : {args.index}")
    print(f"Vectors    : {index.size}")
    print(f"Dimension  : {index.dim}")

    if index.size > 0 and index._metadata:
        sources = {m.get("source_file", "unknown") for m in index._metadata}
        motion_types = {}
        for m in index._metadata:
            mt = m.get("motion_type", "unclassified")
            motion_types[mt] = motion_types.get(mt, 0) + 1

        print(f"Sources    : {len(sources)} unique files")
        print("Motion types:")
        for mt, count in sorted(motion_types.items(), key=lambda x: -x[1]):
            print(f"  {mt:20s} {count}")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        prog="motion_search",
        description="Build and query FAISS motion-embedding indices.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- build-index ---------------------------------------------------
    p_build = subparsers.add_parser("build-index", help="Build an index from .npy files.")
    p_build.add_argument("--data-dir", required=True, help="Directory of .npy motion files.")
    p_build.add_argument("--model", required=True, help="Path to encoder model (.pt or .onnx).")
    p_build.add_argument("--output", required=True, help="Output index path.")
    p_build.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    p_build.add_argument("--device", default=None, help="PyTorch device (e.g. cuda:0).")
    p_build.add_argument(
        "--use-ivf", action="store_true", default=None, help="Force IVF index."
    )

    # -- search --------------------------------------------------------
    p_search = subparsers.add_parser("search", help="Search an index.")
    p_search.add_argument("--index", required=True, help="Path to a saved index.")
    p_search.add_argument("--query", required=True, help="Path to a query .npy file.")
    p_search.add_argument("--k", type=int, default=10, help="Number of neighbours.")
    p_search.add_argument("--model", default=None, help="Encoder model for raw-sequence queries.")
    p_search.add_argument("--device", default=None, help="PyTorch device.")

    # -- stats ---------------------------------------------------------
    p_stats = subparsers.add_parser("stats", help="Print index statistics.")
    p_stats.add_argument("--index", required=True, help="Path to a saved index.")

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    commands = {
        "build-index": _build_index,
        "search": _search,
        "stats": _stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
