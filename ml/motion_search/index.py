"""FAISS index wrapper for motion embedding retrieval.

Supports flat (exact) search via ``IndexFlatIP`` and approximate search via
``IndexIVFFlat`` for large datasets (>100 k vectors).  All vectors are
L2-normalised before insertion so that inner-product search is equivalent to
cosine similarity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from ..constants import MOTION_EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Threshold above which we switch to IVF for efficiency.
_IVF_THRESHOLD = 100_000
# Default number of IVF clusters (adjusted at build time).
_IVF_DEFAULT_NLIST = 256


@dataclass
class SearchResult:
    """A single search hit returned by :pymeth:`MotionIndex.search`."""

    rank: int
    score: float
    metadata: dict = field(default_factory=dict)


class MotionIndex:
    """FAISS index with a JSON metadata sidecar.

    Parameters
    ----------
    dim:
        Dimensionality of the embedding vectors.  Must match the encoder
        output (``MOTION_EMBEDDING_DIM`` by default).
    use_ivf:
        If *None* (default) the index type is chosen automatically based on
        the dataset size at :pymeth:`build` time.  Set explicitly to force a
        particular index type.
    nlist:
        Number of IVF clusters when ``use_ivf`` is enabled.
    """

    def __init__(
        self,
        dim: int = MOTION_EMBEDDING_DIM,
        use_ivf: Optional[bool] = None,
        nlist: int = _IVF_DEFAULT_NLIST,
    ) -> None:
        self.dim = dim
        self._use_ivf = use_ivf
        self._nlist = nlist
        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Build a new index from an embedding matrix and metadata list.

        Parameters
        ----------
        embeddings:
            ``(N, dim)`` float32 array of embedding vectors.
        metadata:
            List of *N* dicts — each dict describes the source clip
            (e.g. source file, frame range, motion type).
        """
        embeddings = self._validate(embeddings, len(metadata))
        embeddings = self._normalise(embeddings)

        n = embeddings.shape[0]
        use_ivf = self._use_ivf if self._use_ivf is not None else (n >= _IVF_THRESHOLD)

        if use_ivf:
            nlist = min(self._nlist, max(1, n // 39))
            quantiser = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
            logger.info("Built IVF index with %d clusters for %d vectors", nlist, n)
        else:
            self._index = faiss.IndexFlatIP(self.dim)
            logger.info("Built flat index for %d vectors", n)

        self._index.add(embeddings)
        self._metadata = list(metadata)

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Incrementally add vectors to an existing index.

        For IVF indices the index must already be trained (via :pymeth:`build`).
        """
        if self._index is None:
            raise RuntimeError("Index not initialised — call build() or load() first.")
        embeddings = self._validate(embeddings, len(metadata))
        embeddings = self._normalise(embeddings)
        self._index.add(embeddings)
        self._metadata.extend(metadata)
        logger.info("Added %d vectors (total: %d)", embeddings.shape[0], self._index.ntotal)

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Find the *k* nearest neighbours for a query vector.

        Parameters
        ----------
        query:
            ``(dim,)`` or ``(1, dim)`` float32 query embedding.
        k:
            Number of neighbours to return.

        Returns
        -------
        list[SearchResult]
            Sorted by descending similarity score.
        """
        if self._index is None:
            raise RuntimeError("Index not initialised — call build() or load() first.")

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dim {query.shape[1]} != index dim {self.dim}")
        query = self._normalise(query)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    metadata=self._metadata[idx] if idx < len(self._metadata) else {},
                )
            )
        return results

    def save(self, path: Path) -> None:
        """Persist the FAISS index and metadata sidecar to disk.

        Writes two files:
        * ``<path>``       — binary FAISS index
        * ``<path>.meta``  — JSON metadata array
        """
        if self._index is None:
            raise RuntimeError("Index not initialised — nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        meta_path = path.with_suffix(path.suffix + ".meta")
        meta_path.write_text(json.dumps(self._metadata, indent=2))
        logger.info("Saved index (%d vectors) to %s", self._index.ntotal, path)

    @classmethod
    def load(cls, path: Path, dim: int = MOTION_EMBEDDING_DIM) -> "MotionIndex":
        """Load a previously saved index from disk.

        Parameters
        ----------
        path:
            Path to the FAISS index file (the ``.meta`` sidecar is loaded
            automatically).
        dim:
            Expected embedding dimensionality.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        obj = cls(dim=dim)
        obj._index = faiss.read_index(str(path))
        meta_path = path.with_suffix(path.suffix + ".meta")
        if meta_path.exists():
            obj._metadata = json.loads(meta_path.read_text())
        else:
            logger.warning("Metadata sidecar not found at %s", meta_path)
            obj._metadata = [{} for _ in range(obj._index.ntotal)]
        logger.info("Loaded index with %d vectors from %s", obj._index.ntotal, path)
        return obj

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, embeddings: np.ndarray, expected_n: int) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected (N, {self.dim}) array, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != expected_n:
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) != metadata count ({expected_n})"
            )
        return embeddings

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows in-place and return the array."""
        faiss.normalize_L2(vectors)
        return vectors
