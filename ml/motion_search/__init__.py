"""Motion search module — FAISS-backed nearest-neighbour retrieval for motion embeddings."""

from .index import MotionIndex, SearchResult
from .embedder import BatchEmbedder

__all__ = ["MotionIndex", "SearchResult", "BatchEmbedder"]
