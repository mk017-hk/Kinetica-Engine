"""Motion Embeddings — train a neural encoder that maps motion sequences to embedding vectors.

Provides similarity search and clustering over motion data by encoding
variable-length joint-position sequences into L2-normalized 128D vectors.

Key classes:
    MotionEncoder       — Temporal CNN encoder (66D input -> 128D embedding)
    MotionSequenceDataset — Dataset loader for .npy motion files
    EmbeddingTrainer    — Training loop with triplet loss and hard mining
    export_encoder_onnx — Export trained encoder to ONNX for C++ inference
"""

from .encoder import MotionEncoder, TemporalResBlock
from .dataset import MotionSequenceDataset
from .trainer import EmbeddingTrainer
from .export_onnx import export_encoder_onnx

__all__ = [
    "MotionEncoder",
    "TemporalResBlock",
    "MotionSequenceDataset",
    "EmbeddingTrainer",
    "export_encoder_onnx",
]
