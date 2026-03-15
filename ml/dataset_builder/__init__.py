"""Dataset builder module for converting football match footage into skeleton motion datasets.

Pipeline: video -> frame extraction -> pose detection -> skeleton normalization -> sequence export

All constants are imported from ml.constants to stay in sync with the C++ core.
"""

from .frame_extractor import FrameExtractor
from .pose_detector import PoseDetector
from .skeleton_normalizer import SkeletonNormalizer
from .sequence_exporter import SequenceExporter
from .pipeline import DatasetPipeline

__all__ = [
    "FrameExtractor",
    "PoseDetector",
    "SkeletonNormalizer",
    "SequenceExporter",
    "DatasetPipeline",
]
