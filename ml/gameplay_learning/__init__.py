"""Gameplay learning module for extracting animation sequences from match footage.

Combines pose detection, multi-player tracking, and motion segmentation to
automatically produce labeled motion clips from broadcast sports video.

All constants are imported from ml.constants to stay in sync with the C++ core.
"""

from .player_tracker import PlayerTracker, Track
from .motion_segmenter import MotionSegmenter, MotionClip
from .pipeline import GameplayLearningPipeline, PipelineConfig

__all__ = [
    "PlayerTracker",
    "Track",
    "MotionSegmenter",
    "MotionClip",
    "GameplayLearningPipeline",
    "PipelineConfig",
]
