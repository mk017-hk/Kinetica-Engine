"""Motion Prediction — train a temporal transformer that predicts future animation frames.

Given a partial sequence of skeletal animation frames (22 joints x 6D rotation = 132D
per frame), the model autoregressively predicts future frames for use in animation
forecasting, motion in-betweening, and real-time anticipation.

Key classes:
    MotionPredictor           — Temporal transformer (132D input -> 132D predicted frames)
    MotionPredictionDataset   — Dataset loader for .npy motion sequence files
    MotionPredictionTrainer   — Training loop with MSE loss and cosine scheduling
    export_predictor          — Export trained model to ONNX for C++ inference
"""

from .model import MotionPredictor
from .dataset import MotionPredictionDataset
from .trainer import MotionPredictionTrainer
from .export_onnx import export_predictor

__all__ = [
    "MotionPredictor",
    "MotionPredictionDataset",
    "MotionPredictionTrainer",
    "export_predictor",
]
