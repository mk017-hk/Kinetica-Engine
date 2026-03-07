"""Export trained PyTorch models to ONNX for C++ inference via ONNX Runtime.

Each exported model produces a .onnx file that the C++ side loads directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from ..models.motion_diffusion import MotionDiffusionModel
from ..models.motion_transformer import MotionTransformer
from ..models.condition_encoder import ConditionEncoder
from ..models.temporal_conv_net import TemporalConvNet
from ..models.style_encoder import StyleEncoder
from ..models.motion_encoder import MotionEncoder, MOTION_INPUT_DIM
from ..models.constants import FRAME_DIM, CONDITION_DIM, FEATURE_DIM_SEGMENTER, STYLE_INPUT_DIM

log = logging.getLogger(__name__)


class DiffusionDenoiserWrapper(nn.Module):
    """Wraps the denoising step (transformer + condition encoder) for ONNX export.

    ONNX graph: (noisy_motion, timestep, condition) -> predicted_noise
    The DDIM loop runs in C++ calling this per step.
    """

    def __init__(self, diffusion_model: MotionDiffusionModel):
        super().__init__()
        self.cond_encoder = diffusion_model.condition_encoder
        self.transformer = diffusion_model.transformer

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        cond_emb = self.cond_encoder(condition)
        return self.transformer(x_t, t, cond_emb)


def export_diffusion_denoiser(model: MotionDiffusionModel, output_path: str | Path,
                               seq_len: int = 64, opset: int = 17) -> Path:
    """Export the diffusion denoiser (transformer + condition encoder) to ONNX.

    The noise scheduler and DDIM loop are reimplemented in C++ since they're
    pure math — no neural network weights.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = DiffusionDenoiserWrapper(model)
    wrapper.eval()

    # Dummy inputs
    batch = 1
    x_t = torch.randn(batch, seq_len, FRAME_DIM)
    t = torch.zeros(batch, dtype=torch.long)
    condition = torch.randn(batch, CONDITION_DIM)

    torch.onnx.export(
        wrapper, (x_t, t, condition), str(output_path),
        opset_version=opset,
        input_names=["noisy_motion", "timestep", "condition"],
        output_names=["predicted_noise"],
        dynamic_axes={
            "noisy_motion": {0: "batch", 1: "seq_len"},
            "timestep": {0: "batch"},
            "condition": {0: "batch"},
            "predicted_noise": {0: "batch", 1: "seq_len"},
        },
    )
    log.info(f"Exported diffusion denoiser: {output_path}")
    return output_path


def export_classifier(model: TemporalConvNet, output_path: str | Path,
                       opset: int = 17) -> Path:
    """Export the TCN motion classifier to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input: [1, 256, 70]
    x = torch.randn(1, 256, FEATURE_DIM_SEGMENTER)

    torch.onnx.export(
        model, (x,), str(output_path),
        opset_version=opset,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )
    log.info(f"Exported classifier: {output_path}")
    return output_path


def export_style_encoder(model: StyleEncoder, output_path: str | Path,
                          opset: int = 17) -> Path:
    """Export the style encoder to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input: [1, 128, 201]
    x = torch.randn(1, 128, STYLE_INPUT_DIM)

    torch.onnx.export(
        model, (x,), str(output_path),
        opset_version=opset,
        input_names=["motion_features"],
        output_names=["style_embedding"],
        dynamic_axes={
            "motion_features": {0: "batch", 1: "seq_len"},
            "style_embedding": {0: "batch"},
        },
    )
    log.info(f"Exported style encoder: {output_path}")
    return output_path


def export_motion_encoder(model: MotionEncoder, output_path: str | Path,
                           seq_len: int = 64, opset: int = 17) -> Path:
    """Export the motion encoder to ONNX for C++ inference.

    Input:  joint_positions [batch, seq_len, 66]
    Output: motion_embedding [batch, 128]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    x = torch.randn(1, seq_len, MOTION_INPUT_DIM)

    torch.onnx.export(
        model, (x,), str(output_path),
        opset_version=opset,
        input_names=["joint_positions"],
        output_names=["motion_embedding"],
        dynamic_axes={
            "joint_positions": {0: "batch", 1: "seq_len"},
            "motion_embedding": {0: "batch"},
        },
    )
    log.info(f"Exported motion encoder: {output_path}")
    return output_path
