"""Export a trained MotionEncoder to ONNX for C++ inference via ONNX Runtime.

The exported model uses the same I/O contract expected by the C++ OnnxInference
pipeline:
    - Input name:  ``joint_positions``  shape [batch, seq_len, 66]
    - Output name: ``motion_embedding`` shape [batch, 128]
    - Dynamic axes on batch (dim 0) and seq_len (dim 1)
    - ONNX opset 17
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..constants import DEFAULT_SEQ_LEN, MOTION_INPUT_DIM, ONNX_OPSET
from .encoder import MotionEncoder

log = logging.getLogger(__name__)


def export_encoder_onnx(
    model: MotionEncoder,
    output_path: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    opset: int = ONNX_OPSET,
) -> Path:
    """Export a trained MotionEncoder to ONNX format.

    The exported graph accepts variable-length sequences thanks to dynamic
    axes on the batch and sequence-length dimensions. The C++ side loads
    this file directly with ONNX Runtime.

    Args:
        model: Trained MotionEncoder instance.
        output_path: Destination path for the .onnx file.
        seq_len: Sequence length for the dummy input (does not constrain
            the exported model due to dynamic axes).
        opset: ONNX opset version (must be >= 17 for full operator support).

    Returns:
        Resolved path to the exported .onnx file.

    Raises:
        RuntimeError: If the ONNX export fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, seq_len, MOTION_INPUT_DIM, device=device)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        opset_version=opset,
        input_names=["joint_positions"],
        output_names=["motion_embedding"],
        dynamic_axes={
            "joint_positions": {0: "batch", 1: "seq_len"},
            "motion_embedding": {0: "batch"},
        },
    )

    log.info("Exported motion encoder to ONNX: %s", output_path)
    return output_path


def load_and_export(
    checkpoint_path: str | Path,
    output_path: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    opset: int = ONNX_OPSET,
) -> Path:
    """Load a checkpoint and export the model to ONNX.

    Convenience function that loads model weights from a training checkpoint
    and delegates to :func:`export_encoder_onnx`.

    Args:
        checkpoint_path: Path to a .pt checkpoint (must contain
            ``model_state_dict``).
        output_path: Destination path for the .onnx file.
        seq_len: Sequence length for the dummy input.
        opset: ONNX opset version.

    Returns:
        Resolved path to the exported .onnx file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model = MotionEncoder()
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded model weights from %s", checkpoint_path)

    return export_encoder_onnx(model, output_path, seq_len=seq_len, opset=opset)
