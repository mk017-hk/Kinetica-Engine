"""ONNX export for the MotionPredictor model.

Exports the trained temporal transformer to ONNX format compatible with the
C++ ``OnnxInference`` runtime.  After export, an ``onnxruntime`` validation
pass confirms numerical agreement with the PyTorch model.

ONNX graph:
    Input:  ``context_frames``  [batch, seq_len, 132]
    Output: ``predicted_frames`` [batch, seq_len, 132]
    Dynamic axes on batch (dim 0) and seq_len (dim 1).
    Opset: 17
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..constants import FRAME_DIM, ONNX_OPSET
from .model import MotionPredictor


def export_predictor(
    model: MotionPredictor,
    output_path: str | Path,
    seq_len: int = 32,
    validate: bool = True,
    verbose: bool = True,
) -> Path:
    """Export a trained :class:`MotionPredictor` to ONNX.

    Args:
        model: Trained model instance (will be set to eval mode).
        output_path: Destination ``.onnx`` file path.
        seq_len: Sequence length used for the dummy input during tracing.
        validate: If *True*, run onnxruntime inference and compare outputs
            to the PyTorch model.
        verbose: Print model size and I/O information.

    Returns:
        Resolved path to the exported ONNX file.

    Raises:
        RuntimeError: If validation detects a numerical mismatch.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Dummy input for tracing
    dummy_input = torch.randn(1, seq_len, model.frame_dim, device=device)

    # Export
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["context_frames"],
        output_names=["predicted_frames"],
        dynamic_axes={
            "context_frames": {0: "batch", 1: "seq_len"},
            "predicted_frames": {0: "batch", 1: "seq_len"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
    )

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ONNX model exported to: {output_path}")
        print(f"  Model size:  {size_mb:.2f} MB")
        print(f"  Opset:       {ONNX_OPSET}")
        print(f"  Input:       context_frames  [batch, seq_len, {model.frame_dim}]")
        print(f"  Output:      predicted_frames [batch, seq_len, {model.frame_dim}]")
        print(f"  Dynamic axes: batch (0), seq_len (1)")

    if validate:
        _validate_onnx(model, output_path, dummy_input, verbose)

    return output_path.resolve()


def _validate_onnx(
    model: MotionPredictor,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    verbose: bool,
) -> None:
    """Validate ONNX output matches PyTorch output within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        if verbose:
            print("  [WARN] onnxruntime not installed — skipping validation.")
        return

    # PyTorch reference output
    with torch.no_grad():
        pt_output = model(dummy_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_inputs = {"context_frames": dummy_input.cpu().numpy()}
    ort_output = session.run(["predicted_frames"], ort_inputs)[0]

    # Numerical comparison
    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))

    if verbose:
        print(f"  Validation:  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    tolerance = 1e-4
    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX validation failed: max absolute difference {max_diff:.2e} "
            f"exceeds tolerance {tolerance:.2e}."
        )

    if verbose:
        print("  Validation:  PASSED")
