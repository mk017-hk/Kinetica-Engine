"""Batch embedding computation using a trained motion encoder.

Supports both PyTorch checkpoints (``.pt``) and ONNX models (``.onnx``)
for flexible deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import MOTION_EMBEDDING_DIM, MOTION_INPUT_DIM

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Compute motion embeddings in batches using a trained encoder.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` (PyTorch) or ``.onnx`` model file.
    batch_size:
        Number of sequences processed per forward pass.
    device:
        PyTorch device string (ignored for ONNX).  Defaults to ``"cuda"``
        when a GPU is available, ``"cpu"`` otherwise.
    """

    def __init__(
        self,
        model_path: Path,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self._backend: str = ""
        self._model = None
        self._session = None
        self._device = device

        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_files(
        self,
        data_dir: Path,
        pattern: str = "*.npy",
    ) -> tuple[np.ndarray, list[dict]]:
        """Embed every ``.npy`` motion file in *data_dir*.

        Each ``.npy`` file is expected to contain an array of shape
        ``(T, MOTION_INPUT_DIM)`` where *T* is the number of frames.

        Returns
        -------
        embeddings:
            ``(N, MOTION_EMBEDDING_DIM)`` float32 array.
        metadata:
            List of *N* dicts with keys ``source_file``, ``num_frames``,
            and ``frame_range``.
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {pattern} files found in {data_dir}")

        sequences: list[np.ndarray] = []
        metadata: list[dict] = []

        for fpath in files:
            seq = np.load(fpath).astype(np.float32)
            if seq.ndim != 2 or seq.shape[1] != MOTION_INPUT_DIM:
                logger.warning(
                    "Skipping %s: expected shape (T, %d), got %s",
                    fpath.name,
                    MOTION_INPUT_DIM,
                    seq.shape,
                )
                continue
            sequences.append(seq)
            metadata.append(
                {
                    "source_file": str(fpath.name),
                    "num_frames": int(seq.shape[0]),
                    "frame_range": [0, int(seq.shape[0])],
                }
            )

        if not sequences:
            raise ValueError("No valid motion sequences found after filtering.")

        logger.info("Loaded %d sequences from %s", len(sequences), data_dir)
        embeddings = self._embed_batch(sequences)
        return embeddings, metadata

    def embed(self, sequence: np.ndarray) -> np.ndarray:
        """Embed a single motion sequence.

        Parameters
        ----------
        sequence:
            ``(T, MOTION_INPUT_DIM)`` float32 array.

        Returns
        -------
        np.ndarray
            ``(MOTION_EMBEDDING_DIM,)`` float32 embedding vector.
        """
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != MOTION_INPUT_DIM:
            raise ValueError(
                f"Expected (T, {MOTION_INPUT_DIM}), got {sequence.shape}"
            )
        embeddings = self._embed_batch([sequence])
        return embeddings[0]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx()
        elif suffix in (".pt", ".pth"):
            self._load_torch()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _load_torch(self) -> None:
        import torch  # type: ignore[import-untyped]

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        # Support both raw model and state-dict-in-checkpoint conventions.
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            raise ValueError(
                "Checkpoint contains a state_dict but no model object.  "
                "Export the full model or provide an ONNX file."
            )
        else:
            model = checkpoint

        model.to(device)
        model.eval()
        self._model = model
        self._backend = "torch"
        logger.info("Loaded PyTorch model from %s (device=%s)", self.model_path, device)

    def _load_onnx(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        providers = ort.get_available_providers()
        # Prefer CUDA if available.
        if "CUDAExecutionProvider" in providers:
            selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            selected = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(self.model_path), providers=selected)
        self._backend = "onnx"
        logger.info("Loaded ONNX model from %s (providers=%s)", self.model_path, selected)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def _embed_batch(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Run batched inference over a list of variable-length sequences.

        Each sequence is zero-padded to the length of the longest sequence
        in its batch.
        """
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(sequences), self.batch_size):
            batch_seqs = sequences[start : start + self.batch_size]
            padded, lengths = self._pad_sequences(batch_seqs)

            if self._backend == "torch":
                embs = self._infer_torch(padded, lengths)
            else:
                embs = self._infer_onnx(padded)

            all_embeddings.append(embs)

        return np.concatenate(all_embeddings, axis=0)

    def _infer_torch(self, padded: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        import torch  # type: ignore[import-untyped]

        with torch.no_grad():
            x = torch.from_numpy(padded).to(self._device)
            out = self._model(x)
            # If the model returns a tuple/dict, take the embedding tensor.
            if isinstance(out, dict):
                out = out.get("embedding", out.get("z", next(iter(out.values()))))
            elif isinstance(out, (tuple, list)):
                out = out[0]
            return out.cpu().numpy().astype(np.float32)

    def _infer_onnx(self, padded: np.ndarray) -> np.ndarray:
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        result = self._session.run([output_name], {input_name: padded})
        return np.asarray(result[0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad a list of ``(T_i, D)`` arrays to ``(B, T_max, D)``."""
        max_len = max(s.shape[0] for s in sequences)
        dim = sequences[0].shape[1]
        batch_size = len(sequences)

        padded = np.zeros((batch_size, max_len, dim), dtype=np.float32)
        lengths = np.zeros(batch_size, dtype=np.int64)
        for i, seq in enumerate(sequences):
            t = seq.shape[0]
            padded[i, :t, :] = seq
            lengths[i] = t
        return padded, lengths
