#pragma once

// MotionTransformer is now baked into the ONNX denoiser model.
// Trained in Python: python/hypermotion/models/motion_transformer.py
// This header is kept for reference only.

namespace hm::ml {
// The motion transformer (~17.6M params) is fused into the diffusion denoiser
// ONNX graph and runs automatically during MotionDiffusionModel::generate().
} // namespace hm::ml
