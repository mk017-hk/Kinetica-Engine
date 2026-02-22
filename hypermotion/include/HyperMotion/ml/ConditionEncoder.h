#pragma once

// ConditionEncoder is now baked into the ONNX denoiser model.
// Trained in Python: python/hypermotion/models/condition_encoder.py
// This header is kept for reference only.

namespace hm::ml {
// The condition encoder (78D -> 256D) is fused into the diffusion denoiser
// ONNX graph and runs automatically during MotionDiffusionModel::generate().
} // namespace hm::ml
