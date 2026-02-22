#pragma once

// StyleTrainer is now implemented in the Python training pipeline.
// This header is kept for backward compatibility but contains no class.
// See: python/hypermotion/training/style_trainer.py
//
// Usage:
//   python scripts/hm_style.py --train --data player_clips/ --epochs 200 --output styles/
//   python scripts/hm_style.py --export --encoder styles/style_encoder_final.pt --output models/

namespace hm::style {
// Training-only — use the Python pipeline.
} // namespace hm::style
