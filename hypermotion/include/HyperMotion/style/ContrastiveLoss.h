#pragma once

// ContrastiveLoss is now implemented in the Python training pipeline.
// This header is kept for backward compatibility but contains no class.
// See: python/hypermotion/models/contrastive_loss.py

namespace hm::style {
// Training-only — use the Python pipeline:
//   python scripts/hm_style.py --train --data player_clips/ --epochs 200
} // namespace hm::style
