#!/usr/bin/env python3
"""HyperMotion style fingerprinting CLI.

Usage:
  # Train style encoder
  python hm_style.py --train --data player_clips/ --epochs 200 --output styles/

  # Encode players (compute embeddings with trained encoder)
  python hm_style.py --encode --data player_clips/ --encoder styles/style_encoder_final.pt --output styles/

  # Export encoder to ONNX
  python hm_style.py --export --encoder styles/style_encoder_final.pt --output models/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def do_train(args):
    from hypermotion.models.style_encoder import StyleEncoder
    from hypermotion.data.style_dataset import StylePairDataset, pad_collate
    from hypermotion.training.style_trainer import StyleTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    dataset = StylePairDataset(args.data, augment=True)
    if len(dataset) == 0:
        log.error(f"No valid player clip pairs found in {args.data}")
        sys.exit(1)
    log.info(f"Loaded {len(dataset)} clip pairs from {len(dataset.player_clips)} players")

    dataloader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        collate_fn=pad_collate, num_workers=4, pin_memory=True,
    )

    model = StyleEncoder()

    # Resume
    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Resumed from {args.encoder}")

    trainer = StyleTrainer(model, lr=args.lr, temperature=0.07, device=device)
    losses = trainer.train(
        dataloader, epochs=args.epochs,
        checkpoint_dir=args.output, checkpoint_every=50,
    )
    log.info(f"Training complete. Final loss: {losses[-1]:.4f}")


def do_encode(args):
    from hypermotion.models.style_encoder import StyleEncoder
    from hypermotion.data.style_dataset import _extract_style_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleEncoder()
    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    style_library = {}

    for player_dir in sorted(data_path.iterdir()):
        if not player_dir.is_dir():
            continue
        player_id = player_dir.name
        embeddings = []

        for clip_file in sorted(player_dir.glob("*.json")):
            try:
                clip_data = json.loads(clip_file.read_text())
                feat = _extract_style_features(clip_data.get("frames", []))
                if feat is None:
                    continue

                x = torch.from_numpy(feat).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(x).cpu().numpy()[0]
                embeddings.append(emb)

            except (json.JSONDecodeError, KeyError):
                continue

        if embeddings:
            # Average embeddings across clips, then L2-normalize
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

            style_library[player_id] = {
                "playerID": player_id,
                "playerName": player_id,
                "embedding": avg_emb.tolist(),
                "numClips": len(embeddings),
                "strideLengthScale": 1.0,
                "armSwingIntensity": 1.0,
                "sprintLeanAngle": 0.0,
                "hipRotationScale": 1.0,
                "kneeLiftScale": 1.0,
                "cadenceScale": 1.0,
                "decelerationSharpness": 1.0,
                "turnLeadBody": 0.0,
            }
            log.info(f"  {player_id}: {len(embeddings)} clips -> embedding computed")

    # Save style library
    library_path = output_path / "style_library.json"
    library_path.write_text(json.dumps(style_library, indent=2))
    log.info(f"Saved style library ({len(style_library)} players): {library_path}")


def do_export(args):
    from hypermotion.models.style_encoder import StyleEncoder
    from hypermotion.export.onnx_export import export_style_encoder

    model = StyleEncoder()
    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    output_path = Path(args.output) / "style_encoder.onnx"
    export_style_encoder(model, output_path)
    log.info("Export complete.")


def main():
    parser = argparse.ArgumentParser(description="HyperMotion Style Fingerprinting")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train style encoder")
    group.add_argument("--encode", action="store_true", help="Compute player embeddings")
    group.add_argument("--export", action="store_true", help="Export encoder to ONNX")

    parser.add_argument("--data", type=str, default=None,
                        help="Path to player clips directory")
    parser.add_argument("--encoder", type=str, default=None,
                        help="Path to trained encoder checkpoint (.pt)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="styles",
                        help="Output directory")
    args = parser.parse_args()

    if args.train:
        if not args.data:
            log.error("--data is required for training")
            sys.exit(1)
        do_train(args)
    elif args.encode:
        if not args.data:
            log.error("--data is required for encoding")
            sys.exit(1)
        do_encode(args)
    elif args.export:
        do_export(args)


if __name__ == "__main__":
    main()
