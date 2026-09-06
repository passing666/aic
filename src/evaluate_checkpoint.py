"""Evaluate a saved SegFormer checkpoint on the fixed validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from train import CLASS_NAMES, SegmentationDataset, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    dataset = SegmentationDataset(args.val_csv, args.data_root, args.image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    miou, ious, predicted_ignore_ratio = evaluate(model, loader, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "validation_samples": len(dataset),
        "miou": miou,
        "ious": ious,
        "class_names": list(CLASS_NAMES),
        "predicted_ignore_ratio_on_valid_pixels": predicted_ignore_ratio,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
