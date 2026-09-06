"""Evaluate a checkpoint on the fixed validation split with controlled color shifts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import SegformerForSemanticSegmentation

from train import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD, IGNORE_INDEX, NUM_CLASSES


def load_image(
    path: Path,
    image_size: int,
    brightness: float,
    contrast: float,
    color: float,
    gamma: float,
    blur: float,
    noise: float,
) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if color != 1.0:
            image = ImageEnhance.Color(image).enhance(color)
        if blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    if gamma != 1.0:
        image_array = np.power(np.clip(image_array, 0.0, 1.0), gamma)
    if noise > 0:
        rng = np.random.default_rng(2026)
        image_array += rng.normal(0.0, noise, image_array.shape).astype(np.float32)
    return np.clip(image_array, 0.0, 1.0)


def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    data_root: Path,
    image_size: int,
    brightness: float,
    contrast: float,
    color: float,
    gamma: float,
    blur: float,
    noise: float,
    device: torch.device,
) -> dict[str, object]:
    intersections = [0] * NUM_CLASSES
    unions = [0] * NUM_CLASSES
    valid_pixels = 0
    predicted_ignore = 0
    started_at = time.time()
    model.eval()
    with torch.no_grad():
        for row in rows:
            image_path = data_root / "images" / f"{row['id']}.png"
            mask_path = data_root / "masks" / f"{row['id']}.png"
            image = load_image(image_path, image_size, brightness, contrast, color, gamma, blur, noise)
            with Image.open(mask_path) as mask_file:
                mask = np.asarray(mask_file.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST), dtype=np.int64)
            tensor = torch.from_numpy((image - IMAGE_MEAN) / IMAGE_STD).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            valid = mask != IGNORE_INDEX
            valid_pixels += int(valid.sum())
            predicted_ignore += int(((predictions == IGNORE_INDEX) & valid).sum())
            for class_id in range(1, NUM_CLASSES):
                predicted = (predictions == class_id) & valid
                actual = (mask == class_id) & valid
                intersections[class_id] += int((predicted & actual).sum())
                unions[class_id] += int((predicted | actual).sum())
    ious = [None] + [intersections[i] / unions[i] if unions[i] else None for i in range(1, NUM_CLASSES)]
    valid_ious = [value for value in ious if value is not None]
    return {
        "brightness": brightness,
        "contrast": contrast,
        "color": color,
        "gamma": gamma,
        "blur": blur,
        "noise": noise,
        "image_size": image_size,
        "samples": len(rows),
        "miou": sum(valid_ious) / len(valid_ious) if valid_ious else 0.0,
        "ious": ious,
        "predicted_ignore_ratio_on_valid_pixels": predicted_ignore / valid_pixels if valid_pixels else 0.0,
        "elapsed_seconds": time.time() - started_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--color", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    with args.val_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    results = [
        evaluate(model, rows, args.data_root, args.image_size, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, device),
        evaluate(model, rows, args.data_root, args.image_size, 1.10, 1.0, 1.0, 1.0, 0.0, 0.0, device),
        evaluate(model, rows, args.data_root, args.image_size, 1.10, 1.05, 1.0, 1.0, 0.0, 0.0, device),
        evaluate(model, rows, args.data_root, args.image_size, 1.0, 1.0, args.color, 1.0, 0.0, 0.0, device),
        evaluate(model, rows, args.data_root, args.image_size, 1.0, 1.0, 1.0, args.gamma, 0.0, 0.0, device),
        evaluate(model, rows, args.data_root, args.image_size, 1.0, 1.0, 1.0, 1.0, args.blur, args.noise, device),
    ]
    output = {
        "checkpoint": str(args.checkpoint),
        "validation_samples": len(rows),
        "class_names": list(CLASS_NAMES),
        "scenarios": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())