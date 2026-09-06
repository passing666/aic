"""Compare full-image and sliding-window inference on the fixed validation split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from train import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD, IGNORE_INDEX, NUM_CLASSES


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def normalize(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((image - IMAGE_MEAN) / IMAGE_STD).permute(2, 0, 1)


def predict_logits(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = normalize(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        return torch.nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        ).squeeze(0).cpu()


def resize_logits(model: torch.nn.Module, image: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray((image * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    logits = predict_logits(model, resized, device)
    return torch.nn.functional.interpolate(
        logits.unsqueeze(0), size=image.shape[:2], mode="bilinear", align_corners=False
    ).squeeze(0)


def sliding_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    window_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    height, width = image.shape[:2]
    if height < window_size or width < window_size:
        raise ValueError("Sliding-window inference requires images at least as large as the window")
    y_positions = list(range(0, height - window_size + 1, stride))
    x_positions = list(range(0, width - window_size + 1, stride))
    if y_positions[-1] != height - window_size:
        y_positions.append(height - window_size)
    if x_positions[-1] != width - window_size:
        x_positions.append(width - window_size)
    accumulated = torch.zeros((NUM_CLASSES, height, width), dtype=torch.float32)
    counts = torch.zeros((1, height, width), dtype=torch.float32)
    for top in y_positions:
        for left in x_positions:
            window = image[top : top + window_size, left : left + window_size]
            accumulated[:, top : top + window_size, left : left + window_size] += predict_logits(
                model, window, device
            )
            counts[:, top : top + window_size, left : left + window_size] += 1
    return accumulated / counts.clamp_min(1)


def tta_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
    vertical: bool,
) -> torch.Tensor:
    variants = [(image, None)]
    variants.append((np.flip(image, axis=1).copy(), "horizontal"))
    if vertical:
        variants.append((np.flip(image, axis=0).copy(), "vertical"))
        variants.append((np.flip(image, axis=(0, 1)).copy(), "both"))

    accumulated = None
    for variant, flip in variants:
        logits = resize_logits(model, variant, image_size, device)
        if flip in ("horizontal", "both"):
            logits = torch.flip(logits, dims=(2,))
        if flip in ("vertical", "both"):
            logits = torch.flip(logits, dims=(1,))
        accumulated = logits if accumulated is None else accumulated + logits
    return accumulated / len(variants)


def update_metrics(
    logits: torch.Tensor,
    mask: np.ndarray,
    intersections: list[int],
    unions: list[int],
    ignore_predictions: list[int],
    valid_pixels: list[int],
) -> None:
    predictions = logits.argmax(dim=0).numpy()
    valid = mask != IGNORE_INDEX
    valid_pixels[0] += int(valid.sum())
    ignore_predictions[0] += int(((predictions == IGNORE_INDEX) & valid).sum())
    for class_id in range(1, NUM_CLASSES):
        predicted = (predictions == class_id) & valid
        actual = (mask == class_id) & valid
        intersections[class_id] += int((predicted & actual).sum())
        unions[class_id] += int((predicted | actual).sum())


def evaluate_strategy(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    data_root: Path,
    strategy: str,
    device: torch.device,
    image_size: int,
    stride: int,
) -> dict[str, object]:
    intersections = [0] * NUM_CLASSES
    unions = [0] * NUM_CLASSES
    ignore_predictions = [0]
    valid_pixels = [0]
    started_at = time.time()
    for row in rows:
        image_path = data_root / "images" / f"{row['id']}.png"
        mask_path = data_root / "masks" / f"{row['id']}.png"
        image = load_rgb(image_path)
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("L"), dtype=np.int64)
        if strategy == "resize":
            logits = resize_logits(model, image, image_size, device)
        elif strategy == "full":
            logits = predict_logits(model, image, device)
        elif strategy == "sliding":
            logits = sliding_logits(model, image, image_size, stride, device)
        elif strategy == "tta-h":
            logits = tta_logits(model, image, image_size, device, vertical=False)
        elif strategy == "tta-hv":
            logits = tta_logits(model, image, image_size, device, vertical=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        update_metrics(logits, mask, intersections, unions, ignore_predictions, valid_pixels)
    ious = [None] + [intersections[i] / unions[i] if unions[i] else None for i in range(1, NUM_CLASSES)]
    valid_ious = [value for value in ious if value is not None]
    return {
        "strategy": strategy,
        "image_size": image_size if strategy != "full" else 1024,
        "stride": stride if strategy == "sliding" else None,
        "miou": sum(valid_ious) / len(valid_ious) if valid_ious else 0.0,
        "ious": ious,
        "predicted_ignore_ratio_on_valid_pixels": ignore_predictions[0] / valid_pixels[0],
        "samples": len(rows),
        "elapsed_seconds": time.time() - started_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
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
    model.eval()
    results = [
        evaluate_strategy(model, rows, args.data_root, "resize", device, args.image_size, args.stride),
        evaluate_strategy(model, rows, args.data_root, "full", device, args.image_size, args.stride),
        evaluate_strategy(model, rows, args.data_root, "sliding", device, args.image_size, args.stride),
        evaluate_strategy(model, rows, args.data_root, "tta-h", device, args.image_size, args.stride),
        evaluate_strategy(model, rows, args.data_root, "tta-hv", device, args.image_size, args.stride),
    ]
    output = {
        "checkpoint": str(args.checkpoint),
        "validation_samples": len(rows),
        "class_names": list(CLASS_NAMES),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())