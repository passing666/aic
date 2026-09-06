"""Generate single-channel PNG predictions for the competition test images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from train import IMAGE_MEAN, IMAGE_STD, NUM_CLASSES


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = (image.height, image.width)
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(image_array).permute(2, 0, 1), original_size


def predict_directory(
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> int:
    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            sizes = []
            for image_path in batch_paths:
                image_tensor, original_size = load_image(image_path, image_size)
                images.append(image_tensor)
                sizes.append(original_size)
            logits = model(pixel_values=torch.stack(images).to(device)).logits
            for image_path, image_logits, original_size in zip(batch_paths, logits, sizes):
                image_logits = torch.nn.functional.interpolate(
                    image_logits.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
                )
                prediction = image_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                if prediction.min() < 0 or prediction.max() >= NUM_CLASSES:
                    raise ValueError(f"Invalid prediction values for {image_path}")
                Image.fromarray(prediction).save(output_dir / image_path.name)
    return len(image_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    count = predict_directory(model, args.input_dir, args.output_dir, args.image_size, args.batch_size, device)
    print(f"generated={count} output_dir={args.output_dir} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())