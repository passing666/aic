"""Audit a submission and create prediction visualizations without labels."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NUM_CLASSES = 9
CLASS_NAMES = (
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
)
CLASS_COLORS = (
    (0, 0, 0),
    (128, 128, 128),
    (220, 70, 70),
    (70, 110, 220),
    (40, 170, 220),
    (220, 160, 50),
    (60, 170, 80),
    (170, 120, 210),
    (245, 235, 60),
)


def list_prediction_names(predictions: Path) -> list[str]:
    if predictions.is_dir():
        return sorted(path.name for path in predictions.glob("*.png"))
    if predictions.suffix.lower() != ".zip":
        raise ValueError(f"Predictions must be a directory or zip: {predictions}")
    with zipfile.ZipFile(predictions) as archive:
        return sorted(Path(name).name for name in archive.namelist() if name.lower().endswith(".png"))


def read_prediction(predictions: Path, name: str) -> Image.Image:
    if predictions.is_dir():
        return Image.open(predictions / name).convert("L")
    with zipfile.ZipFile(predictions) as archive:
        matching = [entry for entry in archive.namelist() if Path(entry).name == name]
        if len(matching) != 1:
            raise ValueError(f"Could not uniquely find {name} in {predictions}")
        return Image.open(io.BytesIO(archive.read(matching[0]))).convert("L")


def colorize(mask: Image.Image) -> Image.Image:
    labels = np.asarray(mask, dtype=np.uint8)
    if int(labels.max()) >= NUM_CLASSES:
        raise ValueError(f"Invalid prediction label {int(labels.max())}")
    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    return Image.fromarray(palette[labels], mode="RGB")


def make_triptych(image: Image.Image, prediction: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    colored = colorize(prediction)
    overlay = Image.blend(image, colored, alpha=0.45)
    width, height = image.size
    triptych = Image.new("RGB", (width * 3, height), "white")
    triptych.paste(image, (0, 0))
    triptych.paste(colored, (width, 0))
    triptych.paste(overlay, (width * 2, 0))
    draw = ImageDraw.Draw(triptych)
    draw.text((8, 8), "image", fill="white", stroke_width=2, stroke_fill="black")
    draw.text((width + 8, 8), "prediction", fill="white", stroke_width=2, stroke_fill="black")
    draw.text((width * 2 + 8, 8), "overlay", fill="white", stroke_width=2, stroke_fill="black")
    return triptych


def analyze_submission(
    predictions: Path,
    images_dir: Path,
    output_dir: Path,
    visualization_count: int,
) -> dict[str, object]:
    image_names = sorted(path.name for path in images_dir.glob("*.png"))
    prediction_names = list_prediction_names(predictions)
    missing = sorted(set(image_names) - set(prediction_names))
    extra = sorted(set(prediction_names) - set(image_names))
    if missing or extra:
        raise ValueError(f"Prediction names do not match images: missing={missing[:5]} extra={extra[:5]}")
    if not prediction_names:
        raise ValueError(f"No PNG predictions found in {predictions}")

    counts = [0] * NUM_CLASSES
    image_size = None
    for name in prediction_names:
        with read_prediction(predictions, name) as prediction:
            if prediction.size != (1024, 1024):
                raise ValueError(f"Unexpected prediction size for {name}: {prediction.size}")
            values = np.asarray(prediction, dtype=np.uint8)
            if int(values.max()) >= NUM_CLASSES:
                raise ValueError(f"Invalid prediction label {int(values.max())} in {name}")
            image_counts = np.bincount(values.ravel(), minlength=NUM_CLASSES)
            counts = [count + int(value) for count, value in zip(counts, image_counts)]
            image_size = prediction.size

    total_pixels = sum(counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "triptychs"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    selected_names = prediction_names[:visualization_count]
    if visualization_count < len(prediction_names):
        step = max(1, len(prediction_names) // visualization_count)
        selected_names = prediction_names[::step][:visualization_count]
    for name in selected_names:
        with Image.open(images_dir / name) as image, read_prediction(predictions, name) as prediction:
            make_triptych(image, prediction).save(visualization_dir / name)

    result = {
        "predictions": str(predictions),
        "images_dir": str(images_dir),
        "image_count": len(image_names),
        "prediction_count": len(prediction_names),
        "size": list(image_size or ()),
        "visualization_count": len(selected_names),
        "classes": {
            str(class_id): {
                "name": CLASS_NAMES[class_id],
                "pixels": counts[class_id],
                "ratio": counts[class_id] / total_pixels if total_pixels else 0.0,
            }
            for class_id in range(NUM_CLASSES)
        },
    }
    (output_dir / "submission_prediction_distribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction directory or zip file")
    parser.add_argument("--images-dir", type=Path, required=True, help="Original test image directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visualization-count", type=int, default=20)
    args = parser.parse_args()
    if args.visualization_count <= 0:
        raise ValueError("--visualization-count must be positive")
    result = analyze_submission(args.predictions, args.images_dir, args.output_dir, args.visualization_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())