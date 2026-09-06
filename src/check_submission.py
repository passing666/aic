"""Validate competition segmentation PNG files before upload."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

NUM_CLASSES = 9


def check_submission(prediction_dir: Path, expected_count: int, expected_size: int) -> None:
    files = sorted(prediction_dir.glob("*.png"))
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} PNG files, found {len(files)}")
    for path in files:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"{path.name}: expected mode L, found {image.mode}")
            if image.size != (expected_size, expected_size):
                raise ValueError(f"{path.name}: expected {expected_size}x{expected_size}, found {image.size}")
            minimum, maximum = image.getextrema()
            if minimum < 0 or maximum >= NUM_CLASSES:
                raise ValueError(f"{path.name}: pixel value outside 0-{NUM_CLASSES - 1}")
    print(f"submission check passed: files={len(files)} size={expected_size}x{expected_size} mode=L labels=0-{NUM_CLASSES - 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--expected-size", type=int, default=1024)
    args = parser.parse_args()
    check_submission(args.prediction_dir, args.expected_count, args.expected_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())