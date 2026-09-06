"""Validate and package prediction PNGs directly at the ZIP root."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--expected-size", type=int, default=1024)
    args = parser.parse_args()
    paths = sorted(args.predictions.glob("*.png"))
    if len(paths) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} PNG files, found {len(paths)}")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (args.expected_size, args.expected_size) or image.mode != "L":
                raise ValueError(f"invalid image format: {path.name}, size={image.size}, mode={image.mode}")
        with Image.open(path) as image:
            values = set(image.getdata())
        if not values.issubset(set(range(9))):
            raise ValueError(f"invalid label values in {path.name}: {sorted(values)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        if len(names) != args.expected_count or any("/" in name for name in names):
            raise RuntimeError("submission archive must contain PNGs directly at its root")
    print(f"packaged={len(paths)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
