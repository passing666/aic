"""Benchmark SegFormer inference throughput for candidate batch sizes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    if args.image_size <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("image-size must be positive, warmup non-negative, and steps positive")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device).eval()
    results = []
    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        images = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)
        try:
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(pixel_values=images).logits
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.steps):
                    model(pixel_values=images).logits
                if device.type == "cuda":
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "seconds_per_step": elapsed / args.steps,
                "images_per_second": batch_size * args.steps / elapsed,
            }
            if device.type == "cuda":
                result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "status": "oom"}
        results.append(result)
        print(result)
    print({"checkpoint": str(args.checkpoint), "image_size": args.image_size, "device": str(device), "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
