"""Train a small SegFormer semantic-segmentation baseline.

The split CSV stores sample IDs and may contain Windows paths. This script
uses the sample ID with --data-root so the same split works on AutoDL/Linux.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

NUM_CLASSES = 9
IGNORE_INDEX = 0
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
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


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        image_size: int,
        train: bool,
        multiscale: bool = False,
        scale_min: float = 0.5,
        scale_max: float = 1.25,
        color_jitter: float = 0.0,
        vertical_flip: bool = False,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.train = train
        self.multiscale = multiscale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.color_jitter = color_jitter
        self.vertical_flip = vertical_flip
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            self.sample_ids = [row["id"] for row in csv.DictReader(file)]
        if not self.sample_ids:
            raise ValueError(f"No samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.vertical_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if self.multiscale:
            scale = random.uniform(self.scale_min, self.scale_max)
            scaled_width = max(1, round(image.width * scale))
            scaled_height = max(1, round(image.height * scale))
            image = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
            mask = mask.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
            source_left = random.randint(0, max(scaled_width - self.image_size, 0))
            source_top = random.randint(0, max(scaled_height - self.image_size, 0))
            source_width = min(scaled_width, self.image_size)
            source_height = min(scaled_height, self.image_size)
            box = (source_left, source_top, source_left + source_width, source_top + source_height)
            destination_left = random.randint(0, max(self.image_size - scaled_width, 0))
            destination_top = random.randint(0, max(self.image_size - scaled_height, 0))
            fill_color = tuple((IMAGE_MEAN * 255).round().astype(int))
            image_canvas = Image.new("RGB", (self.image_size, self.image_size), fill_color)
            mask_canvas = Image.new("L", (self.image_size, self.image_size), IGNORE_INDEX)
            image_canvas.paste(image.crop(box), (destination_left, destination_top))
            mask_canvas.paste(mask.crop(box), (destination_left, destination_top))
            image, mask = image_canvas, mask_canvas
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.color_jitter:
            lower = 1.0 - self.color_jitter
            upper = 1.0 + self.color_jitter
            enhancements = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
            for enhancement in random.sample(enhancements, len(enhancements)):
                image = enhancement(image).enhance(random.uniform(lower, upper))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[index]
        image_path = self.data_root / "images" / f"{sample_id}.png"
        mask_path = self.data_root / "masks" / f"{sample_id}.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
        if self.train:
            image, mask = self._train_transform(image, mask)
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
        mask_array = np.asarray(mask, dtype=np.int64)
        if np.any((mask_array < 0) | (mask_array >= NUM_CLASSES)):
            raise ValueError(f"Invalid label in {mask_path}")
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


def compute_miou(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, list[float | None]]:
    predictions = logits.argmax(dim=1)
    ious: list[float | None] = [None] * NUM_CLASSES
    valid = targets != IGNORE_INDEX
    for class_id in range(1, NUM_CLASSES):
        predicted = (predictions == class_id) & valid
        actual = (targets == class_id) & valid
        union = (predicted | actual).sum().item()
        intersection = (predicted & actual).sum().item()
        ious[class_id] = intersection / union if union else None
    valid_ious = [value for value in ious if value is not None]
    return (sum(valid_ious) / len(valid_ious) if valid_ious else 0.0), ious


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float | None], float]:
    model.eval()
    total_intersection = [0] * NUM_CLASSES
    total_union = [0] * NUM_CLASSES
    total_valid_pixels = 0
    total_predicted_ignore = 0
    with torch.no_grad():
        for images, masks in loader:
            outputs = model(pixel_values=images.to(device)).logits
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            predictions = outputs.argmax(dim=1).cpu()
            valid = masks != IGNORE_INDEX
            total_valid_pixels += int(valid.sum())
            total_predicted_ignore += int(((predictions == IGNORE_INDEX) & valid).sum())
            for class_id in range(1, NUM_CLASSES):
                predicted = (predictions == class_id) & valid
                actual = (masks == class_id) & valid
                total_intersection[class_id] += int((predicted & actual).sum())
                total_union[class_id] += int((predicted | actual).sum())
    ious = [None] + [total_intersection[i] / total_union[i] if total_union[i] else None for i in range(1, NUM_CLASSES)]
    valid_ious = [value for value in ious if value is not None]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    predicted_ignore_ratio = total_predicted_ignore / total_valid_pixels if total_valid_pixels else 0.0
    return miou, ious, predicted_ignore_ratio


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_miou: float,
    run_config: dict[str, object],
) -> None:
    state = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "run_config": run_config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def restore_random_state(checkpoint: dict[str, object]) -> None:
    random.setstate(checkpoint["python_random_state"])  # type: ignore[arg-type]
    np.random.set_state(checkpoint["numpy_random_state"])  # type: ignore[arg-type]
    torch.set_rng_state(checkpoint["torch_random_state"])  # type: ignore[arg-type]
    cuda_random_state = checkpoint.get("cuda_random_state")
    if cuda_random_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_random_state)  # type: ignore[arg-type]


def validate_resume_config(saved: dict[str, object], current: dict[str, object]) -> None:
    immutable_keys = (
        "model", "pretrained", "num_classes", "ignore_index", "image_size", "batch_size",
        "learning_rate", "amp", "scheduler", "warmup_epochs", "multiscale", "scale_min", "scale_max",
        "color_jitter", "vertical_flip",
    )
    legacy_defaults = {
        "multiscale": False,
        "scale_min": 0.5,
        "scale_max": 1.25,
        "color_jitter": 0.0,
        "vertical_flip": False,
    }
    mismatches = [
        key for key in immutable_keys
        if saved.get(key, legacy_defaults.get(key)) != current.get(key)
    ]
    if mismatches:
        details = ", ".join(f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}" for key in mismatches)
        raise ValueError(f"Resume configuration does not match checkpoint: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing images/ and masks/")
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV containing an id column")
    parser.add_argument("--val-csv", type=Path, required=True, help="CSV containing an id column")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exp001_baseline"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--vertical-flip", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", default="nvidia/mit-b0")
    parser.add_argument("--log-file", type=Path, help="Write stdout and stderr to this file as well as the terminal")
    parser.add_argument("--resume", type=Path, help="Resume from a complete training checkpoint")
    parser.add_argument("--overwrite-output-dir", action="store_true", help="Allow a fresh run to replace experiment records")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    if args.resume and args.overwrite_output_dir:
        raise ValueError("--resume and --overwrite-output-dir cannot be used together")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        raise ValueError("--warmup-epochs must be between 0 and --epochs")
    if args.scale_min <= 0 or args.scale_max < args.scale_min:
        raise ValueError("--scale-min must be positive and no greater than --scale-max")
    if args.color_jitter < 0 or args.color_jitter >= 1:
        raise ValueError("--color-jitter must be in [0, 1)")
    record_paths = ("run_config.json", "history.json", "history.jsonl", "summary.json", "latest_checkpoint.pt", "best_model")
    existing_records = [name for name in record_paths if (args.output_dir / name).exists()]
    if existing_records and not args.resume and not args.overwrite_output_dir:
        raise FileExistsError(
            f"Output directory already contains experiment records: {', '.join(existing_records)}. "
            "Use a new --output-dir or pass --overwrite-output-dir."
        )
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a" if args.resume else "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
        sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]
    started_at = time.time()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    train_dataset = SegmentationDataset(
        args.train_csv,
        args.data_root,
        args.image_size,
        train=True,
        multiscale=args.multiscale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        color_jitter=args.color_jitter,
        vertical_flip=args.vertical_flip,
    )
    val_dataset = SegmentationDataset(args.val_csv, args.data_root, args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    config = SegformerConfig.from_pretrained(args.pretrained, num_labels=NUM_CLASSES, semantic_loss_ignore_index=IGNORE_INDEX)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained, config=config, ignore_mismatched_sizes=True
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    scaler_enabled = device.type == "cuda" and not args.no_amp
    run_config = {
        "model": "SegFormer MIT-B0",
        "pretrained": args.pretrained,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": list(CLASS_NAMES),
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "multiscale": args.multiscale,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "color_jitter": args.color_jitter,
        "vertical_flip": args.vertical_flip,
        "num_workers": args.num_workers,
        "amp": scaler_enabled,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    if args.scheduler == "cosine":
        cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        if args.warmup_epochs:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs),
                    cosine,
                ],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = cosine
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    best_miou = -1.0
    history: list[dict[str, object]] = []
    history_path = args.output_dir / "history.json"
    history_jsonl_path = args.output_dir / "history.jsonl"
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_run_config = checkpoint.get("run_config")
        if not isinstance(saved_run_config, dict):
            raise ValueError("Resume checkpoint is missing run_config")
        validate_resume_config(saved_run_config, run_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", scheduler.state_dict()))
        best_miou = float(checkpoint["best_miou"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        if args.epochs < start_epoch:
            raise ValueError(f"--epochs must be at least {start_epoch} when resuming epoch {completed_epoch}")
        if not history_path.is_file():
            raise FileNotFoundError(f"Resume history does not exist: {history_path}")
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list) or not loaded_history or int(loaded_history[-1]["epoch"]) != completed_epoch:
            raise ValueError("history.json does not end at the checkpoint epoch")
        history = loaded_history
        restore_random_state(checkpoint)
        print(json.dumps({"resumed_from": str(args.resume), "completed_epoch": completed_epoch}, ensure_ascii=False))
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"run_config": run_config}, ensure_ascii=False))
    history_jsonl = history_jsonl_path.open("a" if args.resume else "w", encoding="utf-8")

    print(f"device={device} train={len(train_dataset)} val={len(val_dataset)} image_size={args.image_size}")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started_at = time.time()
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(pixel_values=images.to(device)).logits
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = torch.nn.functional.cross_entropy(logits, masks.to(device), ignore_index=IGNORE_INDEX)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        miou, ious, predicted_ignore_ratio = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_miou": miou,
            "ious": ious,
            "predicted_ignore_ratio_on_valid_pixels": predicted_ignore_ratio,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - epoch_started_at,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        history_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        history_jsonl.flush()
        write_json(history_path, history)
        scheduler.step()
        if miou > best_miou:
            best_miou = miou
            model.save_pretrained(args.output_dir / "best_model")
            torch.save({"epoch": epoch, "val_miou": miou}, args.output_dir / "best_metrics.pt")
            write_json(
                args.output_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "val_miou": miou,
                    "ious": ious,
                    "predicted_ignore_ratio_on_valid_pixels": predicted_ignore_ratio,
                },
            )
            save_checkpoint(args.output_dir / "best_checkpoint.pt", model, optimizer, scaler, scheduler, epoch, best_miou, run_config)
        save_checkpoint(args.output_dir / "latest_checkpoint.pt", model, optimizer, scaler, scheduler, epoch, best_miou, run_config)
    history_jsonl.close()
    session_elapsed_seconds = time.time() - started_at
    elapsed_seconds = sum(float(record["elapsed_seconds"]) for record in history)
    best_record = max(history, key=lambda item: float(item["val_miou"]))
    summary = {
        "best_epoch": best_record["epoch"],
        "best_miou": best_miou,
        "final_miou": history[-1]["val_miou"],
        "final_epoch": history[-1]["epoch"],
        "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
        "best_checkpoint": str(args.output_dir / "best_checkpoint.pt"),
        "resumed_from": str(args.resume) if args.resume else None,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
        "session_elapsed_seconds": session_elapsed_seconds,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if log_handle:
        log_handle.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())