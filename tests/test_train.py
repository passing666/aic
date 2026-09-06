from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import SegmentationDataset, compute_miou, restore_random_state, save_checkpoint, validate_resume_config


class DatasetTests(unittest.TestCase):
    def test_multiscale_transform_preserves_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            (root / "split.csv").write_text("id\nsample\n", encoding="utf-8")
            Image.fromarray(np.full((12, 12, 3), 128, dtype=np.uint8)).save(root / "images" / "sample.png")
            Image.fromarray(np.tile(np.arange(1, 9, dtype=np.uint8), (12, 2))[:, :12]).save(
                root / "masks" / "sample.png"
            )
            dataset = SegmentationDataset(
                root / "split.csv",
                root,
                image_size=16,
                train=True,
                multiscale=True,
                scale_min=1.0,
                scale_max=1.0,
                color_jitter=0.1,
                vertical_flip=True,
            )

            image, mask = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(tuple(mask.shape), (16, 16))
        self.assertTrue(set(mask.unique().tolist()).issubset(set(range(9))))
        self.assertIn(0, mask.unique().tolist())

    def test_validation_transform_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            (root / "split.csv").write_text("id\nsample\n", encoding="utf-8")
            Image.fromarray(np.full((12, 12, 3), 128, dtype=np.uint8)).save(root / "images" / "sample.png")
            Image.fromarray(np.ones((12, 12), dtype=np.uint8)).save(root / "masks" / "sample.png")
            dataset = SegmentationDataset(root / "split.csv", root, image_size=16, train=False)

            first = dataset[0]
            second = dataset[0]

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_contains_training_state_and_restores_random_state(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        run_config = {"model": "test"}

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest_checkpoint.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler, 3, 0.75, run_config)
            expected_values = (random.random(), float(np.random.random()), float(torch.rand(1)))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            restore_random_state(checkpoint)
            restored_values = (random.random(), float(np.random.random()), float(torch.rand(1)))

        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["best_miou"], 0.75)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("scaler_state_dict", checkpoint)
        self.assertIn("scheduler_state_dict", checkpoint)
        self.assertEqual(restored_values, expected_values)

    def test_resume_rejects_changed_training_configuration(self) -> None:
        saved = {
            "model": "SegFormer MIT-B0",
            "pretrained": "mit-b0",
            "num_classes": 9,
            "ignore_index": 0,
            "image_size": 768,
            "batch_size": 2,
            "learning_rate": 6e-5,
            "amp": True,
        }
        current = dict(saved, image_size=1024)

        with self.assertRaisesRegex(ValueError, "image_size"):
            validate_resume_config(saved, current)

    def test_resume_accepts_legacy_default_augmentation_configuration(self) -> None:
        saved = {
            "model": "SegFormer MIT-B0",
            "pretrained": "mit-b0",
            "num_classes": 9,
            "ignore_index": 0,
            "image_size": 768,
            "batch_size": 2,
            "learning_rate": 6e-5,
            "amp": True,
            "scheduler": "cosine",
            "warmup_epochs": 2,
        }
        current = dict(
            saved,
            multiscale=False,
            scale_min=0.5,
            scale_max=1.25,
            color_jitter=0.0,
            vertical_flip=False,
        )

        validate_resume_config(saved, current)


class MetricTests(unittest.TestCase):
    def test_ignore_target_pixels_do_not_affect_miou(self) -> None:
        targets = torch.tensor([[[0, 1], [2, 2]]])
        predictions = torch.tensor([[[2, 1], [2, 1]]])
        logits = torch.nn.functional.one_hot(predictions, num_classes=9).permute(0, 3, 1, 2).float()

        miou, ious = compute_miou(logits, targets)

        self.assertEqual(ious[1], 0.5)
        self.assertEqual(ious[2], 0.5)
        self.assertEqual(miou, 0.5)


if __name__ == "__main__":
    unittest.main()