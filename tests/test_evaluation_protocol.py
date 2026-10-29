import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import data
from engine import evaluate, train_model
import main as cli


class FakeFood101:
    """Metadata-only dataset: split tests never read or download images."""

    def __init__(self, root, split, download, transform):
        self.split = split
        self.transform = transform
        self.classes = [str(i) for i in range(101)]

    def __len__(self):
        return 75750 if self.split == "train" else 25250


class EvaluationProtocolTests(unittest.TestCase):
    @patch("data.Food101", FakeFood101)
    def test_official_splits_and_limits_are_separate_and_reproducible(self):
        train, val, test, _ = data.create_loaders("unused", 32)
        self.assertEqual([len(x.dataset) for x in (train, val, test)],
                         [68175, 7575, 25250])
        self.assertTrue(set(train.dataset.indices).isdisjoint(val.dataset.indices))
        self.assertEqual(len(set(train.dataset.indices) | set(val.dataset.indices)), 75750)
        self.assertEqual(val.dataset.dataset.split, "train")
        self.assertEqual(test.dataset.dataset.split, "test")
        self.assertEqual([type(t).__name__ for t in val.dataset.dataset.transform.transforms],
                         ["Resize", "ToTensor", "Normalize"])
        limited = data.create_loaders("unused", 32, max_train=2000, max_test=500)
        repeated = data.create_loaders("unused", 32, max_train=2000, max_test=500)
        self.assertEqual([len(x.dataset) for x in limited[:3]], [1800, 200, 500])
        for first, second in zip(limited[:3], repeated[:3]):
            self.assertEqual(first.dataset.indices, second.dataset.indices)
        for limit in (0, -1):
            with self.assertRaises(ValueError):
                data.create_loaders("unused", 32, max_test=limit)

    def test_evaluate_counts_actual_samples_including_partial_batch(self):
        logits = torch.eye(5) * 4
        labels = torch.tensor([0, 1, 2, 3, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        loss, top1, top5, confusion = evaluate(nn.Identity(), loader,
                                              nn.CrossEntropyLoss(), "cpu", 5)
        self.assertAlmostEqual(loss, nn.CrossEntropyLoss()(logits, labels).item(), places=6)
        self.assertEqual(top1, 0.8)
        self.assertEqual(top5, 1.0)
        self.assertEqual(confusion.sum().item(), 5)
        empty = DataLoader(TensorDataset(logits[:0], labels[:0]))
        with self.assertRaisesRegex(ValueError, "at least one image"):
            evaluate(nn.Identity(), empty, nn.CrossEntropyLoss(), "cpu", 5)

    def test_checkpoint_selection_uses_validation_and_restores_best(self):
        model = nn.Linear(2, 5)
        loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        snapshots = []

        def validation(current_model, current_loader, criterion, device):
            self.assertIs(current_loader, val_loader)
            snapshots.append({k: v.clone() for k, v in current_model.state_dict().items()})
            return 1.0, (0.9 if len(snapshots) == 1 else 0.2), 1.0, None

        val_loader = object()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, \
             patch("engine.evaluate", side_effect=validation):
            with contextlib.redirect_stdout(io.StringIO()):
                train_model(model, loader, val_loader, nn.CrossEntropyLoss(), optimizer,
                            "cpu", 2, Path(folder) / "best.pth")
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, snapshots[0][name]))
        self.assertFalse(torch.equal(snapshots[0]["weight"], snapshots[1]["weight"]))

    def test_cli_records_full_and_subset_evaluations_and_preserves_previous_metrics(self):
        dataset = TensorDataset(torch.eye(5), torch.arange(5))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            checkpoint = Path(folder) / "best_mobilenetv3_food101.pth"
            model = nn.Linear(5, 5)
            torch.save(model.state_dict(), checkpoint)
            original_checkpoint = checkpoint.read_bytes()
            metrics_path = Path(folder) / "metrics.json"
            old_metrics = '{"top1_accuracy": 0.61}'
            metrics_path.write_text(old_metrics, encoding="utf-8")
            for count in (3, 5):
                loader = DataLoader(Subset(dataset, list(range(count))), batch_size=2)
                args = argparse.Namespace(mode="evaluate", data_dir="unused", output_dir=folder,
                                          epochs=10, batch_size=2, learning_rate=0.001,
                                          workers=0, max_train_samples=None,
                                          max_test_samples=3 if count == 3 else None, gradcam_count=6)
                argv = ["main.py", "--mode", "evaluate", "--output-dir", folder]
                if count == 3:
                    argv += ["--max-test-samples", "3"]
                with patch.object(cli, "parse_args", return_value=args), \
                     patch.object(cli, "create_loaders", return_value=(loader, loader, loader, list("abcde"))), \
                     patch.object(cli, "FoodTransferModel", return_value=model), \
                     patch.object(cli, "plot_confusion_matrix"), \
                     patch.object(cli.torch.cuda, "is_available", return_value=False), \
                     patch.object(cli.sys, "argv", argv), \
                     contextlib.redirect_stdout(io.StringIO()):
                    cli.main()
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(metrics["test_samples"], count)
                self.assertEqual(metrics["official_test_samples"], 5)
                self.assertEqual(metrics["is_full_test_set"], count == 5)
                self.assertEqual(metrics["command"][1:], argv)
                self.assertEqual(metrics["args"]["max_test_samples"], args.max_test_samples)
                self.assertEqual(metrics["checkpoint_path"], str(checkpoint.resolve()))
                self.assertTrue(metrics["evaluated_at"].endswith("+00:00"))
            backups = list(Path(folder).glob("metrics.backup-*.json"))
            self.assertEqual(len(backups), 2)
            self.assertIn(old_metrics, [p.read_text(encoding="utf-8") for p in backups])
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)


if __name__ == "__main__":
    unittest.main()
