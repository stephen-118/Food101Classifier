import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import compare_strategies
import data
from engine import train_model
from test_evaluation_protocol import FakeFood101


class CPUOnlyBatch:
    """Exercise CUDA measurement ordering on CPU without allocating CUDA tensors."""

    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        return self.tensor


class TrainingMeasurementTests(unittest.TestCase):
    def test_cpu_times_totals_best_epoch_and_no_cuda_calls(self):
        model = nn.Linear(2, 5)
        loader = [(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, \
             patch("engine.time.perf_counter", side_effect=[0, 2, 3, 4, 10, 13, 14, 16]), \
             patch("engine.evaluate", side_effect=[(1., .9, 1., None), (1., .8, 1., None)]), \
             patch("engine.torch.cuda.synchronize") as sync, \
             patch("engine.torch.cuda.reset_peak_memory_stats") as reset, \
             patch("engine.torch.cuda.max_memory_allocated") as peak, \
             contextlib.redirect_stdout(io.StringIO()):
            history = train_model(model, loader, loader, nn.CrossEntropyLoss(),
                                  torch.optim.SGD(model.parameters(), lr=.1), "cpu", 2,
                                  Path(folder) / "best.pth")
        self.assertEqual(history["train_seconds"], [2, 3])
        self.assertEqual(history["val_seconds"], [1, 2])
        self.assertEqual(history["total_train_seconds"], 5)
        self.assertEqual(history["total_val_seconds"], 3)
        self.assertEqual(history["best_epoch"], 1)
        self.assertEqual(history["best_val_accuracy"], .9)
        self.assertEqual(history["train_peak_allocated_mib"], [None, None])
        self.assertIsNone(history["peak_train_allocated_mib"])
        sync.assert_not_called()
        reset.assert_not_called()
        peak.assert_not_called()

    def test_cuda_sync_reset_and_peak_read_happen_before_validation(self):
        events = []
        model = nn.Linear(2, 5)
        loader = [(CPUOnlyBatch(torch.ones(2, 2)),
                   CPUOnlyBatch(torch.zeros(2, dtype=torch.long)))]

        def clock():
            events.append("clock")
            return float(len(events))

        def validation(*args):
            events.append("validation")
            return 1., .9, 1., None

        with patch("engine.time.perf_counter", side_effect=clock), \
             patch("engine.evaluate", side_effect=validation), \
             patch("engine.torch.cuda.synchronize", side_effect=lambda device: events.append("sync")), \
             patch("engine.torch.cuda.reset_peak_memory_stats", side_effect=lambda device: events.append("reset")), \
             patch("engine.torch.cuda.max_memory_allocated",
                   side_effect=lambda device: events.append("peak") or 7 * 1024 ** 2), \
             patch("engine.torch.save"), \
             patch("engine.torch.load", return_value=model.state_dict()), \
             contextlib.redirect_stdout(io.StringIO()):
            history = train_model(model, loader, loader, nn.CrossEntropyLoss(),
                                  torch.optim.SGD(model.parameters(), lr=.1), "cuda:0", 1,
                                  Path("unused.pth"))
        self.assertEqual(events, ["sync", "reset", "clock", "sync", "clock", "peak",
                                  "sync", "clock", "validation", "sync", "clock"])
        self.assertEqual(history["train_peak_allocated_mib"], [7.])
        self.assertEqual(history["peak_train_allocated_mib"], 7.)

    @patch("data.Food101", FakeFood101)
    def test_training_seed_changes_order_without_changing_data_split(self):
        first = data.create_loaders("unused", 32, max_train=20, seed=37)
        second = data.create_loaders("unused", 32, max_train=20, seed=99)
        for left, right in zip(first[:3], second[:3]):
            self.assertEqual(left.dataset.indices, right.dataset.indices)
        self.assertNotEqual(list(first[0].sampler), list(second[0].sampler))

    def test_comparison_runs_in_order_with_matching_settings_and_writes_summary(self):
        commands = []

        def fake_run(command, log_path):
            commands.append(command)
            config = {"data": {"train_samples": 68175, "validation_samples": 7575},
                      "environment": {"gpu_name": "test GPU"}, "total_parameters": 100,
                      "trainable_parameters": 50 if len(commands) == 1 else 100,
                      "best_epoch": 2, "training": {"total_train_seconds": 10.,
                      "total_val_seconds": 2., "peak_train_allocated_mib": 7., "best_val_accuracy": .8}}
            (log_path.parent / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            (log_path.parent / "metrics.json").write_text(json.dumps({"test_samples": 25250,
                "is_full_test_set": True, "top1_accuracy": .7, "top5_accuracy": .9}), encoding="utf-8")

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, \
             patch.object(compare_strategies.sys, "argv", ["compare_strategies.py", "--output-dir", folder]), \
             patch.object(compare_strategies, "run_experiment", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            compare_strategies.main()
            rows = json.loads((Path(folder) / "comparison_seed37.json").read_text(encoding="utf-8"))
            self.assertTrue((Path(folder) / "comparison_seed37.csv").exists())
        self.assertEqual([row["strategy"] for row in rows], ["frozen", "full"])
        self.assertEqual(len(commands), 2)
        for command, strategy in zip(commands, ("frozen", "full")):
            for flag, expected in (("--epochs", "10"), ("--batch-size", "32"),
                                   ("--seed", "37"), ("--learning-rate", "0.001"),
                                   ("--strategy", strategy), ("--mode", "all")):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertTrue(command[command.index("--output-dir") + 1].endswith(f"{strategy}_seed37"))
            self.assertNotIn("--max-train-samples", command)
            self.assertNotIn("--max-test-samples", command)

    def test_comparison_stops_when_first_experiment_fails(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, \
             patch.object(compare_strategies.sys, "argv", ["compare_strategies.py", "--output-dir", folder]), \
             patch.object(compare_strategies, "run_experiment",
                          side_effect=subprocess.CalledProcessError(1, "test")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                compare_strategies.main()
            run.assert_called_once()
            self.assertFalse((Path(folder) / "comparison_seed37.csv").exists())


if __name__ == "__main__":
    unittest.main()
