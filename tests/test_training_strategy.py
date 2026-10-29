import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import main as cli
from model import FoodTransferModel


class TrainingStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_default_freezes_backbone_and_keeps_classifier_trainable(self):
        model = FoodTransferModel(num_classes=5, use_pretrained=False)
        self.assertTrue(model.freeze_backbone)
        self.assertTrue(all(not p.requires_grad for p in model.network.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.network.classifier.parameters()))

    def test_training_step_updates_backbone_and_batchnorm_only_for_full_strategy(self):
        for frozen in (True, False):
            with self.subTest(freeze_backbone=frozen):
                torch.manual_seed(37)
                model = FoodTransferModel(num_classes=5, use_pretrained=False,
                                          freeze_backbone=frozen)
                self.assertIs(model.train(), model)
                self.assertTrue(model.network.classifier.training)
                batchnorm = next(m for m in model.network.features.modules()
                                 if isinstance(m, nn.BatchNorm2d))
                self.assertEqual(batchnorm.training, not frozen)
                batches_before = batchnorm.num_batches_tracked.item()
                backbone_before = {k: v.clone() for k, v in model.network.features.state_dict().items()}
                classifier_before = model.network.classifier[3].weight.detach().clone()
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=0.001)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 32, 32)), torch.tensor([0, 1]))
                loss.backward()
                self.assertTrue(all((p.grad is None) == frozen
                                    for p in model.network.features.parameters()))
                optimizer.step()
                backbone_changed = any(not torch.equal(value, backbone_before[name])
                                       for name, value in model.network.features.state_dict().items())
                self.assertEqual(backbone_changed, not frozen)
                self.assertEqual(batchnorm.num_batches_tracked.item() - batches_before,
                                 0 if frozen else 1)
                self.assertFalse(torch.equal(model.network.classifier[3].weight, classifier_before))
                self.assertIs(model.eval(), model)
                self.assertTrue(all(not m.training for m in model.modules()))
                model.train()
                self.assertEqual(batchnorm.training, not frozen)

    def test_strategy_does_not_change_checkpoint_format(self):
        frozen = FoodTransferModel(num_classes=5, use_pretrained=False)
        full = FoodTransferModel(num_classes=5, use_pretrained=False, freeze_backbone=False)
        full.load_state_dict(frozen.state_dict(), strict=True)
        self.assertEqual(list(frozen.state_dict()), list(full.state_dict()))

    def test_cli_defaults_to_frozen_and_rejects_unknown_strategy(self):
        with patch.object(cli.sys, "argv", ["main.py"]):
            self.assertEqual(cli.parse_args().strategy, "frozen")
        with patch.object(cli.sys, "argv", ["main.py", "--strategy", "unknown"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_cli_training_passes_strategy_and_optimizes_only_trainable_parameters(self):
        for strategy in ("frozen", "full"):
            with self.subTest(strategy=strategy), \
                 tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                model = FoodTransferModel(num_classes=5, use_pretrained=False,
                                          freeze_backbone=strategy == "frozen")
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)))

                def check_optimizer(current_model, train_loader, val_loader, criterion,
                                    optimizer, device, epochs, checkpoint_path):
                    self.assertIs(current_model, model)
                    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
                    expected = {id(p) for p in model.parameters() if p.requires_grad}
                    self.assertEqual(actual, expected)
                    backbone = {id(p) for p in model.network.features.parameters()}
                    self.assertEqual(bool(actual & backbone), strategy == "full")
                    return {"train_loss": [1.0], "val_loss": [1.0], "val_accuracy": [0.5],
                            "best_epoch": 1, "best_val_accuracy": 0.5,
                            "total_train_seconds": 2.0, "total_val_seconds": 1.0,
                            "peak_train_allocated_mib": None}

                argv = ["main.py", "--mode", "train", "--strategy", strategy,
                        "--output-dir", folder, "--epochs", "1"]
                with patch.object(cli.sys, "argv", argv), \
                     patch.object(cli, "create_loaders", return_value=(loader, loader, loader, list("abcde"))), \
                     patch.object(cli, "FoodTransferModel", return_value=model) as factory, \
                     patch.object(cli, "train_model", side_effect=check_optimizer) as train, \
                     patch.object(cli, "plot_history"), \
                     patch.object(cli.torch.cuda, "is_available", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    cli.main()
                factory.assert_called_once_with(num_classes=5, use_pretrained=True,
                                                freeze_backbone=strategy == "frozen")
                train.assert_called_once()
                config = json.loads((Path(folder) / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["status"], "completed")
                self.assertEqual(config["seed"], 37)
                self.assertEqual(config["best_epoch"], 1)
                self.assertEqual(config["total_parameters"], sum(p.numel() for p in model.parameters()))
                self.assertEqual(config["trainable_parameters"],
                                 sum(p.numel() for p in model.parameters() if p.requires_grad))
                self.assertIsNone(config["training"]["peak_train_allocated_mib"])
                self.assertIn("pytorch", config["environment"])
                self.assertIn("torchvision", config["environment"])
                self.assertTrue(config["pretrained_weights"]["loaded_for_this_run"])


if __name__ == "__main__":
    unittest.main()
