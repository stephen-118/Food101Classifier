import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision

from data import SPLIT_SEED, VALIDATION_FRACTION, build_transforms, create_loaders
from engine import evaluate, train_model
from model import PRETRAINED_WEIGHTS, FoodTransferModel
from visualize import plot_confusion_matrix, plot_history, save_gradcam_examples


def parse_args():
    # Define command-line arguments.
    parser = argparse.ArgumentParser(description="Food-101 transfer learning with MobileNetV3-Small")
    parser.add_argument("--mode", choices=["train", "evaluate", "gradcam", "all"], default="all")
    parser.add_argument("--strategy", choices=["frozen", "full"], default="frozen",
                        help="frozen: train the classification head only; full: fine-tune all layers")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=37,
                        help="Training RNG seed; the data split always uses seed 37")
    parser.add_argument("--workers", type=int, default=0,
                        help="Use 0 on Windows; increase on Linux if desired")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit the official training pool BEFORE its 90/10 train/validation split")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Limit official test images for a quick check; omit for final results")
    parser.add_argument("--gradcam-count", type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def save_json(path, value, backup=False):
    if backup and path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        print(f"Previous {path.name} backed up to: {backup_path.resolve()}")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def build_run_config(args, model, device, loaders, use_pretrained, checkpoint):
    train_loader, val_loader, test_loader = loaders
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "strategy": args.strategy,
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "working_directory": str(Path.cwd()),
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "data": {
            "dataset": "Food-101",
            "train_source": "official train",
            "validation_source": "official train",
            "test_source": "official test",
            "validation_fraction": VALIDATION_FRACTION,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "official_test_samples": len(getattr(test_loader.dataset, "dataset", test_loader.dataset)),
            "train_transforms": repr(build_transforms(True)),
            "evaluation_transforms": repr(build_transforms(False)),
        },
        "pretrained_weights": {
            "name": f"MobileNet_V3_Small_Weights.{PRETRAINED_WEIGHTS.name}",
            "url": PRETRAINED_WEIGHTS.url,
            "loaded_for_this_run": use_pretrained,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
            "precision": "float32 (no AMP)",
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_path": str(checkpoint.resolve()),
        "best_epoch": None,
        "training": None,
        "measurement_scope": {
            "train_seconds": "Training batch loop including data loading, transfers, forward, backward, optimizer steps; CUDA synchronized before/after",
            "val_seconds": "Validation only, measured separately with CUDA synchronization",
            "train_peak_allocated_mib": "Peak PyTorch tensor memory allocated on the selected CUDA device during the training batch loop, including resident model/optimizer tensors; null on CPU",
            "excluded": "Model setup, checkpoint I/O, plots, final test, and Grad-CAM",
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Use the same FP32 precision settings for both strategies.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_mobilenetv3_food101.pth"
    # Use the GPU if CUDA is available; otherwise, use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = create_loaders(
        args.data_dir, args.batch_size, args.workers,
        args.max_train_samples, args.max_test_samples, seed=args.seed
    )
    print(f"Samples: train={len(train_loader.dataset)}, "
          f"validation={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print("Train: update model parameters; validation: select checkpoint by Top-1; "
          "official test: report final Top-1 / Top-5.")

    use_pretrained = args.mode in ("train", "all")
    # Create the model.
    model = FoodTransferModel(
        num_classes=len(classes),
        use_pretrained=use_pretrained,
        freeze_backbone=args.strategy == "frozen",
    ).to(device)
    # Define the loss function.
    criterion = nn.CrossEntropyLoss()
    config_path = output_dir / "run_config.json"
    run_config = build_run_config(args, model, device,
                                  (train_loader, val_loader, test_loader), use_pretrained, checkpoint)
    save_json(config_path, run_config, backup=True)

    # Train the model.
    if args.mode in ("train", "all"):
        print(f"Training strategy: {args.strategy}")
        # Pass only trainable parameters to the optimizer.
        trainable_parameters = filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        )
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
        run_config["optimizer"] = {"name": "Adam", "defaults": optimizer.defaults}
        save_json(config_path, run_config)
        try:
            history = train_model(model, train_loader, val_loader, criterion,
                                  optimizer, device, args.epochs, checkpoint)
        except Exception as error:
            run_config.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_json(config_path, run_config)
            raise
        # Persist training measurements before test evaluation or Grad-CAM can allocate memory.
        run_config["best_epoch"] = history["best_epoch"]
        run_config["training"] = {key: history[key] for key in (
            "best_val_accuracy", "total_train_seconds", "total_val_seconds", "peak_train_allocated_mib")}
        run_config["status"] = "training_complete"
        save_json(config_path, run_config)
        save_json(output_dir / "history.json", history)
        plot_history(history, output_dir)
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Evaluate the model on the test set.
    if args.mode in ("evaluate", "all"):
        official_test_samples = len(test_loader.dataset.dataset)
        print(f"Evaluating checkpoint: {checkpoint.resolve()}")
        test_loss, top1, top5, confusion = evaluate(
            model, test_loader, criterion, device, len(classes)
        )
        # Count images actually processed, rather than inferring this from CLI limits.
        test_samples = int(confusion.sum().item())
        is_full_test_set = test_samples == official_test_samples
        print(f"Official test samples evaluated: {test_samples}/{official_test_samples} "
              f"({'full test set' if is_full_test_set else 'subset / quick check'})")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Top-1 accuracy: {top1:.2%}")
        print(f"Top-5 accuracy: {top5:.2%}")
        plot_confusion_matrix(confusion, classes, output_dir)
        metrics = {
            "test_loss": test_loss,
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "evaluation_split": "Food-101 official test",
            "test_samples": test_samples,
            "official_test_samples": official_test_samples,
            "is_full_test_set": is_full_test_set,
            "checkpoint_path": str(checkpoint.resolve()),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            # Keep argv as a list to preserve spaces and argument boundaries.
            "command": [sys.executable, *sys.argv],
            "args": vars(args),
            "split_seed": SPLIT_SEED,
            "seed": args.seed,
            "validation_fraction": VALIDATION_FRACTION,
        }
        metrics_path = output_dir / "metrics.json"
        save_json(metrics_path, metrics, backup=True)
        run_config["test"] = {"test_loss": test_loss, "top1_accuracy": top1,
                              "top5_accuracy": top5, "test_samples": test_samples,
                              "is_full_test_set": is_full_test_set}
        save_json(config_path, run_config)
        print(f"Test metrics saved in: {metrics_path.resolve()}")

    # Generate Grad-CAM visualizations.
    if args.mode in ("gradcam", "all"):
        save_gradcam_examples(model, test_loader, classes, device,
                              output_dir, args.gradcam_count)
        print(f"Grad-CAM images saved in: {output_dir.resolve()}")

    run_config.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    save_json(config_path, run_config)
    print(f"Run configuration saved in: {config_path.resolve()}")


if __name__ == "__main__":
    main()
