import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch
import torch.nn as nn

from data import SPLIT_SEED, VALIDATION_FRACTION, create_loaders
from engine import evaluate, train_model
from model import FoodTransferModel
from visualize import plot_confusion_matrix, plot_history, save_gradcam_examples


def parse_args():
    # Define command-line arguments.
    parser = argparse.ArgumentParser(description="Food-101 transfer learning with MobileNetV3-Small")
    parser.add_argument("--mode", choices=["train", "evaluate", "gradcam", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0,
                        help="Use 0 on Windows; increase on Linux if desired")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit the official training pool BEFORE its 90/10 train/validation split")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Limit official test images for a quick check; omit for final results")
    parser.add_argument("--gradcam-count", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SPLIT_SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_mobilenetv3_food101.pth"
    # Use the GPU if CUDA is available; otherwise, use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = create_loaders(
        args.data_dir, args.batch_size, args.workers,
        args.max_train_samples, args.max_test_samples
    )
    print(f"Samples: train={len(train_loader.dataset)}, "
          f"validation={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print("Train: update model parameters; validation: select checkpoint by Top-1; "
          "official test: report final Top-1 / Top-5.")

    use_pretrained = args.mode in ("train", "all")
    # Create the model.
    model = FoodTransferModel(num_classes=len(classes),
                              use_pretrained=use_pretrained).to(device)
    # Define the loss function.
    criterion = nn.CrossEntropyLoss()

    # Train the model.
    if args.mode in ("train", "all"):
        # Pass only trainable parameters to the optimizer.
        trainable_parameters = filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        )
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
        history = train_model(model, train_loader, val_loader, criterion,
                              optimizer, device, args.epochs, checkpoint)
        plot_history(history, output_dir)
        with open(output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
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
            "validation_fraction": VALIDATION_FRACTION,
        }
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = output_dir / f"metrics.backup-{timestamp}.json"
            shutil.copy2(metrics_path, backup_path)
            print(f"Previous metrics backed up to: {backup_path.resolve()}")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        print(f"Test metrics saved in: {metrics_path.resolve()}")

    # Generate Grad-CAM visualizations.
    if args.mode in ("gradcam", "all"):
        save_gradcam_examples(model, test_loader, classes, device,
                              output_dir, args.gradcam_count)
        print(f"Grad-CAM images saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
