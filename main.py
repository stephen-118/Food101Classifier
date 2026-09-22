import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from data import create_loaders
from engine import evaluate, train_model
from model import FoodTransferModel
from visualize import plot_confusion_matrix, plot_history, save_gradcam_examples


def parse_args():
    # 命令行参数
    parser = argparse.ArgumentParser(description="Food-101 transfer learning with MobileNetV3-Small")
    parser.add_argument("--mode", choices=["train", "evaluate", "gradcam", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0,
                        help="Use 0 on Windows; increase on Linux if desired")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--gradcam-count", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(37)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_mobilenetv3_food101.pth"
    # 有 NVIDIA GPU 和 CUDA 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = create_loaders(
        args.data_dir, args.batch_size, args.workers,
        args.max_train_samples, args.max_test_samples
    )
    print(f"Samples: train={len(train_loader.dataset)}, "
          f"validation={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    use_pretrained = args.mode in ("train", "all")
    # 创建模型
    model = FoodTransferModel(num_classes=len(classes),
                              use_pretrained=use_pretrained).to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练
    if args.mode in ("train", "all"):
        # 只把可训练参数交给优化器
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

    # 评估
    if args.mode in ("evaluate", "all"):
        test_loss, top1, top5, confusion = evaluate(
            model, test_loader, criterion, device, len(classes)
        )
        print(f"Test loss: {test_loss:.4f}")
        print(f"Top-1 accuracy: {top1:.2%}")
        print(f"Top-5 accuracy: {top5:.2%}")
        plot_confusion_matrix(confusion, classes, output_dir)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump({"test_loss": test_loss, "top1_accuracy": top1,
                       "top5_accuracy": top5}, file, indent=2)

    # 绘图
    if args.mode in ("gradcam", "all"):
        save_gradcam_examples(model, test_loader, classes, device,
                              output_dir, args.gradcam_count)
        print(f"Grad-CAM images saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
