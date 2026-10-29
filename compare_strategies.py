"""Run frozen/full experiments sequentially in separate processes and directories."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def run_experiment(command, log_path):
    print(f"Running: {subprocess.list2cmdline(command)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="For smoke checks only; omit for full experiments")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="For smoke checks only; omit for full experiments")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in ("frozen", "full"):
        output = root / f"{strategy}_seed{args.seed}"
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).with_name("main.py")),
                   "--mode", "all", "--strategy", strategy, "--output-dir", str(output)]
        for name, value in vars(args).items():
            if name != "output_dir" and value is not None:
                command += [f"--{name.replace('_', '-')}", str(value)]
        # Each process exits before the next starts, releasing its CUDA allocations.
        run_experiment(command, output / "run.log")
        with open(output / "run_config.json", encoding="utf-8") as file:
            config = json.load(file)
        with open(output / "metrics.json", encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append({
            "strategy": strategy,
            "seed": args.seed,
            "epochs": args.epochs,
            "train_samples": config["data"]["train_samples"],
            "validation_samples": config["data"]["validation_samples"],
            "test_samples": metrics["test_samples"],
            "is_full_test_set": metrics["is_full_test_set"],
            "gpu_name": config["environment"]["gpu_name"],
            "total_parameters": config["total_parameters"],
            "trainable_parameters": config["trainable_parameters"],
            "best_epoch": config["best_epoch"],
            **config["training"],
            "test_top1_accuracy": metrics["top1_accuracy"],
            "test_top5_accuracy": metrics["top5_accuracy"],
            "output_dir": str(output),
        })
    with open(root / f"comparison_seed{args.seed}.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    with open(root / f"comparison_seed{args.seed}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison saved in: {root / f'comparison_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()
