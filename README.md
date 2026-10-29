# Food-101 Transfer Learning Classifier

```text
ImageNet-pretrained MobileNetV3-Small
              ↓
  Freeze or fine-tune the convolutional feature extractor
              ↓
Replace the 1,000-class output layer with a 101-class layer
              ↓
       Train on Food-101
```

## Quick Test Run

```bash
python main.py --mode all --epochs 2 --max-train-samples 2000 --max-test-samples 500
```

The first run automatically downloads the Food-101 dataset and the ImageNet-pretrained weights. This small sample size is intended only to verify that the program works correctly.

## Full Training

```bash
python main.py --mode all --epochs 10 --batch-size 32
```

By default (`--strategy frozen`), the convolutional feature extractor is frozen,
while the entire classification head is trained.

## Training Strategies

| Strategy | Parameters updated | Backbone BatchNorm during training |
| --- | --- | --- |
| `frozen` (default) | Entire classification head | Evaluation mode; running statistics stay fixed |
| `full` | Backbone and entire classification head | Training mode; running statistics are updated |

The optimizer includes only parameters with `requires_grad=True` for either
strategy. During evaluation and Grad-CAM, the entire model uses evaluation mode
regardless of the strategy. Existing weight checkpoints remain compatible.

For comparison experiments, keep the data split and other settings the same
and use separate output directories for each strategy:

```bash
python main.py --mode all --strategy frozen --seed 37 --epochs 10 --batch-size 32 --learning-rate 0.001 --output-dir outputs/frozen_seed37
python main.py --mode all --strategy full --seed 37 --epochs 10 --batch-size 32 --learning-rate 0.001 --output-dir outputs/full_seed37
```

Each directory stores its own best validation checkpoint and test results. To
evaluate a saved model, select the corresponding directory:

```bash
python main.py --mode evaluate --strategy full --output-dir outputs/full_seed37
```

The selected strategy is recorded in `metrics.json` under `args.strategy` for
the current invocation. For an existing checkpoint, this argument does not
establish which strategy was used in its original training run.

## Time, Memory, and Experiment Records

Both strategies use the same measurement scope. `history.json` preserves the
loss/accuracy curves and adds:

| Field | Meaning |
| --- | --- |
| `train_seconds` | Per-epoch training batch-loop wall time in seconds |
| `val_seconds` | Per-epoch validation wall time, recorded separately |
| `total_train_seconds` | Sum of all epoch training times; excludes validation |
| `total_val_seconds` | Sum of all epoch validation times |
| `train_peak_allocated_mib` | Per-epoch peak PyTorch tensor memory allocated on the selected GPU, in MiB |
| `peak_train_allocated_mib` | Maximum of the per-epoch training peaks |
| `best_epoch` | One-based epoch saved as the best validation Top-1 checkpoint; ties keep the earlier epoch |
| `best_val_accuracy` | Validation Top-1 accuracy at that epoch |

Training time uses `time.perf_counter()` around the batch loop, including data
loading, device transfers, forward/backward passes, and optimizer steps.
CUDA is synchronized before and after timing. Model setup, checkpoint I/O,
plots, final test evaluation, and Grad-CAM are excluded. The sum covers the
requested epoch count (10 by default), rather than total process runtime.

At each epoch, CUDA peak statistics are reset before the training batch loop.
The peak is read with `torch.cuda.max_memory_allocated()` immediately after
training, before validation. Values are divided by `1024 ** 2`. This measures
**peak allocated PyTorch tensor memory**, including resident model, gradient,
and optimizer tensors. It is not reserved allocator memory or total GPU usage
from `nvidia-smi`. CPU runs record memory values as JSON `null`.

Every invocation saves `run_config.json`, including the full command argument
list, parsed arguments, strategy, training seed, fixed split seed, data sources
and sample counts, transforms, resolved pretrained weight version/URL,
PyTorch/torchvision versions, GPU model, FP32 precision settings, total and
trainable parameter counts, checkpoint path, and run status. Training runs add
the best epoch and aggregate time/memory measurements **before** test evaluation
or Grad-CAM. The optimizer defaults are also recorded. Existing run configurations
are backed up as `run_config.backup-<UTC timestamp>.json` before a new invocation.
Evaluation/Grad-CAM-only runs leave training measurements and best epoch unset;
they cannot infer an old checkpoint's original training configuration.

`--seed` defaults to 37 and seeds Python, NumPy, PyTorch, and loader shuffling.
Workers seed Python/NumPy from their PyTorch worker seed. The split seed remains
37 when the training seed changes. Both strategies use FP32 without AMP or TF32;
exact bitwise reproducibility across different hardware/software is not implied.

## Sequential Comparison Experiments

To run the two complete 10-epoch experiments sequentially using the same Python
environment, data split, seed, initialization, augmentation, batch size,
optimizer, learning rate, precision, and visible GPU:

```powershell
python compare_strategies.py
```

This runs `--mode all` first with `frozen`, then with `full`, each in a separate
process. Results and terminal logs are saved to `outputs/frozen_seed37/` and
`outputs/full_seed37/`. After both succeed, `outputs/comparison_seed37.json` and
`outputs/comparison_seed37.csv` compare accuracy, training/validation times,
peak allocated PyTorch tensor memory, best epoch, and parameter counts. A failed
experiment stops the sequence. No sample limits are applied by default.

For a short workflow check only:

```powershell
python compare_strategies.py --epochs 1 --batch-size 4 --max-train-samples 20 --max-test-samples 10 --output-dir outputs/measurement_smoke
```

This comparison answers how the strategies differ under the **same settings and
10-epoch training budget**; it does not establish each strategy's best result
after separate hyperparameter tuning. Optional repeats with `--seed 38` and
`--seed 39` produce separate directories/summaries while retaining the same data
split. Use the same program version and environment for both strategies.

## Training, Validation, and Test Data

The official Food-101 training split is shuffled with seed 37 and divided into
90% training and 10% validation data. The official test split is kept separate.

| Subset | Images without sample limits | Purpose |
| --- | ---: | --- |
| Training | 68,175 | Update model parameters |
| Validation | 7,575 | Select the best checkpoint using validation Top-1 accuracy |
| Official test | 25,250 | Report final Top-1 and Top-5 accuracy for the selected checkpoint |

Validation uses no training augmentation. Test scores are never used to select
checkpoints. After training, the best validation checkpoint is reloaded before
test evaluation. `--mode train` trains and saves a checkpoint without running
the test evaluation; `--mode all` also evaluates it and generates Grad-CAM images.

`--max-train-samples` limits the official training pool **before** the 90/10
split: for example, 2,000 images gives 1,800 training and 200 validation images.
`--max-test-samples` independently limits the official test split for quick checks.
Results from such a subset should not be reported as full test-set results.

## Evaluation and Grad-CAM

```bash
python main.py --mode evaluate
python main.py --mode gradcam --gradcam-count 10
```

Results are saved in `outputs/`, including training curves, a confusion matrix, metrics in JSON format, Grad-CAM images, and the best model weights in `best_mobilenetv3_food101.pth`.

To re-evaluate an existing checkpoint on all 25,250 official test images:

```powershell
python main.py --mode evaluate --output-dir outputs 2>&1 | Tee-Object -FilePath outputs/evaluation-full-test.log
```

Run this in an environment with the project's PyTorch dependencies installed
(for example, after `conda activate pytorch`). Omit `--max-test-samples` for the
final evaluation. This command loads the existing checkpoint without retraining.
It saves the terminal output, including the sample count and final scores, to
`outputs/evaluation-full-test.log`.

`metrics.json` keeps `test_loss`, `top1_accuracy`, and `top5_accuracy` (accuracies
are fractions between 0 and 1). It also records the actual number of evaluated
test images (`test_samples`), the official test size (`official_test_samples`),
whether the full test split was evaluated (`is_full_test_set`), checkpoint path,
UTC evaluation time, command argument list, parsed arguments, and split settings.
The arguments and split settings describe the current invocation, not the
historical training configuration of a previously saved checkpoint.
Before replacing an existing `metrics.json`, the program copies it to
`metrics.backup-<UTC timestamp>.json` so earlier results remain available.
Older metrics files containing only scores cannot confirm whether sample limits
were used; re-evaluate the checkpoint to obtain a traceable result.

## Project Structure

- `model.py`: Defines the transfer learning model.
- `data.py`: Handles dataset loading and preprocessing.
- `engine.py`: Contains the training and evaluation logic.
- `visualize.py`: Generates charts and Grad-CAM visualizations.
- `main.py`: Provides the main command-line entry point.
- `compare_strategies.py`: Runs sequential comparison experiments and exports a summary.

## Tests

```bash
python -m unittest discover -s tests -v
```
