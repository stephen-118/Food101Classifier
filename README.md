# Food-101 Transfer Learning Classifier

```text
ImageNet-pretrained MobileNetV3-Small
              ↓
  Freeze the convolutional feature extractor
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

The convolutional feature extractor is frozen, while the entire classification head is trained.

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
