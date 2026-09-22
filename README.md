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

Only the new classification layer is trained, so this setup typically requires far fewer epochs than training a CNN from scratch.

## Evaluation and Grad-CAM

```bash
python main.py --mode evaluate
python main.py --mode gradcam --gradcam-count 10
```

Results are saved in `outputs/`, including training curves, a confusion matrix, metrics in JSON format, Grad-CAM images, and the best model weights in `best_mobilenetv3_food101.pth`.

## Project Structure

- `model.py`: Defines the transfer learning model.
- `data.py`: Handles dataset loading and preprocessing.
- `engine.py`: Contains the training and evaluation logic.
- `visualize.py`: Generates charts and Grad-CAM visualizations.
- `main.py`: Provides the main command-line entry point.
