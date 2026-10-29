from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from data import denormalize


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.activations, self.gradients = model, None, None
        # Capture the feature maps produced by the target convolutional layer.
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        # Capture gradients of the target class score with respect to the feature maps.
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_index=None):
        self.model.eval()
        # Enable Grad-CAM backpropagation through the frozen convolutional layers.
        image = image.detach().requires_grad_(True)
        output = self.model(image)
        class_index = output.argmax(1).item() if class_index is None else class_index
        self.model.zero_grad()
        output[0, class_index].backward()
        # Compute an importance weight for each feature map.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # Generate the heatmap.
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # Upsample the heatmap to the input image size.
        heatmap = F.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        # Normalize the heatmap to [0, 1].
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp(min=1e-8)
        # Compute the confidence for the selected class.
        confidence = torch.softmax(output, dim=1)[0, class_index].item()
        return heatmap.cpu(), class_index, confidence

    def close(self):
        self.forward_hook.remove(); self.backward_hook.remove()

# Plot the training metrics in two panels and save the figure.
def plot_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    # Create a figure with two subplots.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Plot training and validation loss.
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
    # Plot validation accuracy.
    axes[1].plot(epochs, history["val_accuracy"], marker="o")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    # Adjust subplot spacing automatically.
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=160)
    plt.close(fig)

# Plot the classifier's confusion matrix as a heatmap and save the figure.
def plot_confusion_matrix(matrix, classes, output_dir):
    # Display the confusion matrix.
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    # Show a tick for every fifth class.
    ticks = np.arange(0, len(classes), 5)
    # Set the horizontal and vertical axis ticks.
    ax.set_xticks(ticks, [classes[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticks(ticks, [classes[i] for i in ticks], fontsize=6)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    # Add a color bar.
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(Path(output_dir) / "confusion_matrix.png", dpi=160)
    plt.close(fig)

# Select sample images and generate and save their Grad-CAM visualizations.
def save_gradcam_examples(model, loader, classes, device, output_dir, count=6):
    # Create a Grad-CAM instance.
    gradcam, saved = GradCAM(model, model.gradcam_layer), 0
    for images, labels in loader:
        for image, label in zip(images, labels):
            # Generate the Grad-CAM heatmap.
            heatmap, predicted, confidence = gradcam.generate(image.unsqueeze(0).to(device))
            # Undo normalization and rearrange [C, H, W] to [H, W, C].
            original = denormalize(image).permute(1, 2, 0).numpy()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            # Display the original image and its true class.
            axes[0].imshow(original); axes[0].set_title(f"Truth: {classes[label.item()]}")
            # Display the Grad-CAM heatmap.
            axes[1].imshow(heatmap.numpy(), cmap="jet"); axes[1].set_title("Grad-CAM")
            # Overlay the heatmap on the original image.
            axes[2].imshow(original); axes[2].imshow(heatmap.numpy(), cmap="jet", alpha=0.45)
            # Display the predicted class and its confidence.
            axes[2].set_title(f"Prediction: {classes[predicted]}\nConfidence: {confidence:.1%}")
            for ax in axes: ax.axis("off")
            fig.tight_layout(); fig.savefig(Path(output_dir) / f"gradcam_{saved + 1}.png", dpi=160); plt.close(fig)
            saved += 1
            if saved >= count:
                gradcam.close(); return
    gradcam.close()
