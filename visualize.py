from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from data import denormalize


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.activations, self.gradients = model, None, None
        # 记录卷积层输出的特征图
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        # 目标类别对特征图的梯度
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_index=None):
        self.model.eval()
        # 让 Grad-CAM 的反向传播仍能经过已被冻结的卷积层
        image = image.detach().requires_grad_(True)
        output = self.model(image)
        class_index = output.argmax(1).item() if class_index is None else class_index
        self.model.zero_grad()
        output[0, class_index].backward()
        # 计算每张特征图的重要性
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 生成热力图
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # 放大热力图
        heatmap = F.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        # 把热力图调整到0-1之间
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp(min=1e-8)
        # 计算置信度
        confidence = torch.softmax(output, dim=1)[0, class_index].item()
        return heatmap.cpu(), class_index, confidence

    def close(self):
        self.forward_hook.remove(); self.backward_hook.remove()

# 把训练过程中记录的指标画成两张曲线图，并保存为图片
def plot_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    # 创建画布和两个子图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # 训练损失和验证损失
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
    # 验证准确率
    axes[1].plot(epochs, history["val_accuracy"], marker="o")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    # 自动调整布局
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=160)
    plt.close(fig)

# 把分类模型的混淆矩阵画成热力图，并保存为图片
def plot_confusion_matrix(matrix, classes, output_dir):
    # 显示混淆矩阵
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    # 隔五个类别显示一个刻度
    ticks = np.arange(0, len(classes), 5)
    # 设置横纵轴刻度
    ax.set_xticks(ticks, [classes[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticks(ticks, [classes[i] for i in ticks], fontsize=6)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    # 添加颜色条
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(Path(output_dir) / "confusion_matrix.png", dpi=160)
    plt.close(fig)

# 从数据集中选取若干张图片，生成并保存 Grad-CAM 可视化结果
def save_gradcam_examples(model, loader, classes, device, output_dir, count=6):
    # 创建 Grad-CAM 对象
    gradcam, saved = GradCAM(model, model.gradcam_layer), 0
    for images, labels in loader:
        for image, label in zip(images, labels):
            # 生成 Grad-CAM
            heatmap, predicted, confidence = gradcam.generate(image.unsqueeze(0).to(device))
            # 反归一化 & [C, H, W] → [H, W, C]
            original = denormalize(image).permute(1, 2, 0).numpy()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            # 显示原图和真实类别
            axes[0].imshow(original); axes[0].set_title(f"Truth: {classes[label.item()]}")
            # 显示 Grad-CAM 热力图
            axes[1].imshow(heatmap.numpy(), cmap="jet"); axes[1].set_title("Grad-CAM")
            # 叠加原图与热力图
            axes[2].imshow(original); axes[2].imshow(heatmap.numpy(), cmap="jet", alpha=0.45)
            # 显示预测类别和置信度
            axes[2].set_title(f"Prediction: {classes[predicted]}\nConfidence: {confidence:.1%}")
            for ax in axes: ax.axis("off")
            fig.tight_layout(); fig.savefig(Path(output_dir) / f"gradcam_{saved + 1}.png", dpi=160); plt.close(fig)
            saved += 1
            if saved >= count:
                gradcam.close(); return
    gradcam.close()
