import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

PRETRAINED_WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT

# MobileNetV3-Small transfer-learning model for Food-101
class FoodTransferModel(nn.Module):

    def __init__(self, num_classes=101, use_pretrained=True, freeze_backbone=True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        # Load MobileNet with pretrained visual features.
        if use_pretrained:
            weights = PRETRAINED_WEIGHTS
        else:
            weights = None
        self.network = mobilenet_v3_small(weights=weights)

        # Freeze the feature extractor or fine-tune it along with the classification head.
        for parameter in self.network.features.parameters():
            parameter.requires_grad = not freeze_backbone

        # Get the number of input features for the output layer.
        input_features = self.network.classifier[3].in_features
        # Replace the output layer with one for the dataset's 101 classes.
        self.network.classifier[3] = nn.Linear(input_features, num_classes)

    @property # Allow access as an attribute.
    # Expose the final convolutional feature block for heatmap generation.
    def gradcam_layer(self):
        return self.network.features[-1]

    def forward(self, x):
        return self.network(x)

    # Keep the frozen pretrained feature extractor stable while training the classifier.
    def train(self, mode=True):
        super().train(mode)
        # Frozen features keep fixed BatchNorm statistics; full fine-tuning updates them.
        if self.freeze_backbone:
            self.network.features.eval()
        return self
