import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# MobileNetV3-Small transfer-learning model for Food-101
class FoodTransferModel(nn.Module):

    def __init__(self, num_classes=101, use_pretrained=True): # Use pretrained weights by default.
        super().__init__()
        # Load MobileNet with pretrained visual features.
        if use_pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
        self.network = mobilenet_v3_small(weights=weights)

        # Preserve the ImageNet feature extractor and train only the classification head.
        for parameter in self.network.features.parameters():
            # Disable gradient computation and parameter updates.
            parameter.requires_grad = False

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
        # Keep BatchNorm running statistics fixed and use the stored mean and variance.
        self.network.features.eval()
        return self
