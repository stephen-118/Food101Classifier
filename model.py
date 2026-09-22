import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# MobileNetV3-Small transfer-learning model for Food-101
class FoodTransferModel(nn.Module):

    def __init__(self, num_classes=101, use_pretrained=True): # 默认使用训练参数
        super().__init__()
        # 加载预训练 MobileNet，学习视觉知识
        if use_pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
        self.network = mobilenet_v3_small(weights=weights)

        # 保留 ImageNet 学到的视觉知识，只训练新的分类器
        for parameter in self.network.features.parameters():
            # 不计算梯度也不更新参数
            parameter.requires_grad = False

        # 获取该层输入的特征数量
        input_features = self.network.classifier[3].in_features
        # 将第三层的输出改为现在dataset的101个类别
        self.network.classifier[3] = nn.Linear(input_features, num_classes)

    @property # 可以像访问变量一样使用它
    # 生成热力图的最后一个卷积特征层
    def gradcam_layer(self):
        return self.network.features[-1]

    def forward(self, x):
        return self.network(x)

    # 训练分类器时，让冻结的预训练特征层继续保持稳定
    def train(self, mode=True):
        super().train(mode)
        # BatchNorm不再更新统计值，使用训练期间保存的均值和方差
        self.network.features.eval()
        return self
