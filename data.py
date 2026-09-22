import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101

# 为了适配MobileNetV3-small model需要的参数
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 使用训练集的时候数据增强
def build_transforms(training):
    # 将原图尺寸改为统一的224 x 224尺寸
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), # 随机水平翻转
                  transforms.RandomRotation(10)] # 随机旋转-10°到10°
    steps += [transforms.ToTensor(), # 转换为tensor类型
              transforms.Normalize(MEAN, STD)] # 标准化
    return transforms.Compose(steps)

# 在数据集中随机选择指定数量的样本测试
def _limited_indices(length, maximum, seed):
    # 没给最大数量或最大值大于总样本数量 -> 样本总量
    if maximum is None or maximum >= length:
        return list(range(length))
    # 创建随机数生成器并接收一个固定种子
    generator = torch.Generator().manual_seed(seed)
    # 随机排列并选择前maximum个值
    return torch.randperm(length, generator=generator)[:maximum].tolist()

# 下载并划分数据集
def create_loaders(data_dir, batch_size, workers=0, max_train=None, max_test=None):
    # 增强版数据集，用于训练
    train_augmented = Food101(data_dir, split="train", download=True, transform=build_transforms(True))
    # 普通版数据集，用于验证
    train_plain = Food101(data_dir, split="train", download=False, transform=build_transforms(False))
    # 测试集
    test_data = Food101(data_dir, split="test", download=True, transform=build_transforms(False))
    indices = _limited_indices(len(train_augmented), max_train, seed=37)
    if len(indices) < 2:
        raise ValueError("Training requires at least 2 images.")
    # 生成随机数并打乱索引
    generator = torch.Generator().manual_seed(37)
    shuffled = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
    # 随机取90%数据作为训练集，10%数据作为验证集
    val_size = max(1, int(len(shuffled) * 0.1))
    train_set = Subset(train_augmented, shuffled[val_size:])
    val_set = Subset(train_plain, shuffled[:val_size])
    # 指定数量随机选取图片作为测试集
    test_set = Subset(test_data, _limited_indices(len(test_data), max_test, seed=37))
    # 小优化：用dict函数封装，减少重复代码
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_set, shuffle=True, **options),
            DataLoader(val_set, shuffle=False, **options),
            DataLoader(test_set, shuffle=False, **options),
            train_augmented.classes) # train_augmented.classes保存了类别名称与label编号的对应表

# 反标准化：方便用matplotlib显示
def denormalize(image):
    # 将均值转成tensor类型并reshape成和图片转成的tensor的shape一样，便于广播
    mean = torch.tensor(MEAN).reshape(3, 1, 1)
    std = torch.tensor(STD).reshape(3, 1, 1)
    # 把图像张量从 GPU 显存移动到 CPU 内存，并将结果限制在【0,1】区间内
    return (image.cpu() * std + mean).clamp(0, 1)