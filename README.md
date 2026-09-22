# Food-101 迁移学习分类项目

```text
ImageNet 预训练 MobileNetV3-Small
              ↓
       冻结卷积特征提取层
              ↓
将 1000 类输出层替换为 101 类
              ↓
        在 Food-101 上训练
```

## 快速试跑

```bash
python main.py --mode all --epochs 2 --max-train-samples 2000 --max-test-samples 500
```

第一次运行会自动下载 Food-101 和 ImageNet 预训练参数。少量数据只适合检查程序。

## 完整训练

```bash
python main.py --mode all --epochs 10 --batch-size 32
```

这里只训练新的分类层，所以通常不需要像从零训练 CNN 那样运行很多轮。

## 单独评估或生成 Grad-CAM

```bash
python main.py --mode evaluate
python main.py --mode gradcam --gradcam-count 10
```

结果保存在 `outputs/`：训练曲线、混淆矩阵、指标 JSON、Grad-CAM 图片和
`best_mobilenetv3_food101.pth` 最佳模型参数。

文件分工：`model.py` 是迁移学习模型，`data.py` 是数据，`engine.py` 是训练评估，
`visualize.py` 是图表和 Grad-CAM，`main.py` 是运行入口。
