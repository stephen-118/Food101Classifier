import torch

# 评估函数
def evaluate(model, loader, criterion, device, num_classes=101):
    model.eval()
    total_loss = correct1 = correct5 = total = 0
    # 创建混淆矩阵，行 -> 真实类别；列 -> 预测类别
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    # 关闭梯度
    with torch.no_grad():
        for images, labels in loader:
            # 把数据移动到GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 统计损失
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            # 找出每张图片得分最高的5个类别
            top5 = outputs.topk(5, dim=1).indices
            # top1准确率
            correct1 += (top5[:, 0] == labels).sum().item()
            # top5准确率
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
            # 更新混淆矩阵 （把真实标签和预测标签移回 CPU）
            for true_label, predicted_label in zip(labels.cpu(), top5[:, 0].cpu()):
                confusion[true_label, predicted_label] += 1
    return total_loss / total, correct1 / total, correct5 / total, confusion

# 训练循环
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_path):
    # 保存训练历史
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_accuracy = -1.0
    # 一个 epoch 表示模型看完一次完整训练集
    for epoch in range(epochs):
        model.train()
        loss_sum = sample_count = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            # 清空上一批梯度
            optimizer.zero_grad()
            # 前向传播并计算损失
            loss = criterion(model(images), labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)
            if step % 100 == 0:
                print(f"  batch {step}/{len(train_loader)}, loss={loss.item():.4f}")
        train_loss = loss_sum / sample_count
        # 用验证集检查模型表现
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2%}")
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
    # 训练结束后重新加载最佳参数
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history
