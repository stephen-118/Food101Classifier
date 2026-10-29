import torch

# Evaluate the model.
def evaluate(model, loader, criterion, device, num_classes=101):
    """Compute metrics for the supplied validation or test loader without updating weights."""
    model.eval()
    total_loss = correct1 = correct5 = total = 0
    # Create a confusion matrix: rows are true classes, columns are predicted classes.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    # Disable gradient computation.
    with torch.no_grad():
        for images, labels in loader:
            # Move the batch to the selected device.
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Accumulate the loss weighted by batch size.
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            # Find the five highest-scoring classes for each image.
            top5 = outputs.topk(5, dim=1).indices
            # Count correct Top-1 predictions.
            correct1 += (top5[:, 0] == labels).sum().item()
            # Count labels that appear among the Top-5 predictions.
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
            # Update the confusion matrix after moving labels and predictions to CPU.
            for true_label, predicted_label in zip(labels.cpu(), top5[:, 0].cpu()):
                confusion[true_label, predicted_label] += 1
    if total == 0:
        raise ValueError("Evaluation requires at least one image.")
    return total_loss / total, correct1 / total, correct5 / total, confusion

# Run the training loop.
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_path):
    """Update weights on train data and select the checkpoint by validation Top-1 only.

    The official test loader is deliberately not part of the training interface.
    """
    # Track the training history.
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_accuracy = -1.0
    # Each epoch makes one complete pass through the training set.
    for epoch in range(epochs):
        model.train()
        loss_sum = sample_count = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            # Clear gradients from the previous batch.
            optimizer.zero_grad()
            # Run the forward pass and compute the loss.
            loss = criterion(model(images), labels)
            # Backpropagate the loss.
            loss.backward()
            # Update the trainable parameters.
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)
            if step % 100 == 0:
                print(f"  batch {step}/{len(train_loader)}, loss={loss.item():.4f}")
        train_loss = loss_sum / sample_count
        # Evaluate model performance on the validation set.
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2%}")
        # Select checkpoints using validation Top-1, never official test scores.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
    # Reload the best model weights after training.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history
