import time

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
    if epochs < 1:
        raise ValueError("Training requires at least one epoch.")
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "train_seconds": [], "val_seconds": [], "train_peak_allocated_mib": [],
               "best_epoch": None, "best_val_accuracy": None}
    best_accuracy = -1.0
    # Each epoch makes one complete pass through the training set.
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = sample_count = 0
        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_start = time.perf_counter()
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
        if use_cuda:
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start
        # Read the training-only peak before validation, test evaluation, or Grad-CAM.
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_cuda else None
        if sample_count == 0:
            raise ValueError("Training requires at least one image.")
        train_loss = loss_sum / sample_count
        # Evaluate model performance on the validation set.
        if use_cuda:
            torch.cuda.synchronize(device)
        val_start = time.perf_counter()
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        if use_cuda:
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - val_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["train_peak_allocated_mib"].append(peak_mib)
        memory_text = f"{peak_mib:.2f} MiB" if peak_mib is not None else "N/A (CPU)"
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2%}, "
              f"train={train_seconds:.2f}s, validation={val_seconds:.2f}s, "
              f"train_peak_allocated={memory_text}")
        # Select checkpoints using validation Top-1, never official test scores.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            history["best_epoch"] = epoch + 1
            history["best_val_accuracy"] = val_accuracy
    history["total_train_seconds"] = sum(history["train_seconds"])
    history["total_val_seconds"] = sum(history["val_seconds"])
    peaks = [value for value in history["train_peak_allocated_mib"] if value is not None]
    history["peak_train_allocated_mib"] = max(peaks) if peaks else None
    print(f"Training complete: best_epoch={history['best_epoch']}, "
          f"training={history['total_train_seconds']:.2f}s, "
          f"validation={history['total_val_seconds']:.2f}s")
    # Reload the best model weights after training.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history
