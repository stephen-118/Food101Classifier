import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101

# Input size and normalization settings for MobileNetV3-Small.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 37
VALIDATION_FRACTION = 0.1

# Apply data augmentation to training images.
def build_transforms(training):
    # Resize all images to 224 x 224 pixels.
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), # Randomly flip images horizontally.
                  transforms.RandomRotation(10)] # Randomly rotate between -10 and 10 degrees.
    steps += [transforms.ToTensor(), # Convert images to tensors.
              transforms.Normalize(MEAN, STD)] # Normalize each channel.
    return transforms.Compose(steps)

# Randomly select up to the requested number of sample indices.
def _limited_indices(length, maximum, seed):
    if maximum is not None and maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    # Use all samples if no limit is set or the limit covers the entire dataset.
    if maximum is None or maximum >= length:
        return list(range(length))
    # Create a random number generator with a fixed seed.
    generator = torch.Generator().manual_seed(seed)
    # Shuffle the indices and select the first maximum entries.
    return torch.randperm(length, generator=generator)[:maximum].tolist()

# Download the dataset and create training, validation, and test splits.
def create_loaders(data_dir, batch_size, workers=0, max_train=None, max_test=None):
    """Split official training data 90/10; reserve official test data for final reporting.

    max_train limits the pool BEFORE splitting it into train/validation subsets.
    Validation Top-1 selects the checkpoint; test scores never select checkpoints.
    """
    # Use augmented images for training.
    train_augmented = Food101(data_dir, split="train", download=True, transform=build_transforms(True))
    # Use images without augmentation for validation.
    train_plain = Food101(data_dir, split="train", download=False, transform=build_transforms(False))
    # Load the official test split.
    test_data = Food101(data_dir, split="test", download=True, transform=build_transforms(False))
    indices = _limited_indices(len(train_augmented), max_train, seed=SPLIT_SEED)
    if len(indices) < 2:
        raise ValueError("Training requires at least 2 images.")
    # Shuffle the indices using a fixed random seed.
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    shuffled = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
    # Reserve 10% of the selected samples for validation and use the rest for training.
    val_size = max(1, int(len(shuffled) * VALIDATION_FRACTION))
    train_set = Subset(train_augmented, shuffled[val_size:])
    val_set = Subset(train_plain, shuffled[:val_size])
    # Randomly limit the test split to the requested number of images.
    test_set = Subset(test_data, _limited_indices(len(test_data), max_test, seed=SPLIT_SEED))
    # Share DataLoader options to avoid repeating the same arguments.
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_set, shuffle=True, **options),
            DataLoader(val_set, shuffle=False, **options),
            DataLoader(test_set, shuffle=False, **options),
            train_augmented.classes) # Class names are ordered by their numeric labels.

# Undo normalization for display with matplotlib.
def denormalize(image):
    # Reshape the channel statistics to [C, 1, 1] for broadcasting over the image.
    mean = torch.tensor(MEAN).reshape(3, 1, 1)
    std = torch.tensor(STD).reshape(3, 1, 1)
    # Move the image tensor to CPU, undo normalization, and clamp values to [0, 1].
    return (image.cpu() * std + mean).clamp(0, 1)
