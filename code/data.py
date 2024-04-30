from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch

def get_dataloaders(transformations, batch_size=32, root_dir='./images'):
    dataset = datasets.ImageFolder(root=root_dir, transform=transformations)

    # Calculate split sizes for 80% train, 10% validation, and 10% test
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Count the number of images per class in the training dataset
    class_counts = {}
    for idx in train_dataset.indices:
        label = dataset.targets[idx]
        class_counts[label] = class_counts.get(label, 0) + 1

    # Calculate weights for each class
    weights = [1.0 / class_counts[dataset.targets[idx]] for idx in train_dataset.indices]
    sample_weights = torch.DoubleTensor(weights)

    # Create a WeightedRandomSampler for the training dataset
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main_transformations():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
