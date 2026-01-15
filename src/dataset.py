"""
Data loading and preprocessing utilities for image classification
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class ImageClassificationDataset(Dataset):
    """Custom Dataset for image classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform: Transformations to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = self.transform(image)
        
        return image, label


def get_transforms(img_height=224, img_width=224, augment=False):
    """
    Get image transformations for training/validation
    
    Args:
        img_height (int): Target image height
        img_width (int): Target image width
        augment (bool): Whether to apply data augmentation
    
    Returns:
        Albumentations compose object with transforms
    """
    if augment:
        transform = A.Compose([
            A.Resize(img_height, img_width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_height, img_width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def get_torchvision_transforms(img_height=224, img_width=224, augment=False):
    """
    Alternative: Get torchvision transformations
    
    Args:
        img_height (int): Target image height
        img_width (int): Target image width
        augment (bool): Whether to apply data augmentation
    
    Returns:
        Torchvision compose object with transforms
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_data_from_folders(data_dir, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    """
    Load data from folder structure: data_dir/class_name/image.jpg
    
    Args:
        data_dir (str): Path to data directory
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names)
    """
    import random
    random.seed(seed)
    
    all_paths = []
    all_labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Collect all image paths and labels
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_paths.append(img_path)
                all_labels.append(class_to_idx[class_name])
    
    # Shuffle data
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    # Split data
    n_samples = len(all_paths)
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)
    
    train_paths = list(all_paths[:train_end])
    train_labels = list(all_labels[:train_end])
    
    val_paths = list(all_paths[train_end:val_end])
    val_labels = list(all_labels[train_end:val_end])
    
    test_paths = list(all_paths[val_end:])
    test_labels = list(all_labels[val_end:])
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names
