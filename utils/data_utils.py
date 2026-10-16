"""
Data utilities for building datasets and data loaders.
Provides functions for CIFAR-10, CIFAR-100, and ImageNet datasets.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from typing import Tuple, Optional, List


def build_cifar100_data(data_path: str = 'data/CIFAR100', input_size: int = 32, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build CIFAR-100 data loaders"""
    print('==> Using CIFAR-100 Dataset')

    # CIFAR-100 normalization values
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load CIFAR-100 datasets
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=val_transform
    )

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader


def build_cifar10_data(data_path: str = 'data/CIFAR10', input_size: int = 32, batch_size: int = 64, workers: int = 4,
                       dist_sample: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build CIFAR-10 data loaders"""
    print('==> Using CIFAR-10 Dataset')

    # CIFAR-10 normalization values
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=val_transform
    )

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader


def build_imagenet_data(data_path: str = '/local/a/imagenet/imagenet2012/', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build ImageNet data loaders"""
    print('==> Using ImageNet Dataset')

    # ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Load ImageNet datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'val'),
        transform=val_transform
    )

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Custom TinyImageNet dataset that handles the special validation set structure"""
    
    def __init__(self, root, wnid_to_idx, transform=None):
        self.root = root
        self.transform = transform
        
        # Use the class mapping from train dataset
        self.class_to_idx = wnid_to_idx
        self.classes = list(wnid_to_idx.keys())
        
        # For validation, read from val_annotations.txt
        self.samples = []
        val_images_dir = os.path.join(root, 'images')
        val_annotations_file = os.path.join(root, 'val_annotations.txt')
        
        if not os.path.exists(val_annotations_file):
            raise FileNotFoundError(f"Validation annotations file not found: {val_annotations_file}")
        
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    if class_name in self.class_to_idx:
                        img_path = os.path.join(val_images_dir, img_name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, self.class_to_idx[class_name]))

        assert len(self.samples) >= 10000, f"Unexpected #val samples: {len(self.samples)}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def build_tinyimagenet_data(data_path: str = 'data/TinyImageNet', input_size: int = 64, batch_size: int = 64, workers: int = 4,
                            dist_sample: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build TinyImageNet data loaders with train/val/test splits"""
    print('==> Using TinyImageNet Dataset')

    # TinyImageNet normalization values (using ImageNet values as standard)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation/Test transforms
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Load TinyImageNet datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=train_transform
    )

    wnid_to_idx = train_dataset.class_to_idx

    val_dataset = TinyImageNetDataset(
        root=os.path.join(data_path, 'val'),
        wnid_to_idx=wnid_to_idx,
        transform=val_transform
    )

    # Using  val as test since test had no labels
    test_dataset = val_dataset

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader 