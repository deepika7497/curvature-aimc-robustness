"""
Generic data loader utility for multiple datasets.
Supports ImageNet, Tiny ImageNet, CIFAR-10/100, and MNIST with indexing capabilities.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Dict, List, Optional, Tuple, Union
import logging


class IndexedDataset(Dataset):
    """Dataset wrapper that returns indices along with data."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        data, target = self.dataset[index]
        return index, data, target
    
    def __len__(self) -> int:
        return len(self.dataset)


class DictToObj:
    """Convert dictionary to object for easy attribute access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_transform(
    test_transform,
    train_transform,
    val_transform,
    mean,
    std,
    augment,
    img_dim,
    padding_crop,
    resize=False):

    if(test_transform == None):
        test_transform = transforms.Compose([
                                                transforms.Resize((img_dim, img_dim)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std),
                                            ])

    if(val_transform == None):    
        val_transform = test_transform

    if(train_transform == None):
        if augment:
            transforms_list = [
                transforms.RandomCrop(img_dim, padding=padding_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if resize:
                transforms_list = [transforms.Resize((img_dim, img_dim))] + transforms_list
                train_transform = transforms.Compose(transforms_list)
            else:
                train_transform = transforms.Compose(transforms_list)
        else:
            train_transform = test_transform

    return train_transform, val_transform, test_transform


def load_dataset(
    dataset='CIFAR10',
    train_batch_size=128,
    test_batch_size=128,
    val_split=0.1,
    augment=True,
    padding_crop=4,
    shuffle=True,
    random_seed=None,
    resize_shape=None,
    mean=None,
    std=None,
    train_transform=None,
    test_transform=None,
    val_transform=None,
    index=None,
    logger=None,
    root_path='data',
    workers=None,
    pin_memory=True,
    indexed=False):
    """
    Loads and prepares a dataset for training, validation, and testing.

    Parameters:
    dataset (str): Name of the dataset. Options include 'MNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet'.
    train_batch_size (int): Batch size for the training dataset.
    test_batch_size (int): Batch size for the testing dataset.
    val_split (float): Percentage of the training data to be used as validation dataset.
    augment (bool): Flag to apply random horizontal flip and padding shift for data augmentation.
    padding_crop (int): Number of pixels to pad the image for cropping.
    shuffle (bool): Flag to shuffle the training and testing dataset.
    random_seed (int, optional): Seed for shuffling to reproduce results.
    resize_shape (tuple, optional): Shape to resize the images, if required.
    mean (list, optional): Mean values for normalization.
    std (list, optional): Standard deviation values for normalization.
    train_transform (torchvision.transforms.Compose, optional): Custom transformations for training data.
    test_transform (torchvision.transforms.Compose, optional): Custom transformations for testing data.
    val_transform (torchvision.transforms.Compose, optional): Custom transformations for validation data.
    index (list, optional): Indices for the dataset.
    logger (logging.Logger, optional): Logger for logging information.
    root_path (str): Root path for the datasets.
    workers (int, optional): Number of worker threads for data loading.
    pin_memory (bool): Whether to pin memory for faster GPU transfer.
    indexed (bool): Whether to return indexed datasets (returns indices with data).

    Returns:
    dataset_obj (DictToObj): An object containing dataset loaders and relevant dataset information.

    Raises:
    ValueError: If the specified dataset is not supported.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Load dataset
    if (dataset.lower() == 'mnist'):
        if(mean == None):
            mean = [0.1307]
        if(std == None):
            std = [0.3081]
        img_dim = 28
        img_ch = 1
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'MNIST')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=test_transform)
        
    elif(dataset.lower() == 'cifar10'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR10')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=test_transform)
    
    elif(dataset.lower() == 'cifar100'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR100')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    elif(dataset.lower() == 'tinyimagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]
        root = os.path.join(root_path, 'TinyImageNet')
        img_dim = 64
        resize = False
        if(resize_shape == None):
            resize_shape = (32, 32)
            img_dim = 32
        else:
            img_dim = resize_shape[0]
            resize = True
        img_ch = 3
        num_classes = 200
        num_worker = 4
        
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=resize)

        # For TinyImageNet, we need to handle the special directory structure
        train_data_path = os.path.join(root, 'train')
        val_data_path = os.path.join(root, 'val')
        
        # Check if TinyImageNet is downloaded
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(
                f"TinyImageNet dataset not found at {train_data_path}. "
                f"Please download it using: python utils/download_tiny_imagenet.py"
            )
        
        trainset = torchvision.datasets.ImageFolder(
            root=train_data_path,
            transform=train_transform)
        
        valset = torchvision.datasets.ImageFolder(
            root=train_data_path,
            transform=val_transform)
        
        testset = torchvision.datasets.ImageFolder(
            root=val_data_path,
            transform=test_transform)
  
    elif(dataset.lower() == 'imagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]

        img_dim = 224
        img_ch = 3
        num_classes = 1000
        num_worker = 8
        # Use the absolute path for ImageNet as specified in memory
        datapath = '/local/a/imagenet/imagenet2012/'

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'train'),
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'train'),
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'val'),
            transform=test_transform)

    else:
        logger.error("Unsupported dataset")
        raise ValueError("Unsupported dataset")
    
    # Split the training dataset into training and validation sets
    logger.info('Forming the sampler for train and validation split')
    if index is None:
        num_train = len(trainset)
        ind = list(range(num_train))
    else:
        num_train = len(index)
        ind = index
    
    split = int(val_split * num_train)
    logger.info(f'Split counts 0 {split} {num_train}')

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(ind)

    train_idx, val_idx = ind[split:], ind[:split]
    valset = torch.utils.data.Subset(trainset, val_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)
    
    # Create indexed datasets if requested
    if indexed:
        trainset = IndexedDataset(trainset)
        valset = IndexedDataset(valset)
        testset = IndexedDataset(testset)
    
    num_worker = num_worker if workers is None else workers
    
    # Load dataloader
    logger.info('Loading data to the dataloader')
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_worker)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_worker)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_worker)

    transforms_dict = {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform
    }

    return_dict = {
        'name': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': trainset,
        'val_dataset': valset,
        'test_dataset': testset,
        'num_classes': num_classes,
        'train_length': num_train,
        'testset': testset,
        'mean': mean,
        'std': std,
        'img_dim': img_dim,
        'img_ch': img_ch,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'val_split': val_split,
        'padding_crop': padding_crop,
        'augment': augment,
        'random_seed': random_seed,
                 'shuffle': shuffle,
         'indexed': indexed,
         'transforms': DictToObj(**transforms_dict),
         'num_worker': num_worker,
         'metadata': {
            'mean': mean,
            'std': std,
            'padding_crop': padding_crop,
            'num_workers': num_worker
        }
    }

    dataset_obj = DictToObj(**return_dict)
    logger.info(f"Dataset loaded successfully: {len(trainset)} train, {len(valset)} val, {len(testset)} test")
    return dataset_obj


def create_subset_dataloader(
    full_loader: DataLoader,
    indices: List[int],
    batch_size: Optional[int] = None,
    shuffle: bool = True
) -> DataLoader:
    """Create a data loader for a subset of the data."""
    
    if batch_size is None:
        batch_size = full_loader.batch_size
    
    # Get the full dataset
    full_dataset = full_loader.dataset
    
    # Create subset
    subset_dataset = Subset(full_dataset, indices)
    
    # Create data loader
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=full_loader.num_workers,
        pin_memory=full_loader.pin_memory
    )
    
    return subset_loader