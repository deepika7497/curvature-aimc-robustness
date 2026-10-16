#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute curvature for training and validation datasets.

Given a dataset and model name, compute curvature for each training and validation sample
and save as dictionaries with sample index as key.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.data_utils import build_imagenet_data, build_cifar10_data, build_cifar100_data, build_tinyimagenet_data
from utils.model_utils import create_cifar10_model, create_cifar100_model, create_tinyimagenet_model, load_checkpoint
from utils.curv_utils import compute_curvature_for_dataset
from utils.training_utils import validate
import torchvision.models as models

def get_checkpoint_path(model_name, dataset_name, seed=42):
    """Get checkpoint path for a given model and dataset."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', dataset_name.lower())
    pattern = f'{model_name}_seed{seed}_*'
    saved_runs = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(saved_runs) == 0:
        raise FileNotFoundError(f"No saved runs found for {model_name} on {dataset_name} with seed {seed}")
    else:
        # Sort by directory name to get the latest run (assumes timestamp in name)
        saved_runs.sort()
        latest_run = saved_runs[-1]
        # Find a checkpoint file in latest_run, prefer *best.pth or last.pth, fallback to any .pth
        checkpoint_files = []
        for filename in os.listdir(latest_run):
            if filename.endswith('.pth'):
                checkpoint_files.append(filename)
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint (.pth) files found in latest run directory: {latest_run}")
        best_checkpoint = next((f for f in checkpoint_files if 'best' in f), None)
        if best_checkpoint is None:
            raise FileNotFoundError(f"No best checkpoint found in latest run directory: {latest_run}")
        checkpoint_path = os.path.join(latest_run, best_checkpoint)
        return checkpoint_path


def load_model(model_name, dataset_name, checkpoint_path=None, seed=42, device='cpu'):
    """Load model with automatic or manual checkpoint loading.
    
    Args:
        model_name: Name of the model architecture (e.g., 'resnet18', 'mobilenetv2')
        dataset_name: Name of the dataset ('imagenet', 'cifar10', 'cifar100', 'tinyimagenet')
        checkpoint_path: Optional explicit path to checkpoint file. If None, will auto-find for non-ImageNet datasets.
        seed: Random seed used during training (default: 42, used for auto-finding checkpoints)
        device: Device to load checkpoint on
        
    Returns:
        model: Loaded model
    """
    # Create model based on dataset
    if dataset_name == 'imagenet':
        # Load directly from torchvision
        print(f"Loading pretrained ImageNet weights from torchvision for {model_name}...")
        
        # Map model names to torchvision functions
        if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            model_fn = getattr(models, model_name)
            try:
                weights = 'IMAGENET1K_V1'
                model = model_fn(weights=weights)
            except TypeError:
                # Fallback for older torchvision versions
                model = model_fn(pretrained=True)
        elif model_name == 'mobilenetv2':
            try:
                model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            except TypeError:
                model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f"Unsupported ImageNet model: {model_name}. "
                           f"Supported: resnet18, resnet34, resnet50, resnet101, resnet152, mobilenetv2")
        return model
    
    # For non-ImageNet datasets, create model and load checkpoint
    elif dataset_name == 'cifar10':
        model = create_cifar10_model(model_name)
    elif dataset_name == 'cifar100':
        model = create_cifar100_model(model_name)
    elif dataset_name == 'tinyimagenet':
        model = create_tinyimagenet_model(model_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Auto-find checkpoint if not provided
    if checkpoint_path is None:
        print(f"No checkpoint provided. Auto-finding checkpoint for {model_name} on {dataset_name} with seed {seed}...")
        try:
            checkpoint_path = get_checkpoint_path(model_name, dataset_name, seed=seed)
            print(f"Found checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = load_checkpoint(model, checkpoint_path, device=device)
        if checkpoint:
            print(f"Checkpoint loaded successfully!")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'best_acc' in checkpoint:
                print(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
    elif checkpoint_path:
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    return model


def compute_curvature(model_name, dataset_name, checkpoint_path=None, batch_size=64, 
                     workers=4, device='cuda', save_dir=None, seed=42, debug=False):
    """Compute curvature for both training and validation datasets.
    
    Args:
        model_name: Name of the model architecture
        dataset_name: Name of dataset ('imagenet', 'cifar10', 'cifar100', 'tinyimagenet')
        checkpoint_path: Optional path to checkpoint file. If None, will auto-find for non-ImageNet datasets.
        batch_size: Batch size for computation
        workers: Number of data loading workers
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save curvature results
        seed: Random seed used during training (default: 42, used for auto-finding checkpoints)
        debug: Enable debug output
    """
    
    print(f"\n{'='*80}")
    print(f"Computing curvature for {dataset_name} with {model_name}")
    print(f"{'='*80}\n")
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if curvature files already exist
    train_curvature_file = os.path.join(save_dir, f'{model_name}_{dataset_name}_train_curvature.npy')
    val_curvature_file = os.path.join(save_dir, f'{model_name}_{dataset_name}_val_curvature.npy')
    
    train_exists = os.path.exists(train_curvature_file)
    val_exists = os.path.exists(val_curvature_file)
    
    train_curvature_dict = None
    val_curvature_dict = None
    
    if train_exists and val_exists:
        print(f"\n{'='*80}")
        print(f"BOTH CURVATURE FILES ALREADY EXIST")
        print(f"{'='*80}")
        print(f"Train file: {train_curvature_file}")
        print(f"Val file: {val_curvature_file}")
        print(f"Loading existing curvature data...")
        print(f"{'='*80}\n")
        
        train_curvature_dict = np.load(train_curvature_file, allow_pickle=True).item()
        val_curvature_dict = np.load(val_curvature_file, allow_pickle=True).item()
        return train_curvature_file, train_curvature_dict, val_curvature_file, val_curvature_dict
    
    if train_exists:
        print(f"\nTraining curvature file already exists: {train_curvature_file}")
        print(f"Loading existing training curvature data...")
        train_curvature_dict = np.load(train_curvature_file, allow_pickle=True).item()
    
    if val_exists:
        print(f"\nValidation curvature file already exists: {val_curvature_file}")
        print(f"Loading existing validation curvature data...")
        val_curvature_dict = np.load(val_curvature_file, allow_pickle=True).item()
    
    # Build data loaders (training set, no shuffling)
    if dataset_name.lower() == 'imagenet':
        train_loader, val_loader = build_imagenet_data(
            data_path='/local/a/imagenet/imagenet2012/',
            input_size=224,
            batch_size=batch_size,
            workers=workers
        )
    elif dataset_name.lower() == 'cifar10':
        train_loader, val_loader = build_cifar10_data(
            data_path='data/CIFAR10',
            input_size=32,
            batch_size=batch_size,
            workers=workers
        )
    elif dataset_name.lower() == 'cifar100':
        train_loader, val_loader = build_cifar100_data(
            data_path='data/CIFAR100',
            input_size=32,
            batch_size=batch_size,
            workers=workers
        )
    elif dataset_name.lower() == 'tinyimagenet':
        train_loader, val_loader, test_loader = build_tinyimagenet_data(
            data_path='data/TinyImageNet',
            input_size=64,
            batch_size=batch_size,
            workers=workers
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load model
    print(f"Loading model...")
    model = load_model(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device
    )
    model = model.to(device)
    model.eval()
    
    # Define loss criterion for validation (needs scalar output)
    val_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    print(f"\nChecking model validation accuracy for sanity check...")
    val_loss, val_acc, val_acc5 = validate(model, val_loader, val_criterion, device)
    print(f"Validation accuracy: {val_acc:.2f}%")
    if val_acc < 50:
        print(f"Validation accuracy is less than 50%. Exiting...")
        sys.exit(1)
    
    # Define loss criterion for curvature computation (needs per-sample output)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Compute training curvature if not exists
    if not train_exists:
        print(f"\nComputing curvature for training set...")
        print(f"Dataset size: {len(train_loader.dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        
        train_curvature_file, train_curvature_dict = compute_curvature_for_dataset(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            save_dir=save_dir,
            debug=debug,
            split='train'
        )
        
        print(f"\n{'='*80}")
        print(f"TRAINING CURVATURE COMPUTATION COMPLETE")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of training samples: {len(train_curvature_dict)}")
        print(f"Saved to: {train_curvature_file}")
        print(f"{'='*80}\n")
    
    # Compute validation curvature if not exists
    if not val_exists:
        print(f"\nComputing curvature for validation set...")
        print(f"Dataset size: {len(val_loader.dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        
        val_curvature_file, val_curvature_dict = compute_curvature_for_dataset(
            model=model,
            criterion=criterion,
            train_loader=val_loader,
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            save_dir=save_dir,
            debug=debug,
            split='val'
        )
        
        print(f"\n{'='*80}")
        print(f"VALIDATION CURVATURE COMPUTATION COMPLETE")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of validation samples: {len(val_curvature_dict)}")
        print(f"Saved to: {val_curvature_file}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"ALL CURVATURE COMPUTATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(train_curvature_dict)}")
    print(f"Validation samples: {len(val_curvature_dict)}")
    print(f"Train file: {train_curvature_file}")
    print(f"Val file: {val_curvature_file}")
    print(f"{'='*80}\n")
    
    return train_curvature_file, train_curvature_dict, val_curvature_file, val_curvature_dict


def main():
    """Main function to compute curvature."""
    parser = argparse.ArgumentParser(
        description='Compute curvature for training and validation datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture (e.g., resnet18, mobilenetv2, resnet50)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet', 'cifar10', 'cifar100', 'tinyimagenet'],
                        help='Dataset to compute curvature for')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (optional - will auto-find if not provided for non-ImageNet datasets)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used during training (default: 42, used for auto-finding checkpoints)')
    
    # Computation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for computation (default: 64)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (default: cuda if available)')
    
    # Save directory
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom directory to save curvature results (default: aihwkit_experiments/scores)')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    else:
        if args.dataset == 'imagenet':
            print(f"  Checkpoint: None (using pretrained torchvision weights)")
        else:
            print(f"  Checkpoint: Auto-find (seed={args.seed})")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.workers}")
    if args.save_dir:
        print(f"  Save directory: {args.save_dir}")
    else:
        print(f"  Save directory: aihwkit_experiments/scores (default)")
    print(f"  Debug: {args.debug}")
    
    # Compute curvature
    train_file, train_dict, val_file, val_dict = compute_curvature(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
        debug=args.debug
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"\nTraining set:")
    print(f"  Number of samples: {len(train_dict)}")
    print(f"  Curvature file: {train_file}")
    
    # Print training sample statistics
    train_values = list(train_dict.values())
    print(f"  Curvature statistics:")
    print(f"    Min: {np.min(train_values):.6f}")
    print(f"    Max: {np.max(train_values):.6f}")
    print(f"    Mean: {np.mean(train_values):.6f}")
    print(f"    Std: {np.std(train_values):.6f}")
    
    print(f"\nValidation set:")
    print(f"  Number of samples: {len(val_dict)}")
    print(f"  Curvature file: {val_file}")
    
    # Print validation sample statistics
    val_values = list(val_dict.values())
    print(f"  Curvature statistics:")
    print(f"    Min: {np.min(val_values):.6f}")
    print(f"    Max: {np.max(val_values):.6f}")
    print(f"    Mean: {np.mean(val_values):.6f}")
    print(f"    Std: {np.std(val_values):.6f}")
    print("="*80)


if __name__ == "__main__":
    main()

