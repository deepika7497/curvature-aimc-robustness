#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train models on CIFAR-10, CIFAR-100, and TinyImageNet datasets.

This script trains models from scratch or fine-tunes existing models
and saves checkpoints to results/checkpoints/dataset_name/model_name/.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pickle
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.data_utils import build_cifar10_data, build_cifar100_data, build_tinyimagenet_data
from utils.model_utils import create_cifar10_model, create_cifar100_model, create_tinyimagenet_model, save_checkpoint, load_checkpoint
from utils.training_utils import train_epoch, validate, seed_all
from utils.curv_utils import create_indexed_dataloader


def train_epoch_with_sample_loss(model, train_loader, criterion, per_sample_criterion, 
                                 optimizer, device, epoch, sample_losses):
    """Train for one epoch and record per-sample losses.
    
    Args:
        model: The model to train
        train_loader: Training data loader (indexed)
        criterion: Loss criterion for backprop
        per_sample_criterion: Loss criterion with reduction='none' for recording
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        sample_losses: Dictionary to store sample losses
        
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (indices, inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Record per-sample losses
        with torch.no_grad():
            per_sample_losses = per_sample_criterion(outputs, targets)
            for idx, sample_loss in zip(indices, per_sample_losses):
                sample_losses[idx.item()].append(sample_loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


def create_model(model_name, dataset_name):
    """Create a model for the specified dataset.
    
    Args:
        model_name: Name of the model architecture
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'tinyimagenet')
        
    Returns:
        model: Created model
    """
    if dataset_name == 'cifar10':
        return create_cifar10_model(model_name)
    elif dataset_name == 'cifar100':
        return create_cifar100_model(model_name)
    elif dataset_name == 'tinyimagenet':
        return create_tinyimagenet_model(model_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def train_model(model_name, dataset_name, batch_size=128, workers=4, epochs=200, 
                lr=0.1, momentum=0.9, weight_decay=1e-4, device='cuda', 
                save_dir=None, checkpoint_path=None, seed=42):
    """Train a model on the specified dataset.
    
    Args:
        model_name: Name of the model architecture
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'tinyimagenet')
        batch_size: Batch size for training
        workers: Number of data loading workers
        epochs: Number of training epochs
        lr: Initial learning rate
        momentum: SGD momentum
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
        checkpoint_path: Path to existing checkpoint to resume from
        seed: Random seed for reproducibility
        
    Returns:
        best_checkpoint_path: Path to the saved best checkpoint
    """
    
    # Set random seed
    seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL ON {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Momentum: {momentum}")
    print(f"Weight decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Build data loaders
    if dataset_name.lower() == 'cifar10':
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
        train_loader, val_loader, _ = build_tinyimagenet_data(
            data_path='data/TinyImageNet',
            input_size=64,
            batch_size=batch_size,
            workers=workers
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # Create indexed data loader for sample loss tracking
    print("Creating indexed data loader for sample loss tracking...")
    indexed_train_loader, indexed_train_dataset = create_indexed_dataloader(train_loader)
    print(f"  Indexed training samples: {len(indexed_train_dataset)}")
    print(f"  Indexed training batches: {len(indexed_train_loader)}\n")
    
    # Create model
    print(f"Creating model: {model_name}")
    model = create_model(model_name, dataset_name)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')  # For recording individual losses
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    
    # Initialize sample losses storage
    sample_losses = defaultdict(list)
    
    # Learning rate scheduler - use different schedulers for different datasets
    if dataset_name.lower() == 'tinyimagenet':
        # For TinyImageNet, use MultiStepLR with decay at specific epochs
        if epochs >= 300:
            milestones = [90, 180, 240]
        elif epochs >= 200:
            milestones = [60, 120, 160]
        else:
            milestones = [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.8)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print(f"Using MultiStepLR with milestones at epochs: {milestones}")
    else:
        # For CIFAR datasets, use CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"Using CosineAnnealingLR with T_max={epochs}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(model, checkpoint_path, device=device)
        if checkpoint:
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"  Resuming from epoch {start_epoch}")
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f"  Previous best accuracy: {best_acc:.2f}%")
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"  Optimizer state loaded")
        print()
    
    # Determine save directory
    if save_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints')
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
        save_dir = os.path.join(base_dir, dataset_name, f'{model_name}_seed{seed}_{timestamp}')
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}\n")
    
    best_epoch = start_epoch
    best_checkpoint_path = None
    
    # Initialize lists to store training history
    training_history = {
        'epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_accuracies_top5': []
    }
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}")
        
        # Train for one epoch with sample loss recording
        train_loss, train_acc = train_epoch_with_sample_loss(
            model, indexed_train_loader, criterion, per_sample_criterion, 
            optimizer, device, epoch, sample_losses
        )
        
        # Validate
        val_loss, val_acc, val_acc5 = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
        
        # Save regular checkpoint at every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                arch=model_name,
                save_dir=save_dir,
                filename=f'{model_name}_epoch{epoch+1}.pth'
            )
            print(f"✓ Checkpoint saved: {checkpoint_path} at epoch {epoch+1}")
        
        # Save best checkpoint separately
        if is_best:
            best_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                arch=model_name,
                save_dir=save_dir,
                filename=f'{model_name}_best.pth'
            )
            print(f"✓ New best accuracy: {best_acc:.2f}% at epoch {epoch+1}")
            print(f"✓ Best checkpoint saved: {best_checkpoint_path}")
        
        # Record training history
        training_history['epochs'].append(epoch + 1)
        training_history['train_losses'].append(train_loss)
        training_history['train_accuracies'].append(train_acc)
        training_history['val_losses'].append(val_loss)
        training_history['val_accuracies'].append(val_acc)
        training_history['val_accuracies_top5'].append(val_acc5)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Acc@5: {val_acc5:.2f}%")
        print(f"  Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch+1})")
        
        # Save last checkpoint
        if epoch == epochs - 1:
            last_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                arch=model_name,
                save_dir=save_dir,
                filename=f'{model_name}_last.pth'
            )
            print(f"✓ Last checkpoint saved: {last_checkpoint_path}")
    
    # Compute cumulative (summed) sample losses
    print(f"\nComputing cumulative sample losses...")
    cumulative_sample_losses = {}
    for idx, losses in sample_losses.items():
        cumulative_sample_losses[idx] = sum(losses)
    
    print(f"Total samples tracked: {len(cumulative_sample_losses)}")
    print(f"Average cumulative loss: {sum(cumulative_sample_losses.values()) / len(cumulative_sample_losses):.4f}")
    
    # Save cumulative sample losses to scores directory
    scores_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    os.makedirs(scores_dir, exist_ok=True)
    
    sample_loss_file = os.path.join(
        scores_dir, 
        f'{model_name}_{dataset_name}_seed{seed}_sample_losses.pkl'
    )
    
    print(f"Saving cumulative sample losses...")
    with open(sample_loss_file, 'wb') as f:
        pickle.dump(cumulative_sample_losses, f)
    print(f"Cumulative sample losses saved to: {sample_loss_file}")
    
    # Save training history (metrics) to scores directory
    training_metrics_file = os.path.join(
        scores_dir,
        f'{model_name}_{dataset_name}_seed{seed}_training_metrics.pkl'
    )
    
    training_metrics = {
        'epochs': training_history['epochs'],
        'train_losses': training_history['train_losses'],
        'train_accuracies': training_history['train_accuracies'],
        'val_losses': training_history['val_losses'],
        'val_accuracies': training_history['val_accuracies'],
        'val_accuracies_top5': training_history['val_accuracies_top5'],
        'best_epoch': best_epoch + 1,
        'best_val_acc': best_acc,
        'config': {
            'model': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay
        }
    }
    
    print(f"Saving training metrics...")
    with open(training_metrics_file, 'wb') as f:
        pickle.dump(training_metrics, f)
    print(f"Training metrics saved to: {training_metrics_file}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {best_acc:.2f}% (Epoch {best_epoch+1})")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"All checkpoints saved to: {save_dir}")
    print(f"Cumulative sample losses saved to: {sample_loss_file}")
    print(f"Training metrics saved to: {training_metrics_file}")
    print(f"{'='*80}\n")
    
    return best_checkpoint_path


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(
        description='Train models on CIFAR-10, CIFAR-100, and TinyImageNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet18 on CIFAR-10
  python train_model.py --model resnet18 --dataset cifar10 --epochs 200
  
  # Train ResNet50 on CIFAR-100 with custom learning rate
  python train_model.py --model resnet50 --dataset cifar100 --epochs 200 --lr 0.05
  
  # Train MobileNetV2 on TinyImageNet
  python train_model.py --model mobilenetv2 --dataset tinyimagenet --epochs 200
  
  # Resume training from checkpoint
  python train_model.py --model resnet18 --dataset cifar10 --resume /path/to/checkpoint.pth
        """
    )
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture (e.g., resnet18, resnet50, mobilenetv2)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='Dataset to train on')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    
    # Data loading
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available)')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom directory to save checkpoints (default: results/checkpoints/dataset/model_seedX_timestamp)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Workers: {args.workers}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    if args.save_dir:
        print(f"  Save directory: {args.save_dir}")
    
    # Train the model
    best_checkpoint = train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        checkpoint_path=args.resume,
        seed=args.seed
    )
    
    # Construct expected file paths
    scores_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    sample_loss_file = os.path.join(
        scores_dir, 
        f'{args.model}_{args.dataset}_seed{args.seed}_sample_losses.pkl'
    )
    training_metrics_file = os.path.join(
        scores_dir,
        f'{args.model}_{args.dataset}_seed{args.seed}_training_metrics.pkl'
    )
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Cumulative sample losses: {sample_loss_file}")
    print(f"Training metrics: {training_metrics_file}")
    print("="*80)


if __name__ == "__main__":
    main()

