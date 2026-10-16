"""
Utility functions for plotting training metrics.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional


def load_training_metrics(metrics_file: str) -> Dict:
    """
    Load training metrics from a pickle file.
    
    Args:
        metrics_file: Path to the training_metrics.pkl file
        
    Returns:
        Dictionary containing training metrics and experiment config
    """
    with open(metrics_file, 'rb') as f:
        return pickle.load(f)


def plot_training_curves(metrics_file: str, save_path: Optional[str] = None, 
                        figsize: tuple = (12, 8), dpi: int = 100) -> None:
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        metrics_file: Path to the training_metrics.pkl file
        save_path: Optional path to save the plot (if None, displays the plot)
        figsize: Figure size as (width, height)
        dpi: DPI for the saved figure
    """
    # Load metrics
    metrics = load_training_metrics(metrics_file)
    
    # Extract data
    epochs = metrics['epochs']
    train_losses = metrics['train_losses']
    train_accuracies = metrics['train_accuracies']
    val_losses = metrics['val_losses']
    val_accuracies = metrics['val_accuracies']
    config = metrics['experiment_config']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Plot accuracy
    ax1.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot loss
    ax2.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add experiment info as text
    exp_info = f"Dataset: {config['dataset']}\n"
    exp_info += f"Arch: {config['arch']}\n"
    exp_info += f"Type: {config['experiment_type']}\n"
    exp_info += f"Strategy: {config['strategy']}\n"
    exp_info += f"Percentage: {config['percentage']}%\n"
    exp_info += f"Best Val Acc: {max(val_accuracies):.2f}%"
    
    fig.text(0.02, 0.02, exp_info, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_experiments(metrics_files: List[str], labels: List[str], 
                            save_path: Optional[str] = None, 
                            figsize: tuple = (12, 8), dpi: int = 100) -> None:
    """
    Plot multiple experiments on the same graph for comparison.
    
    Args:
        metrics_files: List of paths to training_metrics.pkl files
        labels: List of labels for each experiment
        save_path: Optional path to save the plot
        figsize: Figure size as (width, height)
        dpi: DPI for the saved figure
    """
    if len(metrics_files) != len(labels):
        raise ValueError("Number of metrics files must match number of labels")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_files)))
    
    for i, (metrics_file, label, color) in enumerate(zip(metrics_files, labels, colors)):
        # Load metrics
        metrics = load_training_metrics(metrics_file)
        
        epochs = metrics['epochs']
        train_accuracies = metrics['train_accuracies']
        val_accuracies = metrics['val_accuracies']
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        
        # Plot accuracy
        ax1.plot(epochs, val_accuracies, color=color, label=f'{label} (Val)', linewidth=2)
        ax1.plot(epochs, train_accuracies, color=color, linestyle='--', 
                label=f'{label} (Train)', linewidth=1, alpha=0.7)
        
        # Plot loss
        ax2.plot(epochs, val_losses, color=color, label=f'{label} (Val)', linewidth=2)
        ax2.plot(epochs, train_losses, color=color, linestyle='--', 
                label=f'{label} (Train)', linewidth=1, alpha=0.7)
    
    # Configure accuracy plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Configure loss plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_training_summary(metrics_file: str) -> None:
    """
    Print a summary of training metrics.
    
    Args:
        metrics_file: Path to the training_metrics.pkl file
    """
    metrics = load_training_metrics(metrics_file)
    
    config = metrics['experiment_config']
    train_accuracies = metrics['train_accuracies']
    val_accuracies = metrics['val_accuracies']
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Dataset: {config['dataset']}")
    print(f"Architecture: {config['arch']}")
    print(f"Experiment Type: {config['experiment_type']}")
    print(f"Strategy: {config['strategy']}")
    print(f"Percentage: {config['percentage']}%")
    print(f"Original Seed: {config['original_seed']}")
    print(f"Current Seed: {config['seed']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Batch Size: {config['batch_size']}")
    print("-" * 60)
    print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Best Val Accuracy: {max(val_accuracies):.2f}% (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    print(f"Min Val Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_utils.py <metrics_file> [save_path]")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file {metrics_file} not found")
        sys.exit(1)
    
    # Print summary
    print_training_summary(metrics_file)
    
    # Create plot
    plot_training_curves(metrics_file, save_path) 