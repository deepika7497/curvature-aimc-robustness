#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize highest and lowest curvature samples from CIFAR-100.

This script creates a publication-ready figure showing samples with
the highest and lowest curvature values, similar to the CSG visualization.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def load_curvature(scores_dir, model_name, dataset_name, split='val'):
    """Load curvature values from numpy file."""
    curvature_file = os.path.join(
        scores_dir,
        f'{model_name}_{dataset_name}_{split}_curvature.npy'
    )
    
    if not os.path.exists(curvature_file):
        raise FileNotFoundError(f"Curvature file not found: {curvature_file}")
    
    print(f"Loading curvature from: {curvature_file}")
    curvature_dict = np.load(curvature_file, allow_pickle=True).item()
    print(f"  Loaded {len(curvature_dict)} samples")
    return curvature_dict


def load_cifar100_dataset(data_path='data/CIFAR100', split='train'):
    """Load CIFAR-100 dataset without normalization for visualization."""
    dataset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=(split == 'train'),
        download=True,
        transform=transforms.ToTensor()  # Only convert to tensor, no normalization
    )
    return dataset


def get_cifar100_class_names():
    """Get CIFAR-100 class names."""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]


def create_bins(curvature_dict, n_bins=10):
    """Create curvature bins (deciles or quantiles).
    
    Args:
        curvature_dict: Dictionary mapping sample index to curvature
        n_bins: Number of bins (default: 10 for deciles)
        
    Returns:
        bins: List of (bin_idx, sample_indices, curvature_values) tuples
    """
    indices = np.array(list(curvature_dict.keys()))
    curvatures = np.array([curvature_dict[idx] for idx in indices])
    
    sorted_idx = np.argsort(curvatures)
    sorted_indices = indices[sorted_idx]
    sorted_curvatures = curvatures[sorted_idx]
    
    bin_size = len(sorted_indices) // n_bins
    bins = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            end_idx = len(sorted_indices)
        else:
            end_idx = (i + 1) * bin_size
        
        bin_indices = sorted_indices[start_idx:end_idx]
        bin_curvatures = sorted_curvatures[start_idx:end_idx]
        bins.append((i, bin_indices, bin_curvatures))
    
    return bins


def select_classes_for_comparison(d1_indices, d10_indices, dataset, n_rows=4, n_cols=5):
    """Select classes that have samples in both D1 and D10 deciles for comparison.
    
    Args:
        d1_indices: Set of sample indices in D1 (lowest curvature decile)
        d10_indices: Set of sample indices in D10 (highest curvature decile)
        dataset: CIFAR-100 dataset
        n_rows: Number of classes to select
        n_cols: Minimum number of samples required per class in each decile
        
    Returns:
        List of class indices to use for both panels
    """
    # Group by class for D1 and D10
    d1_class_groups = {}
    d10_class_groups = {}
    
    for idx in d1_indices:
        if idx >= len(dataset):
            continue
        _, class_idx = dataset[idx]
        if class_idx not in d1_class_groups:
            d1_class_groups[class_idx] = []
        d1_class_groups[class_idx].append(idx)
    
    for idx in d10_indices:
        if idx >= len(dataset):
            continue
        _, class_idx = dataset[idx]
        if class_idx not in d10_class_groups:
            d10_class_groups[class_idx] = []
        d10_class_groups[class_idx].append(idx)
    
    # Find classes that appear in both D1 and D10 AND have enough samples in each
    common_classes = []
    for class_idx in d1_class_groups.keys():
        if class_idx in d10_class_groups:
            d1_count = len(d1_class_groups[class_idx])
            d10_count = len(d10_class_groups[class_idx])
            # Only include if both deciles have at least n_cols samples
            if d1_count >= n_cols and d10_count >= n_cols:
                # Prefer classes with more samples in both deciles
                common_classes.append((class_idx, d1_count + d10_count, d1_count, d10_count))
    
    # Sort by total count (descending)
    common_classes.sort(key=lambda x: x[1], reverse=True)
    
    # Select top n_rows classes
    selected_classes = [c[0] for c in common_classes[:n_rows]]
    
    if len(selected_classes) < n_rows:
        print(f"Warning: Only found {len(selected_classes)} classes with at least {n_cols} samples in both D1 and D10.")
        print(f"  Requested {n_rows} classes, but only {len(selected_classes)} meet the criteria.")
    
    return selected_classes


def select_samples_from_decile(decile_indices, curvature_dict, dataset, class_indices, n_cols=5):
    """Select samples from specified classes within a decile.
    
    Args:
        decile_indices: Set of sample indices in the decile
        curvature_dict: Dictionary mapping index to curvature value
        dataset: CIFAR-100 dataset
        class_indices: List of class indices to select from (in order)
        n_cols: Number of samples per class
        
    Returns:
        List of (index, curvature, image, class_idx, class_name) tuples, grouped by class.
        Each class contributes exactly n_cols samples (or fewer if not enough available).
    """
    # Group by class within the decile
    class_groups = {}
    for idx in decile_indices:
        if idx >= len(dataset) or idx not in curvature_dict:
            continue
        _, class_idx = dataset[idx]
        if class_idx in class_indices:
            if class_idx not in class_groups:
                class_groups[class_idx] = []
            curvature = curvature_dict[idx]
            class_groups[class_idx].append((idx, curvature))
    
    # Select samples from each class in order
    selected = []
    class_names = get_cifar100_class_names()
    
    for class_idx in class_indices:
        if class_idx not in class_groups:
            # If class not found, add empty slots to maintain row structure
            continue
        
        class_samples = class_groups[class_idx]
        # Sort samples within class by curvature (ascending)
        class_samples = sorted(class_samples, key=lambda x: x[1], reverse=False)
        
        # Take first n_cols samples from this class
        for idx, curvature in class_samples[:n_cols]:
            image, _ = dataset[idx]
            selected.append((idx, curvature, image, class_idx, class_names[class_idx]))
    
    return selected


def denormalize_image(tensor):
    """Convert tensor image to PIL Image for display.
    
    CIFAR-100 normalization: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    But we're loading without normalization, so just convert tensor to numpy.
    """
    # Convert from CHW to HWC and clamp to [0, 1]
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def create_curvature_visualization(d1_samples, d10_samples, save_path, 
                                   n_rows=4, n_cols=5):
    """Create a publication-ready visualization of curvature samples from D1 and D10 deciles.
    
    Args:
        d1_samples: List of (idx, curvature, image, class_idx, class_name) for D1 (lowest curvature decile)
        d10_samples: List of (idx, curvature, image, class_idx, class_name) for D10 (highest curvature decile)
        save_path: Path to save the figure
        n_rows: Number of rows in each panel
        n_cols: Number of columns in each panel
    """
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
    
    # Arrange side by side with class labels in center
    # Layout: [left panel] [center labels] [right panel] [bottom labels]
    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, 
                          height_ratios=[1, 0.05],
                          width_ratios=[1, 0.12, 1],
                          hspace=0.0, wspace=0.02,
                          left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    # Panel (a): D1 (lowest curvature decile) samples (left)
    gs_a = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0, 0], 
                                            hspace=0.01, wspace=0.01)
    
    # Group samples by class to ensure each row is one class
    d1_by_class = {}
    for idx, curvature, image, class_idx, class_name in d1_samples:
        if class_idx not in d1_by_class:
            d1_by_class[class_idx] = []
        d1_by_class[class_idx].append((idx, curvature, image, class_name))
    
    # Get class order from selected classes (maintain order)
    class_names = get_cifar100_class_names()
    class_order = []
    seen_classes = set()
    for _, _, _, class_idx, _ in d1_samples:
        if class_idx not in seen_classes:
            class_order.append((class_idx, class_names[class_idx]))
            seen_classes.add(class_idx)
    
    # Plot images: each row is one class
    for row, (class_idx, class_name) in enumerate(class_order[:n_rows]):
        if class_idx not in d1_by_class:
            continue
        class_samples = d1_by_class[class_idx][:n_cols]  # Take at most n_cols samples
        
        for col, (idx, curvature, image, _) in enumerate(class_samples):
            if col >= n_cols:
                break
            ax_img = fig.add_subplot(gs_a[row, col])
            img_array = denormalize_image(image)
            ax_img.imshow(img_array)
            ax_img.axis('off')
    
    # Center: Class labels (vertical)
    gs_labels = gridspec.GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs[0, 1], 
                                                 hspace=0.01)
    
    for row, (class_idx, class_name) in enumerate(class_order[:n_rows]):
        ax_label = fig.add_subplot(gs_labels[row, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, class_name.replace('_', ' ').title(), 
                     fontsize=20, ha='center', va='center',
                     transform=ax_label.transAxes, fontweight='bold',
                     rotation=90)
    
    # Panel (b): D10 (highest curvature decile) samples (right)
    gs_b = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0, 2], 
                                            hspace=0.01, wspace=0.01)
    
    # Group samples by class to ensure each row is one class
    d10_by_class = {}
    for idx, curvature, image, class_idx, class_name in d10_samples:
        if class_idx not in d10_by_class:
            d10_by_class[class_idx] = []
        d10_by_class[class_idx].append((idx, curvature, image, class_name))
    
    # Plot images: each row is one class (use same class order as D1)
    for row, (class_idx, class_name) in enumerate(class_order[:n_rows]):
        if class_idx not in d10_by_class:
            continue
        class_samples = d10_by_class[class_idx][:n_cols]  # Take at most n_cols samples
        
        for col, (idx, curvature, image, _) in enumerate(class_samples):
            if col >= n_cols:
                break
            ax_img = fig.add_subplot(gs_b[row, col])
            img_array = denormalize_image(image)
            ax_img.imshow(img_array)
            ax_img.axis('off')
    
    # Bottom labels: (a) and (b)
    ax_label_a = fig.add_subplot(gs[1, 0])
    ax_label_a.axis('off')
    ax_label_a.text(0.5, 0.5, 'D1 (Lowest Curvature Decile)', 
                   fontsize=24, fontweight='bold', ha='center', va='center',
                   transform=ax_label_a.transAxes)
    
    ax_label_center = fig.add_subplot(gs[1, 1])
    ax_label_center.axis('off')
    
    ax_label_b = fig.add_subplot(gs[1, 2])
    ax_label_b.axis('off')
    ax_label_b.text(0.5, 0.5, 'D10 (Highest Curvature Decile)', 
                   fontsize=24, fontweight='bold', ha='center', va='center',
                   transform=ax_label_b.transAxes)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved curvature visualization to: {save_path}")
    plt.close()


def main():
    """Main function to generate curvature sample visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize highest and lowest curvature samples from CIFAR-100',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., resnet18, resnet34, resnet50)')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'],
                        help='Dataset name (currently only CIFAR-100 supported)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split (default: val)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples per panel (default: 30)')
    parser.add_argument('--n_rows', type=int, default=4,
                        help='Number of rows in each panel (default: 6)')
    parser.add_argument('--n_cols', type=int, default=5,
                        help='Number of columns in each panel (default: 5)')
    
    parser.add_argument('--scores_dir', type=str, default=None,
                        help='Directory containing curvature files (default: aihwkit_experiments/scores/)')
    parser.add_argument('--data_path', type=str, default='data/CIFAR100',
                        help='Path to CIFAR-100 dataset (default: data/CIFAR100)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: aihwkit_experiments/results/curvature_visualization/)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CURVATURE SAMPLE VISUALIZATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Number of samples per panel: {args.n_samples}")
    print(f"Grid: {args.n_rows} rows x {args.n_cols} columns")
    print("="*80 + "\n")
    
    if args.scores_dir is None:
        args.scores_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 
                                     'curvature_visualization', args.dataset, args.model)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load curvature data
    print("Loading curvature data...")
    curvature_dict = load_curvature(args.scores_dir, args.model, args.dataset, args.split)
    
    # Load dataset
    print(f"\nLoading {args.dataset} {args.split} dataset...")
    dataset = load_cifar100_dataset(args.data_path, args.split)
    print(f"  Loaded {len(dataset)} samples")
    
    # Create deciles
    print(f"\nCreating 10 deciles from curvature data...")
    bins = create_bins(curvature_dict, n_bins=10)
    print(f"  Created {len(bins)} deciles")
    
    # Get D1 (bin 0, lowest curvature) and D10 (bin 9, highest curvature)
    d1_bin_idx, d1_indices, d1_curvatures = bins[0]
    d10_bin_idx, d10_indices, d10_curvatures = bins[9]
    
    d1_indices_set = set(d1_indices)
    d10_indices_set = set(d10_indices)
    
    print(f"  D1 (lowest curvature decile): {len(d1_indices)} samples")
    print(f"    Curvature range: [{np.min(d1_curvatures):.6e}, {np.max(d1_curvatures):.6e}]")
    print(f"  D10 (highest curvature decile): {len(d10_indices)} samples")
    print(f"    Curvature range: [{np.min(d10_curvatures):.6e}, {np.max(d10_curvatures):.6e}]")
    
    # Select classes for comparison (same classes for both panels)
    print(f"\nSelecting {args.n_rows} classes that appear in both D1 and D10...")
    print(f"  Requiring at least {args.n_cols} samples per class in each decile...")
    selected_classes = select_classes_for_comparison(d1_indices_set, d10_indices_set, dataset, 
                                                      n_rows=args.n_rows, n_cols=args.n_cols)
    class_names = get_cifar100_class_names()
    print(f"  Selected classes: {[class_names[c] for c in selected_classes]}")
    
    if len(selected_classes) == 0:
        raise ValueError(f"No classes found with at least {args.n_cols} samples in both D1 and D10 deciles. "
                        f"Try reducing --n_cols or --n_rows.")
    
    # Select samples from D1 and D10 deciles
    print(f"\nSelecting samples from D1 decile...")
    d1_samples = select_samples_from_decile(
        d1_indices_set, curvature_dict, dataset, selected_classes, n_cols=args.n_cols
    )
    print(f"  Selected {len(d1_samples)} samples from D1")
    
    print(f"\nSelecting samples from D10 decile...")
    d10_samples = select_samples_from_decile(
        d10_indices_set, curvature_dict, dataset, selected_classes, n_cols=args.n_cols
    )
    print(f"  Selected {len(d10_samples)} samples from D10")
    
    # Create visualization
    print(f"\nCreating visualization...")
    save_path = os.path.join(args.save_dir, 
                             f'curvature_samples_{args.split}_seed{args.seed}.png')
    create_curvature_visualization(
        d1_samples, d10_samples, save_path,
        n_rows=args.n_rows, n_cols=args.n_cols
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

