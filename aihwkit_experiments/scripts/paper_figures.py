#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate paper-ready figures for curvature vs analog-only failure rate analysis.

This script creates a single publication-quality plot showing analog-only failure rates
across curvature bins for all hardware configurations.
"""

import os
import sys
import argparse
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def load_failing_indices(failing_indices_dir, split='val', fp=False, profile=None):
    """Load failing indices from analog inference results directory."""
    if fp:
        failing_file = os.path.join(failing_indices_dir, f'failed_indices_{split}_fp.pkl')
    else:
        if profile is None:
            raise ValueError("profile must be specified when loading analog failures")
        failing_file = os.path.join(failing_indices_dir, f'failed_indices_{split}_{profile}.pkl')
    
    if not os.path.exists(failing_file):
        raise FileNotFoundError(f"Failing indices file not found: {failing_file}")
    
    print(f"Loading failing indices from: {failing_file}")
    with open(failing_file, 'rb') as f:
        failing_indices = pickle.load(f)
    
    print(f"  Loaded {len(failing_indices)} failing samples")
    return set(failing_indices)


def create_bins(curvature_dict, n_bins=10):
    """Create curvature bins (deciles or quantiles)."""
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


def compute_bin_statistics_analog_only(bins, analog_only_failures, fp_failures):
    """Compute statistics for each curvature bin for analog-only failures."""
    bin_stats = []
    
    for bin_idx, sample_indices, curvatures in bins:
        n_samples = len(sample_indices)
        n_fp_failures = sum(1 for idx in sample_indices if idx in fp_failures)
        n_fp_correct = n_samples - n_fp_failures
        n_analog_only_failures = sum(1 for idx in sample_indices if idx in analog_only_failures)
        failure_rate = n_analog_only_failures / n_fp_correct if n_fp_correct > 0 else 0
        avg_curvature = np.mean(curvatures)
        
        bin_stats.append({
            'bin_idx': bin_idx,
            'n_samples': n_samples,
            'n_fp_correct': n_fp_correct,
            'n_failures': n_analog_only_failures,
            'failure_rate': failure_rate,
            'avg_curvature': avg_curvature,
            'min_curvature': np.min(curvatures),
            'max_curvature': np.max(curvatures),
        })
    
    return bin_stats


def format_profile_name(profile):
    """Format profile name for display in plot."""
    name_map = {
        'ideal': 'Ideal',
        'only_ADC': 'ADC Only',
        'only_wnoise': 'WNoise Only',
        'only_PCM': 'PCM Only',
        'ADC_and_wnoise': 'ADC + WNoise',
        'ADC_and_PCM': 'ADC + PCM',
        'wnoise_and_PCM': 'WNoise + PCM',
        'ADC_wnoise_and_PCM': 'ADC + WNoise + PCM'
    }
    return name_map.get(profile, profile.replace('_', ' ').title())


def save_csv_results(bin_stats_dict, save_path, profiles=None):
    """Save bin statistics to CSV file.
    
    Args:
        bin_stats_dict: Dictionary mapping profile name to bin statistics
        save_path: Path to save the CSV file
        profiles: List of profiles to include (if None, uses all in bin_stats_dict)
    """
    if profiles is None:
        profiles = list(bin_stats_dict.keys())
    
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['Profile', 'Bin', 'Failure_Rate_Percent', 'Avg_Curvature', 
                     'Min_Curvature', 'Max_Curvature', 'N_Samples', 'N_FP_Correct', 'N_Failures']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for profile in profiles:
            if profile not in bin_stats_dict:
                continue
            
            bin_stats = sorted(bin_stats_dict[profile], key=lambda x: x['bin_idx'])
            for stat in bin_stats:
                writer.writerow({
                    'Profile': profile,
                    'Bin': stat['bin_idx'] + 1,
                    'Failure_Rate_Percent': stat['failure_rate'] * 100,
                    'Avg_Curvature': stat['avg_curvature'],
                    'Min_Curvature': stat['min_curvature'],
                    'Max_Curvature': stat['max_curvature'],
                    'N_Samples': stat['n_samples'],
                    'N_FP_Correct': stat['n_fp_correct'],
                    'N_Failures': stat['n_failures']
                })
    
    print(f"Saved CSV results to: {save_path}")


def plot_paper_figure(bin_stats_dict, model_name, dataset_name, split, save_path, 
                      profiles=None, n_bins=10):
    """Generate publication-ready figure showing analog-only failure rates across curvature bins.
    
    Args:
        bin_stats_dict: Dictionary mapping profile name to bin statistics
        model_name: Name of the model
        dataset_name: Name of the dataset
        split: 'train' or 'val'
        save_path: Path to save the plot
        profiles: List of profiles to include (if None, uses all in bin_stats_dict)
        n_bins: Number of bins (for x-axis labels)
    """
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    if profiles is None:
        profiles = list(bin_stats_dict.keys())
    
    # Formal color scheme: professional, colorblind-friendly palette
    # Using a combination of distinct, muted colors suitable for academic publication
    formal_colors = [
        '#2E86AB',  # Deep blue
        '#A23B72',  # Deep purple/magenta
        '#C73E1D',  # Deep red
        '#6A994E',  # Forest green
        '#5F4B8B',  # Deep violet
        '#D4A574',  # Tan/brown
        '#0077BE',  # Ocean blue
        '#8B4513',  # Saddle brown
    ]
    
    # If we need more colors, extend with additional professional colors
    if len(profiles) > len(formal_colors):
        additional_colors = [
            '#4682B4',  # Steel blue
            '#CD5C5C',  # Indian red
            '#556B2F',  # Dark olive green
            '#9370DB',  # Medium purple
            '#20B2AA',  # Light sea green
            '#B8860B',  # Dark goldenrod (replaces light orange)
            '#6495ED',  # Cornflower blue
            '#708090',  # Slate gray
        ]
        formal_colors.extend(additional_colors)
    
    colors = formal_colors[:len(profiles)]
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    bin_indices = list(range(n_bins))
    bin_indices_array = np.array(bin_indices)
    
    for i, profile in enumerate(profiles):
        if profile not in bin_stats_dict:
            continue
            
        bin_stats = sorted(bin_stats_dict[profile], key=lambda x: x['bin_idx'])
        failure_rates = [stat['failure_rate'] * 100 for stat in bin_stats]
        profile_label = format_profile_name(profile)
        
        ax.plot(bin_indices, failure_rates, 
               marker=markers[i % len(markers)], 
               linestyle='-', 
               linewidth=2.5,
               markersize=8,
               color=colors[i], 
               label=profile_label,
               alpha=0.85,
               markeredgewidth=0.5,
               markeredgecolor=colors[i])
    
    ax.set_xlabel('Input Loss Curvature Deciles (Low → High)', fontsize=16, fontweight='bold', color='#2C2C2C')
    ax.set_ylabel('Analog-Only Failure Rate (%)', fontsize=16, fontweight='bold', color='#2C2C2C')
    
    ax.set_xticks(bin_indices)
    ax.set_xticklabels([f'{i+1}' for i in bin_indices], color='#2C2C2C')
    ax.tick_params(colors='#2C2C2C', width=1)
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#666666')
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, 
              ncol=2, framealpha=0.95, edgecolor='#CCCCCC', facecolor='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved paper figure to: {save_path}")
    plt.close()


def main():
    """Main function to generate paper figures."""
    parser = argparse.ArgumentParser(
        description='Generate paper-ready figures for curvature vs analog-only failure rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., resnet18, resnet34, resnet50)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet', 'cifar10', 'cifar100', 'tinyimagenet'],
                        help='Dataset name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of curvature bins (default: 10 for deciles)')
    
    parser.add_argument('--scores_dir', type=str, default=None,
                        help='Directory containing curvature files (default: aihwkit_experiments/scores/)')
    parser.add_argument('--failing_indices_dir', type=str, default=None,
                        help='Directory containing failing indices (default: auto-detect from analog_inference2 results)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: aihwkit_experiments/results/paper_figures/)')
    parser.add_argument('--profiles', nargs='+', type=str, default=None,
                        choices=['ideal', 'only_ADC', 'only_wnoise', 'only_PCM', 
                                'ADC_and_wnoise', 'ADC_and_PCM', 'wnoise_and_PCM', 'ADC_wnoise_and_PCM'],
                        help='Profiles to include (default: all non-ideal profiles)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PAPER FIGURE GENERATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Number of bins: {args.n_bins}")
    print("="*80 + "\n")
    
    if args.scores_dir is None:
        args.scores_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    
    if args.failing_indices_dir is None:
        analog_inference_base = os.path.join(os.path.dirname(__file__), '..', 'results', 'analog_inference2', args.dataset.lower(), args.model.lower())
        if os.path.exists(analog_inference_base):
            args.failing_indices_dir = analog_inference_base
            print(f"Auto-detected failing indices directory: {args.failing_indices_dir}\n")
        else:
            raise FileNotFoundError(f"No analog inference results found at: {analog_inference_base}")
    
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'paper_figures', args.dataset, args.model)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.profiles is None:
        profiles = ['only_ADC', 'only_wnoise', 'only_PCM', 
                   'ADC_and_wnoise', 'ADC_and_PCM', 'wnoise_and_PCM', 'ADC_wnoise_and_PCM']
    else:
        profiles = args.profiles
    
    print(f"Profiles to include: {profiles}\n")
    
    splits = ['train', 'val']
    
    for split in splits:
        print("\n" + "="*80)
        print(f"Processing {split.upper()} split")
        print("="*80 + "\n")
        
        try:
            print(f"Loading data for {split} split...")
            curvature_dict = load_curvature(args.scores_dir, args.model, args.dataset, split)
            
            fp_failures = load_failing_indices(args.failing_indices_dir, split, fp=True)
            
            print(f"\nCreating {args.n_bins} curvature bins...")
            bins = create_bins(curvature_dict, n_bins=args.n_bins)
            
            bin_stats_dict = {}
            
            for profile in profiles:
                print(f"\nProcessing profile: {profile}")
                try:
                    analog_failures = load_failing_indices(args.failing_indices_dir, split, fp=False, profile=profile)
                    analog_only_failures = analog_failures - fp_failures
                    
                    bin_stats = compute_bin_statistics_analog_only(bins, analog_only_failures, fp_failures)
                    bin_stats_dict[profile] = bin_stats
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    print(f"Skipping profile {profile}\n")
                    continue
            
            if not bin_stats_dict:
                print(f"Warning: No profiles were successfully processed for {split} split. Skipping...")
                continue
            
            png_path = os.path.join(args.save_dir, f'curvature_failure_rate_{split}_seed{args.seed}.png')
            csv_path = os.path.join(args.save_dir, f'curvature_failure_rate_{split}_seed{args.seed}.csv')
            
            plot_paper_figure(bin_stats_dict, args.model, args.dataset, split, png_path, 
                             profiles=list(bin_stats_dict.keys()), n_bins=args.n_bins)
            save_csv_results(bin_stats_dict, csv_path, profiles=list(bin_stats_dict.keys()))
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {split} split due to missing files\n")
            continue
    
    print("\n" + "="*80)
    print("FIGURE AND CSV GENERATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

