#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze relationship between curvature and analog failures.

This script performs statistical analysis to understand how curvature relates to
analog hardware failures. It includes:
- Curvature binning and failure rate analysis
- Comparison of FP and Analog failures
- Correlation analysis (Spearman)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def load_curvature(scores_dir, model_name, dataset_name, split='val'):
    """Load curvature values from numpy file.
    
    Args:
        scores_dir: Directory containing score files
        model_name: Name of the model (e.g., 'resnet18')
        dataset_name: Name of the dataset (e.g., 'cifar10')
        split: 'train' or 'val' to specify which split to load
        
    Returns:
        Dictionary mapping sample index to curvature value
    """
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
    """Load failing indices from analog inference results directory.
    
    Args:
        failing_indices_dir: Directory containing failing indices
        split: 'train' or 'val'
        fp: If True, load FP failures; otherwise load analog failures
        profile: Profile name (e.g., 'ideal', 'only_ADC', etc.) - required if fp=False
        
    Returns:
        Set of failing sample indices
    """
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


def compute_failure_statistics(analog_failures, fp_failures, total_samples):
    """Compute failure statistics for FP and analog.
    
    Args:
        analog_failures: Set of indices that fail in analog
        fp_failures: Set of indices that fail in full precision
        total_samples: Total number of samples
        
    Returns:
        Dictionary with failure statistics
    """
    analog_only = analog_failures - fp_failures
    fp_accuracy = (total_samples - len(fp_failures)) / total_samples * 100
    analog_accuracy = (total_samples - len(analog_failures)) / total_samples * 100
    
    print(f"\nFailure breakdown:")
    print(f"  Total samples: {total_samples}")
    print(f"  FP failures: {len(fp_failures)} (FP accuracy: {fp_accuracy:.2f}%)")
    print(f"  Analog failures: {len(analog_failures)} (Analog accuracy: {analog_accuracy:.2f}%)")
    print(f"  Analog-specific failures (analog fails, FP succeeds): {len(analog_only)}")
    print(f"  Overlap (both fail): {len(analog_failures & fp_failures)}")
    
    return {
        'fp_failures': len(fp_failures),
        'analog_failures': len(analog_failures),
        'analog_only_failures': len(analog_only),
        'overlap': len(analog_failures & fp_failures),
        'fp_accuracy': fp_accuracy,
        'analog_accuracy': analog_accuracy,
        'total_samples': total_samples
    }


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


def compute_bin_statistics(bins, failing_indices):
    """Compute statistics for each curvature bin.
    
    Args:
        bins: List of (bin_idx, sample_indices, curvature_values) tuples
        failing_indices: Set of failing sample indices
        
    Returns:
        bin_stats: List of dictionaries with statistics for each bin
    """
    bin_stats = []
    
    for bin_idx, sample_indices, curvatures in bins:
        n_samples = len(sample_indices)
        n_failures = sum(1 for idx in sample_indices if idx in failing_indices)
        failure_rate = n_failures / n_samples if n_samples > 0 else 0
        avg_curvature = np.mean(curvatures)
        
        bin_stats.append({
            'bin_idx': bin_idx,
            'n_samples': n_samples,
            'n_failures': n_failures,
            'failure_rate': failure_rate,
            'avg_curvature': avg_curvature,
            'min_curvature': np.min(curvatures),
            'max_curvature': np.max(curvatures),
        })
    
    return bin_stats


def compute_bin_statistics_analog_only(bins, analog_only_failures, fp_failures):
    """Compute statistics for each curvature bin for analog-only failures.
    
    For analog-only failures, the denominator should be samples that were
    correctly classified by FP (not all samples in the bin).
    
    Args:
        bins: List of (bin_idx, sample_indices, curvature_values) tuples
        analog_only_failures: Set of analog-only failing sample indices (analog fails, FP succeeds)
        fp_failures: Set of FP failing sample indices
        
    Returns:
        bin_stats: List of dictionaries with statistics for each bin
    """
    bin_stats = []
    
    for bin_idx, sample_indices, curvatures in bins:
        n_samples = len(sample_indices)
        n_fp_failures = sum(1 for idx in sample_indices if idx in fp_failures)
        n_fp_correct = n_samples - n_fp_failures  # Samples that FP got right
        n_analog_only_failures = sum(1 for idx in sample_indices if idx in analog_only_failures)
        # Failure rate = analog-only failures / samples that FP got correct
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


def plot_failure_rate_vs_curvature(bin_stats, model_name, dataset_name, split, save_path, title_suffix=""):
    """Plot failure rate vs curvature bin.
    
    Args:
        bin_stats: List of bin statistics
        model_name: Name of the model
        dataset_name: Name of the dataset
        split: 'train' or 'val'
        save_path: Path to save the plot
        title_suffix: Additional text to add to title (e.g., "FP", "Analog")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bin_indices = [stat['bin_idx'] for stat in bin_stats]
    failure_rates = [stat['failure_rate'] * 100 for stat in bin_stats]
    avg_curvatures = [stat['avg_curvature'] for stat in bin_stats]
    
    bars = ax1.bar(bin_indices, failure_rates, alpha=0.7, color='steelblue', edgecolor='black')
    for i, (bar, rate) in enumerate(zip(bars, failure_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Curvature Bin (sorted by curvature)', fontsize=12)
    ax1.set_ylabel('Failure Rate (%)', fontsize=12)
    title = f'Failure Rate vs Curvature Bin\n{model_name} - {dataset_name} ({split})'
    if title_suffix:
        title += f' - {title_suffix}'
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(avg_curvatures, failure_rates, s=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Average Curvature in Bin', fontsize=12)
    ax2.set_ylabel('Failure Rate (%)', fontsize=12)
    title = f'Failure Rate vs Average Curvature\n{model_name} - {dataset_name} ({split})'
    if title_suffix:
        title += f' - {title_suffix}'
    ax2.set_title(title, fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    z = np.polyfit(avg_curvatures, failure_rates, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(avg_curvatures), max(avg_curvatures), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label=f'Linear fit: y={z[0]:.2e}x+{z[1]:.2f}')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved failure rate plot to: {save_path}")
    plt.close()


def plot_analog_failure_rate_stacked(bin_stats_fp, bin_stats_analog_only, bin_stats_analog, 
                                     model_name, dataset_name, split, save_path, profile=None):
    """Plot analog failure rate with stacked bars (FP failures at bottom, Analog-only on top).
    
    Args:
        bin_stats_fp: List of bin statistics for FP failures
        bin_stats_analog_only: List of bin statistics for Analog-only failures
        bin_stats_analog: List of bin statistics for total Analog failures
        model_name: Name of the model
        dataset_name: Name of the dataset
        split: 'train' or 'val'
        save_path: Path to save the plot
        profile: Profile name (optional, for title)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort all by bin_idx to ensure alignment
    bin_stats_fp = sorted(bin_stats_fp, key=lambda x: x['bin_idx'])
    bin_stats_analog_only = sorted(bin_stats_analog_only, key=lambda x: x['bin_idx'])
    bin_stats_analog = sorted(bin_stats_analog, key=lambda x: x['bin_idx'])
    
    bin_indices = [stat['bin_idx'] for stat in bin_stats_analog]
    failure_rates_fp = [stat['failure_rate'] * 100 for stat in bin_stats_fp]
    failure_rates_analog_only = [stat['failure_rate'] * 100 for stat in bin_stats_analog_only]
    failure_rates_analog = [stat['failure_rate'] * 100 for stat in bin_stats_analog]
    avg_curvatures = [stat['avg_curvature'] for stat in bin_stats_analog]
    
    # Stacked bar chart: FP at bottom, Analog-only on top
    bars_fp = ax1.bar(bin_indices, failure_rates_fp, alpha=0.7, color='steelblue', edgecolor='black', label='FP Failures')
    bars_analog_only = ax1.bar(bin_indices, failure_rates_analog_only, bottom=failure_rates_fp, alpha=0.7, 
            color='coral', edgecolor='black', label='Analog-Only Failures')
    
    # Add value labels on top of bars
    total_rates = [fp + ao for fp, ao in zip(failure_rates_fp, failure_rates_analog_only)]
    for i, (bar_fp, bar_ao, fp_rate, ao_rate, total_rate) in enumerate(zip(bars_fp, bars_analog_only, failure_rates_fp, failure_rates_analog_only, total_rates)):
        # Label on top of FP bar (if visible)
        if fp_rate > 0:
            ax1.text(bar_fp.get_x() + bar_fp.get_width() / 2., fp_rate,
                    f'{fp_rate:.2f}%', ha='center', va='bottom', fontsize=8)
        # Label on top of total stacked bar
        ax1.text(bar_ao.get_x() + bar_ao.get_width() / 2., total_rate,
                f'{total_rate:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Curvature Bin (sorted by curvature)', fontsize=12)
    ax1.set_ylabel('Failure Rate (%)', fontsize=12)
    title = f'Failure Rate vs Curvature Bin\n{model_name} - {dataset_name} ({split})'
    if profile:
        title += f' - {profile}'
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot with total analog failure rate
    ax2.scatter(avg_curvatures, failure_rates_analog, s=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Average Curvature in Bin', fontsize=12)
    ax2.set_ylabel('Failure Rate (%)', fontsize=12)
    title = f'Failure Rate vs Average Curvature\n{model_name} - {dataset_name} ({split})'
    if profile:
        title += f' - {profile}'
    ax2.set_title(title, fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    z = np.polyfit(avg_curvatures, failure_rates_analog, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(avg_curvatures), max(avg_curvatures), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label=f'Linear fit: y={z[0]:.2e}x+{z[1]:.2f}')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved failure rate plot to: {save_path}")
    plt.close()


def plot_profile_comparison_curvature(bin_stats_dict, model_name, dataset_name, split, save_path, 
                                      plot_type='analog_only', bin_stats_fp=None):
    """Plot comparison of different profiles in the same plot for curvature analysis.
    
    Args:
        bin_stats_dict: Dictionary mapping profile name to bin statistics
        model_name: Name of the model
        dataset_name: Name of the dataset
        split: 'train' or 'val'
        save_path: Path to save the plot
        plot_type: 'analog_only' or 'analog_total' - what to plot
        bin_stats_fp: Optional bin statistics for FP (for stacked plot)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    profiles = list(bin_stats_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    profile_colors = dict(zip(profiles, colors))
    
    # Get bin indices from first profile (all should have same bins)
    first_profile = profiles[0]
    bin_stats_first = sorted(bin_stats_dict[first_profile], key=lambda x: x['bin_idx'])
    bin_indices = [stat['bin_idx'] for stat in bin_stats_first]
    
    # Bar chart comparison
    bin_indices_array = np.array(bin_indices)
    bar_width = 0.8 / len(profiles) if len(profiles) > 1 else 0.8
    
    if plot_type == 'analog_only' and bin_stats_fp is not None:
        bin_stats_fp = sorted(bin_stats_fp, key=lambda x: x['bin_idx'])
        failure_rates_fp = [stat['failure_rate'] * 100 for stat in bin_stats_fp]
        bars_fp = ax.bar(bin_indices, failure_rates_fp, alpha=0.4, color='gray', edgecolor='black', 
               label='FP Failures', width=0.8)
    
    all_bars = []
    for i, profile in enumerate(profiles):
        bin_stats = sorted(bin_stats_dict[profile], key=lambda x: x['bin_idx'])
        failure_rates = [stat['failure_rate'] * 100 for stat in bin_stats]
        color = profile_colors[profile]
        
        if len(profiles) > 1:
            x_positions = bin_indices_array + (i - (len(profiles) - 1) / 2) * bar_width
        else:
            x_positions = bin_indices_array
        
        if plot_type == 'analog_only' and bin_stats_fp is not None:
            # Stack on top of FP
            bars = ax.bar(x_positions, failure_rates, bottom=failure_rates_fp,
                   width=bar_width, alpha=0.8, color=color, edgecolor='black', label=profile)
            # Add labels on top of stacked bars
            for j, (bar, rate, fp_rate) in enumerate(zip(bars, failure_rates, failure_rates_fp)):
                total_height = fp_rate + rate
                ax.text(bar.get_x() + bar.get_width() / 2., total_height,
                       f'{rate:.2f}%', ha='center', va='bottom', fontsize=8)
        else:
            bars = ax.bar(x_positions, failure_rates, 
                   width=bar_width, alpha=0.8, color=color, edgecolor='black', label=profile)
            # Add labels on top of bars
            for bar, rate in zip(bars, failure_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{rate:.2f}%', ha='center', va='bottom', fontsize=8)
        all_bars.append(bars)
    
    ax.set_xlabel('Curvature Bin (sorted by curvature)', fontsize=12)
    ax.set_ylabel('Analog-Only Failure Rate (%)', fontsize=12)
    title = f'Analog-Only Failure Rate vs Curvature Bin\n{model_name} - {dataset_name} ({split})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved profile comparison plot to: {save_path}")
    plt.close()


def compute_correlation(curvature_dict, failing_indices, name=""):
    """Compute Spearman correlation between curvature and failure.
    
    Args:
        curvature_dict: Dictionary mapping sample index to curvature
        failing_indices: Set of failing sample indices
        name: Name for the correlation (e.g., "FP" or "Analog")
        
    Returns:
        correlation: Spearman correlation coefficient
        p_value: P-value for the correlation
    """
    common_indices = list(curvature_dict.keys())
    curvatures = np.array([curvature_dict[idx] for idx in common_indices])
    failures = np.array([1 if idx in failing_indices else 0 for idx in common_indices])
    
    correlation, p_value = stats.spearmanr(curvatures, failures)
    
    print(f"\nSpearman Correlation ({name}):")
    print(f"  Correlation coefficient: {correlation:.6f}")
    print(f"  P-value: {p_value:.6e}")
    
    return correlation, p_value


def save_statistics(bin_stats_fp, bin_stats_analog, bin_stats_analog_only, failure_stats, 
                    correlation_fp, p_value_fp, correlation_analog, p_value_analog,
                    correlation_analog_only, p_value_analog_only, save_path, profile=None):
    """Save all statistics to a text file.
    
    Args:
        bin_stats_fp: List of bin statistics for FP failures
        bin_stats_analog: List of bin statistics for Analog failures
        bin_stats_analog_only: List of bin statistics for Analog-only failures
        failure_stats: Dictionary with overall failure statistics
        correlation_fp: Spearman correlation coefficient for FP
        p_value_fp: P-value for FP correlation
        correlation_analog: Spearman correlation coefficient for Analog
        p_value_analog: P-value for Analog correlation
        correlation_analog_only: Spearman correlation coefficient for Analog-only
        p_value_analog_only: P-value for Analog-only correlation
        save_path: Path to save the statistics file
        profile: Profile name (optional, for header)
    """
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CURVATURE-FAILURE STATISTICAL ANALYSIS\n")
        if profile:
            f.write(f"Profile: {profile}\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {failure_stats['total_samples']}\n")
        f.write(f"FP failures: {failure_stats['fp_failures']}\n")
        f.write(f"FP accuracy: {failure_stats['fp_accuracy']:.2f}%\n")
        f.write(f"Analog failures: {failure_stats['analog_failures']}\n")
        f.write(f"Analog accuracy: {failure_stats['analog_accuracy']:.2f}%\n")
        f.write(f"Analog-specific failures: {failure_stats['analog_only_failures']}\n")
        f.write(f"Overlap (both fail): {failure_stats['overlap']}\n")
        f.write("\n")
        
        f.write("BIN STATISTICS - FP FAILURES\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Bin':>4} {'N Samples':>10} {'N Failures':>12} {'Failure Rate':>14} {'Avg Curvature':>16}\n")
        f.write("-"*80 + "\n")
        for stat in bin_stats_fp:
            f.write(f"{stat['bin_idx']:>4} {stat['n_samples']:>10} {stat['n_failures']:>12} "
                   f"{stat['failure_rate']*100:>13.2f}% {stat['avg_curvature']:>16.6e}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("BIN STATISTICS - ANALOG FAILURES\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Bin':>4} {'N Samples':>10} {'N Failures':>12} {'Failure Rate':>14} {'Avg Curvature':>16}\n")
        f.write("-"*80 + "\n")
        for stat in bin_stats_analog:
            f.write(f"{stat['bin_idx']:>4} {stat['n_samples']:>10} {stat['n_failures']:>12} "
                   f"{stat['failure_rate']*100:>13.2f}% {stat['avg_curvature']:>16.6e}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("BIN STATISTICS - ANALOG-ONLY FAILURES (Analog-FP)\n")
        f.write("Note: Failure rate = analog-only failures / samples that FP got correct\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Bin':>4} {'N Samples':>10} {'N FP Correct':>13} {'N Failures':>12} {'Failure Rate':>14} {'Avg Curvature':>16}\n")
        f.write("-"*80 + "\n")
        for stat in bin_stats_analog_only:
            f.write(f"{stat['bin_idx']:>4} {stat['n_samples']:>10} {stat['n_fp_correct']:>13} {stat['n_failures']:>12} "
                   f"{stat['failure_rate']*100:>13.2f}% {stat['avg_curvature']:>16.6e}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("CORRELATION ANALYSIS - FP FAILURES\n")
        f.write("-"*80 + "\n")
        f.write(f"Spearman Correlation Coefficient: {correlation_fp:.6f}\n")
        f.write(f"P-value: {p_value_fp:.6e}\n")
        if p_value_fp < 0.001:
            f.write("Statistical significance: *** (p < 0.001)\n")
        elif p_value_fp < 0.01:
            f.write("Statistical significance: ** (p < 0.01)\n")
        elif p_value_fp < 0.05:
            f.write("Statistical significance: * (p < 0.05)\n")
        else:
            f.write("Statistical significance: Not significant (p >= 0.05)\n")
        f.write("\n")
        
        f.write("CORRELATION ANALYSIS - ANALOG FAILURES\n")
        f.write("-"*80 + "\n")
        f.write(f"Spearman Correlation Coefficient: {correlation_analog:.6f}\n")
        f.write(f"P-value: {p_value_analog:.6e}\n")
        if p_value_analog < 0.001:
            f.write("Statistical significance: *** (p < 0.001)\n")
        elif p_value_analog < 0.01:
            f.write("Statistical significance: ** (p < 0.01)\n")
        elif p_value_analog < 0.05:
            f.write("Statistical significance: * (p < 0.05)\n")
        else:
            f.write("Statistical significance: Not significant (p >= 0.05)\n")
        f.write("\n")
        
        f.write("CORRELATION ANALYSIS - ANALOG-ONLY FAILURES (Analog-FP)\n")
        f.write("-"*80 + "\n")
        f.write(f"Spearman Correlation Coefficient: {correlation_analog_only:.6f}\n")
        f.write(f"P-value: {p_value_analog_only:.6e}\n")
        if p_value_analog_only < 0.001:
            f.write("Statistical significance: *** (p < 0.001)\n")
        elif p_value_analog_only < 0.01:
            f.write("Statistical significance: ** (p < 0.01)\n")
        elif p_value_analog_only < 0.05:
            f.write("Statistical significance: * (p < 0.05)\n")
        else:
            f.write("Statistical significance: Not significant (p >= 0.05)\n")
        f.write("="*80 + "\n")
    
    print(f"Saved statistics to: {save_path}")


def main():
    """Main function to perform statistical analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze relationship between curvature and analog failures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., resnet18, resnet34, resnet50)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet', 'cifar10', 'cifar100', 'tinyimagenet'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to analyze (default: val)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of curvature bins (default: 10 for deciles)')
    
    parser.add_argument('--scores_dir', type=str, default=None,
                        help='Directory containing curvature files (default: aihwkit_experiments/scores/)')
    parser.add_argument('--failing_indices_dir', type=str, default=None,
                        help='Directory containing failing indices (default: auto-detect from analog_inference2 results)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: aihwkit_experiments/results/basic_stats2/)')
    parser.add_argument('--profiles', nargs='+', type=str, default=None,
                        choices=['ideal', 'only_ADC', 'only_wnoise', 'only_PCM', 
                                'ADC_and_wnoise', 'ADC_and_PCM', 'wnoise_and_PCM', 'ADC_wnoise_and_PCM'],
                        help='Profiles to analyze (default: all profiles)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CURVATURE-FAILURE STATISTICAL ANALYSIS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Seed: {args.seed}")
    print(f"Number of bins: {args.n_bins}")
    print("="*80 + "\n")
    
    if args.scores_dir is None:
        args.scores_dir = os.path.join(os.path.dirname(__file__), '..', 'scores')
    
    if args.failing_indices_dir is None:
        # Look in analog_inference2 results directory
        analog_inference_base = os.path.join(os.path.dirname(__file__), '..', 'results', 'analog_inference2', args.dataset.lower(), args.model.lower())
        if os.path.exists(analog_inference_base):
            args.failing_indices_dir = analog_inference_base
            print(f"Auto-detected failing indices directory: {args.failing_indices_dir}\n")
        else:
            raise FileNotFoundError(f"No analog inference results found at: {analog_inference_base}")
    
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'basic_stats2', args.dataset, args.model)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine which profiles to analyze
    if args.profiles is None:
        profiles = ['ideal', 'only_ADC', 'only_wnoise', 'only_PCM', 
                   'ADC_and_wnoise', 'ADC_and_PCM', 'wnoise_and_PCM', 'ADC_wnoise_and_PCM']
    else:
        profiles = args.profiles
    
    print(f"Profiles to analyze: {profiles}\n")
    
    print(f"Loading data...")
    curvature_dict = load_curvature(args.scores_dir, args.model, args.dataset, args.split)
    
    # Load FP failures once (same for all profiles)
    fp_failures = load_failing_indices(args.failing_indices_dir, args.split, fp=True)
    
    print(f"\nCreating {args.n_bins} curvature bins...")
    bins = create_bins(curvature_dict, n_bins=args.n_bins)
    
    # Compute FP bin statistics once
    print(f"\nComputing bin statistics for FP failures...")
    bin_stats_fp = compute_bin_statistics(bins, fp_failures)
    
    # Analyze each profile
    for profile in profiles:
        print(f"\n{'='*80}")
        print(f"ANALYZING PROFILE: {profile}")
        print(f"{'='*80}\n")
        
        try:
            analog_failures = load_failing_indices(args.failing_indices_dir, args.split, fp=False, profile=profile)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping profile {profile}\n")
            continue
        
        failure_stats = compute_failure_statistics(analog_failures, fp_failures, len(curvature_dict))
        analog_only_failures = analog_failures - fp_failures
        
        print(f"\nComputing bin statistics for Analog failures ({profile})...")
        bin_stats_analog = compute_bin_statistics(bins, analog_failures)
        
        print(f"\nComputing bin statistics for Analog-only failures ({profile})...")
        bin_stats_analog_only = compute_bin_statistics_analog_only(bins, analog_only_failures, fp_failures)
        
        print(f"\nBin Statistics - Analog Failures ({profile}):")
        print(f"{'Bin':>4} {'N Samples':>10} {'N Failures':>12} {'Failure Rate':>14} {'Avg Curvature':>16}")
        print("-"*70)
        for stat in bin_stats_analog:
            print(f"{stat['bin_idx']:>4} {stat['n_samples']:>10} {stat['n_failures']:>12} "
                  f"{stat['failure_rate']*100:>13.2f}% {stat['avg_curvature']:>16.6e}")
        
        print(f"\nBin Statistics - Analog-only Failures ({profile}):")
        print(f"{'Bin':>4} {'N Samples':>10} {'N FP Correct':>13} {'N Failures':>12} {'Failure Rate':>14} {'Avg Curvature':>16}")
        print("-"*85)
        for stat in bin_stats_analog_only:
            print(f"{stat['bin_idx']:>4} {stat['n_samples']:>10} {stat['n_fp_correct']:>13} {stat['n_failures']:>12} "
                  f"{stat['failure_rate']*100:>13.2f}% {stat['avg_curvature']:>16.6e}")
        
        correlation_fp, p_value_fp = compute_correlation(curvature_dict, fp_failures, "FP")
        correlation_analog, p_value_analog = compute_correlation(curvature_dict, analog_failures, f"Analog ({profile})")
        correlation_analog_only, p_value_analog_only = compute_correlation(curvature_dict, analog_only_failures, f"Analog-Only ({profile})")
        
        stats_file_path = os.path.join(args.save_dir, f'statistics_{profile}_{args.split}_seed{args.seed}.txt')
        save_statistics(bin_stats_fp, bin_stats_analog, bin_stats_analog_only, failure_stats, 
                       correlation_fp, p_value_fp, correlation_analog, p_value_analog,
                       correlation_analog_only, p_value_analog_only, stats_file_path, profile=profile)
        
        # Generate analog-only failure rate plot for this profile
        analog_only_plot_path = os.path.join(args.save_dir, f'failure_rate_vs_curvature_analog_only_{profile}_{args.split}_seed{args.seed}.png')
        plot_failure_rate_vs_curvature(bin_stats_analog_only, args.model, args.dataset, args.split, analog_only_plot_path, f"Analog-Only Failures ({profile})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

