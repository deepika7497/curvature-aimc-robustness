#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Poster motivation figure: FP vs full analog non-ideality (ADC+WNoise+PCM).

Four x-axis groups: ResNet-18 CIFAR-10, ResNet-18 CIFAR-100, ResNet-50 CIFAR-10,
ResNet-50 CIFAR-100. Two bars per group: digital FP32 PyTorch inference (no AIHWKit)
and analog inference with the same profile as ``ADC_wnoise_and_PCM`` in analog_inference2.py
(9-bit ADC + additive weight noise + PCM program/read/drift). Matches paper_figures
hardware assumptions. No extra training.

Use ``--plot_only`` to regenerate the PNG from a saved CSV without running inference.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
sys.path.append(os.path.join(_scripts_dir, '../..'))

from utils.data_utils import build_cifar10_data, build_cifar100_data
from utils.eval_utils import validate_model
from utils.training_utils import seed_all
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.rpu_base import cuda

from analog_inference2 import get_rpu_config, load_model, get_checkpoint_path

# Same profile name / flags as paper_figures.py and analog_inference2 ``ADC_wnoise_and_PCM``
PROFILE_ADC_WNOISE_PCM = [True, True, True]

# (model_name, dataset_name) — order is x-axis left → right
DEFAULT_GROUPS = [
    ('resnet18', 'cifar10'),
    ('resnet18', 'cifar100'),
    ('resnet50', 'cifar10'),
    ('resnet50', 'cifar100'),
]


def build_val_loader(dataset_name, batch_size, workers):
    d = dataset_name.lower()
    if d == 'cifar10':
        _, val_loader = build_cifar10_data(
            data_path='data/CIFAR10',
            input_size=32,
            batch_size=batch_size,
            workers=workers,
        )
    elif d == 'cifar100':
        _, val_loader = build_cifar100_data(
            data_path='data/CIFAR100',
            input_size=32,
            batch_size=batch_size,
            workers=workers,
        )
    else:
        raise ValueError(f'Unsupported dataset for this plot: {dataset_name}')
    return val_loader


def _to_float(x):
    return float(x.item() if torch.is_tensor(x) else x)


def eval_fp_and_analog_profile(
    model_name,
    dataset_name,
    checkpoint_path,
    device,
    batch_size,
    workers,
    print_freq,
):
    val_loader = build_val_loader(dataset_name, batch_size, workers)

    fp_model = load_model(model_name, dataset_name, checkpoint_path, device=device)
    fp_model = fp_model.to(device)
    fp_model.eval()
    fp_acc = _to_float(validate_model(val_loader, fp_model, device=device, print_freq=print_freq))

    if not cuda.is_compiled():
        analog_device = 'cpu'
        print('WARNING: AIHWKIT without CUDA; analog on CPU (slow).')
    else:
        analog_device = device

    base = load_model(model_name, dataset_name, checkpoint_path, device=device)
    rpu_config = get_rpu_config(profile=PROFILE_ADC_WNOISE_PCM)
    analog = convert_to_analog(base, rpu_config)
    analog.eval()
    analog.program_analog_weights()
    analog = analog.to(analog_device)
    analog_acc = _to_float(
        validate_model(val_loader, analog, device=analog_device, print_freq=print_freq)
    )

    return fp_acc, analog_acc


def xtick_label(model_name: str, dataset_name: str) -> str:
    m = 'ResNet-18' if model_name.lower() == 'resnet18' else 'ResNet-50'
    d = 'CIFAR-10' if dataset_name.lower() == 'cifar10' else 'CIFAR-100'
    return f'{m}\n{d}'


POSTER_TITLE_DEFAULT = 'Change in Network Accuracy with hardware noise'


def save_bar_plot(labels, fp_accs, analog_accs, out_png, title=None):
    """Poster-oriented: large type, bar-top %; legend below title."""
    plt.rcParams.update(
        {
            'font.size': 16,
            'font.family': 'sans-serif',
            'axes.labelsize': 20,
            'axes.titlesize': 22,
            'xtick.labelsize': 14,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.linewidth': 1.2,
        }
    )
    n = len(labels)
    x = np.arange(n, dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.patch.set_facecolor('white')

    ax.bar(
        x - width / 2,
        fp_accs,
        width,
        label='Ideal (digital) inference',
        color='#3d5a80',
        edgecolor='#1e293b',
        linewidth=1.0,
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        analog_accs,
        width,
        label='Inference with hardware noise',
        color='#ee6c4d',
        edgecolor='#9f1239',
        linewidth=1.0,
        hatch='//',
        alpha=0.95,
        zorder=2,
    )

    ax.set_ylabel('Inference Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis='y', alpha=0.4, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    finite = [v for v in (fp_accs + analog_accs) if not (isinstance(v, float) and np.isnan(v))]
    mx = float(max(finite)) if finite else 100.0
    # Headroom above tallest bar so %-labels (placed at height + dy) stay inside the axes
    y_top = min(mx + 14.0, 125.0)
    ax.set_ylim(0, y_top)
    ax.set_yticks(np.arange(0, 101, 10))

    use_title = title if title else POSTER_TITLE_DEFAULT
    ax.set_title(use_title, fontweight='bold', pad=36)

    dy = 1.2
    for i in range(n):
        fp = fp_accs[i]
        an = analog_accs[i]
        if np.isnan(fp) or np.isnan(an):
            continue
        ax.text(
            x[i] - width / 2,
            fp + dy,
            f'{fp:.1f}%',
            ha='center',
            va='bottom',
            fontsize=15,
            fontweight='bold',
            color='#0f172a',
        )
        ax.text(
            x[i] + width / 2,
            an + dy,
            f'{an:.1f}%',
            ha='center',
            va='bottom',
            fontsize=15,
            fontweight='bold',
            color='#7c2d12',
        )

    ax.legend(
        bbox_to_anchor=(0.5, 0.99),
        loc='lower center',
        ncol=2,
        framealpha=0.97,
        edgecolor='#cbd5e1',
        columnspacing=1.2,
    )
    fig.subplots_adjust(top=0.82, bottom=0.14)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, facecolor='white', bbox_inches='tight', pad_inches=0.4)
    plt.close()
    print(f'Saved figure: {out_png}')


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'model',
                'dataset',
                'val_acc_fp_pct',
                'val_acc_ADC_WNoise_PCM_pct',
                'degradation_pp',
                'analog_profile',
            ]
        )
        for r in rows:
            w.writerow(r)
    print(f'Saved CSV: {path}')


def load_results_from_csv(path):
    """Rebuild bar-plot inputs from a CSV written by this script."""
    labels = []
    fp_list = []
    analog_list = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row['model'].strip()
            dataset_name = row['dataset'].strip()
            labels.append(xtick_label(model_name, dataset_name))
            fp_s = row.get('val_acc_fp_pct', '').strip()
            an_s = row.get('val_acc_ADC_WNoise_PCM_pct', '').strip()
            if not fp_s or not an_s:
                fp_list.append(float('nan'))
                analog_list.append(float('nan'))
            else:
                fp_list.append(float(fp_s))
                analog_list.append(float(an_s))
    return labels, fp_list, analog_list


def main():
    p = argparse.ArgumentParser(
        description='Poster bars: FP vs ADC+WNoise+PCM (paper profile) for RN18/RN50 × CIFAR-10/100'
    )
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--print_freq', type=int, default=1000)
    p.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='If set, use this checkpoint for every group (debug only)',
    )
    p.add_argument(
        '--skip_missing',
        action='store_true',
        help='Skip groups with no checkpoint instead of failing',
    )
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--title', type=str, default=None)
    p.add_argument(
        '--plot_only',
        action='store_true',
        help='Skip inference; redraw PNG from CSV (--data_csv or default fp_vs_*_seed*.csv in --out_dir)',
    )
    p.add_argument(
        '--data_csv',
        type=str,
        default=None,
        help='CSV for --plot_only (defaults to same path as the script’s output CSV for this seed)',
    )
    args = p.parse_args()

    seed_all(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            _scripts_dir, '..', 'results', 'poster_motivation', 'cifar_bars'
        )
    os.makedirs(args.out_dir, exist_ok=True)
    profile_tag = 'ADC_WNoise_PCM'
    png = os.path.join(args.out_dir, f'fp_vs_{profile_tag}_val_accuracy_seed{args.seed}.png')
    csv_path = os.path.join(args.out_dir, f'fp_vs_{profile_tag}_val_accuracy_seed{args.seed}.csv')

    if args.plot_only:
        csv_in = args.data_csv or csv_path
        if not os.path.isfile(csv_in):
            raise FileNotFoundError(f'--plot_only: CSV not found: {csv_in}')
        print(f'--plot_only: loading {csv_in}')
        labels, fp_list, analog_list = load_results_from_csv(csv_in)
        save_bar_plot(labels, fp_list, analog_list, png, title=args.title)
        print(f'Wrote plot only: {png}')
        return

    labels = []
    fp_list = []
    analog_list = []
    csv_rows = []
    profile_name = 'ADC_wnoise_and_PCM'

    for model_name, dataset_name in DEFAULT_GROUPS:
        label = xtick_label(model_name, dataset_name)
        labels.append(label)

        ckpt = args.checkpoint
        if ckpt is None:
            try:
                ckpt = get_checkpoint_path(model_name, dataset_name, args.seed)
            except FileNotFoundError as e:
                if args.skip_missing:
                    print(f'SKIP {model_name} {dataset_name}: {e}')
                    fp_list.append(float('nan'))
                    analog_list.append(float('nan'))
                    csv_rows.append([model_name, dataset_name, '', '', '', profile_name])
                    continue
                raise

        print(f'=== {model_name} / {dataset_name} ===')
        print(f'Checkpoint: {ckpt}')
        fp_acc, analog_acc = eval_fp_and_analog_profile(
            model_name,
            dataset_name,
            ckpt,
            args.device,
            args.batch_size,
            args.workers,
            args.print_freq,
        )
        drop_pp = fp_acc - analog_acc
        print(f'  Digital: {fp_acc:.2f}%  |  Analog: {analog_acc:.2f}%  |  Drop: {drop_pp:.2f} pp')
        fp_list.append(fp_acc)
        analog_list.append(analog_acc)
        csv_rows.append(
            [
                model_name,
                dataset_name,
                f'{fp_acc:.4f}',
                f'{analog_acc:.4f}',
                f'{drop_pp:.4f}',
                profile_name,
            ]
        )

    save_csv(csv_path, csv_rows)
    save_bar_plot(labels, fp_list, analog_list, png, title=args.title)


if __name__ == '__main__':
    main()
