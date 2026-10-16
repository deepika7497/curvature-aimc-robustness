# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Generic inference on ImageNet and CIFAR-10/100 datasets with analog models.

Compare original PyTorch model vs analog converted model with different non-ideality profiles.
Supports loading from existing checkpoints and multiple model architectures.
"""
# pylint: disable=invalid-name

import os
import sys
import argparse
import torch
import torch.nn as nn
import pickle
import glob
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.data_utils import (
    build_imagenet_data,
    build_cifar10_data,
    build_cifar100_data,
    build_tinyimagenet_data
)
from utils.eval_utils import validate_model_with_failed_indices_indexed, save_failed_indices
from utils.model_utils import (
    create_cifar10_model,
    create_cifar100_model,
    create_tinyimagenet_model,
    load_checkpoint,
)
from utils.generic_data_loader import IndexedDataset
from utils.training_utils import seed_all
import torchvision.models as models
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)
from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.utils import WeightRemapType, WeightClipType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.inference.calibration import (
    calibrate_input_ranges, InputRangeCalibrationType
)

# --- Global ADC settings (used for all profiles with ADC) ---
ADC_BITS = 9                 # single resolution everywhere
ADC_OUT_BOUND = 12.0         # match typical analog out_bound (from ideal runs)


def get_rpu_config(profile: list[bool] = [True, True, True]):
    """Get RPU configuration for analog inference."""
    cfg = InferenceRPUConfig()

    # Keep mapping/clip/remap fixed across profiles
    cfg.mapping = MappingParameter(
        max_input_size=512,
        max_output_size=512,
        digital_bias=True,
        weight_scaling_omega=0.6,
    )
    cfg.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
    cfg.clip.type = WeightClipType.LAYER_GAUSSIAN
    cfg.clip.sigma = 2.5

    # Start from "perfect" forward and then add non-idealities
    fwd = cfg.forward
    fwd.is_perfect = False
    fwd.out_noise = 0.0
    fwd.inp_noise = 0.0
    fwd.out_res = -1.0      # no ADC
    fwd.inp_res = -1.0      # no DAC
    fwd.w_noise = 0.0
    fwd.w_noise_type = WeightNoiseType.NONE
    
    # Default: no drift, no PCM noise
    cfg.noise_model = PCMLikeNoiseModel(
        prog_noise_scale=0.0,
        read_noise_scale=0.0,
        drift_scale=0.0,
    )
    cfg.drift_compensation = None

    # Enable input-range calibration for PTQ
    cfg.pre_post.input_range.enable = True
    cfg.pre_post.input_range.learn_input_range = False
    cfg.pre_post.input_range.init_from_data = 0
    cfg.forward.noise_management = NoiseManagementType.NONE
    cfg.forward.bound_management = BoundManagementType.NONE

    if profile[0]:
        # ADC quantization + bound, no weight/device noise
        fwd.out_bound = ADC_OUT_BOUND
        fwd.out_res = 1.0 / (2**ADC_BITS)
        cfg.forward.noise_management = NoiseManagementType.ABS_MAX
        cfg.forward.bound_management = BoundManagementType.NONE

    if profile[1]:
        # Only short-term output-referred weight noise
        fwd.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        fwd.w_noise = 0.02

    if profile[2]:
        # Only PCM program+drift/read noise, no MAC/ADC noise
        cfg.noise_model = PCMLikeNoiseModel(
            g_max=25.0,
            prog_noise_scale=1.0,
            read_noise_scale=1.0,
            drift_scale=1.0
        )

    return cfg


def get_checkpoint_path(model_name, dataset_name, seed=42):
    """Get checkpoint path for a given model and dataset."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', dataset_name.lower())
    pattern = f'{model_name}_seed{seed}_*'
    saved_runs = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(saved_runs) == 0:
        raise FileNotFoundError(f"No saved runs found for {model_name} on {dataset_name} with seed {seed}")
    else:
        saved_runs.sort()
        latest_run = saved_runs[-1]
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


def load_model(model_name, dataset_name, checkpoint_path=None, device='cpu'):
    """Load model - ImageNet from PyTorch, others from checkpoints."""
    # ImageNet: use pretrained PyTorch weights
    if dataset_name == 'imagenet':
        print(f"Loading pretrained ImageNet weights from torchvision for {model_name}...")
        if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            model_fn = getattr(models, model_name)
            try:
                weights = 'IMAGENET1K_V1'
                model = model_fn(weights=weights)
            except TypeError:
                model = model_fn(pretrained=True)
        elif model_name == 'mobilenetv2':
            try:
                model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            except TypeError:
                model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f"Unsupported ImageNet model: {model_name}. "
                           f"Supported: resnet18, resnet34, resnet50, resnet101, resnet152, mobilenetv2")
    
    # Other datasets: load from checkpoint
    else:
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(model_name, dataset_name)
            print(f"Auto-detected checkpoint: {checkpoint_path}")
        
        if dataset_name == 'cifar10':
            model = create_cifar10_model(model_name)
        elif dataset_name == 'cifar100':
            model = create_cifar100_model(model_name)
        elif dataset_name == 'tinyimagenet':
            model = create_tinyimagenet_model(model_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        checkpoint = load_checkpoint(model, checkpoint_path, device=device)
        if checkpoint:
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'best_acc' in checkpoint:
                print(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model


def save_analog_model(analog_model, dataset_name, model_name, profile, save_dir):
    """Save the analog model to the appropriate directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{model_name}_analog_{profile}_{timestamp}.pth'
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        'model_state_dict': analog_model.state_dict(),
        'model_name': model_name,
        'dataset': dataset_name,
        'profile': profile,
        'timestamp': timestamp,
    }, save_path)
    
    print(f"Analog model saved to: {save_path}")
    return save_path


def run_inference(model_name, dataset_name, batch_size=64, workers=4, checkpoint_path=None,
                device='cuda', save_dir=None, use_calibration=False, num_cal_samples=1000, seed=42):
    """Run inference on specified dataset for both original and analog models.
    
    Note: The profile parameter is kept for backward compatibility but all profiles
    will be evaluated (ideal_analog, adc_only, wnoise_only, pcm_only, full).
    """
    
    # Build data loaders
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
    
    # Wrap training dataset with IndexedDataset for tracking failed indices
    train_dataset_indexed = IndexedDataset(train_loader.dataset)
    train_loader_indexed = torch.utils.data.DataLoader(
        train_dataset_indexed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    # Wrap validation dataset with IndexedDataset for tracking failed indices
    val_dataset_indexed = IndexedDataset(val_loader.dataset)
    val_loader_indexed = torch.utils.data.DataLoader(
        val_dataset_indexed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    # Load original model
    print(f"Loading original model...")
    original_model = load_model(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    original_model = original_model.to(device)
    original_model.eval()
    
    print(f"\nEvaluating original model on {dataset_name} TRAINING set...")
    # Use indexed loader to track failing indices for FP model on TRAINING set
    train_acc, fp_train_failing_indices = validate_model_with_failed_indices_indexed(
        train_loader_indexed, original_model, device=device, print_freq=50
    )
    print(f"Full Precision Model Training Accuracy: {train_acc:.2f}%")
    print(f"Number of failed training samples (FP): {len(fp_train_failing_indices)}")
    
    print(f"\nEvaluating original model on {dataset_name} VALIDATION set...")
    # Use indexed loader to track failing indices for FP model on VALIDATION set
    val_acc, fp_val_failing_indices = validate_model_with_failed_indices_indexed(
        val_loader_indexed, original_model, device=device, print_freq=50
    )
    print(f"Full Precision Model Validation Accuracy: {val_acc:.2f}%")
    print(f"Number of failed validation samples (FP): {len(fp_val_failing_indices)}")

    # Set up the results directory for saving failing indices
    if save_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'analog_inference2')
        results_dir = os.path.join(base_dir, dataset_name.lower(), model_name.lower())
    else:
        results_dir = os.path.join(save_dir, dataset_name.lower(), model_name.lower())
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults directory: {results_dir}")

    # Save full precision model failing indices (both training and validation)
    print(f"\nSaving full precision model failing indices...")
    
    # Save training failing indices
    fp_train_failed_indices_file = 'failed_indices_train_fp.pkl'
    save_failed_indices(fp_train_failing_indices, results_dir, fp_train_failed_indices_file)
    print(f"  Training FP failing indices saved ({len(fp_train_failing_indices)} samples)")

    fp_val_failed_indices_file = 'failed_indices_val_fp.pkl'
    save_failed_indices(fp_val_failing_indices, results_dir, fp_val_failed_indices_file)
    print(f"  Validation FP failing indices saved ({len(fp_val_failing_indices)} samples)")
    
    # Handle CUDA support for analog model
    if not cuda.is_compiled():
        print("WARNING: AIHWKIT was built without CUDA support. Keeping analog model on CPU.")
        analog_device = "cpu"
    else:
        analog_device = device

    # Load model for analog conversion
    baseline_model = load_model(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Initialize results storage
    all_results = {}
    profile_summary = []  # Store profile name, config, train_acc, val_acc for each profile
    
    # Evaluate each profile separately
    profiles = {
        'ideal': [False, False, False],
        'only_ADC': [True, False, False],
        'only_wnoise': [False, True, False],
        'only_PCM': [False, False, True],
        'ADC_and_wnoise': [True, True, False],
        'ADC_and_PCM': [True, False, True],
        'wnoise_and_PCM': [False, True, True],
        'ADC_wnoise_and_PCM': [True, True, True]
    }
    
    for profile_name, profile_config in profiles.items():
        print(f"\n{'='*80}")
        print(f"Evaluating profile: {profile_name}")
        print(f"{'='*80}\n")
        
        # Convert to analog with this profile
        rpu_config = get_rpu_config(profile=profile_config)
        print("\nPretty-print of non-default settings:\n")
        print(rpu_config)

        analog_model = convert_to_analog(baseline_model, rpu_config)
        analog_model.eval()
        analog_model.program_analog_weights()
        print("\nC++ RPUCudaTile information:\n")
        print(next(analog_model.analog_tiles()).tile)

        analog_model = analog_model.to(analog_device)
        print(f"Analog model device: {analog_device}")
        
        # Save the analog model
        save_analog_model(analog_model, dataset_name, model_name, profile_name, results_dir)
        
        # Evaluate analog model
        print(f"\nEvaluating analog model on {dataset_name} with profile {profile_name}...")
        if not cuda.is_compiled():
            print("WARNING: Analog inference on CPU will be VERY SLOW!")

        all_train_failed_indices = {}
        all_val_failed_indices = {}

        # Evaluate on TRAINING set
        print(f"  Evaluating on training set...")
        train_analog_acc, train_failed_indices = validate_model_with_failed_indices_indexed(
            train_loader_indexed, analog_model, device=analog_device, print_freq=50
        )
        train_analog_acc_val = train_analog_acc.item()
        all_train_failed_indices['train'] = train_failed_indices
        
        # Save training failed indices
        train_failed_indices_file = f'failed_indices_train_{profile_name}.pkl'
        save_failed_indices(train_failed_indices, results_dir, train_failed_indices_file)
        
        print(f"  Training Analog Model Accuracy: {train_analog_acc:.2f}%")
        print(f"  Number of failed training samples: {len(train_failed_indices)}")
        
        # Evaluate on VALIDATION set
        val_analog_acc, val_failed_indices = validate_model_with_failed_indices_indexed(
            val_loader_indexed, analog_model, device=analog_device, print_freq=50
        )
        val_analog_acc_val = val_analog_acc.item()
        all_val_failed_indices['val'] = val_failed_indices
        
        val_failed_indices_file = f'failed_indices_val_{profile_name}.pkl'
        save_failed_indices(val_failed_indices, results_dir, val_failed_indices_file)
        
        print(f"  Validation Analog Model Accuracy: {val_analog_acc:.2f}%")
        print(f"  Number of failed validation samples: {len(val_failed_indices)}")
        
        # Add FP failing indices to the dictionaries
        all_train_failed_indices['fp'] = fp_train_failing_indices
        all_val_failed_indices['fp'] = fp_val_failing_indices
        
        # Save all TRAINING failed indices together for this profile
        all_train_failed_file = os.path.join(results_dir, f'all_failed_indices_train_{profile_name}.pkl')
        with open(all_train_failed_file, 'wb') as f:
            pickle.dump(all_train_failed_indices, f)
        print(f"\nAll training failed indices for profile {profile_name} saved to: {all_train_failed_file}")
        
        # Save all VALIDATION failed indices together for this profile
        all_val_failed_file = os.path.join(results_dir, f'all_failed_indices_val_{profile_name}.pkl')
        with open(all_val_failed_file, 'wb') as f:
            pickle.dump(all_val_failed_indices, f)
        print(f"All validation failed indices for profile {profile_name} saved to: {all_val_failed_file}")
        
        # Store results for this profile
        all_results[profile_name] = {
            'analog_train_acc': train_analog_acc_val,
            'analog_val_acc': val_analog_acc_val,
            'train_failed_indices': all_train_failed_indices,
            'val_failed_indices': all_val_failed_indices,
        }
        
        # Store summary information
        profile_summary.append({
            'profile_name': profile_name,
            'config': profile_config,
            'train_acc': train_analog_acc_val,
            'val_acc': val_analog_acc_val,
        })
        
        # Print results for this profile
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {dataset_name.upper()} - {model_name.upper()} (profile: {profile_name})")
        print(f"{'='*80}")
        print(f"Original Model TRAINING Accuracy: {train_acc:.2f}%")
        print(f"  Failed training samples (FP): {len(fp_train_failing_indices)}")
        print(f"\nOriginal Model VALIDATION Accuracy: {val_acc:.2f}%")
        print(f"  Failed validation samples (FP): {len(fp_val_failing_indices)}")
        print(f"\nAnalog Model TRAINING Accuracy:")
        print(f"  {train_analog_acc_val:.2f}% (drop: {train_acc.item() - train_analog_acc_val:.2f}%)")
        print(f"         Failed training samples: {len(all_train_failed_indices['train'])}")
        print(f"\nAnalog Model VALIDATION Accuracy:")
        print(f"  {val_analog_acc_val:.2f}% (drop: {val_acc.item() - val_analog_acc_val:.2f}%)")
        print(f"         Failed validation samples: {len(all_val_failed_indices['val'])}")
        print(f"{'='*80}\n")
    
    # Print summary of all profiles
    print(f"\n{'='*80}")
    print(f"PROFILE SUMMARY - {dataset_name.upper()} - {model_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Profile Name':<25} {'Config (ADC/Wnoise/PCM)':<30} {'Train Acc (%)':<15} {'Val Acc (%)':<15}")
    print(f"{'-'*80}")
    for summary in profile_summary:
        config_str = f"[{int(summary['config'][0])}, {int(summary['config'][1])}, {int(summary['config'][2])}]"
        print(f"{summary['profile_name']:<25} {config_str:<30} {summary['train_acc']:<15.2f} {summary['val_acc']:<15.2f}")
    print(f"{'='*80}\n")
    
    # Save profile summary to file
    summary_file = os.path.join(results_dir, f'profile_summary_{model_name}_{dataset_name}.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'model_name': model_name,
            'dataset': dataset_name,
            'fp_train_acc': train_acc.item(),
            'fp_val_acc': val_acc.item(),
            'profiles': profile_summary
        }, f)
    print(f"Profile summary saved to: {summary_file}")
    
    # Also save as human-readable text file
    summary_txt_file = os.path.join(results_dir, f'profile_summary_{model_name}_{dataset_name}.txt')
    with open(summary_txt_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"PROFILE SUMMARY - {dataset_name.upper()} - {model_name.upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Full Precision Model:\n")
        f.write(f"  Training Accuracy: {train_acc.item():.2f}%\n")
        f.write(f"  Validation Accuracy: {val_acc.item():.2f}%\n\n")
        f.write(f"{'Profile Name':<25} {'Config (ADC/Wnoise/PCM)':<30} {'Train Acc (%)':<15} {'Val Acc (%)':<15} {'Train Drop':<15} {'Val Drop':<15}\n")
        f.write(f"{'-'*100}\n")
        for summary in profile_summary:
            config_str = f"[{int(summary['config'][0])}, {int(summary['config'][1])}, {int(summary['config'][2])}]"
            train_drop = train_acc.item() - summary['train_acc']
            val_drop = val_acc.item() - summary['val_acc']
            f.write(f"{summary['profile_name']:<25} {config_str:<30} {summary['train_acc']:<15.2f} {summary['val_acc']:<15.2f} {train_drop:<15.2f} {val_drop:<15.2f}\n")
        f.write(f"{'='*80}\n")
    print(f"Profile summary (text) saved to: {summary_txt_file}")
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        # Training set results
        'train_acc': train_acc.item(),
        'fp_train_failing_indices': fp_train_failing_indices,
        # Validation set results
        'val_acc': val_acc.item(),
        'fp_val_failing_indices': fp_val_failing_indices,
        'results_dir': results_dir,
        'all_results': all_results,
        'use_calibration': use_calibration,
    }


def main():
    """Main function to run inference on multiple datasets."""
    parser = argparse.ArgumentParser(
        description='Generic Model Inference: Original vs Analog with Non-Ideality Profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture (e.g., resnet18, mobilenetv2, resnet50)')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10', 'cifar100', 'tinyimagenet'],
                        help='Dataset to run inference on')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to existing checkpoint file (auto-detected if not provided)')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    
    # PTQ calibration
    parser.add_argument('--use_calibration', action='store_true',
                        help='Use PTQ calibration (default: disabled)')
    parser.add_argument('--num_cal_samples', type=int, default=1000,
                        help='Number of samples to use for PTQ calibration (default: 1000)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom directory to save analog model and results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    elif args.dataset == 'imagenet':
        print(f"  Checkpoint: None (will use pretrained PyTorch/torchvision weights)")
    else:
        print(f"  Checkpoint: Auto-detect from checkpoints directory")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PTQ Calibration: {'Disabled' if not args.use_calibration else f'Enabled ({args.num_cal_samples} samples)'}")
    
    result = run_inference(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        device=args.device,
        save_dir=args.save_dir,
        use_calibration=args.use_calibration,
        num_cal_samples=args.num_cal_samples,
        seed=args.seed
    )
    
    
if __name__ == "__main__":
    main()
