import torch
import numpy as np
import time
import os
import pickle
import csv
from utils.training_utils import accuracy
from utils.training_utils import AverageMeter


@torch.no_grad()
def validate_model(data_loader, model, device=None, print_freq=100):
    """Validate model and return accuracy"""
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Batch [{i}/{len(data_loader)}] - '
                  f'Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}, '
                  f'Time: {batch_time.avg:.2f}s/batch')

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}, Avg Time: {batch_time.avg:.2f}s/batch\n')

    return top1.avg


@torch.no_grad()
def validate_model_with_failed_indices_indexed(data_loader, model, device=None, print_freq=100):
    """Validate model using indexed dataloader and return accuracy along with indices of failed samples"""
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    
    failed_indices = []
    total_samples = 0

    end = time.time()
    for i, (indices, images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Record failed samples with their actual indices
        _, predicted = output.max(1)
        failed_mask = predicted != target
        if failed_mask.any():
            # Get indices of failed samples in this batch
            batch_failed_indices = torch.where(failed_mask)[0]
            for failed_idx in batch_failed_indices:
                # Get the actual dataset index
                actual_idx = indices[failed_idx].item()
                failed_indices.append(actual_idx)

        total_samples += images.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Batch [{i}/{len(data_loader)}] - '
                  f'Processed: {total_samples} samples, '
                  f'Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}')

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n')

    return top1.avg, failed_indices


def save_failed_indices(failed_indices, index_dir, filename):
    """Save failed indices to a file"""
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, filename), 'wb') as f:
        pickle.dump(failed_indices, f)
    print(f"Failed indices saved to: {os.path.join(index_dir, filename)}")


def print_results_table(results_data):
    """Print results in a formatted table"""
    print("\n" + "="*180)
    print("QUANTIZATION RESULTS TABLE")
    print("="*180)
    
    # Header
    header = f"{'Bit Width':<12} {'Train Acc':<10} {'Train Drop':<12} {'Train Failed':<13} {'Test Acc':<10} {'Test Drop':<10} {'BRECQ Train':<12} {'BRECQ Drop':<12} {'BRECQ Failed':<13} {'BRECQ Test':<12} {'BRECQ Test Drop':<12}"
    print(header)
    print("-" * 180)
    
    # Data rows
    for bit_width, data in results_data.items():
        # Handle BRECQ data (might be None)
        brecq_train = f"{data['train_acc_brecq']:.2f}" if data['train_acc_brecq'] is not None else "N/A"
        brecq_drop = f"{data['train_drop_brecq']:.2f}" if data['train_drop_brecq'] is not None else "N/A"
        brecq_failed = str(data['train_failed_brecq']) if data['train_failed_brecq'] is not None else "N/A"
        brecq_test = f"{data['test_acc_brecq']:.2f}" if data['test_acc_brecq'] is not None else "N/A"
        brecq_test_drop = f"{data['test_drop_brecq']:.2f}" if data['test_drop_brecq'] is not None else "N/A"
        
        row = f"{bit_width:<12} {data['train_acc']:<10.2f} {data['train_drop']:<12.2f} {data['train_failed']:<13} {data['test_acc']:<10.2f} {data['test_drop']:<10.2f} {brecq_train:<12} {brecq_drop:<12} {brecq_failed:<13} {brecq_test:<12} {brecq_test_drop:<12}"
        print(row)
    
    print("="*180)


def save_results_to_csv(results_data, csv_path):
    """Save results to CSV file"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Bit_Width', 'Train_Acc', 'Train_Drop', 'Train_Failed', 'Test_Acc', 'Test_Drop', 
                     'Train_Acc (BRECQ)', 'Train_Drop (BRECQ)', 'Train_Failed (BRECQ)', 'Test_Acc (BRECQ)', 'Test_Drop (BRECQ)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for bit_width, data in results_data.items():
            writer.writerow({
                'Bit_Width': bit_width,
                'Train_Acc': f"{data['train_acc']:.2f}",
                'Train_Drop': f"{data['train_drop']:.2f}",
                'Train_Failed': data['train_failed'],
                'Test_Acc': f"{data['test_acc']:.2f}",
                'Test_Drop': f"{data['test_drop']:.2f}",
                'Train_Acc (BRECQ)': f"{data['train_acc_brecq']:.2f}" if data['train_acc_brecq'] is not None else "N/A",
                'Train_Drop (BRECQ)': f"{data['train_drop_brecq']:.2f}" if data['train_drop_brecq'] is not None else "N/A",
                'Train_Failed (BRECQ)': data['train_failed_brecq'] if data['train_failed_brecq'] is not None else "N/A",
                'Test_Acc (BRECQ)': f"{data['test_acc_brecq']:.2f}" if data['test_acc_brecq'] is not None else "N/A",
                'Test_Drop (BRECQ)': f"{data['test_drop_brecq']:.2f}" if data['test_drop_brecq'] is not None else "N/A"
            })
    
    print(f"Results saved to CSV: {csv_path}")