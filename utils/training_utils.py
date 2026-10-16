"""
Training utilities for model training and evaluation.
Provides common training functions, argument parsers, and evaluation metrics.
"""

import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import torch.distributed as dist
from typing import Tuple, Optional


def seed_all(seed=1029):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank=0, debug=False):
    """Train for one epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 40 == 0 and rank == 0:
            progress.display(i)

    return losses.avg, top1.avg


def validate(model, val_loader, criterion, device, rank=0, debug=False):
    """Validate the model"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 40 == 0 and rank == 0:
                progress.display(i)

        if rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


#########################Argument Parsing#########################
def create_common_parser(description='BRECQ Quantization'):
    """Create a common argument parser with shared parameters"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default=None, type=str, help='path to dataset')
    parser.add_argument('--checkpoint_pth', type=str, default='', help='path to trained model checkpoint')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet',
                                'resnet18_cifar', 'resnet50_cifar', 'mobilenetv2_cifar',
                                'resnet18_tinyimagenet', 'resnet50_tinyimagenet', 'mobilenetv2_tinyimagenet'])

    return parser


def add_training_args(parser):
    """Add training specific arguments"""
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--save_path', default='./experiments/results/checkpoints', type=str, help='path to save checkpoints')
    parser.add_argument('--evaluate_only', default=False, type=bool, help='only evaluate the model')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='use distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--rank', default=0, type=int, help='rank of distributed process')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')

    return parser


def add_curriculum_args(parser):
    """Add curriculum sampling arguments"""
    parser.add_argument('--index_file', type=str, default=None, help='path to index file')
    parser.add_argument('--curr_strategy', type=str, default='default', choices=['default', 'progressive_favor', 'progressive_against', 'switch_favor', 'cyclic', 'exponential', 'curvature_adaptive'], help='curriculum sampling strategy')
    parser.add_argument('--curr_switch_epoch', type=int, default=None, help='epoch at which to switch curriculum strategy')
    parser.add_argument('--curr_bias_strength', type=float, default=1.0, help='strength of the curriculum bias')
    parser.add_argument('--curr_min_weight', type=float, default=0.1, help='minimum weight for any sample')
    parser.add_argument('--curr_max_weight', type=float, default=10.0, help='maximum weight for any sample')
    return parser

