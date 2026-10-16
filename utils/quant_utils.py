import torch
import numpy as np
import libraries.BRECQ.linklink as link
import time
from utils.training_utils import accuracy
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


@torch.no_grad()
def validate_model_distributed(test_loader, ann, print_freq=100):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    ann.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_iter = len(test_loader)
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = ann(input)

            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))

            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Test: [{i + 1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    top1_sum = torch.Tensor([top1.avg * top1.count]).cuda()
    top5_sum = torch.Tensor([top5.avg * top5.count]).cuda()
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item() / total_num.item()
    final_top1 = top1_sum.item() / total_num.item()
    final_top5 = top5_sum.item() / total_num.item()

    print(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
          Loss {final_loss:.3f}\ttotal_num={total_num.item()}\t')

    return final_top1


def get_calibration_data(
    data_loader: DataLoader, 
    num_samples: int = 1024,
    indexed: bool = False
    ) -> torch.Tensor:
    """Extract calibration data from a data loader."""
    
    calibration_data = []
    total_samples = 0
    
    for batch in data_loader:
        if indexed:
            _, data, _ = batch
        else:
            data, _ = batch
            
        if total_samples >= num_samples:
            break
        
        batch_size = data.size(0)
        samples_needed = min(batch_size, num_samples - total_samples)
        
        calibration_data.append(data[:samples_needed])
        total_samples += samples_needed
    
    return torch.cat(calibration_data, dim=0)

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
                  f'Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}')

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n')

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


def add_quantization_args(parser):
    """Add quantization specific arguments to the parser"""
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')
    parser.add_argument('--target_layer_name', type=str, default=None, 
                        help='name of specific layer to quantize (if None, quantize all layers)')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    return parser


def add_dist_args(parser):
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    return parser