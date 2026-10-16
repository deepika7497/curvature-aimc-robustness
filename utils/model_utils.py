"""
Model utilities for model creation and management.
Provides functions for creating models and managing checkpoints.
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Optional, Dict, Any


def create_cifar100_model(model_name: str) -> nn.Module:
    """Create a CIFAR-100 model without pretrained weights"""
    # Use CIFAR-optimized models from models/ directory
    try:
        from models import get_model, list_models
        
        # Map to CIFAR-specific model names
        if model_name in ['resnet18', 'resnet18_cifar']:
            return get_model('resnet18_cifar', num_classes=100)
        elif model_name in ['resnet50', 'resnet50_cifar']:
            return get_model('resnet50_cifar', num_classes=100)
        elif model_name in ['mobilenetv2', 'mobilenetv2_cifar']:
            return get_model('mobilenetv2_cifar', num_classes=100)
        else:
            raise ValueError(f"Unsupported CIFAR-100 model: {model_name}")
    except ImportError:
        raise ValueError(f"Could not import models. Please ensure the models directory is properly set up.")


def create_cifar10_model(model_name: str) -> nn.Module:
    """Create a CIFAR-10 model without pretrained weights"""
    # Use CIFAR-optimized models from models/ directory
    try:
        from models import get_model, list_models
        
        # Map to CIFAR-specific model names
        if model_name in ['resnet18', 'resnet18_cifar']:
            return get_model('resnet18_cifar', num_classes=10)
        elif model_name in ['resnet50', 'resnet50_cifar']:
            return get_model('resnet50_cifar', num_classes=10)
        elif model_name in ['mobilenetv2', 'mobilenetv2_cifar']:
            return get_model('mobilenetv2_cifar', num_classes=10)
        else:
            raise ValueError(f"Unsupported CIFAR-10 model: {model_name}")
    except ImportError:
        raise ValueError(f"Could not import models. Please ensure the models directory is properly set up.")


def create_imagenet_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Create an ImageNet model"""
    if pretrained:
        # Use BRECQ hubconf for ImageNet pretrained weights
        try:
            sys.path.append('libraries/BRECQ')
            import hubconf
            return getattr(hubconf, model_name)(pretrained=True)
        except (ImportError, AttributeError):
            print("Warning: BRECQ pretrained weights not available, using local model")
            pretrained = False
    
    # Use local models from models/ directory
    try:
        from models import get_model, list_models
        
        if model_name in list_models():
            return get_model(model_name, num_classes=1000)
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list_models()}")
    except ImportError:
        # Final fallback to torchvision models
        if model_name == 'resnet18':
            from torchvision.models import resnet18
            return resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            from torchvision.models import resnet50
            return resnet50(pretrained=pretrained)
        elif model_name == 'mobilenetv2':
            from torchvision.models import mobilenet_v2
            return mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def create_tinyimagenet_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Create a TinyImageNet model (200 classes)"""
    # Use local models from models/ directory
    try:
        from models import get_model, list_models
        
        if model_name in list_models():
            return get_model(model_name, num_classes=200)
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list_models()}")
    except ImportError:
        # Final fallback to torchvision models
        if model_name == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=pretrained)
            # Modify the final layer for 200 classes
            model.fc = nn.Linear(model.fc.in_features, 200)
            return model
        elif model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=pretrained)
            # Modify the final layer for 200 classes
            model.fc = nn.Linear(model.fc.in_features, 200)
            return model
        elif model_name == 'mobilenetv2':
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2(pretrained=pretrained)
            # Modify the final layer for 200 classes
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
            return model
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def save_checkpoint(model: nn.Module, optimizer, epoch: int, best_acc: float, 
                   arch: str, save_dir: str, filename: str = None) -> str:
    """Save model checkpoint"""
    if filename is None:
        filename = f'{arch}_best.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'arch': arch
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """Load model checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"=> loaded checkpoint '{checkpoint_path}'")
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return {}


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and non-trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': total_params
    } 