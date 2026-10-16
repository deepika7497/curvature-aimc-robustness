"""
Models package for Robust Dataset Project.
Contains production-ready model implementations from BRECQ.
"""

# Import all model functions
from .resnet import (
    ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
)

from .mobilenetv2 import MobileNetV2, mobilenetv2

from .mnasnet import MNASNet, mnasnet

from .regnet import (
    RegNet, regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m
)

# Import CIFAR-optimized models
from .cifar_models import (
    ResNet18_CIFAR, ResNet50_CIFAR, MobileNetV2_CIFAR
)

# Import TinyImageNet-optimized models
from .tiny_imagenet_models import (
    ResNet18_TinyImageNet, ResNet50_TinyImageNet, MobileNetV2_TinyImageNet
)

# Define what gets imported with "from models import *"
__all__ = [
    # ResNet models
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    
    # MobileNetV2 models
    'MobileNetV2', 'mobilenetv2',
    
    # MNASNet models
    'MNASNet', 'mnasnet',
    
    # RegNet models
    'RegNet', 'regnetx_200m', 'regnetx_400m', 'regnetx_600m', 'regnetx_800m',
    'regnetx_1600m', 'regnetx_3200m', 'regnetx_4000m', 'regnetx_6400m',
    'regnety_200m', 'regnety_400m', 'regnety_600m', 'regnety_800m',
    'regnety_1600m', 'regnety_3200m', 'regnety_4000m', 'regnety_6400m',
    
    # CIFAR-optimized models
    'ResNet18_CIFAR', 'ResNet50_CIFAR', 'MobileNetV2_CIFAR'

    # TinyImageNet-optimized models
    'ResNet18_TinyImageNet', 'ResNet50_TinyImageNet', 'MobileNetV2_TinyImageNet'
]

# Model registry for easy access
MODEL_REGISTRY = {
    # ResNet family
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    
    # MobileNetV2
    'mobilenetv2': mobilenetv2,
    
    # MNASNet
    'mnasnet': mnasnet,
    
    # RegNet family
    'regnetx_200m': regnetx_200m,
    'regnetx_400m': regnetx_400m,
    'regnetx_600m': regnetx_600m,
    'regnetx_800m': regnetx_800m,
    'regnetx_1600m': regnetx_1600m,
    'regnetx_3200m': regnetx_3200m,
    'regnetx_4000m': regnetx_4000m,
    'regnetx_6400m': regnetx_6400m,
    'regnety_200m': regnety_200m,
    'regnety_400m': regnety_400m,
    'regnety_600m': regnety_600m,
    'regnety_800m': regnety_800m,
    'regnety_1600m': regnety_1600m,
    'regnety_3200m': regnety_3200m,
    'regnety_4000m': regnety_4000m,
    'regnety_6400m': regnety_6400m,
    
    # CIFAR-optimized models (same architecture for both CIFAR-10 and CIFAR-100)
    'resnet18_cifar': ResNet18_CIFAR,
    'resnet50_cifar': ResNet50_CIFAR,
    'mobilenetv2_cifar': MobileNetV2_CIFAR,
    
    # TinyImageNet-optimized models
    'resnet18_tinyimagenet': ResNet18_TinyImageNet,
    'resnet50_tinyimagenet': ResNet50_TinyImageNet,
    'mobilenetv2_tinyimagenet': MobileNetV2_TinyImageNet,
}

def get_model(model_name, **kwargs):
    """
    Get a model by name.
    
    Args:
        model_name (str): Name of the model to get
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        torch.nn.Module: The requested model
    
    Raises:
        ValueError: If model_name is not found in the registry
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name](**kwargs)

def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
