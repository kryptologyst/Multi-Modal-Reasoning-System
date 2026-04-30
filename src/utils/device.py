"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.deterministic = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over other devices.
        
    Returns:
        The best available torch device.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about the current device.
    
    Returns:
        Dictionary containing device information.
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "current_gpu": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
        })
    elif device.type == "mps":
        info.update({
            "mps_available": True,
            "mps_built": torch.backends.mps.is_built(),
        })
    else:
        info.update({
            "cpu_only": True,
        })
    
    return info


def clear_cache() -> None:
    """Clear GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory_usage() -> dict:
    """Get current memory usage.
    
    Returns:
        Dictionary containing memory usage information.
    """
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info.update({
            "cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved(),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved": torch.cuda.max_memory_reserved(),
        })
    
    return memory_info
