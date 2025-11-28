# core.py
import numpy as np
from typing import List, Optional, Union
from .utils import Results, _calculate_bins


def available_backends() -> List[str]:
    """
    Return list of available computation backends.
    
    Returns:
        List of available backend names
    """
    backends = []
    
    try:
        import torch
        backends.append('torch')
    except ImportError:
        pass
        
    try:
        import jax
        import jax.numpy as jnp
        backends.append('jax')
    except ImportError:
        pass
        
    return backends


def compute_tpcf(
    images: np.ndarray, 
    voxel_size: float = 1.0,
    bins: Union[int, np.ndarray] = 100,
    backend: str = 'torch',
    batch_size: int = 8,
    device: Optional[str] = None
) -> Union[Results, List[Results]]:
    """
    Compute Two-Point Correlation Function for 3D binary images.
    
    Args:
        images: 3D or 4D array of images (single or batch)
        voxel_size: Physical size of voxels
        bins: Number of bins or array of bin edges
        backend: Computation backend ('torch' or 'jax')
        batch_size: Batch size for processing multiple images
        device: Device for computation (only for torch backend)
        
    Returns:
        Single Results object for single image, list of Results for batch
    """
    backend = backend.lower()
    available = available_backends()
    
    if backend not in available:
        raise ValueError(
            f"Unsupported backend: '{backend}'. "
            f"Available backends: {available}"
        )
    
    if backend == 'torch':
        from .torch_backend import two_point_correlation_torch
        return two_point_correlation_torch(
            images, voxel_size, bins, batch_size, device
        )
    
    elif backend == 'jax':
        from .jax_backend import two_point_correlation_jax
        return two_point_correlation_jax(
            images, voxel_size, bins, batch_size
        )