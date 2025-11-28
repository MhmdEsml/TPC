# torch_backend.py
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from .utils import Results, _parse_histogram, _calculate_bins


def _compute_radial_sum_torch(
    distances: torch.Tensor,
    autocorr: torch.Tensor, 
    bins: np.ndarray,
    bin_sizes: np.ndarray
) -> torch.Tensor:
    """
    Compute radial sum for autocorrelation data using PyTorch.
    """
    # Ensure batch dimension
    if distances.ndim == 3:
        distances = distances.unsqueeze(0)
        autocorr = autocorr.unsqueeze(0)
    
    batch_size, depth, height, width = distances.shape
    distances_flat = distances.reshape(batch_size, -1)
    autocorr_flat = autocorr.reshape(batch_size, -1)
    
    device = distances.device
    dtype = distances.dtype
    
    bins_tensor = torch.tensor(bins[:-1], device=device, dtype=dtype)
    bin_sizes_tensor = torch.tensor(bin_sizes, device=device, dtype=dtype)
    
    radial_averages = torch.zeros(batch_size, len(bins_tensor), device=device, dtype=torch.float32)
    
    for bin_idx in range(len(bins_tensor)):
        radius = bins_tensor[bin_idx]
        bin_size = bin_sizes_tensor[bin_idx]
        
        # Create mask for current radial bin
        mask = (distances_flat <= radius) & (distances_flat > (radius - bin_size))
        
        # Compute average within bin
        counts = mask.sum(dim=1, keepdim=True)
        values = autocorr_flat * mask.float()
        sums = values.sum(dim=1)
        
        radial_averages[:, bin_idx] = torch.where(
            counts.squeeze(1) > 0, 
            sums / counts.squeeze(1), 
            torch.tensor(0.0, device=device)
        )
    
    return radial_averages


def _compute_radial_profile_torch(
    autocorr_data: torch.Tensor,
    bins: np.ndarray,
    porosity_factors: np.ndarray,
    voxel_size: float = 1.0
) -> Union[Results, List[Results]]:
    """
    Compute radial profile from autocorrelation data.
    """
    device = autocorr_data.device
    batch_size = autocorr_data.shape[0]
    spatial_shape = autocorr_data.shape[1:]
    
    # Create distance grid
    coordinate_grids = torch.meshgrid(
        [torch.arange(dim_size, device=device) for dim_size in spatial_shape], 
        indexing="ij"
    )
    coordinates = torch.stack(coordinate_grids)
    
    center = torch.round(
        torch.tensor(spatial_shape, device=device).reshape(-1, 1, 1, 1) / 2
    )
    centered_coordinates = coordinates - center
    distances = torch.sqrt(torch.sum(centered_coordinates ** 2, dim=0))
    distances = distances.expand(batch_size, *spatial_shape)
    
    # Compute radial averages
    bin_sizes = np.diff(bins)
    radial_averages = _compute_radial_sum_torch(distances, autocorr_data, bins, bin_sizes)
    
    # Normalize by maximum autocorrelation
    max_autocorr = autocorr_data.reshape(batch_size, -1).max(dim=1)[0].unsqueeze(1)
    normalized_autocorr = (radial_averages / max_autocorr).cpu().numpy()
    
    # Convert porosity factors to array
    if np.isscalar(porosity_factors):
        porosity_array = np.full(batch_size, porosity_factors)
    else:
        porosity_array = np.asarray(porosity_factors)
    
    # Create results for each batch item
    results = []
    for i in range(batch_size):
        histogram_data = _parse_histogram([normalized_autocorr[i], bins], voxel_size)
        
        result = Results()
        result.distance = histogram_data.bin_centers
        result.bin_centers = histogram_data.bin_centers
        result.bin_edges = histogram_data.bin_edges
        result.bin_widths = histogram_data.bin_widths
        result.probability = normalized_autocorr[i]
        result.probability_scaled = normalized_autocorr[i] * porosity_array[i]
        result.pdf = histogram_data.pdf * porosity_array[i]
        
        results.append(result)
    
    return results[0] if batch_size == 1 else results


def two_point_correlation_torch(
    images: np.ndarray,
    voxel_size: float = 1.0,
    bins: int = 100,
    batch_size: int = 8,
    device: Optional[str] = None
) -> List[Results]:
    """
    Compute two-point correlation function using PyTorch backend.
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    single_image = images.ndim == 3
    if single_image:
        images = images[None, ...]
    
    # Binarize images
    binary_images = (images > 0.5).astype(np.float32)
    
    # Calculate bins and porosity factors
    bins_array = _calculate_bins(binary_images.shape[1:], bins)
    porosity_factors = binary_images.mean(axis=(-3, -2, -1))
    
    num_images = binary_images.shape[0]
    all_results = []
    
    # Process in batches
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        current_batch = binary_images[start_idx:end_idx]
        current_porosity = porosity_factors[start_idx:end_idx]
        
        # Convert to PyTorch tensor
        batch_tensor = torch.tensor(current_batch, device=device, dtype=torch.float32)
        
        # Compute autocorrelation via FFT
        fourier_transform = torch.fft.ifftshift(
            torch.fft.rfftn(
                torch.fft.fftshift(batch_tensor, dim=(-3, -2, -1)), 
                dim=(-3, -2, -1)
            ), 
            dim=(-3, -2, -1)
        )
        
        power_spectrum = torch.abs(fourier_transform) ** 2
        
        autocorrelation = torch.abs(
            torch.fft.ifftshift(
                torch.fft.irfftn(
                    torch.fft.fftshift(power_spectrum, dim=(-3, -2, -1)), 
                    dim=(-3, -2, -1)
                ), 
                dim=(-3, -2, -1)
            )
        )
        
        # Compute radial profiles
        batch_results = _compute_radial_profile_torch(
            autocorrelation, 
            bins_array, 
            current_porosity, 
            voxel_size
        )
        
        if isinstance(batch_results, list):
            all_results.extend(batch_results)
        else:
            all_results.append(batch_results)
    
    return all_results[0] if single_image else all_results