# jax_backend.py
import numpy as np
import jax.numpy as jnp
import jax
from typing import List, Tuple, Optional
from .utils import Results, _parse_histogram, _calculate_bins


def _compute_radial_sum_jax(
    distances: jnp.ndarray, 
    autocorr: jnp.ndarray, 
    bins: jnp.ndarray, 
    bin_sizes: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute radial sum for autocorrelation data using JAX.
    """
    # Ensure batch dimension
    if distances.ndim == 3:
        distances = distances[None, ...]
        autocorr = autocorr[None, ...]
    
    batch_size, depth, height, width = distances.shape
    distances_flat = distances.reshape(batch_size, -1)
    autocorr_flat = autocorr.reshape(batch_size, -1)
    
    bins_jax = jnp.asarray(bins[:-1])
    bin_sizes_jax = jnp.asarray(bin_sizes)
    
    def compute_single_bin(radius: float, bin_size: float) -> jnp.ndarray:
        """Compute average for a single radial bin."""
        mask = (distances_flat <= radius) & (distances_flat > (radius - bin_size))
        values = jnp.where(mask, autocorr_flat, 0.0)
        sums = jnp.sum(values, axis=1)
        counts = jnp.sum(mask, axis=1)
        return jnp.where(counts > 0, sums / counts, 0.0)
    
    # Vectorize over bins
    radial_sums = jax.vmap(compute_single_bin)(bins_jax, bin_sizes_jax)
    return radial_sums.T


def _compute_radial_profile_jax(
    autocorr_data: jnp.ndarray, 
    bins: np.ndarray, 
    porosity_factors: np.ndarray, 
    voxel_size: float = 1.0
) -> List[Results]:
    """
    Compute radial profile from autocorrelation data.
    """
    # Ensure batch dimension
    single_image = autocorr_data.ndim == 3
    if single_image:
        autocorr_data = autocorr_data[None, ...]
    
    batch_size = autocorr_data.shape[0]
    spatial_shape = autocorr_data.shape[1:]
    
    # Create distance grid
    indices = jnp.indices(spatial_shape)
    center = jnp.round(jnp.array(spatial_shape).reshape(-1, 1, 1, 1) / 2)
    centered_indices = indices - center
    distances = jnp.sqrt(jnp.sum(centered_indices ** 2, axis=0))
    distances = jnp.broadcast_to(distances, (batch_size,) + spatial_shape)
    
    # Compute radial averages
    bin_sizes = np.diff(bins)
    radial_averages = _compute_radial_sum_jax(distances, autocorr_data, bins, bin_sizes)
    
    # Normalize by maximum autocorrelation
    max_autocorr = jnp.max(autocorr_data.reshape(batch_size, -1), axis=1, keepdims=True)
    normalized_autocorr = radial_averages / max_autocorr
    normalized_autocorr_np = np.array(normalized_autocorr)
    
    # Convert porosity factors to array
    if np.isscalar(porosity_factors):
        porosity_array = np.full(batch_size, porosity_factors)
    else:
        porosity_array = np.asarray(porosity_factors)
    
    # Create results for each batch item
    results = []
    for i in range(batch_size):
        histogram_data = _parse_histogram([normalized_autocorr_np[i], bins], voxel_size)
        
        result = Results()
        result.distance = histogram_data.bin_centers
        result.bin_centers = histogram_data.bin_centers
        result.bin_edges = histogram_data.bin_edges
        result.bin_widths = histogram_data.bin_widths
        result.probability = normalized_autocorr_np[i]
        result.probability_scaled = normalized_autocorr_np[i] * porosity_array[i]
        result.pdf = histogram_data.pdf * porosity_array[i]
        
        results.append(result)
    
    return results[0] if single_image else results


def two_point_correlation_jax(
    images: np.ndarray,
    voxel_size: float = 1.0,
    bins: int = 100,
    batch_size: int = 8
) -> List[Results]:
    """
    Compute two-point correlation function using JAX backend.
    """
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
    
    @jax.jit
    def compute_autocorrelation(batch_images: jnp.ndarray) -> jnp.ndarray:
        """Compute autocorrelation via FFT."""
        # FFT of input images
        fourier_transform = jnp.fft.ifftshift(
            jnp.fft.rfftn(
                jnp.fft.fftshift(batch_images, axes=(-3, -2, -1)), 
                axes=(-3, -2, -1)
            ), 
            axes=(-3, -2, -1)
        )
        
        # Power spectrum
        power_spectrum = jnp.abs(fourier_transform) ** 2
        
        # Inverse FFT to get autocorrelation
        autocorrelation = jnp.abs(
            jnp.fft.ifftshift(
                jnp.fft.irfftn(
                    jnp.fft.fftshift(power_spectrum, axes=(-3, -2, -1)), 
                    axes=(-3, -2, -1)
                ), 
                axes=(-3, -2, -1)
            )
        )
        
        return autocorrelation
    
    # Process in batches
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        current_batch = binary_images[start_idx:end_idx]
        current_porosity = porosity_factors[start_idx:end_idx]
        
        valid_batch_size = current_batch.shape[0]
        
        # Pad batch if necessary
        if valid_batch_size < batch_size:
            padding = batch_size - valid_batch_size
            current_batch = np.pad(
                current_batch, 
                ((0, padding), (0, 0), (0, 0), (0, 0)), 
                mode="constant"
            )
        
        # Compute autocorrelation
        batch_jax = jnp.asarray(current_batch)
        autocorrelation_jax = compute_autocorrelation(batch_jax)
        autocorrelation_np = np.asarray(jax.device_get(autocorrelation_jax))
        
        # Compute radial profiles
        batch_results = _compute_radial_profile_jax(
            autocorrelation_np[:valid_batch_size], 
            bins_array, 
            current_porosity, 
            voxel_size
        )
        
        if isinstance(batch_results, list):
            all_results.extend(batch_results)
        else:
            all_results.append(batch_results)
    
    return all_results[0] if single_image else all_results