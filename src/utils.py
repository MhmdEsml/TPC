# utils.py
import numpy as np
from typing import List, Union, Tuple


class Results:
    """Container for two-point correlation function results."""
    
    def __init__(self):
        self.distance = None
        self.bin_centers = None
        self.bin_edges = None
        self.bin_widths = None
        self.probability = None
        self.probability_scaled = None
        self.pdf = None
    
    def __repr__(self) -> str:
        return (
            f"Results(distance_shape={getattr(self.distance, 'shape', None)}, "
            f"probability_shape={getattr(self.probability, 'shape', None)})"
        )


def _calculate_bins(
    image_shape: Tuple[int, int, int], 
    bins: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Calculate bin edges for radial profile.
    
    Args:
        image_shape: Shape of the input image (depth, height, width)
        bins: Number of bins or array of bin edges
        
    Returns:
        Array of bin edges
    """
    if isinstance(bins, int):
        # Calculate maximum radius (half of smallest dimension)
        max_radius = int(np.ceil(np.min(image_shape) / 2))
        bin_size = int(np.ceil(max_radius / bins))
        bins_array = np.arange(0, max_radius + bin_size, bin_size)
        return bins_array
    else:
        return np.asarray(bins)


def _parse_histogram(
    histogram_data: List, 
    voxel_size: float = 1.0
) -> Results:
    """
    Parse histogram data into Results object with physical units.
    
    Args:
        histogram_data: List containing [probability, bin_edges]
        voxel_size: Physical size of voxels for unit conversion
        
    Returns:
        Results object with histogram information
    """
    probability, bin_edges = histogram_data
    
    # Calculate bin properties
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Create histogram results
    histogram = Results()
    histogram.pdf = probability.copy()
    histogram.relfreq = probability * bin_widths
    histogram.bin_centers = bin_centers * voxel_size
    histogram.bin_edges = bin_edges * voxel_size
    histogram.bin_widths = bin_widths * voxel_size
    
    return histogram