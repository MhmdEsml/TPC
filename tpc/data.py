# data.py
import os
import urllib.parse
import urllib.request
from typing import List, Tuple, Dict, Any, Optional, Iterator
import numpy as np


# Dataset information
POROUS_MEDIA_DATASETS = {
    "Berea": "Berea/Berea_2d25um_binary.raw/Berea_2d25um_binary.raw",
    "BanderaBrown": "Bandera Brown/BanderaBrown_2d25um_binary.raw/BanderaBrown_2d25um_binary.raw",
    "BanderaGray": "Bandera Gray/BanderaGray_2d25um_binary.raw/BanderaGray_2d25um_binary.raw",
    "Bentheimer": "Bentheimer/Bentheimer_2d25um_binary.raw/Bentheimer_2d25um_binary.raw",
    "BSG": "Berea Sister Gray/BSG_2d25um_binary.raw/BSG_2d25um_binary.raw",
    "BUG": "Berea Upper Gray/BUG_2d25um_binary.raw/BUG_2d25um_binary.raw",
    "BuffBerea": "Buff Berea/BB_2d25um_binary.raw/BB_2d25um_binary.raw",
    "CastleGate": "CastleGate/CastleGate_2d25um_binary.raw/CastleGate_2d25um_binary.raw",
    "Kirby": "Kirby/Kirby_2d25um_binary.raw/Kirby_2d25um_binary.raw",
    "Leopard": "Leopard/Leopard_2d25um_binary.raw/Leopard_2d25um_binary.raw",
    "Parker": "Parker/Parker_2d25um_binary.raw/Parker_2d25um_binary.raw",
}

DATASET_BASE_URL = "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-317"


def download_dataset(
    dataset_name: str, 
    data_directory: str = "datasets", 
    force_download: bool = False
) -> str:
    """
    Download a porous media dataset.
    """
    if dataset_name not in POROUS_MEDIA_DATASETS:
        available_datasets = list(POROUS_MEDIA_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available_datasets}"
        )
    
    # Create data directory if it doesn't exist
    os.makedirs(data_directory, exist_ok=True)
    
    output_path = os.path.join(data_directory, f"{dataset_name}.raw")
    
    # Check if dataset already exists
    if os.path.exists(output_path) and not force_download:
        print(f"Dataset '{dataset_name}' already exists at: {output_path}")
        print("Use force_download=True to re-download")
        return output_path
    
    # Construct download URL
    relative_path = POROUS_MEDIA_DATASETS[dataset_name]
    encoded_path = "/".join(urllib.parse.quote(part) for part in relative_path.split("/"))
    download_url = f"{DATASET_BASE_URL}/{encoded_path}"

    try:
        print(f"Downloading dataset '{dataset_name}' from: {download_url}")
        urllib.request.urlretrieve(download_url, output_path)
        print(f"Downloaded to: {output_path}")
        return output_path
    except Exception as error:
        print(f"Download failed: {error}")
        raise IOError(f"Failed to download dataset '{dataset_name}'") from error


def dataset_exists(dataset_name: str, data_directory: str = "datasets") -> bool:
    """
    Check if a dataset exists locally.
    """
    dataset_path = os.path.join(data_directory, f"{dataset_name}.raw")
    return os.path.exists(dataset_path)


def get_dataset_path(dataset_name: str, data_directory: str = "datasets") -> str:
    """
    Get the local file path for a dataset.
    """
    return os.path.join(data_directory, f"{dataset_name}.raw")


def load_volume(
    raw_file_path: str, 
    dimensions: Tuple[int, int, int]
) -> np.ndarray:
    """
    Load 3D volume from raw binary file.
    """
    depth, height, width = dimensions
    data_type = np.uint8
    total_voxels = depth * height * width
    
    print(f"Loading 3D volume from: {raw_file_path}")
    
    try:
        # Read raw binary data
        volume_data = np.fromfile(raw_file_path, dtype=data_type)
        
        # Check if data size matches expected dimensions
        if volume_data.size != total_voxels:
            raise ValueError(
                f"File size {volume_data.size} does not match "
                f"expected size {total_voxels} for dimensions {dimensions}"
            )
        
        # Reshape to 3D volume
        volume = volume_data.reshape((depth, height, width))
        
        return volume
        
    except Exception as error:
        print(f"Failed to load volume: {error}")
        raise IOError(f"Could not load volume from {raw_file_path}") from error


from typing import Iterator, Dict, Any, Tuple
import numpy as np

def create_train_generator(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    batch_size: int,
    compute_porosity: bool = True,
    seed: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Create infinite generator that yields random 3D patches from volume.
    Reproducible if seed is provided.
    """

    # Local RNG
    rng = np.random.default_rng(seed)

    volume_depth, volume_height, volume_width = volume.shape
    patch_depth, patch_height, patch_width = patch_size

    max_depth = volume_depth - patch_depth
    max_height = volume_height - patch_height
    max_width = volume_width - patch_width

    print(
        f"Creating training data generator:\n"
        f"  Volume shape: {volume.shape}\n"
        f"  Patch size: {patch_size}\n"
        f"  Batch size: {batch_size}\n"
        f"  Compute porosity: {compute_porosity}\n"
        f"  Seed: {seed}"
    )

    while True:
        image_batch = []
        porosity_batch = []

        for _ in range(batch_size):
            # Sample patch location
            sd = rng.integers(0, max_depth + 1)
            sh = rng.integers(0, max_height + 1)
            sw = rng.integers(0, max_width + 1)

            patch = volume[
                sd:sd + patch_depth,
                sh:sh + patch_height,
                sw:sw + patch_width
            ].astype(np.float32)

            # Random flips
            if rng.random() > 0.5:
                patch = np.flip(patch, axis=0)
            if rng.random() > 0.5:
                patch = np.flip(patch, axis=1)
            if rng.random() > 0.5:
                patch = np.flip(patch, axis=2)

            if compute_porosity:
                porosity_batch.append(np.mean(patch == 0).astype(np.float32))

            image_batch.append(patch.copy())

        images = np.stack(image_batch)[..., None]

        batch = {'images': images}

        if compute_porosity:
            batch['porosity'] = np.asarray(porosity_batch)

        yield batch


