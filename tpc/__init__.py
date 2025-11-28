# __init__.py
from .core import compute_tpcf, available_backends
from .utils import Results
from .data import (
    load_volume, 
    create_train_generator, 
    download_dataset,
    dataset_exists, 
    get_dataset_path
)

__version__ = "0.1.0"

__all__ = [
    "compute_tpcf",
    "available_backends", 
    "Results",
    "load_volume",
    "create_train_generator", 
    "download_dataset",
    "dataset_exists",
    "get_dataset_path",
    "__version__",
]