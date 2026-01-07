"""
PyTorch Dataset and DataLoader utilities for chest X-ray images.

Implements a custom dataset class with support for:
- Lazy loading for memory efficiency
- Class-weighted sampling for imbalanced data
- Stratified train/val/test splitting
"""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .transforms import get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for Chest X-Ray images.
    
    Loads images lazily from disk with support for custom transforms.
    
    Attributes:
        image_paths: List of paths to image files.
        labels: List of integer labels corresponding to each image.
        class_names: List of class name strings.
        transform: Optional transform to apply to images.
    """
    
    # Class name to index mapping
    CLASS_TO_IDX = {
        "Normal": 0,
        "Pneumonia": 1,
        "Tuberculosis": 2,
    }
    
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    
    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Callable | None = None,
        class_names: list[str] | None = None,
    ):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files.
            labels: List of integer labels for each image.
            transform: Optional transform to apply to images.
            class_names: Optional list of class names. Defaults to
                ["Normal", "Pneumonia", "Tuberculosis"].
        """
        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ["Normal", "Pneumonia", "Tuberculosis"]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (image_tensor, label).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling class imbalance.
        
        Uses inverse frequency weighting.
        
        Returns:
            Tensor of weights for each class.
        """
        labels_array = np.array(self.labels)
        class_counts = np.bincount(labels_array, minlength=len(self.class_names))
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_names)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for weighted random sampling.
        
        Returns:
            Array of weights for each sample.
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = np.array([class_weights[label] for label in self.labels])
        return sample_weights


def load_dataset_from_directory(
    data_dir: str | Path,
    class_names: list[str] | None = None,
) -> tuple[list[Path], list[int], list[str]]:
    """Load image paths and labels from a directory structure.
    
    Supports two directory structures:
    
    Structure 1 (flat):
        data_dir/
            Normal/
            Pneumonia/
            Tuberculosis/
    
    Structure 2 (pre-split):
        data_dir/
            train/
                normal/
                pneumonia/
                tuberculosis/
            val/
            test/
    
    Args:
        data_dir: Root directory containing class subdirectories.
        class_names: Optional list of class names to load.
            Defaults to ["Normal", "Pneumonia", "Tuberculosis"].
            
    Returns:
        Tuple of (image_paths, labels, class_names).
    """
    data_dir = Path(data_dir)
    
    if class_names is None:
        class_names = ["Normal", "Pneumonia", "Tuberculosis"]
    
    # Map display names to folder names (handles case differences)
    folder_names = [name.lower() for name in class_names]
    
    image_paths = []
    labels = []
    
    for idx, (class_name, folder_name) in enumerate(zip(class_names, folder_names)):
        # Try direct class folder first
        class_dir = data_dir / class_name
        if not class_dir.exists():
            class_dir = data_dir / folder_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_name}")
            continue
        
        # Find all images
        class_images = (
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )
        
        image_paths.extend(class_images)
        labels.extend([idx] * len(class_images))
        
        logger.info(f"Found {len(class_images)} images for class '{class_name}'")
    
    logger.info(f"Total: {len(image_paths)} images across {len(class_names)} classes")
    
    return image_paths, labels, class_names


def load_presplit_dataset(
    data_dir: str | Path,
    split: str = "train",
    class_names: list[str] | None = None,
) -> tuple[list[Path], list[int], list[str]]:
    """Load from pre-split dataset (train/val/test folders).
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories.
        split: Which split to load ("train", "val", "test").
        class_names: Optional list of class names.
        
    Returns:
        Tuple of (image_paths, labels, class_names).
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    return load_dataset_from_directory(split_dir, class_names)


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    use_weighted_sampling: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train, validation, and test dataloaders.
    
    Automatically detects if dataset is pre-split (has train/val/test folders)
    or needs to be split.
    
    Args:
        data_dir: Path to the dataset directory.
        batch_size: Batch size for dataloaders.
        image_size: Target image size.
        num_workers: Number of worker processes for data loading.
        train_ratio: Proportion of data for training (if not pre-split).
        val_ratio: Proportion of data for validation (if not pre-split).
        test_ratio: Proportion of data for testing (if not pre-split).
        seed: Random seed for reproducibility.
        use_weighted_sampling: Whether to use weighted random sampling
            to handle class imbalance.
        pin_memory: Whether to pin memory for faster GPU transfer.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    data_dir = Path(data_dir)
    class_names = ["Normal", "Pneumonia", "Tuberculosis"]
    
    # Check if dataset is pre-split
    is_presplit = (data_dir / "train").exists() and (data_dir / "val").exists()
    
    if is_presplit:
        logger.info("Detected pre-split dataset structure")
        train_paths, train_labels, class_names = load_presplit_dataset(data_dir, "train", class_names)
        val_paths, val_labels, _ = load_presplit_dataset(data_dir, "val", class_names)
        test_paths, test_labels, _ = load_presplit_dataset(data_dir, "test", class_names)
    else:
        logger.info("Creating train/val/test splits from flat directory")
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Load all data
        image_paths, labels, class_names = load_dataset_from_directory(data_dir)
        
        # Convert to numpy for sklearn
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=seed,
        )
        
        # Second split: val vs test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            train_size=relative_val_ratio,
            stratify=temp_labels,
            random_state=seed,
        )
    
    logger.info(f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create transforms
    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        image_paths=list(train_paths),
        labels=list(train_labels),
        transform=train_transform,
        class_names=class_names,
    )
    
    val_dataset = ChestXRayDataset(
        image_paths=list(val_paths),
        labels=list(val_labels),
        transform=eval_transform,
        class_names=class_names,
    )
    
    test_dataset = ChestXRayDataset(
        image_paths=list(test_paths),
        labels=list(test_labels),
        transform=eval_transform,
        class_names=class_names,
    )
    
    # Create weighted sampler for training (handles class imbalance)
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle_train = False  # Sampler handles shuffling
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader, class_names

