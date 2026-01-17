"""
Image transformation pipelines for training and evaluation.

Implements data augmentation strategies optimized for medical imaging,
including careful handling of intensity normalization and geometric transforms.
"""

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class RandomCLAHE:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) randomly.
    
    CLAHE enhances local contrast, particularly useful for medical images
    with low contrast regions. Applied to the L channel in LAB color space
    to preserve color information.
    
    Note: CLAHE object is created lazily in __call__ to support multiprocessing
    (cv2.CLAHE objects cannot be pickled).
    
    Args:
        p: Probability of applying CLAHE (default: 0.2)
        clip_limit: Contrast limiting threshold (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
    """
    
    def __init__(
        self,
        p: float = 0.2,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        # Don't create CLAHE here - it can't be pickled for multiprocessing
        self._clahe = None
    
    def _get_clahe(self):
        """Lazily create CLAHE object (for multiprocessing compatibility)."""
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
        return self._clahe
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE with probability p.
        
        Args:
            img: PIL Image (RGB)
            
        Returns:
            PIL Image with CLAHE applied (or unchanged if not triggered)
        """
        if np.random.random() > self.p:
            return img
        
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Convert RGB to LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel (luminance) - create lazily for pickling
        clahe = self._get_clahe()
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(result)
    
    def __getstate__(self):
        """Remove unpicklable CLAHE object for multiprocessing."""
        state = self.__dict__.copy()
        state['_clahe'] = None
        return state


def get_train_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    **augmentation_config: Any,
) -> transforms.Compose:
    """Get training data augmentation pipeline.
    
    Applies a series of augmentations designed for medical imaging:
    - Random CLAHE for contrast enhancement (helps low-contrast images)
    - Random horizontal flip (chest X-rays are roughly symmetric)
    - Random rotation (small angles to simulate patient positioning)
    - Color jitter for brightness/contrast (simulate different exposure settings)
    - Random resized crop (simulate different zoom levels)
    
    Args:
        image_size: Target image size (square).
        mean: Normalization mean. Defaults to ImageNet values.
        std: Normalization std. Defaults to ImageNet values.
        **augmentation_config: Additional augmentation parameters.
        
    Returns:
        Composed transform pipeline for training.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet
    
    # Extract augmentation parameters with defaults (optimized based on ablation study)
    # Ablation results: brightness +2.8%, hflip +2.2%, contrast +1.6%, rotation -0.5%
    horizontal_flip = augmentation_config.get("horizontal_flip", True)
    rotation_degrees = augmentation_config.get("rotation_degrees", 10)  # Reduced from 15 (ablation showed rotation hurts)
    brightness_range = augmentation_config.get("brightness_range", [0.8, 1.2])
    contrast_range = augmentation_config.get("contrast_range", [0.8, 1.2])
    clahe_probability = augmentation_config.get("clahe_probability", 0.2)
    
    transform_list = [
        # CLAHE for contrast enhancement (especially helps low-contrast TB images)
        # Applied early before geometric transforms to enhance on original resolution
        RandomCLAHE(p=clahe_probability, clip_limit=2.0, tile_grid_size=(8, 8)),
        
        # Resize to slightly larger for random cropping
        transforms.Resize(int(image_size * 1.1)),
        
        # Random crop to target size
        transforms.RandomCrop(image_size),
        
        # Horizontal flip (X-rays are roughly symmetric)
        transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        
        # Small rotation to simulate patient positioning variance
        transforms.RandomRotation(degrees=rotation_degrees),
        
        # Brightness and contrast adjustment
        # ColorJitter expects values as max deviation from 1, e.g., 0.2 means range [0.8, 1.2]
        transforms.ColorJitter(
            brightness=brightness_range[1] - 1,  # e.g., 1.2 - 1 = 0.2
            contrast=contrast_range[1] - 1,
        ),
        
        # Random affine for slight perspective changes
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics (for transfer learning)
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(transform_list)


def get_eval_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Get evaluation (validation/test) transform pipeline.
    
    Applies minimal transformations for consistent evaluation:
    - Resize to target size
    - Center crop (if needed)
    - Normalize
    
    Args:
        image_size: Target image size (square).
        mean: Normalization mean. Defaults to ImageNet values.
        std: Normalization std. Defaults to ImageNet values.
        
    Returns:
        Composed transform pipeline for evaluation.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.05)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Get inference transform pipeline for single image prediction.
    
    Same as eval transforms but explicitly named for clarity in deployment.
    
    Args:
        image_size: Target image size (square).
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Composed transform pipeline for inference.
    """
    return get_eval_transforms(image_size, mean, std)


def denormalize(
    tensor: torch.Tensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W).
        mean: Normalization mean used.
        std: Normalization std used.
        
    Returns:
        Denormalized tensor suitable for visualization.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std_t + mean_t

