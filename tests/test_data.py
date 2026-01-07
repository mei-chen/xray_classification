"""Tests for data loading and transforms."""

import pytest
import torch
from PIL import Image
import tempfile
from pathlib import Path
import numpy as np

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
    denormalize,
)
from src.data.dataset import ChestXRayDataset


class TestTransforms:
    """Test suite for data transforms."""
    
    def test_train_transform_output_shape(self):
        """Test that train transforms produce correct output shape."""
        transform = get_train_transforms(image_size=224)
        
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        output = transform(img)
        
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32
    
    def test_eval_transform_output_shape(self):
        """Test that eval transforms produce correct output shape."""
        transform = get_eval_transforms(image_size=224)
        
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        output = transform(img)
        
        assert output.shape == (3, 224, 224)
    
    def test_denormalize(self):
        """Test denormalization recovers approximate original values."""
        # Create normalized tensor
        original = torch.rand(3, 224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Normalize
        normalized = original.clone()
        for c in range(3):
            normalized[c] = (normalized[c] - mean[c]) / std[c]
        
        # Denormalize
        recovered = denormalize(normalized, mean, std)
        
        assert torch.allclose(original, recovered, atol=1e-5)
    
    def test_different_image_sizes(self):
        """Test transforms work with different target sizes."""
        for size in [128, 224, 256]:
            transform = get_eval_transforms(image_size=size)
            img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
            
            output = transform(img)
            
            assert output.shape == (3, size, size)


class TestChestXRayDataset:
    """Test suite for ChestXRayDataset."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset with temporary images."""
        # Create temporary images
        image_paths = []
        labels = []
        
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img_path = tmp_path / f"image_{i}.png"
            img.save(img_path)
            image_paths.append(img_path)
            labels.append(i % 3)  # 3 classes
        
        return ChestXRayDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_eval_transforms(image_size=224),
        )
    
    def test_dataset_length(self, sample_dataset):
        """Test dataset length is correct."""
        assert len(sample_dataset) == 10
    
    def test_getitem(self, sample_dataset):
        """Test getting items from dataset."""
        image, label = sample_dataset[0]
        
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 3
    
    def test_class_weights(self, sample_dataset):
        """Test class weight calculation."""
        weights = sample_dataset.get_class_weights()
        
        assert weights.shape == (3,)
        assert all(w > 0 for w in weights)

