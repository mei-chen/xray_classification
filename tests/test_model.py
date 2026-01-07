"""Tests for model architecture and utilities."""

import pytest
import torch

from src.models.architecture import XRayClassifier, create_model


class TestXRayClassifier:
    """Test suite for XRayClassifier model."""
    
    def test_model_creation(self):
        """Test that model can be created successfully."""
        model = create_model(
            architecture="efficientnet_b0",
            num_classes=3,
            pretrained=False,  # Don't download weights in tests
            freeze_backbone=True,
        )
        
        assert isinstance(model, XRayClassifier)
        assert model.num_classes == 3
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = create_model(
            architecture="efficientnet_b0",
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )
        model.eval()
        
        # Random batch of 4 images
        batch = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            output = model(batch)
        
        assert output.shape == (4, 3)
    
    def test_output_range(self):
        """Test that softmax outputs sum to 1."""
        model = create_model(
            architecture="efficientnet_b0",
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )
        model.eval()
        
        batch = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(batch)
            probs = torch.softmax(output, dim=1)
        
        # Check probabilities sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    
    def test_freeze_unfreeze(self):
        """Test backbone freezing and unfreezing."""
        model = create_model(
            architecture="efficientnet_b0",
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )
        
        # Initially frozen - should have fewer trainable params
        trainable_frozen = model.get_trainable_params()
        total_params = model.get_total_params()
        
        assert trainable_frozen < total_params
        
        # Unfreeze
        model.unfreeze_backbone()
        trainable_unfrozen = model.get_trainable_params()
        
        assert trainable_unfrozen == total_params
    
    def test_different_architectures(self):
        """Test that different architectures can be created."""
        architectures = ["efficientnet_b0", "resnet50"]
        
        for arch in architectures:
            model = create_model(
                architecture=arch,
                num_classes=3,
                pretrained=False,
                freeze_backbone=True,
            )
            
            assert model.architecture == arch
            
            # Test forward pass
            batch = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(batch)
            
            assert output.shape == (1, 3)

