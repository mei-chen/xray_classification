"""
Model architectures for chest X-ray classification.

Implements transfer learning with modern CNN architectures from timm library.
Supports multiple backbones with customizable classification heads.
"""

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class XRayClassifier(nn.Module):
    """Chest X-Ray Classifier with transfer learning backbone.
    
    Uses a pretrained CNN backbone with a custom classification head.
    Supports progressive unfreezing for fine-tuning.
    
    Attributes:
        backbone: Pretrained CNN feature extractor.
        classifier: Classification head.
        num_classes: Number of output classes.
    """
    
    SUPPORTED_ARCHITECTURES = [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "resnet50",
        "resnet101",
        "densenet121",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
    ]
    
    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        """Initialize the classifier.
        
        Args:
            architecture: Name of the backbone architecture.
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained ImageNet weights.
            dropout: Dropout rate before final classifier.
            freeze_backbone: Whether to freeze backbone weights initially.
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Create backbone using timm
        logger.info(f"Creating {architecture} backbone (pretrained={pretrained})")
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global average pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")
        
        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights using Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W).
            
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification.
        
        Useful for visualization and debugging.
        
        Args:
            x: Input tensor.
            
        Returns:
            Feature tensor from backbone.
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def unfreeze_backbone_layers(self, num_layers: int = -1):
        """Progressively unfreeze backbone layers from the end.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                -1 means unfreeze all.
        """
        if num_layers == -1:
            self.unfreeze_backbone()
            return
        
        # Get all named parameters
        params = list(self.backbone.named_parameters())
        
        # Unfreeze the last num_layers
        for name, param in params[-num_layers:]:
            param.requires_grad = True
            logger.debug(f"Unfroze: {name}")
        
        logger.info(f"Unfroze last {num_layers} backbone parameters")
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> XRayClassifier:
    """Factory function to create a model.
    
    Args:
        architecture: Backbone architecture name.
        num_classes: Number of output classes.
        pretrained: Use pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone initially.
        **kwargs: Additional arguments (ignored).
        
    Returns:
        Configured XRayClassifier model.
    """
    model = XRayClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    
    logger.info(f"Created model: {architecture}")
    logger.info(f"  Total params: {model.get_total_params():,}")
    logger.info(f"  Trainable params: {model.get_trainable_params():,}")
    
    return model


def load_model(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    **model_kwargs: Any,
) -> XRayClassifier:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model to.
        **model_kwargs: Model configuration arguments.
        
    Returns:
        Loaded model in eval mode.
    """
    logger.info(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_kwargs.update({
            "architecture": config.get("architecture", model_kwargs.get("architecture", "efficientnet_b0")),
            "num_classes": config.get("num_classes", model_kwargs.get("num_classes", 3)),
            "dropout": config.get("dropout", model_kwargs.get("dropout", 0.3)),
        })
    
    # Create model (don't freeze backbone for inference)
    model = create_model(freeze_backbone=False, pretrained=False, **model_kwargs)
    
    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model

