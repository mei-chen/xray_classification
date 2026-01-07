"""
Configuration management utilities.

Handles loading, validation, and access to configuration files.
Ensures reproducibility through proper seed setting.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing configuration values.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict[str, Any], save_path: str | Path) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        save_path: Path where to save the configuration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (CPU and CUDA)
    
    It also configures PyTorch for deterministic operations where possible.
    
    Args:
        seed: Integer seed value. Default is 42.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Configure PyTorch for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """Get the appropriate torch device based on configuration.
    
    Args:
        device_config: Device specification. Options:
            - "auto": Automatically detect best available device
            - "cuda": Force CUDA GPU
            - "mps": Force Apple Silicon GPU
            - "cpu": Force CPU
            
    Returns:
        torch.device object for the specified device.
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


@dataclass
class TrainingConfig:
    """Dataclass for training configuration with type hints."""
    
    # Seed
    seed: int = 42
    
    # Data
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3
    num_classes: int = 3
    
    # Training
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    use_amp: bool = True
    
    # Paths
    model_dir: str = "models"
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        """Create TrainingConfig from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            TrainingConfig instance with values from the file.
        """
        config = load_config(config_path)
        
        return cls(
            seed=config.get("seed", 42),
            image_size=config.get("data", {}).get("image_size", 224),
            batch_size=config.get("data", {}).get("batch_size", 32),
            num_workers=config.get("data", {}).get("num_workers", 4),
            architecture=config.get("model", {}).get("architecture", "efficientnet_b0"),
            pretrained=config.get("model", {}).get("pretrained", True),
            dropout=config.get("model", {}).get("dropout", 0.3),
            num_classes=config.get("data", {}).get("num_classes", 3),
            epochs=config.get("training", {}).get("epochs", 50),
            learning_rate=config.get("training", {}).get("optimizer", {}).get("learning_rate", 0.001),
            weight_decay=config.get("training", {}).get("optimizer", {}).get("weight_decay", 0.01),
            use_amp=config.get("training", {}).get("use_amp", True),
            model_dir=config.get("paths", {}).get("model_dir", "models"),
        )

