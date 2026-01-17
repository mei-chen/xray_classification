"""
Augmentation Ablation Study

Tests different augmentation configurations to identify which ones
improve model performance. Uses shortened training (5-7 epochs) to
quickly evaluate each configuration.

Usage:
    python -m src.models.augmentation_ablation
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ChestXRayDataset, load_presplit_dataset
from src.data.transforms import RandomCLAHE
from src.models.architecture import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    horizontal_flip: bool = False
    rotation_degrees: int = 0
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0
    affine_translate: float = 0.0
    affine_scale: tuple = (1.0, 1.0)
    clahe_probability: float = 0.0


# Define ablation configurations
ABLATION_CONFIGS = [
    # Baseline: No augmentation
    AblationConfig(name="baseline_no_aug"),
    
    # Individual augmentations
    AblationConfig(name="hflip_only", horizontal_flip=True),
    AblationConfig(name="rotation_only", rotation_degrees=15),
    AblationConfig(name="brightness_only", brightness_jitter=0.2),
    AblationConfig(name="contrast_only", contrast_jitter=0.2),
    AblationConfig(name="affine_only", affine_translate=0.05, affine_scale=(0.95, 1.05)),
    AblationConfig(name="clahe_only", clahe_probability=0.3),
    
    # Progressive combinations
    AblationConfig(
        name="geometric_only",
        horizontal_flip=True,
        rotation_degrees=15,
        affine_translate=0.05,
        affine_scale=(0.95, 1.05),
    ),
    AblationConfig(
        name="photometric_only",
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        clahe_probability=0.2,
    ),
    
    # Full augmentation (current default)
    AblationConfig(
        name="full_augmentation",
        horizontal_flip=True,
        rotation_degrees=15,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        affine_translate=0.05,
        affine_scale=(0.95, 1.05),
        clahe_probability=0.2,
    ),
    
    # Full without CLAHE (to measure CLAHE impact)
    AblationConfig(
        name="full_no_clahe",
        horizontal_flip=True,
        rotation_degrees=15,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        affine_translate=0.05,
        affine_scale=(0.95, 1.05),
        clahe_probability=0.0,
    ),
]


def build_transforms(config: AblationConfig, image_size: int = 224):
    """Build transform pipeline from ablation config."""
    from torchvision import transforms
    
    transform_list = []
    
    # CLAHE (if enabled)
    if config.clahe_probability > 0:
        transform_list.append(RandomCLAHE(p=config.clahe_probability))
    
    # Resize
    transform_list.append(transforms.Resize(int(image_size * 1.1)))
    transform_list.append(transforms.RandomCrop(image_size))
    
    # Horizontal flip
    if config.horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Rotation
    if config.rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=config.rotation_degrees))
    
    # Color jitter
    if config.brightness_jitter > 0 or config.contrast_jitter > 0:
        transform_list.append(transforms.ColorJitter(
            brightness=config.brightness_jitter,
            contrast=config.contrast_jitter,
        ))
    
    # Affine
    if config.affine_translate > 0 or config.affine_scale != (1.0, 1.0):
        transform_list.append(transforms.RandomAffine(
            degrees=0,
            translate=(config.affine_translate, config.affine_translate) if config.affine_translate > 0 else None,
            scale=config.affine_scale if config.affine_scale != (1.0, 1.0) else None,
        ))
    
    # Standard transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int = 224):
    """Standard eval transforms (no augmentation)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.05)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_names = ["Normal", "Pneumonia", "Tuberculosis"]
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = (all_preds[mask] == all_labels[mask]).mean()
    
    return running_loss / total, correct / total, per_class_acc


def run_ablation_experiment(
    config: AblationConfig,
    data_dir: Path,
    device: torch.device,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> dict:
    """Run a single ablation experiment."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {config.name}")
    logger.info(f"{'='*60}")
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Build transforms
    train_transform = build_transforms(config)
    eval_transform = get_eval_transforms()
    
    # Load dataset paths and labels
    train_paths, train_labels, class_names = load_presplit_dataset(data_dir, "train")
    val_paths, val_labels, _ = load_presplit_dataset(data_dir, "val")
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform,
        class_names=class_names,
    )
    val_dataset = ChestXRayDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=eval_transform,
        class_names=class_names,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model (fresh for each experiment)
    model = create_model(
        architecture="efficientnet_b0",
        num_classes=3,
        pretrained=True,
        dropout=0.3,
    )
    model = model.to(device)
    
    # Freeze backbone for faster ablation
    for name, param in model.named_parameters():
        if "classifier" not in name and "head" not in name:
            param.requires_grad = False
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_per_class = {}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_per_class = per_class_acc
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
    
    elapsed_time = time.time() - start_time
    
    return {
        "config_name": config.name,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "per_class_acc": best_per_class,
        "train_time_seconds": elapsed_time,
        "history": history,
    }


def main():
    """Run the full ablation study."""
    # Configuration
    data_dir = Path("data/raw/chest-xray-dataset")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Run ablation experiments
    results = []
    
    for config in tqdm(ABLATION_CONFIGS, desc="Ablation experiments"):
        try:
            result = run_ablation_experiment(
                config=config,
                data_dir=data_dir,
                device=device,
                num_epochs=5,  # Short training for ablation
                batch_size=32,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error running {config.name}: {e}")
            results.append({
                "config_name": config.name,
                "error": str(e),
            })
    
    # Save results
    output_path = output_dir / "augmentation_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("AUGMENTATION ABLATION RESULTS")
    print("="*80)
    print(f"{'Configuration':<25} {'Val Acc':>10} {'Normal':>10} {'Pneumonia':>10} {'TB':>10}")
    print("-"*80)
    
    # Sort by validation accuracy
    valid_results = [r for r in results if "best_val_acc" in r]
    valid_results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    
    baseline_acc = None
    for r in valid_results:
        if r["config_name"] == "baseline_no_aug":
            baseline_acc = r["best_val_acc"]
            break
    
    for r in valid_results:
        per_class = r.get("per_class_acc", {})
        delta = ""
        if baseline_acc and r["config_name"] != "baseline_no_aug":
            diff = r["best_val_acc"] - baseline_acc
            delta = f" ({'+' if diff >= 0 else ''}{diff*100:.1f}%)"
        
        print(f"{r['config_name']:<25} {r['best_val_acc']*100:>9.2f}%{delta}"
              f" {per_class.get('Normal', 0)*100:>9.1f}%"
              f" {per_class.get('Pneumonia', 0)*100:>9.1f}%"
              f" {per_class.get('Tuberculosis', 0)*100:>9.1f}%")
    
    print("="*80)
    print("\nKey findings:")
    if valid_results:
        best = valid_results[0]
        print(f"  Best config: {best['config_name']} ({best['best_val_acc']*100:.2f}%)")
        
        # Find CLAHE impact
        full_with = next((r for r in valid_results if r["config_name"] == "full_augmentation"), None)
        full_without = next((r for r in valid_results if r["config_name"] == "full_no_clahe"), None)
        if full_with and full_without:
            clahe_impact = full_with["best_val_acc"] - full_without["best_val_acc"]
            print(f"  CLAHE impact: {'+' if clahe_impact >= 0 else ''}{clahe_impact*100:.2f}%")


if __name__ == "__main__":
    main()
