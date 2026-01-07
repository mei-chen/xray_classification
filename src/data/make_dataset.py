"""
Data preparation and preprocessing script.

This script handles the complete data preparation pipeline:
1. Loads raw data from the downloaded dataset
2. Performs data validation and cleaning
3. Creates train/val/test splits
4. Saves processed data with metadata

Usage:
    python -m src.data.make_dataset
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import load_config, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_image(image_path: Path) -> dict | None:
    """Analyze a single image and return its properties.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary with image properties or None if invalid.
    """
    try:
        with Image.open(image_path) as img:
            return {
                "path": str(image_path),
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
            }
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return None


def validate_dataset(data_dir: Path, class_names: list[str]) -> dict:
    """Validate the dataset and compute statistics.
    
    Args:
        data_dir: Path to the dataset directory.
        class_names: List of expected class names.
        
    Returns:
        Dictionary with validation results and statistics.
    """
    stats = {
        "valid_images": 0,
        "invalid_images": 0,
        "class_distribution": {},
        "image_sizes": [],
        "issues": [],
    }
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            stats["issues"].append(f"Missing class directory: {class_name}")
            continue
        
        # Find all images
        images = (
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )
        
        valid_count = 0
        for img_path in tqdm(images, desc=f"Validating {class_name}"):
            info = analyze_image(img_path)
            if info:
                valid_count += 1
                stats["image_sizes"].append(info["size"])
            else:
                stats["invalid_images"] += 1
        
        stats["valid_images"] += valid_count
        stats["class_distribution"][class_name] = valid_count
    
    # Compute size statistics
    if stats["image_sizes"]:
        sizes = np.array(stats["image_sizes"])
        stats["size_stats"] = {
            "min_width": int(sizes[:, 0].min()),
            "max_width": int(sizes[:, 0].max()),
            "mean_width": float(sizes[:, 0].mean()),
            "min_height": int(sizes[:, 1].min()),
            "max_height": int(sizes[:, 1].max()),
            "mean_height": float(sizes[:, 1].mean()),
        }
        del stats["image_sizes"]  # Remove raw data
    
    return stats


def create_split_files(
    data_dir: Path,
    output_dir: Path,
    class_names: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create train/val/test split files.
    
    Creates JSON files listing image paths for each split,
    enabling reproducible data loading.
    
    Args:
        data_dir: Source data directory.
        output_dir: Directory to save split files.
        class_names: List of class names.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed for splitting.
        
    Returns:
        Dictionary with split statistics.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_paths = []
    all_labels = []
    
    for idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        images = (
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )
        all_paths.extend([str(p) for p in images])
        all_labels.extend([idx] * len(images))
    
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    
    # Stratified split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        train_size=train_ratio,
        stratify=all_labels,
        random_state=seed,
    )
    
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=seed,
    )
    
    # Save splits
    splits = {
        "train": {"paths": train_paths.tolist(), "labels": train_labels.tolist()},
        "val": {"paths": val_paths.tolist(), "labels": val_labels.tolist()},
        "test": {"paths": test_paths.tolist(), "labels": test_labels.tolist()},
    }
    
    for split_name, split_data in splits.items():
        split_file = output_dir / f"{split_name}_split.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} split: {len(split_data['paths'])} samples")
    
    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "class_names": class_names,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_sizes": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        "class_distribution": {
            "train": {class_names[i]: int((train_labels == i).sum()) for i in range(len(class_names))},
            "val": {class_names[i]: int((val_labels == i).sum()) for i in range(len(class_names))},
            "test": {class_names[i]: int((test_labels == i).sum()) for i in range(len(class_names))},
        },
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    """Main data preparation pipeline."""
    # Load configuration
    config_path = Path("configs/train_config.yaml")
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {
            "seed": 42,
            "data": {
                "raw_data_path": "data/raw/chest-xray-dataset",
                "processed_data_path": "data/processed",
                "classes": ["Normal", "Pneumonia", "Tuberculosis"],
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            }
        }
    
    seed = config.get("seed", 42)
    set_seed(seed)
    
    data_config = config.get("data", {})
    raw_data_dir = Path(data_config.get("raw_data_path", "data/raw/chest-xray-dataset"))
    processed_dir = Path(data_config.get("processed_data_path", "data/processed"))
    class_names = data_config.get("classes", ["Normal", "Pneumonia", "Tuberculosis"])
    
    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Check if raw data exists
    if not raw_data_dir.exists():
        logger.error(f"Raw data not found at: {raw_data_dir}")
        logger.error("Please run: python -m src.data.download_dataset")
        return
    
    # Step 1: Validate dataset
    logger.info("\nStep 1: Validating dataset...")
    validation_stats = validate_dataset(raw_data_dir, class_names)
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  Valid images: {validation_stats['valid_images']}")
    logger.info(f"  Invalid images: {validation_stats['invalid_images']}")
    logger.info(f"  Class distribution:")
    for class_name, count in validation_stats['class_distribution'].items():
        logger.info(f"    {class_name}: {count}")
    
    if validation_stats.get("size_stats"):
        logger.info(f"  Image sizes:")
        ss = validation_stats["size_stats"]
        logger.info(f"    Width: {ss['min_width']}-{ss['max_width']} (mean: {ss['mean_width']:.0f})")
        logger.info(f"    Height: {ss['min_height']}-{ss['max_height']} (mean: {ss['mean_height']:.0f})")
    
    # Step 2: Create splits
    logger.info("\nStep 2: Creating train/val/test splits...")
    metadata = create_split_files(
        data_dir=raw_data_dir,
        output_dir=processed_dir,
        class_names=class_names,
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
        test_ratio=data_config.get("test_ratio", 0.15),
        seed=seed,
    )
    
    logger.info(f"\nSplit sizes:")
    for split, size in metadata["split_sizes"].items():
        logger.info(f"  {split}: {size}")
    
    logger.info(f"\nProcessed data saved to: {processed_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

