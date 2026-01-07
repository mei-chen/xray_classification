"""
Download the Chest X-Ray Dataset from Kaggle.

This script downloads the dataset using kagglehub and organizes it
into the project's data directory structure.

Usage:
    python -m src.data.download_dataset
"""

import logging
import shutil
from pathlib import Path

import kagglehub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset(destination: str | Path = "data/raw") -> Path:
    """Download the Chest X-Ray dataset from Kaggle.
    
    Uses kagglehub to download the dataset and copies it to the
    project's data/raw directory.
    
    Args:
        destination: Directory where to save the raw data.
        
    Returns:
        Path to the downloaded dataset.
        
    Note:
        Requires Kaggle API credentials to be set up.
        See: https://github.com/Kaggle/kaggle-api#api-credentials
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Chest X-Ray Dataset from Kaggle...")
    logger.info("Dataset: muhammadrehan00/chest-xray-dataset")
    
    # Download using kagglehub
    kaggle_path = kagglehub.dataset_download("muhammadrehan00/chest-xray-dataset")
    kaggle_path = Path(kaggle_path)
    
    logger.info(f"Downloaded to: {kaggle_path}")
    
    # Copy to our data directory
    dest_path = destination / "chest-xray-dataset"
    if dest_path.exists():
        logger.info(f"Destination already exists: {dest_path}")
        logger.info("Removing existing data to refresh...")
        shutil.rmtree(dest_path)
    
    logger.info(f"Copying data to: {dest_path}")
    shutil.copytree(kaggle_path, dest_path)
    
    # Log dataset structure
    logger.info("\nDataset structure:")
    for class_dir in sorted(dest_path.iterdir()):
        if class_dir.is_dir():
            num_images = len(list(class_dir.glob("*")))
            logger.info(f"  {class_dir.name}: {num_images} images")
    
    logger.info(f"\nDataset successfully saved to: {dest_path}")
    
    return dest_path


def verify_dataset(data_path: str | Path) -> dict:
    """Verify the downloaded dataset and return statistics.
    
    Args:
        data_path: Path to the dataset directory.
        
    Returns:
        Dictionary with dataset statistics.
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    stats = {}
    total_images = 0
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png")) + \
                     list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg"))
            class_name = class_dir.name
            stats[class_name] = len(images)
            total_images += len(images)
    
    stats["total"] = total_images
    
    return stats


if __name__ == "__main__":
    # Download the dataset
    dataset_path = download_dataset()
    
    # Verify and print statistics
    stats = verify_dataset(dataset_path)
    
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    for class_name, count in stats.items():
        if class_name != "total":
            print(f"  {class_name}: {count} images")
    print(f"\n  Total: {stats['total']} images")
    print("=" * 50)

