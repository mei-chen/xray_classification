"""
Data validation and cleaning script for chest X-ray images.

Performs quality checks:
- Image integrity verification (corrupted files)
- Size validation (minimum dimensions)
- Aspect ratio checks
- Near-duplicate detection using perceptual hashing
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Validation thresholds
MIN_IMAGE_SIZE = 100  # Minimum dimension in pixels
MAX_ASPECT_RATIO = 3.0  # Maximum allowed aspect ratio


def verify_image_integrity(image_path: Path) -> tuple[bool, str]:
    """Check if an image file is valid and can be opened.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def check_image_dimensions(image_path: Path) -> tuple[bool, str, dict]:
    """Check if image dimensions meet minimum requirements.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Tuple of (is_valid, issue_type, metadata).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
        metadata = {
            "width": width,
            "height": height,
            "mode": mode,
            "aspect_ratio": width / height if height > 0 else 0,
        }
        
        # Check minimum size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False, "too_small", metadata
        
        # Check aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > MAX_ASPECT_RATIO:
            return False, "bad_aspect_ratio", metadata
        
        return True, "", metadata
        
    except Exception as e:
        return False, f"error: {e}", {}


def find_duplicates_phash(
    image_paths: list[Path],
    threshold: int = 5,
) -> list[tuple[Path, Path, int]]:
    """Find near-duplicate images using perceptual hashing.
    
    Args:
        image_paths: List of image paths to check.
        threshold: Maximum hash distance to consider as duplicate.
        
    Returns:
        List of (image1, image2, distance) tuples for duplicates.
    """
    try:
        import imagehash
    except ImportError:
        logger.warning("imagehash not installed. Skipping duplicate detection.")
        logger.warning("Install with: pip install imagehash")
        return []
    
    logger.info("Computing perceptual hashes for duplicate detection...")
    
    hashes = {}
    duplicates = []
    
    for path in tqdm(image_paths, desc="Computing hashes"):
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img.convert("RGB"))
            
            # Check against existing hashes
            for existing_path, existing_hash in hashes.items():
                distance = h - existing_hash
                if distance <= threshold:
                    duplicates.append((path, existing_path, distance))
                    break
            else:
                hashes[path] = h
                
        except Exception as e:
            logger.debug(f"Could not hash {path}: {e}")
    
    return duplicates


def validate_dataset(
    data_dir: Path,
    check_duplicates: bool = True,
    duplicate_threshold: int = 5,
) -> dict:
    """Validate all images in a dataset directory.
    
    Args:
        data_dir: Root directory containing images.
        check_duplicates: Whether to check for duplicates.
        duplicate_threshold: Hash distance threshold for duplicates.
        
    Returns:
        Dictionary with validation results.
    """
    data_dir = Path(data_dir)
    
    # Find all images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(data_dir.rglob(ext))
    
    logger.info(f"Found {len(image_paths)} images in {data_dir}")
    
    # Results storage
    results = {
        "total_images": len(image_paths),
        "valid_images": 0,
        "issues": {
            "corrupted": [],
            "too_small": [],
            "bad_aspect_ratio": [],
            "duplicates": [],
        },
        "class_distribution": defaultdict(int),
        "size_stats": {
            "widths": [],
            "heights": [],
        },
    }
    
    # Validate each image
    logger.info("Validating image integrity and dimensions...")
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Validating images"):
        # Get class from path (parent directory name)
        class_name = path.parent.name
        results["class_distribution"][class_name] += 1
        
        # Check integrity
        is_valid, error = verify_image_integrity(path)
        if not is_valid:
            results["issues"]["corrupted"].append({
                "path": str(path),
                "error": error,
            })
            continue
        
        # Check dimensions
        is_valid, issue_type, metadata = check_image_dimensions(path)
        if not is_valid:
            results["issues"][issue_type].append({
                "path": str(path),
                "metadata": metadata,
            })
            continue
        
        # Track stats for valid images
        results["valid_images"] += 1
        results["size_stats"]["widths"].append(metadata["width"])
        results["size_stats"]["heights"].append(metadata["height"])
        valid_paths.append(path)
    
    # Check for duplicates
    if check_duplicates and valid_paths:
        duplicates = find_duplicates_phash(valid_paths, duplicate_threshold)
        results["issues"]["duplicates"] = [
            {
                "image1": str(d[0]),
                "image2": str(d[1]),
                "distance": d[2],
            }
            for d in duplicates
        ]
    
    # Compute size statistics
    if results["size_stats"]["widths"]:
        import numpy as np
        widths = np.array(results["size_stats"]["widths"])
        heights = np.array(results["size_stats"]["heights"])
        results["size_stats"] = {
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "height_mean": float(np.mean(heights)),
            "height_std": float(np.std(heights)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
        }
    
    return results


def print_validation_report(results: dict):
    """Print a summary of validation results.
    
    Args:
        results: Validation results dictionary.
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total images:  {results['total_images']}")
    print(f"   Valid images:  {results['valid_images']}")
    print(f"   Issues found:  {results['total_images'] - results['valid_images']}")
    
    print(f"\n📁 Class Distribution:")
    for class_name, count in sorted(results["class_distribution"].items()):
        print(f"   {class_name}: {count}")
    
    print(f"\n📐 Size Statistics (valid images):")
    stats = results.get("size_stats", {})
    if "width_mean" in stats:
        print(f"   Width:  {stats['width_min']}-{stats['width_max']} px (mean: {stats['width_mean']:.0f})")
        print(f"   Height: {stats['height_min']}-{stats['height_max']} px (mean: {stats['height_mean']:.0f})")
    
    print(f"\n⚠️  Issues Found:")
    issues = results["issues"]
    
    n_corrupted = len(issues.get("corrupted", []))
    n_small = len(issues.get("too_small", []))
    n_aspect = len(issues.get("bad_aspect_ratio", []))
    n_duplicates = len(issues.get("duplicates", []))
    
    print(f"   Corrupted files:     {n_corrupted}")
    print(f"   Too small (<{MIN_IMAGE_SIZE}px): {n_small}")
    print(f"   Bad aspect ratio:    {n_aspect}")
    print(f"   Duplicate pairs:     {n_duplicates}")
    
    # Show details if issues exist
    if n_corrupted > 0 and n_corrupted <= 10:
        print(f"\n   Corrupted files:")
        for item in issues["corrupted"]:
            print(f"     - {item['path']}")
    
    if n_small > 0 and n_small <= 10:
        print(f"\n   Too small files:")
        for item in issues["too_small"]:
            meta = item['metadata']
            print(f"     - {item['path']} ({meta['width']}x{meta['height']})")
    
    if n_duplicates > 0 and n_duplicates <= 10:
        print(f"\n   Duplicate pairs:")
        for item in issues["duplicates"]:
            print(f"     - {Path(item['image1']).name} <-> {Path(item['image2']).name} (dist={item['distance']})")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    if results['valid_images'] == results['total_images']:
        print("✅ All images passed validation!")
    else:
        print("📋 Recommendations:")
        if n_corrupted > 0:
            print("   - Remove or replace corrupted files")
        if n_small > 0:
            print("   - Remove images smaller than 100x100 pixels")
        if n_duplicates > 0:
            print("   - Review and remove duplicate images to prevent data leakage")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate chest X-ray dataset for quality issues"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/chest-xray-dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip duplicate detection (faster)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=5,
        help="Hash distance threshold for duplicate detection (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    args = parser.parse_args()
    
    # Run validation
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    results = validate_dataset(
        data_dir=data_dir,
        check_duplicates=not args.no_duplicates,
        duplicate_threshold=args.duplicate_threshold,
    )
    
    # Print report
    print_validation_report(results)
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for JSON
        results["class_distribution"] = dict(results["class_distribution"])
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()


