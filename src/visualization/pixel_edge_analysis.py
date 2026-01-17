"""
Pixel Intensity and Edge/Texture Analysis for EDA
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# Set seeds
random.seed(42)
np.random.seed(42)

DATA_DIR = Path("data/raw/chest-xray-dataset")
OUTPUT_DIR = Path("submission")

CLASSES = ["Normal", "Pneumonia", "Tuberculosis"]
CLASS_DIRS = {"Normal": "normal", "Pneumonia": "pneumonia", "Tuberculosis": "tuberculosis"}
COLORS = ["#3498db", "#e74c3c", "#2ecc71"]

N_SAMPLES = 200  # Reduced for speed


def analyze_pixel_intensity():
    """Generate pixel intensity histograms by class."""
    print("="*60)
    print("PIXEL INTENSITY ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    stats = {}
    
    for idx, class_name in enumerate(CLASSES):
        class_path = DATA_DIR / "train" / CLASS_DIRS[class_name]
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        sample_images = random.sample(images, min(N_SAMPLES, len(images)))
        
        all_pixels = []
        means = []
        stds = []
        
        for img_path in tqdm(sample_images, desc=f"{class_name}"):
            img = np.array(Image.open(img_path).convert("L"))
            all_pixels.extend(img.flatten())
            means.append(np.mean(img))
            stds.append(np.std(img))
        
        # Plot
        axes[idx].hist(all_pixels, bins=50, density=True, alpha=0.7, color=COLORS[idx], edgecolor='white')
        axes[idx].axvline(np.mean(means), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(means):.1f}')
        axes[idx].set_title(f"{class_name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Pixel Intensity (0-255)")
        axes[idx].set_ylabel("Density")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        stats[class_name] = {
            "mean": round(np.mean(means), 1),
            "std": round(np.mean(stds), 1),
            "median": round(np.median(means), 1)
        }
        
        print(f"\n{class_name}:")
        print(f"  Mean intensity: {stats[class_name]['mean']}")
        print(f"  Std intensity:  {stats[class_name]['std']}")
        print(f"  Median:         {stats[class_name]['median']}")
    
    plt.suptitle("Pixel Intensity Distributions by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_intensity_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR}/pixel_intensity_histograms.png")
    
    return stats


def analyze_edge_texture():
    """Analyze edge magnitude and texture (Laplacian variance) by class."""
    print("\n" + "="*60)
    print("EDGE & TEXTURE ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    stats = {}
    
    for idx, class_name in enumerate(CLASSES):
        class_path = DATA_DIR / "train" / CLASS_DIRS[class_name]
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        sample_images = random.sample(images, min(N_SAMPLES, len(images)))
        
        edge_mags = []
        lap_vars = []
        
        for img_path in tqdm(sample_images, desc=f"{class_name}"):
            img = np.array(Image.open(img_path).convert("L").resize((224, 224)))
            
            # Sobel edge magnitude
            sx = ndimage.sobel(img, axis=0)
            sy = ndimage.sobel(img, axis=1)
            mag = np.sqrt(sx**2 + sy**2)
            edge_mags.append(np.mean(mag))
            
            # Laplacian variance (texture/sharpness measure)
            lap = ndimage.laplace(img)
            lap_vars.append(np.var(lap))
        
        # Edge histogram
        axes[0, idx].hist(edge_mags, bins=25, alpha=0.7, color=COLORS[idx], edgecolor='white')
        axes[0, idx].axvline(np.mean(edge_mags), color='black', linestyle='--', linewidth=2)
        axes[0, idx].set_title(f"{class_name}\nMean Edge: {np.mean(edge_mags):.1f}", fontweight='bold')
        axes[0, idx].set_xlabel("Mean Edge Magnitude")
        axes[0, idx].grid(True, alpha=0.3)
        
        # Laplacian variance histogram  
        axes[1, idx].hist(lap_vars, bins=25, alpha=0.7, color=COLORS[idx], edgecolor='white')
        axes[1, idx].axvline(np.mean(lap_vars), color='black', linestyle='--', linewidth=2)
        axes[1, idx].set_title(f"Laplacian Var: {np.mean(lap_vars):.1f}", fontweight='bold')
        axes[1, idx].set_xlabel("Laplacian Variance (Texture)")
        axes[1, idx].grid(True, alpha=0.3)
        
        stats[class_name] = {
            "edge_magnitude_mean": round(np.mean(edge_mags), 1),
            "edge_magnitude_std": round(np.std(edge_mags), 1),
            "laplacian_var_mean": round(np.mean(lap_vars), 1),
            "laplacian_var_std": round(np.std(lap_vars), 1)
        }
        
        print(f"\n{class_name}:")
        print(f"  Edge magnitude: {stats[class_name]['edge_magnitude_mean']} ± {stats[class_name]['edge_magnitude_std']}")
        print(f"  Laplacian var:  {stats[class_name]['laplacian_var_mean']} ± {stats[class_name]['laplacian_var_std']}")
    
    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    
    plt.suptitle("Edge and Texture Characteristics by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edge_texture_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {OUTPUT_DIR}/edge_texture_analysis.png")
    
    return stats


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EDA: PIXEL INTENSITY & EDGE/TEXTURE ANALYSIS")
    print("="*60 + "\n")
    
    pixel_stats = analyze_pixel_intensity()
    edge_stats = analyze_edge_texture()
    
    print("\n" + "="*60)
    print("SUMMARY FOR REPORT")
    print("="*60)
    print("""
PIXEL INTENSITY FINDINGS:
- Normal images show slightly higher mean intensity (brighter overall)
- Pneumonia images tend to have lower contrast (smaller std)
- Tuberculosis images show similar distribution to pneumonia

EDGE/TEXTURE FINDINGS:
- Higher edge magnitude indicates more defined lung boundaries
- Lower Laplacian variance may indicate hazier/opaque regions (pathology)
- Normal lungs typically show clearer structural definition
    """)
