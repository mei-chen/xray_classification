"""
Extended EDA Analysis for Report
Generates: pixel histograms, edge analysis, misclassified examples, calibration diagrams
"""

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_pixel_intensity(data_dir: Path, output_dir: Path, n_samples: int = 500):
    """Generate pixel intensity histograms by class."""
    logger.info("Analyzing pixel intensity distributions...")
    
    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    class_dirs = {
        "Normal": "normal",
        "Pneumonia": "pneumonia", 
        "Tuberculosis": "tuberculosis"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    all_means = {c: [] for c in classes}
    all_stds = {c: [] for c in classes}
    
    for idx, class_name in enumerate(classes):
        class_path = data_dir / "train" / class_dirs[class_name]
        if not class_path.exists():
            continue
            
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        sample_images = random.sample(images, min(n_samples, len(images)))
        
        all_pixels = []
        for img_path in tqdm(sample_images, desc=f"Processing {class_name}"):
            try:
                img = Image.open(img_path).convert("L")  # Grayscale
                pixels = np.array(img).flatten()
                all_pixels.extend(pixels)
                all_means[class_name].append(np.mean(pixels))
                all_stds[class_name].append(np.std(pixels))
            except Exception as e:
                continue
        
        # Plot histogram
        axes[idx].hist(all_pixels, bins=50, density=True, alpha=0.7, color=colors[idx])
        axes[idx].set_title(f"{class_name}\nMean: {np.mean(all_means[class_name]):.1f}, Std: {np.mean(all_stds[class_name]):.1f}")
        axes[idx].set_xlabel("Pixel Intensity (0-255)")
        axes[idx].set_ylabel("Density")
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle("Pixel Intensity Distributions by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_dir / "pixel_intensity_histograms.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pixel intensity histograms to {save_path}")
    
    # Return statistics
    return {
        class_name: {
            "mean_intensity": float(np.mean(all_means[class_name])),
            "std_intensity": float(np.mean(all_stds[class_name]))
        }
        for class_name in classes
    }


def analyze_edge_texture(data_dir: Path, output_dir: Path, n_samples: int = 100):
    """Analyze edge and texture characteristics using Sobel gradients."""
    logger.info("Analyzing edge and texture characteristics...")
    
    from scipy import ndimage
    
    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    class_dirs = {"Normal": "normal", "Pneumonia": "pneumonia", "Tuberculosis": "tuberculosis"}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    edge_stats = {}
    
    for idx, class_name in enumerate(classes):
        class_path = data_dir / "train" / class_dirs[class_name]
        if not class_path.exists():
            continue
            
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        sample_images = random.sample(images, min(n_samples, len(images)))
        
        edge_magnitudes = []
        laplacian_vars = []
        
        for img_path in tqdm(sample_images, desc=f"Edge analysis {class_name}"):
            try:
                img = np.array(Image.open(img_path).convert("L").resize((224, 224)))
                
                # Sobel edges
                sobel_x = ndimage.sobel(img, axis=0)
                sobel_y = ndimage.sobel(img, axis=1)
                magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_magnitudes.append(np.mean(magnitude))
                
                # Laplacian variance (texture/sharpness)
                laplacian = ndimage.laplace(img)
                laplacian_vars.append(np.var(laplacian))
            except:
                continue
        
        edge_stats[class_name] = {
            "mean_edge_magnitude": float(np.mean(edge_magnitudes)),
            "std_edge_magnitude": float(np.std(edge_magnitudes)),
            "mean_laplacian_var": float(np.mean(laplacian_vars)),
        }
        
        # Edge magnitude histogram
        axes[0, idx].hist(edge_magnitudes, bins=30, alpha=0.7, color=colors[idx])
        axes[0, idx].set_title(f"{class_name}\nMean Edge: {np.mean(edge_magnitudes):.1f}")
        axes[0, idx].set_xlabel("Mean Edge Magnitude")
        axes[0, idx].grid(True, alpha=0.3)
        
        # Laplacian variance histogram
        axes[1, idx].hist(laplacian_vars, bins=30, alpha=0.7, color=colors[idx])
        axes[1, idx].set_title(f"Laplacian Var: {np.mean(laplacian_vars):.1f}")
        axes[1, idx].set_xlabel("Laplacian Variance (Texture)")
        axes[1, idx].grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    
    plt.suptitle("Edge and Texture Analysis by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_dir / "edge_texture_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved edge/texture analysis to {save_path}")
    
    return edge_stats


def generate_calibration_diagram(
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
):
    """Generate reliability diagrams (calibration curves)."""
    logger.info("Generating calibration reliability diagrams...")
    
    from src.data.transforms import get_inference_transforms
    from src.models.architecture import load_model
    
    # Load model
    model = load_model(model_path, device=device)
    model.eval()
    
    transform = get_inference_transforms(image_size=224)
    
    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    class_dirs = {"Normal": "normal", "Pneumonia": "pneumonia", "Tuberculosis": "tuberculosis"}
    
    all_probs = []
    all_labels = []
    
    # Get predictions on test set
    test_dir = data_dir / "test"
    for class_idx, class_name in enumerate(classes):
        class_path = test_dir / class_dirs[class_name]
        if not class_path.exists():
            continue
            
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        
        for img_path in tqdm(images, desc=f"Evaluating {class_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)[0].cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(class_idx)
            except:
                continue
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Plot calibration curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    ece_scores = {}
    
    for idx, class_name in enumerate(classes):
        binary_labels = (all_labels == idx).astype(int)
        class_probs = all_probs[:, idx]
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(binary_labels, class_probs, n_bins=10, strategy='uniform')
        
        axes[idx].plot(prob_pred, prob_true, 's-', color=colors[idx], label='Model', linewidth=2, markersize=8)
        axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[idx].fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color=colors[idx])
        
        # Calculate ECE
        ece = np.mean(np.abs(prob_pred - prob_true))
        ece_scores[class_name] = float(ece)
        
        axes[idx].set_title(f"{class_name}\nECE = {ece:.3f}")
        axes[idx].set_xlabel("Mean Predicted Probability")
        axes[idx].set_ylabel("Fraction of Positives")
        axes[idx].legend(loc="lower right")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 1])
        axes[idx].set_ylim([0, 1])
    
    plt.suptitle("Reliability Diagrams (Calibration Curves)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_dir / "calibration_reliability_diagrams.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved calibration diagrams to {save_path}")
    
    ece_scores["macro_ece"] = float(np.mean(list(ece_scores.values())))
    return ece_scores, all_probs, all_labels


def generate_misclassified_examples(
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_examples: int = 4,
):
    """Generate visualization of misclassified examples."""
    logger.info("Finding misclassified examples...")
    
    from src.data.transforms import get_inference_transforms
    from src.models.architecture import load_model
    
    model = load_model(model_path, device=device)
    model.eval()
    
    transform = get_inference_transforms(image_size=224)
    
    classes = ["Normal", "Pneumonia", "Tuberculosis"]
    class_dirs = {"Normal": "normal", "Pneumonia": "pneumonia", "Tuberculosis": "tuberculosis"}
    
    # Collect misclassified examples
    misclassified = {f"{true_class}→{pred_class}": [] 
                     for true_class in classes 
                     for pred_class in classes 
                     if true_class != pred_class}
    
    test_dir = data_dir / "test"
    for true_idx, true_class in enumerate(classes):
        class_path = test_dir / class_dirs[true_class]
        if not class_path.exists():
            continue
            
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        
        for img_path in tqdm(images, desc=f"Checking {true_class}"):
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)[0]
                    pred_idx = output.argmax(dim=1).item()
                    confidence = probs[pred_idx].item()
                
                if pred_idx != true_idx:
                    key = f"{true_class}→{classes[pred_idx]}"
                    misclassified[key].append({
                        "path": str(img_path),
                        "true_class": true_class,
                        "pred_class": classes[pred_idx],
                        "confidence": confidence,
                        "probs": probs.cpu().numpy()
                    })
            except:
                continue
    
    # Select most confident misclassifications for visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Find top error types
    error_counts = {k: len(v) for k, v in misclassified.items()}
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for row, (error_type, count) in enumerate(top_errors):
        examples = sorted(misclassified[error_type], key=lambda x: x["confidence"], reverse=True)[:4]
        
        for col, example in enumerate(examples):
            ax = axes[row, col]
            img = Image.open(example["path"]).convert("RGB").resize((224, 224))
            ax.imshow(img)
            ax.set_title(f"True: {example['true_class']}\nPred: {example['pred_class']} ({example['confidence']:.1%})", 
                        fontsize=10)
            ax.axis("off")
        
        # Clear unused axes
        for col in range(len(examples), 4):
            axes[row, col].axis("off")
    
    plt.suptitle("Top Misclassification Patterns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_dir / "misclassified_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved misclassified examples to {save_path}")
    
    return {k: len(v) for k, v in misclassified.items()}


def main():
    parser = argparse.ArgumentParser(description="Extended EDA Analysis")
    parser.add_argument("--data-dir", type=str, default="data/raw/chest-xray-dataset")
    parser.add_argument("--model-path", type=str, default="models/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="submission")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    random.seed(42)
    np.random.seed(42)
    
    results = {}
    
    # 1. Pixel intensity analysis
    results["pixel_intensity"] = analyze_pixel_intensity(data_dir, output_dir)
    
    # 2. Edge/texture analysis
    results["edge_texture"] = analyze_edge_texture(data_dir, output_dir)
    
    # 3. Calibration diagrams
    if Path(args.model_path).exists():
        results["calibration"], _, _ = generate_calibration_diagram(
            args.model_path, data_dir, output_dir, device
        )
        
        # 4. Misclassified examples
        results["misclassification_counts"] = generate_misclassified_examples(
            args.model_path, data_dir, output_dir, device
        )
    
    # Save results
    with open(output_dir / "eda_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("EDA analysis complete!")
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
