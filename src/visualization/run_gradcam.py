"""
Generate Grad-CAM visualizations for chest X-ray classification.

Creates attention heatmaps showing which regions of the image
the model focuses on when making predictions.
"""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.config import get_device, load_config, set_seed
from src.data.transforms import get_inference_transforms, denormalize
from src.models.architecture import load_model
from src.visualization.visualize import GradCAM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_target_layer(model):
    """Get the target convolutional layer for Grad-CAM.
    
    For EfficientNet, we target the last conv layer in the backbone.
    
    Args:
        model: The model to extract layer from.
        
    Returns:
        The target layer for Grad-CAM.
    """
    target_layer = None
    
    # Find the last Conv2d layer in the backbone
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            layer_name = name
    
    if target_layer is not None:
        logger.info(f"Using target layer: {layer_name}")
    
    return target_layer


def generate_gradcam_grid(
    model,
    image_paths: list[Path],
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
    save_path: Path = None,
):
    """Generate a grid of Grad-CAM visualizations.
    
    Args:
        model: Trained model.
        image_paths: List of image paths to visualize.
        class_names: List of class names.
        device: Device for inference.
        image_size: Image size for preprocessing.
        save_path: Path to save the figure.
    """
    n_images = len(image_paths)
    
    # Get target layer
    target_layer = get_target_layer(model)
    if target_layer is None:
        raise ValueError("Could not find target convolution layer")
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    # Prepare transforms
    transform = get_inference_transforms(image_size=image_size)
    
    # Create figure
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="Generating Grad-CAM")):
        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
        
        # Generate Grad-CAM
        cam = gradcam.generate(input_tensor, pred_class)
        
        # Resize CAM
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image_size, image_size),
                Image.BILINEAR,
            )
        ) / 255.0
        
        # Prepare image for display
        img_display = np.array(image.resize((image_size, image_size)))
        img_normalized = img_display.astype(np.float32) / 255.0
        
        # Get true class from path
        true_class = image_path.parent.name.capitalize()
        if true_class.lower() == "normal":
            true_idx = 0
        elif true_class.lower() == "pneumonia":
            true_idx = 1
        else:
            true_idx = 2
        
        is_correct = pred_class == true_idx
        color = "green" if is_correct else "red"
        
        # Plot original
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f"Original\nTrue: {true_class}", fontsize=10)
        axes[idx, 0].axis("off")
        
        # Plot Grad-CAM heatmap
        axes[idx, 1].imshow(cam_resized, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[idx, 1].axis("off")
        
        # Plot overlay
        heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.5 * img_normalized + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay", fontsize=10)
        axes[idx, 2].axis("off")
        
        # Plot probability distribution
        y_pos = np.arange(len(class_names))
        probs_np = probs.cpu().numpy()
        colors = ['#2ecc71' if i == pred_class else '#3498db' for i in range(len(class_names))]
        
        axes[idx, 3].barh(y_pos, probs_np, color=colors)
        axes[idx, 3].set_yticks(y_pos)
        axes[idx, 3].set_yticklabels(class_names)
        axes[idx, 3].set_xlim(0, 1)
        axes[idx, 3].set_title(
            f"Pred: {class_names[pred_class]} ({probs_np[pred_class]:.1%})",
            fontsize=10,
            color=color,
        )
        axes[idx, 3].set_xlabel("Probability")
    
    plt.suptitle(
        "Grad-CAM Explainability Analysis\n"
        "(Bright regions = high importance for prediction)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Grad-CAM grid to: {save_path}")
    
    plt.close()


def get_sample_images(data_dir: Path, n_per_class: int = 2) -> list[Path]:
    """Get sample images from each class.
    
    Args:
        data_dir: Path to dataset directory.
        n_per_class: Number of images to sample per class.
        
    Returns:
        List of image paths.
    """
    sample_paths = []
    class_folders = ["normal", "pneumonia", "tuberculosis"]
    
    # Look in test folder
    test_dir = data_dir / "test"
    if not test_dir.exists():
        test_dir = data_dir
    
    for class_name in class_folders:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # Get all images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if len(images) >= n_per_class:
            sampled = random.sample(images, n_per_class)
        else:
            sampled = images
        
        sample_paths.extend(sampled)
        logger.info(f"Sampled {len(sampled)} images from {class_name}")
    
    return sample_paths


def main():
    """Main entry point for Grad-CAM visualization."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for chest X-ray classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/chest-xray-dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=3,
        help="Number of images to sample per class",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a specific image (overrides random sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.model_path,
        device=device,
        architecture=config.get("model", {}).get("architecture", "efficientnet_b0"),
        num_classes=config.get("data", {}).get("num_classes", 3),
    )
    model.eval()
    
    # Class names
    class_names = config.get("data", {}).get("classes", ["Normal", "Pneumonia", "Tuberculosis"])
    
    # Get image paths
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = get_sample_images(
            Path(args.data_dir),
            n_per_class=args.n_per_class,
        )
    
    if not image_paths:
        logger.error("No images found!")
        return
    
    logger.info(f"Generating Grad-CAM for {len(image_paths)} images...")
    
    # Generate visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_gradcam_grid(
        model=model,
        image_paths=image_paths,
        class_names=class_names,
        device=device,
        image_size=config.get("data", {}).get("image_size", 224),
        save_path=output_dir / "gradcam_analysis.png",
    )
    
    logger.info("Done! Grad-CAM visualizations saved.")


if __name__ == "__main__":
    main()


