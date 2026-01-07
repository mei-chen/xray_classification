"""
Visualization utilities for training and model interpretability.

Includes:
- Training history plots
- Sample predictions visualization
- Grad-CAM attention maps
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data.transforms import denormalize


def plot_training_history(
    history: dict[str, list[float]] | str | Path,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot training history curves.
    
    Args:
        history: Training history dict or path to JSON file.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    """
    if isinstance(history, (str, Path)):
        with open(history, "r") as f:
            history = json.load(f)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[1].plot(epochs, history["train_f1"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], "r-", label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score (Macro)")
    axes[1].set_title("Training and Validation F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history["learning_rate"], "g-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    num_samples: int = 12,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Visualize model predictions on sample images.
    
    Args:
        model: Trained model.
        dataloader: Data loader to sample from.
        class_names: List of class names.
        device: Device for inference.
        num_samples: Number of samples to visualize.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    """
    model.eval()
    
    # Get samples
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()
    
    # Create grid
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalize and convert to displayable format
        img = denormalize(images[idx])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Get prediction info
        pred_label = preds[idx].item()
        true_label = labels[idx].item()
        confidence = probs[idx, pred_label].item()
        
        # Set border color based on correctness
        is_correct = pred_label == true_label
        border_color = "green" if is_correct else "red"
        
        ax.imshow(img)
        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]} ({confidence:.2%})",
            fontsize=10,
            color=border_color,
        )
        ax.axis("off")
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


class GradCAM:
    """Grad-CAM implementation for model interpretability.
    
    Generates class activation maps showing which regions
    of the image contributed most to the prediction.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ):
        """Initialize Grad-CAM.
        
        Args:
            model: The model to explain.
            target_layer: The convolutional layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Heatmap array of shape (H, W).
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)  # ReLU
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def visualize_gradcam(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
    target_class: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Generate and visualize Grad-CAM for an image.
    
    Args:
        model: Trained model.
        image_path: Path to the input image.
        class_names: List of class names.
        device: Device for inference.
        image_size: Image size for preprocessing.
        target_class: Optional target class for Grad-CAM.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        
    Returns:
        Dictionary with prediction info and Grad-CAM.
    """
    from src.data.transforms import get_inference_transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    transform = get_inference_transforms(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get the target layer (last conv layer of backbone)
    # For EfficientNet, this is typically in the final block
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise ValueError("Could not find target convolution layer")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
    
    # Generate CAM for target class (or predicted class)
    target = target_class if target_class is not None else pred_class
    cam = gradcam.generate(input_tensor, target)
    
    # Resize CAM to original image size
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image_size, image_size),
            Image.BILINEAR,
        )
    ) / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_display = np.array(image.resize((image_size, image_size)))
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    
    # Grad-CAM heatmap
    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM: {class_names[target]}", fontsize=12)
    axes[1].axis("off")
    
    # Overlay
    img_normalized = img_display.astype(np.float32) / 255.0
    heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = 0.6 * img_normalized + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {class_names[pred_class]} ({probs[pred_class]:.2%})",
        fontsize=12,
    )
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        "predicted_class": class_names[pred_class],
        "predicted_index": pred_class,
        "confidence": probs[pred_class].item(),
        "probabilities": {
            name: probs[i].item()
            for i, name in enumerate(class_names)
        },
        "gradcam": cam_resized,
    }

