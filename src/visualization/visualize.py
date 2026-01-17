"""
Visualization utilities for training and model interpretability.

Includes:
- Training history plots
- Sample predictions visualization
- Grad-CAM attention maps
- Layer-CAM attention maps
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


class LayerCAM:
    """Layer-CAM implementation for model interpretability.
    
    Layer-CAM uses element-wise multiplication of positive gradients 
    with activations (spatial weighting), providing finer-grained 
    attention maps compared to Grad-CAM's global average pooling.
    
    Reference: Jiang et al., "LayerCAM: Exploring Hierarchical Class 
    Activation Maps for Localization" (2021)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ):
        """Initialize Layer-CAM.
        
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
        """Generate Layer-CAM heatmap.
        
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
        
        # Compute Layer-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Layer-CAM: element-wise multiplication with positive gradients
        # This preserves spatial information better than Grad-CAM
        positive_gradients = F.relu(gradients)
        
        # Element-wise weighted combination (spatial weighting)
        cam = (positive_gradients * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)  # ReLU to keep only positive contributions
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


class ScoreCAM:
    """Score-CAM implementation for gradient-free model interpretability.
    
    Score-CAM uses activation maps as masks and measures the change in
    output score, avoiding gradient noise issues.
    
    Note: This is slower than Grad-CAM/Layer-CAM but can be more stable.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ):
        """Initialize Score-CAM.
        
        Args:
            model: The model to explain.
            target_layer: The convolutional layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register hook
        target_layer.register_forward_hook(self._save_activation)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Generate Score-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.
            batch_size: Batch size for processing activation masks.
            
        Returns:
            Heatmap array of shape (H, W).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass to get activations and prediction
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            activations = self.activations[0]  # (C, H, W)
            n_channels = activations.shape[0]
            
            # Upsample activations to input size
            h, w = input_tensor.shape[2:]
            upsampled_acts = F.interpolate(
                activations.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )[0]  # (C, H, W)
            
            # Normalize each channel to [0, 1]
            upsampled_acts = upsampled_acts - upsampled_acts.view(n_channels, -1).min(dim=1)[0].view(n_channels, 1, 1)
            upsampled_acts = upsampled_acts / (upsampled_acts.view(n_channels, -1).max(dim=1)[0].view(n_channels, 1, 1) + 1e-8)
            
            # Compute scores for each channel
            scores = torch.zeros(n_channels, device=input_tensor.device)
            
            for i in range(0, n_channels, batch_size):
                batch_end = min(i + batch_size, n_channels)
                masks = upsampled_acts[i:batch_end].unsqueeze(1)  # (B, 1, H, W)
                
                # Apply mask to input
                masked_inputs = input_tensor * masks  # (B, C, H, W)
                
                # Get predictions
                outputs = self.model(masked_inputs)
                probs = F.softmax(outputs, dim=1)
                scores[i:batch_end] = probs[:, target_class]
            
            # Normalize scores
            scores = F.relu(scores)
            scores = scores / (scores.sum() + 1e-8)
            
            # Weighted sum of activations
            cam = (scores.view(-1, 1, 1) * upsampled_acts).sum(dim=0)
            cam = F.relu(cam)
            
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


def visualize_cam_comparison(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
    target_class: int | None = None,
    methods: list[str] = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Generate and compare multiple CAM methods for an image.
    
    Args:
        model: Trained model.
        image_path: Path to the input image.
        class_names: List of class names.
        device: Device for inference.
        image_size: Image size for preprocessing.
        target_class: Optional target class for CAM.
        methods: List of methods to compare. Default: ["gradcam", "layercam"]
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        
    Returns:
        Dictionary with prediction info and CAM results.
    """
    from src.data.transforms import get_inference_transforms
    
    if methods is None:
        methods = ["gradcam", "layercam"]
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    transform = get_inference_transforms(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get the target layer (last conv layer of backbone)
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise ValueError("Could not find target convolution layer")
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
    
    target = target_class if target_class is not None else pred_class
    
    # Generate CAMs for each method
    cam_results = {}
    
    for method in methods:
        if method.lower() == "gradcam":
            cam_generator = GradCAM(model, target_layer)
        elif method.lower() == "layercam":
            cam_generator = LayerCAM(model, target_layer)
        elif method.lower() == "scorecam":
            cam_generator = ScoreCAM(model, target_layer)
        else:
            raise ValueError(f"Unknown CAM method: {method}")
        
        cam = cam_generator.generate(input_tensor, target)
        
        # Resize CAM to image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image_size, image_size),
                Image.BILINEAR,
            )
        ) / 255.0
        
        cam_results[method] = cam_resized
    
    # Create visualization
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Original image
    img_display = np.array(image.resize((image_size, image_size)))
    img_normalized = img_display.astype(np.float32) / 255.0
    
    # Top row: Original + CAM heatmaps
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    for idx, method in enumerate(methods):
        axes[0, idx + 1].imshow(cam_results[method], cmap="jet")
        axes[0, idx + 1].set_title(f"{method.upper()}", fontsize=12, fontweight='bold')
        axes[0, idx + 1].axis("off")
    
    # Bottom row: Prediction info + overlays
    axes[1, 0].text(
        0.5, 0.5,
        f"Prediction: {class_names[pred_class]}\n"
        f"Confidence: {probs[pred_class]:.2%}\n\n"
        f"Class Probabilities:\n" +
        "\n".join([f"  {name}: {probs[i]:.2%}" for i, name in enumerate(class_names)]),
        ha='center', va='center',
        fontsize=11,
        transform=axes[1, 0].transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    axes[1, 0].axis("off")
    
    for idx, method in enumerate(methods):
        heatmap_colored = plt.cm.jet(cam_results[method])[:, :, :3]
        overlay = 0.6 * img_normalized + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title(f"{method.upper()} Overlay", fontsize=12)
        axes[1, idx + 1].axis("off")
    
    plt.suptitle(
        f"CAM Comparison - Target: {class_names[target]}",
        fontsize=14, fontweight='bold', y=1.02,
    )
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
        "cam_results": cam_results,
    }


def generate_multi_cam_analysis(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    class_names: list[str],
    device: torch.device,
    image_size: int = 224,
    methods: list[str] = None,
    save_path: str | Path | None = None,
) -> None:
    """Generate a comprehensive CAM analysis grid for multiple images.
    
    Args:
        model: Trained model.
        image_paths: List of paths to input images.
        class_names: List of class names.
        device: Device for inference.
        image_size: Image size for preprocessing.
        methods: CAM methods to use. Default: ["gradcam", "layercam"]
        save_path: Path to save the figure.
    """
    from src.data.transforms import get_inference_transforms
    
    if methods is None:
        methods = ["gradcam", "layercam"]
    
    n_images = len(image_paths)
    n_cols = 1 + len(methods) * 2  # Original + (heatmap + overlay) for each method
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    transform = get_inference_transforms(image_size=image_size)
    
    # Get target layer
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    for row_idx, image_path in enumerate(image_paths):
        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
        
        img_display = np.array(image.resize((image_size, image_size)))
        img_normalized = img_display.astype(np.float32) / 255.0
        
        # Original image
        axes[row_idx, 0].imshow(img_display)
        axes[row_idx, 0].set_title(
            f"True: {Path(image_path).parent.name}\n"
            f"Pred: {class_names[pred_class]} ({probs[pred_class]:.0%})",
            fontsize=10,
        )
        axes[row_idx, 0].axis("off")
        
        # Generate CAMs
        col_idx = 1
        for method in methods:
            if method.lower() == "gradcam":
                cam_generator = GradCAM(model, target_layer)
            elif method.lower() == "layercam":
                cam_generator = LayerCAM(model, target_layer)
            else:
                cam_generator = ScoreCAM(model, target_layer)
            
            cam = cam_generator.generate(input_tensor, pred_class)
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (image_size, image_size), Image.BILINEAR,
                )
            ) / 255.0
            
            # Heatmap
            axes[row_idx, col_idx].imshow(cam_resized, cmap="jet")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"{method.upper()}", fontsize=11, fontweight='bold')
            axes[row_idx, col_idx].axis("off")
            col_idx += 1
            
            # Overlay
            heatmap_colored = plt.cm.jet(cam_resized)[:, :, :3]
            overlay = 0.6 * img_normalized + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            axes[row_idx, col_idx].imshow(overlay)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"{method.upper()} Overlay", fontsize=11)
            axes[row_idx, col_idx].axis("off")
            col_idx += 1
    
    plt.suptitle("Multi-Method CAM Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved CAM analysis to: {save_path}")
    
    plt.close()

