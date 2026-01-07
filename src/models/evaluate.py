"""
Model evaluation script with comprehensive metrics and visualizations.

Generates:
- Classification report with precision, recall, F1
- Confusion matrix
- ROC curves and AUC scores
- Grad-CAM visualizations for interpretability
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_device, load_config, set_seed
from src.data import create_dataloaders
from src.models.architecture import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        class_names: list[str],
        device: torch.device,
        output_dir: Path,
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained model to evaluate.
            test_loader: Test data loader.
            class_names: List of class names.
            device: Device for inference.
            output_dir: Directory to save evaluation results.
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def get_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all predictions and probabilities.
        
        Returns:
            Tuple of (predictions, labels, probabilities).
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> dict:
        """Compute all evaluation metrics.
        
        Args:
            preds: Predicted labels.
            labels: True labels.
            probs: Prediction probabilities.
            
        Returns:
            Dictionary with all metrics.
        """
        # Classification report
        report = classification_report(
            labels,
            preds,
            target_names=self.class_names,
            output_dict=True,
        )
        
        # Overall accuracy
        accuracy = (preds == labels).mean()
        
        # ROC AUC (one-vs-rest for multiclass)
        try:
            # Binarize labels for OvR
            n_classes = len(self.class_names)
            labels_onehot = np.eye(n_classes)[labels]
            auc_scores = {}
            for i, class_name in enumerate(self.class_names):
                auc_scores[class_name] = roc_auc_score(labels_onehot[:, i], probs[:, i])
            auc_macro = np.mean(list(auc_scores.values()))
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            auc_scores = {}
            auc_macro = None
        
        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "auc_scores": auc_scores,
            "auc_macro": auc_macro,
            "num_samples": len(labels),
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
    ):
        """Plot and save confusion matrix.
        
        Args:
            preds: Predicted labels.
            labels: True labels.
            normalize: Whether to normalize the matrix.
        """
        cm = confusion_matrix(labels, preds)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved confusion matrix to: {save_path}")
    
    def plot_roc_curves(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
    ):
        """Plot and save ROC curves for each class.
        
        Args:
            labels: True labels.
            probs: Prediction probabilities.
        """
        n_classes = len(self.class_names)
        labels_onehot = np.eye(n_classes)[labels]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
            auc = roc_auc_score(labels_onehot[:, i], probs[:, i])
            
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f"{class_name} (AUC = {auc:.3f})",
            )
        
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves (One-vs-Rest)", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "roc_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved ROC curves to: {save_path}")
    
    def plot_class_distribution(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
    ):
        """Plot true vs predicted class distribution.
        
        Args:
            labels: True labels.
            preds: Predicted labels.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        true_counts = np.bincount(labels, minlength=len(self.class_names))
        axes[0].bar(self.class_names, true_counts, color="steelblue", edgecolor="black")
        axes[0].set_title("True Class Distribution", fontsize=12)
        axes[0].set_ylabel("Count")
        for i, count in enumerate(true_counts):
            axes[0].text(i, count + 1, str(count), ha="center", fontsize=10)
        
        # Predicted distribution
        pred_counts = np.bincount(preds, minlength=len(self.class_names))
        axes[1].bar(self.class_names, pred_counts, color="coral", edgecolor="black")
        axes[1].set_title("Predicted Class Distribution", fontsize=12)
        axes[1].set_ylabel("Count")
        for i, count in enumerate(pred_counts):
            axes[1].text(i, count + 1, str(count), ha="center", fontsize=10)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved class distribution to: {save_path}")
    
    def run_evaluation(self) -> dict:
        """Run full evaluation pipeline.
        
        Returns:
            Dictionary with all evaluation results.
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        preds, labels, probs = self.get_predictions()
        
        # Compute metrics
        metrics = self.compute_metrics(preds, labels, probs)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Test samples: {metrics['num_samples']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['auc_macro']:
            logger.info(f"Macro AUC: {metrics['auc_macro']:.4f}")
        
        logger.info("\nPer-class metrics:")
        for class_name in self.class_names:
            report = metrics['classification_report'][class_name]
            logger.info(
                f"  {class_name}: "
                f"P={report['precision']:.3f}, "
                f"R={report['recall']:.3f}, "
                f"F1={report['f1-score']:.3f}"
            )
        
        logger.info("=" * 60)
        
        # Generate plots
        self.plot_confusion_matrix(preds, labels)
        self.plot_roc_curves(labels, probs)
        self.plot_class_distribution(labels, preds)
        
        # Save metrics to JSON
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {
            "accuracy": float(metrics["accuracy"]),
            "auc_macro": float(metrics["auc_macro"]) if metrics["auc_macro"] else None,
            "auc_scores": {k: float(v) for k, v in metrics["auc_scores"].items()},
            "num_samples": int(metrics["num_samples"]),
            "per_class": {
                name: {
                    "precision": float(metrics["classification_report"][name]["precision"]),
                    "recall": float(metrics["classification_report"][name]["recall"]),
                    "f1-score": float(metrics["classification_report"][name]["f1-score"]),
                    "support": int(metrics["classification_report"][name]["support"]),
                }
                for name in self.class_names
            },
        }
        
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Saved metrics to: {metrics_path}")
        
        return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Chest X-Ray Classifier")
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
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Create test dataloader
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("raw_data_path", "data/raw/chest-xray-dataset"))
    
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=data_config.get("batch_size", 32),
        image_size=data_config.get("image_size", 224),
        num_workers=data_config.get("num_workers", 4),
        seed=seed,
        use_weighted_sampling=False,
    )
    
    # Load model
    model = load_model(
        args.model_path,
        device=device,
        architecture=config.get("model", {}).get("architecture", "efficientnet_b0"),
        num_classes=data_config.get("num_classes", 3),
    )
    
    # Run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=Path(args.output_dir),
    )
    
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()

