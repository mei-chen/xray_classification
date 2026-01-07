"""
Training pipeline for chest X-ray classification.

Implements a complete training loop with:
- Mixed precision training
- Learning rate scheduling with warmup
- Early stopping
- Model checkpointing
- Comprehensive logging
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_device, load_config, set_seed
from src.data import create_dataloaders
from src.models.architecture import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for the chest X-ray classifier.
    
    Handles the complete training pipeline including optimization,
    validation, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device,
        class_names: list[str],
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration dictionary.
            device: Device to train on.
            class_names: List of class names for reporting.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.class_names = class_names
        
        # Training config
        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 50)
        self.use_amp = train_config.get("use_amp", True) and device.type == "cuda"
        self.gradient_clip = train_config.get("gradient_clip_val", 1.0)
        
        # Model config
        model_config = config.get("model", {})
        self.freeze_backbone = model_config.get("freeze_backbone", True)
        self.unfreeze_at_epoch = model_config.get("unfreeze_at_epoch", 5)
        
        # Setup loss function
        self._setup_loss_function(train_config.get("loss", {}))
        
        # Setup optimizer and scheduler
        self._setup_optimizer(train_config.get("optimizer", {}))
        self._setup_scheduler(train_config.get("scheduler", {}))
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        es_config = train_config.get("early_stopping", {})
        self.early_stopping_enabled = es_config.get("enabled", True)
        self.patience = es_config.get("patience", 10)
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("paths", {}).get("model_dir", "models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "learning_rate": [],
        }
    
    def _setup_loss_function(self, loss_config: dict):
        """Setup the loss function with optional class weights."""
        label_smoothing = loss_config.get("label_smoothing", 0.1)
        use_class_weights = loss_config.get("use_class_weights", True)
        
        weight = None
        if use_class_weights:
            # Get class weights from training dataset
            weight = self.train_loader.dataset.get_class_weights().to(self.device)
            logger.info(f"Using class weights: {weight.tolist()}")
        
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
    
    def _setup_optimizer(self, opt_config: dict):
        """Setup the optimizer."""
        self.learning_rate = opt_config.get("learning_rate", 0.001)
        weight_decay = opt_config.get("weight_decay", 0.01)
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    
    def _setup_scheduler(self, sched_config: dict):
        """Setup learning rate scheduler with warmup."""
        warmup_epochs = sched_config.get("warmup_epochs", 3)
        min_lr = sched_config.get("min_lr", 1e-6)
        
        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = self.epochs * len(self.train_loader)
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps - warmup_steps,
            eta_min=min_lr,
        )
        
        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Tuple of (average loss, F1 score).
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return avg_loss, f1
    
    @torch.no_grad()
    def validate(self) -> tuple[float, float, dict]:
        """Validate the model.
        
        Returns:
            Tuple of (average loss, F1 score, classification report).
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        # Full classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True,
        )
        
        return avg_loss, f1, report
    
    def save_checkpoint(
        self,
        epoch: int,
        val_f1: float,
        is_best: bool = False,
    ):
        """Save a training checkpoint.
        
        Args:
            epoch: Current epoch.
            val_f1: Validation F1 score.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_f1": val_f1,
            "config": {
                "architecture": self.config.get("model", {}).get("architecture"),
                "num_classes": self.config.get("data", {}).get("num_classes", 3),
                "dropout": self.config.get("model", {}).get("dropout"),
            },
            "class_names": self.class_names,
            "history": self.history,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_f1={val_f1:.4f}")
    
    def train(self) -> dict:
        """Run the full training loop.
        
        Returns:
            Training history dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Progressive unfreezing
            if self.freeze_backbone and epoch == self.unfreeze_at_epoch:
                logger.info(f"Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
                # Reset optimizer for unfrozen parameters
                self._setup_optimizer(self.config.get("training", {}).get("optimizer", {}))
            
            # Training
            train_loss, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_f1, report = self.validate()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)
            self.history["learning_rate"].append(current_lr)
            
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Checkpointing
            is_best = val_f1 > self.best_metric
            if is_best:
                self.best_metric = val_f1
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch, val_f1, is_best)
            
            # Early stopping
            if self.early_stopping_enabled:
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info(f"Best Val F1: {self.best_metric:.4f}")
        logger.info(f"Training Time: {training_time/60:.2f} minutes")
        logger.info("=" * 60)
        
        # Save final history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("raw_data_path", "data/raw/chest-xray-dataset"))
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=data_config.get("batch_size", 32),
        image_size=data_config.get("image_size", 224),
        num_workers=data_config.get("num_workers", 4),
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
        test_ratio=data_config.get("test_ratio", 0.15),
        seed=seed,
        pin_memory=data_config.get("pin_memory", True),
    )
    
    # Create model
    model_config = config.get("model", {})
    model = create_model(
        architecture=model_config.get("architecture", "efficientnet_b0"),
        num_classes=data_config.get("num_classes", 3),
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", True),
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_names=class_names,
    )
    
    trainer.train()
    
    logger.info("\nTraining complete! Best model saved to: models/best_model.pt")


if __name__ == "__main__":
    main()

