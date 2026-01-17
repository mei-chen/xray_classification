"""
Hyperparameter tuning with structured logging.

Supports:
- Grid search, random search
- Experiment tracking with JSON logs (MLflow-compatible format)
- Search space definition for key hyperparameters
"""

import argparse
import json
import logging
import itertools
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import get_device, load_config, set_seed
from src.data import create_dataloaders
from src.models.architecture import create_model
from src.models.train import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default search space for hyperparameters
DEFAULT_SEARCH_SPACE = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 3e-3],
    "weight_decay": [0.001, 0.01, 0.1],
    "dropout": [0.2, 0.3, 0.5],
    "backbone_lr_factor": [0.05, 0.1, 0.2],
    "unfreeze_at_epoch": [3, 5, 10],
    "label_smoothing": [0.0, 0.1, 0.2],
    "batch_size": [16, 32, 64],
    "image_size": [224, 256, 384],
}


class ExperimentLogger:
    """Simple experiment logger with MLflow-compatible JSON format."""
    
    def __init__(self, log_dir: Path):
        """Initialize the experiment logger.
        
        Args:
            log_dir: Directory to save experiment logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
        self.run_id = 0
    
    def log_run(
        self,
        params: dict,
        metrics: dict,
        tags: dict = None,
        artifacts: list = None,
    ) -> int:
        """Log a single experiment run.
        
        Args:
            params: Hyperparameters used.
            metrics: Evaluation metrics.
            tags: Optional tags for the run.
            artifacts: Optional list of artifact paths.
            
        Returns:
            Run ID.
        """
        self.run_id += 1
        
        run = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
            "artifacts": artifacts or [],
        }
        
        self.experiments.append(run)
        
        # Save incrementally
        self._save_logs()
        
        return self.run_id
    
    def _save_logs(self):
        """Save experiment logs to JSON."""
        log_path = self.log_dir / "experiment_logs.json"
        with open(log_path, "w") as f:
            json.dump({
                "experiments": self.experiments,
                "best_run": self.get_best_run(),
            }, f, indent=2)
    
    def get_best_run(self, metric: str = "val_f1", higher_is_better: bool = True) -> dict:
        """Get the best run by a specific metric.
        
        Args:
            metric: Metric to optimize.
            higher_is_better: Whether higher values are better.
            
        Returns:
            Best run dictionary.
        """
        if not self.experiments:
            return {}
        
        if higher_is_better:
            best = max(self.experiments, key=lambda x: x["metrics"].get(metric, 0))
        else:
            best = min(self.experiments, key=lambda x: x["metrics"].get(metric, float("inf")))
        
        return best
    
    def print_summary(self):
        """Print a summary of all experiments."""
        if not self.experiments:
            logger.info("No experiments logged yet.")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER TUNING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total runs: {len(self.experiments)}")
        
        # Sort by val_f1
        sorted_runs = sorted(
            self.experiments,
            key=lambda x: x["metrics"].get("val_f1", 0),
            reverse=True,
        )
        
        logger.info("\nTop 5 runs by validation F1:")
        logger.info("-" * 80)
        for i, run in enumerate(sorted_runs[:5], 1):
            params = run["params"]
            metrics = run["metrics"]
            logger.info(
                f"{i}. Run {run['run_id']}: "
                f"val_f1={metrics.get('val_f1', 0):.4f}, "
                f"val_loss={metrics.get('val_loss', 0):.4f} | "
                f"lr={params.get('learning_rate', 'N/A')}, "
                f"wd={params.get('weight_decay', 'N/A')}, "
                f"dropout={params.get('dropout', 'N/A')}"
            )
        
        logger.info("\nBest hyperparameters:")
        best = sorted_runs[0]
        for key, value in best["params"].items():
            logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80)


def generate_param_combinations(
    search_space: dict,
    method: str = "grid",
    n_samples: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate hyperparameter combinations.
    
    Args:
        search_space: Dictionary mapping param names to possible values.
        method: "grid" for grid search, "random" for random search.
        n_samples: Number of samples for random search.
        seed: Random seed.
        
    Returns:
        List of parameter dictionaries.
    """
    if method == "grid":
        # Full grid search
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    elif method == "random":
        # Random search
        random.seed(seed)
        combinations = []
        for _ in range(n_samples):
            combo = {
                key: random.choice(values)
                for key, values in search_space.items()
            }
            combinations.append(combo)
        return combinations
    
    else:
        raise ValueError(f"Unknown search method: {method}")


def run_single_experiment(
    params: dict,
    base_config: dict,
    device: torch.device,
    max_epochs: int = 15,
) -> dict:
    """Run a single training experiment with given hyperparameters.
    
    Args:
        params: Hyperparameters to use.
        base_config: Base configuration dictionary.
        device: Device to train on.
        max_epochs: Maximum epochs (shorter for tuning).
        
    Returns:
        Dictionary with metrics.
    """
    # Update config with hyperparameters
    config = base_config.copy()
    
    # Update optimizer settings
    if "learning_rate" in params:
        config["training"]["optimizer"]["learning_rate"] = params["learning_rate"]
    if "weight_decay" in params:
        config["training"]["optimizer"]["weight_decay"] = params["weight_decay"]
    if "backbone_lr_factor" in params:
        config["training"]["optimizer"]["backbone_lr_factor"] = params["backbone_lr_factor"]
    
    # Update model settings
    if "dropout" in params:
        config["model"]["dropout"] = params["dropout"]
    if "unfreeze_at_epoch" in params:
        config["model"]["unfreeze_at_epoch"] = params["unfreeze_at_epoch"]
    
    # Update loss settings
    if "label_smoothing" in params:
        config["training"]["loss"]["label_smoothing"] = params["label_smoothing"]
    
    # Update data settings
    if "batch_size" in params:
        config["data"]["batch_size"] = params["batch_size"]
    if "image_size" in params:
        config["data"]["image_size"] = params["image_size"]
    
    # Reduce epochs for tuning
    config["training"]["epochs"] = max_epochs
    config["training"]["early_stopping"]["patience"] = 5
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create dataloaders
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("raw_data_path", "data/raw/chest-xray-dataset"))
    
    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=data_config.get("batch_size", 32),
        image_size=data_config.get("image_size", 224),
        num_workers=data_config.get("num_workers", 4),
        seed=seed,
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
    
    history = trainer.train()
    
    # Return final metrics
    return {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "train_f1": history["train_f1"][-1] if history["train_f1"] else None,
        "val_f1": history["val_f1"][-1] if history["val_f1"] else None,
        "best_val_f1": trainer.best_metric,
        "epochs_trained": len(history["train_loss"]),
        "generalization_gap": (
            history["train_f1"][-1] - history["val_f1"][-1]
            if history["train_f1"] and history["val_f1"]
            else None
        ),
    }


def run_hyperparameter_search(
    base_config: dict,
    search_space: dict = None,
    method: str = "random",
    n_samples: int = 10,
    max_epochs: int = 15,
    log_dir: str = "reports/hp_tuning",
):
    """Run hyperparameter search.
    
    Args:
        base_config: Base configuration dictionary.
        search_space: Search space (uses default if None).
        method: "grid" or "random".
        n_samples: Number of samples for random search.
        max_epochs: Max epochs per run.
        log_dir: Directory for experiment logs.
    """
    search_space = search_space or DEFAULT_SEARCH_SPACE
    
    # Generate parameter combinations
    param_combinations = generate_param_combinations(
        search_space=search_space,
        method=method,
        n_samples=n_samples,
    )
    
    logger.info(f"Running {len(param_combinations)} experiments...")
    logger.info(f"Search method: {method}")
    logger.info(f"Max epochs per run: {max_epochs}")
    
    # Initialize logger
    exp_logger = ExperimentLogger(log_dir)
    
    # Get device
    device = get_device(base_config.get("device", "auto"))
    
    # Run experiments
    for i, params in enumerate(param_combinations):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EXPERIMENT {i + 1}/{len(param_combinations)}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Parameters: {params}")
        
        try:
            metrics = run_single_experiment(
                params=params,
                base_config=base_config,
                device=device,
                max_epochs=max_epochs,
            )
            
            exp_logger.log_run(
                params=params,
                metrics=metrics,
                tags={"method": method, "max_epochs": max_epochs},
            )
            
            logger.info(f"Results: val_f1={metrics.get('val_f1', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            exp_logger.log_run(
                params=params,
                metrics={"error": str(e)},
                tags={"status": "failed"},
            )
    
    # Print summary
    exp_logger.print_summary()
    
    return exp_logger


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Search method",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of random samples",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help="Maximum epochs per experiment",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="reports/hp_tuning",
        help="Directory for experiment logs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick search with reduced space",
    )
    args = parser.parse_args()
    
    # Load base config
    base_config = load_config(args.config)
    
    # Define search space
    if args.quick:
        # Reduced search space for quick testing
        search_space = {
            "learning_rate": [5e-4, 1e-3],
            "weight_decay": [0.01],
            "dropout": [0.3],
            "backbone_lr_factor": [0.1],
        }
    else:
        search_space = DEFAULT_SEARCH_SPACE
    
    # Run search
    run_hyperparameter_search(
        base_config=base_config,
        search_space=search_space,
        method=args.method,
        n_samples=args.n_samples,
        max_epochs=args.max_epochs,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
