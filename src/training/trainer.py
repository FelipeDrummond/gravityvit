"""Trainer class for GravityViT training."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.training.losses import create_loss_fn
from src.training.metrics import MetricTracker
from src.utils import get_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator with AMP, gradient clipping, checkpointing,
    early stopping, and MLflow experiment tracking.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        cfg: Hydra configuration
        test_loader: Optional test dataloader
        class_weights: Optional class weights for loss function
        class_names: Optional class names for reporting
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        test_loader: Optional[DataLoader] = None,
        class_weights: Optional[torch.Tensor] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names

        # Device setup
        self.device = get_device(cfg.train.get("device", "auto"))
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)

        # Loss function
        self.criterion = create_loss_fn(cfg, class_weights, self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=tuple(cfg.train.optimizer.betas),
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = cfg.train.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training enabled")

        # Gradient clipping
        self.grad_clip = cfg.train.get("grad_clip", 1.0)

        # Early stopping
        self.early_stopping_enabled = cfg.train.early_stopping.enabled
        self.patience = cfg.train.early_stopping.patience
        self.es_metric = cfg.train.early_stopping.metric
        self.es_mode = cfg.train.early_stopping.mode
        self.best_metric = float("-inf") if self.es_mode == "max" else float("inf")
        self.patience_counter = 0

        # Checkpointing
        self.save_best = cfg.train.checkpoint.save_best
        self.save_last = cfg.train.checkpoint.save_last
        self.ckpt_metric = cfg.train.checkpoint.metric
        self.ckpt_mode = cfg.train.checkpoint.mode
        self.best_ckpt_metric = (
            float("-inf") if self.ckpt_mode == "max" else float("inf")
        )

        # Output directory
        self.output_dir = Path(cfg.hydra.run.dir if "hydra" in cfg else "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_checkpoint_path: Optional[Path] = None

        # Metrics
        self.train_metrics = MetricTracker(class_names)
        self.val_metrics = MetricTracker(class_names)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_epochs = self.cfg.train.scheduler.get("warmup_epochs", 5)
        min_lr = self.cfg.train.scheduler.get("min_lr", 1e-6)
        total_epochs = self.cfg.train.epochs

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    def train(self) -> dict:
        """
        Main training loop.

        Returns:
            Dictionary with final metrics and best checkpoint path
        """
        self._setup_mlflow()

        epochs = self.cfg.train.epochs
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            self._log_epoch_metrics(train_metrics, val_metrics)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.2e}"
            )

            is_best = self._check_best_model(val_metrics)
            if self.save_best and is_best:
                self._save_checkpoint(val_metrics, is_best=True)
            if self.save_last:
                self._save_checkpoint(val_metrics, is_best=False)

            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            self.scheduler.step()

        self._log_final_artifacts()

        final_results = {
            "best_val_accuracy": self.best_metric if self.es_mode == "max" else None,
            "best_checkpoint": (
                str(self.best_checkpoint_path) if self.best_checkpoint_path else None
            ),
            "final_epoch": self.current_epoch + 1,
        }

        mlflow.end_run()

        return final_results

    def _train_epoch(self) -> dict:
        """Run single training epoch."""
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            self.train_metrics.update(outputs, targets, loss.item())
            self.global_step += 1

        return self.train_metrics.compute()

    @torch.no_grad()
    def _validate_epoch(self) -> dict:
        """Run validation pass."""
        self.model.eval()
        self.val_metrics.reset()

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.val_metrics.update(outputs, targets, loss.item())

        return self.val_metrics.compute()

    def _check_early_stopping(self, metrics: dict) -> bool:
        """Check if early stopping should be triggered."""
        if not self.early_stopping_enabled:
            return False

        current = metrics.get(self.es_metric.replace("val_", ""), 0)

        if self.es_mode == "max":
            improved = current > self.best_metric
        else:
            improved = current < self.best_metric

        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience

    def _check_best_model(self, metrics: dict) -> bool:
        """Check if current model is best so far."""
        current = metrics.get(self.ckpt_metric.replace("val_", ""), 0)

        if self.ckpt_mode == "max":
            is_best = current > self.best_ckpt_metric
        else:
            is_best = current < self.best_ckpt_metric

        if is_best:
            self.best_ckpt_metric = current

        return is_best

    def _save_checkpoint(self, metrics: dict, is_best: bool) -> Path:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        if is_best:
            path = self.output_dir / "best_model.pt"
            self.best_checkpoint_path = path
        else:
            path = self.output_dir / "last_model.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        return path

    def _setup_mlflow(self):
        """Initialize MLflow experiment tracking."""
        if self.cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)

        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        run_name = self.cfg.mlflow.run_name or f"{self.cfg.model.name}_run"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params(
            {
                "model": self.cfg.model.name,
                "model_type": self.cfg.model.type,
                "epochs": self.cfg.train.epochs,
                "batch_size": self.cfg.data.batch_size,
                "learning_rate": self.cfg.train.optimizer.lr,
                "weight_decay": self.cfg.train.optimizer.weight_decay,
                "optimizer": self.cfg.train.optimizer.name,
                "scheduler": self.cfg.train.scheduler.name,
                "loss": self.cfg.train.loss.name,
                "label_smoothing": self.cfg.train.loss.get("label_smoothing", 0.0),
                "mixed_precision": self.use_amp,
                "grad_clip": self.grad_clip,
                "device": str(self.device),
            }
        )

        for key, value in self.cfg.mlflow.get("tags", {}).items():
            mlflow.set_tag(key, value)

    def _log_epoch_metrics(self, train_metrics: dict, val_metrics: dict):
        """Log metrics to MLflow."""
        step = self.current_epoch

        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value, step=step)

        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value, step=step)

        mlflow.log_metric(
            "learning_rate", self.optimizer.param_groups[0]["lr"], step=step
        )

    def _log_final_artifacts(self):
        """Log final artifacts to MLflow."""
        # Save config
        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(self.cfg, config_path)
        mlflow.log_artifact(str(config_path))

        # Confusion matrix
        self._log_confusion_matrix()

        # Sample predictions
        self._log_sample_predictions()

        # Classification report
        report = self.val_metrics.get_classification_report()
        if report:
            report_path = self.output_dir / "classification_report.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

        # Best model
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            mlflow.log_artifact(str(self.best_checkpoint_path))

    def _log_confusion_matrix(self):
        """Save and log confusion matrix visualization."""
        cm = self.val_metrics.get_confusion_matrix()
        if cm.size == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names or range(cm.shape[0]),
            yticklabels=self.class_names or range(cm.shape[1]),
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()

        cm_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(cm_path))

    def _log_sample_predictions(self, num_samples: int = 16):
        """Log grid of sample predictions."""
        self.model.eval()

        images_list = []
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)

                images_list.append(inputs.cpu())
                preds_list.append(preds.cpu())
                targets_list.append(targets)

                if sum(len(p) for p in preds_list) >= num_samples:
                    break

        images = torch.cat(images_list)[:num_samples]
        preds = torch.cat(preds_list)[:num_samples]
        targets = torch.cat(targets_list)[:num_samples]

        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for idx in range(num_samples):
            ax = axes[idx]
            img = images[idx].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            ax.imshow(img)

            pred_label = (
                self.class_names[preds[idx]] if self.class_names else preds[idx].item()
            )
            true_label = (
                self.class_names[targets[idx]]
                if self.class_names
                else targets[idx].item()
            )

            color = "green" if preds[idx] == targets[idx] else "red"
            ax.set_title(
                f"Pred: {pred_label}\nTrue: {true_label}", fontsize=8, color=color
            )
            ax.axis("off")

        for idx in range(num_samples, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        pred_path = self.output_dir / "sample_predictions.png"
        fig.savefig(pred_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(pred_path))


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
