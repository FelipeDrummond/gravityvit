"""Metrics tracking for GravityViT training."""

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


class MetricTracker:
    """
    Accumulates predictions across batches and computes metrics.

    Usage:
        tracker = MetricTracker(class_names)
        for batch in loader:
            preds = model(inputs)
            tracker.update(preds, targets)
        metrics = tracker.compute()
    """

    def __init__(self, class_names: Optional[list[str]] = None):
        """
        Args:
            class_names: List of class names for reporting
        """
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.all_preds: list[np.ndarray] = []
        self.all_targets: list[np.ndarray] = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """
        Update with batch predictions.

        Args:
            preds: Model outputs of shape (N, C) or predictions (N,)
            targets: Target labels of shape (N,) as tensor or numpy array
            loss: Optional batch loss value
        """
        if preds.dim() == 2:
            preds = preds.argmax(dim=1)

        self.all_preds.append(preds.cpu().numpy())

        # Handle both tensor and numpy array inputs for targets
        if isinstance(targets, np.ndarray):
            self.all_targets.append(targets)
        else:
            self.all_targets.append(targets.cpu().numpy())

        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary of metric name -> value
        """
        if not self.all_preds:
            return {}

        preds = np.concatenate(self.all_preds)
        targets = np.concatenate(self.all_targets)

        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "balanced_accuracy": balanced_accuracy_score(targets, preds),
            "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                targets, preds, average="weighted", zero_division=0
            ),
        }

        if self.num_batches > 0:
            metrics["loss"] = self.total_loss / self.num_batches

        return metrics

    def get_confusion_matrix(self, labels: Optional[int] = None) -> np.ndarray:
        """
        Return confusion matrix from accumulated predictions.

        Args:
            labels: Number of classes. If provided, ensures the confusion matrix
                has shape (labels, labels) even if some classes have no samples.
        """
        if not self.all_preds:
            return np.array([])

        preds = np.concatenate(self.all_preds)
        targets = np.concatenate(self.all_targets)

        if labels is not None:
            return confusion_matrix(targets, preds, labels=list(range(labels)))
        return confusion_matrix(targets, preds)

    def get_classification_report(self) -> str:
        """Return formatted classification report."""
        if not self.all_preds:
            return ""

        preds = np.concatenate(self.all_preds)
        targets = np.concatenate(self.all_targets)

        return classification_report(
            targets,
            preds,
            target_names=self.class_names,
            zero_division=0,
        )


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy from batch predictions.

    Args:
        preds: Model outputs of shape (N, C) or predictions (N,)
        targets: Target labels of shape (N,)

    Returns:
        Accuracy as float
    """
    if preds.dim() == 2:
        preds = preds.argmax(dim=1)

    return (preds == targets).float().mean().item()
