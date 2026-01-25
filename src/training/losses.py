"""Loss functions for GravityViT training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
        weight: Class weights for imbalanced data
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (N, C)
            target: Target labels of shape (N,)

        Returns:
            Scalar loss
        """
        n_classes = pred.size(-1)

        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Apply label smoothing
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute log softmax
        log_probs = F.log_softmax(pred, dim=-1)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            # Expand weights to match batch: (C,) -> (N, C)
            sample_weights = weight[target]  # (N,)
            loss = -(one_hot * log_probs).sum(dim=-1)  # (N,)
            loss = (loss * sample_weights).mean()
        else:
            loss = -(one_hot * log_probs).sum(dim=-1).mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weights (tensor of shape (C,) or None)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (N, C)
            target: Target labels of shape (N,)

        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_t = alpha[target]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_loss_fn(
    cfg: DictConfig,
    class_weights: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Factory function to create loss function based on config.

    Args:
        cfg: Training configuration
        class_weights: Optional class weights for imbalanced data
        device: Target device

    Returns:
        Loss function module
    """
    loss_name = cfg.train.loss.get("name", "cross_entropy")
    label_smoothing = cfg.train.loss.get("label_smoothing", 0.0)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_name == "cross_entropy":
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights,
            )
        else:
            return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "focal_loss":
        gamma = cfg.train.loss.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
