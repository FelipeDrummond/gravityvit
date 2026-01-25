"""Evaluation analysis utilities for GravityViT."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.metrics import confusion_matrix


def normalize_confusion_matrix(
    cm: np.ndarray, mode: Literal["true", "pred", "all"]
) -> np.ndarray:
    """
    Normalize a confusion matrix.

    Args:
        cm: Raw confusion matrix as numpy array
        mode: Normalization mode:
            - "true": Normalize over true labels (rows sum to 1)
            - "pred": Normalize over predictions (columns sum to 1)
            - "all": Normalize over all samples (matrix sums to 1)

    Returns:
        Normalized confusion matrix as numpy array
    """
    cm_float = cm.astype(float)

    if mode == "true":
        row_sums = cm_float.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return cm_float / row_sums
    elif mode == "pred":
        col_sums = cm_float.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        return cm_float / col_sums
    elif mode == "all":
        total = cm_float.sum()
        return cm_float / total if total > 0 else cm_float
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")


@dataclass
class ConfusedPair:
    """Represents a pair of frequently confused classes."""

    true_class: str
    predicted_class: str
    count: int
    percentage: float  # Percentage of true class samples misclassified as predicted


@dataclass
class HighConfidenceMisclassification:
    """Represents a high-confidence misclassification."""

    sample_index: int
    true_class: str
    predicted_class: str
    confidence: float
    true_class_prob: float


class EvaluationAnalyzer:
    """
    Comprehensive evaluation analysis for classification results.

    Args:
        predictions: Model predictions as class indices
        targets: Ground truth labels as class indices
        probabilities: Optional prediction probabilities of shape (N, C)
        class_names: Optional list of class names
    """

    def __init__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.predictions = predictions
        self.targets = targets
        self.probabilities = probabilities
        self.class_names = class_names

        if class_names:
            self.num_classes = len(class_names)
        elif len(predictions) == 0 and len(targets) == 0:
            self.num_classes = 0
        else:
            self.num_classes = int(max(predictions.max(), targets.max()) + 1)

        self._confusion_matrix: Optional[np.ndarray] = None

    def _get_class_name(self, idx: int) -> str:
        """Get class name by index."""
        if self.class_names is not None and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    @property
    def confusion_matrix_raw(self) -> np.ndarray:
        """Compute and cache raw confusion matrix."""
        if self._confusion_matrix is None:
            if self.num_classes == 0:
                self._confusion_matrix = np.array([]).reshape(0, 0)
            else:
                self._confusion_matrix = confusion_matrix(
                    self.targets,
                    self.predictions,
                    labels=list(range(self.num_classes)),
                )
        return self._confusion_matrix

    def get_normalized_confusion_matrix(
        self, normalize: Literal["true", "pred", "all"] = "true"
    ) -> np.ndarray:
        """
        Return normalized confusion matrix.

        Args:
            normalize: Normalization mode:
                - "true": Normalize over true labels (rows sum to 1)
                - "pred": Normalize over predictions (columns sum to 1)
                - "all": Normalize over all samples (matrix sums to 1)

        Returns:
            Normalized confusion matrix as numpy array
        """
        return normalize_confusion_matrix(self.confusion_matrix_raw, normalize)

    def find_most_confused_pairs(self, top_k: int = 10) -> list[ConfusedPair]:
        """
        Find the most frequently confused class pairs.

        Args:
            top_k: Number of top confused pairs to return

        Returns:
            List of ConfusedPair objects sorted by confusion count
        """
        cm = self.confusion_matrix_raw
        cm_normalized = self.get_normalized_confusion_matrix(normalize="true")

        pairs = []
        for true_idx in range(self.num_classes):
            for pred_idx in range(self.num_classes):
                if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                    pairs.append(
                        ConfusedPair(
                            true_class=self._get_class_name(true_idx),
                            predicted_class=self._get_class_name(pred_idx),
                            count=int(cm[true_idx, pred_idx]),
                            percentage=float(cm_normalized[true_idx, pred_idx] * 100),
                        )
                    )

        pairs.sort(key=lambda p: p.count, reverse=True)
        return pairs[:top_k]

    def find_high_confidence_misclassifications(
        self, threshold: float = 0.9
    ) -> list[HighConfidenceMisclassification]:
        """
        Find samples that were misclassified with high confidence.

        Args:
            threshold: Minimum confidence for predicted class to be considered
                "high confidence"

        Returns:
            List of HighConfidenceMisclassification objects sorted by confidence

        Raises:
            ValueError: If probabilities were not provided
        """
        if self.probabilities is None:
            raise ValueError(
                "Probabilities required for high-confidence misclassification analysis"
            )

        misclassified_mask = self.predictions != self.targets
        misclassified_indices = np.where(misclassified_mask)[0]

        misclassifications = []
        for idx in misclassified_indices:
            probs = self.probabilities[idx]
            confidence = probs.max()

            if confidence >= threshold:
                pred_class = int(self.predictions[idx])
                true_class = int(self.targets[idx])

                misclassifications.append(
                    HighConfidenceMisclassification(
                        sample_index=int(idx),
                        true_class=self._get_class_name(true_class),
                        predicted_class=self._get_class_name(pred_class),
                        confidence=float(confidence),
                        true_class_prob=float(probs[true_class]),
                    )
                )

        misclassifications.sort(key=lambda m: m.confidence, reverse=True)
        return misclassifications

    def get_per_class_accuracy(self) -> dict[str, float]:
        """
        Compute per-class accuracy (recall).

        Returns:
            Dictionary mapping class name to accuracy
        """
        cm = self.confusion_matrix_raw

        per_class_acc = {}
        for i in range(self.num_classes):
            class_name = self._get_class_name(i)
            total = cm[i].sum()
            if total > 0:
                per_class_acc[class_name] = float(cm[i, i] / total)
            else:
                per_class_acc[class_name] = 0.0

        return per_class_acc
