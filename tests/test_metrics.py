"""Tests for metrics module."""

import numpy as np
import torch

from src.training.metrics import MetricTracker, compute_accuracy


class TestComputeAccuracy:
    def test_perfect_predictions(self):
        """Should return 1.0 for perfect predictions."""
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])

        acc = compute_accuracy(preds, targets)
        assert acc == 1.0

    def test_zero_accuracy(self):
        """Should return 0.0 for completely wrong predictions."""
        preds = torch.tensor([1, 2, 3, 0])
        targets = torch.tensor([0, 1, 2, 3])

        acc = compute_accuracy(preds, targets)
        assert acc == 0.0

    def test_partial_accuracy(self):
        """Should handle partial correct predictions."""
        preds = torch.tensor([0, 1, 0, 0])
        targets = torch.tensor([0, 1, 2, 3])

        acc = compute_accuracy(preds, targets)
        assert acc == 0.5

    def test_with_logits(self):
        """Should handle logit inputs (N, C)."""
        # Logits where class 0 wins for all samples
        preds = torch.zeros(4, 5)
        preds[:, 0] = 10.0
        targets = torch.zeros(4, dtype=torch.long)

        acc = compute_accuracy(preds, targets)
        assert acc == 1.0


class TestMetricTracker:
    def test_single_batch_accuracy(self):
        """Should compute correct accuracy for single batch."""
        tracker = MetricTracker()

        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])

        tracker.update(preds, targets)
        metrics = tracker.compute()

        assert metrics["accuracy"] == 1.0

    def test_multiple_batches(self):
        """Should accumulate across multiple batches."""
        tracker = MetricTracker()

        # Batch 1: 100% correct
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        # Batch 2: 50% correct
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 0]))

        metrics = tracker.compute()
        assert metrics["accuracy"] == 0.75

    def test_reset(self):
        """Should clear accumulated predictions."""
        tracker = MetricTracker()

        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 0]))
        tracker.reset()
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))

        metrics = tracker.compute()
        assert metrics["accuracy"] == 1.0

    def test_loss_tracking(self):
        """Should track average loss."""
        tracker = MetricTracker()

        tracker.update(torch.tensor([0]), torch.tensor([0]), loss=1.0)
        tracker.update(torch.tensor([0]), torch.tensor([0]), loss=2.0)
        tracker.update(torch.tensor([0]), torch.tensor([0]), loss=3.0)

        metrics = tracker.compute()
        assert metrics["loss"] == 2.0

    def test_confusion_matrix(self):
        """Should return correct confusion matrix."""
        tracker = MetricTracker()

        # Class 0: 2 correct, 1 predicted as class 1
        # Class 1: 1 correct, 1 predicted as class 0
        preds = torch.tensor([0, 0, 1, 1, 0])
        targets = torch.tensor([0, 0, 0, 1, 1])

        tracker.update(preds, targets)
        cm = tracker.get_confusion_matrix()

        # Expected:
        # True 0: pred 0=2, pred 1=1
        # True 1: pred 0=1, pred 1=1
        expected = np.array([[2, 1], [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_f1_macro(self):
        """Should compute macro F1 score."""
        tracker = MetricTracker()

        preds = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([0, 0, 1, 1])

        tracker.update(preds, targets)
        metrics = tracker.compute()

        assert metrics["f1_macro"] == 1.0

    def test_balanced_accuracy(self):
        """Should compute balanced accuracy."""
        tracker = MetricTracker()

        # Imbalanced: 3 samples of class 0, 1 sample of class 1
        # All class 0 correct, class 1 wrong
        preds = torch.tensor([0, 0, 0, 0])
        targets = torch.tensor([0, 0, 0, 1])

        tracker.update(preds, targets)
        metrics = tracker.compute()

        # Regular accuracy: 3/4 = 0.75
        # Balanced: (1.0 + 0.0) / 2 = 0.5
        assert metrics["accuracy"] == 0.75
        assert metrics["balanced_accuracy"] == 0.5

    def test_with_class_names(self):
        """Should include class names in report."""
        class_names = ["cat", "dog"]
        tracker = MetricTracker(class_names=class_names)

        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        report = tracker.get_classification_report()

        assert "cat" in report
        assert "dog" in report

    def test_empty_tracker(self):
        """Should handle empty tracker gracefully."""
        tracker = MetricTracker()

        metrics = tracker.compute()
        cm = tracker.get_confusion_matrix()
        report = tracker.get_classification_report()

        assert metrics == {}
        assert cm.size == 0
        assert report == ""
