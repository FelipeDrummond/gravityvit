"""Tests for analysis module."""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - Must be set before importing pyplot

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.analysis import (  # noqa: E402
    ConfusedPair,
    EvaluationAnalyzer,
    HighConfidenceMisclassification,
    plot_confusion_matrix,
    plot_per_class_accuracy,
)


class TestEvaluationAnalyzer:
    """Tests for EvaluationAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets for testing."""
        np.random.seed(42)
        predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
        targets = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2, 0])
        probabilities = np.random.dirichlet([1, 1, 1], size=10)
        # Set high confidence for some samples
        probabilities[0] = [0.95, 0.03, 0.02]  # Correct, high conf
        probabilities[4] = [0.05, 0.92, 0.03]  # Wrong (target=0, pred=1), high conf
        class_names = ["cat", "dog", "bird"]
        return predictions, targets, probabilities, class_names

    def test_confusion_matrix_raw(self, sample_data):
        """Should compute raw confusion matrix."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        cm = analyzer.confusion_matrix_raw
        assert cm.shape == (3, 3)
        assert cm.sum() == len(predictions)

    def test_normalized_confusion_matrix_true(self, sample_data):
        """Normalized by true labels should have rows summing to 1."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        cm_norm = analyzer.get_normalized_confusion_matrix(normalize="true")
        # Each row should sum to 1 (or 0 if no samples)
        row_sums = cm_norm.sum(axis=1)
        for i, s in enumerate(row_sums):
            if analyzer.confusion_matrix_raw[i].sum() > 0:
                assert abs(s - 1.0) < 1e-6

    def test_normalized_confusion_matrix_pred(self, sample_data):
        """Normalized by predictions should have columns summing to 1."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        cm_norm = analyzer.get_normalized_confusion_matrix(normalize="pred")
        col_sums = cm_norm.sum(axis=0)
        for i, s in enumerate(col_sums):
            if analyzer.confusion_matrix_raw[:, i].sum() > 0:
                assert abs(s - 1.0) < 1e-6

    def test_normalized_confusion_matrix_all(self, sample_data):
        """Normalized by all should sum to 1."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        cm_norm = analyzer.get_normalized_confusion_matrix(normalize="all")
        assert abs(cm_norm.sum() - 1.0) < 1e-6

    def test_find_most_confused_pairs(self, sample_data):
        """Should find confused pairs sorted by count."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        pairs = analyzer.find_most_confused_pairs(top_k=5)

        assert all(isinstance(p, ConfusedPair) for p in pairs)
        assert len(pairs) <= 5
        # Should be sorted by count descending
        for i in range(len(pairs) - 1):
            assert pairs[i].count >= pairs[i + 1].count
        # Should not include diagonal (correct predictions)
        for p in pairs:
            assert p.true_class != p.predicted_class

    def test_find_high_confidence_misclassifications(self, sample_data):
        """Should find high-confidence misclassifications."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        errors = analyzer.find_high_confidence_misclassifications(threshold=0.9)

        assert all(isinstance(e, HighConfidenceMisclassification) for e in errors)
        for e in errors:
            assert e.confidence >= 0.9
            assert e.true_class != e.predicted_class

    def test_high_confidence_without_probs_raises(self, sample_data):
        """Should raise when probabilities not provided."""
        predictions, targets, _, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities=None, class_names=class_names
        )

        with pytest.raises(ValueError, match="Probabilities required"):
            analyzer.find_high_confidence_misclassifications()

    def test_get_per_class_accuracy(self, sample_data):
        """Should compute per-class accuracy."""
        predictions, targets, probabilities, class_names = sample_data
        analyzer = EvaluationAnalyzer(
            predictions, targets, probabilities, class_names
        )

        per_class_acc = analyzer.get_per_class_accuracy()

        assert len(per_class_acc) == 3
        assert all(0.0 <= acc <= 1.0 for acc in per_class_acc.values())
        assert all(name in per_class_acc for name in class_names)

    def test_without_class_names(self):
        """Should work without class names using indices."""
        predictions = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 1, 0])

        analyzer = EvaluationAnalyzer(predictions, targets)

        assert analyzer.num_classes == 2
        per_class_acc = analyzer.get_per_class_accuracy()
        assert "0" in per_class_acc
        assert "1" in per_class_acc

    def test_empty_arrays_without_class_names(self):
        """Should handle empty arrays without class names."""
        predictions = np.array([])
        targets = np.array([])

        analyzer = EvaluationAnalyzer(predictions, targets)

        assert analyzer.num_classes == 0
        assert analyzer.get_per_class_accuracy() == {}
        assert analyzer.find_most_confused_pairs() == []

    def test_empty_arrays_with_class_names(self):
        """Should use class_names length when arrays are empty."""
        predictions = np.array([])
        targets = np.array([])
        class_names = ["cat", "dog", "bird"]

        analyzer = EvaluationAnalyzer(
            predictions, targets, class_names=class_names
        )

        assert analyzer.num_classes == 3
        per_class_acc = analyzer.get_per_class_accuracy()
        assert len(per_class_acc) == 3
        assert all(acc == 0.0 for acc in per_class_acc.values())


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_cm(self):
        """Sample confusion matrix."""
        return np.array([[10, 2, 1], [3, 15, 2], [1, 2, 12]])

    @pytest.fixture
    def class_names(self):
        """Sample class names."""
        return ["cat", "dog", "bird"]

    def test_plot_confusion_matrix_returns_figure(self, sample_cm, class_names):
        """Should return matplotlib Figure."""
        import matplotlib.pyplot as plt

        fig = plot_confusion_matrix(sample_cm, class_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_normalized(self, sample_cm, class_names):
        """Should handle normalized confusion matrix."""
        import matplotlib.pyplot as plt

        fig = plot_confusion_matrix(sample_cm, class_names, normalize="true")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_saves_file(self, sample_cm, class_names):
        """Should save figure when output_path provided."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.png"
            fig = plot_confusion_matrix(sample_cm, class_names, output_path=output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_plot_per_class_accuracy_returns_figure(self):
        """Should return matplotlib Figure."""
        import matplotlib.pyplot as plt

        per_class_acc = {"cat": 0.9, "dog": 0.85, "bird": 0.92}
        fig = plot_per_class_accuracy(per_class_acc)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_per_class_accuracy_saves_file(self):
        """Should save figure when output_path provided."""
        import matplotlib.pyplot as plt

        per_class_acc = {"cat": 0.9, "dog": 0.85, "bird": 0.92}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "acc.png"
            fig = plot_per_class_accuracy(per_class_acc, output_path=output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_plot_per_class_accuracy_sorted(self):
        """Should sort classes by accuracy."""
        import matplotlib.pyplot as plt

        per_class_acc = {"a": 0.5, "b": 0.9, "c": 0.7}
        fig = plot_per_class_accuracy(per_class_acc)

        # Get y-axis labels from the figure
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Should be sorted descending (highest first since inverted y-axis)
        assert labels == ["b", "c", "a"]
        plt.close(fig)


class TestMetricTrackerExtensions:
    """Tests for MetricTracker extensions."""

    def test_get_per_class_metrics(self):
        """Should return per-class precision, recall, f1, support."""
        import torch
        from src.training.metrics import MetricTracker

        tracker = MetricTracker(class_names=["cat", "dog"])
        preds = torch.tensor([0, 1, 0, 1, 0])
        targets = torch.tensor([0, 1, 0, 0, 1])

        tracker.update(preds, targets)
        per_class = tracker.get_per_class_metrics()

        assert "cat" in per_class
        assert "dog" in per_class
        assert set(per_class["cat"].keys()) == {"precision", "recall", "f1", "support"}
        assert per_class["cat"]["support"] == 3  # 3 true cats
        assert per_class["dog"]["support"] == 2  # 2 true dogs

    def test_get_predictions(self):
        """Should return accumulated predictions and targets."""
        import torch
        from src.training.metrics import MetricTracker

        tracker = MetricTracker()

        # Add multiple batches
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        tracker.update(torch.tensor([2, 0]), torch.tensor([2, 1]))

        preds, targets = tracker.get_predictions()

        np.testing.assert_array_equal(preds, [0, 1, 2, 0])
        np.testing.assert_array_equal(targets, [0, 1, 2, 1])

    def test_get_predictions_empty(self):
        """Should return empty arrays when no predictions."""
        from src.training.metrics import MetricTracker

        tracker = MetricTracker()
        preds, targets = tracker.get_predictions()

        assert len(preds) == 0
        assert len(targets) == 0

    def test_get_per_class_metrics_empty(self):
        """Should return empty dict when no predictions."""
        from src.training.metrics import MetricTracker

        tracker = MetricTracker()
        per_class = tracker.get_per_class_metrics()

        assert per_class == {}
