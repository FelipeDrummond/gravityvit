"""Standalone evaluation script for GravityViT models.

Loads a trained checkpoint and runs comprehensive evaluation on the test set,
logging all metrics, analysis, and visualizations to MLflow.

Usage:
    python scripts/evaluate.py checkpoint=outputs/experiment/best_model.pt
    python scripts/evaluate.py checkpoint=path/to/model.pt data.batch_size=64
"""

import logging
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.analysis import (  # noqa: E402
    EvaluationAnalyzer,
    plot_confusion_matrix,
    plot_per_class_accuracy,
)
from src.data import create_dataloader  # noqa: E402
from src.training.metrics import MetricTracker  # noqa: E402

logger = logging.getLogger(__name__)


def load_checkpoint_and_config(checkpoint_path: Path) -> tuple[dict, DictConfig]:
    """Load checkpoint and extract config.

    Warning:
        This function uses pickle deserialization (weights_only=False) to load
        the config stored in the checkpoint. Only load checkpoints from trusted
        sources, as malicious checkpoint files could execute arbitrary code.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Note: weights_only=False is required to load the config dict stored in checkpoint.
    # Only load checkpoints from trusted sources.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required_keys = ["config", "model_state_dict"]
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required '{key}' key")

    config = OmegaConf.create(checkpoint["config"])
    return checkpoint, config


def create_model_from_config(cfg: DictConfig) -> torch.nn.Module:
    """Create model based on checkpoint config."""
    model_type = cfg.model.type
    num_classes = cfg.model.num_classes

    if model_type == "cnn":
        import torchvision.models as models

        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "vit":
        import timm

        model = timm.create_model(
            cfg.model.backbone,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=cfg.model.dropout,
            attn_drop_rate=cfg.model.attn_dropout,
        )

    elif model_type == "multiview_vit":
        raise NotImplementedError("MultiViewViT not yet implemented")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[MetricTracker, np.ndarray]:
    """Run model evaluation on a dataloader."""
    model.eval()
    tracker = MetricTracker(class_names=class_names)
    all_probs = []

    for inputs, targets in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)

        tracker.update(outputs, targets)
        all_probs.append(probs.cpu().numpy())

    if not all_probs:
        return tracker, np.array([])

    probabilities = np.concatenate(all_probs, axis=0)
    return tracker, probabilities


def log_to_mlflow(
    tracker: MetricTracker,
    analyzer: EvaluationAnalyzer,
    output_dir: Path,
    class_names: list[str],
) -> None:
    """Log all evaluation results to MLflow."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log summary metrics
    summary_metrics = tracker.compute()
    for name, value in summary_metrics.items():
        mlflow.log_metric(f"test_{name}", value)
    logger.info("Logged summary metrics to MLflow")

    # Log per-class metrics
    per_class_metrics = tracker.get_per_class_metrics()
    for class_name, metrics in per_class_metrics.items():
        for metric_name, value in metrics.items():
            if metric_name != "support":
                mlflow.log_metric(f"test_{class_name}_{metric_name}", value)
    logger.info("Logged per-class metrics to MLflow")

    # Log per-class accuracy
    per_class_acc = analyzer.get_per_class_accuracy()
    for class_name, acc in per_class_acc.items():
        mlflow.log_metric(f"test_{class_name}_accuracy", acc)

    # Log confused pairs as text artifact
    confused_pairs = analyzer.find_most_confused_pairs(top_k=20)
    confused_path = output_dir / "confused_pairs.txt"
    with open(confused_path, "w") as f:
        f.write("Most Confused Class Pairs\n")
        f.write("=" * 60 + "\n\n")
        for pair in confused_pairs:
            f.write(
                f"{pair.true_class} -> {pair.predicted_class}: "
                f"{pair.count} samples ({pair.percentage:.1f}%)\n"
            )
    mlflow.log_artifact(str(confused_path))
    logger.info("Logged confused pairs to MLflow")

    # Log high-confidence misclassifications
    try:
        high_conf_errors = analyzer.find_high_confidence_misclassifications(
            threshold=0.9
        )
        mlflow.log_metric("test_high_conf_errors_count", len(high_conf_errors))
        logger.info(f"Found {len(high_conf_errors)} high-confidence misclassifications")

        if high_conf_errors:
            errors_path = output_dir / "high_confidence_errors.txt"
            with open(errors_path, "w") as f:
                f.write("High-Confidence Misclassifications (>90% confidence)\n")
                f.write("=" * 60 + "\n\n")
                for err in high_conf_errors[:50]:
                    f.write(f"Sample {err.sample_index}:\n")
                    f.write(f"  True: {err.true_class}\n")
                    f.write(f"  Pred: {err.predicted_class}\n")
                    f.write(f"  Confidence: {err.confidence:.4f}\n")
                    f.write(f"  True class prob: {err.true_class_prob:.4f}\n\n")
            mlflow.log_artifact(str(errors_path))
    except ValueError as e:
        logger.warning(f"Could not analyze high-confidence errors: {e}")

    # Log confusion matrix visualization (normalized)
    cm = analyzer.confusion_matrix_raw
    fig = plot_confusion_matrix(cm, class_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_normalized.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))
    logger.info("Logged confusion matrix to MLflow")

    # Log per-class accuracy visualization
    fig = plot_per_class_accuracy(per_class_acc)
    acc_path = output_dir / "per_class_accuracy.png"
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(acc_path))
    logger.info("Logged per-class accuracy chart to MLflow")

    # Log classification report
    report = tracker.get_classification_report()
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(str(report_path))
    logger.info("Logged classification report to MLflow")


def log_summary(tracker: MetricTracker, analyzer: EvaluationAnalyzer) -> None:
    """Log summary of evaluation results to console."""
    metrics = tracker.compute()

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
    logger.info("=" * 60)

    per_class_acc = analyzer.get_per_class_accuracy()
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1])

    logger.info("\nLowest Accuracy Classes:")
    for class_name, acc in sorted_acc[:5]:
        logger.info(f"  {class_name}: {acc:.4f}")

    logger.info("\nHighest Accuracy Classes:")
    for class_name, acc in sorted_acc[-5:]:
        logger.info(f"  {class_name}: {acc:.4f}")

    confused = analyzer.find_most_confused_pairs(top_k=5)
    logger.info("\nMost Confused Pairs:")
    for pair in confused:
        logger.info(
            f"  {pair.true_class} -> {pair.predicted_class}: "
            f"{pair.count} ({pair.percentage:.1f}%)"
        )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""
    if not hasattr(cfg, "checkpoint") or cfg.checkpoint is None:
        raise ValueError(
            "checkpoint path required. "
            "Usage: python scripts/evaluate.py checkpoint=path/to/model.pt"
        )

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint and get training config
    checkpoint, train_cfg = load_checkpoint_and_config(checkpoint_path)

    # Use data config from checkpoint, but allow overrides from CLI
    data_cfg = train_cfg.data
    if hasattr(cfg, "data"):
        data_cfg = OmegaConf.merge(data_cfg, cfg.data)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create model and load weights
    model = create_model_from_config(train_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logger.info(f"Loaded model: {train_cfg.model.name}")

    # Get class names
    class_names = list(data_cfg.class_names)

    # Create test dataloader
    test_loader = create_dataloader(
        root=data_cfg.root,
        split="test",
        mode="single",
        time_scale=data_cfg.get("time_scale", 1.0),
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        image_size=data_cfg.image_size,
        mean=list(data_cfg.mean),
        std=list(data_cfg.std),
        class_names=class_names,
        shuffle=False,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Run evaluation
    logger.info("Running evaluation...")
    tracker, probabilities = run_evaluation(model, test_loader, device, class_names)

    # Get predictions for analysis
    predictions, targets = tracker.get_predictions()

    # Create analyzer
    analyzer = EvaluationAnalyzer(
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
        class_names=class_names,
    )

    # Log summary to console
    log_summary(tracker, analyzer)

    # Setup MLflow and log all results
    if train_cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(train_cfg.mlflow.tracking_uri)

    mlflow.set_experiment(train_cfg.mlflow.experiment_name)

    run_name = f"eval_{train_cfg.model.name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("checkpoint", str(checkpoint_path))
        mlflow.log_param("model", train_cfg.model.name)
        mlflow.log_param("test_samples", len(test_loader.dataset))

        output_dir = Path(
            OmegaConf.select(cfg, "hydra.run.dir", default="outputs/evaluation")
        )
        log_to_mlflow(tracker, analyzer, output_dir, class_names)

        logger.info(f"\nAll results logged to MLflow experiment: {train_cfg.mlflow.experiment_name}")


if __name__ == "__main__":
    main()
