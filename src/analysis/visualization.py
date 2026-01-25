"""Visualization utilities for evaluation results."""

from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.analyzer import normalize_confusion_matrix


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: Optional[Literal["true", "pred", "all"]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    show_values: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix as a seaborn heatmap.

    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names for axis labels
        normalize: Normalization mode (None for raw counts):
            - "true": Normalize over true labels (rows)
            - "pred": Normalize over predictions (columns)
            - "all": Normalize over all samples
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        cmap: Colormap name
        show_values: Whether to show values in cells

    Returns:
        Matplotlib Figure object
    """
    if normalize:
        cm_plot = normalize_confusion_matrix(cm, normalize)
    else:
        cm_plot = cm.astype(int)

    fig, ax = plt.subplots(figsize=figsize)

    fmt = ".2f" if normalize else "d"
    if not show_values or len(class_names) > 15:
        annot = False
    else:
        annot = True

    sns.heatmap(
        cm_plot,
        annot=annot,
        fmt=fmt if annot else "",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    title = "Confusion Matrix"
    if normalize:
        title += f" (normalized: {normalize})"
    ax.set_title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_accuracy(
    per_class_acc: dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (12, 8),
    color: str = "steelblue",
) -> plt.Figure:
    """
    Plot per-class accuracy as a sorted horizontal bar chart.

    Args:
        per_class_acc: Dictionary mapping class names to accuracy values
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        color: Bar color

    Returns:
        Matplotlib Figure object
    """
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    class_names = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(class_names))
    bars = ax.barh(y_pos, accuracies, color=color, alpha=0.8)

    mean_acc = np.mean(accuracies)
    ax.axvline(x=mean_acc, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_acc:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            min(acc + 0.01, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
