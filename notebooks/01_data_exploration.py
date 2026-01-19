"""Gravity Spy Dataset Exploration.

This notebook explores the Gravity Spy training dataset from Zenodo
(DOI: 10.5281/zenodo.1476156).

The dataset contains spectrogram images of "glitches" - transient noise
artifacts in LIGO gravitational wave detector data, classified into
22 morphological categories.

Run with: marimo run notebooks/01_data_exploration.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Gravity Spy Dataset Exploration

        This notebook explores the Gravity Spy training dataset from Zenodo
        (DOI: 10.5281/zenodo.1476156).

        The dataset contains spectrogram images of "glitches" - transient noise
        artifacts in LIGO gravitational wave detector data, classified into
        22 morphological categories.

        ## Contents
        1. Load Dataset
        2. Dataset Structure
        3. Class Distribution
        4. Train/Val/Test Split Statistics
        5. Time-Scale Views Analysis
        6. Sample Visualizations
        7. Class Imbalance Analysis
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Data paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "gravityspy"
    METADATA_PATH = DATA_DIR / "trainingset_v1d0_metadata.csv"
    HDF5_PATH = DATA_DIR / "trainingsetv1d0.h5"

    # Time scales
    TIME_SCALES = [0.5, 1.0, 2.0, 4.0]

    print(f"Data directory: {DATA_DIR}")
    print(f"Metadata file exists: {METADATA_PATH.exists()}")
    print(f"HDF5 file exists: {HDF5_PATH.exists()}")

    return (
        DATA_DIR,
        HDF5_PATH,
        METADATA_PATH,
        PROJECT_ROOT,
        TIME_SCALES,
        Path,
        h5py,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(mo):
    mo.md("## 1. Load Dataset")
    return


@app.cell
def _(METADATA_PATH, pd):
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Total samples: {len(metadata)}")
    print(f"\nMetadata columns: {list(metadata.columns)}")
    metadata.head()
    return (metadata,)


@app.cell
def _(HDF5_PATH, h5py):
    def explore_h5_structure(h5_file, prefix=""):
        """Recursively print HDF5 file structure."""
        for key in h5_file.keys():
            item = h5_file[key]
            full_path = f"{prefix}/{key}" if prefix else key

            if isinstance(item, h5py.Group):
                print(f"Group: {full_path}")
                explore_h5_structure(item, full_path)
            else:
                print(f"Dataset: {full_path}, Shape: {item.shape}, Dtype: {item.dtype}")

    with h5py.File(HDF5_PATH, "r") as _f:
        print("HDF5 File Structure:")
        print("=" * 50)
        explore_h5_structure(_f)

    return (explore_h5_structure,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Dataset Structure

        The Gravity Spy dataset contains:
        - Spectrogram images of glitches from LIGO detectors (H1 and L1)
        - Each sample has 4 time-scale views: 0.5s, 1.0s, 2.0s, and 4.0s
        - 22 morphological classes representing different glitch types
        - Pre-defined train/validation/test splits
        """
    )
    return


@app.cell
def _(metadata):
    # Get all unique classes and sample types
    classes = sorted(metadata["label"].unique())
    sample_types = metadata["sample_type"].unique()

    print(f"Number of classes: {len(classes)}")
    print(f"\nClasses:")
    for _i, cls in enumerate(classes, 1):
        print(f"  {_i:2}. {cls}")

    print(f"\nSample types: {list(sample_types)}")

    return classes, sample_types


@app.cell
def _(mo):
    mo.md("## 3. Class Distribution")
    return


@app.cell
def _(DATA_DIR, metadata, plt):
    # Class distribution
    class_counts = metadata["label"].value_counts().sort_index()

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    ax1 = axes1[0]
    bars = ax1.barh(range(len(class_counts)), class_counts.values)
    ax1.set_yticks(range(len(class_counts)))
    ax1.set_yticklabels(class_counts.index, fontsize=9)
    ax1.set_xlabel("Number of Samples")
    ax1.set_title("Class Distribution")

    # Add value labels
    for _i, (bar, val) in enumerate(zip(bars, class_counts.values)):
        ax1.text(val + 10, _i, str(val), va="center", fontsize=8)

    # Percentage pie chart
    ax2 = axes1[1]
    # Group small classes for better visualization
    threshold = len(metadata) * 0.03  # 3% threshold
    large_classes = class_counts[class_counts >= threshold]
    small_classes_sum = class_counts[class_counts < threshold].sum()
    if small_classes_sum > 0:
        import pandas as pd_inner

        pie_data = pd_inner.concat(
            [large_classes, pd_inner.Series({"Other (< 3%)": small_classes_sum})]
        )
    else:
        pie_data = large_classes

    ax2.pie(
        pie_data.values,
        labels=pie_data.index,
        autopct="%1.1f%%",
        textprops={"fontsize": 8},
    )
    ax2.set_title("Class Distribution (%)")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    fig1

    return (
        ax1,
        ax2,
        axes1,
        bars,
        class_counts,
        fig1,
        large_classes,
        pie_data,
        small_classes_sum,
        threshold,
    )


@app.cell
def _(class_counts, metadata):
    # Summary statistics
    print(f"Class Distribution Statistics:")
    print(f"  Total samples: {len(metadata):,}")
    print(f"  Mean per class: {class_counts.mean():.1f}")
    print(f"  Std per class: {class_counts.std():.1f}")
    print(f"  Min: {class_counts.min()} ({class_counts.idxmin()})")
    print(f"  Max: {class_counts.max()} ({class_counts.idxmax()})")
    print(
        f"  Imbalance ratio (max/min): {class_counts.max() / class_counts.min():.1f}x"
    )
    return


@app.cell
def _(mo):
    mo.md("## 4. Train/Val/Test Split Statistics")
    return


@app.cell
def _(metadata):
    # Split statistics
    split_counts = metadata["sample_type"].value_counts()
    split_percentages = (split_counts / len(metadata) * 100).round(1)

    print("Dataset Splits:")
    print("=" * 40)
    for _split_type in ["train", "validation", "test"]:
        if _split_type in split_counts.index:
            count = split_counts[_split_type]
            pct = split_percentages[_split_type]
            print(f"  {_split_type:12}: {count:5,} samples ({pct}%)")

    return split_counts, split_percentages


@app.cell
def _(DATA_DIR, metadata, plt):
    # Class distribution per split
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    for _ax, _split_type in zip(axes2, ["train", "validation", "test"]):
        if _split_type in metadata["sample_type"].values:
            split_data = metadata[metadata["sample_type"] == _split_type]
            split_class_counts = split_data["label"].value_counts().sort_index()

            _ax.barh(range(len(split_class_counts)), split_class_counts.values)
            _ax.set_yticks(range(len(split_class_counts)))
            _ax.set_yticklabels(split_class_counts.index, fontsize=8)
            _ax.set_xlabel("Number of Samples")
            _ax.set_title(f"{_split_type.capitalize()} Set ({len(split_data):,} samples)")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "split_distribution.png", dpi=150, bbox_inches="tight")
    fig2

    return axes2, fig2


@app.cell
def _(metadata):
    # Detailed split statistics per class
    split_table = (
        metadata.groupby(["label", "sample_type"]).size().unstack(fill_value=0)
    )
    split_table["total"] = split_table.sum(axis=1)
    split_table = split_table.sort_values("total", ascending=False)

    print("\nSamples per Class by Split:")
    print(split_table.to_string())
    split_table

    return (split_table,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Time-Scale Views Analysis

        Each glitch sample has 4 spectrogram views at different time durations:
        - **0.5s**: Captures fast transients, high time resolution
        - **1.0s**: Balanced view
        - **2.0s**: Shows medium-duration features
        - **4.0s**: Captures long-duration glitch morphology
        """
    )
    return


@app.cell
def _(HDF5_PATH, h5py):
    # Load sample images from HDF5 to analyze time scales
    # HDF5 structure: {class}/{split}/{sample_id}/{timescale}.png
    time_scale_names = ["0.5.png", "1.0.png", "2.0.png", "4.0.png"]

    with h5py.File(HDF5_PATH, "r") as _f:
        print("HDF5 Structure: {class}/{split}/{sample_id}/{timescale}.png")
        print(f"\nTime scales available: {time_scale_names}")

        # Get a sample to show shape
        first_class = list(_f.keys())[0]
        first_split = list(_f[first_class].keys())[0]
        first_sample = list(_f[first_class][first_split].keys())[0]
        sample_img = _f[first_class][first_split][first_sample]["1.0.png"][:]
        print(f"Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}")

    return (time_scale_names,)


@app.cell
def _(DATA_DIR, HDF5_PATH, TIME_SCALES, time_scale_names, h5py, metadata, np, plt):
    # Visualize time-scale views for random samples
    # HDF5 structure: {class}/{split}/{sample_id}/{timescale}.png
    with h5py.File(HDF5_PATH, "r") as _f:
        # Select 3 random samples from metadata
        np.random.seed(42)
        sample_indices = np.random.choice(len(metadata), 3, replace=False)

        fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12))

        for row, idx in enumerate(sample_indices):
            row_data = metadata.iloc[idx]
            _label = row_data["label"]
            _split = row_data["sample_type"]
            _sample_id = row_data["gravityspy_id"]

            for col, (ts_name, time_scale) in enumerate(
                zip(time_scale_names, TIME_SCALES)
            ):
                # Access: class/split/sample_id/timescale.png
                _img = _f[_label][_split][_sample_id][ts_name][:]

                # Handle channel-first format (1, H, W)
                if len(_img.shape) == 3 and _img.shape[0] == 1:
                    _img = _img.squeeze(0)

                _ax = axes3[row, col]
                _ax.imshow(_img, cmap="viridis", aspect="auto")

                if row == 0:
                    _ax.set_title(f"{time_scale}s view", fontsize=12)
                if col == 0:
                    _ax.set_ylabel(f"{_label}", fontsize=10)
                _ax.axis("off")

        plt.suptitle(
            "Multi-Scale Spectrogram Views\n(Same glitch at different time durations)",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(DATA_DIR / "time_scale_views.png", dpi=150, bbox_inches="tight")
    fig3

    return axes3, fig3, sample_indices


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Sample Visualizations

        Visualize sample spectrograms for each of the 22 glitch classes.
        """
    )
    return


@app.cell
def _(classes, metadata):
    # Create mapping from class to sample metadata rows
    class_to_samples = {}
    for _label in classes:
        samples = metadata[metadata["label"] == _label][
            ["gravityspy_id", "sample_type"]
        ].values.tolist()
        class_to_samples[_label] = samples

    print(f"Class to sample mapping created for {len(class_to_samples)} classes")

    return (class_to_samples,)


@app.cell
def _(DATA_DIR, HDF5_PATH, class_to_samples, classes, h5py, np, plt):
    # Visualize one sample from each class (using 1.0s time scale)
    # HDF5 structure: {class}/{split}/{sample_id}/{timescale}.png
    with h5py.File(HDF5_PATH, "r") as _f:
        # Create grid for all 22 classes
        n_cols = 5
        n_rows = (len(classes) + n_cols - 1) // n_cols

        fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes4_flat = axes4.flatten()

        np.random.seed(42)

        for _i, _label in enumerate(classes):
            _ax = axes4_flat[_i]

            # Get random sample from this class
            if class_to_samples[_label]:
                sample_idx = np.random.randint(len(class_to_samples[_label]))
                _sample_id, _split = class_to_samples[_label][sample_idx]

                # Access: class/split/sample_id/1.0.png
                _img = _f[_label][_split][_sample_id]["1.0.png"][:]

                # Handle channel-first format (1, H, W)
                if len(_img.shape) == 3 and _img.shape[0] == 1:
                    _img = _img.squeeze(0)

                _ax.imshow(_img, cmap="viridis", aspect="auto")

            _ax.set_title(
                f"{_label}\n(n={len(class_to_samples[_label])})", fontsize=10
            )
            _ax.axis("off")

        # Hide unused subplots
        for _j in range(len(classes), len(axes4_flat)):
            axes4_flat[_j].axis("off")

        plt.suptitle(
            "Sample Spectrograms for Each Glitch Class (1.0s view)",
            fontsize=14,
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(DATA_DIR / "class_samples.png", dpi=150, bbox_inches="tight")
    fig4

    return axes4, axes4_flat, fig4, n_cols, n_rows


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Class Imbalance Analysis

        Analyzing the degree of class imbalance to inform training strategies.
        """
    )
    return


@app.cell
def _(classes, metadata, np):
    # Calculate class weights (inverse frequency)
    class_counts_imb = metadata["label"].value_counts()
    total_samples = len(metadata)
    n_classes = len(classes)

    # Different weighting strategies
    weights_inverse = total_samples / (n_classes * class_counts_imb)
    weights_sqrt = np.sqrt(weights_inverse)

    # Normalize weights
    weights_inverse_norm = weights_inverse / weights_inverse.sum() * n_classes
    weights_sqrt_norm = weights_sqrt / weights_sqrt.sum() * n_classes

    print("Class Imbalance Analysis")
    print("=" * 60)
    print(f"\nImbalance Metrics:")
    print(
        f"  Maximum samples: {class_counts_imb.max():,} ({class_counts_imb.idxmax()})"
    )
    print(
        f"  Minimum samples: {class_counts_imb.min():,} ({class_counts_imb.idxmin()})"
    )
    print(f"  Imbalance ratio: {class_counts_imb.max() / class_counts_imb.min():.1f}x")
    print(
        f"  Coefficient of variation: {class_counts_imb.std() / class_counts_imb.mean():.2%}"
    )

    return (
        class_counts_imb,
        n_classes,
        total_samples,
        weights_inverse,
        weights_inverse_norm,
        weights_sqrt,
        weights_sqrt_norm,
    )


@app.cell
def _(DATA_DIR, class_counts_imb, np, plt):
    # Visualize imbalance
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

    # Sorted bar chart
    ax5_1 = axes5[0]
    sorted_counts = class_counts_imb.sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_counts)))
    bars5 = ax5_1.barh(range(len(sorted_counts)), sorted_counts.values, color=colors)
    ax5_1.set_yticks(range(len(sorted_counts)))
    ax5_1.set_yticklabels(sorted_counts.index, fontsize=9)
    ax5_1.set_xlabel("Number of Samples")
    ax5_1.set_title("Classes Sorted by Sample Count")
    ax5_1.axvline(
        class_counts_imb.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {class_counts_imb.mean():.0f}",
    )
    ax5_1.legend()

    # Log scale distribution
    ax5_2 = axes5[1]
    ax5_2.bar(
        range(len(class_counts_imb)), sorted(class_counts_imb.values, reverse=True)
    )
    ax5_2.set_yscale("log")
    ax5_2.set_xlabel("Class Rank")
    ax5_2.set_ylabel("Number of Samples (log scale)")
    ax5_2.set_title("Class Distribution (Log Scale)")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "class_imbalance.png", dpi=150, bbox_inches="tight")
    fig5

    return ax5_1, ax5_2, axes5, bars5, colors, fig5, sorted_counts


@app.cell
def _(class_counts_imb, pd, weights_inverse_norm, weights_sqrt_norm):
    # Recommended class weights for training
    print("\nRecommended Class Weights for Training:")
    print("=" * 60)
    print("\nUsing inverse frequency weighting (normalized):")
    print("-" * 40)

    weight_df = pd.DataFrame(
        {
            "samples": class_counts_imb,
            "weight_inv": weights_inverse_norm,
            "weight_sqrt": weights_sqrt_norm,
        }
    ).sort_values("samples")

    print(weight_df.round(3).to_string())
    weight_df

    return (weight_df,)


@app.cell
def _(DATA_DIR, classes, np, weights_inverse_norm):
    # Save class weights as numpy arrays for training
    weight_array = np.array([weights_inverse_norm[cls] for cls in sorted(classes)])
    np.save(DATA_DIR / "class_weights.npy", weight_array)
    print(f"\nClass weights saved to: {DATA_DIR / 'class_weights.npy'}")

    return (weight_array,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Summary

        ### Dataset Overview
        - **Total samples**: ~8,500 labeled glitches
        - **Classes**: 22 morphological categories
        - **Views**: 4 time-scale spectrograms per sample (0.5s, 1.0s, 2.0s, 4.0s)
        - **Split**: ~80% train, ~10% validation, ~10% test

        ### Key Findings
        1. **Class Imbalance**: Significant imbalance exists between classes
           - Recommend using weighted loss or oversampling strategies
           - Class weights have been computed and saved

        2. **Multi-scale Views**: Different time scales capture different aspects of glitch morphology
           - Short views (0.5s) good for fast transients
           - Long views (4.0s) capture extended structure
           - Multi-view architecture should leverage all scales

        3. **Visual Distinctiveness**: Many classes have distinctive visual patterns
           - ViT attention mechanisms may learn to focus on relevant features
           - Cross-attention fusion can learn which time scales matter for each class
        """
    )
    return


@app.cell
def _(TIME_SCALES, class_counts, classes, metadata):
    # Final summary statistics
    print("Dataset Summary")
    print("=" * 50)
    print(f"Total samples: {len(metadata):,}")
    print(f"Number of classes: {len(classes)}")
    print(f"Time scales: {TIME_SCALES}")
    print(f"\nSplit sizes:")
    for _split in ["train", "validation", "test"]:
        if _split in metadata["sample_type"].values:
            n = len(metadata[metadata["sample_type"] == _split])
            print(f"  {_split}: {n:,} ({n/len(metadata)*100:.1f}%)")
    print(f"\nClass imbalance ratio: {class_counts.max() / class_counts.min():.1f}x")
    return


if __name__ == "__main__":
    app.run()
