# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GravityViT applies Vision Transformer architectures to classify transient noise artifacts ("glitches") in LIGO gravitational wave detector data. The goal is to explore whether attention-based models can outperform CNN baselines and provide interpretable insights into glitch morphology.

**Project Status**: Data pipeline complete. Core model implementation in progress. See Linear board for current sprint tasks.

## Architecture

```
4 Time-Scale Views (0.5s, 1.0s, 2.0s, 4.0s spectrograms)
         ↓
   Shared ViT Encoder (ViT-B/16 from Timm library)
         ↓
   Cross-Attention Fusion Module (learns view importance)
         ↓
   MLP Classification Head (22 glitch classes)
```

Key design decisions:
- Multi-view approach processes spectrograms at different time scales simultaneously
- Single shared ViT encoder processes all views (parameter efficient)
- Cross-attention fusion learns which time scales matter for each glitch type
- Comparison against CNN baseline from Bahaadini et al. 2018 (97.1% accuracy)

## Development Environment

Supports both **Apple Silicon (MPS)** for local development and **CUDA** for training on Linux VMs.

```bash
# Create virtual environment with uv (recommended)
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# For CUDA on Linux VM (after activation):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Set up pre-commit hooks
pre-commit install
```

Device selection is automatic (`train.device: auto`) - priority: CUDA > MPS > CPU.

## Commands

```bash
# Download dataset
python scripts/download_data.py                # HDF5 file (~3.3 GB)
python scripts/download_data.py --include-png  # Also PNG archive (~9.6 GB)

# Training with Hydra
python scripts/train.py model=vit                           # Single-view ViT
python scripts/train.py model=baseline_cnn                  # CNN baseline
python scripts/train.py model=vit data.batch_size=64 train.epochs=50
python scripts/train.py model=vit train.loss.name=focal_loss  # Focal loss for imbalance

# Tests
pytest tests/                           # All tests
pytest tests/test_trainer.py            # Single test file
pytest tests/test_trainer.py::TestTrainerTraining::test_full_training_loop -v  # Single test

# Code quality
pre-commit run --all-files

# MLflow UI
mlflow ui  # http://localhost:5000
```

## Data Pipeline

1. **HDF5 Structure**: `{class}/{split}/{sample_id}/{timescale}.png` - grayscale spectrograms
2. **GravitySpyDataset** (`src/data/dataset.py`):
   - Lazy-loads HDF5 for multiprocessing compatibility
   - `multi_view=True` returns tensor of shape `(4, 3, H, W)` stacking all time scales
   - `multi_view=False` returns single view `(3, H, W)` (grayscale replicated to RGB)
   - Resizes to 224x224 for ViT input
3. **Class weights**: Pre-computed in `data/gravityspy/class_weights.npy` (set `data.use_class_weights: true`)

## Training Infrastructure

The `Trainer` class (`src/training/trainer.py`) handles:
- **AMP**: Automatic mixed precision on CUDA (`train.mixed_precision: true`)
- **Gradient clipping**: Default 1.0 (`train.grad_clip`)
- **LR schedule**: Linear warmup (5 epochs) + cosine annealing
- **Early stopping**: Monitors `val_accuracy` with patience=10
- **Checkpointing**: Saves `best_model.pt` and `last_model.pt` to Hydra output dir
- **MLflow logging**: Metrics per epoch, confusion matrix, classification report

**Loss functions** (`src/training/losses.py`):
- `cross_entropy` with optional label smoothing (default 0.1)
- `focal_loss` with configurable gamma (for class imbalance)

**Metrics** (`src/training/metrics.py`):
- `MetricTracker` accumulates predictions across batches
- Computes: accuracy, balanced_accuracy, f1_macro, f1_weighted

## Hydra Configuration

Configs compose: `config.yaml` → `model/*.yaml`, `data/*.yaml`, `train/*.yaml`

Override any value via CLI:
```bash
python scripts/train.py train.optimizer.lr=3e-4 train.scheduler.warmup_epochs=10
python scripts/train.py +experiment.name=my_experiment  # Add new key
```

Outputs go to `outputs/{experiment.name}/{timestamp}/`

## Dataset Details

Gravity Spy training set (Zenodo DOI: 10.5281/zenodo.1476156):
- ~8,500 labeled glitches across 22 morphological classes
- Each sample has 4 spectrogram views: 0.5s, 1.0s, 2.0s, 4.0s duration
- Significant class imbalance (see `notebooks/01_data_exploration.py`)
