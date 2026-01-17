# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GravityViT applies Vision Transformer architectures to classify transient noise artifacts ("glitches") in LIGO gravitational wave detector data. The goal is to explore whether attention-based models can outperform CNN baselines and provide interpretable insights into glitch morphology.

## Project Status

Project scaffolding complete. Core implementation in progress. See Linear board for current sprint tasks.

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
# Create conda environment
conda env create -f environment.yml
conda activate gravityvit

# For CUDA on Linux VM (after activation):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Set up pre-commit hooks
pre-commit install
```

Device selection is automatic (`train.device: auto` in config) - detects CUDA > MPS > CPU.

## Commands

```bash
# Training (Hydra configuration)
python scripts/train.py model=baseline_cnn
python scripts/train.py model=multiview_vit
python scripts/train.py model=vit data.batch_size=32 train.epochs=100

# MLflow experiment tracking
mlflow ui  # Opens dashboard at http://localhost:5000

# Code quality
pre-commit run --all-files
pytest tests/
```

## Project Structure

```
src/
├── data/          # Dataset loader, transforms
├── models/        # CNN baseline, ViT, multi-view ViT
├── training/      # Training loop, losses, metrics
├── analysis/      # Attention visualization, GradCAM
└── utils.py       # Device detection, helpers

configs/
├── config.yaml    # Main config (composes others)
├── model/         # Model configs (baseline_cnn, vit, multiview_vit)
├── data/          # Dataset config (gravityspy)
└── train/         # Training config (default)

scripts/           # Entry points
notebooks/         # Analysis notebooks
tests/             # Unit tests
```

## Technology Stack

- **PyTorch 2.0+** with **Timm** for pretrained ViT models
- **Hydra** for configuration management
- **MLflow** for experiment tracking
- **scikit-learn** for evaluation metrics
- **Pre-commit** with black, isort, flake8

## Dataset

Gravity Spy training set from Zenodo (DOI: 10.5281/zenodo.1476156):
- ~8,500 labeled glitches across 22 morphological classes
- Each sample has 4 spectrogram views at different time durations (0.5s, 1.0s, 2.0s, 4.0s)
