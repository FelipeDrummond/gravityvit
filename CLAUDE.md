# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GravityViT applies Vision Transformer architectures to classify transient noise artifacts ("glitches") in LIGO gravitational wave detector data. The goal is to explore whether attention-based models can outperform CNN baselines and provide interpretable insights into glitch morphology.

## Project Status

This repository is in the **planning and initial setup phase**. Only documentation files exist currently. See PROJECT_SUMMARY.MD for the 6-week sprint breakdown and 18-ticket implementation roadmap.

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

This project supports both **Apple Silicon (MPS)** for local development and **CUDA** for training on Linux VMs.

```bash
# Create conda environment
conda env create -f environment.yml
conda activate gravityvit

# For CUDA on Linux VM (after activation):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Device selection is automatic (`train.device: auto` in config) - detects CUDA > MPS > CPU.

## Planned Commands

Once implemented, the main entry points will be:

```bash
# Install dependencies (alternative to conda)
pip install -r requirements.txt

# Download Gravity Spy dataset from Zenodo
python scripts/download_data.py

# Train models (Hydra configuration)
python scripts/train.py model=baseline_cnn
python scripts/train.py model=multiview_vit fusion=cross_attention

# Evaluate and visualize attention
python scripts/evaluate.py --visualize-attention
```

Configuration uses Hydra with YAML files in `configs/`. Override via CLI:
```bash
python scripts/train.py data.batch_size=32 train.epochs=100
```

## Planned Project Structure

```
src/
├── data/          # GravitySpy dataset loader, transforms, Zenodo download
├── models/        # CNN baseline, single-view ViT, multi-view ViT with cross-attention
├── training/      # Training loop, focal loss, weighted CE, metrics
└── analysis/      # Attention visualization, GradCAM, interpretability

configs/           # Hydra YAML configs for model, data, training
scripts/           # Entry points: download_data.py, train.py, evaluate.py
notebooks/         # Data exploration, baseline validation, attention analysis
```

## Technology Stack

- **PyTorch 2.0+** with **Timm** for pretrained ViT models
- **Hydra** for configuration management
- **Weights & Biases** for experiment tracking
- **scikit-learn** for evaluation metrics

## Dataset

Gravity Spy training set from Zenodo (DOI: 10.5281/zenodo.1476156):
- ~8,500 labeled glitches across 22 morphological classes
- Each sample has 4 spectrogram views at different time durations
