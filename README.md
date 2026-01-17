# GravityViT: Vision Transformer-Based Multi-View Fusion for LIGO Glitch Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project explores Vision Transformer (ViT) architectures for classifying transient noise artifacts ("glitches") in LIGO gravitational wave detector data. We investigate whether attention-based models can outperform the CNN baselines used in the [Gravity Spy](https://gravityspy.org/) project, and whether attention patterns provide interpretable insights into glitch morphology.

### Key Contributions
- First ViT baseline on the Gravity Spy dataset
- Multi-view cross-attention fusion across time scales (0.5s, 1.0s, 2.0s, 4.0s)
- Attention-based interpretability analysis comparing ViT attention rollout vs CNN GradCAM
- Evaluation of pretrained vs scratch-trained models on scientific imaging data

## Background

LIGO detectors are susceptible to instrumental and environmental noise transients that can obscure or mimic gravitational wave signals. The Gravity Spy project categorizes these glitches into 22+ morphological classes using Q-transform spectrograms. Current classifiers use CNNs with simple fusion strategies (concatenation, mean/max pooling). We hypothesize that:

1. ViTs can capture long-range frequency-time dependencies better than CNNs
2. Cross-attention between time scales enables more sophisticated multi-view reasoning
3. Attention weights provide physically interpretable feature importance

## Dataset

**Gravity Spy Training Set** ([Zenodo: 10.5281/zenodo.1476156](https://zenodo.org/record/1476156))
- ~8,500 labeled glitches from LIGO O1/O2 observing runs
- 22 classes including: Blip, Koi_Fish, Scratchy, Whistle, Tomte, etc.
- 4 spectrogram views per glitch: 0.5s, 1.0s, 2.0s, 4.0s duration
- Standard train/val/test splits provided

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-View GravityViT                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │ 0.5s    │  │ 1.0s    │  │ 2.0s    │  │ 4.0s    │      │
│   │ View    │  │ View    │  │ View    │  │ View    │      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│        │            │            │            │            │
│        ▼            ▼            ▼            ▼            │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │ViT Enc │  │ViT Enc │  │ViT Enc │  │ViT Enc │      │
│   │(shared)│  │(shared)│  │(shared)│  │(shared)│      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│        │            │            │            │            │
│        └──────┬─────┴─────┬─────┴─────┬──────┘            │
│               │           │           │                    │
│               ▼           ▼           ▼                    │
│        ┌──────────────────────────────────┐               │
│        │     Cross-Attention Fusion       │               │
│        │   (Learnable view importance)    │               │
│        └─────────────┬────────────────────┘               │
│                      │                                     │
│                      ▼                                     │
│              ┌───────────────┐                            │
│              │  MLP Head     │                            │
│              │  (22 classes) │                            │
│              └───────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
gravityvit/
├── configs/                    # Hydra configuration files
│   ├── model/
│   ├── data/
│   └── train/
├── src/
│   ├── data/
│   │   ├── dataset.py         # GravitySpy dataset loader
│   │   ├── transforms.py      # Augmentations for spectrograms
│   │   └── download.py        # Zenodo download utilities
│   ├── models/
│   │   ├── baseline_cnn.py    # Gravity Spy CNN reproduction
│   │   ├── vit.py             # Single-view ViT
│   │   ├── multiview_vit.py   # Cross-attention multi-view ViT
│   │   └── fusion.py          # Fusion strategy implementations
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Class-weighted CE, focal loss
│   │   └── metrics.py         # Per-class recall, confusion matrix
│   └── analysis/
│       ├── attention_viz.py   # Attention rollout visualization
│       ├── gradcam.py         # GradCAM for CNN baseline
│       └── interpretability.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_validation.ipynb
│   └── 03_attention_analysis.ipynb
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/gravityvit.git
cd gravityvit

# Create environment
conda create -n gravityvit python=3.10
conda activate gravityvit

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py --output-dir data/
```

## Quick Start

```bash
# Train baseline CNN (reproduction)
python scripts/train.py model=baseline_cnn data.batch_size=32

# Train single-view ViT
python scripts/train.py model=vit_b16 data.view=1.0s

# Train multi-view ViT with cross-attention
python scripts/train.py model=multiview_vit fusion=cross_attention

# Evaluate and generate attention visualizations
python scripts/evaluate.py checkpoint=outputs/best.ckpt --visualize-attention
```

## Experiments

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Gravity Spy CNN (paper) | 97.1% | - | Bahaadini et al. 2018 |
| Baseline CNN (ours) | TBD | TBD | Reproduction |
| ViT-B/16 (single view) | TBD | TBD | ImageNet pretrained |
| ViT-B/16 (multi-view concat) | TBD | TBD | Late fusion |
| **GravityViT (cross-attn)** | TBD | TBD | Our method |

## References

```bibtex
@article{zevin2017gravity,
  title={Gravity Spy: integrating advanced LIGO detector characterization, machine learning, and citizen science},
  author={Zevin, Michael and others},
  journal={Classical and Quantum Gravity},
  volume={34},
  number={6},
  pages={064003},
  year={2017}
}

@article{bahaadini2018machine,
  title={Machine learning for Gravity Spy: Glitch classification and dataset},
  author={Bahaadini, Sara and others},
  journal={Information Sciences},
  volume={444},
  pages={172--186},
  year={2018}
}

@article{wu2024advancing,
  title={Advancing Glitch Classification in Gravity Spy: Multi-view Fusion with Attention-based Machine Learning},
  author={Wu, Yunan and others},
  journal={arXiv preprint arXiv:2401.12913},
  year={2024}
}

@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- LIGO Scientific Collaboration for the Gravity Spy dataset
- Zooniverse volunteers who labeled the training data
- Gravity Spy team for baseline implementations and documentation