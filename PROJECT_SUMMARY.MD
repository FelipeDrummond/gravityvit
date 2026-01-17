# Project Summary

## Repository Title

```
gravityvit
```

**Alternative options:**
- `gw-glitch-vit`
- `gravity-spy-transformers`
- `ligo-glitch-classifier`

I recommend **`gravityvit`** - it's concise, memorable, and clearly communicates the intersection of gravitational waves + Vision Transformers.

---

## Repository Description (for GitHub)

**Short (GitHub limit ~350 chars):**
```
Vision Transformer-based multi-view fusion for LIGO gravitational wave glitch classification. Explores cross-attention mechanisms across time scales (0.5s-4.0s) and provides interpretability analysis comparing ViT attention to CNN baselines on the Gravity Spy dataset.
```

**Extended (for README header or project page):**
```
🔭 GravityViT: Applying Vision Transformers to gravitational wave detector noise classification.

This project investigates whether attention-based architectures can improve upon CNN baselines for classifying transient noise artifacts in LIGO data, and whether attention patterns provide physically interpretable insights into glitch morphology.
```

---

## Ticket Summary

| ID | Title | Priority | Estimate | Epic |
|----|-------|----------|----------|------|
| GVIT-001 | Project scaffolding and environment setup | High | 2-3h | Setup |
| GVIT-002 | Dataset download and exploration | High | 2-3h | Setup |
| GVIT-003 | PyTorch dataset and dataloader | High | 3-4h | Setup |
| GVIT-004 | Reproduce Gravity Spy CNN baseline | High | 3-4h | Baseline |
| GVIT-005 | Training infrastructure and logging | High | 2-3h | Baseline |
| GVIT-006 | Evaluation metrics and analysis | Medium | 2h | Baseline |
| GVIT-007 | Single-view ViT implementation | High | 3-4h | ViT |
| GVIT-008 | Multi-view ViT with concat fusion | Medium | 2-3h | ViT |
| GVIT-009 | Multi-view ViT with cross-attention | High | 4-5h | ViT |
| GVIT-010 | Temporal positional encoding | Low | 2h | ViT |
| GVIT-011 | Attention visualization pipeline | Medium | 3-4h | Analysis |
| GVIT-012 | GradCAM for CNN comparison | Medium | 2h | Analysis |
| GVIT-013 | Interpretability analysis notebook | Medium | 3-4h | Analysis |
| GVIT-014 | Hyperparameter tuning | Medium | 3-4h | Experiments |
| GVIT-015 | Final evaluation and results | High | 3-4h | Experiments |
| GVIT-016 | Class imbalance experiments | Medium | 2-3h | Experiments |
| GVIT-017 | Code documentation and cleanup | Low | 2-3h | Release |
| GVIT-018 | Reproducibility package | Low | 2h | Release |

**Total estimated time: ~50-60 hours**

---

## Suggested Sprint Plan

### Sprint 1: Foundation (Week 1)
- GVIT-001: Project scaffolding
- GVIT-002: Dataset download and exploration
- GVIT-003: PyTorch dataset implementation

### Sprint 2: Baseline (Week 2)
- GVIT-004: CNN baseline reproduction
- GVIT-005: Training infrastructure
- GVIT-006: Evaluation metrics

### Sprint 3: ViT Core (Week 3)
- GVIT-007: Single-view ViT
- GVIT-008: Multi-view concat fusion

### Sprint 4: Novel Contribution (Week 4)
- GVIT-009: Cross-attention fusion
- GVIT-014: Hyperparameter tuning

### Sprint 5: Analysis (Week 5)
- GVIT-011: Attention visualization
- GVIT-012: GradCAM comparison
- GVIT-013: Interpretability notebook

### Sprint 6: Polish (Week 6)
- GVIT-015: Final evaluation
- GVIT-016: Class imbalance (if time)
- GVIT-017: Documentation
- GVIT-018: Reproducibility

---

## Key Dependencies Between Tickets

```
GVIT-001 ──► GVIT-002 ──► GVIT-003 ──┬──► GVIT-004 (CNN baseline)
                                      │
                                      └──► GVIT-007 (ViT) ──► GVIT-008 ──► GVIT-009
                                                │
                                                └──► GVIT-011 (attention viz)

GVIT-005 (training infra) ──► GVIT-004, GVIT-007, GVIT-008, GVIT-009

GVIT-006 (metrics) ──► GVIT-015 (final eval)
```

---

## First Steps After Creating Repository

1. **Create GitHub repo** with name `gravityvit`
2. **Copy README.md** from this draft
3. **Import tickets** into Linear (copy from TICKETS.md)
4. **Set up local environment:**
   ```bash
   git clone https://github.com/YOURUSERNAME/gravityvit.git
   cd gravityvit
   conda create -n gravityvit python=3.10
   conda activate gravityvit
   pip install torch torchvision timm hydra-core wandb scikit-learn matplotlib pandas jupyter
   ```
5. **Create folder structure** as specified in README
6. **Start GVIT-002** (download data) while setting up the rest