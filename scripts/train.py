"""Training script for GravityViT models.

Run with Hydra configuration:
    python scripts/train.py model=vit
    python scripts/train.py model=baseline_cnn data.batch_size=64
    python scripts/train.py model=multiview_vit train.epochs=50
"""

import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import create_dataloaders, get_class_weights
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(cfg: DictConfig) -> torch.nn.Module:
    """
    Create model based on configuration.

    Uses stub implementations until real models are implemented.
    """
    model_type = cfg.model.type
    num_classes = cfg.model.num_classes

    if model_type == "cnn":
        import torchvision.models as models

        model = models.resnet50(
            weights="IMAGENET1K_V2" if cfg.model.pretrained else None
        )
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        logger.info(f"Created ResNet50 baseline model with {num_classes} classes")

    elif model_type == "vit":
        import timm

        model = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=num_classes,
            drop_rate=cfg.model.dropout,
            attn_drop_rate=cfg.model.attn_dropout,
        )
        logger.info(
            f"Created ViT model ({cfg.model.backbone}) with {num_classes} classes"
        )

    elif model_type == "multiview_vit":
        raise NotImplementedError(
            "MultiViewViT not yet implemented. Use model=vit or model=baseline_cnn"
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training entry point."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)

    multi_view = cfg.model.type == "multiview_vit"
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, multi_view=multi_view
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    model = create_model(cfg)

    from src.utils import get_device

    device = get_device(cfg.train.get("device", "auto"))
    class_weights = get_class_weights(cfg, device)

    class_names = train_loader.dataset.classes

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        test_loader=test_loader,
        class_weights=class_weights,
        class_names=class_names,
    )

    results = trainer.train()

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {results.get('best_val_accuracy', 'N/A')}")
    logger.info(f"Best checkpoint: {results.get('best_checkpoint', 'N/A')}")

    return results


if __name__ == "__main__":
    main()
