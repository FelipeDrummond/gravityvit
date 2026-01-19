"""DataLoader factory for Gravity Spy dataset.

Creates train, validation, and test DataLoaders with appropriate
transforms and settings.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import GravitySpyDataset
from .transforms import get_eval_transforms, get_train_transforms


def multiview_collate_fn(batch):
    """Custom collate function for multi-view mode.

    Converts list of (images_dict, label) to batched format.

    Returns:
        images: Dict mapping time_scale -> batched tensor (B, C, H, W)
        labels: Tensor of shape (B,)
    """
    images_list, labels = zip(*batch)

    # Get time scales from first sample
    time_scales = list(images_list[0].keys())

    # Stack images for each time scale
    batched_images = {}
    for ts in time_scales:
        batched_images[ts] = torch.stack([img[ts] for img in images_list])

    labels = torch.tensor(labels, dtype=torch.long)
    return batched_images, labels


def create_dataloader(
    root: Union[str, Path],
    split: Literal["train", "validation", "test"],
    mode: Literal["single", "multi"] = "single",
    time_scale: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augmentation: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader for a specific split.

    Args:
        root: Path to data directory
        split: Data split ('train', 'validation', 'test')
        mode: Loading mode ('single' or 'multi')
        time_scale: Time scale for single-view mode
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        augmentation: Augmentation settings dict
        class_names: List of class names for label encoding
        shuffle: Whether to shuffle (default: True for train, False otherwise)

    Returns:
        Configured DataLoader
    """
    # Set default shuffle based on split
    if shuffle is None:
        shuffle = split == "train"

    # Get appropriate transforms
    if split == "train":
        transform = get_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            augmentation=augmentation,
        )
    else:
        transform = get_eval_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
        )

    # Create dataset
    dataset = GravitySpyDataset(
        root=root,
        split=split,
        mode=mode,
        time_scale=time_scale,
        transform=transform,
        class_names=class_names,
    )

    # Select collate function based on mode
    collate_fn = multiview_collate_fn if mode == "multi" else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=split == "train",
        persistent_workers=num_workers > 0,
    )


def create_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test DataLoaders from Hydra config.

    Args:
        cfg: Hydra config with 'data' section containing:
            - root: Data directory path
            - batch_size: Batch size
            - num_workers: Number of workers
            - pin_memory: Pin memory flag
            - image_size: Target image size
            - mean: Normalization mean
            - std: Normalization std
            - augmentation: Augmentation settings
            - class_names: List of class names
            - mode: 'single' or 'multi' (optional, default 'single')
            - time_scale: Time scale for single mode (optional, default 1.0)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    data_cfg = cfg.data

    # Extract config values
    root = data_cfg.root
    batch_size = data_cfg.batch_size
    num_workers = data_cfg.num_workers
    pin_memory = data_cfg.pin_memory
    image_size = data_cfg.image_size
    mean = list(data_cfg.mean)
    std = list(data_cfg.std)
    class_names = list(data_cfg.class_names)

    # Get augmentation config
    aug_cfg = data_cfg.get("augmentation", {})
    augmentation = (
        {
            "horizontal_flip": aug_cfg.get("horizontal_flip", True),
            "vertical_flip": aug_cfg.get("vertical_flip", False),
            "rotation": aug_cfg.get("rotation", 15),
            "color_jitter": aug_cfg.get("color_jitter", 0.1),
        }
        if aug_cfg.get("enabled", True)
        else None
    )

    # Get mode settings (check model config for multiview)
    mode = "multi" if cfg.get("model", {}).get("name") == "multiview_vit" else "single"
    time_scale = data_cfg.get("time_scale", 1.0)

    # Common kwargs for all loaders
    common_kwargs = {
        "root": root,
        "mode": mode,
        "time_scale": time_scale,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "class_names": class_names,
    }

    train_loader = create_dataloader(
        split="train",
        batch_size=batch_size,
        augmentation=augmentation,
        **common_kwargs,
    )

    val_loader = create_dataloader(
        split="validation",
        batch_size=batch_size,
        augmentation=None,
        **common_kwargs,
    )

    test_loader = create_dataloader(
        split="test",
        batch_size=batch_size,
        augmentation=None,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader, class_names
