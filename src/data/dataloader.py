"""DataLoader factory for Gravity Spy dataset.

Creates train, validation, and test DataLoaders with appropriate
transforms and settings.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import (
    GravitySpyDataset,
    worker_init_fn,
)
from .transforms import (
    MultiViewTransform,
    get_eval_transforms,
    get_train_transforms,
)


class CollateError(Exception):
    """Raised when batch collation fails."""

    pass


class MultiViewTransformCollate:
    """Picklable collate function that applies consistent transforms to multi-view data.

    This class wraps a MultiViewTransform and applies it during collation,
    ensuring all views in each sample receive the same random spatial transforms.
    """

    def __init__(self, transform: "MultiViewTransform"):
        """
        Args:
            transform: MultiViewTransform instance to apply
        """
        self.transform = transform

    def __call__(
        self, batch: List[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply consistent transforms then collate."""
        transformed_batch = []
        for images_dict, label in batch:
            transformed_images = self.transform(images_dict)
            transformed_batch.append((transformed_images, label))
        return multiview_collate_fn(transformed_batch)


def multiview_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Custom collate function for multi-view mode.

    Converts list of (images_dict, label) to batched format.

    Args:
        batch: List of (images_dict, label) tuples from dataset

    Returns:
        images: Dict mapping time_scale string -> batched tensor (B, C, H, W)
        labels: Tensor of shape (B,)

    Raises:
        CollateError: If batch is empty, has inconsistent keys, or shape mismatches
    """
    if len(batch) == 0:
        raise CollateError("Empty batch received")

    images_list, labels = zip(*batch)

    time_scales = list(images_list[0].keys())

    # Validate all samples have consistent time scales
    for i, img_dict in enumerate(images_list):
        if set(img_dict.keys()) != set(time_scales):
            raise CollateError(
                f"Sample {i} has inconsistent time scales: "
                f"{set(img_dict.keys())} vs expected {set(time_scales)}"
            )

    batched_images: Dict[str, torch.Tensor] = {}
    for ts in time_scales:
        tensors = [img[ts] for img in images_list]
        # Validate shapes before stacking
        first_shape = tensors[0].shape
        for i, t in enumerate(tensors):
            if t.shape != first_shape:
                raise CollateError(
                    f"Shape mismatch at time_scale '{ts}', sample {i}: "
                    f"{t.shape} vs expected {first_shape}"
                )
        batched_images[ts] = torch.stack(tensors)  # shape: (B, C, H, W)

    labels_tensor = torch.tensor(labels, dtype=torch.long)  # shape: (B,)
    return batched_images, labels_tensor


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
    consistent_multiview_transforms: bool = True,
    sample_size: float = 1.0,
    seed: int = 42,
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
        consistent_multiview_transforms: If True (default), apply same random
            spatial transforms to all views in multi-view mode. Only affects
            training transforms.
        sample_size: Fraction of data to use (0.0-1.0). Useful for quick experiments.
        seed: Random seed for reproducible sampling.

    Returns:
        Configured DataLoader
    """
    # Set default shuffle based on split
    if shuffle is None:
        shuffle = split == "train"

    # Get appropriate transforms
    if split == "train":
        base_transform = get_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            augmentation=augmentation,
        )
    else:
        base_transform = get_eval_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
        )

    # For multi-view mode with training, wrap transform to ensure consistency
    if mode == "multi" and split == "train" and consistent_multiview_transforms:
        # Create a MultiViewTransform wrapper, but we'll apply it differently
        # since the dataset applies transforms per-image
        # We need to use MultiViewTransform at the dataset level
        multiview_transform = MultiViewTransform(
            base_transform, consistent_spatial=True
        )
        transform = None  # Dataset won't apply per-image transforms
    else:
        multiview_transform = None
        transform = base_transform

    # Create dataset
    dataset = GravitySpyDataset(
        root=root,
        split=split,
        mode=mode,
        time_scale=time_scale,
        transform=transform,
        class_names=class_names,
        sample_size=sample_size,
        seed=seed,
    )

    # If using multiview transform, we need a custom collate that applies it
    if multiview_transform is not None:
        collate_fn = MultiViewTransformCollate(multiview_transform)
    elif mode == "multi":
        collate_fn = multiview_collate_fn
    else:
        collate_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=split == "train",
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
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

    # Get sample_size for quick experiments
    sample_size = data_cfg.get("sample_size", 1.0)
    seed = cfg.get("experiment", {}).get("seed", 42)

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
        "sample_size": sample_size,
        "seed": seed,
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
