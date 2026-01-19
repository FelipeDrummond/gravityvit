"""Data transforms for Gravity Spy spectrograms.

Provides transform pipelines for training (with augmentation) and
evaluation (no augmentation).
"""

from typing import Dict, List, Literal, Optional

import torch
from torchvision import transforms


def get_transforms(
    image_size: int = 224,
    mode: Literal["train", "eval"] = "train",
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augmentation: Optional[Dict] = None,
) -> transforms.Compose:
    """Build transform pipeline for spectrograms.

    Args:
        image_size: Target image size (square)
        mode: 'train' for training (with augmentation), 'eval' for evaluation
        normalize: Whether to apply normalization
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        augmentation: Dict with augmentation settings:
            - horizontal_flip: bool
            - vertical_flip: bool
            - rotation: float (max degrees)
            - color_jitter: float (jitter strength)

    Returns:
        Composed transform pipeline
    """
    # Default to ImageNet normalization stats
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transform_list = []

    # Resize to target size
    transform_list.append(transforms.Resize((image_size, image_size), antialias=True))

    # Training augmentations
    if mode == "train" and augmentation is not None:
        if augmentation.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if augmentation.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        rotation = augmentation.get("rotation", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        color_jitter = augmentation.get("color_jitter", 0)
        if color_jitter > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                )
            )

    # Normalization
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_train_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augmentation: Optional[Dict] = None,
) -> transforms.Compose:
    """Get transform pipeline for training.

    Convenience wrapper around get_transforms with mode='train'.
    """
    if augmentation is None:
        # Default augmentation settings from config
        augmentation = {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation": 15,
            "color_jitter": 0.1,
        }
    return get_transforms(
        image_size=image_size,
        mode="train",
        normalize=True,
        mean=mean,
        std=std,
        augmentation=augmentation,
    )


def get_eval_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Get transform pipeline for evaluation (val/test).

    Convenience wrapper around get_transforms with mode='eval'.
    """
    return get_transforms(
        image_size=image_size,
        mode="eval",
        normalize=True,
        mean=mean,
        std=std,
        augmentation=None,
    )


class MultiViewTransform:
    """Apply the same transform to all views in multi-view mode.

    For training, this ensures spatial augmentations (rotation, flip)
    are applied consistently across all time scales.

    Time scale keys are strings (e.g., "0.5", "1.0", "2.0", "4.0") to avoid
    floating-point comparison issues.
    """

    def __init__(self, transform: transforms.Compose, consistent_spatial: bool = True):
        """
        Args:
            transform: Transform to apply
            consistent_spatial: If True, apply same random spatial transform
                               to all views (recommended for training)
        """
        self.transform = transform
        self.consistent_spatial = consistent_spatial

    def __call__(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transform to all views.

        Args:
            images: Dict mapping time_scale string key -> image tensor

        Returns:
            Dict mapping time_scale string key -> transformed image tensor
        """
        if self.consistent_spatial:
            # Get random state for consistent transforms
            state = torch.get_rng_state()

            transformed = {}
            for ts, img in images.items():
                torch.set_rng_state(state)
                transformed[ts] = self.transform(img)
            return transformed
        else:
            return {ts: self.transform(img) for ts, img in images.items()}
