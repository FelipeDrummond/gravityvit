"""Gravity Spy PyTorch Dataset implementation.

Supports single-view and multi-view loading modes for spectrogram images
from the Gravity Spy glitch classification dataset.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GravitySpyDataset(Dataset):
    """PyTorch Dataset for Gravity Spy glitch spectrograms.

    Supports two loading modes:
    - Single-view: Returns (image, label) for a specified time scale
    - Multi-view: Returns (dict of images, label) with all 4 time scales

    Args:
        root: Path to data directory containing HDF5 and metadata CSV
        split: Data split to use ('train', 'validation', 'test')
        mode: Loading mode ('single' or 'multi')
        time_scale: Time scale for single-view mode (0.5, 1.0, 2.0, or 4.0)
        transform: Optional transform to apply to images
        class_names: List of class names (for label encoding)
    """

    TIME_SCALES = [0.5, 1.0, 2.0, 4.0]
    TIME_SCALE_FILES = ["0.5.png", "1.0.png", "2.0.png", "4.0.png"]

    def __init__(
        self,
        root: Union[str, Path],
        split: Literal["train", "validation", "test"] = "train",
        mode: Literal["single", "multi"] = "single",
        time_scale: float = 1.0,
        transform=None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.time_scale = time_scale
        self.transform = transform
        self._h5_file = None  # Initialize early for clean __del__

        # Validate time scale
        if time_scale not in self.TIME_SCALES:
            raise ValueError(
                f"time_scale must be one of {self.TIME_SCALES}, got {time_scale}"
            )
        self.time_scale_file = f"{time_scale}.png"

        # Load metadata
        metadata_path = self.root / "trainingset_v1d0_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["sample_type"] == split]
        self.metadata = self.metadata.reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split '{split}'")

        # Set up class names and label encoding
        if class_names is None:
            class_names = sorted(self.metadata["label"].unique())
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.num_classes = len(class_names)

        # HDF5 file path
        self.h5_path = self.root / "trainingsetv1d0.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # We don't keep HDF5 file open to support multiprocessing
        self._h5_file = None

    def _get_h5_file(self):
        """Get HDF5 file handle (lazy loading for multiprocessing support)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_image(self, label: str, split: str, sample_id: str, ts_file: str):
        """Load a single image from HDF5 file.

        Returns:
            Tensor of shape (C, H, W) where C=3 (RGB expanded from grayscale)
        """
        h5 = self._get_h5_file()
        # HDF5 structure: {class}/{split}/{sample_id}/{timescale}.png
        img = h5[label][split][sample_id][ts_file][:]

        # Handle channel-first format (1, H, W) -> expand to (3, H, W) for RGB
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Expand grayscale to RGB (3, H, W) for pretrained models
        img = np.stack([img, img, img], axis=0)

        return torch.from_numpy(img)

    def __getitem__(self, idx: int):
        """Get a sample by index.

        Returns:
            Single-view mode: (image, label) where image is (C, H, W)
            Multi-view mode: (images_dict, label) where images_dict maps
                             time_scale -> image tensor
        """
        row = self.metadata.iloc[idx]
        label_name = row["label"]
        sample_id = row["gravityspy_id"]
        split = row["sample_type"]

        label = self.class_to_idx[label_name]

        if self.mode == "single":
            image = self._load_image(label_name, split, sample_id, self.time_scale_file)
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        else:  # multi-view mode
            images = {}
            for ts, ts_file in zip(self.TIME_SCALES, self.TIME_SCALE_FILES):
                img = self._load_image(label_name, split, sample_id, ts_file)
                if self.transform is not None:
                    img = self.transform(img)
                images[ts] = img
            return images, label

    def close(self):
        """Close HDF5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced training.

        Returns:
            Tensor of shape (num_classes,) with normalized weights
        """
        class_counts = self.metadata["label"].value_counts()
        total = len(self.metadata)
        n_classes = self.num_classes

        weights = []
        for cls in self.class_names:
            count = class_counts.get(cls, 1)
            weight = total / (n_classes * count)
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize so weights sum to n_classes
        weights = weights / weights.sum() * n_classes
        return weights
