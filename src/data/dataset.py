"""Gravity Spy PyTorch Dataset implementation.

Supports single-view and multi-view loading modes for spectrogram images
from the Gravity Spy glitch classification dataset.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Use string keys for time scales to avoid float comparison issues
TIME_SCALE_KEYS = ["0.5", "1.0", "2.0", "4.0"]
TIME_SCALE_VALUES = [0.5, 1.0, 2.0, 4.0]
TIME_SCALE_FILES = ["0.5.png", "1.0.png", "2.0.png", "4.0.png"]


class HDF5AccessError(Exception):
    """Raised when HDF5 data access fails."""

    pass


class GravitySpyDataset(Dataset):
    """PyTorch Dataset for Gravity Spy glitch spectrograms.

    Supports two loading modes:
    - Single-view: Returns (image, label) for a specified time scale
    - Multi-view: Returns (dict of images, label) with all 4 time scales

    This dataset is designed to work safely with PyTorch's multiprocessing
    DataLoader. Each worker process opens its own HDF5 file handle.

    Can be used as a context manager:
        with GravitySpyDataset(root, split="train") as dataset:
            image, label = dataset[0]

    Args:
        root: Path to data directory containing HDF5 and metadata CSV
        split: Data split to use ('train', 'validation', 'test')
        mode: Loading mode ('single' or 'multi')
        time_scale: Time scale for single-view mode (0.5, 1.0, 2.0, or 4.0)
        transform: Optional transform to apply to images
        class_names: List of class names (for label encoding). If provided,
            must contain all classes present in the split's metadata.
        sample_size: Fraction of data to use (0.0-1.0). Useful for quick experiments.
        seed: Random seed for reproducible sampling when sample_size < 1.0.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Literal["train", "validation", "test"] = "train",
        mode: Literal["single", "multi"] = "single",
        time_scale: float = 1.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        class_names: Optional[List[str]] = None,
        sample_size: float = 1.0,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.time_scale = time_scale
        self.transform = transform
        self._h5_file: Optional[h5py.File] = None
        self._worker_id: Optional[int] = None

        # Validate time scale
        if time_scale not in TIME_SCALE_VALUES:
            raise ValueError(
                f"time_scale must be one of {TIME_SCALE_VALUES}, got {time_scale}"
            )
        self.time_scale_key = str(time_scale)
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

        # Sample data if sample_size < 1.0 (for quick experiments)
        if sample_size < 1.0:
            n_samples = max(1, int(len(self.metadata) * sample_size))
            self.metadata = self.metadata.sample(n=n_samples, random_state=seed)
            self.metadata = self.metadata.reset_index(drop=True)

        # Get classes present in this split
        split_classes = set(self.metadata["label"].unique())

        # Set up class names and label encoding
        if class_names is None:
            class_names = sorted(split_classes)
        else:
            # Validate that all split classes are in provided class_names
            missing_classes = split_classes - set(class_names)
            if missing_classes:
                raise ValueError(
                    f"class_names missing classes present in '{split}' split: "
                    f"{sorted(missing_classes)}"
                )

        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.num_classes = len(class_names)

        # HDF5 file path - validate existence but don't open yet
        self.h5_path = self.root / "trainingsetv1d0.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

    def _get_h5_file(self) -> h5py.File:
        """Get HDF5 file handle with proper multiprocessing support.

        Each worker process gets its own file handle. The handle is opened
        lazily on first access and cached per-worker.
        """
        # Check if we're in a different worker than before
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else None

        # If worker changed or file not open, (re)open it
        if self._h5_file is None or self._worker_id != current_worker_id:
            # Close existing handle if any
            if self._h5_file is not None:
                try:
                    self._h5_file.close()
                except Exception:
                    pass
            self._h5_file = h5py.File(self.h5_path, "r")
            self._worker_id = current_worker_id

        return self._h5_file

    def __len__(self) -> int:
        return len(self.metadata)

    def _validate_image_array(self, img: np.ndarray, context: str) -> np.ndarray:
        """Validate and normalize image array from HDF5.

        Args:
            img: Raw image array from HDF5
            context: Description of the image for error messages

        Returns:
            Normalized 2D grayscale array with shape (H, W) and values in [0, 1]

        Raises:
            ValueError: If image has unexpected shape or dtype
        """
        # Handle various possible shapes
        if img.ndim == 2:
            # Already (H, W) - expected case
            pass
        elif img.ndim == 3:
            if img.shape[0] == 1:
                # (1, H, W) -> (H, W)
                img = img.squeeze(0)
            elif img.shape[2] == 1:
                # (H, W, 1) -> (H, W)
                img = img.squeeze(2)
            elif img.shape[0] == 3:
                # (3, H, W) - already RGB, take first channel
                logger.warning(
                    f"{context}: Found RGB image (3, H, W), using first channel"
                )
                img = img[0]
            elif img.shape[2] == 3:
                # (H, W, 3) - RGB channels-last, take first channel
                logger.warning(
                    f"{context}: Found RGB image (H, W, 3), using first channel"
                )
                img = img[:, :, 0]
            else:
                raise ValueError(
                    f"{context}: Unexpected 3D image shape {img.shape}. "
                    "Expected (1, H, W), (H, W, 1), (3, H, W), or (H, W, 3)"
                )
        else:
            raise ValueError(
                f"{context}: Unexpected image dimensions {img.ndim}. "
                f"Expected 2D or 3D array, got shape {img.shape}"
            )

        # Validate and normalize dtype/range
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype in (np.float32, np.float64):
            img = img.astype(np.float32)
            # Check if already normalized
            if img.max() > 1.0:
                if img.max() <= 255.0:
                    img = img / 255.0
                elif img.max() <= 65535.0:
                    img = img / 65535.0
                else:
                    raise ValueError(
                        f"{context}: Float image has unexpected range "
                        f"[{img.min():.2f}, {img.max():.2f}]"
                    )
        else:
            raise ValueError(
                f"{context}: Unexpected image dtype {img.dtype}. "
                "Expected uint8, uint16, float32, or float64"
            )

        return img

    def _load_image(
        self, label: str, split: str, sample_id: str, ts_file: str
    ) -> torch.Tensor:
        """Load a single image from HDF5 file.

        Args:
            label: Class label (HDF5 group name)
            split: Data split (HDF5 subgroup name)
            sample_id: Sample identifier (HDF5 subgroup name)
            ts_file: Time scale filename (HDF5 dataset name)

        Returns:
            Tensor of shape (3, H, W) - RGB expanded from grayscale

        Raises:
            HDF5AccessError: If the requested data path doesn't exist in HDF5
        """
        h5 = self._get_h5_file()
        h5_path = f"{label}/{split}/{sample_id}/{ts_file}"

        try:
            img = h5[label][split][sample_id][ts_file][:]
        except KeyError as e:
            raise HDF5AccessError(
                f"Failed to load image from HDF5 path '{h5_path}': {e}. "
                f"Sample may be missing or corrupted."
            ) from e

        # Validate and normalize image
        context = f"Image at '{h5_path}'"
        img = self._validate_image_array(img, context)  # shape: (H, W)

        # Expand grayscale to RGB
        img = np.stack([img, img, img], axis=0)  # shape: (3, H, W)

        return torch.from_numpy(img)  # shape: (3, H, W)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[Dict[str, torch.Tensor], int]]:
        """Get a sample by index.

        Returns:
            Single-view mode: (image, label) where image is (C, H, W)
            Multi-view mode: (images_dict, label) where images_dict maps
                             time_scale string key -> image tensor

        Raises:
            HDF5AccessError: If the sample data cannot be loaded from HDF5
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
            images: Dict[str, torch.Tensor] = {}
            for ts_key, ts_file in zip(TIME_SCALE_KEYS, TIME_SCALE_FILES):
                img = self._load_image(label_name, split, sample_id, ts_file)
                if self.transform is not None:
                    img = self.transform(img)
                images[ts_key] = img
            return images, label

    def close(self) -> None:
        """Close HDF5 file handle."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
            self._worker_id = None

    def __enter__(self) -> "GravitySpyDataset":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures file handle is closed."""
        self.close()

    def __del__(self):
        """Destructor - attempt to close file handle."""
        self.close()

    def get_class_weights(self, validate_all_classes: bool = True) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced training.

        Args:
            validate_all_classes: If True (default), raise error if any class
                in class_names has zero samples. If False, assign weight of 0
                to missing classes.

        Returns:
            Tensor of shape (num_classes,) with normalized weights

        Raises:
            ValueError: If validate_all_classes=True and a class has no samples
        """
        class_counts = self.metadata["label"].value_counts()
        total = len(self.metadata)
        n_classes = self.num_classes

        weights = []
        for cls in self.class_names:
            count = class_counts.get(cls, 0)
            if count == 0:
                if validate_all_classes:
                    raise ValueError(
                        f"Class '{cls}' has no samples in '{self.split}' split. "
                        "Either remove from class_names or set validate_all_classes=False"
                    )
                weights.append(0.0)
            else:
                weight = total / (n_classes * count)
                weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)

        # Normalize so non-zero weights sum to n_classes
        non_zero_sum = weights.sum()
        if non_zero_sum > 0:
            weights = weights / non_zero_sum * n_classes

        return weights


def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for DataLoader multiprocessing.

    This ensures each worker has a unique random seed and properly
    initializes its HDF5 file handle.

    Usage:
        DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Set unique seed for this worker
        seed = worker_info.seed % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
