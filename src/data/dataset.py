"""GravitySpy dataset loader for HDF5 spectrograms."""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class GravitySpyDataset(Dataset):
    """
    Dataset for Gravity Spy spectrograms stored in HDF5 format.

    HDF5 structure: {class}/{split}/{sample_id}/{timescale}.png

    Args:
        hdf5_path: Path to HDF5 file
        metadata: DataFrame with columns [gravityspy_id, label, sample_type]
        split: One of 'train', 'validation', 'test'
        time_scale: Time scale to use (0.5, 1.0, 2.0, 4.0) or None for multi-view
        input_size: Target image size (default 224)
        multi_view: If True, return all 4 time scales stacked
    """

    TIME_SCALES = [0.5, 1.0, 2.0, 4.0]
    TIME_SCALE_FILES = ["0.5.png", "1.0.png", "2.0.png", "4.0.png"]

    def __init__(
        self,
        hdf5_path: Path,
        metadata: pd.DataFrame,
        split: str,
        time_scale: float = 1.0,
        input_size: int = 224,
        multi_view: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.input_size = input_size
        self.multi_view = multi_view
        self.time_scale = time_scale
        self._h5file: Optional[h5py.File] = None

        if not multi_view and time_scale not in self.TIME_SCALES:
            raise ValueError(f"time_scale must be one of {self.TIME_SCALES}")

        self.time_scale_file = f"{time_scale}.png"

        split_map = {"train": "train", "val": "validation", "test": "test"}
        split_name = split_map.get(split, split)
        self.samples = metadata[metadata["sample_type"] == split_name][
            ["gravityspy_id", "label"]
        ].values.tolist()

        self.classes = sorted(metadata["label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def _get_h5file(self) -> h5py.File:
        """Lazy-load HDF5 file (for multiprocessing compatibility)."""
        if self._h5file is None:
            self._h5file = h5py.File(self.hdf5_path, "r")
        return self._h5file

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample_id, label = self.samples[idx]
        label_idx = self.class_to_idx[label]

        h5file = self._get_h5file()

        if self.multi_view:
            views = []
            for ts_file in self.TIME_SCALE_FILES:
                img = h5file[label]["train"][sample_id][ts_file][:]
                img_tensor = self._preprocess(img)
                views.append(img_tensor)
            # Stack: (4, 3, H, W)
            images = torch.stack(views, dim=0)
        else:
            # Find correct split in HDF5
            for split_name in ["train", "validation", "test"]:
                if sample_id in h5file[label][split_name]:
                    img = h5file[label][split_name][sample_id][self.time_scale_file][:]
                    break
            images = self._preprocess(img)

        return images, label_idx

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize and convert to RGB."""
        # img shape: (1, H, W) or (H, W)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

        # Normalize to [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # Resize to input_size: (1, H, W) -> (1, input_size, input_size)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Convert grayscale to RGB by repeating: (1, H, W) -> (3, H, W)
        img_tensor = img_tensor.expand(3, -1, -1)

        return img_tensor

    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()


def create_dataloaders(
    cfg: DictConfig, multi_view: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        cfg: Hydra configuration
        multi_view: If True, each sample returns all 4 time scales

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(cfg.data.data_dir)
    hdf5_path = data_dir / cfg.data.hdf5_file
    metadata_path = data_dir / cfg.data.metadata_file

    metadata = pd.read_csv(metadata_path)

    sample_size = cfg.data.get("sample_size", 1.0)
    if sample_size < 1.0:
        seed = cfg.experiment.get("seed", 42)
        sampled_dfs = []
        for split_type in metadata["sample_type"].unique():
            split_df = metadata[metadata["sample_type"] == split_type]
            sampled_dfs.append(split_df.sample(frac=sample_size, random_state=seed))
        metadata = pd.concat(sampled_dfs, ignore_index=True)

    time_scale = cfg.data.get("default_time_scale", 1.0)
    input_size = cfg.data.get("input_size", 224)

    train_dataset = GravitySpyDataset(
        hdf5_path=hdf5_path,
        metadata=metadata,
        split="train",
        time_scale=time_scale,
        input_size=input_size,
        multi_view=multi_view,
    )

    val_dataset = GravitySpyDataset(
        hdf5_path=hdf5_path,
        metadata=metadata,
        split="val",
        time_scale=time_scale,
        input_size=input_size,
        multi_view=multi_view,
    )

    test_dataset = GravitySpyDataset(
        hdf5_path=hdf5_path,
        metadata=metadata,
        split="test",
        time_scale=time_scale,
        input_size=input_size,
        multi_view=multi_view,
    )

    batch_size = cfg.data.get("batch_size", 32)
    num_workers = cfg.data.get("num_workers", 4)
    pin_memory = cfg.data.get("pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(cfg: DictConfig, device: torch.device) -> Optional[torch.Tensor]:
    """Load precomputed class weights if configured."""
    if not cfg.data.get("use_class_weights", False):
        return None

    weights_path = Path(cfg.data.class_weights_file)
    if not weights_path.exists():
        return None

    weights = np.load(weights_path)
    return torch.from_numpy(weights).float().to(device)
