"""Tests for dataset module."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.dataset import GravitySpyDataset


@pytest.fixture
def data_paths():
    """Return paths to test data if available."""
    data_dir = Path("data/gravityspy")
    hdf5_path = data_dir / "trainingsetv1d0.h5"
    metadata_path = data_dir / "trainingset_v1d0_metadata.csv"

    if not hdf5_path.exists() or not metadata_path.exists():
        pytest.skip("Dataset files not available")

    return hdf5_path, metadata_path


@pytest.fixture
def metadata(data_paths):
    """Load metadata DataFrame."""
    _, metadata_path = data_paths
    return pd.read_csv(metadata_path)


class TestGravitySpyDataset:
    def test_dataset_length(self, data_paths, metadata):
        """Should return correct number of samples."""
        hdf5_path, _ = data_paths

        train_count = len(metadata[metadata["sample_type"] == "train"])

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
            time_scale=1.0,
        )

        assert len(dataset) == train_count

    def test_single_view_output_shape(self, data_paths, metadata):
        """Single-view should return (3, 224, 224) tensor."""
        hdf5_path, _ = data_paths

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
            time_scale=1.0,
            input_size=224,
            multi_view=False,
        )

        images, label = dataset[0]

        assert images.shape == (3, 224, 224)
        assert images.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes

    def test_multi_view_output_shape(self, data_paths, metadata):
        """Multi-view should return (4, 3, 224, 224) tensor."""
        hdf5_path, _ = data_paths

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
            multi_view=True,
            input_size=224,
        )

        images, label = dataset[0]

        assert images.shape == (4, 3, 224, 224)
        assert images.dtype == torch.float32

    def test_different_splits(self, data_paths, metadata):
        """Should load different splits correctly."""
        hdf5_path, _ = data_paths

        train_ds = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
        )

        val_ds = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="val",
        )

        test_ds = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="test",
        )

        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(metadata)

    def test_class_mapping(self, data_paths, metadata):
        """Should create consistent class to index mapping."""
        hdf5_path, _ = data_paths

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
        )

        assert dataset.num_classes == 22
        assert len(dataset.class_to_idx) == 22
        assert len(dataset.classes) == 22

    def test_invalid_time_scale_raises(self, data_paths, metadata):
        """Should raise for invalid time scale."""
        hdf5_path, _ = data_paths

        with pytest.raises(ValueError, match="time_scale must be"):
            GravitySpyDataset(
                hdf5_path=hdf5_path,
                metadata=metadata,
                split="train",
                time_scale=3.0,
            )

    def test_image_normalization(self, data_paths, metadata):
        """Images should be normalized to [0, 1] range."""
        hdf5_path, _ = data_paths

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
        )

        images, _ = dataset[0]

        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_custom_input_size(self, data_paths, metadata):
        """Should resize to custom input size."""
        hdf5_path, _ = data_paths

        dataset = GravitySpyDataset(
            hdf5_path=hdf5_path,
            metadata=metadata,
            split="train",
            input_size=128,
        )

        images, _ = dataset[0]
        assert images.shape == (3, 128, 128)


class TestCreateDataloaders:
    def test_create_dataloaders(self, data_paths):
        """Should create train/val/test loaders."""
        from omegaconf import OmegaConf

        from src.data.dataset import create_dataloaders

        cfg = OmegaConf.create(
            {
                "data": {
                    "data_dir": "data/gravityspy",
                    "hdf5_file": "trainingsetv1d0.h5",
                    "metadata_file": "trainingset_v1d0_metadata.csv",
                    "default_time_scale": 1.0,
                    "input_size": 224,
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                }
            }
        )

        train_loader, val_loader, test_loader = create_dataloaders(cfg)

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
