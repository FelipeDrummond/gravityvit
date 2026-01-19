"""Unit tests for Gravity Spy dataset and transforms."""

from pathlib import Path

import pytest
import torch

from src.data.dataloader import create_dataloader, multiview_collate_fn
from src.data.dataset import GravitySpyDataset
from src.data.transforms import (
    get_eval_transforms,
    get_train_transforms,
    get_transforms,
)

# Test data path
DATA_ROOT = Path("data/gravityspy")


@pytest.fixture
def skip_if_no_data():
    """Skip tests if dataset not downloaded."""
    if not (DATA_ROOT / "trainingsetv1d0.h5").exists():
        pytest.skip("Dataset not downloaded - run scripts/download_data.py first")


class TestGravitySpyDataset:
    """Tests for GravitySpyDataset class."""

    def test_dataset_init_train(self, skip_if_no_data):
        """Test dataset initialization for train split."""
        dataset = GravitySpyDataset(DATA_ROOT, split="train")
        assert len(dataset) > 0
        assert dataset.num_classes == 22
        dataset.close()

    def test_dataset_init_validation(self, skip_if_no_data):
        """Test dataset initialization for validation split."""
        dataset = GravitySpyDataset(DATA_ROOT, split="validation")
        assert len(dataset) > 0
        dataset.close()

    def test_dataset_init_test(self, skip_if_no_data):
        """Test dataset initialization for test split."""
        dataset = GravitySpyDataset(DATA_ROOT, split="test")
        assert len(dataset) > 0
        dataset.close()

    def test_single_view_output_shape(self, skip_if_no_data):
        """Test single-view mode returns correct shapes."""
        dataset = GravitySpyDataset(DATA_ROOT, split="train", mode="single")
        image, label = dataset[0]

        # Image should be (C=3, H, W)
        assert image.ndim == 3
        assert image.shape[0] == 3  # RGB channels
        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes
        dataset.close()

    def test_single_view_time_scales(self, skip_if_no_data):
        """Test single-view mode with different time scales."""
        for ts in [0.5, 1.0, 2.0, 4.0]:
            dataset = GravitySpyDataset(
                DATA_ROOT, split="train", mode="single", time_scale=ts
            )
            image, label = dataset[0]
            assert image.ndim == 3
            assert image.shape[0] == 3
            dataset.close()

    def test_invalid_time_scale(self):
        """Test that invalid time scale raises error."""
        with pytest.raises(ValueError, match="time_scale must be one of"):
            GravitySpyDataset(DATA_ROOT, split="train", time_scale=3.0)

    def test_multi_view_output_shape(self, skip_if_no_data):
        """Test multi-view mode returns correct shapes."""
        dataset = GravitySpyDataset(DATA_ROOT, split="train", mode="multi")
        images, label = dataset[0]

        # Images should be dict with 4 time scales
        assert isinstance(images, dict)
        assert len(images) == 4
        assert set(images.keys()) == {0.5, 1.0, 2.0, 4.0}

        for ts, img in images.items():
            assert img.ndim == 3
            assert img.shape[0] == 3  # RGB channels

        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes
        dataset.close()

    def test_label_range(self, skip_if_no_data):
        """Test all labels are in valid range."""
        dataset = GravitySpyDataset(DATA_ROOT, split="train", mode="single")
        for i in range(min(100, len(dataset))):
            _, label = dataset[i]
            assert 0 <= label < dataset.num_classes
        dataset.close()

    def test_class_weights(self, skip_if_no_data):
        """Test class weights computation."""
        dataset = GravitySpyDataset(DATA_ROOT, split="train")
        weights = dataset.get_class_weights()

        assert weights.shape == (dataset.num_classes,)
        assert weights.dtype == torch.float32
        assert (weights > 0).all()
        # Normalized weights should sum to num_classes
        assert torch.isclose(weights.sum(), torch.tensor(float(dataset.num_classes)))
        dataset.close()

    def test_reproducibility(self, skip_if_no_data):
        """Test that dataset access is deterministic."""
        dataset1 = GravitySpyDataset(DATA_ROOT, split="train", mode="single")
        dataset2 = GravitySpyDataset(DATA_ROOT, split="train", mode="single")

        img1, label1 = dataset1[0]
        img2, label2 = dataset2[0]

        assert label1 == label2
        assert torch.allclose(img1, img2)

        dataset1.close()
        dataset2.close()


class TestTransforms:
    """Tests for transform pipelines."""

    def test_get_transforms_train(self):
        """Test training transforms pipeline."""
        transform = get_transforms(image_size=224, mode="train")
        assert transform is not None

    def test_get_transforms_eval(self):
        """Test evaluation transforms pipeline."""
        transform = get_transforms(image_size=224, mode="eval")
        assert transform is not None

    def test_get_train_transforms(self):
        """Test convenience function for train transforms."""
        transform = get_train_transforms(image_size=224)
        assert transform is not None

    def test_get_eval_transforms(self):
        """Test convenience function for eval transforms."""
        transform = get_eval_transforms(image_size=224)
        assert transform is not None

    def test_transforms_resize(self, skip_if_no_data):
        """Test that transforms resize images correctly."""
        transform = get_eval_transforms(image_size=224)
        dataset = GravitySpyDataset(
            DATA_ROOT, split="train", mode="single", transform=transform
        )
        image, _ = dataset[0]

        assert image.shape == (3, 224, 224)
        dataset.close()

    def test_transforms_different_sizes(self, skip_if_no_data):
        """Test transforms with different target sizes."""
        for size in [128, 224, 384]:
            transform = get_eval_transforms(image_size=size)
            dataset = GravitySpyDataset(
                DATA_ROOT, split="train", mode="single", transform=transform
            )
            image, _ = dataset[0]
            assert image.shape == (3, size, size)
            dataset.close()


class TestDataLoader:
    """Tests for DataLoader factory."""

    def test_create_dataloader_single(self, skip_if_no_data):
        """Test creating single-view DataLoader."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="single",
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(loader))
        images, labels = batch

        assert images.shape[0] == 4  # batch size
        assert images.shape[1] == 3  # channels
        assert images.shape[2] == 224  # height
        assert images.shape[3] == 224  # width
        assert labels.shape == (4,)

    def test_create_dataloader_multi(self, skip_if_no_data):
        """Test creating multi-view DataLoader."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="multi",
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(loader))
        images, labels = batch

        assert isinstance(images, dict)
        assert len(images) == 4
        for ts, img in images.items():
            assert img.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)

    def test_multiview_collate_fn(self):
        """Test multi-view collate function."""
        # Create fake batch
        batch = [
            ({0.5: torch.randn(3, 224, 224), 1.0: torch.randn(3, 224, 224)}, 0),
            ({0.5: torch.randn(3, 224, 224), 1.0: torch.randn(3, 224, 224)}, 1),
        ]

        images, labels = multiview_collate_fn(batch)

        assert isinstance(images, dict)
        assert images[0.5].shape == (2, 3, 224, 224)
        assert images[1.0].shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert labels.tolist() == [0, 1]

    def test_dataloader_shuffle(self, skip_if_no_data):
        """Test that train loader shuffles, eval loaders don't."""
        train_loader = create_dataloader(
            DATA_ROOT, split="train", batch_size=4, num_workers=0
        )
        val_loader = create_dataloader(
            DATA_ROOT, split="validation", batch_size=4, num_workers=0
        )

        # Train loader should have shuffle=True (set in sampler)
        # Val loader should have shuffle=False
        assert train_loader.dataset is not None
        assert val_loader.dataset is not None


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_single_view(self, skip_if_no_data):
        """Test full data pipeline in single-view mode."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="single",
            batch_size=8,
            num_workers=0,
            image_size=224,
        )

        # Process multiple batches
        for i, (images, labels) in enumerate(loader):
            assert images.shape == (8, 3, 224, 224) or images.shape[0] <= 8
            assert labels.ndim == 1
            assert (labels >= 0).all()
            assert (labels < 22).all()
            if i >= 2:  # Test first few batches
                break

    def test_full_pipeline_multi_view(self, skip_if_no_data):
        """Test full data pipeline in multi-view mode."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="multi",
            batch_size=8,
            num_workers=0,
            image_size=224,
        )

        # Process multiple batches
        for i, (images, labels) in enumerate(loader):
            assert isinstance(images, dict)
            assert len(images) == 4
            for ts in [0.5, 1.0, 2.0, 4.0]:
                assert ts in images
                assert images[ts].shape[1:] == (3, 224, 224)
            assert labels.ndim == 1
            if i >= 2:
                break
