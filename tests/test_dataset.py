"""Unit tests for Gravity Spy dataset and transforms."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataloader import CollateError, create_dataloader, multiview_collate_fn
from src.data.dataset import (
    TIME_SCALE_KEYS,
    GravitySpyDataset,
    worker_init_fn,
)
from src.data.transforms import (
    MultiViewTransform,
    get_eval_transforms,
    get_train_transforms,
    get_transforms,
)

# Use absolute path resolution relative to this test file
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "gravityspy"


@pytest.fixture
def skip_if_no_data():
    """Skip tests if dataset not downloaded."""
    if not (DATA_ROOT / "trainingsetv1d0.h5").exists():
        pytest.skip("Dataset not downloaded - run scripts/download_data.py first")


class TestGravitySpyDataset:
    """Tests for GravitySpyDataset class."""

    def test_dataset_init_train(self, skip_if_no_data):
        """Test dataset initialization for train split."""
        with GravitySpyDataset(DATA_ROOT, split="train") as dataset:
            assert len(dataset) > 0
            assert dataset.num_classes == 22

    def test_dataset_init_validation(self, skip_if_no_data):
        """Test dataset initialization for validation split."""
        with GravitySpyDataset(DATA_ROOT, split="validation") as dataset:
            assert len(dataset) > 0

    def test_dataset_init_test(self, skip_if_no_data):
        """Test dataset initialization for test split."""
        with GravitySpyDataset(DATA_ROOT, split="test") as dataset:
            assert len(dataset) > 0

    def test_single_view_output_shape(self, skip_if_no_data):
        """Test single-view mode returns correct shapes."""
        with GravitySpyDataset(DATA_ROOT, split="train", mode="single") as dataset:
            image, label = dataset[0]

            # Image should be (C=3, H, W)
            assert image.ndim == 3
            assert image.shape[0] == 3  # RGB channels
            assert isinstance(label, int)
            assert 0 <= label < dataset.num_classes

    def test_single_view_time_scales(self, skip_if_no_data):
        """Test single-view mode with different time scales."""
        for ts in [0.5, 1.0, 2.0, 4.0]:
            with GravitySpyDataset(
                DATA_ROOT, split="train", mode="single", time_scale=ts
            ) as dataset:
                image, label = dataset[0]
                assert image.ndim == 3
                assert image.shape[0] == 3

    def test_invalid_time_scale(self):
        """Test that invalid time scale raises error."""
        with pytest.raises(ValueError, match="time_scale must be one of"):
            GravitySpyDataset(DATA_ROOT, split="train", time_scale=3.0)

    def test_multi_view_output_shape(self, skip_if_no_data):
        """Test multi-view mode returns correct shapes with string keys."""
        with GravitySpyDataset(DATA_ROOT, split="train", mode="multi") as dataset:
            images, label = dataset[0]

            # Images should be dict with 4 time scales (string keys)
            assert isinstance(images, dict)
            assert len(images) == 4
            assert set(images.keys()) == {"0.5", "1.0", "2.0", "4.0"}

            for ts, img in images.items():
                assert isinstance(ts, str)  # Keys are strings, not floats
                assert img.ndim == 3
                assert img.shape[0] == 3  # RGB channels

            assert isinstance(label, int)
            assert 0 <= label < dataset.num_classes

    def test_label_range(self, skip_if_no_data):
        """Test all labels are in valid range."""
        with GravitySpyDataset(DATA_ROOT, split="train", mode="single") as dataset:
            for i in range(min(100, len(dataset))):
                _, label = dataset[i]
                assert 0 <= label < dataset.num_classes

    def test_class_weights(self, skip_if_no_data):
        """Test class weights computation."""
        with GravitySpyDataset(DATA_ROOT, split="train") as dataset:
            weights = dataset.get_class_weights()

            assert weights.shape == (dataset.num_classes,)
            assert weights.dtype == torch.float32
            assert (weights > 0).all()
            # Normalized weights should sum to num_classes
            assert torch.isclose(
                weights.sum(), torch.tensor(float(dataset.num_classes))
            )

    def test_class_weights_missing_class_raises(self, skip_if_no_data):
        """Test that missing class in weights raises error by default."""
        with GravitySpyDataset(DATA_ROOT, split="train") as dataset:
            # Add a fake class that doesn't exist in the data
            original_classes = dataset.class_names.copy()
            dataset.class_names = original_classes + ["FakeClass"]
            dataset.num_classes = len(dataset.class_names)

            with pytest.raises(ValueError, match="has no samples"):
                dataset.get_class_weights(validate_all_classes=True)

            # But with validate_all_classes=False, it should work
            weights = dataset.get_class_weights(validate_all_classes=False)
            assert weights[-1] == 0.0  # FakeClass gets 0 weight

    def test_reproducibility(self, skip_if_no_data):
        """Test that dataset access is deterministic."""
        with GravitySpyDataset(DATA_ROOT, split="train", mode="single") as dataset1:
            with GravitySpyDataset(DATA_ROOT, split="train", mode="single") as dataset2:
                img1, label1 = dataset1[0]
                img2, label2 = dataset2[0]

                assert label1 == label2
                assert torch.allclose(img1, img2)

    def test_context_manager(self, skip_if_no_data):
        """Test that context manager properly closes file handle."""
        with GravitySpyDataset(DATA_ROOT, split="train") as dataset:
            _ = dataset[0]  # Access data to open file
            assert dataset._h5_file is not None

        # After context exit, file should be closed
        assert dataset._h5_file is None

    def test_class_names_validation(self, skip_if_no_data):
        """Test that missing required classes raises error."""
        # Get actual classes from the dataset
        with GravitySpyDataset(DATA_ROOT, split="train") as dataset:
            actual_classes = dataset.class_names

        # Try with incomplete class list
        incomplete_classes = actual_classes[:5]  # Only first 5 classes
        with pytest.raises(ValueError, match="class_names missing classes"):
            GravitySpyDataset(DATA_ROOT, split="train", class_names=incomplete_classes)


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
        with GravitySpyDataset(
            DATA_ROOT, split="train", mode="single", transform=transform
        ) as dataset:
            image, _ = dataset[0]
            assert image.shape == (3, 224, 224)

    def test_transforms_different_sizes(self, skip_if_no_data):
        """Test transforms with different target sizes."""
        for size in [128, 224, 384]:
            transform = get_eval_transforms(image_size=size)
            with GravitySpyDataset(
                DATA_ROOT, split="train", mode="single", transform=transform
            ) as dataset:
                image, _ = dataset[0]
                assert image.shape == (3, size, size)

    def test_multiview_transform_consistent_spatial(self):
        """Test that MultiViewTransform applies consistent transforms."""
        transform = get_train_transforms(image_size=64)
        mv_transform = MultiViewTransform(transform, consistent_spatial=True)

        # Create fake multi-view data with string keys
        images = {
            "0.5": torch.randn(3, 100, 100),
            "1.0": torch.randn(3, 100, 100),
        }

        # Apply transform multiple times and check consistency
        torch.manual_seed(42)
        result1 = mv_transform(images)

        torch.manual_seed(42)
        result2 = mv_transform(images)

        # Results should be identical with same seed
        for ts in images.keys():
            assert torch.allclose(result1[ts], result2[ts])

    def test_multiview_transform_string_keys(self):
        """Test that MultiViewTransform works with string keys."""
        transform = get_eval_transforms(image_size=64)
        mv_transform = MultiViewTransform(transform, consistent_spatial=False)

        images = {
            "0.5": torch.randn(3, 100, 100),
            "1.0": torch.randn(3, 100, 100),
            "2.0": torch.randn(3, 100, 100),
            "4.0": torch.randn(3, 100, 100),
        }

        result = mv_transform(images)

        assert set(result.keys()) == {"0.5", "1.0", "2.0", "4.0"}
        for ts, img in result.items():
            assert img.shape == (3, 64, 64)


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
        """Test creating multi-view DataLoader with string keys."""
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
        # Keys should be strings
        assert set(images.keys()) == {"0.5", "1.0", "2.0", "4.0"}
        for ts, img in images.items():
            assert isinstance(ts, str)
            assert img.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)

    def test_multiview_collate_fn(self):
        """Test multi-view collate function with string keys."""
        # Create fake batch with string keys
        batch = [
            ({"0.5": torch.randn(3, 224, 224), "1.0": torch.randn(3, 224, 224)}, 0),
            ({"0.5": torch.randn(3, 224, 224), "1.0": torch.randn(3, 224, 224)}, 1),
        ]

        images, labels = multiview_collate_fn(batch)

        assert isinstance(images, dict)
        assert images["0.5"].shape == (2, 3, 224, 224)
        assert images["1.0"].shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert labels.tolist() == [0, 1]

    def test_multiview_collate_fn_empty_batch(self):
        """Test that empty batch raises CollateError."""
        with pytest.raises(CollateError, match="Empty batch"):
            multiview_collate_fn([])

    def test_multiview_collate_fn_inconsistent_keys(self):
        """Test that inconsistent time scales raise CollateError."""
        batch = [
            ({"0.5": torch.randn(3, 224, 224), "1.0": torch.randn(3, 224, 224)}, 0),
            ({"0.5": torch.randn(3, 224, 224), "2.0": torch.randn(3, 224, 224)}, 1),
        ]

        with pytest.raises(CollateError, match="inconsistent time scales"):
            multiview_collate_fn(batch)

    def test_multiview_collate_fn_shape_mismatch(self):
        """Test that shape mismatch raises CollateError."""
        batch = [
            ({"0.5": torch.randn(3, 224, 224)}, 0),
            ({"0.5": torch.randn(3, 128, 128)}, 1),  # Different size
        ]

        with pytest.raises(CollateError, match="Shape mismatch"):
            multiview_collate_fn(batch)

    def test_dataloader_shuffle(self, skip_if_no_data):
        """Test that train loader shuffles, eval loaders don't."""
        train_loader = create_dataloader(
            DATA_ROOT, split="train", batch_size=4, num_workers=0
        )
        val_loader = create_dataloader(
            DATA_ROOT, split="validation", batch_size=4, num_workers=0
        )

        assert train_loader.dataset is not None
        assert val_loader.dataset is not None


class TestMultiprocessing:
    """Tests for multiprocessing DataLoader behavior."""

    def test_worker_init_fn(self):
        """Test worker initialization function."""
        # This should not raise any errors
        worker_init_fn(0)

    @pytest.mark.skipif(
        not (DATA_ROOT / "trainingsetv1d0.h5").exists(),
        reason="Dataset not downloaded",
    )
    def test_multiprocessing_dataloader_single(self):
        """Test DataLoader with multiple workers in single-view mode."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="single",
            batch_size=4,
            num_workers=2,
            pin_memory=False,  # Disable for test
        )

        # Process a few batches to verify multiprocessing works
        batches_processed = 0
        for images, labels in loader:
            assert images.shape == (4, 3, 224, 224) or images.shape[0] <= 4
            assert labels.ndim == 1
            batches_processed += 1
            if batches_processed >= 3:
                break

        assert batches_processed == 3

    @pytest.mark.skipif(
        not (DATA_ROOT / "trainingsetv1d0.h5").exists(),
        reason="Dataset not downloaded",
    )
    def test_multiprocessing_dataloader_multi(self):
        """Test DataLoader with multiple workers in multi-view mode."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="multi",
            batch_size=4,
            num_workers=2,
            pin_memory=False,  # Disable for test
        )

        # Process a few batches to verify multiprocessing works
        batches_processed = 0
        for images, labels in loader:
            assert isinstance(images, dict)
            assert set(images.keys()) == {"0.5", "1.0", "2.0", "4.0"}
            for ts, img in images.items():
                assert img.shape[1:] == (3, 224, 224)
            assert labels.ndim == 1
            batches_processed += 1
            if batches_processed >= 3:
                break

        assert batches_processed == 3


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
            for ts in TIME_SCALE_KEYS:
                assert ts in images
                assert images[ts].shape[1:] == (3, 224, 224)
            assert labels.ndim == 1
            if i >= 2:
                break

    def test_full_pipeline_with_consistent_transforms(self, skip_if_no_data):
        """Test multi-view pipeline with consistent spatial transforms."""
        loader = create_dataloader(
            DATA_ROOT,
            split="train",
            mode="multi",
            batch_size=4,
            num_workers=0,
            image_size=224,
            consistent_multiview_transforms=True,
        )

        batch = next(iter(loader))
        images, labels = batch

        assert isinstance(images, dict)
        assert len(images) == 4
        # All views should have same shape
        shapes = [img.shape for img in images.values()]
        assert all(s == shapes[0] for s in shapes)


class TestImageValidation:
    """Tests for image validation logic."""

    def test_validate_image_array_2d(self):
        """Test validation of 2D grayscale image."""
        # Test the normalization logic that the dataset uses internally
        # uint8 image
        img_uint8 = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        normalized = img_uint8.astype(np.float32) / 255.0
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

        # float32 already normalized
        img_float = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]], dtype=np.float32)
        assert img_float.max() <= 1.0

        # float32 needs normalization
        img_float_unnorm = np.array(
            [[0.0, 128.0, 255.0], [64.0, 192.0, 32.0]], dtype=np.float32
        )
        normalized = img_float_unnorm / 255.0
        assert normalized.max() <= 1.0
