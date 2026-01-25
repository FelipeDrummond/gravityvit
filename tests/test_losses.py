"""Tests for loss functions."""

import pytest
import torch

from src.training.losses import FocalLoss, LabelSmoothingCrossEntropy, create_loss_fn


class TestLabelSmoothingCrossEntropy:
    def test_no_smoothing_matches_ce(self):
        """Without smoothing, should match standard CrossEntropy."""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.0)
        ce_fn = torch.nn.CrossEntropyLoss()

        pred = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))

        loss = loss_fn(pred, target)
        ce_loss = ce_fn(pred, target)

        assert torch.allclose(loss, ce_loss, atol=1e-5)

    def test_smoothing_reduces_confidence(self):
        """Smoothing should make loss higher for perfect predictions."""
        no_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)
        with_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Create "confident" predictions
        pred = torch.zeros(4, 10)
        pred[range(4), range(4)] = 10.0
        target = torch.arange(4)

        loss_no_smooth = no_smooth(pred, target)
        loss_smooth = with_smooth(pred, target)

        assert loss_smooth > loss_no_smooth

    def test_with_class_weights(self):
        """Should apply class weights correctly."""
        weights = torch.ones(10)
        weights[0] = 2.0  # Double weight for class 0

        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.0, weight=weights)

        pred = torch.randn(8, 10)
        target = torch.zeros(8, dtype=torch.long)

        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestFocalLoss:
    def test_gamma_zero_matches_ce(self):
        """With gamma=0, focal loss should match CE."""
        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()

        pred = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))

        focal_loss = focal(pred, target)
        ce_loss = ce(pred, target)

        assert torch.allclose(focal_loss, ce_loss, atol=1e-5)

    def test_high_gamma_reduces_easy_example_weight(self):
        """Higher gamma should reduce loss for easy examples."""
        focal_low = FocalLoss(gamma=1.0)
        focal_high = FocalLoss(gamma=3.0)

        # Create confident correct predictions
        pred = torch.zeros(4, 10)
        pred[range(4), range(4)] = 10.0
        target = torch.arange(4)

        loss_low = focal_low(pred, target)
        loss_high = focal_high(pred, target)

        assert loss_high < loss_low

    def test_with_alpha(self):
        """Should apply class weights (alpha) correctly."""
        alpha = torch.ones(10)
        focal = FocalLoss(gamma=2.0, alpha=alpha)

        pred = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))

        loss = focal(pred, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestCreateLossFn:
    def test_create_ce_without_smoothing(self):
        """Should create standard CE loss."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"train": {"loss": {"name": "cross_entropy", "label_smoothing": 0.0}}}
        )

        loss_fn = create_loss_fn(cfg)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_create_ce_with_smoothing(self):
        """Should create label smoothing CE."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"train": {"loss": {"name": "cross_entropy", "label_smoothing": 0.1}}}
        )

        loss_fn = create_loss_fn(cfg)
        assert isinstance(loss_fn, LabelSmoothingCrossEntropy)

    def test_create_focal_loss(self):
        """Should create focal loss."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"train": {"loss": {"name": "focal_loss", "focal_gamma": 2.0}}}
        )

        loss_fn = create_loss_fn(cfg)
        assert isinstance(loss_fn, FocalLoss)

    def test_unknown_loss_raises(self):
        """Should raise for unknown loss."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"train": {"loss": {"name": "unknown"}}})

        with pytest.raises(ValueError, match="Unknown loss"):
            create_loss_fn(cfg)
