"""Tests for Trainer class."""

import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, load_checkpoint


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def dummy_dataloader():
    """Create dummy dataloader for testing."""
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def trainer_config():
    """Create minimal trainer config."""
    return OmegaConf.create(
        {
            "model": {
                "name": "test_model",
                "type": "cnn",
            },
            "data": {
                "batch_size": 8,
            },
            "train": {
                "epochs": 2,
                "device": "cpu",
                "mixed_precision": False,
                "grad_clip": 1.0,
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "name": "cosine",
                    "warmup_epochs": 1,
                    "min_lr": 1e-6,
                },
                "loss": {
                    "name": "cross_entropy",
                    "label_smoothing": 0.0,
                },
                "early_stopping": {
                    "enabled": True,
                    "patience": 5,
                    "metric": "val_accuracy",
                    "mode": "max",
                },
                "checkpoint": {
                    "save_best": True,
                    "save_last": True,
                    "metric": "val_accuracy",
                    "mode": "max",
                },
            },
            "mlflow": {
                "experiment_name": "test",
                "tracking_uri": None,
                "run_name": "test_run",
                "tags": {},
            },
            "hydra": {
                "run": {
                    "dir": "outputs/test",
                }
            },
        }
    )


class TestTrainerInit:
    def test_trainer_initialization(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Should initialize trainer correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            assert trainer.device == torch.device("cpu")
            assert trainer.use_amp is False
            assert trainer.grad_clip == 1.0
            assert trainer.patience == 5

    def test_model_moved_to_device(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Model should be moved to specified device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            param = next(trainer.model.parameters())
            assert param.device == torch.device("cpu")


class TestTrainerTraining:
    def test_train_epoch_decreases_loss(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Loss should generally decrease during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            metrics1 = trainer._train_epoch()
            assert "loss" in metrics1
            assert "accuracy" in metrics1

    def test_validate_epoch(self, simple_model, dummy_dataloader, trainer_config):
        """Validation should compute metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            metrics = trainer._validate_epoch()

            assert "loss" in metrics
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert "balanced_accuracy" in metrics

    def test_full_training_loop(self, simple_model, dummy_dataloader, trainer_config):
        """Should complete full training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.epochs = 2

            with (
                patch("mlflow.set_experiment"),
                patch("mlflow.start_run"),
                patch("mlflow.log_params"),
                patch("mlflow.log_metric"),
                patch("mlflow.log_artifact"),
                patch("mlflow.set_tag"),
                patch("mlflow.end_run"),
            ):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

                results = trainer.train()

            assert "final_epoch" in results
            assert results["final_epoch"] == 2


class TestEarlyStopping:
    def test_early_stopping_triggers(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Early stopping should trigger after patience exhausted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.early_stopping.patience = 3

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            # Simulate decreasing accuracy (no improvement)
            trainer.best_metric = 0.9
            assert not trainer._check_early_stopping({"accuracy": 0.85})  # counter=1
            assert not trainer._check_early_stopping({"accuracy": 0.84})  # counter=2
            assert trainer._check_early_stopping(
                {"accuracy": 0.83}
            )  # counter=3, triggers

    def test_early_stopping_resets_on_improvement(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Patience counter should reset on improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            trainer.best_metric = 0.8
            trainer._check_early_stopping({"accuracy": 0.75})  # No improvement
            assert trainer.patience_counter == 1

            trainer._check_early_stopping({"accuracy": 0.85})  # Improvement
            assert trainer.patience_counter == 0


class TestCheckpointing:
    def test_save_checkpoint(self, simple_model, dummy_dataloader, trainer_config):
        """Should save checkpoint correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            trainer.current_epoch = 5
            trainer.global_step = 100
            path = trainer._save_checkpoint({"accuracy": 0.9}, is_best=True)

            assert path.exists()
            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["epoch"] == 5
            assert checkpoint["global_step"] == 100
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self, simple_model, dummy_dataloader, trainer_config):
        """Should load checkpoint correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            path = trainer._save_checkpoint({"accuracy": 0.9}, is_best=True)

            # Create new model and load checkpoint
            new_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 32 * 32, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

            checkpoint = load_checkpoint(path, new_model)

            assert checkpoint["epoch"] == 0
            assert "model_state_dict" in checkpoint


class TestBestModelTracking:
    def test_check_best_model_max_mode(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Should track best model in max mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.checkpoint.mode = "max"

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            assert trainer._check_best_model({"accuracy": 0.7})
            assert trainer._check_best_model({"accuracy": 0.8})
            assert not trainer._check_best_model({"accuracy": 0.75})
            assert trainer._check_best_model({"accuracy": 0.9})

    def test_check_best_model_min_mode(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Should track best model in min mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.checkpoint.mode = "min"
            trainer_config.train.checkpoint.metric = "val_loss"

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            assert trainer._check_best_model({"loss": 1.0})
            assert trainer._check_best_model({"loss": 0.5})
            assert not trainer._check_best_model({"loss": 0.7})
            assert trainer._check_best_model({"loss": 0.3})


class TestTestMethod:
    def test_test_method_runs(self, simple_model, dummy_dataloader, trainer_config):
        """Test method should evaluate on test set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    test_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            metrics = trainer.test()

            assert "loss" in metrics
            assert "accuracy" in metrics
            assert "balanced_accuracy" in metrics

    def test_test_method_raises_without_loader(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Test method should raise if no test_loader provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir

            with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                trainer = Trainer(
                    model=simple_model,
                    train_loader=dummy_dataloader,
                    val_loader=dummy_dataloader,
                    cfg=trainer_config,
                )

            with pytest.raises(ValueError, match="test_loader was not provided"):
                trainer.test()


class TestSchedulerValidation:
    def test_warmup_epochs_validation(
        self, simple_model, dummy_dataloader, trainer_config
    ):
        """Should raise if warmup_epochs >= total epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.epochs = 5
            trainer_config.train.scheduler.warmup_epochs = 5

            with pytest.raises(
                ValueError, match="warmup_epochs.*must be < total epochs"
            ):
                with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                    Trainer(
                        model=simple_model,
                        train_loader=dummy_dataloader,
                        val_loader=dummy_dataloader,
                        cfg=trainer_config,
                    )


class TestAMPWarning:
    def test_amp_warning_on_cpu(
        self, simple_model, dummy_dataloader, trainer_config, caplog
    ):
        """Should warn when mixed precision requested on CPU."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config.hydra.run.dir = tmpdir
            trainer_config.train.mixed_precision = True
            trainer_config.train.device = "cpu"

            with caplog.at_level(logging.WARNING):
                with patch("mlflow.set_experiment"), patch("mlflow.start_run"):
                    trainer = Trainer(
                        model=simple_model,
                        train_loader=dummy_dataloader,
                        val_loader=dummy_dataloader,
                        cfg=trainer_config,
                    )

            assert trainer.use_amp is False
            assert "Mixed precision requested but device is cpu" in caplog.text
