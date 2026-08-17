"""Unit tests for Opacus DP-SGD engine and session epsilon accounting."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client.dp_trainer import DPTrainConfig, DPTrainer
from src.models.network import create_model


def _tiny_loader(n: int = 128, input_dim: int = 784, num_classes: int = 10, batch_size: int = 32):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_dp_training_produces_positive_epsilon() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=1,
        batch_size=32,
        learning_rate=0.05,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_delta=1e-5,
        epsilon_max=50.0,
        target_epsilon=20.0,
        num_rounds=1,
        calibrate_to_target=True,
        poisson_sampling=False,
    )
    trainer = DPTrainer(model, cfg, n_samples=128, device="cpu")
    result = trainer.train(_tiny_loader())
    assert result.epsilon > 0.0
    assert result.epochs_completed == 1
    assert 0.0 <= result.accuracy <= 1.0
    assert result.noise_multiplier > 0.0


def test_epsilon_composes_across_rounds() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=1,
        batch_size=32,
        learning_rate=0.05,
        max_grad_norm=1.0,
        target_delta=1e-5,
        epsilon_max=50.0,
        target_epsilon=20.0,
        num_rounds=2,
        calibrate_to_target=True,
        poisson_sampling=False,
    )
    trainer = DPTrainer(model, cfg, n_samples=128, device="cpu")
    first = trainer.train(_tiny_loader())
    second = trainer.train(_tiny_loader())
    assert second.epsilon > first.epsilon
    assert second.steps_taken > 0


def test_epsilon_cap_stops_before_next_step() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=8,
        batch_size=32,
        learning_rate=0.05,
        noise_multiplier=0.25,
        max_grad_norm=1.0,
        target_delta=1e-5,
        epsilon_max=0.05,
        target_epsilon=5.0,
        num_rounds=1,
        calibrate_to_target=False,
        poisson_sampling=False,
    )
    trainer = DPTrainer(model, cfg, n_samples=256, device="cpu")
    result = trainer.train(_tiny_loader(n=256, batch_size=32))
    assert result.stopped_early
    assert result.reason == "epsilon_cap_exceeded"
    assert result.epochs_completed < cfg.epochs
    skipped = trainer.train(_tiny_loader(n=256, batch_size=32))
    assert skipped.stopped_early
    assert skipped.steps_taken == 0


def test_target_epsilon_calibrates_noise() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=1,
        batch_size=32,
        target_epsilon=4.0,
        epsilon_max=8.0,
        target_delta=1e-5,
        num_rounds=3,
        calibrate_to_target=True,
        poisson_sampling=False,
        noise_multiplier=0.01,
    )
    trainer = DPTrainer(model, cfg, n_samples=128, device="cpu")
    assert trainer.noise_multiplier != 0.01
    assert trainer.noise_multiplier > 0.0


def test_gradient_clipping_configured() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=1,
        batch_size=32,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        epsilon_max=100.0,
        num_rounds=1,
        calibrate_to_target=False,
        poisson_sampling=False,
    )
    trainer = DPTrainer(model, cfg, n_samples=128, device="cpu")
    trainer.train(_tiny_loader())
    assert trainer.privacy_engine is not None
    assert len(trainer.privacy_engine.accountant) > 0
