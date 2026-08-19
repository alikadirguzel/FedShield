"""Opacus DP-SGD with a session-scoped RDP accountant and a pre-step ε cap."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from src.common.logger import setup_logger
from src.models.network import assert_no_batchnorm

logger = setup_logger("fedshield.dp_trainer")


class EpsilonBudgetExceeded(RuntimeError):
    """Raised when local privacy spend would exceed epsilon_max."""

    def __init__(self, epsilon: float, epsilon_max: float) -> None:
        super().__init__(
            f"Privacy budget exceeded: ε={epsilon:.4f} > ε_max={epsilon_max:.4f}"
        )
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max


@dataclass
class DPTrainConfig:
    epochs: int = 2
    learning_rate: float = 0.01
    batch_size: int = 64
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5
    epsilon_max: float = 8.0
    target_epsilon: float = 5.0
    num_rounds: int = 5
    calibrate_to_target: bool = True
    poisson_sampling: bool = True
    dp_enabled: bool = True


@dataclass
class DPTrainResult:
    loss: float
    accuracy: float
    epsilon: float
    delta: float
    epochs_completed: int
    stopped_early: bool
    noise_multiplier: float = 0.0
    steps_taken: int = 0
    reason: Optional[str] = None


def _seeded_cpu_generator() -> torch.Generator:
    """Seed a CPU generator from OS entropy (torchcsprng is unmaintained)."""
    generator = torch.Generator(device="cpu")
    seed = int.from_bytes(os.urandom(8), "little") % (2**63 - 1)
    generator.manual_seed(seed)
    return generator


class DPTrainer:
    """Session-scoped DP trainer: one RDP accountant across federated rounds."""

    def __init__(
        self,
        model: nn.Module,
        config: DPTrainConfig,
        n_samples: int,
        device: Optional[str] = None,
    ) -> None:
        assert_no_batchnorm(model)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            assert_no_batchnorm(model)
        self.model = model.to(self.device)
        self.config = config
        self.n_samples = max(int(n_samples), 1)
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        self._criterion = nn.CrossEntropyLoss()
        self.noise_multiplier = float(config.noise_multiplier)
        self._calibrate()

    def _steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.n_samples / max(int(self.config.batch_size), 1)))

    def _sample_rate(self) -> float:
        # Matches Opacus make_private: sample_rate = 1 / len(DataLoader)
        return 1.0 / float(self._steps_per_epoch())

    def _total_steps(self) -> int:
        return max(1, self._steps_per_epoch() * int(self.config.epochs) * int(self.config.num_rounds))

    def _calibrate(self) -> None:
        if not self.config.dp_enabled:
            self.noise_multiplier = 0.0
            logger.info("DP-SGD off — data isolation + SecAgg only (no gradient noise)")
            return
        if not self.config.calibrate_to_target:
            self.noise_multiplier = float(self.config.noise_multiplier)
            logger.info(
                "Using configured noise_multiplier (calibration disabled)",
                extra={"noise_multiplier": self.noise_multiplier},
            )
            return
        sample_rate = self._sample_rate()
        steps = self._total_steps()
        self.noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=float(self.config.target_epsilon),
                target_delta=float(self.config.target_delta),
                sample_rate=sample_rate,
                steps=steps,
                accountant="rdp",
            )
        )
        logger.info(
            "Calibrated DP noise to session privacy budget",
            extra={
                "target_epsilon": self.config.target_epsilon,
                "epsilon_max": self.config.epsilon_max,
                "delta": self.config.target_delta,
                "sample_rate": sample_rate,
                "session_steps": steps,
                "noise_multiplier": self.noise_multiplier,
            },
        )

    def recalibrate_if_idle(self, num_rounds: int) -> None:
        """Adopt server round count before any accountant steps have been recorded."""
        if not self.config.dp_enabled:
            return
        rounds = max(int(num_rounds), 1)
        if rounds == self.config.num_rounds:
            return
        if len(self.privacy_engine.accountant) > 0:
            logger.warning(
                "Ignoring num_rounds change; privacy accountant already has history",
                extra={"configured": self.config.num_rounds, "requested": rounds},
            )
            return
        self.config.num_rounds = rounds
        self._calibrate()

    def get_epsilon(self) -> float:
        if not self.config.dp_enabled or len(self.privacy_engine.accountant) == 0:
            return 0.0
        return float(self.privacy_engine.get_epsilon(self.config.target_delta))

    def get_underlying_model(self) -> nn.Module:
        model = self.model
        if hasattr(model, "_module"):
            return model._module  # type: ignore[no-any-return]
        return model

    def train(self, train_loader: DataLoader) -> DPTrainResult:
        """Run local training; DP-SGD is optional (off in data_isolation mode)."""
        if not self.config.dp_enabled:
            return self._train_plain(train_loader)
        spent = self.get_epsilon()
        if spent >= self.config.epsilon_max:
            logger.warning(
                "Skipping local training — privacy cap already reached",
                extra={"epsilon": spent, "epsilon_max": self.config.epsilon_max},
            )
            return DPTrainResult(
                loss=0.0,
                accuracy=0.0,
                epsilon=spent,
                delta=self.config.target_delta,
                epochs_completed=0,
                stopped_early=True,
                noise_multiplier=self.noise_multiplier,
                steps_taken=0,
                reason="epsilon_cap_exceeded",
            )

        work = copy.deepcopy(self.model).to(self.device)
        work.train()
        optimizer = torch.optim.SGD(work.parameters(), lr=self.config.learning_rate)
        work, optimizer, private_loader = self.privacy_engine.make_private(
            module=work,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
            noise_generator=_seeded_cpu_generator(),
            poisson_sampling=self.config.poisson_sampling,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epochs_done = 0
        steps_taken = 0
        stopped_early = False
        reason: Optional[str] = None

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for features, labels in private_loader:
                if self.get_epsilon() >= self.config.epsilon_max:
                    stopped_early = True
                    reason = "epsilon_cap_exceeded"
                    break
                features = features.to(self.device)
                labels = labels.to(self.device)
                if labels.numel() == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = work(features)
                loss = self._criterion(logits, labels)
                loss.backward()
                optimizer.step()
                steps_taken += 1

                batch_size = int(labels.size(0))
                epoch_loss += loss.item() * batch_size
                epoch_correct += int((logits.argmax(dim=1) == labels).sum().item())
                epoch_samples += batch_size

            epochs_done = epoch + 1
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            epsilon = self.get_epsilon()
            logger.info(
                "Local DP epoch complete",
                extra={
                    "epoch": epochs_done,
                    "epsilon": epsilon,
                    "loss": epoch_loss / max(epoch_samples, 1),
                    "stopped_early": stopped_early,
                },
            )
            if stopped_early:
                break

        trained = work._module if hasattr(work, "_module") else work
        self.model.load_state_dict(trained.state_dict())

        epsilon = self.get_epsilon()
        return DPTrainResult(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            epsilon=epsilon,
            delta=self.config.target_delta,
            epochs_completed=epochs_done,
            stopped_early=stopped_early,
            noise_multiplier=self.noise_multiplier,
            steps_taken=steps_taken,
            reason=reason,
        )

    def _train_plain(self, train_loader: DataLoader) -> DPTrainResult:
        work = copy.deepcopy(self.model).to(self.device)
        work.train()
        optimizer = torch.optim.SGD(work.parameters(), lr=self.config.learning_rate)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        steps_taken = 0
        epochs_done = 0
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                if labels.numel() == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = work(features)
                loss = self._criterion(logits, labels)
                loss.backward()
                optimizer.step()
                steps_taken += 1
                batch_size = int(labels.size(0))
                epoch_loss += loss.item() * batch_size
                epoch_correct += int((logits.argmax(dim=1) == labels).sum().item())
                epoch_samples += batch_size
            epochs_done = epoch + 1
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        self.model.load_state_dict(work.state_dict())
        return DPTrainResult(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            epsilon=0.0,
            delta=self.config.target_delta,
            epochs_completed=epochs_done,
            stopped_early=False,
            noise_multiplier=0.0,
            steps_taken=steps_taken,
        )

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Non-private evaluation on local hold-out data (metrics only)."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(features)
                loss = self._criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }
