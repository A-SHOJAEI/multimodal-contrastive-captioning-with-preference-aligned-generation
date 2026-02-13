"""Training loop with learning rate scheduling and early stopping."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import ContrastivePreferenceLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for multimodal captioning model with two-phase training.

    Phase 1: Contrastive pre-training
    Phase 2: Preference alignment

    Attributes:
        model: The captioning model
        config: Training configuration
        device: Device to train on
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Extract config sections
        self.training_config = config.get("training", {})
        self.loss_config = config.get("loss", {})
        self.output_config = config.get("output", {})

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss function
        self.criterion = ContrastivePreferenceLoss(
            contrastive_temperature=self.loss_config.get("contrastive_temperature", 0.07),
            contrastive_weight=self.loss_config.get("contrastive_weight", 1.0),
            preference_weight=self.loss_config.get("preference_weight", 0.5),
            preference_beta=self.loss_config.get("preference_beta", 0.1),
            preference_margin=self.loss_config.get("preference_margin", 0.1),
            symmetric_contrastive=self.loss_config.get("use_symmetric_contrastive", True),
            adaptive_weighting=self.loss_config.get("adaptive_weighting", True),
            initial_contrastive_ratio=self.loss_config.get("initial_contrastive_ratio", 0.7),
            final_contrastive_ratio=self.loss_config.get("final_contrastive_ratio", 0.3),
        )

        # Initialize scheduler
        self.scheduler = None

        # Mixed precision scaler
        self.scaler = None
        if self.training_config.get("mixed_precision", False):
            self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf")
        self.patience_counter = 0

        # Create output directories
        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for dir_key in ["output_dir", "logging_dir", "results_dir", "checkpoint_dir"]:
            dir_path = Path(self.output_config.get(dir_key, dir_key.replace("_dir", "")))
            dir_path.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.training_config.get("optimizer", "adamw").lower()
        lr = self.training_config.get("phase1_learning_rate", 0.0001)
        weight_decay = self.training_config.get("weight_decay", 0.01)

        if optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(
                    self.training_config.get("adam_beta1", 0.9),
                    self.training_config.get("adam_beta2", 0.999),
                ),
                eps=self.training_config.get("adam_epsilon", 1e-8),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Created optimizer: {optimizer_name} with lr={lr}")
        return optimizer

    def _create_scheduler(
        self,
        total_steps: int,
        scheduler_type: str = "cosine",
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.

        Args:
            total_steps: Total training steps
            scheduler_type: Type of scheduler ("cosine" or "plateau")

        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=0,
            )
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                verbose=True,
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None

        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler

    def train_epoch(
        self,
        train_loader: DataLoader,
        phase: int = 1,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            phase: Training phase (1 or 2)

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_lm_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Phase {phase} - Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Update loss function progress for adaptive weighting
            if self.criterion.adaptive_weighting:
                progress = self.global_step / max(1, len(train_loader) * self.training_config.get("phase1_epochs", 10))
                self.criterion.set_progress(progress)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                    # Compute contrastive loss
                    loss, loss_dict = self.criterion(
                        image_embeds=outputs["image_embeds"],
                        text_embeds=outputs["text_embeds"],
                    )

                    # Add language modeling loss if available
                    if outputs["lm_loss"] is not None:
                        loss = loss + outputs["lm_loss"]
                        total_lm_loss += outputs["lm_loss"].item()
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                loss, loss_dict = self.criterion(
                    image_embeds=outputs["image_embeds"],
                    text_embeds=outputs["text_embeds"],
                )

                if outputs["lm_loss"] is not None:
                    loss = loss + outputs["lm_loss"]
                    total_lm_loss += outputs["lm_loss"].item()

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.training_config.get("max_grad_norm", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_grad_norm"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.training_config.get("max_grad_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_grad_norm"]
                    )
                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            if "contrastive_loss" in loss_dict:
                total_contrastive_loss += loss_dict["contrastive_loss"]
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

        # Compute averages
        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_contrastive_loss": total_contrastive_loss / num_batches,
        }

        if total_lm_loss > 0:
            avg_metrics["train_lm_loss"] = total_lm_loss / num_batches

        return avg_metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss, loss_dict = self.criterion(
                    image_embeds=outputs["image_embeds"],
                    text_embeds=outputs["text_embeds"],
                )

                total_loss += loss.item()
                if "contrastive_loss" in loss_dict:
                    total_contrastive_loss += loss_dict["contrastive_loss"]
                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_contrastive_loss": total_contrastive_loss / num_batches,
        }

    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

        if is_best:
            best_path = Path(filepath).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        phase: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            phase: Training phase (1 or 2)

        Returns:
            Training history
        """
        # Create scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler_type = self.training_config.get("lr_scheduler", "cosine")
        self.scheduler = self._create_scheduler(total_steps, scheduler_type)

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        patience = self.training_config.get("early_stopping_patience", 3)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"Phase {phase} - Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, phase=phase)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log metrics
            for key, value in {**train_metrics, **val_metrics}.items():
                logger.info(f"{key}: {value:.4f}")
                if key in history:
                    history[key].append(value)
                else:
                    history[key] = [value]

            # Check if best model
            current_metric = val_metrics.get("val_loss", float("inf"))
            is_best = current_metric < self.best_metric if self.best_metric != float("-inf") else True

            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            checkpoint_path = Path(self.output_config.get("checkpoint_dir", "checkpoints"))
            checkpoint_path = checkpoint_path / f"phase{phase}_epoch{epoch}.pt"
            self.save_checkpoint(str(checkpoint_path), is_best=is_best)

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Update scheduler if plateau type
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(current_metric)

        return history
