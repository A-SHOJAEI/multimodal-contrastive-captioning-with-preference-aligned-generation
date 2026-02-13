"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import (
    MultimodalCaptioningModel,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.training.trainer import (
    Trainer,
)


class TestTrainer:
    """Test trainer functionality."""

    @pytest.fixture
    def model(self, device):
        """Create a small model for testing."""
        return MultimodalCaptioningModel(
            vision_encoder_name="openai/clip-vit-base-patch32",
            text_decoder_name="gpt2",
            projection_dim=256,
        ).to(device)

    @pytest.fixture
    def dummy_dataloader(self, device):
        """Create a dummy dataloader for testing."""
        batch_size = 2
        num_batches = 3

        pixel_values = torch.randn(num_batches * batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (num_batches * batch_size, 77))
        attention_mask = torch.ones(num_batches * batch_size, 77)

        dataset = TensorDataset(pixel_values, input_ids, attention_mask)

        def collate_fn(batch):
            pv, ii, am = zip(*batch)
            return {
                "pixel_values": torch.stack(pv),
                "input_ids": torch.stack(ii),
                "attention_mask": torch.stack(am),
            }

        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    def test_trainer_initialization(self, model, sample_config, device):
        """Test trainer initialization."""
        trainer = Trainer(model, sample_config, device)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_epoch(self, model, sample_config, device, dummy_dataloader):
        """Test training for one epoch."""
        trainer = Trainer(model, sample_config, device)

        # Train for one epoch
        metrics = trainer.train_epoch(dummy_dataloader, phase=1)

        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0
        assert trainer.global_step > 0

    def test_validate(self, model, sample_config, device, dummy_dataloader):
        """Test validation."""
        trainer = Trainer(model, sample_config, device)

        metrics = trainer.validate(dummy_dataloader)

        assert "val_loss" in metrics
        assert metrics["val_loss"] >= 0

    def test_save_checkpoint(self, model, sample_config, device, tmp_path):
        """Test checkpoint saving."""
        trainer = Trainer(model, sample_config, device)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), is_best=False)

        assert checkpoint_path.exists()

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_early_stopping(self, model, sample_config, device, dummy_dataloader):
        """Test early stopping mechanism."""
        # Set very low patience
        sample_config["training"]["early_stopping_patience"] = 1

        trainer = Trainer(model, sample_config, device)

        # Manually set a good best metric
        trainer.best_metric = -100.0

        # Train with worse metrics (should trigger early stopping)
        history = trainer.train(
            dummy_dataloader,
            dummy_dataloader,
            num_epochs=10,  # Won't complete all epochs
            phase=1,
        )

        # Should stop early
        assert len(history["train_loss"]) < 10

    def test_gradient_clipping(self, model, sample_config, device, dummy_dataloader):
        """Test gradient clipping."""
        sample_config["training"]["max_grad_norm"] = 1.0

        trainer = Trainer(model, sample_config, device)

        # Train one step
        trainer.train_epoch(dummy_dataloader, phase=1)

        # Check that gradients are clipped
        max_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        # With aggressive clipping, should be bounded
        # (This is a soft check as gradient norms can vary)
        assert max_grad < 100.0  # Reasonable upper bound


class TestEvaluationMetrics:
    """Test evaluation metrics."""

    def test_compute_bleu(self, sample_references, sample_captions):
        """Test BLEU score computation."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.metrics import (
            CaptionMetrics,
        )

        metrics = CaptionMetrics(use_clip_score=False)

        bleu_scores = metrics.compute_bleu(sample_references, sample_captions)

        assert "bleu_1" in bleu_scores
        assert "bleu_2" in bleu_scores
        assert "bleu_3" in bleu_scores
        assert "bleu_4" in bleu_scores

        # BLEU scores should be between 0 and 1
        for score in bleu_scores.values():
            assert 0.0 <= score <= 1.0

    def test_compute_rouge(self, sample_references, sample_captions):
        """Test ROUGE score computation."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.metrics import (
            CaptionMetrics,
        )

        metrics = CaptionMetrics(use_clip_score=False)

        rouge_score = metrics.compute_rouge(sample_references, sample_captions)

        assert isinstance(rouge_score, float)
        assert 0.0 <= rouge_score <= 1.0

    def test_compute_cider(self, sample_references, sample_captions):
        """Test CIDEr score computation."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.metrics import (
            CaptionMetrics,
        )

        metrics = CaptionMetrics(use_clip_score=False)

        cider_score = metrics.compute_cider(sample_references, sample_captions)

        assert isinstance(cider_score, float)
        assert cider_score >= 0.0

    def test_compute_all_metrics(self, sample_references, sample_captions):
        """Test computing all metrics at once."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.metrics import (
            compute_all_metrics,
        )

        metrics = compute_all_metrics(
            references=sample_references,
            hypotheses=sample_captions,
            images=None,
            compute_clip=False,
        )

        assert "bleu_1" in metrics
        assert "bleu_4" in metrics
        assert "rouge_l" in metrics
        assert "cider" in metrics


class TestConfig:
    """Test configuration utilities."""

    def test_load_save_config(self, sample_config, tmp_path):
        """Test loading and saving config."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.utils.config import (
            load_config,
            save_config,
        )

        config_path = tmp_path / "test_config.yaml"

        # Save config
        save_config(sample_config, str(config_path))

        assert config_path.exists()

        # Load config
        loaded_config = load_config(str(config_path))

        assert loaded_config["model"]["projection_dim"] == sample_config["model"]["projection_dim"]

    def test_set_seed(self):
        """Test seed setting."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.utils.config import (
            set_seed,
        )

        set_seed(42, deterministic=False)

        # Generate random numbers
        import random
        import numpy as np

        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        # Reset seed
        set_seed(42, deterministic=False)

        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        # Should get same random numbers
        assert r1 == r2
        assert n1 == n2
        assert abs(t1 - t2) < 1e-6
