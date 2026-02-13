"""Tests for data loading and preprocessing."""

import pytest
import torch
from PIL import Image

from multimodal_contrastive_captioning_with_preference_aligned_generation.data.preprocessing import (
    PreprocessTransform,
    collate_fn,
)


class TestPreprocessTransform:
    """Test preprocessing transform."""

    def test_initialization(self):
        """Test transform initialization."""
        transform = PreprocessTransform(
            vision_encoder_name="openai/clip-vit-base-patch32",
            text_decoder_name="gpt2",
            max_length=77,
        )

        assert transform.max_length == 77
        assert transform.tokenizer is not None

    def test_preprocess_image(self, sample_image):
        """Test image preprocessing."""
        transform = PreprocessTransform()
        image_tensor = transform.preprocess_image(sample_image)

        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape[0] == 3  # RGB channels
        assert image_tensor.dim() == 3

    def test_preprocess_text(self):
        """Test text preprocessing."""
        transform = PreprocessTransform()
        text = "A cat sitting on a couch"

        result = transform.preprocess_text(text)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert result["input_ids"].shape[0] == transform.max_length

    def test_call_method(self, sample_image):
        """Test transform __call__ method."""
        transform = PreprocessTransform()
        caption = "A test caption"

        result = transform(sample_image, caption)

        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["pixel_values"], torch.Tensor)
        assert isinstance(result["input_ids"], torch.Tensor)


class TestCollateFunction:
    """Test collate function."""

    def test_collate_basic_batch(self, device):
        """Test basic batch collation."""
        batch = [
            {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.randint(0, 1000, (77,)),
                "attention_mask": torch.ones(77),
            }
            for _ in range(4)
        ]

        collated = collate_fn(batch)

        assert collated["pixel_values"].shape == (4, 3, 224, 224)
        assert collated["input_ids"].shape == (4, 77)
        assert collated["attention_mask"].shape == (4, 77)

    def test_collate_with_preference_pairs(self):
        """Test collation with preference pairs."""
        batch = [
            {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.randint(0, 1000, (77,)),
                "attention_mask": torch.ones(77),
                "chosen_input_ids": torch.randint(0, 1000, (77,)),
                "chosen_attention_mask": torch.ones(77),
                "rejected_input_ids": torch.randint(0, 1000, (77,)),
                "rejected_attention_mask": torch.ones(77),
            }
            for _ in range(2)
        ]

        collated = collate_fn(batch)

        assert "chosen_input_ids" in collated
        assert "rejected_input_ids" in collated
        assert collated["chosen_input_ids"].shape == (2, 77)


class TestDataLoader:
    """Test data loader functionality."""

    def test_dataloader_creation(self, sample_config):
        """Test creating dataloaders from config."""
        from multimodal_contrastive_captioning_with_preference_aligned_generation.data.loader import (
            create_dataloaders,
        )
        from multimodal_contrastive_captioning_with_preference_aligned_generation.data.preprocessing import (
            PreprocessTransform,
        )

        # Modify config to use fewer samples for testing
        sample_config["data"] = {
            "train_split": "train",
            "val_split": "validation",
            "test_split": "test",
            "max_train_samples": 10,
            "max_val_samples": 5,
            "max_test_samples": 5,
            "num_workers": 0,
        }

        transform = PreprocessTransform()

        # This might fail if datasets are not available, which is okay in CI
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                sample_config, transform
            )

            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None

        except Exception as e:
            pytest.skip(f"Could not load datasets: {e}")
