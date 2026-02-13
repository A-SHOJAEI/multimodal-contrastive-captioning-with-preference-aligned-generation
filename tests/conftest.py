"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "vision_encoder": "openai/clip-vit-base-patch32",
            "text_decoder": "gpt2",
            "projection_dim": 512,
            "freeze_vision_encoder": False,
            "freeze_text_decoder": False,
            "max_caption_length": 77,
        },
        "training": {
            "phase1_epochs": 2,
            "phase2_epochs": 1,
            "phase1_batch_size": 4,
            "phase2_batch_size": 4,
            "phase1_learning_rate": 0.0001,
            "phase2_learning_rate": 0.00003,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "early_stopping_patience": 2,
            "seed": 42,
            "mixed_precision": False,
        },
        "loss": {
            "contrastive_temperature": 0.07,
            "contrastive_weight": 1.0,
            "preference_weight": 0.5,
            "preference_beta": 0.1,
            "preference_margin": 0.1,
            "use_symmetric_contrastive": True,
            "adaptive_weighting": True,
            "initial_contrastive_ratio": 0.7,
            "final_contrastive_ratio": 0.3,
        },
        "output": {
            "output_dir": "test_models",
            "logging_dir": "test_logs",
            "results_dir": "test_results",
            "checkpoint_dir": "test_checkpoints",
        },
    }


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing."""
    batch_size = 4
    return {
        "pixel_values": torch.randn(batch_size, 3, 224, 224).to(device),
        "input_ids": torch.randint(0, 1000, (batch_size, 77)).to(device),
        "attention_mask": torch.ones(batch_size, 77).to(device),
    }


@pytest.fixture
def sample_captions():
    """Sample captions for testing."""
    return [
        "A cat sitting on a couch",
        "A beautiful sunset over the ocean",
        "A person riding a bicycle",
        "A plate of delicious food",
    ]


@pytest.fixture
def sample_references():
    """Sample reference captions for metric testing."""
    return [
        ["A cat sitting on a couch", "A feline resting on a sofa"],
        ["A beautiful sunset over the ocean", "Sunset at the beach"],
        ["A person riding a bicycle", "Someone cycling on a bike"],
        ["A plate of delicious food", "Tasty meal on a dish"],
    ]
