"""Tests for model architecture and components."""

import pytest
import torch

from multimodal_contrastive_captioning_with_preference_aligned_generation.models.components import (
    ContrastivePreferenceLoss,
    ProjectionHead,
    compute_contrastive_loss,
    compute_preference_loss,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import (
    MultimodalCaptioningModel,
)


class TestProjectionHead:
    """Test projection head component."""

    def test_initialization(self):
        """Test projection head initialization."""
        proj_head = ProjectionHead(input_dim=768, output_dim=512)

        assert proj_head is not None

    def test_forward_pass(self, device):
        """Test forward pass through projection head."""
        proj_head = ProjectionHead(input_dim=768, output_dim=512).to(device)

        batch_size = 4
        input_features = torch.randn(batch_size, 768).to(device)

        output = proj_head(input_features)

        assert output.shape == (batch_size, 512)


class TestContrastiveLoss:
    """Test contrastive loss computation."""

    def test_compute_contrastive_loss(self, device):
        """Test contrastive loss calculation."""
        batch_size = 8
        embed_dim = 512

        image_embeds = torch.randn(batch_size, embed_dim).to(device)
        text_embeds = torch.randn(batch_size, embed_dim).to(device)

        loss = compute_contrastive_loss(
            image_embeds, text_embeds, temperature=0.07, symmetric=True
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_symmetric_vs_asymmetric(self, device):
        """Test symmetric vs asymmetric contrastive loss."""
        batch_size = 4
        embed_dim = 256

        image_embeds = torch.randn(batch_size, embed_dim).to(device)
        text_embeds = torch.randn(batch_size, embed_dim).to(device)

        loss_symmetric = compute_contrastive_loss(
            image_embeds, text_embeds, symmetric=True
        )
        loss_asymmetric = compute_contrastive_loss(
            image_embeds, text_embeds, symmetric=False
        )

        # Both should be valid scalars
        assert loss_symmetric.dim() == 0
        assert loss_asymmetric.dim() == 0


class TestPreferenceLoss:
    """Test preference loss computation."""

    def test_compute_preference_loss(self, device):
        """Test preference loss calculation."""
        batch_size = 4

        chosen_logits = torch.randn(batch_size).to(device)
        rejected_logits = torch.randn(batch_size).to(device)

        loss = compute_preference_loss(
            chosen_logits, rejected_logits, beta=0.1, margin=0.0
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_preference_loss_with_margin(self, device):
        """Test preference loss with margin."""
        batch_size = 4

        # Chosen should be better than rejected
        chosen_logits = torch.ones(batch_size).to(device)
        rejected_logits = torch.zeros(batch_size).to(device)

        loss = compute_preference_loss(
            chosen_logits, rejected_logits, beta=0.1, margin=0.1
        )

        assert loss.item() >= 0


class TestContrastivePreferenceLoss:
    """Test joint loss function."""

    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = ContrastivePreferenceLoss(
            contrastive_temperature=0.07,
            contrastive_weight=1.0,
            preference_weight=0.5,
        )

        assert loss_fn is not None
        assert loss_fn.contrastive_temperature == 0.07

    def test_forward_contrastive_only(self, device):
        """Test loss with contrastive only."""
        loss_fn = ContrastivePreferenceLoss()

        batch_size = 4
        embed_dim = 512

        image_embeds = torch.randn(batch_size, embed_dim).to(device)
        text_embeds = torch.randn(batch_size, embed_dim).to(device)

        total_loss, loss_dict = loss_fn(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        assert isinstance(total_loss, torch.Tensor)
        assert "contrastive_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_forward_with_preference(self, device):
        """Test loss with both contrastive and preference."""
        loss_fn = ContrastivePreferenceLoss()

        batch_size = 4
        embed_dim = 512

        image_embeds = torch.randn(batch_size, embed_dim).to(device)
        text_embeds = torch.randn(batch_size, embed_dim).to(device)
        chosen_logits = torch.randn(batch_size).to(device)
        rejected_logits = torch.randn(batch_size).to(device)

        total_loss, loss_dict = loss_fn(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
        )

        assert "contrastive_loss" in loss_dict
        assert "preference_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_adaptive_weighting(self, device):
        """Test adaptive loss weighting."""
        loss_fn = ContrastivePreferenceLoss(adaptive_weighting=True)

        # Set progress
        loss_fn.set_progress(0.0)
        w1_c, w1_p = loss_fn.get_adaptive_weights()

        loss_fn.set_progress(1.0)
        w2_c, w2_p = loss_fn.get_adaptive_weights()

        # Contrastive weight should decrease over time
        assert w1_c >= w2_c


class TestMultimodalCaptioningModel:
    """Test the main captioning model."""

    @pytest.fixture
    def model(self, device):
        """Create model instance for testing."""
        return MultimodalCaptioningModel(
            vision_encoder_name="openai/clip-vit-base-patch32",
            text_decoder_name="gpt2",
            projection_dim=512,
        ).to(device)

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.vision_encoder is not None
        assert model.text_decoder is not None
        assert model.projection_dim == 512

    def test_encode_image(self, model, sample_batch):
        """Test image encoding."""
        pixel_values = sample_batch["pixel_values"]

        image_features = model.encode_image(pixel_values)

        assert isinstance(image_features, torch.Tensor)
        assert image_features.shape[0] == pixel_values.shape[0]

    def test_encode_text(self, model, sample_batch):
        """Test text encoding."""
        input_ids = sample_batch["input_ids"]
        attention_mask = sample_batch["attention_mask"]

        text_features = model.encode_text(input_ids, attention_mask)

        assert isinstance(text_features, torch.Tensor)
        assert text_features.shape[0] == input_ids.shape[0]

    def test_get_contrastive_embeddings(self, model, sample_batch):
        """Test contrastive embedding extraction."""
        pixel_values = sample_batch["pixel_values"]
        input_ids = sample_batch["input_ids"]
        attention_mask = sample_batch["attention_mask"]

        image_embeds, text_embeds = model.get_contrastive_embeddings(
            pixel_values, input_ids, attention_mask
        )

        assert image_embeds.shape == (pixel_values.shape[0], model.projection_dim)
        assert text_embeds.shape == (input_ids.shape[0], model.projection_dim)

    def test_forward_pass(self, model, sample_batch):
        """Test forward pass."""
        outputs = model(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
        )

        assert "image_embeds" in outputs
        assert "text_embeds" in outputs
        assert isinstance(outputs["image_embeds"], torch.Tensor)

    def test_generate_caption(self, model, sample_batch, device):
        """Test caption generation."""
        pixel_values = sample_batch["pixel_values"]

        generated_ids = model.generate_caption(
            pixel_values, max_length=20, num_beams=2
        )

        assert isinstance(generated_ids, torch.Tensor)
        assert generated_ids.shape[0] == pixel_values.shape[0]

    def test_model_parameters(self, model):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0

    def test_freeze_vision_encoder(self, device):
        """Test freezing vision encoder."""
        model = MultimodalCaptioningModel(
            vision_encoder_name="openai/clip-vit-base-patch32",
            text_decoder_name="gpt2",
            projection_dim=512,
            freeze_vision_encoder=True,
        ).to(device)

        # Check that vision encoder parameters are frozen
        vision_params_frozen = all(
            not p.requires_grad for p in model.vision_encoder.parameters()
        )

        assert vision_params_frozen
