"""Data preprocessing utilities for images and text."""

import logging
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, GPT2Tokenizer

logger = logging.getLogger(__name__)


class PreprocessTransform:
    """
    Preprocessing pipeline for images and captions.

    Attributes:
        clip_processor: CLIP processor for image preprocessing
        tokenizer: GPT2 tokenizer for text preprocessing
        max_length: Maximum caption length
    """

    def __init__(
        self,
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        text_decoder_name: str = "gpt2",
        max_length: int = 77,
    ):
        """
        Initialize preprocessing transform.

        Args:
            vision_encoder_name: Name of the CLIP vision encoder
            text_decoder_name: Name of the text decoder (for tokenization)
            max_length: Maximum caption length
        """
        self.max_length = max_length

        try:
            self.clip_processor = CLIPProcessor.from_pretrained(vision_encoder_name)
        except Exception as e:
            logger.warning(f"Could not load CLIP processor: {e}. Using default transforms.")
            self.clip_processor = None
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(text_decoder_name)
            # Add special tokens if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image tensor
        """
        if self.clip_processor is not None:
            # Use CLIP processor
            inputs = self.clip_processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)
        else:
            # Use manual transform
            return self.image_transform(image)

    def preprocess_text(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text caption.

        Args:
            text: Input text caption
            add_special_tokens: Whether to add special tokens
            return_attention_mask: Whether to return attention mask

        Returns:
            Dictionary with input_ids and optionally attention_mask
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": encoded["input_ids"].squeeze(0)}
        if return_attention_mask:
            result["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return result

    def __call__(self, image: Image.Image, caption: str) -> Dict[str, torch.Tensor]:
        """
        Apply preprocessing to image and caption.

        Args:
            image: PIL Image
            caption: Text caption

        Returns:
            Dictionary with preprocessed image and text
        """
        image_tensor = self.preprocess_image(image)
        text_dict = self.preprocess_text(caption)

        return {
            "pixel_values": image_tensor,
            "input_ids": text_dict["input_ids"],
            "attention_mask": text_dict["attention_mask"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors dictionary
    """
    # Stack pixel values
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Stack text tokens
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    result = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Include labels if present (for preference learning)
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])

    # Include preference pairs if present
    if "chosen_input_ids" in batch[0]:
        result["chosen_input_ids"] = torch.stack([item["chosen_input_ids"] for item in batch])
        result["chosen_attention_mask"] = torch.stack(
            [item["chosen_attention_mask"] for item in batch]
        )
        result["rejected_input_ids"] = torch.stack(
            [item["rejected_input_ids"] for item in batch]
        )
        result["rejected_attention_mask"] = torch.stack(
            [item["rejected_attention_mask"] for item in batch]
        )

    return result
