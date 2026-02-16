"""Data loaders for Conceptual Captions and UltraFeedback datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import PreprocessTransform, collate_fn

logger = logging.getLogger(__name__)


class ConceptualCaptionsDataset(Dataset):
    """
    Dataset wrapper for image-caption datasets from HuggingFace.

    Uses nlphuji/flickr30k or similar real image-caption datasets.

    Attributes:
        data: Loaded dataset split
        transform: Preprocessing transform
        max_samples: Maximum number of samples to use
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[PreprocessTransform] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize image-caption dataset from HuggingFace.

        Args:
            split: Dataset split ("train", "validation", or "test")
            transform: Preprocessing transform
            max_samples: Maximum number of samples to load
        """
        self.transform = transform or PreprocessTransform()

        # Map splits for Flickr30k (it only has "test" split on HF)
        hf_split = "test"  # Flickr30k on HF only has test split
        num_samples = max_samples if max_samples else (5000 if split == "train" else 500)

        import os
        smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"

        if smoke_test:
            logger.info(f"SMOKE_TEST: creating minimal synthetic image-caption data for {split}")
            self.data = self._create_smoke_data(min(num_samples, 50), split)
            logger.info(f"Created {len(self.data)} smoke test samples for {split} split")
        else:
            logger.info(f"Loading Flickr30k image-caption dataset for {split} split ({num_samples} samples)")
            try:
                ds = load_dataset("nlphuji/flickr30k", split=hf_split, streaming=False)
                ds = ds.shuffle(seed=42)
                if split == "train":
                    ds = ds.select(range(min(num_samples, len(ds) - 2000)))
                elif split == "validation":
                    offset = max(0, len(ds) - 2000)
                    ds = ds.select(range(offset, min(offset + num_samples, len(ds) - 1000)))
                else:  # test
                    offset = max(0, len(ds) - 1000)
                    ds = ds.select(range(offset, min(offset + num_samples, len(ds))))

                self.data = ds
                logger.info(f"Loaded {len(self.data)} real image-caption samples for {split} split")

            except Exception as e:
                logger.error(f"Failed to load Flickr30k: {e}")
                raise RuntimeError(
                    f"Could not load Flickr30k dataset: {e}. "
                    "Install with: pip install datasets pillow"
                )

    def _create_smoke_data(self, num_samples: int, split: str) -> list:
        """Create minimal synthetic data for smoke test validation."""
        import numpy as np
        data = []
        captions = ["a cat on a chair", "a dog in the park", "sunset over ocean",
                     "city at night", "snowy mountains"]
        for i in range(num_samples):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            data.append({"image": Image.fromarray(img_array), "caption": captions[i % len(captions)]})
        return data

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with preprocessed image and caption
        """
        try:
            # Handle both HuggingFace Dataset objects and lists
            if isinstance(self.data, list):
                sample = self.data[idx]
            else:
                sample = self.data[idx]

            # Handle different dataset formats
            if "image" in sample:
                image = sample["image"]
            elif "img" in sample:
                image = sample["img"]
            else:
                raise KeyError("No image field found in sample")

            if "caption" in sample:
                caption = sample["caption"]
                # Flickr30k returns a list of 5 captions; pick the first
                if isinstance(caption, list):
                    caption = caption[0]
            elif "text" in sample:
                caption = sample["text"]
            elif "captions" in sample:
                # For COCO, pick first caption
                caption = sample["captions"][0] if isinstance(sample["captions"], list) else sample["captions"]
            else:
                raise KeyError("No caption field found in sample")

            # Ensure image is PIL Image in RGB mode
            if not isinstance(image, Image.Image):
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image).convert("RGB")
                    else:
                        image = Image.fromarray(np.array(image)).convert("RGB")
            else:
                image = image.convert("RGB")

            # Apply preprocessing
            processed = self.transform(image, caption)

            return processed

        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}. Returning next sample.")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))


class PreferenceDataset(Dataset):
    """
    Dataset for preference learning with chosen/rejected pairs.

    Attributes:
        data: Loaded UltraFeedback dataset
        transform: Preprocessing transform
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[PreprocessTransform] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize preference dataset.

        Args:
            split: Dataset split
            transform: Preprocessing transform
            max_samples: Maximum number of samples
        """
        self.transform = transform or PreprocessTransform()

        try:
            # Load UltraFeedback dataset
            logger.info(f"Loading UltraFeedback dataset, split={split}")
            self.data = load_dataset("openbmb/UltraFeedback", split=split, streaming=False)

            if max_samples is not None and len(self.data) > max_samples:
                self.data = self.data.shuffle(seed=42).select(range(max_samples))

            logger.info(f"Loaded {len(self.data)} preference samples")

        except Exception as e:
            logger.warning(f"Could not load UltraFeedback: {e}. Using synthetic preference data.")
            # Create synthetic preference dataset for demonstration
            self.data = self._create_synthetic_preference_data(max_samples or 1000)

    def _create_synthetic_preference_data(self, num_samples: int) -> list:
        """Create synthetic preference pairs for demonstration."""
        synthetic_data = []
        prompts = [
            "A photo of a cat",
            "A beautiful sunset",
            "A city street at night",
            "Mountains and lakes",
        ]

        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            synthetic_data.append({
                "prompt": prompt,
                "chosen": f"High quality caption: {prompt} with detailed description",
                "rejected": f"Low quality: {prompt}",
            })

        return synthetic_data

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preference pair sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with chosen and rejected text encodings
        """
        sample = self.data[idx]

        # Get chosen and rejected responses
        chosen = sample.get("chosen", sample.get("response_0", ""))
        rejected = sample.get("rejected", sample.get("response_1", ""))

        # Preprocess both
        chosen_encoded = self.transform.preprocess_text(chosen)
        rejected_encoded = self.transform.preprocess_text(rejected)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for a dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    # Only drop last batch if dataset is larger than batch_size
    # This prevents empty dataloaders for small datasets
    drop_last = len(dataset) > batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def create_dataloaders(
    config: Dict[str, Any],
    transform: Optional[PreprocessTransform] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary
        transform: Preprocessing transform

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Create datasets
    train_dataset = ConceptualCaptionsDataset(
        split=data_config.get("train_split", "train"),
        transform=transform,
        max_samples=data_config.get("max_train_samples"),
    )

    val_dataset = ConceptualCaptionsDataset(
        split=data_config.get("val_split", "validation"),
        transform=transform,
        max_samples=data_config.get("max_val_samples"),
    )

    test_dataset = ConceptualCaptionsDataset(
        split=data_config.get("test_split", "test"),
        transform=transform,
        max_samples=data_config.get("max_test_samples"),
    )

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=training_config.get("phase1_batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=training_config.get("phase1_batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
    )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=training_config.get("phase1_batch_size", 32),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
    )

    return train_loader, val_loader, test_loader
