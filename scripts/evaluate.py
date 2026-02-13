#!/usr/bin/env python
"""
Evaluation script for multimodal captioning model.

Loads a trained model and evaluates it on the test set using multiple metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

from multimodal_contrastive_captioning_with_preference_aligned_generation.data.loader import (
    create_dataloaders,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.data.preprocessing import (
    PreprocessTransform,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.analysis import (
    generate_evaluation_report,
    save_results,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.evaluation.metrics import (
    compute_all_metrics,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import (
    MultimodalCaptioningModel,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.utils.config import (
    get_device,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate multimodal captioning model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_metrics.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for beam search",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Initialize model
    model = MultimodalCaptioningModel(
        vision_encoder_name=config["model"]["vision_encoder"],
        text_decoder_name=config["model"]["text_decoder"],
        projection_dim=config["model"]["projection_dim"],
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    return model, config


def evaluate_model(
    model: MultimodalCaptioningModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_beams: int = 5,
    num_samples: int = None,
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        num_beams: Number of beams for generation
        num_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_references = []
    all_hypotheses = []
    all_images = []

    num_evaluated = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if num_samples is not None and num_evaluated >= num_samples:
                break

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Generate captions
            generated_ids = model.generate_caption(
                pixel_values,
                num_beams=num_beams,
                max_length=77,
            )

            # Decode generated captions
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            generated_captions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # Decode reference captions
            reference_captions = tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=True,
            )

            # Store results
            for gen_cap, ref_cap in zip(generated_captions, reference_captions):
                all_hypotheses.append(gen_cap)
                all_references.append([ref_cap])  # List of references per sample

            num_evaluated += len(generated_captions)

    logger.info(f"Evaluated {num_evaluated} samples")

    # Compute metrics
    logger.info("Computing evaluation metrics")
    metrics = compute_all_metrics(
        references=all_references,
        hypotheses=all_hypotheses,
        images=None,  # CLIP score would require original images
        compute_clip=False,
    )

    # Add sample count
    metrics["num_samples"] = num_evaluated

    return metrics


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device(args.device)

    # Set seed for reproducibility
    set_seed(42)

    try:
        # Load checkpoint
        model, config = load_checkpoint(args.checkpoint, device)

        # Create preprocessing transform
        transform = PreprocessTransform(
            vision_encoder_name=config["model"]["vision_encoder"],
            text_decoder_name=config["model"]["text_decoder"],
            max_length=config["model"]["max_caption_length"],
        )

        # Create test dataloader
        logger.info("Loading test dataset")
        _, _, test_loader = create_dataloaders(config, transform)

        # Evaluate model
        metrics = evaluate_model(
            model,
            test_loader,
            device,
            num_beams=args.num_beams,
            num_samples=args.num_samples,
        )

        # Print results
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"{metric_name:20s}: {value:.4f}")
            else:
                logger.info(f"{metric_name:20s}: {value}")
        logger.info("=" * 70)

        # Save results
        save_results(metrics, args.output, format="json")

        # Generate evaluation report
        report_path = output_path.parent / "evaluation_report.txt"
        generate_evaluation_report(metrics, str(report_path))

        logger.info(f"Results saved to {args.output}")
        logger.info(f"Report saved to {report_path}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
