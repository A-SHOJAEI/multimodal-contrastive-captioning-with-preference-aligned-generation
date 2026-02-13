#!/usr/bin/env python
"""
Inference script for generating captions from images.

Loads a trained model and generates captions for new images.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image

from multimodal_contrastive_captioning_with_preference_aligned_generation.data.preprocessing import (
    PreprocessTransform,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import (
    MultimodalCaptioningModel,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.utils.config import (
    get_device,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--num-captions",
        type=int,
        default=1,
        help="Number of captions to generate per image",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of beam search",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save captions (optional)",
    )

    return parser.parse_args()


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """
    Load model for inference.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Tuple of (model, transform, tokenizer)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint
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

    # Create preprocessing transform
    transform = PreprocessTransform(
        vision_encoder_name=config["model"]["vision_encoder"],
        text_decoder_name=config["model"]["text_decoder"],
        max_length=config["model"]["max_caption_length"],
    )

    logger.info("Model loaded successfully")

    return model, transform, transform.tokenizer


def generate_caption(
    model: MultimodalCaptioningModel,
    image: Image.Image,
    transform: PreprocessTransform,
    tokenizer,
    device: torch.device,
    num_beams: int = 5,
    num_captions: int = 1,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> list:
    """
    Generate captions for an image with confidence scores.

    Args:
        model: Trained model
        image: PIL Image
        transform: Preprocessing transform
        tokenizer: Text tokenizer
        device: Device to run on
        num_beams: Number of beams
        num_captions: Number of captions to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling

    Returns:
        List of dicts with 'caption' and 'confidence' keys
    """
    # Preprocess image
    pixel_values = transform.preprocess_image(image).unsqueeze(0).to(device)

    # Generate captions
    with torch.no_grad():
        generated_ids = model.generate_caption(
            pixel_values,
            num_beams=num_beams,
            max_length=77,
            do_sample=do_sample,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Extract sequences
        if hasattr(generated_ids, 'sequences'):
            sequences = generated_ids.sequences
            # Compute confidence from scores if available
            if hasattr(generated_ids, 'sequences_scores'):
                scores = generated_ids.sequences_scores
                confidences = torch.softmax(scores, dim=0).cpu().tolist()
            else:
                # Default confidence based on beam search
                confidences = [1.0 / num_beams] * sequences.size(0)
        else:
            sequences = generated_ids
            confidences = [1.0]

    # Decode captions
    captions = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    # Return captions with confidence scores
    results = []
    for caption, confidence in zip(captions, confidences):
        results.append({
            'caption': caption,
            'confidence': float(confidence),
        })

    return results


def process_single_image(
    image_path: str,
    model: MultimodalCaptioningModel,
    transform: PreprocessTransform,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """
    Process a single image and generate caption.

    Args:
        image_path: Path to image
        model: Model
        transform: Transform
        tokenizer: Tokenizer
        device: Device
        args: Command line arguments

    Returns:
        Dictionary with image path and captions
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Generate captions
        captions = generate_caption(
            model,
            image,
            transform,
            tokenizer,
            device,
            num_beams=args.num_beams,
            num_captions=args.num_captions,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )

        return {
            "image_path": str(image_path),
            "captions": captions,
        }

    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            "image_path": str(image_path),
            "captions": [],
            "error": str(e),
        }


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Get device
    device = get_device(args.device)

    try:
        # Load model
        model, transform, tokenizer = load_model_for_inference(args.checkpoint, device)

        # Process image(s)
        image_path = Path(args.image)
        results = []

        if image_path.is_file():
            # Single image
            logger.info(f"Processing image: {image_path}")
            result = process_single_image(
                image_path,
                model,
                transform,
                tokenizer,
                device,
                args,
            )
            results.append(result)

        elif image_path.is_dir():
            # Directory of images
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
            image_files = [
                f for f in image_path.iterdir()
                if f.suffix.lower() in image_extensions
            ]

            logger.info(f"Found {len(image_files)} images in directory")

            for img_file in image_files:
                logger.info(f"Processing: {img_file.name}")
                result = process_single_image(
                    img_file,
                    model,
                    transform,
                    tokenizer,
                    device,
                    args,
                )
                results.append(result)

        else:
            logger.error(f"Invalid image path: {image_path}")
            sys.exit(1)

        # Print results
        logger.info("=" * 70)
        logger.info("GENERATED CAPTIONS")
        logger.info("=" * 70)

        for result in results:
            if "error" in result:
                logger.error(f"Failed: {result['image_path']} - {result['error']}")
            else:
                logger.info(f"\nImage: {result['image_path']}")
                for i, caption_data in enumerate(result["captions"], 1):
                    if isinstance(caption_data, dict):
                        caption = caption_data['caption']
                        confidence = caption_data.get('confidence', 1.0)
                        logger.info(f"  Caption {i}: {caption} (confidence: {confidence:.4f})")
                    else:
                        # Fallback for old format
                        logger.info(f"  Caption {i}: {caption_data}")

        logger.info("=" * 70)

        # Save to file if specified
        if args.output:
            import json

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
