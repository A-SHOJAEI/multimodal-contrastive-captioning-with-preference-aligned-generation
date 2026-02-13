#!/usr/bin/env python
"""
Training script for multimodal contrastive captioning with preference alignment.

This script implements a two-phase training pipeline:
1. Phase 1: Contrastive pre-training on image-caption pairs
2. Phase 2: Preference alignment using human feedback data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from multimodal_contrastive_captioning_with_preference_aligned_generation.data.loader import (
    create_dataloaders,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.data.preprocessing import (
    PreprocessTransform,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import (
    MultimodalCaptioningModel,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.training.trainer import (
    Trainer,
)
from multimodal_contrastive_captioning_with_preference_aligned_generation.utils.config import (
    get_device,
    load_config,
    save_config,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multimodal contrastive captioning model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line args
    if args.device:
        config["hardware"]["device"] = args.device
    if args.seed:
        config["training"]["seed"] = args.seed

    # Set random seed
    seed = config.get("training", {}).get("seed", 42)
    deterministic = config.get("training", {}).get("deterministic", True)
    set_seed(seed, deterministic)

    # Get device
    device_name = config.get("hardware", {}).get("device", "cuda")
    device = get_device(device_name)

    # Save config to output directory
    output_dir = Path(config.get("output", {}).get("output_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.yaml"))

    # Initialize MLflow (optional)
    try:
        import mlflow
        mlflow.set_experiment("multimodal_contrastive_captioning")
        mlflow.start_run()
        mlflow.log_params({
            "vision_encoder": config["model"]["vision_encoder"],
            "text_decoder": config["model"]["text_decoder"],
            "projection_dim": config["model"]["projection_dim"],
            "phase1_epochs": config["training"]["phase1_epochs"],
            "phase2_epochs": config["training"]["phase2_epochs"],
            "phase1_lr": config["training"]["phase1_learning_rate"],
            "phase2_lr": config["training"]["phase2_learning_rate"],
        })
        logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")

    try:
        # Create preprocessing transform
        logger.info("Initializing data preprocessing")
        transform = PreprocessTransform(
            vision_encoder_name=config["model"]["vision_encoder"],
            text_decoder_name=config["model"]["text_decoder"],
            max_length=config["model"]["max_caption_length"],
        )

        # Create dataloaders
        logger.info("Loading datasets")
        train_loader, val_loader, test_loader = create_dataloaders(config, transform)
        logger.info(f"Created dataloaders - train: {len(train_loader)} batches, "
                   f"val: {len(val_loader)} batches, test: {len(test_loader)} batches")

        # Initialize model
        logger.info("Initializing model")
        model = MultimodalCaptioningModel(
            vision_encoder_name=config["model"]["vision_encoder"],
            text_decoder_name=config["model"]["text_decoder"],
            projection_dim=config["model"]["projection_dim"],
            freeze_vision_encoder=config["model"]["freeze_vision_encoder"],
            freeze_text_decoder=config["model"]["freeze_text_decoder"],
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

        # Initialize trainer
        logger.info("Initializing trainer")
        trainer = Trainer(model, config, device)

        # Phase 1: Contrastive pre-training
        phase1_epochs = config["training"]["phase1_epochs"]
        if phase1_epochs > 0:
            logger.info("=" * 70)
            logger.info("PHASE 1: Contrastive Pre-training")
            logger.info("=" * 70)

            phase1_history = trainer.train(
                train_loader,
                val_loader,
                num_epochs=phase1_epochs,
                phase=1,
            )

            # Log phase 1 metrics
            try:
                import mlflow
                for epoch, loss in enumerate(phase1_history["train_loss"]):
                    mlflow.log_metric("phase1_train_loss", loss, step=epoch)
                for epoch, loss in enumerate(phase1_history["val_loss"]):
                    mlflow.log_metric("phase1_val_loss", loss, step=epoch)
            except Exception:
                pass

            logger.info("Phase 1 training completed")

        # Phase 2: Preference alignment
        phase2_epochs = config["training"]["phase2_epochs"]
        if phase2_epochs > 0:
            logger.info("=" * 70)
            logger.info("PHASE 2: Preference Alignment")
            logger.info("=" * 70)

            # Freeze vision encoder if specified
            if config["training"]["phase2_freeze_vision"]:
                for param in model.vision_encoder.parameters():
                    param.requires_grad = False
                logger.info("Vision encoder frozen for Phase 2")

            # Update learning rate for phase 2
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = config["training"]["phase2_learning_rate"]

            # Reset scheduler
            trainer.scheduler = None

            phase2_history = trainer.train(
                train_loader,
                val_loader,
                num_epochs=phase2_epochs,
                phase=2,
            )

            # Log phase 2 metrics
            try:
                import mlflow
                for epoch, loss in enumerate(phase2_history["train_loss"]):
                    mlflow.log_metric("phase2_train_loss", loss, step=epoch)
                for epoch, loss in enumerate(phase2_history["val_loss"]):
                    mlflow.log_metric("phase2_val_loss", loss, step=epoch)
            except Exception:
                pass

            logger.info("Phase 2 training completed")

        # Save final model
        final_model_path = output_dir / "final_model.pt"
        trainer.save_checkpoint(str(final_model_path), is_best=False)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Best validation metric: {trainer.best_metric:.4f}")

        # Log final metrics
        try:
            import mlflow
            mlflow.log_metric("best_val_metric", trainer.best_metric)
            mlflow.log_artifact(str(output_dir / "config.yaml"))
            mlflow.end_run()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
