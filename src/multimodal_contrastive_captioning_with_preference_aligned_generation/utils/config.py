"""Configuration utilities for loading and managing YAML configs."""

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the YAML file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {output_path}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


def get_device(device_name: str = "cuda") -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device_name: Requested device name ("cuda", "cpu", or specific device like "cuda:0")

    Returns:
        torch.device object
    """
    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")

    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic behavior
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")
