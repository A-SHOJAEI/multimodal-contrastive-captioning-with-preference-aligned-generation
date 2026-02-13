"""
Multimodal Contrastive Captioning with Preference-Aligned Generation

A vision-language model combining contrastive learning with preference optimization
for high-quality image caption generation.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"
__license__ = "MIT"

from . import data, models, training, evaluation, utils

__all__ = ["data", "models", "training", "evaluation", "utils"]
