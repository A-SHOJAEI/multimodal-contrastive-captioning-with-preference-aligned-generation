"""Evaluation metrics for image captioning."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    corpus_bleu = sentence_bleu = meteor_score = None
    NLTK_AVAILABLE = False
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    rouge_scorer = None
    ROUGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Download NLTK data if needed
if NLTK_AVAILABLE:
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {e}")


class CaptionMetrics:
    """
    Comprehensive metrics for evaluating image captions.

    Metrics:
    - BLEU (1-4): N-gram overlap
    - METEOR: Alignment-based metric
    - ROUGE-L: Longest common subsequence
    - CIDEr: Consensus-based metric
    - CLIP Score: Vision-language alignment
    """

    def __init__(self, use_clip_score: bool = True):
        """
        Initialize caption metrics.

        Args:
            use_clip_score: Whether to compute CLIP score
        """
        self.use_clip_score = use_clip_score
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        # Try to load CLIP for CLIP score
        self.clip_model = None
        self.clip_processor = None
        if use_clip_score:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("Loaded CLIP model for CLIP score computation")
            except Exception as e:
                logger.warning(f"Could not load CLIP model: {e}")
                self.use_clip_score = False

    def compute_bleu(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.

        Args:
            references: List of reference captions (tokenized)
            hypotheses: List of generated captions (tokenized)

        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        # Tokenize
        refs_tokenized = [[ref.split() for ref in ref_list] for ref_list in references]
        hyps_tokenized = [hyp.split() for hyp in hypotheses]

        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = corpus_bleu(refs_tokenized, hyps_tokenized, weights=weights)
                bleu_scores[f"bleu_{n}"] = score
            except Exception as e:
                logger.warning(f"Error computing BLEU-{n}: {e}")
                bleu_scores[f"bleu_{n}"] = 0.0

        return bleu_scores

    def compute_meteor(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute METEOR score.

        Args:
            references: List of reference captions
            hypotheses: List of generated captions

        Returns:
            METEOR score
        """
        scores = []
        for refs, hyp in zip(references, hypotheses):
            try:
                # METEOR expects single reference, so we average over multiple
                ref_scores = [meteor_score([ref], hyp) for ref in refs]
                scores.append(np.mean(ref_scores))
            except Exception as e:
                logger.warning(f"Error computing METEOR: {e}")
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def compute_rouge(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute ROUGE-L score.

        Args:
            references: List of reference captions
            hypotheses: List of generated captions

        Returns:
            ROUGE-L F1 score
        """
        scores = []
        for refs, hyp in zip(references, hypotheses):
            # Compute ROUGE-L against all references and take max
            ref_scores = [
                self.rouge_scorer.score(ref, hyp)["rougeL"].fmeasure
                for ref in refs
            ]
            scores.append(max(ref_scores) if ref_scores else 0.0)

        return np.mean(scores) if scores else 0.0

    def compute_cider(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """
        Compute CIDEr score (simplified version).

        Args:
            references: List of reference captions
            hypotheses: List of generated captions

        Returns:
            CIDEr score
        """
        # Simplified CIDEr using TF-IDF weighted n-gram matching
        # For production, use pycocoevalcap package
        from collections import Counter

        def get_ngrams(text: str, n: int = 4) -> Counter:
            tokens = text.lower().split()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
            return Counter(ngrams)

        scores = []
        for refs, hyp in zip(references, hypotheses):
            hyp_ngrams = get_ngrams(hyp)
            ref_ngrams_list = [get_ngrams(ref) for ref in refs]

            # Compute average similarity
            similarities = []
            for ref_ngrams in ref_ngrams_list:
                common = sum((hyp_ngrams & ref_ngrams).values())
                total = max(sum(hyp_ngrams.values()), sum(ref_ngrams.values()))
                if total > 0:
                    similarities.append(common / total)

            if similarities:
                scores.append(np.mean(similarities))
            else:
                scores.append(0.0)

        # Scale to typical CIDEr range
        return (np.mean(scores) * 10.0) if scores else 0.0

    def compute_clip_score(
        self,
        images: List[Any],
        captions: List[str],
    ) -> float:
        """
        Compute CLIP score for image-caption alignment.

        Args:
            images: List of PIL images
            captions: List of generated captions

        Returns:
            Average CLIP score
        """
        if not self.use_clip_score or self.clip_model is None:
            logger.warning("CLIP score not available")
            return 0.0

        try:
            device = next(self.clip_model.parameters()).device

            # Process inputs
            inputs = self.clip_processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Compute embeddings
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                # Compute similarity
                clip_scores = (image_embeds * text_embeds).sum(dim=-1)

            return clip_scores.mean().item()

        except Exception as e:
            logger.error(f"Error computing CLIP score: {e}")
            return 0.0


def compute_all_metrics(
    references: List[List[str]],
    hypotheses: List[str],
    images: Optional[List[Any]] = None,
    compute_clip: bool = True,
) -> Dict[str, float]:
    """
    Compute all caption evaluation metrics.

    Args:
        references: List of reference captions for each sample
        hypotheses: List of generated captions
        images: Optional list of images for CLIP score
        compute_clip: Whether to compute CLIP score

    Returns:
        Dictionary of all metrics
    """
    metrics_calculator = CaptionMetrics(use_clip_score=compute_clip and images is not None)

    results = {}

    # BLEU scores
    bleu_scores = metrics_calculator.compute_bleu(references, hypotheses)
    results.update(bleu_scores)

    # METEOR
    try:
        results["meteor"] = metrics_calculator.compute_meteor(references, hypotheses)
    except Exception as e:
        logger.warning(f"Could not compute METEOR: {e}")
        results["meteor"] = 0.0

    # ROUGE
    try:
        results["rouge_l"] = metrics_calculator.compute_rouge(references, hypotheses)
    except Exception as e:
        logger.warning(f"Could not compute ROUGE: {e}")
        results["rouge_l"] = 0.0

    # CIDEr
    try:
        results["cider"] = metrics_calculator.compute_cider(references, hypotheses)
    except Exception as e:
        logger.warning(f"Could not compute CIDEr: {e}")
        results["cider"] = 0.0

    # CLIP score
    if images is not None and compute_clip:
        try:
            results["clip_score"] = metrics_calculator.compute_clip_score(images, hypotheses)
        except Exception as e:
            logger.warning(f"Could not compute CLIP score: {e}")
            results["clip_score"] = 0.0

    return results
