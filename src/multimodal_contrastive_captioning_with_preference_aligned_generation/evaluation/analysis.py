"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def generate_evaluation_report(
    metrics: Dict[str, float],
    output_path: str,
) -> None:
    """
    Generate a comprehensive evaluation report.

    Args:
        metrics: Dictionary of computed metrics
        output_path: Path to save the report
    """
    report_lines = [
        "=" * 70,
        "CAPTION GENERATION EVALUATION REPORT",
        "=" * 70,
        "",
        "Overall Metrics:",
        "-" * 70,
    ]

    # Group metrics by category
    bleu_metrics = {k: v for k, v in metrics.items() if k.startswith("bleu")}
    other_metrics = {k: v for k, v in metrics.items() if not k.startswith("bleu")}

    # BLEU scores
    if bleu_metrics:
        report_lines.append("BLEU Scores:")
        for name, value in sorted(bleu_metrics.items()):
            report_lines.append(f"  {name.upper():12s}: {value:.4f}")
        report_lines.append("")

    # Other metrics
    if other_metrics:
        report_lines.append("Other Metrics:")
        for name, value in sorted(other_metrics.items()):
            display_name = name.replace("_", " ").title()
            report_lines.append(f"  {display_name:12s}: {value:.4f}")
        report_lines.append("")

    # Overall assessment
    report_lines.extend([
        "-" * 70,
        "Target Metrics (for reference):",
        "  CLIP Score:   0.7500",
        "  BLEU-4:       0.2800",
        "  CIDEr:        1.1000",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(report_text)

    logger.info(f"Saved evaluation report to {output_path}")

    # Also print to console
    print("\n" + report_text)


def save_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json",
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Output format ("json" or "txt")
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Convert numpy types to Python types
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = value

        with open(output_file, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    elif format == "txt":
        with open(output_file, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Saved results to {output_path}")

    else:
        raise ValueError(f"Unknown format: {format}")


def compute_summary_statistics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics across multiple evaluation runs.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    # Collect all metric names
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    summary = {}
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list]
        summary[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    return summary


def compare_models(
    baseline_metrics: Dict[str, float],
    improved_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare two models and compute improvement percentages.

    Args:
        baseline_metrics: Baseline model metrics
        improved_metrics: Improved model metrics

    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}

    for key in baseline_metrics.keys():
        if key in improved_metrics:
            baseline_val = baseline_metrics[key]
            improved_val = improved_metrics[key]

            if baseline_val != 0:
                improvement_pct = ((improved_val - baseline_val) / abs(baseline_val)) * 100
                improvements[f"{key}_improvement_pct"] = improvement_pct

    return improvements
