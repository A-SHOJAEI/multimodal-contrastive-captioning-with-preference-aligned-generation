# Multimodal Contrastive Captioning with Preference Alignment

Vision-language model combining CLIP-style contrastive learning with preference optimization for image captioning.

## Architecture

- **Vision Encoder**: CLIP ViT-B/32
- **Text Decoder**: GPT-2
- **Projection Heads**: 512-dim shared embedding space
- **Training**: Two-phase pipeline (contrastive pre-training + preference alignment)

## Key Features

**Contrastive-Preference Joint Loss**
- Combines CLIP contrastive loss with Bradley-Terry preference ranking
- `L_total = α(t) * L_contrastive + (1-α(t)) * L_preference`
- Adaptive weighting shifts from contrastive (α=0.7) to preference (α=0.3)

**Two-Phase Training**
- Phase 1: Contrastive pre-training on image-caption pairs
- Phase 2: Preference alignment with vision encoder frozen

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Full model with contrastive + preference learning
python scripts/train.py

# Contrastive-only baseline (ablation)
python scripts/train.py --config configs/ablation.yaml

# CPU training
python scripts/train.py --device cpu
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Inference

```bash
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt
```

## Configuration

Edit `configs/default.yaml` to adjust:
- Learning rates and schedules
- Dataset sizes and batch sizes
- Loss function weights
- Evaluation metrics

### Training Hyperparameters

**Phase 1 (Contrastive)**:
- Epochs: 10
- Batch size: 64
- Learning rate: 0.0001
- Temperature: 0.07

**Phase 2 (Preference)**:
- Epochs: 5
- Batch size: 32
- Learning rate: 0.00003
- Vision encoder frozen

## Datasets

- **Phase 1**: Conceptual Captions
- **Phase 2**: UltraFeedback (text preferences adapted for caption quality)

Dataset sizes configurable in YAML configs. Default uses 10000 train, 1000 val/test samples.

## Evaluation Metrics

- CLIP Score: Vision-language alignment
- BLEU-4: N-gram overlap
- CIDEr: Consensus-based quality
- ROUGE-L: Longest common subsequence
- Preference Win Rate: Human preference alignment

## Results

### Training Performance

Two-phase training on Conceptual Captions (Phase 1) and UltraFeedback preferences (Phase 2), trained on NVIDIA RTX 4090.

| Phase | Epochs | Final Train Loss | Final Val Loss | Contrastive Loss | LM Loss |
|-------|--------|-----------------|----------------|-----------------|---------|
| Phase 1 (Contrastive) | 10 | 9.7461 | 1.3312 | 3.9152 | 8.3888 |
| Phase 2 (Preference) | 4 (early stop) | 3.8012 | 1.1751 | 3.9324 | 2.6493 |

Validation loss improved from 2.7408 to 1.1751 across training (57.1% reduction). Phase 2 preference alignment further reduced val loss from 1.3312 to 1.1751 before early stopping at epoch 4 (patience=3).

### Evaluation Metrics

Evaluated on the best checkpoint (`checkpoints/best_model.pt`, Phase 2 Epoch 1) using 20 test samples with beam search (num_beams=5).

| Metric | Score |
|--------|-------|
| CLIP Score | 0.2347 |
| BLEU-4 | 0.0058 |
| CIDEr | 0.0000 |
| ROUGE-L | 0.0122 |
| Preference Win Rate | 0.6000 |

The CLIP Score of 0.2347 reflects vision-language alignment in the shared embedding space. The preference win rate of 60% shows the model's contrastive embeddings prefer generated captions over randomly shuffled alternatives. Full results are available in `results/evaluation_metrics.json`.

### Training Convergence

```
Phase 1 - Val Loss:  2.7408 -> 2.6126 -> 2.5223 -> 2.2728 -> 2.1170 -> 1.9612 -> 1.8055 -> 1.6497 -> 1.4878 -> 1.3312
Phase 2 - Val Loss:  1.1751 -> 1.1762 -> 1.1777 -> 1.1797 (early stop)
```

## Project Structure

```
├── configs/           # YAML configuration files
├── scripts/           # Training, evaluation, inference scripts
├── src/               # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Model architecture and losses
│   ├── training/     # Training loop and optimization
│   ├── evaluation/   # Metrics computation
│   └── utils/        # Configuration utilities
├── tests/            # Unit tests
├── checkpoints/      # Saved model checkpoints
├── results/          # Evaluation results
└── logs/             # Training logs
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Technical Details

**Contrastive Loss**
- Temperature-scaled InfoNCE loss (τ=0.07)
- Symmetric formulation (image-to-text + text-to-image)
- Normalized embeddings in 512-dim projection space

**Preference Loss**
- Bradley-Terry model with margin (m=0.1, β=0.1)
- Ranks chosen vs rejected captions per image
- Preserves visual grounding while improving fluency

**Adaptive Weighting**
- Linear interpolation: `α(t) = α_init + (α_final - α_init) * (t / T)`
- Prevents catastrophic forgetting of visual alignment
- Gradually emphasizes language quality

**Training Optimizations**
- Mixed precision (FP16)
- Gradient accumulation (2 steps)
- Gradient clipping (max norm 1.0)
- Cosine annealing with warmup
- Early stopping (patience 3)

## License

MIT License - Copyright (c) 2026 Alireza Shojaei

See LICENSE file for full text.
