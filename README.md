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

Run training to generate results in `results/` directory. Example command:

```bash
python scripts/train.py
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output results/metrics.json
```

## Ablation Studies

Compare full model vs contrastive-only baseline:

```bash
# Train both variants
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/ablation.yaml

# Evaluate both
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output results/default.json
python scripts/evaluate.py --checkpoint checkpoints/best_model_ablation.pt --output results/ablation.json
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
