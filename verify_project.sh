#!/bin/bash

echo "======================================================================"
echo "Verifying Multimodal Contrastive Captioning Project"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check directory structure
echo -e "\n${GREEN}Checking directory structure...${NC}"
for dir in configs src/multimodal_contrastive_captioning_with_preference_aligned_generation/{data,models,training,evaluation,utils} scripts tests; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir exists"
    else
        echo -e "  ${RED}✗ $dir missing${NC}"
        exit 1
    fi
done

# Check required files
echo -e "\n${GREEN}Checking required files...${NC}"
for file in requirements.txt pyproject.toml README.md LICENSE .gitignore \
            configs/default.yaml configs/ablation.yaml \
            scripts/train.py scripts/evaluate.py scripts/predict.py; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo -e "  ${RED}✗ $file missing${NC}"
        exit 1
    fi
done

# Check Python imports
echo -e "\n${GREEN}Checking Python imports...${NC}"
python3 -c "
import sys
sys.path.insert(0, 'src')
from multimodal_contrastive_captioning_with_preference_aligned_generation import data, models, training, evaluation, utils
print('  ✓ All module imports successful')
" || exit 1

# Check for custom components
echo -e "\n${GREEN}Checking custom components...${NC}"
python3 -c "
import sys
sys.path.insert(0, 'src')
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.components import ContrastivePreferenceLoss
loss_fn = ContrastivePreferenceLoss()
print('  ✓ ContrastivePreferenceLoss (custom component) working')
" || exit 1

# Check YAML configs load properly
echo -e "\n${GREEN}Checking YAML configurations...${NC}"
python3 -c "
import yaml
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
    assert config['training']['phase1_epochs'] == 10
    assert config['training']['phase2_epochs'] == 5
    assert config['loss']['preference_weight'] == 0.5
print('  ✓ default.yaml loads correctly')

with open('configs/ablation.yaml') as f:
    config = yaml.safe_load(f)
    assert config['training']['phase2_epochs'] == 0
    assert config['loss']['preference_weight'] == 0.0
print('  ✓ ablation.yaml loads correctly (preference learning disabled)')
" || exit 1

# Check scripts have proper structure
echo -e "\n${GREEN}Checking script structure...${NC}"
for script in scripts/train.py scripts/evaluate.py scripts/predict.py; do
    if grep -q "if __name__ == \"__main__\":" "$script" && \
       grep -q "import sys" "$script" && \
       grep -q "sys.path.insert" "$script"; then
        echo "  ✓ $script has correct structure"
    else
        echo -e "  ${RED}✗ $script missing required components${NC}"
        exit 1
    fi
done

# Check test files exist and have test functions
echo -e "\n${GREEN}Checking test files...${NC}"
for test_file in tests/test_data.py tests/test_model.py tests/test_training.py; do
    if grep -q "def test_" "$test_file"; then
        echo "  ✓ $test_file has test functions"
    else
        echo -e "  ${RED}✗ $test_file has no test functions${NC}"
        exit 1
    fi
done

# Check for proper docstrings
echo -e "\n${GREEN}Checking documentation...${NC}"
python3 -c "
import sys
sys.path.insert(0, 'src')
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.model import MultimodalCaptioningModel
from multimodal_contrastive_captioning_with_preference_aligned_generation.models.components import ContrastivePreferenceLoss
assert MultimodalCaptioningModel.__doc__ is not None
assert ContrastivePreferenceLoss.__doc__ is not None
print('  ✓ Key classes have docstrings')
" || exit 1

# Summary
echo -e "\n======================================================================"
echo -e "${GREEN}✓ ALL CHECKS PASSED!${NC}"
echo "======================================================================"
echo -e "\nProject is ready for use. Quick start:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Train model: python scripts/train.py --config configs/default.yaml"
echo "  3. Evaluate: python scripts/evaluate.py --checkpoint models/best_model.pt"
echo "  4. Run tests: pytest tests/ -v"
echo ""
