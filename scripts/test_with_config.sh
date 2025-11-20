#!/bin/bash
# Test script for evaluating on TRUE test set
# Usage: bash test_with_config.sh [checkpoint_path]
#
# IMPORTANT: Activate the conda environment first!
# $ conda activate sam2
# $ bash scripts/test_with_config.sh outputs/instance_classifier/best_model.pt

set -e

echo "=========================================="
echo "Testing on Test Set (NOT Pick Dataset!)"
echo "=========================================="

# Get checkpoint path
if [ -z "$1" ]; then
    # Use best model by default
    CHECKPOINT="/home/c/yuqyan/code/treeAI_new/outputs/instance_classifier/best_model.pt"
else
    CHECKPOINT="$1"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

# Test dataset paths
TEST_IMAGE_DIR="/home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/test/images"
TEST_LABEL_DIR="/home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/test/labels"
TEST_MANIFEST="sam2_instances/12_RGB_SemSegm_640_fL/test/instances_manifest.jsonl"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run testing
python "$SCRIPT_DIR/test_on_test_set.py" \
  --checkpoint "$CHECKPOINT" \
  --test-manifest "$PROJECT_ROOT/$TEST_MANIFEST" \
  --output-root "$PROJECT_ROOT/sam2_instances/12_RGB_SemSegm_640_fL" \
  --classes-yaml /home/c/yuqyan/code/treeAI-segmentation/configs/data/treeAI_classes.yaml \
  --test-image-dir "$TEST_IMAGE_DIR" \
  --test-label-dir "$TEST_LABEL_DIR" \
  --output-dir "$PROJECT_ROOT/outputs/test_on_testset" \
  --num-viz-samples 20

echo ""
echo "=========================================="
echo "Testing complete!"
echo "Results saved to: outputs/test_on_testset/"
echo "=========================================="

