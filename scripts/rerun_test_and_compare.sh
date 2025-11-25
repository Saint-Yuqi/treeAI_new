#!/bin/bash
# Re-run test and compare evaluation methods

echo "=========================================="
echo "Step 1: Re-running test on test set"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Load your trained instance classifier"
echo "  2. Predict on test images"
echo "  3. Save y_true.npy and y_pred.npy"
echo "  4. Compute metrics with sklearn"
echo ""

# Activate conda environment
source ~/.bashrc 2>/dev/null
conda activate sam2

# Run test (using args from your previous config)
python scripts/test_on_test_set.py \
  --checkpoint outputs/instance_classifier/best_model.pt \
  --test-manifest sam2_instances/12_RGB_SemSegm_640_fL/test/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --classes-yaml /home/c/yuqyan/code/treeAI-segmentation/configs/data/treeAI_classes.yaml \
  --output-dir outputs/test_on_testset \
  --test-image-dir /home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/test/images \
  --test-label-dir /home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/test/labels \
  --num-viz-samples 20

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Comparing evaluation methods"
    echo "=========================================="
    echo ""
    
    # Run evaluation comparison
    python evaluate_test_results.py
    
    echo ""
    echo "=========================================="
    echo "✅ Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  - outputs/test_on_testset/test_results.json"
    echo "  - outputs/test_on_testset/evaluation_comparison.json"
    echo "  - outputs/test_on_testset/y_true.npy"
    echo "  - outputs/test_on_testset/y_pred.npy"
else
    echo ""
    echo "❌ Test failed. Please check the error messages above."
    exit 1
fi

