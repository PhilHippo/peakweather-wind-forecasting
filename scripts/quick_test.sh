#!/bin/bash
# Quick test script to verify the temperature forecasting pipeline works
# This runs a very short training to test all components

set -e

echo "=================================================="
echo "Quick Pipeline Test"
echo "=================================================="
echo ""
echo "This will run a quick test of each model with:"
echo "  - 2 epochs only"
echo "  - 10% of training data"
echo "  - Just to verify everything works"
echo ""

MODELS=("tcn" "enhanced_rnn" "improved_stgnn")

for model in "${MODELS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Testing ${model}..."
    echo "--------------------------------------------------"

    python -m experiments.run_temperature_prediction \
        dataset=temperature \
        model="${model}" \
        epochs=2 \
        train_batches=0.1 \
        experiment_name="QuickTest_${model}" \
        batch_size=16 || {
            echo "ERROR: ${model} test failed!"
            exit 1
        }

    echo "✓ ${model} passed!"
done

echo ""
echo "=================================================="
echo "✓ ALL TESTS PASSED!"
echo "=================================================="
echo ""
echo "The pipeline is working correctly. You can now run full experiments with:"
echo "  ./scripts/run_all_deliverables.sh"
echo ""
