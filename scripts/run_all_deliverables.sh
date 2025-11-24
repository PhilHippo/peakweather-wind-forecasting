#!/bin/bash
# Master script to run all experiments required for the project deliverables
#
# Deliverable 1: Implementation of 4 models
# Deliverable 2: Performance assessment with 5 seeds each
# Deliverable 3: Adding rain gauges

set -e  # Exit on error

echo "=================================================="
echo "PeakWeather Temperature Forecasting - Full Pipeline"
echo "=================================================="

# Configuration
NUM_SEEDS=5
DATASET_WEATHER="temperature"
DATASET_ALL="temperature_all_stations"

# Models to evaluate (Deliverable 1)
MODELS=("tcn" "enhanced_rnn" "improved_stgnn" "attn_longterm")

echo ""
echo "This script will run all experiments for the project deliverables:"
echo "  - Deliverable 1: Train 4 models (TCN, EnhancedRNN, ImprovedSTGNN, AttentionLongTermSTGNN)"
echo "  - Deliverable 2: Evaluate each with 5 different seeds on weather stations only"
echo "  - Deliverable 3: Evaluate on all stations (weather + rain gauges)"
echo ""
echo "WARNING: This will take a long time to complete!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# DELIVERABLE 2: Performance Assessment on Weather Stations Only
# ============================================================================
echo ""
echo "=================================================="
echo "DELIVERABLE 2: Weather Stations Only (5 seeds each)"
echo "=================================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Running ${model} with ${NUM_SEEDS} seeds..."
    echo "--------------------------------------------------"

    python scripts/run_multiseed.py \
        --model "${model}" \
        --dataset "${DATASET_WEATHER}" \
        --seeds ${NUM_SEEDS}

    echo "Completed ${model}"
done

# Run NWP baseline for comparison
echo ""
echo "--------------------------------------------------"
echo "Running ICON NWP baseline..."
echo "--------------------------------------------------"
python -m experiments.run_temperature_prediction \
    dataset="${DATASET_WEATHER}" \
    model=icon \
    experiment_name=Temperature_ICON_baseline

# ============================================================================
# DELIVERABLE 3: Adding Rain Gauges
# ============================================================================
echo ""
echo "=================================================="
echo "DELIVERABLE 3: All Stations (Weather + Rain Gauges)"
echo "=================================================="

# Train on all nodes, test on weather stations only
echo ""
echo "Experiment 1: Train on all nodes, test on weather stations only"
for model in "${MODELS[@]}"; do
    echo ""
    echo "Running ${model}..."

    python -m experiments.run_temperature_prediction \
        dataset="${DATASET_ALL}" \
        model="${model}" \
        experiment_name="Temperature_${model}_all_stations_test_weather" \
        seed=42
done

# Train and test on all nodes (separate metrics)
echo ""
echo "Experiment 2: Train and test on all nodes"
for model in "${MODELS[@]}"; do
    echo ""
    echo "Running ${model}..."

    python -m experiments.run_temperature_prediction \
        dataset="${DATASET_ALL}" \
        model="${model}" \
        experiment_name="Temperature_${model}_all_stations" \
        seed=42
done

echo ""
echo "=================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=================================================="
echo ""
echo "Results can be viewed in:"
echo "  - MLflow UI: mlflow ui --port 5000"
echo "  - Logs directory: ./logs/"
echo ""
echo "Next steps:"
echo "  1. Analyze results in MLflow"
echo "  2. Compare MAE and CRPS metrics at t=1,3,6,12,18,24"
echo "  3. Compare weather stations vs rain gauges performance"
echo "  4. Write report"
echo ""
