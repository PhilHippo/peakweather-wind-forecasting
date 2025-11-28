#!/bin/bash

# Train all learnable models for temperature forecasting
# Usage: bash train_all_models.sh

set -e  # Exit on error

# Configuration
DATASET="temperature"
MODELS=("tcn" "rnn" "stgnn" "attn_longterm")
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SUMMARY_LOG="training_summary_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Training All Models for Temperature Forecasting"
echo "================================================================================"
echo "Dataset: ${DATASET}"
echo "Models: ${MODELS[@]}"
echo "Summary log: ${SUMMARY_LOG}"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Initialize summary log
cat > "${SUMMARY_LOG}" <<EOF
===============================================================================
Temperature Forecasting - Training Summary
===============================================================================
Start Time: $(date)
Dataset: ${DATASET}
Models: ${MODELS[@]}

===============================================================================
EOF

# Train each model
for MODEL in "${MODELS[@]}"; do
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}Training Model: ${MODEL}${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
    
    # Record start time
    MODEL_START=$(date +%s)
    MODEL_START_TIME=$(date)
    
    # Create temporary log for this model
    MODEL_LOG="temp_${MODEL}_${TIMESTAMP}.log"
    
    # Run training and capture output
    echo -e "${YELLOW}Running: python -m experiments.run_temperature_prediction dataset=${DATASET} model=${MODEL}${NC}"
    echo ""
    
    if python -m experiments.run_temperature_prediction dataset="${DATASET}" model="${MODEL}" 2>&1 | tee "${MODEL_LOG}"; then
        # Training succeeded
        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))
        
        echo ""
        echo -e "${GREEN}✓ ${MODEL} training completed successfully${NC}"
        echo -e "${GREEN}  Duration: $((MODEL_DURATION / 60))m $((MODEL_DURATION % 60))s${NC}"
        
        # Extract metrics from log
        TEST_MAE=$(grep -oP "test_mae['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" "${MODEL_LOG}" | head -1 || echo "N/A")
        TEST_MSE=$(grep -oP "test_mse['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" "${MODEL_LOG}" | head -1 || echo "N/A")
        VAL_MAE=$(grep -oP "val_mae['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" "${MODEL_LOG}" | tail -1 || echo "N/A")
        BEST_EPOCH=$(grep -oP "best_model_score.*:\s*\K[0-9]+\.[0-9]+" "${MODEL_LOG}" | head -1 || echo "N/A")
        
        # Log to summary
        cat >> "${SUMMARY_LOG}" <<EOF

-------------------------------------------------------------------------------
Model: ${MODEL}
-------------------------------------------------------------------------------
Status: SUCCESS
Start: ${MODEL_START_TIME}
Duration: $((MODEL_DURATION / 60))m $((MODEL_DURATION % 60))s
Best Val MAE: ${VAL_MAE}
Test MAE: ${TEST_MAE}
Test MSE: ${TEST_MSE}
Log file: ${MODEL_LOG}

EOF
        
        echo ""
        echo -e "${GREEN}Metrics:${NC}"
        echo -e "  Val MAE: ${VAL_MAE}"
        echo -e "  Test MAE: ${TEST_MAE}"
        echo -e "  Test MSE: ${TEST_MSE}"
        
    else
        # Training failed
        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))
        
        echo ""
        echo -e "${RED}✗ ${MODEL} training FAILED${NC}"
        echo -e "${RED}  Duration: $((MODEL_DURATION / 60))m $((MODEL_DURATION % 60))s${NC}"
        echo -e "${RED}  Check log: ${MODEL_LOG}${NC}"
        
        # Log to summary
        cat >> "${SUMMARY_LOG}" <<EOF

-------------------------------------------------------------------------------
Model: ${MODEL}
-------------------------------------------------------------------------------
Status: FAILED
Start: ${MODEL_START_TIME}
Duration: $((MODEL_DURATION / 60))m $((MODEL_DURATION % 60))s
Error: See log file for details
Log file: ${MODEL_LOG}

EOF
    fi
    
    echo ""
done

# Final summary
TOTAL_END=$(date +%s)
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Training Complete${NC}"
echo -e "${BLUE}================================================================================${NC}"

# Count successes
SUCCESS_COUNT=$(grep -c "Status: SUCCESS" "${SUMMARY_LOG}" || echo "0")
TOTAL_COUNT=${#MODELS[@]}

cat >> "${SUMMARY_LOG}" <<EOF

===============================================================================
Summary
===============================================================================
End Time: $(date)
Models Completed: ${SUCCESS_COUNT}/${TOTAL_COUNT}

EOF

echo ""
echo -e "${GREEN}Results saved to: ${SUMMARY_LOG}${NC}"
echo ""
echo "Model-specific logs:"
for MODEL in "${MODELS[@]}"; do
    if [ -f "temp_${MODEL}_${TIMESTAMP}.log" ]; then
        echo "  - temp_${MODEL}_${TIMESTAMP}.log"
    fi
done
echo ""

# Display summary table
echo "================================================================================"
echo "Quick Results Summary"
echo "================================================================================"
printf "%-20s %-10s %-12s %-12s\n" "Model" "Status" "Val MAE" "Test MAE"
echo "--------------------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    STATUS=$(grep -A 10 "Model: ${MODEL}" "${SUMMARY_LOG}" | grep "Status:" | awk '{print $2}' || echo "UNKNOWN")
    VAL_MAE=$(grep -A 10 "Model: ${MODEL}" "${SUMMARY_LOG}" | grep "Best Val MAE:" | awk '{print $4}' || echo "N/A")
    TEST_MAE=$(grep -A 10 "Model: ${MODEL}" "${SUMMARY_LOG}" | grep "Test MAE:" | awk '{print $3}' || echo "N/A")
    
    if [ "$STATUS" = "SUCCESS" ]; then
        printf "${GREEN}%-20s %-10s %-12s %-12s${NC}\n" "$MODEL" "$STATUS" "$VAL_MAE" "$TEST_MAE"
    else
        printf "${RED}%-20s %-10s %-12s %-12s${NC}\n" "$MODEL" "$STATUS" "$VAL_MAE" "$TEST_MAE"
    fi
done

echo "================================================================================"
echo ""
echo -e "${BLUE}View full summary: cat ${SUMMARY_LOG}${NC}"
echo -e "${BLUE}View MLflow results: mlflow ui --backend-store-uri ./mlruns${NC}"
echo ""

