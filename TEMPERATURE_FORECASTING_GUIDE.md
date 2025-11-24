# Temperature Forecasting Guide

This guide explains how to run the temperature forecasting experiments for the GDL project deliverables.

## Project Overview

This project implements spatiotemporal graph neural networks (STGNNs) for temperature forecasting using the PeakWeather dataset from MeteoSwiss. The goal is to:

1. **Deliverable 1**: Implement 4 models (TCN, Enhanced RNN, Improved STGNN, Attention-based STGNN)
2. **Deliverable 2**: Evaluate models on weather stations only with MAE and CRPS metrics
3. **Deliverable 3**: Assess impact of including rain gauges (lower quality sensors)

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f conda_env.yml
conda activate peakweather

# For M1/M2/M3 Macs, install PyTorch Geometric via pip:
pip install torch-geometric
```

### 2. Run a Single Experiment

```bash
# Train TCN model (Model0) on temperature forecasting
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# Train Enhanced RNN (Model1)
python -m experiments.run_temperature_prediction dataset=temperature model=enhanced_rnn

# Train Improved STGNN (Model2)
python -m experiments.run_temperature_prediction dataset=temperature model=improved_stgnn

# Train Attention STGNN (Model3)
python -m experiments.run_temperature_prediction dataset=temperature model=attn_longterm
```

### 3. Run All Deliverables (Automated)

```bash
# This will run all required experiments for Deliverables 2 and 3
./scripts/run_all_deliverables.sh
```

**WARNING**: Running all experiments takes many hours. Consider running a subset first.

## Model Descriptions

### Model0: TCN (Temporal Convolutional Network)
- **Purpose**: Non-graph baseline as required by Deliverable 1
- **Architecture**: Stacked temporal convolutional blocks with dilations
- **Config**: `config/model/tcn.yaml`
- **File**: `lib/nn/models/temperature_models.py:TCNModel`
- **Key Features**:
  - Does NOT use graph structure
  - Processes each station independently
  - Temporal receptive field through dilated convolutions

### Model1: Enhanced RNN
- **Purpose**: Baseline RNN with embeddings as required by Deliverable 1
- **Architecture**: GRU with node embeddings and multi-head attention
- **Config**: `config/model/enhanced_rnn.yaml`
- **File**: `lib/nn/models/temperature_models.py:EnhancedRNNModel`
- **Key Features**:
  - Node-specific embeddings to capture station characteristics
  - Self-attention for temporal dependencies
  - Does NOT use graph edges (no spatial message passing)

### Model2: Improved STGNN
- **Purpose**: Baseline STGNN as required by Deliverable 1
- **Architecture**: Temporal RNN + Spatial DiffConv layers
- **Config**: `config/model/improved_stgnn.yaml`
- **File**: `lib/nn/models/temperature_models.py:ImprovedSTGNN`
- **Key Features**:
  - Temporal processing with GRU
  - Spatial processing with diffusion convolution
  - Skip connections and layer normalization

### Model3: Attention-based Long-term STGNN
- **Purpose**: Competitive STGNN from literature (Deliverable 1)
- **Architecture**: Patch-based long-term attention with spatial-temporal awareness
- **Config**: `config/model/attn_longterm.yaml`
- **File**: `lib/nn/models/learnable_models/attention_longterm_stgnn.py`
- **Key Features**:
  - Patching for long-term dependencies (window=168h = 7 days)
  - Multi-head attention across time patches
  - Spatial-temporal aware attention
  - Adaptive graph construction (kNN + attention)

## Dataset Configurations

### Weather Stations Only (Deliverable 2)
```yaml
# config/dataset/temperature.yaml
dataset:
  hparams:
    station_type: weather  # Only high-quality weather stations
    target_channels: [temperature]
    covariate_channels: other  # All 8 sensor variables
  splitting:
    test_start: "2024-04-01"  # As required
```

### All Stations (Deliverable 3)
```yaml
# config/dataset/temperature_all_stations.yaml
dataset:
  hparams:
    station_type: null  # Both weather stations AND rain gauges
```

## Running Experiments with Multiple Seeds

### Deliverable 2 Requirement
"The four selected models should be trained and tested starting from 5 different seeds."

```bash
# Run single model with 5 seeds
python scripts/run_multiseed.py --model tcn --seeds 5

# Run all models with 5 seeds each (as required)
for model in tcn enhanced_rnn improved_stgnn attn_longterm; do
    python scripts/run_multiseed.py --model $model --seeds 5
done
```

## Metrics and Evaluation

### Required Metrics (Deliverable 2)

1. **MAE** (Mean Absolute Error) - for point predictions
2. **CRPS** (Continuous Ranked Probability Score) - via Energy Score for probabilistic predictions

### Time-Specific Metrics
As required: "MAE and CRPS should be reported at t=1, t=3, t=6, t=12, t=18, t=24"

The implementation automatically computes:
- `mae_1h`, `mae_3h`, `mae_6h`, `mae_12h`, `mae_18h`, `mae_24h`
- `ens_1h`, `ens_3h`, `ens_6h`, `ens_12h`, `ens_18h`, `ens_24h` (Energy Score = CRPS)

### NWP Baseline Comparison
As required: "Report NWP performance as reference"

```bash
python -m experiments.run_temperature_prediction dataset=temperature model=icon
```

## Deliverable 3: Rain Gauges Experiments

### Experiment 1: Train on all, test on weather stations only
```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature_all_stations \
    model=improved_stgnn \
    experiment_name=Temperature_AllStations_TestWeather
```

### Experiment 2: Train and test on all nodes
```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature_all_stations \
    model=improved_stgnn \
    experiment_name=Temperature_AllStations
```

**Analysis**: Compare metrics between weather stations and rain gauges separately.

## Viewing Results

### MLflow UI
```bash
mlflow ui --port 5000
# Open browser to http://127.0.0.1:5000
```

All experiments are logged with:
- Model hyperparameters
- Training/validation/test metrics
- MAE and CRPS at specific time horizons
- Training curves
- Best model checkpoints

### Checkpoints
Best models are saved in:
```
logs/Temperature/{model_name}/{date}/{time}/epoch_{n}-step_{s}.ckpt
```

## Configuration Overrides

### Change Hyperparameters
```bash
# Adjust learning rate
python -m experiments.run_temperature_prediction \
    dataset=temperature model=tcn \
    optimizer.hparams.lr=0.001

# Adjust hidden size
python -m experiments.run_temperature_prediction \
    dataset=temperature model=enhanced_rnn \
    model.hparams.hidden_size=256

# Change horizon and window
python -m experiments.run_temperature_prediction \
    dataset=temperature model=improved_stgnn \
    window=12 horizon=48
```

### Loss Functions
```bash
# Use MAE loss (default for point predictions)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn loss_fn=mae

# Use Energy Score (CRPS) for probabilistic predictions
python -m experiments.run_temperature_prediction dataset=temperature model=tcn loss_fn=ens
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `batch_size=16`
- Reduce hidden size: `model.hparams.hidden_size=64`
- Use CPU: `accelerator=cpu`

### Slow Training
- Reduce window size: `window=6`
- Use fewer layers: `model.hparams.num_layers=2`
- Limit training batches for testing: `train_batches=0.1`

### Checkpoint Loading Error
If you get errors with `=` in checkpoint paths, rename them:
```bash
cd logs/Temperature/tcn/2024-11-24/10-30-45/
mv "epoch=50-step=12345.ckpt" "epoch_50-step_12345.ckpt"
```

## Project Structure

```
peakweather-wind-forecasting/
├── experiments/
│   ├── run_wind_prediction.py          # Wind forecasting (original)
│   └── run_temperature_prediction.py   # Temperature forecasting (NEW)
├── lib/
│   ├── datasets/peakweather.py         # Dataset loader
│   ├── metrics/                        # MAE, CRPS, Energy Score
│   └── nn/
│       ├── models/
│       │   ├── temperature_models.py   # TCN, EnhancedRNN, ImprovedSTGNN (NEW)
│       │   ├── baselines/              # Persistence, ICON
│       │   └── learnable_models/       # AttentionLongTermSTGNN, etc.
│       └── predictors/predictor.py     # Training loop
├── config/
│   ├── dataset/
│   │   ├── temperature.yaml            # Weather stations only (NEW)
│   │   └── temperature_all_stations.yaml  # All stations (NEW)
│   └── model/
│       ├── tcn.yaml                    # Model0
│       ├── enhanced_rnn.yaml           # Model1
│       ├── improved_stgnn.yaml         # Model2
│       └── attn_longterm.yaml          # Model3
├── scripts/
│   ├── run_multiseed.py                # Multi-seed runner (NEW)
│   └── run_all_deliverables.sh         # Full pipeline (NEW)
└── logs/                               # Training outputs
```

## Tips for Success

1. **Start Small**: Run a quick test with reduced epochs and batches:
   ```bash
   python -m experiments.run_temperature_prediction \
       dataset=temperature model=tcn \
       epochs=5 train_batches=0.1
   ```

2. **Monitor Training**: Use MLflow UI to track progress in real-time

3. **Save Results**: Export metrics from MLflow for your report:
   ```python
   import mlflow
   client = mlflow.tracking.MlflowClient()
   experiment = client.get_experiment_by_name("Temperature_tcn_seed0")
   runs = client.search_runs(experiment.experiment_id)
   ```

4. **Hyperparameter Tuning**: As required by Deliverable 2, tune:
   - Hidden units
   - Number of layers
   - Dropout probabilities
   - Learning rate
   - Scheduler algorithm

5. **Compare Models**: Use MLflow to compare metrics across models side-by-side

## References

- **Paper**: [PeakWeather Dataset](https://arxiv.org/abs/2506.13652)
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/MeteoSwiss/PeakWeather)
- **Library**: [PeakWeather GitHub](https://github.com/meteoswiss/peakweather)
- **Framework**: [TorchSpatiotemporal](https://torch-spatiotemporal.readthedocs.io)

## Contact

For questions about the project:
- Michele Cattaneo (michele.cattaneo@meteoswiss.ch)
- Daniele Zambon (daniele.zambon@usi.ch)
