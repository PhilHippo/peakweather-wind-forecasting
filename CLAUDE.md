# PeakWeather Wind Forecasting Project

## Project Overview

This is a **spatiotemporal deep learning** research project for **wind forecasting** using the [PeakWeather dataset](https://huggingface.co/datasets/MeteoSwiss/PeakWeather). The dataset contains high-resolution meteorological observations from 302 Swiss weather stations, collected every 10 minutes from January 2017 to March 2025.

The project implements and evaluates various forecasting models including:
- Spatiotemporal Graph Neural Networks (STGNNs)
- Recurrent Neural Networks (RNNs)
- Baseline models (Persistence, ICON NWP forecasts)

**Paper**: [PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning](https://arxiv.org/abs/2506.13652)

## Tech Stack

- **Deep Learning**: PyTorch 2.1, PyTorch Lightning 2.1
- **Graph Neural Networks**: PyTorch Geometric 2.4
- **Spatiotemporal Framework**: [torch-spatiotemporal](https://torch-spatiotemporal.readthedocs.io/) 0.9.5
- **Configuration**: Hydra + OmegaConf
- **Experiment Tracking**: MLflow
- **Dataset**: PeakWeather library
- **Python**: 3.10

## Directory Structure

```
peakweather-wind-forecasting/
├── lib/                              # Core library code
│   ├── datasets/                     # Dataset loading and processing
│   │   └── peakweather.py           # PeakWeather dataset wrapper
│   ├── metrics/                      # Custom metrics
│   │   ├── crps.py                  # CRPS metric
│   │   ├── wind.py                  # Wind-specific metrics (direction, speed)
│   │   ├── point_predictions.py
│   │   └── sample_metrics.py        # Probabilistic metrics
│   ├── nn/                          # Neural network models
│   │   ├── layers/                  # Custom layers
│   │   ├── models/                  # Model implementations
│   │   │   ├── baselines/          # Persistence, ICON
│   │   │   └── learnable_models/   # RNNs, STGNNs
│   │   └── predictors/             # Training wrappers
│   │       └── predictor.py        # Predictor and SamplingPredictor
│   └── nn/utils.py
│
├── experiments/                      # Experiment scripts
│   └── run_wind_prediction.py       # Main training/testing script
│
├── config/                          # Hydra configuration files
│   ├── default.yaml                 # Default training config
│   ├── dataset/                     # Dataset configurations
│   │   ├── wind.yaml
│   │   ├── wind_1d.yaml            # 1-day horizon
│   │   └── wind_6h.yaml            # 6-hour horizon
│   ├── model/                       # Model configurations
│   │   ├── tts_imp.yaml            # Time-then-space STGNN
│   │   ├── rnn_glob.yaml
│   │   ├── rnn_emb.yaml
│   │   ├── pers_st.yaml            # Spatiotemporal persistence
│   │   ├── pers_day.yaml
│   │   ├── icon.yaml               # NWP baseline
│   │   ├── attn_longterm.yaml      # Attention-based STGNN
│   │   └── ...
│   └── lr_scheduler/
│       └── multistep.yaml
│
├── logs/                            # Training logs and checkpoints (created at runtime)
├── mlruns/                          # MLflow tracking data (created at runtime)
├── conda_env.yml                    # Environment dependencies
├── README.md                        # Project documentation
└── usage_instructions.md            # Detailed usage guide
```

## Key Concepts

### Spatiotemporal Forecasting
The project forecasts **wind components (u, v)** across multiple weather stations over time horizons (e.g., 1-24 hours ahead). Models must capture both:
- **Temporal dependencies**: How wind evolves over time
- **Spatial dependencies**: How wind conditions at one station relate to nearby stations

### Graph Structure
Weather stations are represented as nodes in a graph, with edges based on:
- **Distance**: Geographic proximity
- **Topography**: Elevation, terrain features
- Configurable connectivity in `config/dataset/*.yaml`

### Wind Representation
Wind is represented as **two components**:
- **u**: East-West component (m/s)
- **v**: North-South component (m/s)

These can be converted to:
- **Speed**: sqrt(u² + v²)
- **Direction**: arctan2(v, u)

### Probabilistic Forecasting
Some models produce probabilistic forecasts (sampling-based):
- **Loss**: Energy Score (ENS)
- **Predictor**: `SamplingPredictor` (generates multiple samples)
- **Metrics**: Sample MAE, Energy Score, Direction Energy Score

Point forecasting uses:
- **Loss**: MAE
- **Predictor**: `Predictor`
- **Metrics**: MAE, MSE, Direction MAE, Speed MAE

## Running Experiments

### Basic Training
```bash
python -m experiments.run_wind_prediction dataset=wind_1d model=<MODEL_NAME>
```

### Available Models
- `tts_imp`: Time-then-space STGNN (spatiotemporal graph neural network)
- `attn_longterm`: Attention-based long-term STGNN
- `rnn_glob`: Global-local RNN
- `rnn_emb`: RNN with embeddings
- `pers_st`: Spatiotemporal persistence baseline
- `pers_day`: Daily persistence baseline
- `icon`: ICON NWP forecasts (operational baseline)

### Configuration Override Examples
```bash
# Change learning rate
python -m experiments.run_wind_prediction dataset=wind_1d model=tts_imp optimizer.hparams.lr=0.001

# Use Energy Score loss (probabilistic)
python -m experiments.run_wind_prediction dataset=wind_1d model=tts_imp loss_fn=ens

# Load and test a checkpoint
python -m experiments.run_wind_prediction dataset=wind_1d model=tts_imp \
    load_model_path=logs/Wind/tts_imp/2024-11-20/10-30-45/epoch_50-step_12345.ckpt
```

### MLflow Tracking
Start MLflow UI to view experiments:
```bash
mlflow ui --port 5000
# Open browser to http://127.0.0.1:5000
```

## Configuration System (Hydra)

The project uses **Hydra** for hierarchical configuration:

1. **Base config**: `config/default.yaml` - training hyperparameters
2. **Dataset config**: `config/dataset/*.yaml` - data settings
3. **Model config**: `config/model/*.yaml` - model architectures
4. **Override from CLI**: `python -m experiments.run_wind_prediction key=value`

### Important Config Parameters
- `epochs`: Maximum training epochs
- `patience`: Early stopping patience
- `batch_size`: Training batch size
- `horizon`: Forecast horizon (timesteps)
- `window`: Input history length (timesteps)
- `loss_fn`: Loss function (`mae` or `ens`)
- `optimizer.hparams.lr`: Learning rate
- `dataset.hparams`: Target variables, covariates
- `dataset.connectivity`: Graph connectivity settings

## Important Code Patterns

### Model Registration
Models are registered in `experiments/run_wind_prediction.py:get_model_class()`:
```python
def get_model_class(model_str):
    if model_str == 'tts_imp':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'rnn_glob':
        model = models.GlobalLocalRNNModel
    # ... etc
```

### Predictor Selection
- If `loss_fn == "ens"`: Uses `SamplingPredictor` (probabilistic)
- Otherwise: Uses `Predictor` (point predictions)

### Data Scaling
- Input features are scaled using the specified scaler in config
- Wind components (u, v) are typically standard-scaled
- Mask is used to handle missing values

### Model Checkpointing
- Best model saved based on validation metric
- Path format: `logs/{dataset}/{model}/{date}/{time}/epoch_{n}-step_{s}.ckpt`
- **Known issue**: Checkpoint paths with `=` signs can cause parsing errors (workaround: rename with `_`)

## Metrics

### Point Prediction Metrics
- `mae`: Mean Absolute Error (overall)
- `mae_Xh`: MAE at X hours ahead (e.g., `mae_6h`)
- `mse`: Mean Squared Error
- `dir_mae`: Direction MAE (wind direction error)
- `speed_mae`: Speed MAE (wind speed error)

### Probabilistic Metrics (with `loss_fn=ens`)
- `smae`: Sample MAE
- `smse`: Sample MSE
- `ens`: Energy Score
- `dir_ens`: Direction Energy Score
- `speed_ens`: Speed Energy Score
- All available with `_Xh` horizon-specific variants

All metrics are **masked** to handle missing values in the data.

## Important Notes

1. **GPU/CPU**: Automatically detects GPU availability; falls back to CPU
2. **M1/M2/M3 Macs**: PyTorch Geometric must be installed via pip (see README)
3. **Missing Values**: Handled via masks throughout the pipeline
4. **Persistence Models**: Skip training, directly evaluated on test set
5. **ICON Model**: Uses pre-computed NWP forecasts from dataset
6. **NWP Test Set**: Special evaluation mode with extended NWP variables (set `nwp_test_set=true`)

## Development Conventions

- Models inherit from `torch.nn.Module` and implement standard forward pass
- Custom metrics inherit from `tsl.metrics.Metric` or `lib.metrics.SampleMetric`
- Predictors handle training loop via PyTorch Lightning
- Dataset wraps PeakWeather library with project-specific preprocessing
- All paths should be absolute when possible

## Common Tasks

### Adding a New Model
1. Implement model class in `lib/nn/models/`
2. Register in `experiments/run_wind_prediction.py:get_model_class()`
3. Create config file in `config/model/your_model.yaml`
4. Run: `python -m experiments.run_wind_prediction dataset=wind_1d model=your_model`

### Debugging Training
1. Check MLflow UI for metrics/logs
2. Inspect checkpoints in `logs/` directory
3. Reduce batch size or use `limit_train_batches` for quick iteration
4. Use `load_model_path` to skip training and test directly

### Modifying Dataset
1. Edit `config/dataset/*.yaml` for data settings
2. Modify `lib/datasets/peakweather.py` for preprocessing
3. Check data shapes with print statements in `run_wind_prediction.py`
