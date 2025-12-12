# PeakWeather Temperature Forecasting

Spatiotemporal deep learning models for temperature prediction using the [PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) dataset from MeteoSwiss.

## Quick Start

### 1. Setup Environment

```bash
conda env create -f conda_env.yml
conda activate peakweather-env
```

### 2. Train a Model

```bash
python -m experiments.run_temperature_prediction dataset=temperature model=tcn
```

### 3. View Results

```bash
mlflow ui --port 5000
# Open http://127.0.0.1:5000 in your browser
```

## Project Structure

```
peakweather-wind-forecasting/
├── config/                    # Configuration files (Hydra)
│   ├── default.yaml           # Training defaults (epochs, batch size, etc.)
│   ├── dataset/
│   │   └── temperature.yaml   # Dataset settings (features, splits, graph)
│   └── model/                 # Model hyperparameters
│       ├── tcn.yaml           # Temporal Convolutional Network
│       ├── rnn.yaml           # GRU-based RNN
│       ├── stgnn.yaml         # Spatiotemporal Graph Neural Network
│       ├── attn_longterm.yaml # Attention-based STGNN
│       ├── naive.yaml         # Naive baseline
│       ├── moving_avg.yaml    # Moving average baseline
│       └── icon.yaml          # NWP baseline
│
├── experiments/
│   └── run_temperature_prediction.py  # Main training script
│
├── lib/                       # Core library code
│   ├── datasets/
│   │   └── peakweather.py     # Dataset loader
│   ├── metrics/               # Evaluation metrics (MAE, Energy Score)
│   ├── nn/
│   │   ├── models/
│   │   │   ├── learnable_models/  # Neural networks (TCN, RNN, STGNN)
│   │   │   └── baselines/         # Simple baselines (Naive, MovingAvg, ICON)
│   │   └── predictors/
│   │       └── predictor.py   # Training wrapper
│   └── visualization/         # Plotting utilities
│
├── logs/                      # Training outputs and checkpoints
└── docs/                      # Reference papers
```

## Available Models

| Model           | Command               | Description                                   |
| --------------- | --------------------- | --------------------------------------------- |
| TCN             | `model=tcn`           | Temporal Convolutional Network                |
| RNN             | `model=rnn`           | GRU-based recurrent network                   |
| STGNN           | `model=stgnn`         | Graph Neural Network for spatial dependencies |
| Attention STGNN | `model=attn_longterm` | STGNN with long-term attention                |
| Naive           | `model=naive`         | Repeats last observation (baseline)           |
| Moving Avg      | `model=moving_avg`    | Average of last 24 hours (baseline)           |
| ICON NWP        | `model=icon`          | MeteoSwiss numerical weather prediction       |

## Common Commands

### Training Examples

```bash
# Train TCN (default)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# Train STGNN with custom settings
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=stgnn \
    epochs=100 \
    batch_size=64 \
    optimizer.hparams.lr=0.0005

# Train with specific random seed
python -m experiments.run_temperature_prediction dataset=temperature model=tcn seed=42
```

### Probabilistic Forecasting

Use Energy Score loss for uncertainty estimation:

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    loss_fn=ens \
    sampling.mc_samples_train=16
```

### Evaluate Saved Model

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    load_model_path=path/to/checkpoint.ckpt
```

### Run Baselines (No Training)

```bash
python -m experiments.run_temperature_prediction dataset=temperature model=naive
python -m experiments.run_temperature_prediction dataset=temperature model=moving_avg
python -m experiments.run_temperature_prediction dataset=temperature model=icon
```

## Dataset Overview

- **Source**: PeakWeather (302 weather stations)
- **Variables**: temperature, humidity, precipitation, sunshine, pressure, wind
- **Resolution**: Hourly (resampled from 10-minute)
- **Time span**: Jan 2017 - Mar 2025

### Data Splits

| Split      | Period               |
| ---------- | -------------------- |
| Train      | Until Dec 31, 2023   |
| Validation | Jan 1 - Mar 31, 2024 |
| Test       | Apr 1, 2024 onwards  |

### Forecasting Task

- **Input window**: 72 hours (3 days)
- **Forecast horizon**: 24 hours

## Configuration

Override any config value from command line:

```bash
# Change training parameters
python -m experiments.run_temperature_prediction epochs=50 batch_size=32

# Change model hyperparameters
python -m experiments.run_temperature_prediction model=stgnn model.hparams.hidden_size=64

# Change dataset settings
python -m experiments.run_temperature_prediction window=48 horizon=12
```

## Citation

If you use this code or the PeakWeather dataset, please cite:

```bibtex
@misc{zambon2025peakweather,
  title={PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning},
  author={Zambon, Daniele and Cattaneo, Michele and Marisca, Ivan and Bhend, Jonas and Nerini, Daniele and Alippi, Cesare},
  year={2025},
  eprint={2506.13652},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.13652},
}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
