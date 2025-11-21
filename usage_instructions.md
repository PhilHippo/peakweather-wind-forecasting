# Temperature Prediction - Usage Guide

This guide explains how to train and test temperature prediction models using the `run_temperature_prediction.py` script.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Training and Testing Models](#Training-and-Testing-Models)
- [MLflow Tracking](#mlflow-tracking)
- [Configuration](#configuration)
- [Available Models](#available-models)

## Directory Structure


```
peakweather-wind-forecasting/
├── logs/                          # Training logs and checkpoints
│   └── Temperature/               # Dataset name
│       └── tcn/                   # Model name
│       └── enhanced_rnn/
│       └── improved_stgnn/
├── mlruns/                        # MLflow tracking data
│   └── 0/                         # Experiment ID
│       └── <run_id>/              # Individual run data
│           ├── artifacts/         # Model artifacts
│           ├── metrics/           # Training metrics
│           └── params/            # Hyperparameters
│
└── config/                        # Dataset and Models configuration files
    ├── dataset/
    │   └── temperature.yaml      
    └── model/
        ├── tcn.yaml               
        ├── enhanced_rnn.yaml      
        └── improved_stgnn.yaml   
```

  - Best model checkpoint is saved based on validation metric


## Training and Testing Models

### Basic Training Command

Train a model with default settings:

```bash
python -m experiments.run_temperature_prediction dataset=temperature model=tcn
```

### Training with Custom Configuration

Override any configuration parameter:

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    epochs=100 \
    batch_size=64 \
    optimizer.hparams.lr=0.0005
```

### Training with Energy Score Loss

Use Energy Score (probabilistic) loss function:

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    loss_fn=ens \
    sampling.mc_samples_train=16 \
    sampling.mc_samples_eval=11 \
    sampling.mc_samples_test=100
```



### Test with Saved Checkpoint

Load a trained model and run evaluation on the test set:

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    load_model_path=logs/Temperature/tcn/2025-11-20/17-53-24/epoch_61-step_118792.ckpt
```


**Note**: When `load_model_path` is specified:
- Training is **skipped**
- Model is loaded from checkpoint
- Test set evaluation runs automatically
- All metrics are computed and logged

## MLflow Tracking

To view experiment tracking results, start the MLflow UI:

```bash
mlflow ui --port 5000
```

Then open your browser to: `http://127.0.0.1:5000` (or your custom port)

## Configuration

### Dataset Configuration

Edit `config/dataset/temperature.yaml` to modify:
- Target channels (e.g., `temperature`)
- Covariate channels (e.g., `other` weather variables)
- Connectivity settings (graph structure)
- Extended topographical variables

### Model Configuration

Edit `config/model/{model_name}.yaml` to modify:
- Hidden size
- Number of layers
- Dropout rate
- Kernel size (for TCN)
- etc.

### Training Configuration

Edit `config/default.yaml` or override via command line:
- `epochs`: Maximum training epochs
- `patience`: Early stopping patience
- `batch_size`: Batch size for training
- `loss_fn`: Loss function (`mae` or `ens`)
- `optimizer.hparams.lr`: Learning rate

## Available Models

### TCN (Temporal Convolutional Network)

TODO

### RNN
TODO

### STGNN
TODO

## Example Workflow

### 1. Train a Model

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    epochs=200 \
    batch_size=32 \
    loss_fn=ens
```
### 2. Test the Model

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    load_model_path=logs/Temperature/tcn/2025-11-20/17-53-24/epoch_61-step_118792.ckpt
```

## Troubleshooting

if you get mismatched input '=' expecting <EOF>, just change the ckpt "=" signs to something else, like "_". TODO fix this issue
