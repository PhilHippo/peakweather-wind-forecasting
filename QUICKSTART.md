# Quick Start: Temperature Forecasting

## 🚀 Get Started in 5 Minutes

### Step 1: Setup Environment
```bash
# Create and activate conda environment
conda env create -f conda_env.yml
conda activate peakweather

# For Apple Silicon (M1/M2/M3):
pip install torch-geometric
```

### Step 2: Verify Installation
```bash
# Run quick test (5-10 minutes)
./scripts/quick_test.sh
```

### Step 3: Train Your First Model
```bash
# Train TCN model (Model0)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# Train Enhanced RNN (Model1)
python -m experiments.run_temperature_prediction dataset=temperature model=enhanced_rnn

# Train Improved STGNN (Model2)
python -m experiments.run_temperature_prediction dataset=temperature model=improved_stgnn

# Train Attention STGNN (Model3)
python -m experiments.run_temperature_prediction dataset=temperature model=attn_longterm
```

### Step 4: View Results
```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser to: http://127.0.0.1:5000
```

---

## 📚 Full Documentation

For comprehensive documentation, see:
- **[TEMPERATURE_FORECASTING_GUIDE.md](TEMPERATURE_FORECASTING_GUIDE.md)** - Complete usage guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[CLAUDE.md](CLAUDE.md)** - Project overview

---

## 🎯 Run All Deliverables

To run all experiments required for the project:

```bash
./scripts/run_all_deliverables.sh
```

**WARNING**: This runs all models with 5 seeds each and takes many hours!

---

## 📊 What Gets Evaluated

### Deliverable 1: Models Implemented
✅ Model0: TCN (Temporal Convolutional Network)
✅ Model1: Enhanced RNN with Embeddings
✅ Model2: Improved STGNN
✅ Model3: Attention-based Long-term STGNN

### Deliverable 2: Metrics Computed
✅ MAE (Mean Absolute Error)
✅ CRPS (via Energy Score)
✅ Time-specific metrics at t=1,3,6,12,18,24 hours
✅ 5 different seeds per model
✅ Weather stations only

### Deliverable 3: Rain Gauges
✅ Train on all, test on weather stations
✅ Train and test on all (separate metrics)

---

## 💡 Tips

1. **Start Small**: Use `epochs=5 train_batches=0.1` for quick testing
2. **Monitor Progress**: Keep MLflow UI open during training
3. **Save Time**: Run models in parallel on different machines/GPUs
4. **Hyperparameter Tuning**: Override config values from command line
5. **Debug**: Check `logs/` directory for checkpoints and errors

---

## 🔧 Common Commands

```bash
# Train with custom learning rate
python -m experiments.run_temperature_prediction dataset=temperature model=tcn optimizer.hparams.lr=0.001

# Train with probabilistic loss
python -m experiments.run_temperature_prediction dataset=temperature model=tcn loss_fn=ens

# Train with 5 seeds automatically
python scripts/run_multiseed.py --model tcn --seeds 5

# Train on all stations (rain gauges + weather)
python -m experiments.run_temperature_prediction dataset=temperature_all_stations model=improved_stgnn

# Load and test existing checkpoint
python -m experiments.run_temperature_prediction dataset=temperature model=tcn load_model_path=logs/Temperature/tcn/2024-11-24/10-30-45/epoch_50-step_12345.ckpt
```

---

## 📞 Support

For questions:
- Michele Cattaneo: michele.cattaneo@meteoswiss.ch
- Daniele Zambon: daniele.zambon@usi.ch

For bugs/issues, check the documentation first!

---

**Implementation Status**: ✅ Complete and ready to run
