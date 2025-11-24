# Project Deliverables Checklist

## GDL Fall 2025 - PeakWeather Temperature Forecasting

This checklist tracks all mandatory and optional deliverables for the project.

---

## ✅ DELIVERABLE 1: Implementation

### Model0: Non-Graph Baseline (RNN and/or TCN)
- [x] **Implemented**: TCN (Temporal Convolutional Network)
- [x] **Location**: `lib/nn/models/temperature_models.py:TCNModel`
- [x] **Config**: `config/model/tcn.yaml`
- [x] **Differs from repo**: Yes (repo doesn't have this exact TCN architecture)
- [x] **Description**: Stacked dilated temporal convolutions with residual connections
- [x] **Motivation**:
  - Dilated convolutions provide exponentially growing receptive field
  - Batch normalization and residuals prevent degradation
  - Processes each station independently to isolate temporal patterns
  - Provides strong baseline without graph structure

### Model1: Baseline RNN with Embeddings
- [x] **Implemented**: EnhancedRNNModel
- [x] **Location**: `lib/nn/models/temperature_models.py:EnhancedRNNModel`
- [x] **Config**: `config/model/enhanced_rnn.yaml`
- [x] **Checked/Modified**: Yes, enhanced with attention mechanism
- [x] **Motivation for Changes**:
  - Added multi-head self-attention (4 heads) for better temporal modeling
  - Node embeddings capture station-specific characteristics (altitude, location effects)
  - GRU provides efficient sequential modeling with fewer parameters than LSTM
  - No graph edges to isolate embedding effectiveness

### Model2: Baseline STGNN
- [x] **Implemented**: ImprovedSTGNN
- [x] **Location**: `lib/nn/models/temperature_models.py:ImprovedSTGNN`
- [x] **Config**: `config/model/improved_stgnn.yaml`
- [x] **Checked/Modified**: Yes, added skip connections and layer norms
- [x] **Motivation for Changes**:
  - Temporal GRU encoder captures time dependencies
  - DiffConv (k=2) propagates information across graph (2-hop diffusion)
  - Layer normalization stabilizes training
  - Skip connections preserve information flow
  - Node embeddings integrated at both temporal and spatial stages

### Model3: Competitive STGNN from Literature
- [x] **Implemented**: AttentionLongTermSTGNN
- [x] **Location**: `lib/nn/models/learnable_models/attention_longterm_stgnn.py`
- [x] **Config**: `config/model/attn_longterm.yaml`
- [x] **Literature Basis**: Patch-based attention for time series (TST, PatchTST)
- [x] **Motivation**:
  - Patch-based encoding (12h patches) captures sub-daily cycles
  - Long-term attention (up to 168h = 7 days) models weekly patterns
  - Spatial-temporal aware convolution combines local and global patterns
  - Adaptive graph construction learns optimal connectivity
  - Multi-scale architecture (long-term + short-term)
  - Proven effective in spatiotemporal forecasting literature

---

## ✅ DELIVERABLE 2: Performance Assessment

### Metrics Implementation
- [x] **MAE**: Overall and time-specific (t=1,3,6,12,18,24)
- [x] **CRPS**: Energy Score metric (overall and time-specific)
- [x] **Sample Metrics**: SampleMAE, SampleMSE for ensemble evaluation
- [x] **Location**: `experiments/run_temperature_prediction.py` lines 186-207

### NWP Baseline
- [x] **ICON Model**: Implemented for comparison
- [x] **Config**: `config/model/icon.yaml`
- [x] **Note**: Uses 11-member ensemble issued every 3 hours
- [x] **Metrics**: SampleMAE for median, EnergyScore for CRPS

### Dataset Configuration
- [x] **Weather Stations Only**: `config/dataset/temperature.yaml`
- [x] **Test Set**: Starts April 1, 2024 (`test_start: "2024-04-01"`)
- [x] **All Variables**: Uses all 8 sensor types as covariates
- [x] **Mask Handling**: Missing observations properly masked

### Multi-Seed Training
- [x] **5 Seeds Required**: Implemented via `scripts/run_multiseed.py`
- [x] **Automation**: Can run all models with 5 seeds automatically
- [x] **Tracking**: Each seed gets unique experiment name in MLflow

### Hyperparameter Tuning Support
- [x] **Configurable**: All hyperparameters in YAML configs
- [x] **Override**: Command-line overrides supported
- [x] **Parameters Tunable**:
  - Hidden units (`model.hparams.hidden_size`)
  - Number of layers (`model.hparams.num_layers`, `rnn_layers`, `gnn_layers`)
  - Dropout (`model.hparams.dropout`)
  - Learning rate (`optimizer.hparams.lr`)
  - Scheduler (`lr_scheduler`)
  - Window and horizon sizes

---

## ✅ DELIVERABLE 3: Adding Low-Quality Sensors

### Dataset with Rain Gauges
- [x] **Configuration**: `config/dataset/temperature_all_stations.yaml`
- [x] **Setting**: `station_type: null` (includes both weather + rain gauges)

### Experiment 1: Train on All, Test on Weather Stations Only
- [x] **Script**: `experiments/run_temperature_prediction.py`
- [x] **Question**: Does forecasting improve when using rain gauges as input?
- [x] **Approach**:
  ```bash
  python -m experiments.run_temperature_prediction \
      dataset=temperature_all_stations \
      model=improved_stgnn
  ```
- [x] **Analysis**: Compare MAE/CRPS on weather stations with/without rain gauge data

### Experiment 2: Train and Test on All Nodes
- [x] **Script**: Same as Experiment 1
- [x] **Question**: Difference in prediction error between weather stations and rain gauges?
- [x] **Analysis**: Compute separate metrics for each node type
- [x] **Expected**: Lower error on weather stations (higher quality sensors)

---

## 📊 BONUS TASKS (Optional)

### Bonus Task 1: Graph Sensitivity
- [ ] Test different graph hyperparameters (theta, threshold)
- [ ] Construct alternative graphs (e.g., topological, learned)
- [ ] Compare model performance across graph types
- [ ] **Suggested Models**: ImprovedSTGNN, AttentionLongTermSTGNN

### Bonus Task 2: Learnable Graph
- [ ] Implement graph learning mechanism
- [ ] Analyze learned graphs for interpretability
- [ ] Check consistency across training runs
- [ ] **Approach**: Attention-based edge weights or graph structure learning

### Bonus Task 3: External Weather Variables
- [ ] Fetch additional variables (e.g., ERA5 reanalysis)
- [ ] Integrate into pipeline
- [ ] Compare with/without additional data
- [ ] **Suggestions**: Satellite data, radar, climate indices

### Bonus Task 4: Foundation Models
- [ ] Test TimesFM, Chronos, Moirai, or similar
- [ ] Fair comparison with custom models
- [ ] Discuss advantages/limitations
- [ ] **Note**: Zero-shot vs fine-tuned comparison

---

## 🚀 AUTOMATION & TOOLS

### Execution Scripts
- [x] **Multi-Seed Runner**: `scripts/run_multiseed.py`
- [x] **Master Script**: `scripts/run_all_deliverables.sh`
- [x] **Quick Test**: `scripts/quick_test.sh`

### Documentation
- [x] **Quick Start**: `QUICKSTART.md`
- [x] **Full Guide**: `TEMPERATURE_FORECASTING_GUIDE.md`
- [x] **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- [x] **This Checklist**: `PROJECT_DELIVERABLES_CHECKLIST.md`

### Experiment Tracking
- [x] **MLflow Integration**: Automatic logging of all metrics
- [x] **Checkpoint Management**: Best model saved per run
- [x] **Visualization**: Training curves, metrics dashboard

---

## 📝 FINAL DELIVERABLES FOR SUBMISSION

### 1. Code
- [x] All models implemented and tested
- [x] Training scripts functional
- [x] Configuration files complete
- [x] Documentation comprehensive

### 2. Report (To Do)
- [ ] Model architectures and motivations
- [ ] Experimental setup and hyperparameters
- [ ] Results tables (MAE, CRPS at t=1,3,6,12,18,24)
- [ ] Comparison with NWP baseline
- [ ] Analysis of rain gauges impact
- [ ] Visualizations (training curves, predictions, error maps)
- [ ] Discussion and conclusions

### 3. Presentation (To Do)
- [ ] Project overview slides
- [ ] Model descriptions
- [ ] Results summary
- [ ] Key insights
- [ ] Q&A preparation

### 4. Quiz (To Do)
- [ ] Complete final quiz on iCorsi

---

## ⏱️ RECOMMENDED WORKFLOW

### Week 1: Setup & Initial Experiments
1. ✅ Environment setup
2. ✅ Run quick tests to verify installation
3. ⏳ Train initial models (1 seed each)
4. ⏳ Review results in MLflow
5. ⏳ Preliminary hyperparameter exploration

### Week 2: Full Experiments
1. ⏳ Run all 4 models with 5 seeds (Deliverable 2)
2. ⏳ Run NWP baseline
3. ⏳ Monitor training and adjust hyperparameters
4. ⏳ Collect results

### Week 3: Rain Gauges & Analysis
1. ⏳ Run Deliverable 3 experiments
2. ⏳ Analyze results comprehensively
3. ⏳ Create visualizations
4. ⏳ Start report writing

### Week 4: Report & Presentation
1. ⏳ Finalize report
2. ⏳ Create presentation slides
3. ⏳ Practice presentation
4. ⏳ Complete quiz
5. ⏳ Submit all deliverables

---

## ✅ READY TO RUN

**All mandatory deliverables are implemented and ready for execution.**

Next steps:
1. Setup environment: `conda env create -f conda_env.yml`
2. Run quick test: `./scripts/quick_test.sh`
3. Start experiments: `./scripts/run_all_deliverables.sh`
4. Monitor progress: `mlflow ui --port 5000`
5. Analyze and report

---

## 📞 CONTACT

**Project Supervisors**:
- Michele Cattaneo (MeteoSwiss): michele.cattaneo@meteoswiss.ch
- Daniele Zambon (USI): daniele.zambon@usi.ch

**Resources**:
- Paper: https://arxiv.org/abs/2506.13652
- Dataset: https://huggingface.co/datasets/MeteoSwiss/PeakWeather
- Library: https://github.com/meteoswiss/peakweather
- Framework: https://torch-spatiotemporal.readthedocs.io

---

**Last Updated**: 2024-11-24
**Status**: ✅ Implementation Complete - Ready for Experiments
