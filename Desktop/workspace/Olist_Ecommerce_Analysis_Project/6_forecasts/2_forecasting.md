# Task 2: Multi-level Demand Forecasting Models

## Overview

Task 2 is now modularized into four sub-tasks, each focusing on a different class of forecasting models and a final ensemble integration. All sub-tasks are based on Task 1's time series feature engineering results and support multi-level forecasting across product, seller, geography, and time dimensions.

---

## Sub-Tasks Structure

### 2.1 Statistical Forecasting
- **Models**: ARIMA, Prophet, VAR
- **Features**: Automatic parameter selection, holiday/seasonal effects, trend detection
- **Outputs**: `statistical_forecasts.csv`, `statistical_model_performance.csv`, model files, visualizations

### 2.2 Machine Learning Forecasting
- **Models**: XGBoost, LightGBM, Random Forest, Elastic Net
- **Features**: Advanced feature engineering, lag/rolling/cyclical features, feature importance
- **Outputs**: `ml_forecasts.csv`, `ml_model_performance.csv`, `ml_feature_importance.csv`, model files, visualizations

### 2.3 Deep Learning Forecasting (Optional)
- **Models**: GRU (default, low resource), LSTM, Transformer (user-selectable)
- **Features**: Sequence modeling, early stopping, normalization, minimal resource mode
- **Outputs**: `dl_forecasts.csv`, `dl_model_performance.csv`, model files, visualizations
- **Note**: User can choose to skip or include deep learning models at runtime for resource efficiency.

### 2.4 Model Ensemble and Optimization
- **Function**: Integrates outputs from 2.1, 2.2, 2.3 using weighted and type-based ensemble strategies
- **Features**: Performance-based weighting, simple average, type-weighted ensemble, optimal weight calculation
- **Outputs**: `ensemble_forecasts.csv`, `ensemble_model_configs.csv`, `optimal_model_weights.csv`, visualizations
- **Note**: If deep learning results are missing, ensemble will use only available models.

---

## Output Files

- `statistical_forecasts.csv`, `ml_forecasts.csv`, `dl_forecasts.csv` (optional), `ensemble_forecasts.csv`
- `statistical_model_performance.csv`, `ml_model_performance.csv`, `dl_model_performance.csv` (optional), `ensemble_model_configs.csv`, `optimal_model_weights.csv`
- `ml_feature_importance.csv`
- Model files in `output/models/`
- Visualizations in `output/visualizations/`

---

## Model & Pipeline Highlights

- **Multi-model, multi-level forecasting**: Each sub-task supports daily, weekly, monthly, and multi-dimensional aggregation.
- **Flexible deep learning**: GRU is default for low-resource, LSTM/Transformer as user options.
- **Ensemble integration**: Combines all available models for optimal accuracy.
- **Performance metrics**: MAPE, RMSE, MAE, R², feature importance, model weights.
- **Robust error handling**: Missing model outputs are tolerated in ensemble.
- **Efficient resource management**: Minimal default settings, user control over deep learning.

---

## Usage

```bash
cd week6_forecasts
# Run each sub-task as needed:
python 2.1_statistical_forecasting.py
python 2.2_ml_forecasting.py
python 2.3_dl_forecasting.py  # Optional, user can skip or select models
python 2.4_model_ensemble.py
```

---

## Next Steps

1. **Task 3**: Capacity demand prediction
2. **Task 4**: Cost-benefit analysis engine
3. **Task 5**: Risk assessment and mitigation
4. **Task 6**: Precision recommendation generator
5. **Task 7**: Recommendation validation and optimization
6. **Task 8**: Comprehensive report generator

---

## Key Optimizations

- Modular pipeline, easy to maintain and extend
- Efficient data loading and memory management
- User-selectable model complexity for deep learning
- Parallel and incremental processing support
- Comprehensive logging and error handling 