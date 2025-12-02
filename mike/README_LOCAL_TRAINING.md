# Local Model Training and Inference

Complete guide for training and running streamflow forecasting models locally using AutoGluon (the same technology SageMaker Canvas uses).

## Overview

This setup replaces AWS SageMaker with local training and inference:

- **Training**: `train_local.py` - Trains AutoGluon model on your data
- **Inference**: `invoke_local.py` - Uses trained model for forecasting
- **Data Prep**: `data_prep.py` - Downloads and prepares data

## Quick Start

### 1. Install Dependencies

```bash
cd /home/steve/sync/streamflow/mike

# Activate your virtual environment
source ../.venv/bin/activate

# Install AutoGluon (if not already installed)
pip install autogluon
```

### 2. Prepare Data

Run the data preparation script to download the latest data:

```bash
python data_prep.py
```

This creates:
- `Lamar_training_new.csv` - Historical data for training
- `new_future_data.csv` - Features for forecasting
- `forecast_dates.csv` - Dates to forecast
- `screened_stream.csv` - Stream observations

### 3. Train the Model

Train a local AutoGluon model:

```bash
python train_local.py
```

**What it does:**
- Trains multiple model types (Random Forest, XGBoost, Neural Networks, etc.)
- Automatically selects the best model
- Saves trained model to `./lamar_model/`
- Shows leaderboard and feature importance
- **Training time**: ~10 minutes (configurable)

**Output:**
```
============================================================
LOCAL MODEL TRAINING (AutoGluon)
============================================================

Training data shape: (13461, 20)
Features: ['cfslag7', 'lagged_delta', 'second_d', ...]

Training model with AutoGluon...
Model will be saved to: ./lamar_model

[Training progress...]

============================================================
TRAINING COMPLETE!
============================================================

MODEL LEADERBOARD:
                model  score_test  score_val  ...
0  WeightedEnsemble_L2    12.3456   13.4567  ...
1      LightGBM         13.2345   14.3456  ...
...

FEATURE IMPORTANCE:
              importance
cfslag7            0.345
lagged_delta       0.234
ACCUM_SNOWPACK...  0.123
...
```

### 4. Run Forecasts

Generate forecasts using the trained model:

```bash
python invoke_local.py
```

**What it does:**
- Loads the trained model from `./lamar_model/`
- Makes predictions for future dates
- Generates forecast plot
- Updates prediction CSV files

**Output:**
```
============================================================
LOCAL FORECAST GENERATION (no AWS integration)
============================================================

Loading locally trained AutoGluon model...
✓ Model loaded successfully
  Model type: AutoGluon TabularPredictor
  Problem type: regression

Making predictions for 7 future timesteps...
✓ Successfully generated 7 predictions
  Prediction range: 125.34 - 178.92 cfs

[Forecast plot saved]
```

## Files Generated

### Training Outputs

- `lamar_model/` - Trained AutoGluon model directory
  - Contains ensemble of models
  - Model leaderboard
  - Feature importance
  - Training metrics

### Forecast Outputs

- `latest_lamar_prediction_MM-DD-YY.png` - Forecast visualization
- `full_latest_predictions.csv` - Complete prediction history

## Customizing Training

Edit `train_local.py` to customize training:

### Training Time

```python
predictor.fit(
    train_data=train_dataset,
    time_limit=600,  # Change this (seconds)
    ...
)
```

### Model Quality Presets

```python
predictor.fit(
    ...
    presets='medium_quality',  # Change this
)
```

Available presets:
- `'best_quality'` - Highest accuracy, slowest training (~1-2 hours)
- `'high_quality'` - Good balance (~30-60 minutes)
- `'medium_quality'` - Fast training (~10 minutes) **[DEFAULT]**
- `'optimize_for_deployment'` - Fastest inference

### Specific Models

To train only specific models:

```python
predictor.fit(
    train_data=train_dataset,
    hyperparameters={
        'GBM': {},  # LightGBM/XGBoost
        'RF': {},   # Random Forest
        'NN': {},   # Neural Network
    },
    time_limit=600,
)
```

## Model Performance

After training, check model performance:

```bash
python -c "
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load model
predictor = TabularPredictor.load('./lamar_model')

# Load test data
test_data = pd.read_csv('Lamar_training_new.csv')

# Evaluate
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
"
```

## Comparison: Local vs AWS SageMaker

| Aspect | AWS SageMaker | Local Training |
|--------|---------------|----------------|
| **Cost** | Pay per inference | Free (your hardware) |
| **Training Time** | ~10-60 min | ~10-60 min (configurable) |
| **Data Privacy** | Data sent to AWS | Data stays local |
| **Internet** | Required | Not required |
| **Model Updates** | Manual retraining on AWS | Quick local retraining |
| **Technology** | AutoGluon in SageMaker | AutoGluon locally |
| **Performance** | Same | Same |

## Retraining the Model

To retrain with new data:

```bash
# 1. Get latest data
python data_prep.py

# 2. Retrain model (overwrites existing)
python train_local.py

# 3. Generate new forecasts
python invoke_local.py
```

**Recommendation**: Retrain weekly or when significant new data arrives.

## Troubleshooting

### "Model not found"
- Run `train_local.py` first to create the model
- Check that `./lamar_model/` directory exists

### "AutoGluon not installed"
```bash
pip install autogluon
```

### "Out of memory during training"
- Reduce `time_limit` in `train_local.py`
- Use `presets='medium_quality'` instead of `'best_quality'`
- Reduce training data size

### "Training too slow"
- Use `presets='medium_quality'`
- Reduce `time_limit`
- Train on subset of data for testing

### "Poor prediction accuracy"
- Increase training time: `time_limit=3600` (1 hour)
- Use better preset: `presets='best_quality'`
- Check feature importance for data quality issues

## Advanced: Model Analysis

### Feature Importance

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

predictor = TabularPredictor.load('./lamar_model')
train_data = pd.read_csv('Lamar_training_new.csv')

# Get feature importance
importance = predictor.feature_importance(train_data)
print(importance)
```

### Model Inspection

```python
# Get best model name
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")

# Get all models in ensemble
models = predictor.get_model_names()
print(f"Models trained: {models}")

# Model info
info = predictor.info()
print(info)
```

### Prediction Intervals

AutoGluon can provide prediction intervals (uncertainty):

```python
# Get quantile predictions
predictions = predictor.predict(test_data, quantile_levels=[0.1, 0.5, 0.9])
# Returns: [P10, P50, P90] for each prediction
```

## Next Steps

1. **Automate**: Set up cron job to run `data_prep.py` and `invoke_local.py` daily
2. **Monitor**: Track prediction accuracy over time
3. **Tune**: Experiment with different presets and hyperparameters
4. **Ensemble**: Train multiple models and average predictions

## Support

- AutoGluon Docs: https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
- For issues with this setup, check the error messages for troubleshooting steps
