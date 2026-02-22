# Retail Sales Prediction

## Task

Predicting the number of sneakers units sold based on historical data about prices, promotions, and warehouse inventory.

## Data

Files `train.parquet` and `test.parquet` contain the following columns:

| Column | Description |
|--------|-------------|
| `nm_id` | Anonymous product identifier |
| `dt` | Date |
| `price` | Product price on that day |
| `is_promo` | Flag indicating product participation in promotion |
| `prev_leftovers` | Product inventory at the beginning of the day |
| `qty` | Number of units sold (only in train, target feature) |

Data period: 2024-07-04 -- 2025-07-07.

## Solution

### Feature Engineering

Feature generation is implemented in `features_all.py` and divided into 8 blocks, each represented by a separate function:

**Temporal Features** (`add_temporal_features`): calendar date attributes: day of week, month, weekend flags and start/end of month, linear trend, cyclic encoding via sin/cos.

**Price Features** (`add_price_features`): product price dynamics: absolute and relative change, rolling means/minimums/maximums for 7/14/30 days, discount depth from historical maximum, price percentile, days since last price change.

**Promo Features** (`add_promo_features`): promotional activity characteristics: current promo streak duration, days since last promo, promo frequency over 30 days, flags for promo start and end, interaction of promo with discount depth.

**Inventory Features** (`add_leftovers_features`): warehouse inventory dynamics: rolling averages, inventory change, 7-day trend (linear regression slope), low inventory flag.

**Sales Lags** (`add_sales_lag_features`): lagged qty values for 1/7/14/28 days, rolling means and standard deviations of sales, expanding mean. All statistics are shifted by 1 day to prevent leakage.

**Item Features** (`add_item_features`): aggregate product characteristics: average sales, average price, price volatility, number of days in data.

**Price Elasticity** (`add_elasticity_features`): rolling correlation of price and sales over 30 days, pseudo-elasticity as ratio of percentage changes in qty and price.

**Interactions** (`add_interaction_features`): pairwise products: price on promo, day of week on promo, inventory on promo.

### Data Leakage Prevention

For correct feature calculation on test data, train and test are combined into a single timeline. `qty` values in test rows are replaced with NaN (`qty_safe`), after which all lagged and rolling statistics are calculated on this masked column. This ensures that the model does not use information from the future, even if test rows are located between train rows.

### Model

CatBoostRegressor with weighted MAE loss function.

Observation weights: 7.0 for rows with `qty > 0`, 1.0 for zero sales.

Hyperparameter tuning performed via Optuna (50 trials) on the following parameters: learning rate, tree depth, L2 regularization, min data in leaf, bagging temperature, random strength.

### Experiment Tracking

All experiments are logged in MLflow. Each Optuna trial is recorded as a nested run, the final model with best parameters is saved in the parent run along with feature importance.

## Project Structure

```
.
├── README.md
├── features_all.py           # Feature generation (8 blocks)
├── train.parquet             # Training data
├── test.parquet              # Test data
└── submission.csv            # Predictions
```

## Results

| Configuration | Weighted MAE |
|---------------|-------------|
| Baseline (no additional features) | 3.1 |
| With feature engineering | 2.5 |
| With feature engineering + Optuna | 2.45 |

## Usage

### 1. Start MLflow

Before running the notebook, bring up MLflow via Docker Compose:

```bash
docker compose up -d
```

MLflow UI will be available at [http://localhost:5000](http://localhost:5000).

### 2. Run the notebook

```python
import pandas as pd
import numpy as np
from features_all import *
from catboost import CatBoostRegressor, Pool

# Loading and preparation
train_raw = pd.read_parquet("train.parquet")
test_raw = pd.read_parquet("test.parquet")
train_raw["dt"] = pd.to_datetime(train_raw["dt"])
test_raw["dt"] = pd.to_datetime(test_raw["dt"])

# Merging, feature generation, training, prediction
# (see main notebook)
```